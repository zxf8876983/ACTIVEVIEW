#!/usr/bin/env python3
"""Generate RGB-free per-view skeleton data for four semantic HM3D regions."""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path
from typing import Any, Dict, Mapping
import habitat_sim, magnum as mn, numpy as np, quaternion, torch

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from activeview.core.paths import get_data_root, get_habitat_data_root, get_humanoid_urdf_path
from activeview.active_view.camera_pose import camera_rotation_wxyz
from activeview.dataset.babel_clean_dataset_generator import BabelCleanDatasetGenerator,_load_resampled_motion,apply_humanoid_pose,precompute_grounding_offsets,transform_camera_sequence_to_gravity
from activeview.perception.skeleton_definition import get_skeleton_definition
from activeview.perception.skeleton_normalizer import SkeletonNormalizer

URDF=get_humanoid_urdf_path('male_0')

def _sim(root:Path,sid:str,size:int,cameras:int):
 d=root/sid; glb=next(d.glob('*.basis.glb')); nav=next(d.glob('*.basis.navmesh')); b=habitat_sim.SimulatorConfiguration(); b.scene_id=str(glb); b.enable_physics=True
 agents=[]
 for i in range(cameras):
  s=habitat_sim.CameraSensorSpec(); s.uuid=f'color_{i}'; s.sensor_type=habitat_sim.SensorType.COLOR; s.resolution=[size,size]; s.position=mn.Vector3(0,1.1,0); s.hfov=75.0; a=habitat_sim.AgentConfiguration(); a.sensor_specifications=[s]; agents.append(a)
 sim=habitat_sim.Simulator(habitat_sim.Configuration(b,agents)); sim.pathfinder.load_nav_mesh(str(nav)); human=sim.get_articulated_object_manager().add_articulated_object_from_urdf(str(URDF)); return sim,human

def _agent_position(view: Mapping[str, Any], base: np.ndarray) -> np.ndarray:
 p=np.asarray(view.get('snapped_position',view['position']),np.float32).copy(); p[1]=float(base[1]); return p

def _state(view,base):
 p=_agent_position(view,np.asarray(base,np.float32)); q=quaternion.from_float_array(camera_rotation_wxyz(p,base)); st=habitat_sim.AgentState(); st.position=p; st.rotation=q; c=np.eye(4,dtype=np.float32); c[:3,:3]=quaternion.as_rotation_matrix(q).astype(np.float32); c[:3,3]=p+np.array([0,1.1,0],np.float32); return st,c

def _navigation_arrays(placement: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    """Extract per-view geometry and static navigation metadata for persistence."""
    views = list(placement["viewpoints"])
    costs = [
        np.nan
        if view.get("navigation", {}).get("navigation_cost_m") is None
        else float(view["navigation"]["navigation_cost_m"])
        for view in views
    ]
    return {
        "viewpoint_positions": np.asarray([view["position"] for view in views], dtype=np.float32),
        "viewpoint_snapped_positions": np.asarray(
            [view.get("snapped_position", view["position"]) for view in views], dtype=np.float32
        ),
        "viewpoint_agent_positions": np.asarray(
            [_agent_position(view, np.asarray(placement["position"], dtype=np.float32)) for view in views],
            dtype=np.float32,
        ),
        "viewpoint_rotations_wxyz": np.asarray(
            [camera_rotation_wxyz(_agent_position(view, np.asarray(placement["position"], dtype=np.float32)), placement["position"]) for view in views], dtype=np.float32
        ),
        "viewpoint_is_navigable": np.asarray(
            [bool(view.get("navigation", {}).get("is_navigable", False)) for view in views],
            dtype=np.bool_,
        ),
        "viewpoint_is_reachable_from_placement": np.asarray(
            [bool(view.get("navigation", {}).get("is_reachable", False)) for view in views],
            dtype=np.bool_,
        ),
        "viewpoint_navigation_cost_m": np.asarray(costs, dtype=np.float32),
    }

def _worker(args, tasks, manifest, out):
 sk=get_skeleton_definition(backend='h36m_17'); norm=SkeletonNormalizer(skel_def=sk); root=Path(args.scene_root); scene_dir=root/args.scene_id; navmesh_path=str(next(scene_dir.glob('*.basis.navmesh')).resolve()); sim,human=_sim(root,args.scene_id,args.image_size,4); gen=BabelCleanDatasetGenerator(output_root=out/'cache',image_size=args.image_size,target_frames=args.target_frames,camera_height=1.2,device=args.device,pose_backend='ultralytics_yolo26n',yolo_weights=args.yolo_weights); done=[]
 try:
  for num,(region,record) in enumerate(tasks,1):
   placement=next(x for x in manifest['placements_data'] if x['region']==region); base=np.asarray(placement['position'],np.float32); motion=_load_resampled_motion(record,args.target_frames); conv=gen.converter.convert(motion); joints=np.asarray(conv['pose_motion']['joints_array'],np.float32); roots=np.asarray(conv['pose_motion']['transform_array'],np.float32); offsets,_=precompute_grounding_offsets(human,joints,roots,scene_yaw_deg=0); all_skel=[]; all_conf=[]
   for start in range(0,len(placement['viewpoints']),4):
    views=placement['viewpoints'][start:start+4]; states=[]; c2w=[]
    for v in views:
     st,c=_state(v,base); states.append(st); c2w.append(c)
    for i,st in enumerate(states): sim.get_agent(i).set_state(st)
    seqs=[[] for _ in views]
    for fi,(pose,rt) in enumerate(zip(joints,roots)):
     apply_humanoid_pose(human,pose,rt,base_position=base,scene_yaw_deg=0,floor_y=float(base[1]),grounding_offset=float(offsets[fi])); obs=sim.get_sensor_observations(list(range(len(views))))
     for i in range(len(views)): seqs[i].append(np.asarray(obs[i][f'color_{i}'][:,:,:3],np.uint8).copy())
    for vi,frames in enumerate(seqs):
     est,conf=gen.estimator.estimate_sequence(frames); grav=transform_camera_sequence_to_gravity(est,np.repeat(c2w[vi][None,...],args.target_frames,axis=0)); normalized=norm.normalize_sequence(grav,align_canonical=True); all_skel.append(np.transpose(normalized,(2,0,1)).astype(np.float32)); all_conf.append(float(np.mean(conf)))
   rid=str(record['record_id']); path=out/f'{region}/{rid}.npz'; path.parent.mkdir(parents=True,exist_ok=True); nav=_navigation_arrays(placement); np.savez_compressed(path,skeleton=np.stack(all_skel),confidence=np.asarray(all_conf,np.float32),viewpoint_ids=np.arange(32,dtype=np.int32),scene_id=np.asarray(args.scene_id),region=np.asarray(region),placement_id=np.asarray(str(placement.get('placement_id',region))),placement_position=base.astype(np.float32),viewpoint_positions=nav['viewpoint_positions'],viewpoint_snapped_positions=nav['viewpoint_snapped_positions'],viewpoint_agent_positions=nav['viewpoint_agent_positions'],viewpoint_rotations_wxyz=nav['viewpoint_rotations_wxyz'],viewpoint_is_navigable=nav['viewpoint_is_navigable'],viewpoint_is_reachable_from_placement=nav['viewpoint_is_reachable_from_placement'],viewpoint_navigation_cost_m=nav['viewpoint_navigation_cost_m']); done.append({'scene_id':args.scene_id,'region':region,'record_id':rid,'label':record['action_label'],'label_id':int(record['label_id']),'path':str(path.relative_to(out)),'views':32,'placement_id':str(placement.get('placement_id',region)),'placement_position':base.tolist(),'navmesh_path':navmesh_path,'candidate_metadata_manifest':str((Path(args.candidate_dir)/'manifest.json').resolve()),'static_reachability_reference':'placement_position'})
   if num%5==0: (out/f'worker_{args.worker_id}.json').write_text(json.dumps(done,indent=2,ensure_ascii=False))
 finally: sim.close()
 (out/f'worker_{args.worker_id}.json').write_text(json.dumps(done,indent=2,ensure_ascii=False)); print(f'worker {args.worker_id} complete {len(done)}')

def main():
 d=get_data_root(); p=argparse.ArgumentParser(); p.add_argument('--candidate-dir',type=Path,default=d/'datasets/offline/hm3d-minival/00800-TEEsavR23oF/candidate_metadata'); p.add_argument('--manifest',type=Path,default=d/'datasets/stgcn_babel_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed/val.json'); p.add_argument('--output-dir',type=Path,default=d/'datasets/offline/hm3d-minival/00800-TEEsavR23oF'); p.add_argument('--scene-root',type=Path,default=get_habitat_data_root()/'hm3d-minival'); p.add_argument('--scene-id',default='00800-TEEsavR23oF'); p.add_argument('--workers',type=int,default=4); p.add_argument('--worker-id',type=int,default=None); p.add_argument('--image-size',type=int,default=256); p.add_argument('--target-frames',type=int,default=30); p.add_argument('--device',default='cuda:0'); p.add_argument('--yolo-weights',type=Path,default=d/'checkpoints/ultralytics/yolo26n-pose.pt'); p.add_argument('--max-records',type=int,default=None); args=p.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True); cm=json.loads((args.candidate_dir/'manifest.json').read_text()); records=json.loads(args.manifest.read_text()); records=records[:args.max_records] if args.max_records else records
 tasks=[(pl['region'],r) for pl in cm['placements_data'] for r in records]
 if args.worker_id is not None: _worker(args,tasks[args.worker_id::args.workers],cm,args.output_dir); return
 launch=[]
 for wid in range(args.workers): launch.append(subprocess.Popen([sys.executable,str(Path(__file__)),'--candidate-dir',str(args.candidate_dir),'--manifest',str(args.manifest),'--output-dir',str(args.output_dir),'--scene-root',str(args.scene_root),'--scene-id',args.scene_id,'--workers',str(args.workers),'--worker-id',str(wid),'--image-size',str(args.image_size),'--target-frames',str(args.target_frames),'--device',args.device,'--yolo-weights',str(args.yolo_weights)] + (['--max-records',str(args.max_records)] if args.max_records else [])))
 codes=[x.wait() for x in launch]
 if any(c!=0 for c in codes): raise RuntimeError(f'worker failures {codes}')
 items=[]
 for wid in range(args.workers): items.extend(json.loads((args.output_dir/f'worker_{wid}.json').read_text()))
 items.sort(key=lambda x:(x['region'],x['record_id'])); summary={'version':'semantic-region-offline-v2','scene_id':args.scene_id,'records':len(records),'regions':len(cm['placements_data']),'views_per_record':32,'samples':len(items)*32,'target_frames':args.target_frames,'image_size':args.image_size,'workers':args.workers,'rgb_saved':False,'depth_saved':False,'skeleton_shape_per_record':[32,3,args.target_frames,17],'navigation_reference':'placement_position','dynamic_reachability_must_be_recomputed_from_current_robot_position':True,'rotation_reference':'exact_offline_render_state','sensor_height_m':1.1,'target_height_m':0.85,'per_record_navigation_fields':['scene_id','navmesh_path','placement_position','viewpoint_positions','viewpoint_snapped_positions','viewpoint_agent_positions','viewpoint_rotations_wxyz','viewpoint_is_navigable','viewpoint_is_reachable_from_placement','viewpoint_navigation_cost_m'],'items':items}; (args.output_dir/'manifest.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)); print(json.dumps({'records':len(records),'regions':4,'views':len(items)*32}))
if __name__=='__main__': main()
