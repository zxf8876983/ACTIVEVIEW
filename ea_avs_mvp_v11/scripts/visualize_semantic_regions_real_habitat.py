#!/usr/bin/env python3
"""Render one action near representative semantic furniture in HM3D."""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import habitat_sim
import magnum as mn
import numpy as np
import quaternion

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))
from ea_avs_mvp_v11.core.paths import get_data_root, get_habitat_data_root, get_humanoid_urdf_path
from ea_avs_mvp_v11.dataset.babel_clean_dataset_generator import BabelCleanDatasetGenerator, _load_resampled_motion, apply_humanoid_pose, precompute_grounding_offsets, transform_camera_sequence_to_gravity
from ea_avs_mvp_v11.dataset.humanoid_grounding import select_floor_height
from ea_avs_mvp_v11.perception.skeleton_definition import get_skeleton_definition
from ea_avs_mvp_v11.scripts.visualize_household_rgb_to_3d_pipeline import _extract_2d_keypoints, _render_side_by_side, _load_one_record_per_label

URDF = get_humanoid_urdf_path('male_0')

def scene_sim(root: Path, sid: str, size: int):
    d=root/sid; glb=next(d.glob('*.basis.glb')); nav=next(d.glob('*.basis.navmesh'))
    b=habitat_sim.SimulatorConfiguration(); b.scene_id=str(glb); b.enable_physics=True
    s=habitat_sim.CameraSensorSpec(); s.uuid='color_sensor'; s.sensor_type=habitat_sim.SensorType.COLOR; s.resolution=[size,size]; s.position=mn.Vector3(0,1.1,0)
    a=habitat_sim.AgentConfiguration(); a.sensor_specifications=[s]; sim=habitat_sim.Simulator(habitat_sim.Configuration(b,[a])); sim.pathfinder.load_nav_mesh(str(nav))
    h=sim.get_articulated_object_manager().add_articulated_object_from_urdf(str(URDF)); return sim,h

def choose_placement(sim, semantic_json: Path, region: str):
    data=json.loads(semantic_json.read_text())['objects']; labels={'bedroom':('bed',),'kitchen':('kitchen cabinet',),'living_room':('couch',),'dining_area':('dining table','dining chair'),'bathroom':('bath sink',)}
    target_obj=next(x for label in labels[region] for x in data if x['label']==label)
    target=target_obj['center_xyz']; q=np.array([target[0],target[2],-target[1]],np.float32)
    candidates=[]
    for radius in np.linspace(.35,1.4,8):
        for angle in np.linspace(0,2*np.pi,24,endpoint=False):
            p=q+np.array([radius*np.cos(angle),0,radius*np.sin(angle)],np.float32); p=np.asarray(sim.pathfinder.snap_point(p),dtype=np.float32)
            if not np.isfinite(p).all(): continue
            clearance=float(sim.pathfinder.distance_to_closest_obstacle(p)); candidates.append((clearance,float(np.linalg.norm(p-q)),p))
    valid=[x for x in candidates if x[0]>=.28]
    if not valid: valid=candidates
    _,_,p=min(valid,key=lambda x:x[1]);
    ray=habitat_sim.geo.Ray(p+np.array([0,3,0],np.float32),np.array([0,-1,0],np.float32)); floor=select_floor_height(sim.cast_ray(ray).hits,reference_y=float(p[1])); p[1]=floor
    return p, target_obj['label']

def camera(base, floor, yaw_deg, target_y):
    a=math.radians(yaw_deg); pos=base+np.array([2.7*math.sin(a),0,2.7*math.cos(a)],np.float32); sensor=pos+np.array([0,1.1,0],np.float32); t=np.array([base[0],target_y,base[2]],np.float32); d=t-sensor; d/=max(float(np.linalg.norm(d)),1e-8); cy=math.atan2(-float(d[0]),-float(d[2])); cp=math.asin(float(d[1])); rot=quaternion.from_rotation_vector([0,cy,0])*quaternion.from_rotation_vector([cp,0,0]); st=habitat_sim.AgentState(); st.position=pos; st.rotation=rot; c=np.eye(4,dtype=np.float32); c[:3,:3]=quaternion.as_rotation_matrix(rot).astype(np.float32); c[:3,3]=sensor; return st,c

def main():
    root=get_data_root(); p=argparse.ArgumentParser(); p.add_argument('--scene-root',type=Path,default=get_habitat_data_root()/'hm3d-minival'); p.add_argument('--scene-id',default='00800-TEEsavR23oF'); p.add_argument('--manifest',type=Path,default=root/'datasets/stgcn_babel_selected16_habitat_pure_stumble_30frames_yolo26n_camera_fixed/val.json'); p.add_argument('--semantic-json',type=Path,default=root/'visualizations/hm3d_00800_semantic_topdown/furniture_positions.json'); p.add_argument('--output-dir',type=Path,default=root/'visualizations/semantic_regions_00800_rgb_3d'); p.add_argument('--image-size',type=int,default=256); p.add_argument('--target-frames',type=int,default=30); p.add_argument('--device',default='cuda:0'); p.add_argument('--action',default='sit',help='One BABEL action rendered in all four regions'); args=p.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    actions={region:args.action for region in ('bedroom','living_room','kitchen','dining_area')}; records=_load_one_record_per_label(args.manifest); gen=BabelCleanDatasetGenerator(output_root=args.output_dir/'cache',image_size=args.image_size,target_frames=args.target_frames,camera_height=1.2,device=args.device,pose_backend='ultralytics_yolo26n',yolo_weights=root/'checkpoints/ultralytics/yolo26n-pose.pt'); sim,human=scene_sim(args.scene_root,args.scene_id,args.image_size); sk=get_skeleton_definition(backend='h36m_17'); out={'scene_id':args.scene_id,'action':args.action,'regions':{}}
    try:
        for region,action in actions.items():
            base, furniture=choose_placement(sim,args.semantic_json,region); motion=_load_resampled_motion(records[action],args.target_frames); conv=gen.converter.convert(motion); joints=np.asarray(conv['pose_motion']['joints_array'],np.float32); roots=np.asarray(conv['pose_motion']['transform_array'],np.float32); off,cent=precompute_grounding_offsets(human,joints,roots,scene_yaw_deg=0); rgb=[]; c2w=[]
            for i,(j,r) in enumerate(zip(joints,roots)):
                apply_humanoid_pose(human,j,r,base_position=base,scene_yaw_deg=0,floor_y=float(base[1]),grounding_offset=float(off[i])); state,tr=camera(base,float(base[1]),180,float(base[1]+cent[i]+off[i])); sim.get_agent(0).set_state(state); rgb.append(np.asarray(sim.get_sensor_observations()['color_sensor'][:,:,:3],np.uint8).copy()); c2w.append(tr)
            pose3d,conf3d=gen.estimator.estimate_sequence(rgb); pose3d=transform_camera_sequence_to_gravity(pose3d,np.asarray(c2w,np.float32)); kp,conf=_extract_2d_keypoints(gen.estimator,rgb); rid=str(records[action]['record_id']); video=args.output_dir/f'{region}_{action.replace(" ","_")}_{rid}.mp4'; _render_side_by_side(rgb,kp,conf,pose3d,f'{region} ({furniture}) / {action}',rid,video,sk.edges,30); out['regions'][region]={'action':action,'furniture':furniture,'placement_habitat_xyz':base.tolist(),'record_id':rid,'video_path':str(video),'mean_2d_confidence':float(conf.mean()),'mean_3d_confidence':float(conf3d.mean())}
    finally: sim.close()
    (args.output_dir/'manifest.json').write_text(json.dumps(out,indent=2,ensure_ascii=False))

if __name__=='__main__': main()
