#!/usr/bin/env python3
"""Build 4-region/32-view candidate metadata from semantic furniture points."""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import habitat_sim, magnum as mn, numpy as np, quaternion

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ea_avs_mvp_v11.core.paths import get_data_root, get_habitat_data_root

def _camera_rotation(position, target):
    d=np.asarray(target,np.float32)-np.asarray(position,np.float32); d/=max(float(np.linalg.norm(d)),1e-8); yaw=math.atan2(-float(d[0]),-float(d[2])); pitch=math.asin(float(d[1])); q=quaternion.from_rotation_vector([0,yaw,0])*quaternion.from_rotation_vector([pitch,0,0]); return [float(q.w),float(q.x),float(q.y),float(q.z)]

def main():
    data_root = get_data_root()
    p=argparse.ArgumentParser(); p.add_argument('--scene-root',type=Path,default=get_habitat_data_root()/'hm3d-minival'); p.add_argument('--scene-id',default='00800-TEEsavR23oF'); p.add_argument('--region-manifest',type=Path,default=data_root/'visualizations/semantic_regions_00800_rgb_3d/manifest.json'); p.add_argument('--output-dir',type=Path,default=data_root/'datasets/offline/hm3d-minival/00800-TEEsavR23oF/candidate_metadata'); p.add_argument('--num-views',type=int,default=32); args=p.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    d=args.scene_root/args.scene_id; glb=next(d.glob('*.basis.glb')); nav=next(d.glob('*.basis.navmesh')); b=habitat_sim.SimulatorConfiguration(); b.scene_id=str(glb); b.enable_physics=False; a=habitat_sim.AgentConfiguration(); sim=habitat_sim.Simulator(habitat_sim.Configuration(b,[a])); sim.pathfinder.load_nav_mesh(str(nav)); regions=json.loads(args.region_manifest.read_text())['regions']; placements=[]
    try:
      for rid,(region,info) in enumerate(sorted(regions.items())):
        base=np.asarray(info['placement_habitat_xyz'],np.float32); views=[]
        for vid in range(args.num_views):
          radius=(1.5,2.0,2.5,3.0)[vid//8]; az=float((vid%8)*45.0); ang=math.radians(az); raw=base+np.array([radius*math.sin(ang),0,radius*math.cos(ang)],np.float32); snap=np.asarray(sim.pathfinder.snap_point(raw),np.float32); nav_ok=bool(np.isfinite(snap).all() and np.linalg.norm(snap-raw)<=.5 and sim.pathfinder.is_navigable(snap));
          if not np.isfinite(snap).all(): snap=raw.copy()
          path_cost=None; reachable=False
          if nav_ok:
            sp=habitat_sim.ShortestPath(); sp.requested_start=np.asarray(sim.pathfinder.snap_point(base),np.float32); sp.requested_end=snap; reachable=bool(sim.pathfinder.find_path(sp)); path_cost=float(sp.geodesic_distance) if reachable else None
          views.append({'viewpoint_id':vid,'radius_m':radius,'azimuth_deg':az,'position':raw.tolist(),'snapped_position':snap.tolist(),'camera_rotation_wxyz':_camera_rotation(snap,[float(base[0]),float(base[1])+.85,float(base[2])]),'navigation':{'is_navigable':nav_ok,'is_reachable':reachable,'navigation_cost_m':path_cost}})
        placements.append({'placement_id':region,'region':region,'furniture':info['furniture'],'action':info['action'],'position':base.tolist(),'yaw_deg':0.0,'viewpoints':views})
    finally: sim.close()
    manifest={'version':'semantic-region-v1','scene_id':args.scene_id,'scene_root':str(args.scene_root),'source_region_manifest':str(args.region_manifest),'placements':len(placements),'candidate_viewpoints_per_placement':args.num_views,'regions':[x['region'] for x in placements],'expected_actions':980,'expected_view_samples':980*len(placements)*args.num_views,'observation_modalities':['RGB-at-render-time'],'rgb_saved':False,'depth_saved':False,'placements_data':placements}
    manifest['version']='semantic-region-v2'; manifest['navigation_reference']='placement_position'; manifest['dynamic_reachability_required_for_sequential_evaluation']=True; manifest['viewpoint_fields']=['position','snapped_position','camera_rotation_wxyz','navigation.is_navigable','navigation.is_reachable','navigation.navigation_cost_m']; (args.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)); print(json.dumps({'placements':len(placements),'views':args.num_views,'expected_view_samples':manifest['expected_view_samples']}))
if __name__=='__main__': main()
