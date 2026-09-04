#!/usr/bin/env python3
"""文件用途：
    执行离线数据生成、划分或缓存构建入口。

主要输入：
    - 命令行参数与已有运行时数据。
主要输出：
    - 数据集、缓存或清单文件。
项目角色：
    - 属于 data 脚本入口，仅调用正式数据模块。
"""

from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import habitat_sim, magnum as mn, numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from activeview.simulation.habitat.camera import camera_rotation_wxyz
from activeview.core.paths import get_data_root, get_habitat_data_root

def main():
    data_root = get_data_root()
    p=argparse.ArgumentParser(); p.add_argument('--scene-root',type=Path,default=get_habitat_data_root()/'hm3d-minival'); p.add_argument('--scene-id',default='00800-TEEsavR23oF'); p.add_argument('--region-manifest',type=Path,default=data_root/'visualizations/semantic_regions_00800_rgb_3d/manifest.json'); p.add_argument('--placements-file',type=Path,default=None, help='Eight-placement placements.json; bypasses four-region schema.'); p.add_argument('--output-dir',type=Path,default=data_root/'datasets/offline/hm3d-minival/00800-TEEsavR23oF/candidate_metadata'); p.add_argument('--num-views',type=int,default=32); args=p.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.num_views != 32:
      raise ValueError('candidate geometry protocol requires exactly 32 viewpoints')
    d=args.scene_root/args.scene_id; glb=next(d.glob('*.basis.glb')); nav=next(d.glob('*.basis.navmesh')); b=habitat_sim.SimulatorConfiguration(); b.scene_id=str(glb); b.enable_physics=False; a=habitat_sim.AgentConfiguration(); sim=habitat_sim.Simulator(habitat_sim.Configuration(b,[a])); sim.pathfinder.load_nav_mesh(str(nav)); placements=[]
    try:
      if args.placements_file is not None:
        payload=json.loads(args.placements_file.read_text())
        source_placements=payload.get('placements', [])
        if len(source_placements) != 8:
          raise ValueError('placements-file must contain exactly 8 placements')
        for index, info in enumerate(source_placements):
          base=np.asarray(info['position'],np.float32); views=[]
          placement_id=str(info.get('placement_id',f'p{index:02d}'))
          for vid in range(args.num_views):
            radius=(1.5,2.0,2.5,3.0)[vid//8]; az=float((vid%8)*45.0); ang=math.radians(az); raw=base+np.array([radius*math.sin(ang),0,radius*math.cos(ang)],np.float32); snap=np.asarray(sim.pathfinder.snap_point(raw),np.float32); nav_ok=bool(np.isfinite(snap).all() and np.linalg.norm(snap-raw)<=.5 and sim.pathfinder.is_navigable(snap));
            if not np.isfinite(snap).all(): snap=raw.copy()
            path_cost=None; reachable=False
            if nav_ok:
              sp=habitat_sim.ShortestPath(); sp.requested_start=np.asarray(sim.pathfinder.snap_point(base),np.float32); sp.requested_end=snap; reachable=bool(sim.pathfinder.find_path(sp)); path_cost=float(sp.geodesic_distance) if reachable else None
            views.append({'viewpoint_id':vid,'radius_m':radius,'azimuth_deg':az,'position':raw.tolist(),'snapped_position':snap.tolist(),'camera_rotation_wxyz':camera_rotation_wxyz(snap,base).tolist(),'navigation':{'is_navigable':nav_ok,'is_reachable':reachable,'navigation_cost_m':path_cost}})
          placements.append({'placement_id':placement_id,'position':base.tolist(),'yaw_deg':float(info['yaw_deg']),'anchor_label':str(info['anchor_label']),'anchor_object_id':int(info['anchor_object_id']),'viewpoints':views})
      else:
       regions=json.loads(args.region_manifest.read_text())['regions']
       for rid,(region,info) in enumerate(sorted(regions.items())):
        base=np.asarray(info['placement_habitat_xyz'],np.float32); views=[]
        for vid in range(args.num_views):
          radius=(1.5,2.0,2.5,3.0)[vid//8]; az=float((vid%8)*45.0); ang=math.radians(az); raw=base+np.array([radius*math.sin(ang),0,radius*math.cos(ang)],np.float32); snap=np.asarray(sim.pathfinder.snap_point(raw),np.float32); nav_ok=bool(np.isfinite(snap).all() and np.linalg.norm(snap-raw)<=.5 and sim.pathfinder.is_navigable(snap));
          if not np.isfinite(snap).all(): snap=raw.copy()
          path_cost=None; reachable=False
          if nav_ok:
            sp=habitat_sim.ShortestPath(); sp.requested_start=np.asarray(sim.pathfinder.snap_point(base),np.float32); sp.requested_end=snap; reachable=bool(sim.pathfinder.find_path(sp)); path_cost=float(sp.geodesic_distance) if reachable else None
          views.append({'viewpoint_id':vid,'radius_m':radius,'azimuth_deg':az,'position':raw.tolist(),'snapped_position':snap.tolist(),'camera_rotation_wxyz':camera_rotation_wxyz(snap,base).tolist(),'navigation':{'is_navigable':nav_ok,'is_reachable':reachable,'navigation_cost_m':path_cost}})
        placements.append({'placement_id':region,'region':region,'furniture':info['furniture'],'action':info['action'],'position':base.tolist(),'yaw_deg':0.0,'viewpoints':views})
    finally: sim.close()
    manifest={'version':'semantic-region-v1' if args.placements_file is None else 'furniture-placement-v2','scene_id':args.scene_id,'scene_root':str(args.scene_root),'source_region_manifest':str(args.region_manifest) if args.placements_file is None else None,'source_placements_file':str(args.placements_file.resolve()) if args.placements_file is not None else None,'placements':len(placements),'candidate_viewpoints_per_placement':args.num_views,'regions':[x['region'] for x in placements if 'region' in x],'expected_actions':980 if args.placements_file is None else None,'expected_view_samples':980*len(placements)*args.num_views if args.placements_file is None else len(placements)*args.num_views,'observation_modalities':['RGB-at-render-time'],'rgb_saved':False,'depth_saved':False,'placements_data':placements}
    if args.placements_file is None:
      manifest['version']='semantic-region-v2'
    manifest['navigation_reference']='placement_position'; manifest['dynamic_reachability_required_for_sequential_evaluation']=True; manifest['rotation_reference']='exact_offline_render_state'; manifest['sensor_height_m']=1.1; manifest['target_height_m']=0.85; manifest['viewpoint_fields']=['position','snapped_position','camera_rotation_wxyz','navigation.is_navigable','navigation.is_reachable','navigation.navigation_cost_m']; (args.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)); print(json.dumps({'placements':len(placements),'views':args.num_views,'expected_view_samples':manifest['expected_view_samples']}))
if __name__=='__main__': main()
