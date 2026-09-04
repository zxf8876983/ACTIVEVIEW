#!/usr/bin/env python3
"""文件用途：
    执行只读验证或评估入口。

主要输入：
    - 冻结预测、缓存与评估协议。
主要输出：
    - 指标摘要或验证报告。
项目角色：
    - 属于 evaluation 脚本入口，不修改模型和数据。
"""

from __future__ import annotations
import argparse, json, random
from pathlib import Path
import sys
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from activeview.recognition.stgcn.model import STGCN
from activeview.core.paths import get_data_root
from activeview.perception.skeleton import get_skeleton_definition

def entropy(prob):
    return -(prob*np.log(np.clip(prob,1e-8,1))).sum(axis=-1)

def main():
 d=get_data_root(); p=argparse.ArgumentParser(); p.add_argument('--data-dir',type=Path,default=d/'datasets/offline/hm3d-minival/00800-TEEsavR23oF'); p.add_argument('--candidate-dir',type=Path,default=d/'datasets/offline/hm3d-minival/00800-TEEsavR23oF/candidate_metadata'); p.add_argument('--checkpoint',type=Path,default=d/'checkpoints/stgcn_reduced14_kneel_babel_diversity_v1/stgcn_reduced14_kneel_best.pth'); p.add_argument('--label-mapping',type=Path,default=d/'datasets/reduced14_kneel_babel_diversity_v1/raw-train/label_mapping.json'); p.add_argument('--output',type=Path,default=d/'results/semantic_region_offline_baselines.json'); p.add_argument('--device',default='cuda:0'); p.add_argument('--seed',type=int,default=42); args=p.parse_args(); torch.manual_seed(args.seed); random.seed(args.seed)
 m=json.loads((args.data_dir/'manifest.json').read_text()); mapping=json.loads(args.label_mapping.read_text())
 cats=[x for x,_ in sorted(mapping.items(),key=lambda kv:int(kv[1]))]; dev=torch.device(args.device if torch.cuda.is_available() else 'cpu'); model=STGCN(in_channels=3,num_classes=len(cats),graph_strategy='spatial',edge_importance_weighting=True,skel_def=get_skeleton_definition(backend='h36m_17')).to(dev); ck=torch.load(args.checkpoint,map_location=dev,weights_only=False); model.load_state_dict(ck['model_state_dict']); model.eval()
 items=m['items']; candidate=json.loads((args.candidate_dir/'manifest.json').read_text()); views={str(x.get('placement_id',x.get('region'))):x['viewpoints'] for x in candidate['placements_data']}
 result={'data_dir':str(args.data_dir),'checkpoint':str(args.checkpoint),'records':m['records'],'placements':m.get('placements',m.get('regions')),'views_per_record':32,'oracle_definition':'GT-correctness over the reachable candidate pool; if no candidate is correct, fall back to minimum entropy for deterministic reporting','policies':{},'learned_policy':{'status':'NOT EVALUATED','reason':'No Utility Predictor trained on this semantic-region protocol; historical checkpoint is quarantined.'}}
 for policy in ('Fixed','Random','Nearest','Oracle'):
  selected=[]
  for item in items:
   z=np.load(args.data_dir/item['path']); x=torch.from_numpy(z['skeleton']).float().to(dev); x=x.unsqueeze(-1)
   with torch.inference_mode(): prob=torch.softmax(model(x),dim=-1).cpu().numpy()
   item_key=str(item.get('placement_id',item.get('region'))); h=entropy(prob); candidates=views[item_key]; reachable=[v for v in candidates if v['navigation'].get('is_reachable')]
   pool=reachable or candidates; ids=[int(v['viewpoint_id']) for v in pool]
   if policy=='Fixed': idx=min(ids)
   elif policy=='Random': idx=random.Random(f'{args.seed}|{item_key}|{item["record_id"]}').choice(ids)
   elif policy=='Nearest': idx=min(ids,key=lambda j: float(next(v for v in pool if int(v['viewpoint_id'])==j)['navigation'].get('navigation_cost_m') or 1e9))
   else:
    correct_ids=[j for j in ids if int(np.argmax(prob[j]))==int(item['label_id'])]
    idx=min(correct_ids,key=lambda j: float(h[j])) if correct_ids else min(ids,key=lambda j: float(h[j]))
   selected.append((item,int(idx),h,prob))
  targets=[]; preds=[]; hs=[]; per={}
  for item,idx,h,prob in selected:
   t=int(item['label_id']); pr=int(np.argmax(prob[idx])); targets.append(t); preds.append(pr); hs.append(float(h[idx])); item_key=str(item.get('placement_id',item.get('region'))); per.setdefault(item_key,[0,0,[]]); per[item_key][0]+=1; per[item_key][1]+=int(pr==t); per[item_key][2].append(float(h[idx]))
  result['policies'][policy]={'n':len(selected),'accuracy':float(np.mean(np.asarray(preds)==np.asarray(targets))),'mean_entropy':float(np.mean(hs)),'per_placement':{r:{'n':v[0],'accuracy':v[1]/v[0],'mean_entropy':float(np.mean(v[2]))} for r,v in per.items()}}
 args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)); print(json.dumps({k:{q:v[q] for q in ('accuracy','mean_entropy')} for k,v in result['policies'].items()},ensure_ascii=False))
if __name__=='__main__': main()
