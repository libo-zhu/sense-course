"""Evaluate a provided best checkpoint without rerunning training."""
from pathlib import Path
import argparse,json
import torch
from torch.utils.data import Subset
from mac_models import create_model,SharedSpecificFusion
from mac_data import CHANNELS,open_official_test
from mac_demo import select_device
from run_experiments import evaluate,EarlyConcat
ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',default='msh-hsi');p.add_argument('--device',choices=['auto','cpu','mps'],default='auto');p.add_argument('--limit',type=int,default=64,help='0 means full official test; otherwise first N, never a benchmark');a=p.parse_args()
 if a.limit<0:p.error('--limit must be >=0')
 path=ROOT/'results/augsburg-seed42/models'/a.model/'best.pt'
 if not path.is_file():p.error('Unknown saved model; list results/augsburg-seed42/models/')
 torch.set_num_threads(4);device=select_device(a.device)
 mods=['hsi','sar','dsm'] if a.model in ['teacher','moddrop','early-concat'] else [a.model.rsplit('-',1)[1]]
 channels=[CHANNELS['augsburg'][m] for m in mods]
 model=EarlyConcat(channels,7) if a.model=='early-concat' else create_model(channels,7)
 if a.model.startswith('msh-'):model=SharedSpecificFusion(create_model(channels,7),create_model(channels,7))
 saved=torch.load(path,map_location='cpu',weights_only=True);model.load_state_dict(saved['model']);model.to(device).eval()
 ds=open_official_test(ROOT,'augsburg',mods)
 try:
  metrics,_=evaluate(model,ds if a.limit==0 else Subset(ds,range(min(a.limit,len(ds)))),mods,device,7,128)
  result={'model':a.model,'selected_epoch':saved['epoch'],'full_official_test':a.limit==0,'device':str(device),'purpose':'official evaluation' if a.limit==0 else 'checkpoint subset check, not benchmark',**metrics}
  out=ROOT/'outputs/inference';out.mkdir(parents=True,exist_ok=True);(out/f'{a.model}-{a.limit}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
 finally:ds.close()
if __name__=='__main__':main()
