"""Synthetic environment demo, without data download. Never reports benchmark accuracy."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from torch import nn
from mac_models import create_model,JDA,SharedSpecificFusion
from mac_demo import select_device,state_copy,assert_same_state

def main():
 p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','mps','auto'],default='auto');a=p.parse_args()
 torch.set_num_threads(4);torch.manual_seed(42);device=select_device(a.device)
 xs=[torch.rand(8,c,7,7,device=device) for c in [180,4,1]];y=torch.arange(8,device=device)%7
 teacher=create_model([180,4,1],7).to(device);student=create_model([180],7).to(device);regular=create_model([180],7).to(device)
 ce=nn.CrossEntropyLoss();losses={}
 def update(name,model,loss_fn):
  model.train();opt=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.001,momentum=.9)
  values=[];before=state_copy(model)
  for _ in range(3):
   opt.zero_grad(set_to_none=True);loss=loss_fn();assert torch.isfinite(loss);loss.backward()
   assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
   opt.step();values.append(loss.item())
  assert any(not torch.equal(v,model.state_dict()[k].cpu()) for k,v in before.items())
  losses[name]=values
 update('teacher',teacher,lambda:ce(teacher(*xs)[0],y));teacher.eval().requires_grad_(False);before=state_copy(teacher)
 update('regular',regular,lambda:ce(regular(xs[0])[0],y))
 kd=JDA(2,1).to(device)
 def distill():
  with torch.no_grad():tl,tf=teacher(*xs)
  sl,sf=student(xs[0]);return ce(sl,y)+kd(sf,tf,sl,tl)
 update('jad',student,distill);assert_same_state(before,teacher,'teacher frozen')
 fusion=SharedSpecificFusion(student,regular).to(device)
 sf,rf=state_copy(student),state_copy(regular)
 update('fusion',fusion,lambda:ce(fusion(xs[0]),y))
 assert_same_state(sf,student,'student feature frozen',exclude_fc=True);assert_same_state(rf,regular,'regular feature frozen',exclude_fc=True)
 out=ROOT/'outputs/offline-demo';out.mkdir(parents=True,exist_ok=True)
 fusion.eval();torch.save({'model':state_copy(fusion),'purpose':'synthetic environment check'},out/'synthetic.pt')
 restored=SharedSpecificFusion(create_model([180],7),create_model([180],7)).to(device)
 restored.load_state_dict(torch.load(out/'synthetic.pt',map_location='cpu',weights_only=True)['model']);restored.eval()
 with torch.no_grad():assert torch.allclose(fusion(xs[0]),restored(xs[0]),atol=1e-5,rtol=1e-4)
 report={'status':'passed','device':str(device),'purpose':'synthetic environment check, not dataset accuracy','stages':losses,'teacher_frozen':True,'single_input_inference':True,'reload_same':True}
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
