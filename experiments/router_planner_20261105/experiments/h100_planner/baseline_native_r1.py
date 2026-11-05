"""Original supplied MoonEP kernel, one physical/logical rank. Degenerate control.
All 8-source trace rows are concatenated. Every expert is local and no EP
communication is possible. Do NOT use this as same-semantics R8 speed baseline.
Run separately from the logical-R8 experiment, with torchrun --nproc_per_node=1.
"""
from pathlib import Path
import argparse,hashlib,json,statistics,sys,time
import numpy as np
import torch
import torch.distributed as dist

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--layers',default='0,1,2,3,4,5,6,7');ap.add_argument('--groups',type=int,default=9);ap.add_argument('--repeats',type=int,default=20);ap.add_argument('--num-sms',type=int,default=32);ap.add_argument('--compile-only',action='store_true');a=ap.parse_args()
 torch.cuda.set_device(0);torch.set_num_threads(1);dist.init_process_group('nccl');assert dist.get_world_size()==1
 sys.path.insert(0,str(a.root/'inputs/MoonEP'))
 from moonep import Buffer
 import moonep.planning as native
 rows=[];buf=None
 try:
  for layer in map(int,a.layers.split(',')):
   p=a.root/f'outputs/trace_audit/layer_{layer:02d}.npz';x=np.load(p)['topk'];flat=x.reshape(-1,8);S,K=flat.shape
   if buf is None:
    buf=Buffer(S=S,H=16,K=K,E=352,num_ep_ranks=1,B=8,num_sms=a.num_sms,token_padding=128,replica_target=0,planner_only=True,explicitly_destroy=True)
    ctx=buf._require_ctx()
    ks=('R','E','B','S','K','NvS_capacity','NvS','num_vblocks','meta_chunk_padded','TPE_OFF','PLAN_OFF','BARRIER_OFF','TOPK0_OFF','ORDER_OFF','ORDER0_OFF','replica_max','token_padding','num_sms')
    native._get_compiled(*(int(ctx[k]) for k in ks))
    if a.compile_only:
     print('Original MoonEP R1 JIT compile completed; no planner kernel launched.',flush=True)
     return
   tk=torch.tensor(flat,device='cuda',dtype=torch.int32).flatten();tpe=torch.bincount(tk.long(),minlength=352).int();plan,cu=native.allocate_planning_outputs(ctx)
   for _ in range(3):native.launch_planning(ctx,tk,tpe,cu,plan)
   torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):native.launch_planning(ctx,tk,tpe,cu,plan)
   graph.replay();torch.cuda.synchronize()
   raw=torch.where(plan.dst<0,-plan.dst-1,plan.dst).long()
   assert (raw//ctx['NvS']==0).all().item();assert torch.equal(plan.physical_topk,tk)
   assert torch.bincount(raw).max().item()==1;assert (plan.experts_to_copy<0).all().item()
   reference=plan.dst.clone();times=[]
   start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
   for _ in range(a.groups):
    start.record()
    for _ in range(a.repeats):graph.replay()
    end.record();end.synchronize();times.append(start.elapsed_time(end)*1000/a.repeats)
   assert torch.equal(plan.dst,reference)
   rows.append(dict(layer=layer,input_shape=list(x.shape),kernel_shape=[1,S,K],timings_us=times,median_us=statistics.median(times),audit=dict(all_local=True,physical_expert_exact=True,unique_rows=True,no_copies=True)))
   a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(dict(metadata=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,scope='original MoonEP R1 degenerate control, concatenated trace. Not an R8 native latency; no crossrank traffic or balancing. Input tpe precomputed; planner outputs included.',physical_gpus=1,logical_ranks=1,source_sha256=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()),rows=rows),indent=2))
   print(layer,rows[-1]['median_us'],flush=True)
 finally:
  if buf is not None:buf.destroy()
  dist.destroy_process_group()
if __name__=='__main__':main()
