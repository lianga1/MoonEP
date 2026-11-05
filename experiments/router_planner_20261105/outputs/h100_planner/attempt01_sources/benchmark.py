"""One physical H100, eight logical ranks; raw timings and semantic audits.
Fixed support selection is excluded. Graph timings include GPU computation only;
all algorithms also receive eager CUDA-event and synchronized host timing.
"""
import argparse,json,time,traceback,hashlib,platform,sys
from pathlib import Path
import numpy as np
import torch
from materialize import Materializer
from materialize_cuda import CudaMaterializer
from baseline_virtual import VirtualMoonEPPlanner
from bundle_wrapper import BundlePlanner
from aggregate import AggregateRunner
from subset_cuda import SubsetRunner

def audit(x,o,support,base):
 R,S,K=x.shape;assert o.shape==x.shape and ((o>=0)&(o<R)).all()
 assert support[o,x].all();assert np.array_equal(np.bincount(o.ravel(),minlength=R),np.full(R,S*K))
 so=np.sort(o,axis=-1);first=np.concatenate((np.ones((R,S,1),bool),so[:,:,1:]!=so[:,:,:-1]),axis=2)
 remote=int((first&(so!=np.arange(R)[:,None,None])).sum())
 return dict(remote=remote,reduction_pct=100*(1-remote/base),audit_pass=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);ap.add_argument('--layers',default='0,1,2,3,4,5,6,7');ap.add_argument('--output',type=Path,required=True);ap.add_argument('--groups',type=int,default=10);ap.add_argument('--graph-repeats',type=int,default=20);ap.add_argument('--variants',default='bundle,bundle8,aggregate_aos,aggregate_soa,subset_exact,subset_parallel,subset_indexed');a=ap.parse_args();root=a.root
 sys.path.insert(0,str(root/'analysis'));from moonep_cpu import communication
 torch.cuda.set_device(0);torch.set_num_threads(1)
 out=dict(metadata=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,platform=platform.platform(),physical_gpus=1,logical_ranks=8,scope='same saved trace, rt0 seed, fixed256/g2 support. Seed+exchange+output excludes support selection, input statistics, actual data/weight communication. Native R1 must be reported separately.',code_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'experiments/h100_planner').glob('*') if p.is_file()}),rows=[],failures=[])
 def save():a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2))
 for layer in map(int,a.layers.split(',')):
  x=np.load(root/f'outputs/trace_audit/layer_{layer:02d}.npz')['topk'].astype(np.int32);s=np.load(root/f'outputs/replay/layer{layer:02d}_b8_small256_global_4.npz');support=s['support'];seed=np.load(root/f'outputs/rt2_b8_comparison/layer{layer:02d}_native_rt0.npz')['owner'].astype(np.int32)
  base=communication(x,seed,352)['remote_token_destinations'];xt=torch.tensor(x,device='cuda');ot=torch.tensor(seed,device='cuda');st=torch.tensor(support,device='cuda')
  baseline=VirtualMoonEPPlanner(xt);b=baseline.run();torch.cuda.synchronize();assert np.array_equal(b.cpu().numpy().reshape(x.shape),seed)
  b=baseline.run_seed();torch.cuda.synchronize();assert np.array_equal(b.cpu().numpy().reshape(x.shape),seed)
  mat=CudaMaterializer(xt,st);oracle=Materializer(xt,st)
  funcs={'moonep_port_full':baseline.run,'seed_only':baseline.run_seed,'materialize_only':lambda:mat.run(ot)};runners={};expected={}
  def material_audit(res):
   want=oracle.run(res);got=mat.run(res);torch.cuda.synchronize()
   for k in want:assert torch.equal(want[k],got[k]),'materializer mismatch '+k
   raw=got['raw_dst'].cpu().numpy();phys=got['physical_topk'].cpu().numpy()
   assert len(np.unique(raw))==x.size and phys.min()>=0 and phys.max()<8*52
   assert np.array_equal(got['provenance'][got['raw_dst'].long()].cpu().numpy(),(np.arange(x.size)//mat.cap)*mat.NvS+np.arange(x.size)%mat.cap)
  material_audit(ot)
  for name in a.variants.split(','):
   try:
    if name=='bundle':runner=BundlePlanner(xt,st,x.shape[1]);expected[name]=s['owner']
    elif name=='bundle8':
     from bundle_lane8 import Bundle8LanePlanner
     runner=Bundle8LanePlanner(xt,st,x.shape[1]);expected[name]=s['owner']
    elif name.startswith('aggregate'):
     runner=AggregateRunner(xt,st,coalesced=name.endswith('soa'));expected[name]=np.load(root/f'outputs/aggregate_exchange_multipair/layer{layer:02d}_mixed_sizes.npz')['owner']
    elif name.startswith('subset'):
     runner=SubsetRunner(xt,st,h=2,g=8);expected[name]=np.load(root/(f'outputs/pruned_subset_indexed/layer{layer:02d}_h2_g8.npz' if name=='subset_exact' else f'outputs/h100_subset_reference/layer{layer:02d}.npz'))['owner']
    else:raise ValueError(name)
    runners[name]=runner
    if name.startswith('subset'):
     if name=='subset_indexed':runner.prepare_indexed(ot)
     else:runner.prepare(ot)
     if name=='subset_exact':f=lambda r=runner:r.run(ot);full=lambda r=runner:r.full(baseline.run_seed().reshape(x.shape));prepared=lambda r=runner:r.full(ot)
     elif name=='subset_indexed':f=lambda r=runner:r.run_indexed(ot);full=lambda r=runner:r.full_indexed(baseline.run_seed().reshape(x.shape));prepared=lambda r=runner:r.full_indexed(ot)
     else:f=lambda r=runner:r.run_parallel(ot);full=lambda r=runner:r.full_parallel(baseline.run_seed().reshape(x.shape));prepared=lambda r=runner:r.full_parallel(ot)
    else:f=lambda r=runner:r.run(ot);full=lambda r=runner:r.run(baseline.run_seed().reshape(x.shape))
    res=f();torch.cuda.synchronize();cpu=res.cpu().numpy().reshape(x.shape)
    assert np.array_equal(cpu,expected[name]),f'{name} mismatches {int((cpu!=expected[name]).sum())}'
    met=audit(x,cpu,support,base);material_audit(res)
    # Audit the whole measured pipeline too; seed reset must not leak prior state.
    whole=full();torch.cuda.synchronize();assert np.array_equal(whole.cpu().numpy().reshape(x.shape),expected[name]);material_audit(whole)
    funcs[name]=f;funcs[name+'_seed_exchange_output']=lambda full=full:mat.run(full())
    if name.startswith('subset'):
     funcs[name+'_prepare_exchange']=prepared
     if name=='subset_exact':funcs['subset_prepare']=lambda r=runner:r.prepare(ot)
     if name=='subset_indexed':funcs['subset_prepare_indexed']=lambda r=runner:r.prepare_indexed(ot)
    out['rows'].append(dict(layer=layer,variant=name,quality=met,exact_cpu=True,materialized=True,base_remote=base))
   except Exception as exc:
    out['failures'].append(dict(layer=layer,variant=name,error=repr(exc),traceback=traceback.format_exc()));save();print('FAILED',name,repr(exc),flush=True)
  # Fixed repetitions, rotating order, all methods warmed once before timing.
  for f in funcs.values():f()
  torch.cuda.synchronize();times={n:[] for n in funcs};walls={n:[] for n in funcs};names=list(funcs)
  for rep in range(a.groups):
   shift=rep%len(names)
   for name in names[shift:]+names[:shift]:
    torch.cuda.synchronize();start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);wall=time.perf_counter();start.record();ret=funcs[name]();end.record();end.synchronize();walls[name].append((time.perf_counter()-wall)*1000);times[name].append(start.elapsed_time(end))
  row=dict(layer=layer,scope_timings_ms=times,host_timings_ms=walls,graph_timings_ms={},graph_audit={});out['rows'].append(row);save()
  graphs={};graph_returns={}
  def graph_audit(name,result):
   # This runs outside timing. The reference allocates its own tensors and
   # never calls mat.run(), which would overwrite a captured material result.
   if result is None:
    if name=='subset_prepare':
     result=runners['subset_exact'].run(ot);target=expected['subset_exact']
    elif name=='subset_prepare_indexed':
     result=runners['subset_indexed'].run_indexed(ot);target=expected['subset_indexed']
    else:raise AssertionError('Unexpected empty graph result '+name)
   elif name in ('moonep_port_full','seed_only','materialize_only'):target=seed
   elif name.endswith('_seed_exchange_output'):target=expected[name.removesuffix('_seed_exchange_output')]
   elif name.endswith('_prepare_exchange'):target=expected[name.removesuffix('_prepare_exchange')]
   else:target=expected[name]
   if isinstance(result,dict):
    target_tensor=torch.as_tensor(target.astype(np.int32),device=xt.device)
    want=oracle.run(target_tensor)
    assert result.keys()==want.keys(),'Graph materializer output fields mismatch '+name
    for k in want:assert torch.equal(result[k],want[k]),'Graph materializer mismatch '+name+' '+k
   else:
    assert np.array_equal(result.cpu().numpy().reshape(x.shape),target),'Graph owner mismatch '+name
  for name,f in funcs.items():
   if name.startswith('aggregate'):continue  # Thrust grouping currently allocates/synchronizes.
   phase='graph_capture'
   try:
    for _ in range(3):f()
    torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):captured_result=f()
    g.replay();torch.cuda.synchronize();phase='graph_validation';graph_audit(name,captured_result)
    graphs[name]=g;graph_returns[name]=captured_result;row['graph_timings_ms'][name]=[];row['graph_audit'][name]=True
   except Exception as exc:
    row['graph_audit'][name]=False
    out['failures'].append(dict(layer=layer,variant=name,phase=phase,error=repr(exc),traceback=traceback.format_exc()));save()
  names=list(graphs)
  for rep in range(a.graph_repeats):
   shift=rep%len(names)
   for name in names[shift:]+names[:shift]:
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record();graphs[name].replay();end.record();end.synchronize();row['graph_timings_ms'][name].append(start.elapsed_time(end))
  # Check again after all interleaved replays. Keep any invalid raw timings as
  # failure evidence, but mark them unusable rather than reporting a speedup.
  for name,g in graphs.items():
   try:g.replay();torch.cuda.synchronize();graph_audit(name,graph_returns[name])
   except Exception as exc:
    row['graph_audit'][name]=False
    out['failures'].append(dict(layer=layer,variant=name,phase='graph_post_timing_validation',error=repr(exc),traceback=traceback.format_exc()))
  save();print(layer,{n:round(float(np.median(v)),4) for n,v in times.items()},flush=True)
 save()
if __name__=='__main__':main()
