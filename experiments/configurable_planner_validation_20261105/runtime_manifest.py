from pathlib import Path
from importlib import metadata
import hashlib,json,platform,socket,sys
import torch
root=Path(__file__).resolve().parent
files=['moonep/api.py','moonep/__init__.py','moonep/planner_config.py','moonep/planning.py','moonep/sampled_group.py','moonep/sampled_residency.py','moonep/sampled_exchange.py','moonep/sampled_materialize.py','moonep/csrc/sampled_residency.cu','moonep/csrc/sampled_exchange.cu','moonep/csrc/sampled_materialize.cu','tests/test_planner_config.py','tests/test_sampled_configurable.py','tests/test_sampled_configurable_weighted.py','tests/test_sampled_residency_configurable.py','tests/sampled_residency_reference.py','tests/test_sampled_group.py','tests/test_sampled_weights.py','tests/sampled_components.py','tests/benchmark_sampled_group.py','csrc/bindings.cu']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
record=dict(base_commit='45c2432a8da6f465028e833464f0ec25379cce0e',instance='CI-情境智能/omni-dev',host=socket.gethostname(),python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,nccl=torch.cuda.nccl.version(),platform=platform.platform(),packages={d.metadata['Name']:d.version for d in metadata.distributions() if any(x in d.metadata.get('Name','').lower() for x in ['cutlass','cuda','ninja'])},gpu_count=torch.cuda.device_count(),gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],source_sha256={f:sha(root/f) for f in files},native_binary={p.name:sha(p) for p in (root/'moonep').glob('_C*.so')})
p=root/'evidence/runtime_manifest.json'
if p.exists():raise FileExistsError(p)
p.write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n')
print('manifest:',p)
