import os
import torch
from moonep.sampled_exchange import extension as exchange
from moonep.sampled_materialize import extension as materialize
from moonep.sampled_residency import extension as residency
from tests.test_sampled_configurable import CASES
print('Torch',torch.__version__,'CUDA',torch.version.cuda,flush=True)
print('Shared optin',torch.cuda.get_device_properties(0).shared_memory_per_block_optin,flush=True)
materialize()
for E,K,D,G,L,S,B in CASES:
 print('compile',E,K,min(D,8*S),G,flush=True)
 residency(E,K,min(D,8*S),G)
 exchange(K)
print('CUDA extensions ready',flush=True)
