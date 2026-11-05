"""Existing exact GPU bundle exchange with same trace/support as new candidates."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bucket8'))
from bucket_cuda import BucketCudaPlanner,extension

class BundlePlanner(BucketCudaPlanner):
 def __init__(self,routes,support,per_source,rounds=4):
  super().__init__(routes.reshape(-1,8).contiguous(),support,per_source=per_source,rounds=rounds,coordinated=False)
 def run(self,seed):
  return super().run(seed.reshape(-1,8).contiguous())
