"""Exhaustive integer 1/2 volume formula against previous DP, including quotas."""
import sys,itertools
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'analysis'))
from vendor.aggregate_exchange import bounded_dp

def formula(c,w):
 A,B,C,D=c[1,1],c[1,2],c[0,1],c[0,2]
 lo=max(0,w-2*(B+D));hi=min(A+C,w)
 if lo%2!=w%2:lo+=1
 if hi%2!=w%2:hi-=1
 if lo>hi:return -100000000,None
 best=(-100000000,0,0);out=None
 for p in [lo,hi,A,w-2*B]:
  for d in [-1,0,1]:
   s=max(lo,min(hi,p+d))
   if s%2!=w%2:continue
   q=(w-s)//2;p1=min(A,s);p2=min(B,q);z2=q-p2
   score=(p1+p2,-p2,-z2)
   if out is None or score>best:best=score;out=np.array([[0,s-p1,z2],[0,p1,p2]])
 return best[0],out
checks=0
for counts in itertools.product(range(4),repeat=4):
 c=np.array([[0,counts[0],counts[1]],[0,counts[2],counts[3]]],dtype=np.int64)
 cap=20;dp,choice=bounded_dp(c,cap,2)
 for w in range(cap+1):
  gain,out=formula(c,w);assert gain==dp[w],(c,w,gain,dp[w])
  if out is not None:
   rec=np.zeros((2,3),np.int64);v=w
   for i in range(3,-1,-1):
    s=i%2+1;g=i//2;n=choice[i,v];rec[g,s]=n;v-=n*s
   assert np.array_equal(out,rec),(c,w,out,rec)
  checks+=1
print('PASSED',checks,'volume gains and exact quota checks')
