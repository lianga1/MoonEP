#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>
#include <cub/cub.cuh>
#include <vector>
constexpr int K=8,R=8,NC=4*(K+1),BAD=-100000000;
using torch::Tensor;

__global__ void make_states(const int* x,const int* owner,int* state,int* ids,int T,int S){
 int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=T)return;
 int ee[K],oo[K];
 #pragma unroll
 for(int k=0;k<K;k++){ee[k]=x[t*K+k];oo[k]=owner[t*K+k];}
 #pragma unroll
 for(int i=1;i<K;i++){int e=ee[i],o=oo[i],j=i-1;while(j>=0&&ee[j]>e){ee[j+1]=ee[j];oo[j+1]=oo[j];j--;}ee[j+1]=e;oo[j+1]=o;}
 state[t*17]=t/S;
 #pragma unroll
 for(int k=0;k<K;k++){state[t*17+1+k*2]=ee[k];state[t*17+2+k*2]=oo[k];}ids[t]=t;
}
struct StateLess{const int* s;__host__ __device__ bool operator()(int a,int b)const{
 #pragma unroll
 for(int c=0;c<17;c++){int x=s[a*17+c],y=s[b*17+c];if(x!=y)return x<y;}return a<b;
}};
__global__ void mark_heads(const int* ids,const int* state,int* heads,int T){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=T)return;bool eq=i>0;if(i>0){int a=ids[i],b=ids[i-1];
 #pragma unroll
 for(int c=0;c<17;c++)eq=eq&&(state[a*17+c]==state[b*17+c]);}heads[i]=!eq;}
__global__ void groups(const int* ids,const int* heads,const int* scan,int* reps,int* begins,int* inverse,int* counts,int T){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=T)return;int g=scan[i]-1;inverse[ids[i]]=g;if(heads[i]){reps[g]=ids[i];begins[g]=i;}atomicAdd(counts+g,1);}
__global__ void ordinals(const int* ids,const int* scan,const int* begins,int* ord,int T){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<T)ord[ids[i]]=i-begins[scan[i]-1];}
// Materialize each representative once into a column-major class table.
// The following 28 card-pair scans read neighboring class IDs coalescently.
__global__ void compact_states(const int* state,const int* reps,const int* scan,int* compact,int T){
 int g=blockIdx.x*blockDim.x+threadIdx.x;if(g>=T||g>=scan[T-1])return;int t=reps[g];
 #pragma unroll
 for(int c=0;c<17;c++)compact[c*T+g]=state[t*17+c];
}
__global__ void propose(const int* state,const int* reps,const int* remaining,const int* scan,const bool* support,int* keys,int T,int E,int a,int b,int step,const int* compact,bool coalesced){
 int g=blockIdx.x*blockDim.x+threadIdx.x;if(g>=T)return;keys[g]=-1;if(g>=scan[T-1]||remaining[g]==0)return;
 const int* st=state+17*reps[g];int src=coalesced?compact[g]:st[0];int na=0,nb=0;bool lab=true,lba=true;
 #pragma unroll
 for(int k=0;k<K;k++){int e=coalesced?compact[(1+2*k)*T+g]:st[1+2*k],o=coalesced?compact[(2+2*k)*T+g]:st[2+2*k];if(o==a){na++;lab=lab&&support[b*E+e];}if(o==b){nb++;lba=lba&&support[a*E+e];}}
 int va=int(a!=src)-int(b!=src&&nb==0),vb=int(b!=src)-int(a!=src&&na==0);bool ab=na>0&&lab&&va>=0,ba=nb>0&&lba&&vb>=0;if(!ab&&!ba)return;
 int direction=0;if(!ab)direction=1;else if(ba){int sa=va*(K+1)+(K-na),sb=vb*(K+1)+(K-nb);if(sb>sa||(sb==sa&&((g+step+a+b)%2)))direction=1;}
 keys[g]=direction?(2+vb)*(K+1)+nb:va*(K+1)+na;
}
// A block per category gives deterministic exclusive counts in lexicographic
// state order. Group loads are coalesced; no atomic ranking nondeterminism.
__global__ void category_scan(const int* keys,const int* remaining,const int* group_scan,int* offsets,int* counts,int T){
 typedef cub::BlockScan<int,256> Scan;__shared__ typename Scan::TempStorage temp;__shared__ int carry;
 int key=blockIdx.x,tid=threadIdx.x;if(tid==0)carry=0;__syncthreads();int G=group_scan[T-1];
 for(int base=0;base<G;base+=256){int g=base+tid;int v=(g<G&&keys[g]==key)?remaining[g]:0;int prefix,total;Scan(temp).ExclusiveSum(v,prefix,total);int c=carry;if(g<G&&keys[g]==key)offsets[g]=c+prefix;__syncthreads();if(tid==0)carry=c+total;__syncthreads();}
 if(tid==0)counts[key]=carry;
}
__device__ int pair_cap(const int* cnt){int a=0,b=0;for(int s=1;s<=K;s++){a+=s*(cnt[s]+cnt[K+1+s]);b+=s*(cnt[2*(K+1)+s]+cnt[3*(K+1)+s]);}return min(a,b);}
// Exact bounded knapsack with monotonic queues, matching NumPy/Numba tie rules.
// Two CTAs process opposing directions; each residue chain has one thread.
__global__ void solve_dp(const int* cnt,int* workspace,int* choices,int* queues,int stride){
 int dir=blockIdx.x,tid=threadIdx.x,cap=pair_cap(cnt);int* old=workspace+(dir*2)*stride;int* next=old+stride;int* queue=queues+dir*stride;
 for(int w=tid;w<=cap;w+=blockDim.x)old[w]=(w==0?0:BAD);__syncthreads();
 for(int type=0;type<2*K;type++){
  int gain=type/K,size=type%K+1,n=cnt[(dir*2+gain)*(K+1)+size];
  if(tid<size&&tid<=cap){int residue=tid,head=0,tail=0;for(int j=0;j<=(cap-residue)/size;j++){
   while(head<tail&&queue[residue+head*size]<j-n)head++;
   int pos=residue+j*size,val=old[pos]-j*gain;
   while(head<tail){int q=queue[residue+(tail-1)*size];if(old[residue+q*size]-q*gain>val)break;tail--;}
   queue[residue+tail*size]=j;tail++;int q=queue[residue+head*size],previous=old[residue+q*size];
   next[pos]=previous>BAD/2?previous+(j-q)*gain:BAD;choices[(dir*2*K+type)*stride+pos]=previous>BAD/2?j-q:0;
  }}
  __syncthreads();int* swap=old;old=next;next=swap;
 }
 // Sixteen types => final 'old' is direction's first workspace buffer.
}
__global__ void reconstruct(const int* counts,const int* dp,const int* choices,int* accept,int stride){
 __shared__ int bestv[256],bestw[256];int tid=threadIdx.x,cap=pair_cap(counts);int value=0,weight=0;
 for(int w=tid+1;w<=cap;w+=256){int v=dp[w]+dp[2*stride+w];if(v>value||(v==value&&w<weight)){value=v;weight=w;}}
 bestv[tid]=value;bestw[tid]=weight;__syncthreads();
 for(int d=128;d;d>>=1){if(tid<d){int v=bestv[tid+d],w=bestw[tid+d];if(v>bestv[tid]||(v==bestv[tid]&&w<bestw[tid])){bestv[tid]=v;bestw[tid]=w;}}__syncthreads();}
 for(int i=tid;i<NC;i+=256)accept[i]=0;__syncthreads();
 if(tid<2&&bestv[0]>0){int w=bestw[0],dir=tid;for(int type=2*K-1;type>=0;type--){int size=type%K+1,gain=type/K,n=choices[(dir*2*K+type)*stride+w];accept[(dir*2+gain)*(K+1)+size]=n;w-=size*n;}}
}
__global__ void allocate(const int* keys,const int* offsets,const int* accept,int* remaining,int* actions,int T,int a,int b){int g=blockIdx.x*blockDim.x+threadIdx.x;if(g>=T)return;int key=keys[g];if(key<0)return;int n=min(remaining[g],max(0,accept[key]-offsets[g]));if(n){int edge=key<2*(K+1)?a*R+b:b*R+a;actions[edge*T+g]+=n;remaining[g]-=n;}}
__global__ void expand_kernel(const int* owner,const int* inverse,const int* ordinal,const int* actions,int* out,int T){
 int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=T)return;int g=inverse[t],rank=ordinal[t],chosen=-1;
 for(int edge=0;edge<R*R;edge++){int n=actions[edge*T+g];if(rank<n){chosen=edge;break;}rank-=n;}
 #pragma unroll
 for(int k=0;k<K;k++){int v=owner[t*K+k];if(chosen>=0&&v==chosen/R)v=chosen%R;out[t*K+k]=v;}
}

std::vector<Tensor> aggregate_run(Tensor x,Tensor owner,Tensor support,int rounds,bool coalesced){
 TORCH_CHECK(x.is_cuda()&&owner.is_cuda()&&support.is_cuda(),"CUDA tensors required");
 TORCH_CHECK(x.scalar_type()==torch::kInt32&&owner.scalar_type()==torch::kInt32&&support.scalar_type()==torch::kBool,"int32 x/owner and bool support required");
 TORCH_CHECK(x.is_contiguous()&&owner.is_contiguous()&&support.is_contiguous(),"contiguous required");
 TORCH_CHECK(x.dim()==3&&x.size(0)==8&&x.size(2)==8,"requires [8,S,8]");
 c10::cuda::CUDAGuard guard(x.device());auto stream=c10::cuda::getCurrentCUDAStream();int T=x.numel()/K,S=x.size(1),E=support.size(1),stride=T*K+1;auto opt=x.options();
 Tensor state=torch::empty({T,17},opt),ids=torch::empty({T},opt),heads=torch::empty({T},opt),scan=torch::empty({T},opt);
 Tensor reps=torch::empty({T},opt),begins=torch::empty({T},opt),inverse=torch::empty({T},opt),ordinal=torch::empty({T},opt),remaining=torch::empty({T},opt),keys=torch::empty({T},opt),offsets=torch::empty({T},opt);
 Tensor compact=torch::empty({17,T},opt);
 Tensor counts=torch::empty({NC},opt),accept=torch::empty({NC},opt),actions=torch::empty({R*R,T},opt);
 Tensor dp=torch::empty({4,stride},opt),choices=torch::empty({2,2*K,stride},opt),queues=torch::empty({2,stride},opt);
 Tensor cur=owner,other=torch::empty_like(owner);size_t scan_bytes=0;cub::DeviceScan::InclusiveSum(nullptr,scan_bytes,heads.data_ptr<int>(),scan.data_ptr<int>(),T,stream);
 Tensor scan_tmp=torch::empty({(int64_t)scan_bytes},opt.dtype(torch::kUInt8));int pairs[28][2],pi=0;for(int a=0;a<R;a++)for(int b=a+1;b<R;b++){pairs[pi][0]=a;pairs[pi++][1]=b;}
 for(int step=0;step<rounds;step++){
  make_states<<<(T+255)/256,256,0,stream>>>(x.data_ptr<int>(),cur.data_ptr<int>(),state.data_ptr<int>(),ids.data_ptr<int>(),T,S);
  thrust::sort(thrust::cuda::par.on(stream),ids.data_ptr<int>(),ids.data_ptr<int>()+T,StateLess{state.data_ptr<int>()});
  mark_heads<<<(T+255)/256,256,0,stream>>>(ids.data_ptr<int>(),state.data_ptr<int>(),heads.data_ptr<int>(),T);
  cub::DeviceScan::InclusiveSum(scan_tmp.data_ptr(),scan_bytes,heads.data_ptr<int>(),scan.data_ptr<int>(),T,stream);
  cudaMemsetAsync(remaining.data_ptr(),0,T*sizeof(int),stream);cudaMemsetAsync(actions.data_ptr(),0,R*R*T*sizeof(int),stream);
  groups<<<(T+255)/256,256,0,stream>>>(ids.data_ptr<int>(),heads.data_ptr<int>(),scan.data_ptr<int>(),reps.data_ptr<int>(),begins.data_ptr<int>(),inverse.data_ptr<int>(),remaining.data_ptr<int>(),T);
  ordinals<<<(T+255)/256,256,0,stream>>>(ids.data_ptr<int>(),scan.data_ptr<int>(),begins.data_ptr<int>(),ordinal.data_ptr<int>(),T);
  if(coalesced)compact_states<<<(T+255)/256,256,0,stream>>>(state.data_ptr<int>(),reps.data_ptr<int>(),scan.data_ptr<int>(),compact.data_ptr<int>(),T);
  for(int pp=0;pp<28;pp++){int pair=(step*7+pp)%28,a=pairs[pair][0],b=pairs[pair][1];
   propose<<<(T+255)/256,256,0,stream>>>(state.data_ptr<int>(),reps.data_ptr<int>(),remaining.data_ptr<int>(),scan.data_ptr<int>(),support.data_ptr<bool>(),keys.data_ptr<int>(),T,E,a,b,step,compact.data_ptr<int>(),coalesced);
   category_scan<<<NC,256,0,stream>>>(keys.data_ptr<int>(),remaining.data_ptr<int>(),scan.data_ptr<int>(),offsets.data_ptr<int>(),counts.data_ptr<int>(),T);
   solve_dp<<<2,32,0,stream>>>(counts.data_ptr<int>(),dp.data_ptr<int>(),choices.data_ptr<int>(),queues.data_ptr<int>(),stride);
   reconstruct<<<1,256,0,stream>>>(counts.data_ptr<int>(),dp.data_ptr<int>(),choices.data_ptr<int>(),accept.data_ptr<int>(),stride);
   allocate<<<(T+255)/256,256,0,stream>>>(keys.data_ptr<int>(),offsets.data_ptr<int>(),accept.data_ptr<int>(),remaining.data_ptr<int>(),actions.data_ptr<int>(),T,a,b);
  }
  expand_kernel<<<(T+255)/256,256,0,stream>>>(cur.data_ptr<int>(),inverse.data_ptr<int>(),ordinal.data_ptr<int>(),actions.data_ptr<int>(),other.data_ptr<int>(),T);
  cur=other;other=torch::empty_like(owner);
 }
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {cur,scan};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&aggregate_run,"Exact aggregated mixed-size exchange (CUDA)");}
