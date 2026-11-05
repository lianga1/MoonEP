// Two physical ranks, top-8: identical canonical bundle proposals in both paths.
// sort: reread (expert, owner), CUB stable radix sort, then commit.
// prefix: cache executability masks, retain compact owner mask, use stable
// block-local bucket ordinals and bucket-wise block scans instead of sorting.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/block/block_scan.cuh>
#include <cstdint>

namespace {
constexpr int K=8, R=2, STRIDE=9, BUCKETS=72, THREADS=256;
using Key=unsigned long long;
#define CHECK_I32(x) TORCH_CHECK((x).is_cuda() && (x).is_contiguous() && (x).scalar_type()==at::kInt, #x " requires contiguous CUDA int32")

__device__ __forceinline__ unsigned first_bits(unsigned bits,int size) {
  unsigned out=0;
  #pragma unroll
  for(int i=0;i<4;++i) if(i<size) { unsigned bit=bits & (0u-bits);out|=bit;bits^=bit; }
  return out;
}

struct Proposal { int key; unsigned mask; };

__device__ __forceinline__ Proposal choose(unsigned old0,unsigned exec0,unsigned exec1,int source,int token,int step) {
  int best=-1,key=BUCKETS;unsigned move=0;
  const int preferred=(token+step)&3;
  const bool ascending=((token+step)&1)==0;
  #pragma unroll
  for(int a=0;a<2;++a) {
    const int b=1-a,edge=a*2+b;
    const unsigned old_a=a? (old0^255u):old0;
    const unsigned old_b=old_a^255u;
    const unsigned available=old_a & (b?exec1:exec0);
    if(!available)continue;
    const int ca=__popc(old_a),cb=__popc(old_b),na=__popc(available);
    const int tie=int((a<b)==ascending);
    #pragma unroll
    for(int q=0;q<4;++q) {
      const int size=q<3?1<<q:ca;
      if(q<3) { if(size>na)continue; }
      else if(na!=ca || size==1 || size==2 || size==4)continue;
      const int gain=int(a!=source && size==ca)-int(b!=source && cb==0);
      if(gain<0)continue;
      const int size_tie=3-((q-preferred+4)&3);
      const int score=((gain*4+size_tie)*STRIDE+cb)*2+tie;
      // Explicit final tie preserves strict > in ascending CPU edge order.
      const int ordered=score*4+3-edge;
      if(ordered>best) { best=ordered;key=(edge*2+gain)*STRIDE+size;move=q==3?available:first_bits(available,size); }
    }
  }
  return {key,move};
}

__global__ void pack_resident(const bool* support,int* resident,int E) {
  int e=blockIdx.x*blockDim.x+threadIdx.x;
  if(e<E)resident[e]=int(support[e])|(int(support[E+e])<<1);
}

__global__ void cache_executable(const int* routes,const int* resident,int* exec0,int* exec1,int N) {
  int linear=blockIdx.x*blockDim.x+threadIdx.x,t=linear/8,lane=threadIdx.x&7;
  int shift=(threadIdx.x&31)&~7;
  int ex=t<N?resident[routes[t*8+lane]]:0;
  unsigned a=(__ballot_sync(0xffffffffu,t<N && (ex&1))>>shift)&255;
  unsigned b=(__ballot_sync(0xffffffffu,t<N && (ex&2))>>shift)&255;
  if(t<N && lane==0) { exec0[t]=a;exec1[t]=b; }
}

__global__ void reset_compact(const int* seed,int* owner,int* old0,int N) {
  int linear=blockIdx.x*blockDim.x+threadIdx.x,t=linear/8,lane=threadIdx.x&7;
  int shift=(threadIdx.x&31)&~7;
  int x=t<N?seed[t*8+lane]:-1;
  unsigned m=(__ballot_sync(0xffffffffu,t<N && x==0)>>shift)&255;
  if(t<N) { owner[t*8+lane]=x;if(lane==0)old0[t]=m; }
}

__global__ void propose_sort_kernel(const int* routes,const int* owner,const int* resident,
 Key* keys,int* moves,int N,int token_bits,int source,int global_offset,int step) {
  int linear=blockIdx.x*blockDim.x+threadIdx.x,t=linear/8,lane=threadIdx.x&7;
  int shift=(threadIdx.x&31)&~7;
  int old=t<N?owner[t*8+lane]:-1;
  int ex=t<N?resident[routes[t*8+lane]]:0;
  unsigned om=(__ballot_sync(0xffffffffu,t<N && old==0)>>shift)&255;
  unsigned a=(__ballot_sync(0xffffffffu,t<N && (ex&1))>>shift)&255;
  unsigned b=(__ballot_sync(0xffffffffu,t<N && (ex&2))>>shift)&255;
  if(t<N && lane==0) {
    Proposal p=choose(om,a,b,source,global_offset+t,step);
    keys[t]=(Key(p.key)<<token_bits)|Key(t);moves[t]=p.mask;
  }
}

__global__ void sorted_boundaries(const Key* sorted,int* starts,int* ends,int N,int token_bits) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N)return;
  int key=int(sorted[i]>>token_bits);
  if(i==0 || int(sorted[i-1]>>token_bits)!=key)starts[key]=i;
  if(i==N-1 || int(sorted[i+1]>>token_bits)!=key)ends[key]=i+1;
}
__global__ void sorted_counts(const int* starts,const int* ends,int* counts) {
  int b=threadIdx.x;if(b<BUCKETS)counts[b]=ends[b]-starts[b];
}

__global__ void propose_prefix_kernel(const int* old0,const int* exec0,const int* exec1,
 int* keys,int* moves,int* ordinals,int* block_counts,int N,int blocks,int source,int global_offset,int step) {
  __shared__ int wh[8][BUCKETS+1];
  const int lane=threadIdx.x&31,warp=threadIdx.x/32;
  for(int i=threadIdx.x;i<8*(BUCKETS+1);i+=blockDim.x)reinterpret_cast<int*>(wh)[i]=0;
  __syncthreads();
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  Proposal p={BUCKETS,0};
  if(t<N)p=choose(old0[t],exec0[t],exec1[t],source,global_offset+t,step);
  // All 32 lanes participate, including tail padding carrying the sentinel.
  unsigned peers=__match_any_sync(0xffffffffu,p.key);
  unsigned lower=(1u<<lane)-1u;
  int local=__popc(peers&lower);
  if(lane==__ffs(peers)-1)wh[warp][p.key]=__popc(peers);
  __syncthreads();
  if(t<N) {
    int ordinal=local;
    #pragma unroll
    for(int w=0;w<8;++w)if(w<warp)ordinal+=wh[w][p.key];
    keys[t]=p.key;moves[t]=p.mask;ordinals[t]=ordinal;
  }
  if(threadIdx.x<BUCKETS) {
    int count=0;
    #pragma unroll
    for(int w=0;w<8;++w)count+=wh[w][threadIdx.x];
    block_counts[threadIdx.x*blocks+blockIdx.x]=count;
  }
}

__global__ void scan_blocks(const int* block_counts,int* block_prefix,int* counts,int blocks) {
  using Scan=cub::BlockScan<int,THREADS>;
  __shared__ typename Scan::TempStorage temp;
  __shared__ int carry;
  const int bucket=blockIdx.x;
  if(threadIdx.x==0)carry=0;
  __syncthreads();
  for(int base=0;base<blocks;base+=THREADS) {
    int i=base+threadIdx.x;
    int value=i<blocks?block_counts[bucket*blocks+i]:0;
    int prefix,total;
    Scan(temp).ExclusiveSum(value,prefix,total);
    int prior=carry;
    if(i<blocks)block_prefix[bucket*blocks+i]=prior+prefix;
    __syncthreads();
    if(threadIdx.x==0)carry=prior+total;
    __syncthreads();
  }
  if(threadIdx.x==0)counts[bucket]=carry;
}

// One tiny CTA folds both source count tables and computes same-size quotas.
// All ranks derive identical accept[] and global stats; source prefix is rank-local.
__global__ void quota_kernel(const int* gathered,int* prefix,int* accept,
 unsigned long long* stats,int rank) {
  __shared__ int total[BUCKETS];
  int b=threadIdx.x;
  if(b<BUCKETS) {
    total[b]=gathered[b]+gathered[BUCKETS+b];
    prefix[b]=rank?gathered[b]:0;accept[b]=0;
  }
  if(b<4)stats[b]=0;
  __syncthreads();
  if(b<8) {
    int size=b+1,ab0=18+size,ab1=27+size,ba0=36+size,ba1=45+size;
    int x0=total[ab0],x1=total[ab1],y0=total[ba0],y1=total[ba1];
    int n=min(min(x0+x1,y0+y1),max(x1,y1));
    int p=min(n,x1),q=min(n,y1);
    accept[ab0]=n-p;accept[ab1]=p;accept[ba0]=n-q;accept[ba1]=q;
    atomicAdd(stats+0,(unsigned long long)n);
    atomicAdd(stats+1,(unsigned long long)(2*n*size));
    atomicAdd(stats+2,(unsigned long long)(2*n));
    atomicAdd(stats+3,(unsigned long long)(p+q));
  }
}

__global__ void commit_sort_kernel(const Key* sorted,const int* starts,const int* moves,
 const int* prefix,const int* accept,int* owner,int N,int token_bits) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N)return;
  Key p=sorted[i];int key=int(p>>token_bits);
  if(key>=BUCKETS || i-starts[key]+prefix[key]>=accept[key])return;
  int t=int(p&((Key(1)<<token_bits)-1)),dest=(key/STRIDE/2)%2;
  unsigned bits=moves[t];
  while(bits) { int k=__ffs(bits)-1;bits&=bits-1;owner[t*8+k]=dest; }
}

__global__ void commit_prefix_kernel(const int* keys,const int* moves,const int* ordinals,
 const int* block_prefix,const int* source_prefix,const int* accept,int* old0,int* owner,int N,int blocks) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=N)return;
  int key=keys[t];if(key>=BUCKETS)return;
  int ordinal=ordinals[t]+block_prefix[key*blocks+t/THREADS]+source_prefix[key];
  if(ordinal>=accept[key])return;
  int dest=(key/STRIDE/2)%2;unsigned bits=moves[t];
  old0[t]=dest?(unsigned(old0[t])&~bits):(unsigned(old0[t])|bits);
  while(bits) { int k=__ffs(bits)-1;bits&=bits-1;owner[t*8+k]=dest; }
}
} // namespace

int64_t workspace_size(const at::Tensor& keys,const at::Tensor& sorted,int64_t token_bits) {
  c10::cuda::CUDAGuard guard(keys.device());size_t bytes=0;
  C10_CUDA_CHECK(cub::DeviceRadixSort::SortKeys(nullptr,bytes,
    reinterpret_cast<const Key*>(keys.data_ptr<int64_t>()),reinterpret_cast<Key*>(sorted.data_ptr<int64_t>()),
    keys.numel(),int(token_bits),int(token_bits)+7,c10::cuda::getCurrentCUDAStream()));
  return bytes;
}

void prepare(const at::Tensor& routes,const at::Tensor& support,at::Tensor& resident,
 at::Tensor& exec0,at::Tensor& exec1,bool cached) {
  CHECK_I32(routes);CHECK_I32(resident);
  TORCH_CHECK(support.is_cuda() && support.is_contiguous() && support.scalar_type()==at::kBool && support.dim()==2 && support.size(0)==2,"support must be CUDA bool [2,E]");
  c10::cuda::CUDAGuard guard(routes.device());auto stream=c10::cuda::getCurrentCUDAStream();
  int E=support.size(1),N=routes.size(0);
  pack_resident<<<(E+255)/256,256,0,stream>>>(support.data_ptr<bool>(),resident.data_ptr<int>(),E);
  if(cached)cache_executable<<<(N*8+255)/256,256,0,stream>>>(routes.data_ptr<int>(),resident.data_ptr<int>(),exec0.data_ptr<int>(),exec1.data_ptr<int>(),N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void reset(const at::Tensor& seed,at::Tensor& owner,at::Tensor& old0,bool cached) {
  CHECK_I32(seed);CHECK_I32(owner);
  TORCH_CHECK(seed.sizes()==owner.sizes() && seed.dim()==2 && seed.size(1)==8,"seed/owner need matching [S,8]");
  c10::cuda::CUDAGuard guard(owner.device());
  if(cached)reset_compact<<<(seed.numel()+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(seed.data_ptr<int>(),owner.data_ptr<int>(),old0.data_ptr<int>(),seed.size(0));
  else owner.copy_(seed);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void propose_sort(const at::Tensor& routes,const at::Tensor& owner,const at::Tensor& resident,
 at::Tensor& keys,at::Tensor& sorted,at::Tensor& moves,at::Tensor& starts,at::Tensor& ends,
 at::Tensor& counts,at::Tensor& workspace,int64_t token_bits,int64_t source,int64_t offset,int64_t step) {
  c10::cuda::CUDAGuard guard(owner.device());auto stream=c10::cuda::getCurrentCUDAStream();int N=owner.size(0);
  C10_CUDA_CHECK(cudaMemsetAsync(starts.data_ptr<int>(),0,(BUCKETS+1)*4,stream));
  C10_CUDA_CHECK(cudaMemsetAsync(ends.data_ptr<int>(),0,(BUCKETS+1)*4,stream));
  propose_sort_kernel<<<(N*8+255)/256,256,0,stream>>>(routes.data_ptr<int>(),owner.data_ptr<int>(),resident.data_ptr<int>(),
    reinterpret_cast<Key*>(keys.data_ptr<int64_t>()),moves.data_ptr<int>(),N,token_bits,source,offset,step);
  size_t bytes=workspace.numel();
  C10_CUDA_CHECK(cub::DeviceRadixSort::SortKeys(workspace.data_ptr<uint8_t>(),bytes,
    reinterpret_cast<const Key*>(keys.data_ptr<int64_t>()),reinterpret_cast<Key*>(sorted.data_ptr<int64_t>()),N,
    int(token_bits),int(token_bits)+7,stream));
  sorted_boundaries<<<(N+255)/256,256,0,stream>>>(reinterpret_cast<const Key*>(sorted.data_ptr<int64_t>()),starts.data_ptr<int>(),ends.data_ptr<int>(),N,token_bits);
  sorted_counts<<<1,128,0,stream>>>(starts.data_ptr<int>(),ends.data_ptr<int>(),counts.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void propose_prefix(const at::Tensor& old0,const at::Tensor& exec0,const at::Tensor& exec1,
 at::Tensor& keys,at::Tensor& moves,at::Tensor& ordinals,at::Tensor& block_counts,
 at::Tensor& block_prefix,at::Tensor& counts,int64_t source,int64_t offset,int64_t step) {
  c10::cuda::CUDAGuard guard(old0.device());auto stream=c10::cuda::getCurrentCUDAStream();int N=old0.numel(),blocks=(N+255)/256;
  propose_prefix_kernel<<<blocks,256,0,stream>>>(old0.data_ptr<int>(),exec0.data_ptr<int>(),exec1.data_ptr<int>(),keys.data_ptr<int>(),moves.data_ptr<int>(),ordinals.data_ptr<int>(),block_counts.data_ptr<int>(),N,blocks,source,offset,step);
  scan_blocks<<<BUCKETS,256,0,stream>>>(block_counts.data_ptr<int>(),block_prefix.data_ptr<int>(),counts.data_ptr<int>(),blocks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void commit(const at::Tensor& keys,const at::Tensor& sorted,const at::Tensor& moves,const at::Tensor& ordinals,
 const at::Tensor& starts,const at::Tensor& block_prefix,const at::Tensor& gathered,at::Tensor& prefix,
 at::Tensor& accept,at::Tensor& old0,at::Tensor& owner,at::Tensor& stats,int64_t token_bits,int64_t rank,int64_t step,bool cached) {
  TORCH_CHECK(gathered.is_cuda() && gathered.is_contiguous() && gathered.scalar_type()==at::kInt && gathered.numel()==2*BUCKETS,"gathered needs 144 CUDA int32 values, source 0 then source 1");
  TORCH_CHECK(rank>=0 && rank<2 && step>=0 && step<stats.size(0),"invalid rank/step");
  c10::cuda::CUDAGuard guard(owner.device());auto stream=c10::cuda::getCurrentCUDAStream();int N=owner.size(0);
  quota_kernel<<<1,128,0,stream>>>(gathered.data_ptr<int>(),prefix.data_ptr<int>(),accept.data_ptr<int>(),reinterpret_cast<unsigned long long*>(stats.data_ptr<int64_t>()+step*4),rank);
  if(cached)commit_prefix_kernel<<<(N+255)/256,256,0,stream>>>(keys.data_ptr<int>(),moves.data_ptr<int>(),ordinals.data_ptr<int>(),block_prefix.data_ptr<int>(),prefix.data_ptr<int>(),accept.data_ptr<int>(),old0.data_ptr<int>(),owner.data_ptr<int>(),N,(N+255)/256);
  else commit_sort_kernel<<<(N+255)/256,256,0,stream>>>(reinterpret_cast<const Key*>(sorted.data_ptr<int64_t>()),starts.data_ptr<int>(),moves.data_ptr<int>(),prefix.data_ptr<int>(),accept.data_ptr<int>(),owner.data_ptr<int>(),N,token_bits);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("workspace_size",&workspace_size);m.def("prepare",&prepare);m.def("reset",&reset);
  m.def("propose_sort",&propose_sort);m.def("propose_prefix",&propose_prefix);m.def("commit",&commit);
}
