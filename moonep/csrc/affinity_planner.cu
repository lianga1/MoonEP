// GPU并行版：保留MoonEP完整均衡种子、一次动态驻留、严格正收益等量交换。
// 一轮rank对内每个token只提交一个候选，因此并行接受的交换不会共享token。
// 与CPU逐项串行扫描顺序不同；不声称逐对重现CPU启发式结果。
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <vector>
#include <cooperative_groups.h>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_reduce.cuh>

using at::Tensor;

// CUB 3.x removed cub::Max; this device functor also works with CUDA 12.x.
struct IntMax {
  __device__ __forceinline__ int operator()(int a, int b) const { return a > b ? a : b; }
};
#define PTR(x) (x).data_ptr<int>()
#define STREAM at::cuda::getCurrentCUDAStream()
#define CHECK(x) TORCH_CHECK((x).is_cuda() && (x).is_contiguous() && (x).scalar_type()==at::kInt, #x " must be contiguous CUDA int32")

// GPU构造与CPU参考相同的FIFO种子；block内match_any求同专家的稳定局部序号。
__global__ void seed_hist_kernel(const int* experts,int* ordinal,int* hist,int M,int E) {
  extern __shared__ int warp_hist[];
  int tid=threadIdx.x,lane=tid&31,warp=tid/32,p=blockIdx.x*256+tid;
  for(int i=tid;i<8*E;i+=256) warp_hist[i]=0;
  __syncthreads();
  int e=p<M?experts[p]:-1;
  unsigned peers=__match_any_sync(0xffffffff,e);
  if(p<M && lane==__ffs(peers)-1) warp_hist[warp*E+e]=__popc(peers);
  __syncthreads();
  if(p<M) {
    int n=__popc(peers&((1u<<lane)-1));
    for(int w=0;w<warp;w++) n+=warp_hist[w*E+e];
    ordinal[p]=n;
  }
  for(int j=tid;j<E;j+=256) {
    int n=0;for(int w=0;w<8;w++) n+=warp_hist[w*E+j];
    hist[blockIdx.x*E+j]=n;
  }
}

// 每expert一CTA，对chunk直方图分段并行前缀，避免单线程遍历全部100K输入。
__global__ void seed_prefix_kernel(int* hist,int* demand,int E,int blocks) {
  using Scan=cub::BlockScan<int,256>;
  __shared__ Scan::TempStorage temp;
  __shared__ int carry;
  int e=blockIdx.x,tid=threadIdx.x;
  if(tid==0) carry=0;
  __syncthreads();
  for(int first=0;first<blocks;first+=256) {
    int b=first+tid,n=b<blocks?hist[b*E+e]:0,prefix;
    Scan(temp).ExclusiveSum(n,prefix);
    int base=carry;
    if(b<blocks) hist[b*E+e]=base+prefix;
    __syncthreads();
    if(tid==255) carry=base+prefix+n;
    __syncthreads();
  }
  if(tid==0) demand[e]=carry;
}

__global__ void seed_alloc_kernel(const int* demand,int* alloc,int R,int E,int target) {
  __shared__ int balance[16];
  int epn=E/R,tid=threadIdx.x;
  if(tid<R) balance[tid]=-target;
  __syncthreads();
  for(int i=tid;i<R*E;i+=blockDim.x) {
    int r=i/E,e=i%E,n=e/epn==r?demand[e]:0;
    alloc[i]=n;
    if(n) atomicAdd(balance+r,n);
  }
  __syncthreads();
  if(tid) return;
  int z[256],remaining[1024],quotas[16];
  for(int i=0;i<R*R;i++) z[i]=0;
  for(int i=0;i<2*R;i++) {
    int hot=0,cold=0;
    for(int r=1;r<R;r++){if(balance[r]>balance[hot])hot=r;if(balance[r]<balance[cold])cold=r;}
    if(balance[hot]<=0) break;
    int amount=-balance[cold];z[hot*R+cold]=amount;balance[hot]-=amount;balance[cold]=0;
  }
  for(int h=0;h<R;h++) {
    for(int e=0;e<epn;e++) remaining[e]=demand[h*epn+e];
    for(int r=0;r<R;r++) quotas[r]=z[h*R+r];
    for(int iteration=0;iteration<E+R;iteration++) {
      int d=0,best=0;
      for(int r=1;r<R;r++) if(quotas[r]>quotas[d]) d=r;
      if(quotas[d]<=0) break;
      for(int e=1;e<epn;e++) if(remaining[e]>remaining[best]) best=e;
      int take=min(remaining[best],quotas[d]),e=h*epn+best;
      alloc[d*E+e]+=take;alloc[h*E+e]-=take;remaining[best]-=take;quotas[d]-=take;
    }
  }
}

__global__ void seed_map_kernel(const int* e,const int* ordinal,const int* hist,const int* alloc,
                                int* dst,int M,int R,int E) {
  int p=blockIdx.x*blockDim.x+threadIdx.x;
  if(p<M) {
    int expert=e[p],n=ordinal[p]+hist[(p/256)*E+expert],cumulative=0;
    for(int r=0;r<R;r++){cumulative+=alloc[r*E+expert];if(n<cumulative){dst[p]=r;break;}}
  }
}

__global__ void decode_kernel(const int* raw,int* dst,int M,int nvs) {
  int p=blockIdx.x*blockDim.x+threadIdx.x;
  if(p<M) {int x=raw[p]; dst[p]=(x<0 ? -x-1 : x)/nvs;}
}

// 每个CTA统计一个rank的一小批token，在shared memory合并直方图再写全局。
__global__ void frequency_kernel(const int* e,const int* dst,int* freq,int* counts,
                                 int T,int K,int E) {
  extern __shared__ int h[];
  for(int j=threadIdx.x;j<2*E;j+=blockDim.x) h[j]=0;
  __syncthreads();
  int t=blockIdx.x*blockDim.x+threadIdx.x,r=blockIdx.y;
  if(t<T) {
    bool visit=false;
    for(int k=0;k<K;k++) if(dst[t*K+k]==r) visit=true;
    for(int k=0;k<K;k++) {
      int expert=e[t*K+k];
      if(visit) atomicAdd(h+expert,1);
      if(dst[t*K+k]==r) atomicAdd(h+E+expert,1);
    }
  }
  __syncthreads();
  for(int j=threadIdx.x;j<E;j+=blockDim.x) {
    if(h[j]) atomicAdd(freq+r*E+j,h[j]);
    if(h[E+j]) atomicAdd(counts+r*E+j,h[E+j]);
  }
}

// 每rank一个CTA，用并行归约反复选B个专家；不把频率回传CPU。
__global__ void resident_kernel(const int* freq,const int* counts,int* support,int* status,
                                int R,int E,int B,int target) {
  using Reduce=cub::BlockReduce<int,256>;
  __shared__ Reduce::TempStorage temp;
  __shared__ int used,load,best;
  int r=blockIdx.x,tid=threadIdx.x,u=0,n=0;
  for(int e=tid;e<E;e+=256) {
    bool home=(e/(E/R)==r),active=counts[r*E+e]>0;
    support[r*E+e]=home||active;u+=active&&!home;n+=counts[r*E+e];
  }
  int sum=Reduce(temp).Sum(u);
  if(tid==0) used=sum;
  __syncthreads();
  sum=Reduce(temp).Sum(n);
  if(tid==0) load=sum;
  __syncthreads();
  if(used>B || load!=target) {if(tid==0)atomicExch(status,1);return;}
  while(used<B) {
    int value=-1;
    for(int e=tid;e<E;e+=256) if(!support[r*E+e])
      value=max(value,freq[r*E+e]*1024+1023-e);
    int winner=Reduce(temp).Reduce(value,IntMax());
    if(tid==0) best=winner<0?-1:1023-(winner%1024);
    __syncthreads();
    if(best<0) break;
    if(tid==0) {support[r*E+best]=1;used++;}
    __syncthreads();
  }
}

// 四个候选桶：a→b收益1/0、b→a收益1/0；一个token最多属于一个桶。
// Warp聚合atomicAdd，避免每个候选都串行争用队首计数。
__global__ void propose_kernel(const int* e,const int* dst,const int* support,
                               int* queues,int* sizes,int T,int S,int K,int E,int a,int b) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  int pair=-1,group=-1,best=-1;
  if(t<T) {
    int ca=0,cb=0,source=t/S;
    for(int k=0;k<K;k++) {int d=dst[t*K+k];ca+=d==a;cb+=d==b;}
    for(int k=0;k<K;k++) {
      int p=t*K+k,owner=dst[p];
      if(owner!=a && owner!=b) continue;
      int target=owner==a?b:a;
      if(!support[target*E+e[p]]) continue;
      int oldc=owner==a?ca:cb,newc=owner==a?cb:ca;
      int gain=int(owner!=source && oldc==1)-int(target!=source && newc==0);
      // -1不可能与另一侧最大+1组成严格正收益交换，可直接丢弃。
      if(gain>=0 && gain>best) {best=gain;pair=p;group=(owner==a?0:2)+(1-gain);}
    }
  }
  unsigned mask=__activemask(); int lane=threadIdx.x&31;
  for(int g=0;g<4;g++) {
    unsigned bits=__ballot_sync(mask,group==g);
    if(bits) {
      int leader=__ffs(bits)-1,base=0;
      if(lane==leader) base=atomicAdd(sizes+g,__popc(bits));
      base=__shfl_sync(mask,base,leader);
      if(group==g) queues[g*T+base+__popc(bits & ((1u<<lane)-1))]=pair;
    }
  }
}

__global__ void exchange_kernel(int* dst,const int* queues,const int* sizes,int* stats,
                                int T,int a,int b) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int n=min(min(sizes[0]+sizes[1],sizes[2]+sizes[3]),max(sizes[0],sizes[2]));
  if(i==0) {stats[0]+=n; stats[1]+=min(n,sizes[0])+min(n,sizes[2]);}
  if(i<n) {
    int p=i<sizes[0]?queues[i]:queues[T+i-sizes[0]];
    int q=i<sizes[2]?queues[2*T+i]:queues[3*T+i-sizes[2]];
    // 候选token互斥，p/q不会被本批任何其他线程修改；两卡负载不变。
    dst[p]=b; dst[q]=a;
  }
}

// 持久cooperative grid：将全部rank对扫描放入一次launch。所有CTA必须同时驻留；
// host用occupancy计算安全grid上限，不允许普通launch或超额grid去做grid.sync()。
// 每CTA把support压成E个rank位图缓存到shared memory，整轮复用。
// 双缓冲sizes避免每个rank对单独memset以及第三次grid barrier。
template<int FixedK,bool Diagonal=false>
__global__ void persistent_exchange_kernel(const int* e,int* dst,const int* support,
    int* queues,int* sizes,int* stats,int T,int S,int K,int E,int R,int sweeps,int queue_mode) {
  auto grid=cooperative_groups::this_grid();
  extern __shared__ int support_mask[];
  __shared__ int warp_prefix[4][8];
  int tid=threadIdx.x,gid=blockIdx.x*blockDim.x+tid,stride=gridDim.x*blockDim.x;
  int topk=FixedK?FixedK:K,lane=tid&31;
  for(int expert=tid;expert<E;expert+=blockDim.x) {
    int mask=0;for(int r=0;r<R;r++)mask|=(support[r*E+expert]!=0)<<r;
    support_mask[expert]=mask;
  }
  int global_id=gid,global_stride=stride;
  int stages=Diagonal?2*R-3:R*(R-1)/2,teams=R/2;
  int count=Diagonal?sweeps*stages*teams*4:8;
  for(int i=global_id;i<count;i+=global_stride)sizes[i]=0;
  grid.sync();
  int lex_a=0,lex_b=1;
  for(int sweep=0;sweep<sweeps;sweep++) for(int stage=0;stage<stages;stage++) {
    int step=sweep*stages+stage,a,b,team=0;
    if constexpr(Diagonal) {
      // 同一a+b对角线上的rank对互不重叠；每个rank仍按partner ID递增处理。
      // 因此只并发可交换的独立操作，没有改变共享同一rank的依赖顺序。
      int diagonal=stage+1,lo=max(0,diagonal-(R-1)),groups=(diagonal-1)/2-lo+1;
      team=blockIdx.x%groups;a=lo+team;b=diagonal-a;
      gid=(blockIdx.x/groups)*blockDim.x+tid;
      stride=((gridDim.x+groups-1-team)/groups)*blockDim.x;
    } else {
      if(stage==0){lex_a=0;lex_b=1;}
      a=lex_a;b=lex_b;
      if(++lex_b==R){lex_a++;lex_b=lex_a+1;}
    }
    // 对角线版本每阶段有专属小计数数组，只在入口清零一次，避免变长team复用脏桶。
    int* current=sizes+(Diagonal?(step*teams+team)*4:4*(step&1));
    int* work_queue=queues+(Diagonal?team*4*T:0);
    // 固定迭代次数，尾部无效token也参与warp ballot，避免掩码不一致。
    for(int first=0;first<T;first+=stride) {
      int t=first+gid,pair=-1,group=-1,best=-1;
      if(t<T) {
        int ca=0,cb=0,source=t/S;
        #pragma unroll
        for(int k=0;k<topk;k++){int d=dst[t*topk+k];ca+=d==a;cb+=d==b;}
        #pragma unroll
        for(int k=0;k<topk;k++) {
          int p=t*topk+k,owner=dst[p];
          if(owner!=a && owner!=b)continue;
          int target=owner==a?b:a;
          if(!(support_mask[e[p]]&(1<<target)))continue;
          int oldc=owner==a?ca:cb,newc=owner==a?cb:ca;
          int gain=int(owner!=source && oldc==1)-int(target!=source && newc==0);
          if(gain>=0 && gain>best){best=gain;pair=p;group=(owner==a?0:2)+(1-gain);}
        }
      }
      unsigned bits[4];
      for(int g=0;g<4;g++) {
        bits[g]=__ballot_sync(0xffffffff,group==g);
        if(queue_mode) {
          if(lane==0)warp_prefix[g][tid/32]=__popc(bits[g]);
        } else if(bits[g]) {
          int leader=__ffs(bits[g])-1,base=0;
          if(lane==leader)base=atomicAdd(current+g,__popc(bits[g]));
          base=__shfl_sync(0xffffffff,base,leader);
          if(group==g)work_queue[g*T+base+__popc(bits[g]&((1u<<lane)-1))]=pair;
        }
      }
      if(queue_mode) {
        // 每warp先ballot计数；四个线程作8项前缀，每CTA/桶仅一次global atomic。
        __syncthreads();
        if(tid<4) {
          int total=0;for(int w=0;w<8;w++)total+=warp_prefix[tid][w];
          int base=total?atomicAdd(current+tid,total):0;
          for(int w=0;w<8;w++){int n=warp_prefix[tid][w];warp_prefix[tid][w]=base;base+=n;}
        }
        __syncthreads();
        if(group>=0)work_queue[group*T+warp_prefix[group][tid/32]+
                            __popc(bits[group]&((1u<<lane)-1))]=pair;
        __syncthreads();
      }
    }
    // 发布候选；读取四个桶长并优先配对正收益。
    grid.sync();
    int n=min(min(current[0]+current[1],current[2]+current[3]),max(current[0],current[2]));
    if(gid==0){
      if constexpr(Diagonal){atomicAdd(stats,n);atomicAdd(stats+1,min(n,current[0])+min(n,current[2]));}
      else {stats[0]+=n;stats[1]+=min(n,current[0])+min(n,current[2]);}
    }
    for(int i=gid;i<n;i+=stride) {
      int p=i<current[0]?work_queue[i]:work_queue[T+i-current[0]];
      int q=i<current[2]?work_queue[2*T+i]:work_queue[3*T+i-current[2]];
      dst[p]=b;dst[q]=a;
    }
    if constexpr(!Diagonal){if(gid<4)sizes[4*((step+1)&1)+gid]=0;}
    // 发布地址更新并等待全部队列读完，再开始下一rank对。
    grid.sync();
  }
}

// 全局pair连续读；CTA shared histogram减少全局计数器的atomic竞争。
__global__ void shared_count_kernel(const int* e,const int* dst,int* counts,int M,int RE,int E) {
  extern __shared__ int h[];
  for(int key=threadIdx.x;key<RE;key+=blockDim.x)h[key]=0;
  __syncthreads();
  for(int item=0;item<4;item++) {
    int p=blockIdx.x*1024+item*256+threadIdx.x;
    if(p<M)atomicAdd(h+dst[p]*E+e[p],1);
  }
  __syncthreads();
  for(int key=threadIdx.x;key<RE;key+=blockDim.x)if(h[key])atomicAdd(counts+key,h[key]);
}

// 每CTA处理1024对，先在shared分配块内行，再每个活动key只申请一次全局行段。
// global cursor原来每pair一次atomic，现在每CTA/活动(rank,expert)一次。
__global__ void shared_encode_kernel(const int* e,const int* dst,const int* offsets,int* cursor,
    int* raw,const int* status,int M,int K,int RE,int E,int nvs) {
  extern __shared__ int h[];
  if(*status)return;
  for(int key=threadIdx.x;key<RE;key+=blockDim.x)h[key]=0;
  __syncthreads();
  int keys[4],ranks[4],local[4];
  #pragma unroll
  for(int item=0;item<4;item++) {
    int p=blockIdx.x*1024+item*256+threadIdx.x;
    int d=p<M?dst[p]:0,key=p<M?d*E+e[p]:0;
    keys[item]=key;ranks[item]=d;
    local[item]=p<M?atomicAdd(h+key,1):0;
  }
  __syncthreads();
  for(int key=threadIdx.x;key<RE;key+=blockDim.x) {
    int n=h[key];h[key]=n?atomicAdd(cursor+key,n):0;
  }
  __syncthreads();
  #pragma unroll
  for(int item=0;item<4;item++) {
    int p=blockIdx.x*1024+item*256+threadIdx.x;
    if(p<M) {
      int d=ranks[item],key=keys[item],v=d*nvs+offsets[key]+h[key]+local[item];
      bool duplicate=false;
      // 连续pair排布时，每token的此前K项都在同一cache line附近。
      for(int q=p-p%K;q<p;q++)duplicate|=dst[q]==d;
      raw[p]=duplicate?-v-1:v;
    }
  }
}

__global__ void count_kernel(const int* e,const int* dst,int* counts,int M,int E) {
  int p=blockIdx.x*blockDim.x+threadIdx.x;
  if(p<M) atomicAdd(counts+dst[p]*E+e[p],1);
}

// 构造原生MoonEP VM布局：home保留global expert组，所有活动remote放E+B临时组。
__global__ void layout_kernel(const int* counts,int* offsets,int* cu,int* copies,int* zeros,
                              int* remote_stats,int* status,int R,int E,int B,int padding,int nvs) {
  using Scan=cub::BlockScan<int,256>;
  using Reduce=cub::BlockReduce<int,256>;
  __shared__ Scan::TempStorage scan;
  __shared__ Reduce::TempStorage reduce;
  __shared__ int carry;
  int r=blockIdx.x,tid=threadIdx.x;
  for(int b=tid;b<B;b+=256) copies[r*B+b]=-1;
  if(tid==0) carry=0;
  __syncthreads();
  // 稳定expert ID前缀决定活动副本槽；与原生E+B组布局对应。
  for(int first=0;first<E;first+=256) {
    int e=first+tid,active=(e<E && e/(E/R)!=r && counts[r*E+e]>0),prefix;
    Scan(scan).ExclusiveSum(active,prefix);
    int base=carry;
    if(active && base+prefix<B) copies[r*B+base+prefix]=e;
    __syncthreads();
    if(tid==255) carry=base+prefix+active;
    __syncthreads();
  }
  int extras=carry;
  if(extras>B) {if(tid==0)atomicExch(status,2);return;}
  int send=0;
  for(int e=r*(E/R)+tid;e<(r+1)*(E/R);e+=256)
    for(int d=0;d<R;d++) if(d!=r)send+=counts[d*E+e]>0;
  int total_send=Reduce(reduce).Sum(send);
  if(tid==0) {remote_stats[2*r]=extras;remote_stats[2*r+1]=total_send;carry=0;}
  __syncthreads();
  for(int first=0;first<E+B;first+=256) {
    int g=first+tid,expert=-1;
    if(g<E) expert=g/(E/R)==r?g:-1;
    else if(g<E+B) expert=copies[r*B+g-E];
    int cnt=expert<0?0:counts[r*E+expert],padded=cnt?((cnt+padding-1)/padding)*padding:0,prefix;
    Scan(scan).ExclusiveSum(padded,prefix);
    int start=carry+prefix;
    if(g<E+B) {
      if(cnt) offsets[r*E+expert]=start;
      int pos=(r*(E+B)+g)*2;
      zeros[pos]=padded>cnt?start+cnt:0;zeros[pos+1]=padded-cnt;
      cu[r*(E+B)+g]=start+padded;
    }
    __syncthreads();
    if(tid==255)carry=start+padded;
    __syncthreads();
  }
  if(tid==0 && carry>nvs)atomicExch(status,3);
}

// 每个token一次，分配每对唯一专家执行行，并按目的rank做负编码去重。
__global__ void encode_kernel(const int* e,const int* dst,const int* offsets,int* cursor,
                              int* raw,const int* status,int T,int K,int E,int nvs) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;
  if(t>=T || *status) return;
  unsigned seen=0;
  for(int k=0;k<K;k++) {
    int p=t*K+k,d=dst[p],expert=e[p];
    int row=offsets[d*E+expert]+atomicAdd(cursor+d*E+expert,1);
    int v=d*nvs+row; raw[p]=(seen&(1u<<d))?-v-1:v; seen|=1u<<d;
  }
}

// 每rank只写自己的src_info，不依赖跨rank乱序写；后续公共barrier发布。
__global__ void publish_kernel(const int* raw,int* meta,int M,int N,int nvs,
                               int rank,int stride,int src_off) {
  int p=blockIdx.x*blockDim.x+threadIdx.x;
  if(p<M) {
    int v=raw[p];v=v<0?-v-1:v;
    if(v/nvs==rank) meta[rank*stride+src_off+v%nvs]=(p/N)*nvs+p%N;
  }
}

void decode(Tensor raw,Tensor dst,int nvs) {
  CHECK(raw);CHECK(dst); c10::cuda::CUDAGuard guard(raw.device());
  decode_kernel<<<(raw.numel()+255)/256,256,0,STREAM>>>(PTR(raw),PTR(dst),raw.numel(),nvs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void seed(Tensor e,Tensor ordinal,Tensor hist,Tensor demand,Tensor alloc,Tensor dst,int E) {
  CHECK(e);c10::cuda::CUDAGuard guard(e.device());
  int R=e.size(0),S=e.size(1),K=e.size(2),M=R*S*K,blocks=(M+255)/256;
  seed_hist_kernel<<<blocks,256,8*E*sizeof(int),STREAM>>>(PTR(e),PTR(ordinal),PTR(hist),M,E);
  seed_prefix_kernel<<<E,256,0,STREAM>>>(PTR(hist),PTR(demand),E,blocks);
  seed_alloc_kernel<<<1,256,0,STREAM>>>(PTR(demand),PTR(alloc),R,E,S*K);
  seed_map_kernel<<<blocks,256,0,STREAM>>>(PTR(e),PTR(ordinal),PTR(hist),PTR(alloc),PTR(dst),M,R,E);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void prepare(Tensor e,Tensor dst,Tensor freq,Tensor counts,Tensor support,Tensor status,int R,int E,int B) {
  CHECK(e);CHECK(dst);c10::cuda::CUDAGuard guard(e.device());
  int K=e.size(2),S=e.size(1),T=R*S;
  frequency_kernel<<<dim3((T+255)/256,R),256,2*E*sizeof(int),STREAM>>>(PTR(e),PTR(dst),PTR(freq),PTR(counts),T,K,E);
  resident_kernel<<<R,256,0,STREAM>>>(PTR(freq),PTR(counts),PTR(support),PTR(status),R,E,B,S*K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void exchange(Tensor e,Tensor dst,Tensor support,Tensor queues,Tensor sizes,Tensor stats,int E,int a,int b) {
  CHECK(e);CHECK(dst);c10::cuda::CUDAGuard guard(e.device());
  int S=e.size(1),K=e.size(2),T=e.size(0)*S;
  propose_kernel<<<(T+255)/256,256,0,STREAM>>>(PTR(e),PTR(dst),PTR(support),PTR(queues),PTR(sizes),T,S,K,E,a,b);
  exchange_kernel<<<(T+255)/256,256,0,STREAM>>>(PTR(dst),PTR(queues),PTR(sizes),PTR(stats),T,a,b);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void pack(Tensor e,Tensor dst,Tensor counts,Tensor offsets,Tensor cu,Tensor copies,Tensor zeros,
          Tensor remote_stats,Tensor cursor,Tensor raw,Tensor status,int E,int B,int padding,int nvs) {
  CHECK(e);CHECK(dst);c10::cuda::CUDAGuard guard(e.device());
  int R=e.size(0),S=e.size(1),K=e.size(2),T=R*S,M=T*K;
  count_kernel<<<(M+255)/256,256,0,STREAM>>>(PTR(e),PTR(dst),PTR(counts),M,E);
  layout_kernel<<<R,256,0,STREAM>>>(PTR(counts),PTR(offsets),PTR(cu),PTR(copies),PTR(zeros),PTR(remote_stats),PTR(status),R,E,B,padding,nvs);
  encode_kernel<<<(T+255)/256,256,0,STREAM>>>(PTR(e),PTR(dst),PTR(offsets),PTR(cursor),PTR(raw),PTR(status),T,K,E,nvs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
// 初始化时查询，不把occupancy/runtime查询置入Graph热路径。
int persistent_blocks(int device,int K,int E,int requested_per_sm,bool diagonal) {
  c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device));
  int sm=0,active=0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&sm,cudaDevAttrMultiProcessorCount,device));
  auto fn=diagonal?(K==8?persistent_exchange_kernel<8,true>:persistent_exchange_kernel<0,true>):
                   (K==8?persistent_exchange_kernel<8,false>:persistent_exchange_kernel<0,false>);
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,fn,256,E*sizeof(int)));
  TORCH_CHECK(active>0,"persistent kernel has no resident blocks");
  return sm*min(active,requested_per_sm);
}
void persistent_exchange(Tensor e,Tensor dst,Tensor support,Tensor queues,Tensor sizes,Tensor stats,
                         int E,int sweeps,int blocks,int queue_mode,bool diagonal) {
  CHECK(e);CHECK(dst);c10::cuda::CUDAGuard guard(e.device());
  int R=e.size(0),S=e.size(1),K=e.size(2),T=R*S;
  int *ep=PTR(e),*dp=PTR(dst),*sp=PTR(support),*qp=PTR(queues),*zp=PTR(sizes),*tp=PTR(stats);
  void* args[]={&ep,&dp,&sp,&qp,&zp,&tp,&T,&S,&K,&E,&R,&sweeps,&queue_mode};
  auto fn=diagonal?(K==8?persistent_exchange_kernel<8,true>:persistent_exchange_kernel<0,true>):
                   (K==8?persistent_exchange_kernel<8,false>:persistent_exchange_kernel<0,false>);
  C10_CUDA_CHECK(cudaLaunchCooperativeKernel((void*)fn,dim3(blocks),dim3(256),args,E*sizeof(int),STREAM));
}
void pack_shared(Tensor e,Tensor dst,Tensor counts,Tensor offsets,Tensor cu,Tensor copies,Tensor zeros,
          Tensor remote_stats,Tensor cursor,Tensor raw,Tensor status,int E,int B,int padding,int nvs) {
  CHECK(e);CHECK(dst);c10::cuda::CUDAGuard guard(e.device());
  int R=e.size(0),S=e.size(1),K=e.size(2),M=R*S*K,RE=R*E;
  shared_count_kernel<<<(M+1023)/1024,256,RE*sizeof(int),STREAM>>>(PTR(e),PTR(dst),PTR(counts),M,RE,E);
  layout_kernel<<<R,256,0,STREAM>>>(PTR(counts),PTR(offsets),PTR(cu),PTR(copies),PTR(zeros),PTR(remote_stats),PTR(status),R,E,B,padding,nvs);
  shared_encode_kernel<<<(M+1023)/1024,256,RE*sizeof(int),STREAM>>>(PTR(e),PTR(dst),PTR(offsets),PTR(cursor),PTR(raw),PTR(status),M,K,RE,E,nvs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void publish(Tensor raw,Tensor meta,int N,int nvs,int rank,int stride,int src_off) {
  CHECK(raw);CHECK(meta);c10::cuda::CUDAGuard guard(raw.device());
  publish_kernel<<<(raw.numel()+255)/256,256,0,STREAM>>>(PTR(raw),PTR(meta),raw.numel(),N,nvs,rank,stride,src_off);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("persistent_blocks",&persistent_blocks);m.def("persistent_exchange",&persistent_exchange);
  m.def("pack_shared",&pack_shared);
  m.def("seed",&seed);
  m.def("decode",&decode);m.def("prepare",&prepare);m.def("exchange",&exchange);
  m.def("pack",&pack);m.def("publish",&publish);
}
