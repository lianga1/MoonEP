// Exact configurable eight-rank sampled residency. E/K/D/G are compilation
// parameters, B is runtime. Frequency uses fixed 256-record shared tiles.
// Hash equality is NEVER sufficient: sorted expert lists are compared exactly.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <cstdint>

#ifndef MOONEP_E
#error "MOONEP_E/K/D/G must be supplied by sampled_residency.py"
#endif
namespace {
constexpr int R=8,E=MOONEP_E,K=MOONEP_K,D=MOONEP_D,G=MOONEP_G;
constexpr int THREADS=256,PAIRS=R*(R-1)/2,HOME=E/R,WORDS=(E+63)/64;
constexpr int STRIDE=K+1,CLASSES=4*STRIDE,BUCKETS=R*R*STRIDE,RETAINED=2*G;
constexpr int PAIR_SIZES=PAIRS*K,MAXP=PAIR_SIZES*3*G*G,HISTORY_WIDTH=6+2*WORDS;
constexpr int PACKET_WIDTH=2*K+2,PACKET_SIZE=D*PACKET_WIDTH+1;
constexpr int BAD_COPIES=1,BAD_SAMPLE=2,BAD_OWNERSHIP=4,UNSUPPORTED_SEED=8,BUDGET_OVERFLOW=16,INTERNAL_ERROR=32;
using Word=unsigned long long;
static_assert(E>=8&&E<=4096&&E%R==0,"E must be 8..4096 and divisible by R");
static_assert(K>=1&&K<=32&&K<=E,"K must be 1..min(32,E)");
static_assert(D>=1&&D<=4096&&G>=1&&G<=8,"invalid sample/group bounds");

__device__ __forceinline__ void pair_ids(int p,int& a,int& b){
  a=0;while(p>=R-1-a){p-=R-1-a;++a;}b=a+1+p;
}
__device__ __forceinline__ int cls(int dir,int size,int gain){return (dir*STRIDE+size)*2+gain;}
__device__ __forceinline__ int bucket(int a,int b,int size){return (a*R+b)*STRIDE+size;}

// One CTA cooperatively validates the complete native list in O(RE+RB),
// avoiding the old per-entry scans over all B previous slots. All threads call.
__device__ void native_seen(const int* etc,int* seen,int* extra,int* flag,int flag_value,int B){
  int t=threadIdx.x;
  for(int i=t;i<R*E;i+=THREADS)seen[i]=0;
  if(extra&&t<R)extra[t]=0;
  __syncthreads();
  for(int i=t;i<R*B;i+=THREADS){int r=i/B,e=etc[i];
    if(e==-1)continue;
    if(e<0||e>=E||e/HOME==r){atomicOr(flag,flag_value);continue;}
    int old=atomicAdd(seen+r*E+e,1);
    if(old)atomicOr(flag,flag_value);else if(extra)atomicAdd(extra+r,1);
  }
  __syncthreads();
}

__global__ void gather_local(const int* routes,const int* owner,const int64_t* ids,
 const int* etc,int* packet,int* status,int* seen,int S,int rank,int B){
  int tid=threadIdx.x;
  if(tid==0){packet[D*PACKET_WIDTH]=0;*status=0;}
  __syncthreads();
  native_seen(etc,seen,nullptr,packet+D*PACKET_WIDTH,1,B);
  for(int t=tid;t<D;t+=THREADS){int64_t local=ids[t]-int64_t(rank)*S;bool owns=local>=0&&local<S;int bad=0;
    for(int k=0;k<K;++k){int e=owns?routes[local*K+k]:0,o=owns?owner[local*K+k]:0;
      packet[t*PACKET_WIDTH+k]=e;packet[t*PACKET_WIDTH+K+k]=o;
      if(owns){bad|=e<0||e>=E||o<0||o>=R;for(int j=0;j<k;++j)bad|=routes[local*K+j]==e;}}
    packet[t*PACKET_WIDTH+2*K]=int(owns);packet[t*PACKET_WIDTH+2*K+1]=bad;
  }
}

__global__ void mandatory_validate(const int* packet,const int* etc,int* sampled,
 bool* support,int* extra,int* seen,int* status,int B){
  int tid=threadIdx.x;
  if(tid==0)*status=packet[D*PACKET_WIDTH]?BAD_COPIES:0;
  __syncthreads();
  // Rebuild on every select, also when a component harness supplies an external
  // reduced packet without invoking gather_local on this particular object.
  native_seen(etc,seen,extra,status,BAD_COPIES,B);
  for(int i=tid;i<R*E;i+=THREADS)support[i]=(i%E)/HOME==i/E||seen[i]>0;
  if(tid<R&&extra[tid]>B)atomicOr(status,BUDGET_OVERFLOW);
  __syncthreads();
  int bad=0;
  for(int t=tid;t<D;t+=THREADS){
    if(packet[t*PACKET_WIDTH+2*K]!=1)bad|=BAD_OWNERSHIP;
    if(packet[t*PACKET_WIDTH+2*K+1])bad|=BAD_SAMPLE;
    for(int k=0;k<K;++k){int e=packet[t*PACKET_WIDTH+k],o=packet[t*PACKET_WIDTH+K+k];
      sampled[t*2*K+k]=e;sampled[t*2*K+K+k]=o;
      if(e<0||e>=E||o<0||o>=R)bad|=BAD_SAMPLE;
      else if(!support[o*E+e])bad|=UNSUPPORTED_SEED;
    }
  }
  if(bad)atomicOr(status,bad);
}

__device__ __forceinline__ Word mix64(Word x){
  x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;x*=0x94d049bb133111ebULL;return x^(x>>31);
}
__global__ void sample_records(const int* sampled,const int64_t* ids,const bool* support,const int* extra,
 int* classes,int* groups,Word* sig0,Word* sig1,Word* masks,const int* status,int S,int B){
  int p=blockIdx.y,t=blockIdx.x*THREADS+threadIdx.x;if(t>=D)return;
  int a,b;pair_ids(p,a,b);int idx=p*D+t,src=int(ids[t]/S);
  for(int w=0;w<WORDS;++w)masks[idx*WORDS+w]=0;
  for(int k=0;k<K;++k)groups[idx*K+k]=-1;
  classes[idx]=-1;sig0[idx]=sig1[idx]=0;
  if(*status)return;
  int ga[K],gb[K],na=0,nb=0;
  for(int k=0;k<K;++k){int e=sampled[t*2*K+k],o=sampled[t*2*K+K+k];if(o==a)ga[na++]=e;if(o==b)gb[nb++]=e;}
  int va=-2,vb=-2;
  if(na){int v=int(a!=src)-int(b!=src&&nb==0),missing=0;
    for(int k=0;k<na;++k)missing+=!support[b*E+ga[k]];
    if(v>=0&&missing<=B-extra[b])va=v;}
  if(nb){int v=int(b!=src)-int(a!=src&&na==0),missing=0;
    for(int k=0;k<nb;++k)missing+=!support[a*E+gb[k]];
    if(v>=0&&missing<=B-extra[a])vb=v;}
  int dir=-1;
  if(va>vb)dir=0;else if(vb>va)dir=1;else if(va>=0){
    unsigned tie=unsigned(ids[t])+unsigned(a)*0x9e3779b9u+unsigned(b)*0x85ebca6bu;
    tie^=tie>>16;tie*=0x7feb352du;tie^=tie>>15;dir=tie&1u;}
  if(dir<0||(dir?vb:va)<0)return;
  int n=dir?nb:na,g[K];for(int k=0;k<n;++k)g[k]=dir?gb[k]:ga[k];
  for(int k=1;k<n;++k){int v=g[k],j=k-1;while(j>=0&&g[j]>v){g[j+1]=g[j];--j;}g[j+1]=v;}
  Word h=0xcbf29ce484222325ULL,other=0x9e3779b97f4a7c15ULL;
  for(int k=0;k<n;++k){int e=g[k];groups[idx*K+k]=e;
    h=(h^Word(e+1))*0x100000001b3ULL;other=mix64(other^Word(e+1));
    masks[idx*WORDS+e/64]|=Word(1)<<(e%64);}
  sig0[idx]=h;sig1[idx]=other;classes[idx]=cls(dir,n,dir?vb:va);
}

__device__ __forceinline__ bool equal_group(int a,int b,int size,const int* groups){
  for(int k=0;k<size;++k)if(groups[a*K+k]!=groups[b*K+k])return false;return true;
}
__global__ void record_frequencies(const int* classes,const int* groups,const Word* sig0,const Word* sig1,int* freq){
  __shared__ int c[THREADS];__shared__ Word h[THREADS],l[THREADS];
  int lane=threadIdx.x,t=blockIdx.x*THREADS+lane,base=blockIdx.y*D;
  int own_class=t<D?classes[base+t]:-1,n=0,first=D;
  Word own_h=t<D?sig0[base+t]:0,own_l=t<D?sig1[base+t]:0;
  int size=own_class<0?0:(own_class/2)%STRIDE;
  for(int start=0;start<D;start+=THREADS){int j=start+lane;
    c[lane]=j<D?classes[base+j]:-1;h[lane]=j<D?sig0[base+j]:0;l[lane]=j<D?sig1[base+j]:0;
    __syncthreads();
    if(own_class>=0)for(int k=0;k<min(THREADS,D-start);++k){
      if(own_class==c[k]&&own_h==h[k]&&own_l==l[k]&&equal_group(base+t,base+start+k,size,groups)){
        ++n;first=min(first,start+k);}}
    __syncthreads();
  }
  if(t<D)freq[base+t]=first==t?n:0;
}

// Only exact expert-list lexicographic order decides a frequency tie. These
// hashes never stand in for expert IDs when retaining groups.
struct GroupChoice{int id,first;};
struct BestGroup{
  const int* frequencies;const int* groups;
  __device__ __forceinline__ bool better(int a,int b)const{
    if(a<0)return false;if(b<0)return true;
    if(frequencies[a]!=frequencies[b])return frequencies[a]>frequencies[b];
    for(int k=0;k<K;++k){int x=groups[a*K+k],y=groups[b*K+k];if(x!=y)return x<y;}
    return a<b;
  }
  __device__ __forceinline__ GroupChoice operator()(GroupChoice a,GroupChoice b)const{
    return {better(b.id,a.id)?b.id:a.id,min(a.first,b.first)};
  }
};
__global__ void class_topg(const int* classes,const int* freq,const int* groups,int* tops,int* first){
  using Reduce=cub::BlockReduce<GroupChoice,THREADS>;
  __shared__ typename Reduce::TempStorage tmp;__shared__ int selected[G];
  int cc=blockIdx.x,c=cc%CLASSES,base=(cc/CLASSES)*D,tid=threadIdx.x;
  if(tid<G){selected[tid]=-1;tops[cc*G+tid]=-1;}__syncthreads();
  BestGroup better{freq,groups};
  for(int g=0;g<G;++g){GroupChoice local{-1,D};
    for(int t=tid;t<D;t+=THREADS){int rec=base+t;if(classes[rec]!=c)continue;
      local.first=min(local.first,t);if(freq[rec]<=0)continue;
      bool used=false;for(int j=0;j<g;++j)used|=selected[j]==rec;
      if(!used&&better.better(rec,local.id))local.id=rec;
    }
    GroupChoice chosen=Reduce(tmp).Reduce(local,better);
    if(tid==0){selected[g]=chosen.id;tops[cc*G+g]=chosen.id;if(g==0)first[cc]=chosen.first;}
    __syncthreads();if(selected[g]<0)break;
  }
}

// Reference dict insertion order is first sample occurrence of the gain class,
// followed by (-frequency, exact expert tuple). Package indices are ordered
// pair, size, forward group, backward group, skipping gain0+gain0.
__global__ void assemble_groups(const int* tops,const int* first,const int* classes,int* records,
 int* sizes,int* offsets){
  int task=blockIdx.x*THREADS+threadIdx.x;if(task>=PAIR_SIZES)return;
  int p=task/K,size=task%K+1,a,b;pair_ids(p,a,b);
  for(int dir=0;dir<2;++dir){int key=bucket(dir?b:a,dir?a:b,size),base=p*CLASSES,n=0;
    for(int j=0;j<RETAINED;++j)records[key*RETAINED+j]=-1;
    int c0=base+cls(dir,size,0),c1=base+cls(dir,size,1),initial=first[c0]<=first[c1]?0:1;
    for(int pass=0;pass<2;++pass){int c=base+cls(dir,size,pass?1-initial:initial);
      for(int j=0;j<G;++j){int rec=tops[c*G+j];if(rec>=0)records[key*RETAINED+n++]=rec;}}
    sizes[key]=n;
  }
  int ka=bucket(a,b,size),kb=bucket(b,a,size),n=0;
  for(int ia=0;ia<sizes[ka];++ia)for(int ib=0;ib<sizes[kb];++ib){
    int ra=records[ka*RETAINED+ia],rb=records[kb*RETAINED+ib];
    n+=int((classes[ra]&1)+(classes[rb]&1)>0);}
  offsets[task]=n;
}
__global__ void prefix_packages(int* records,int* sizes,int* offsets,int* package_count,int* status){
  // Diagonal and size-zero buckets are never assembled. Reset them every run.
  for(int key=threadIdx.x;key<BUCKETS;key+=THREADS){int a=key/(R*STRIDE),b=(key/STRIDE)%R,size=key%STRIDE;
    if(a==b||size==0){sizes[key]=0;for(int j=0;j<RETAINED;++j)records[key*RETAINED+j]=-1;}}
  if(threadIdx.x==0){int n=0;for(int t=0;t<PAIR_SIZES;++t){int count=offsets[t];offsets[t]=n;n+=count;}
    offsets[PAIR_SIZES]=n;*package_count=n;if(n>MAXP)atomicOr(status,INTERNAL_ERROR);}
}
__global__ void fill_packages(const int* classes,const int* records,const int* sizes,const int* offsets,int* packages){
  int task=blockIdx.x*THREADS+threadIdx.x;if(task>=PAIR_SIZES)return;
  int p=task/K,size=task%K+1,a,b;pair_ids(p,a,b);int out=offsets[task];
  int ka=bucket(a,b,size),kb=bucket(b,a,size);
  for(int ia=0;ia<sizes[ka];++ia)for(int ib=0;ib<sizes[kb];++ib){
    int ra=records[ka*RETAINED+ia],rb=records[kb*RETAINED+ib];
    if((classes[ra]&1)+(classes[rb]&1)==0)continue;
    if(out<MAXP){int* q=packages+out*5;q[0]=a;q[1]=b;q[2]=size;q[3]=ra;q[4]=rb;}++out;}
}

__device__ __forceinline__ int match_gain(int x0,int x1,int y0,int y1){
  int n=min(min(x0+x1,y0+y1),max(x1,y1));return min(n,x1)+min(n,y1);
}
__device__ __forceinline__ bool enabled(const Word* mask,const Word* resident){
  for(int w=0;w<WORDS;++w)if(mask[w]&~resident[w])return false;return true;
}
struct Score{int gain,cost,idx;};
struct BestScore{
  __device__ __forceinline__ Score operator()(Score a,Score b)const{
    if(b.gain<=0)return a;if(a.gain<=0)return b;
    int64_t lhs=int64_t(b.gain)*a.cost,rhs=int64_t(a.gain)*b.cost;
    if(lhs!=rhs)return lhs>rhs?b:a;
    if(a.gain!=b.gain)return b.gain>a.gain?b:a;
    if(a.cost!=b.cost)return b.cost<a.cost?b:a;
    return b.idx<a.idx?b:a;
  }
};
__global__ void global_greedy(const Word* masks,const int* freq,const int* classes,const int* records,
 const int* sizes,const int* packages,const int* package_count,bool* support,int* extra,
 int* output_etc,int64_t* history,int* history_count,const int* status,int B){
  using Reduce=cub::BlockReduce<Score,THREADS>;
  __shared__ typename Reduce::TempStorage tmp;
  __shared__ Word resident[R][WORDS];__shared__ int current_extra[R],e0[BUCKETS],e1[BUCKETS];
  __shared__ Score winner;__shared__ int accepted;
  int t=threadIdx.x;
  for(int i=t;i<R*WORDS;i+=THREADS){int r=i/WORDS,w=i%WORDS;Word m=0;
    for(int bit=0;bit<64;++bit){int e=w*64+bit;if(e<E&&support[r*E+e])m|=Word(1)<<bit;}resident[r][w]=m;}
  if(t<R)current_extra[t]=extra[t];if(t==0)accepted=0;__syncthreads();
  for(int iteration=0;iteration<R*B&&*status==0;++iteration){
    for(int key=t;key<BUCKETS;key+=THREADS){int x0=0,x1=0,b=(key/STRIDE)%R;
      for(int j=0;j<sizes[key];++j){int rec=records[key*RETAINED+j];if(enabled(masks+rec*WORDS,resident[b])){
        if(classes[rec]&1)x1+=freq[rec];else x0+=freq[rec];}}
      e0[key]=x0;e1[key]=x1;
    }
    __syncthreads();Score best{0,0,MAXP};
    for(int p=t;p<*package_count;p+=THREADS){const int* q=packages+p*5;int a=q[0],b=q[1],size=q[2],ra=q[3],rb=q[4];
      int ca=0,cb=0;
      for(int w=0;w<WORDS;++w){ca+=__popcll(masks[rb*WORDS+w]&~resident[a][w]);
        cb+=__popcll(masks[ra*WORDS+w]&~resident[b][w]);}
      int cost=ca+cb;if(cost==0||current_extra[a]+ca>B||current_extra[b]+cb>B)continue;
      int ka=bucket(a,b,size),kb=bucket(b,a,size),x0=0,x1=0,y0=0,y1=0;
      for(int j=0;j<sizes[ka];++j){int rec=records[ka*RETAINED+j];bool ok=true;
        for(int w=0;w<WORDS;++w)if(masks[rec*WORDS+w]&~(resident[b][w]|masks[ra*WORDS+w])){ok=false;break;}
        if(ok){if(classes[rec]&1)x1+=freq[rec];else x0+=freq[rec];}}
      for(int j=0;j<sizes[kb];++j){int rec=records[kb*RETAINED+j];bool ok=true;
        for(int w=0;w<WORDS;++w)if(masks[rec*WORDS+w]&~(resident[a][w]|masks[rb*WORDS+w])){ok=false;break;}
        if(ok){if(classes[rec]&1)y1+=freq[rec];else y0+=freq[rec];}}
      int gain=match_gain(x0,x1,y0,y1)-match_gain(e0[ka],e1[ka],e0[kb],e1[kb]);
      best=BestScore()(best,Score{gain,cost,p});
    }
    Score found=Reduce(tmp).Reduce(best,BestScore());if(t==0)winner=found;__syncthreads();
    if(winner.gain<=0)break;
    if(t==0){const int* q=packages+winner.idx*5;int a=q[0],b=q[1];int64_t* h=history+accepted*HISTORY_WIDTH;
      h[0]=winner.idx;h[1]=winner.gain;h[2]=a;h[3]=b;h[4]=q[2];h[5]=winner.cost;
      for(int w=0;w<WORDS;++w){Word aa=masks[q[4]*WORDS+w]&~resident[a][w],ab=masks[q[3]*WORDS+w]&~resident[b][w];
        h[6+w]=int64_t(aa);h[6+WORDS+w]=int64_t(ab);current_extra[a]+=__popcll(aa);current_extra[b]+=__popcll(ab);
        resident[a][w]|=aa;resident[b][w]|=ab;}++accepted;
    }
    __syncthreads();
  }
  for(int i=t;i<R*E;i+=THREADS){int r=i/E,e=i%E;support[i]=(resident[r][e/64]>>(e%64))&1;}
  if(t<R){extra[t]=current_extra[t];int n=0;
    for(int e=0;e<E;++e)if(e/HOME!=t&&((resident[t][e/64]>>(e%64))&1)){if(n<B)output_etc[t*B+n]=e;++n;}
    while(n<B)output_etc[t*B+n++]=-1;}
  if(t==0)*history_count=accepted;
}

void check(const at::Tensor& tensor,const at::Tensor& reference,at::ScalarType dtype,int64_t count,const char* name){
  TORCH_CHECK(tensor.is_cuda()&&tensor.device()==reference.device()&&tensor.scalar_type()==dtype&&tensor.is_contiguous()&&tensor.numel()==count,"invalid ",name);
}
} // namespace

void gather(const at::Tensor& routes,const at::Tensor& owner,const at::Tensor& ids,const at::Tensor& etc,
 at::Tensor& packet,at::Tensor& status,at::Tensor& seen,int64_t S,int64_t rank,int64_t B){
  TORCH_CHECK(routes.is_cuda(),"routes must be CUDA");
  TORCH_CHECK(B>=1&&B<=E-HOME&&S>=1&&int64_t(R)*S>=D&&rank>=0&&rank<R,"invalid B/S/rank");
  check(routes,routes,at::kInt,S*K,"local routes");check(owner,routes,at::kInt,S*K,"local owner");
  check(ids,routes,at::kLong,D,"sample IDs");check(etc,routes,at::kInt,R*B,"native copies");
  check(packet,routes,at::kInt,PACKET_SIZE,"packet");check(status,routes,at::kInt,1,"status");check(seen,routes,at::kInt,R*E,"copy seen");
  c10::cuda::CUDAGuard guard(routes.device());auto stream=c10::cuda::getCurrentCUDAStream();
  gather_local<<<1,THREADS,0,stream>>>(routes.data_ptr<int>(),owner.data_ptr<int>(),ids.data_ptr<int64_t>(),etc.data_ptr<int>(),packet.data_ptr<int>(),status.data_ptr<int>(),seen.data_ptr<int>(),S,rank,B);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void select_residency(const at::Tensor& packet,const at::Tensor& ids,const at::Tensor& etc,
 at::Tensor& samples,at::Tensor& support,at::Tensor& extra,at::Tensor& seen,at::Tensor& classes,at::Tensor& groups,
 at::Tensor& sig0,at::Tensor& sig1,at::Tensor& masks,at::Tensor& freq,at::Tensor& tops,at::Tensor& first,
 at::Tensor& records,at::Tensor& sizes,at::Tensor& offsets,at::Tensor& packages,at::Tensor& package_count,
 at::Tensor& output_etc,at::Tensor& history,at::Tensor& history_count,at::Tensor& status,int64_t S,int64_t B){
  TORCH_CHECK(packet.is_cuda()&&B>=1&&B<=E-HOME&&S>=1&&int64_t(R)*S>=D,"invalid packet/B/S");
  check(packet,packet,at::kInt,PACKET_SIZE,"packet");check(ids,packet,at::kLong,D,"sample IDs");check(etc,packet,at::kInt,R*B,"native copies");
  check(samples,packet,at::kInt,D*2*K,"samples");check(support,packet,at::kBool,R*E,"support");
  check(extra,packet,at::kInt,R,"extra");check(seen,packet,at::kInt,R*E,"copy seen");
  check(classes,packet,at::kInt,PAIRS*D,"classes");check(groups,packet,at::kInt,PAIRS*D*K,"expert groups");
  check(sig0,packet,at::kLong,PAIRS*D,"signature hash0");check(sig1,packet,at::kLong,PAIRS*D,"signature hash1");
  check(masks,packet,at::kLong,PAIRS*D*WORDS,"masks");check(freq,packet,at::kInt,PAIRS*D,"frequencies");
  check(tops,packet,at::kInt,PAIRS*CLASSES*G,"tops");check(first,packet,at::kInt,PAIRS*CLASSES,"first");
  check(records,packet,at::kInt,BUCKETS*RETAINED,"records");check(sizes,packet,at::kInt,BUCKETS,"sizes");
  check(offsets,packet,at::kInt,PAIR_SIZES+1,"package offsets");check(packages,packet,at::kInt,MAXP*5,"packages");
  check(package_count,packet,at::kInt,1,"package count");check(output_etc,packet,at::kInt,R*B,"output copies");
  check(history,packet,at::kLong,R*B*HISTORY_WIDTH,"history");check(history_count,packet,at::kInt,1,"history count");check(status,packet,at::kInt,1,"status");
  c10::cuda::CUDAGuard guard(packet.device());auto stream=c10::cuda::getCurrentCUDAStream();
  auto w=[](at::Tensor& x){return reinterpret_cast<Word*>(x.data_ptr<int64_t>());};
  mandatory_validate<<<1,THREADS,0,stream>>>(packet.data_ptr<int>(),etc.data_ptr<int>(),samples.data_ptr<int>(),support.data_ptr<bool>(),extra.data_ptr<int>(),seen.data_ptr<int>(),status.data_ptr<int>(),B);
  dim3 grid((D+THREADS-1)/THREADS,PAIRS);
  sample_records<<<grid,THREADS,0,stream>>>(samples.data_ptr<int>(),ids.data_ptr<int64_t>(),support.data_ptr<bool>(),extra.data_ptr<int>(),classes.data_ptr<int>(),groups.data_ptr<int>(),w(sig0),w(sig1),w(masks),status.data_ptr<int>(),S,B);
  record_frequencies<<<grid,THREADS,0,stream>>>(classes.data_ptr<int>(),groups.data_ptr<int>(),w(sig0),w(sig1),freq.data_ptr<int>());
  class_topg<<<PAIRS*CLASSES,THREADS,0,stream>>>(classes.data_ptr<int>(),freq.data_ptr<int>(),groups.data_ptr<int>(),tops.data_ptr<int>(),first.data_ptr<int>());
  assemble_groups<<<(PAIR_SIZES+THREADS-1)/THREADS,THREADS,0,stream>>>(tops.data_ptr<int>(),first.data_ptr<int>(),classes.data_ptr<int>(),records.data_ptr<int>(),sizes.data_ptr<int>(),offsets.data_ptr<int>());
  prefix_packages<<<1,THREADS,0,stream>>>(records.data_ptr<int>(),sizes.data_ptr<int>(),offsets.data_ptr<int>(),package_count.data_ptr<int>(),status.data_ptr<int>());
  fill_packages<<<(PAIR_SIZES+THREADS-1)/THREADS,THREADS,0,stream>>>(classes.data_ptr<int>(),records.data_ptr<int>(),sizes.data_ptr<int>(),offsets.data_ptr<int>(),packages.data_ptr<int>());
  global_greedy<<<1,THREADS,0,stream>>>(w(masks),freq.data_ptr<int>(),classes.data_ptr<int>(),records.data_ptr<int>(),sizes.data_ptr<int>(),packages.data_ptr<int>(),package_count.data_ptr<int>(),support.data_ptr<bool>(),extra.data_ptr<int>(),output_etc.data_ptr<int>(),history.data_ptr<int64_t>(),history_count.data_ptr<int>(),status.data_ptr<int>(),B);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
// Diagnostic split used to force hash collisions in correctness tests. The
// normal runtime always calls select_residency and recomputes both hashes.
void count_frequencies(const at::Tensor& classes,const at::Tensor& groups,
 const at::Tensor& sig0,const at::Tensor& sig1,at::Tensor& freq){
  TORCH_CHECK(classes.is_cuda(),"classes must be CUDA");
  check(classes,classes,at::kInt,PAIRS*D,"classes");check(groups,classes,at::kInt,PAIRS*D*K,"expert groups");
  check(sig0,classes,at::kLong,PAIRS*D,"signature hash0");check(sig1,classes,at::kLong,PAIRS*D,"signature hash1");
  check(freq,classes,at::kInt,PAIRS*D,"frequencies");
  c10::cuda::CUDAGuard guard(classes.device());auto stream=c10::cuda::getCurrentCUDAStream();
  record_frequencies<<<dim3((D+THREADS-1)/THREADS,PAIRS),THREADS,0,stream>>>(classes.data_ptr<int>(),groups.data_ptr<int>(),
    reinterpret_cast<const Word*>(sig0.data_ptr<int64_t>()),reinterpret_cast<const Word*>(sig1.data_ptr<int64_t>()),freq.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("gather",&gather);m.def("select",&select_residency);m.def("count_frequencies",&count_frequencies);}
