// Exact configurable eight-rank sampled residency. E/K/D/G are compilation
// parameters, B is runtime. Frequency uses fixed 256-record shared tiles.
// Hash equality is NEVER sufficient: sorted expert lists are compared exactly.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <cub/device/device_merge_sort.cuh>
#include <cstdint>
#include <algorithm>

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

// Reuse the work for one retained direction group across all reciprocal
// packages containing it. The generic kernel above remains the fallback.
// sizeof includes the actual CUB scratch storage and structure alignment.
using GreedyReduce=cub::BlockReduce<Score,THREADS>;
constexpr int GREEDY_ROWS=BUCKETS*RETAINED;
constexpr int PACKAGES_PER_THREAD=(MAXP+THREADS-1)/THREADS;
struct GreedyTableStorage {
  typename GreedyReduce::TempStorage reduction;
  Word resident[R][WORDS];
  int current_extra[R],e0[BUCKETS],e1[BUCKETS];
  int missing[GREEDY_ROWS],after0[GREEDY_ROWS],after1[GREEDY_ROWS];
  Score winner;
  int accepted;
  unsigned changed_targets;
};
constexpr bool USE_GREEDY_TABLE=sizeof(GreedyTableStorage)<=48*1024;

// A bucket retains at most 2*G groups, all with the same expert count. For
// small K/G, assign one bit to each (retained group, expert slot), including
// duplicate expert IDs across groups. Exact coverage of those slots replaces
// repeated E-wide mask scans; it is not a probabilistic hash or approximation.
struct CompactGreedyTableStorage {
  GreedyTableStorage core;
  unsigned coverage[GREEDY_ROWS],present[BUCKETS];
};
constexpr bool USE_COMPACT_GREEDY_TABLE=RETAINED*K<=32&&
  sizeof(CompactGreedyTableStorage)<=48*1024;


// A template is intentional: configurations exceeding the static shared-memory
// limit must not instantiate this kernel (or its large local package arrays).
template<int Specialization=0>
__global__ void global_greedy_table(const Word* masks,const int* freq,const int* classes,const int* records,
 const int* sizes,const int* packages,const int* package_count,bool* support,int* extra,
 int* output_etc,int64_t* history,int* history_count,const int* status,int B){
  __shared__ GreedyTableStorage s;
  const int t=threadIdx.x,n_packages=*package_count;
  for(int i=t;i<R*WORDS;i+=THREADS){int r=i/WORDS,w=i%WORDS;Word m=0;
    for(int bit=0;bit<64;++bit){int e=w*64+bit;if(e<E&&support[r*E+e])m|=Word(1)<<bit;}s.resident[r][w]=m;}
  if(t<R)s.current_extra[t]=extra[t];
  if(t==0){s.accepted=0;s.changed_targets=(1u<<R)-1;}

  // Only two constant-index arrays are kept per thread. For E352/K8/G2 this
  // is 2*11 int values; unrolling allows register scalarization, subject to
  // compiler register pressure. Package order and package contents are intact.
  int forward_slot[PACKAGES_PER_THREAD],backward_slot[PACKAGES_PER_THREAD];
  #pragma unroll
  for(int i=0;i<PACKAGES_PER_THREAD;++i){
    forward_slot[i]=backward_slot[i]=-1;
    int p=t+i*THREADS;
    // The generic path skips all package reads when status is nonzero. Keep
    // that property even though metadata is now loaded before the iterations.
    if(p<n_packages&&*status==0){
      const int* q=packages+p*5;
      int ka=bucket(q[0],q[1],q[2]),kb=bucket(q[1],q[0],q[2]);
      for(int j=0;j<sizes[ka];++j)if(records[ka*RETAINED+j]==q[3])forward_slot[i]=ka*RETAINED+j;
      for(int j=0;j<sizes[kb];++j)if(records[kb*RETAINED+j]==q[4])backward_slot[i]=kb*RETAINED+j;
    }
  }
  __syncthreads();

  for(int iteration=0;iteration<R*B&&*status==0;++iteration){
    // All cached values for old->new depend only on resident[new]. Every
    // target is initialized once; after accepting a package, only targets
    // receiving at least one genuinely new expert need their tables rebuilt.
    // Keep full package rescoring below: both endpoints' global budgets still
    // participate in feasibility, and the original deterministic tie remains.
    const unsigned changed=s.changed_targets;
    // The baseline counts include every retained group that is already
    // executable, including groups with zero standalone communication gain.
    for(int key=t;key<BUCKETS;key+=THREADS){int x0=0,x1=0,b=(key/STRIDE)%R;
      if(!(changed&(1u<<b)))continue;
      for(int j=0;j<sizes[key];++j){int rec=records[key*RETAINED+j];if(enabled(masks+rec*WORDS,s.resident[b])){
        if(classes[rec]&1)x1+=freq[rec];else x0+=freq[rec];}}
      s.e0[key]=x0;s.e1[key]=x1;
    }
    for(int row=t;row<GREEDY_ROWS;row+=THREADS){
      int key=row/RETAINED,j=row%RETAINED,target=(key/STRIDE)%R;
      if(!(changed&(1u<<target)))continue;
      if(j>=sizes[key]){s.missing[row]=s.after0[row]=s.after1[row]=0;continue;}
      int rec=records[row],missing=0,x0=0,x1=0;
      for(int w=0;w<WORDS;++w)missing+=__popcll(masks[rec*WORDS+w]&~s.resident[target][w]);
      // A zero-cost direction still participates: its reverse direction can
      // produce positive gain. Count the complete bucket after adding this
      // group's experts, not only the newly enabled records.
      for(int k=0;k<sizes[key];++k){int other=records[key*RETAINED+k];bool ok=true;
        for(int w=0;w<WORDS;++w)if(masks[other*WORDS+w]&~(s.resident[target][w]|masks[rec*WORDS+w])){ok=false;break;}
        if(ok){if(classes[other]&1)x1+=freq[other];else x0+=freq[other];}
      }
      s.missing[row]=missing;s.after0[row]=x0;s.after1[row]=x1;
    }
    __syncthreads();
    Score best{0,0,MAXP};
    #pragma unroll
    for(int i=0;i<PACKAGES_PER_THREAD;++i){int p=t+i*THREADS;
      if(p<n_packages){
        int sa=forward_slot[i],sb=backward_slot[i],ka=sa/RETAINED,kb=sb/RETAINED;
        int a=ka/(R*STRIDE),b=(ka/STRIDE)%R;
        // a->b adds the forward group to b, while b->a adds the reverse
        // group to a. Budgets remain global across all 28 card pairs.
        int ca=s.missing[sb],cb=s.missing[sa],cost=ca+cb;
        if(cost==0||s.current_extra[a]+ca>B||s.current_extra[b]+cb>B)continue;
        int gain=match_gain(s.after0[sa],s.after1[sa],s.after0[sb],s.after1[sb])
          -match_gain(s.e0[ka],s.e1[ka],s.e0[kb],s.e1[kb]);
        best=BestScore()(best,Score{gain,cost,p});
      }
    }
    Score found=GreedyReduce(s.reduction).Reduce(best,BestScore());
    if(t==0)s.winner=found;__syncthreads();
    if(s.winner.gain<=0)break;
    if(t==0){const int* q=packages+s.winner.idx*5;int a=q[0],b=q[1];int64_t* h=history+s.accepted*HISTORY_WIDTH;
      h[0]=s.winner.idx;h[1]=s.winner.gain;h[2]=a;h[3]=b;h[4]=q[2];h[5]=s.winner.cost;
      s.changed_targets=0;
      for(int w=0;w<WORDS;++w){Word aa=masks[q[4]*WORDS+w]&~s.resident[a][w],ab=masks[q[3]*WORDS+w]&~s.resident[b][w];
        h[6+w]=int64_t(aa);h[6+WORDS+w]=int64_t(ab);s.current_extra[a]+=__popcll(aa);s.current_extra[b]+=__popcll(ab);
        if(aa)s.changed_targets|=1u<<a;if(ab)s.changed_targets|=1u<<b;
        s.resident[a][w]|=aa;s.resident[b][w]|=ab;}++s.accepted;
    }
    __syncthreads();
  }
  for(int i=t;i<R*E;i+=THREADS){int r=i/E,e=i%E;support[i]=(s.resident[r][e/64]>>(e%64))&1;}
  if(t<R){extra[t]=s.current_extra[t];int n=0;
    for(int e=0;e<E;++e)if(e/HOME!=t&&((s.resident[t][e/64]>>(e%64))&1)){if(n<B)output_etc[t*B+n]=e;++n;}
    while(n<B)output_etc[t*B+n++]=-1;}
  if(t==0)*history_count=s.accepted;
}

__global__ void global_greedy_capped(const Word* masks,const int* freq,const int* classes,const int* records,
 const int* sizes,const int* packages,const int* package_count,const int* selected_ids,bool* support,int* extra,
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
    for(int p=t;p<*package_count;p+=THREADS){int original=selected_ids[p];const int* q=packages+original*5;int a=q[0],b=q[1],size=q[2],ra=q[3],rb=q[4];
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
      best=BestScore()(best,Score{gain,cost,original});
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


template<int Specialization=0>
__global__ void global_greedy_table_active(const Word* masks,const int* freq,const int* classes,const int* records,
 const int* sizes,const int* packages,const int* package_count,const int* selected_ids,bool* support,int* extra,
 int* output_etc,int64_t* history,int* history_count,const int* status,int B,const int* active_keys,const int* active_rows,const int* active_counts){
  __shared__ GreedyTableStorage s;
  const int t=threadIdx.x,n_packages=*package_count;
  const int n_active_keys=active_counts[0],n_active_rows=active_counts[1];
  for(int i=t;i<R*WORDS;i+=THREADS){int r=i/WORDS,w=i%WORDS;Word m=0;
    for(int bit=0;bit<64;++bit){int e=w*64+bit;if(e<E&&support[r*E+e])m|=Word(1)<<bit;}s.resident[r][w]=m;}
  if(t<R)s.current_extra[t]=extra[t];
  if(t==0){s.accepted=0;s.changed_targets=(1u<<R)-1;}

  // Only two constant-index arrays are kept per thread. For E352/K8/G2 this
  // is 2*11 int values; unrolling allows register scalarization, subject to
  // compiler register pressure. Package order and package contents are intact.
  int forward_slot[PACKAGES_PER_THREAD],backward_slot[PACKAGES_PER_THREAD];
  #pragma unroll
  for(int i=0;i<PACKAGES_PER_THREAD;++i){
    forward_slot[i]=backward_slot[i]=-1;
    int p=t+i*THREADS;
    // The generic path skips all package reads when status is nonzero. Keep
    // that property even though metadata is now loaded before the iterations.
    if(p<n_packages&&*status==0){
      const int* q=packages+selected_ids[p]*5;
      int ka=bucket(q[0],q[1],q[2]),kb=bucket(q[1],q[0],q[2]);
      for(int j=0;j<sizes[ka];++j)if(records[ka*RETAINED+j]==q[3])forward_slot[i]=ka*RETAINED+j;
      for(int j=0;j<sizes[kb];++j)if(records[kb*RETAINED+j]==q[4])backward_slot[i]=kb*RETAINED+j;
    }
  }
  __syncthreads();

  for(int iteration=0;iteration<R*B&&*status==0;++iteration){
    // All cached values for old->new depend only on resident[new]. Every
    // target is initialized once; after accepting a package, only targets
    // receiving at least one genuinely new expert need their tables rebuilt.
    // Keep full package rescoring below: both endpoints' global budgets still
    // participate in feasibility, and the original deterministic tie remains.
    const unsigned changed=s.changed_targets;
    // The baseline counts include every retained group that is already
    // executable, including groups with zero standalone communication gain.
    for(int entry=t;entry<n_active_keys;entry+=THREADS){int key=active_keys[entry];
      int x0=0,x1=0,b=(key/STRIDE)%R;
      if(!(changed&(1u<<b)))continue;
      for(int j=0;j<sizes[key];++j){int rec=records[key*RETAINED+j];if(enabled(masks+rec*WORDS,s.resident[b])){
        if(classes[rec]&1)x1+=freq[rec];else x0+=freq[rec];}}
      s.e0[key]=x0;s.e1[key]=x1;
    }
    for(int entry=t;entry<n_active_rows;entry+=THREADS){
      int row=active_rows[entry],key=row/RETAINED,j=row%RETAINED,target=(key/STRIDE)%R;
      if(!(changed&(1u<<target)))continue;
      if(j>=sizes[key]){s.missing[row]=s.after0[row]=s.after1[row]=0;continue;}
      int rec=records[row],missing=0,x0=0,x1=0;
      for(int w=0;w<WORDS;++w)missing+=__popcll(masks[rec*WORDS+w]&~s.resident[target][w]);
      // A zero-cost direction still participates: its reverse direction can
      // produce positive gain. Count the complete bucket after adding this
      // group's experts, not only the newly enabled records.
      for(int k=0;k<sizes[key];++k){int other=records[key*RETAINED+k];bool ok=true;
        for(int w=0;w<WORDS;++w)if(masks[other*WORDS+w]&~(s.resident[target][w]|masks[rec*WORDS+w])){ok=false;break;}
        if(ok){if(classes[other]&1)x1+=freq[other];else x0+=freq[other];}
      }
      s.missing[row]=missing;s.after0[row]=x0;s.after1[row]=x1;
    }
    __syncthreads();
    Score best{0,0,MAXP};
    #pragma unroll
    for(int i=0;i<PACKAGES_PER_THREAD;++i){int p=t+i*THREADS;
      if(p<n_packages){
        int sa=forward_slot[i],sb=backward_slot[i],ka=sa/RETAINED,kb=sb/RETAINED;
        int a=ka/(R*STRIDE),b=(ka/STRIDE)%R;
        // a->b adds the forward group to b, while b->a adds the reverse
        // group to a. Budgets remain global across all 28 card pairs.
        int ca=s.missing[sb],cb=s.missing[sa],cost=ca+cb;
        if(cost==0||s.current_extra[a]+ca>B||s.current_extra[b]+cb>B)continue;
        int gain=match_gain(s.after0[sa],s.after1[sa],s.after0[sb],s.after1[sb])
          -match_gain(s.e0[ka],s.e1[ka],s.e0[kb],s.e1[kb]);
        best=BestScore()(best,Score{gain,cost,selected_ids[p]});
      }
    }
    Score found=GreedyReduce(s.reduction).Reduce(best,BestScore());
    if(t==0)s.winner=found;__syncthreads();
    if(s.winner.gain<=0)break;
    if(t==0){const int* q=packages+s.winner.idx*5;int a=q[0],b=q[1];int64_t* h=history+s.accepted*HISTORY_WIDTH;
      h[0]=s.winner.idx;h[1]=s.winner.gain;h[2]=a;h[3]=b;h[4]=q[2];h[5]=s.winner.cost;
      s.changed_targets=0;
      for(int w=0;w<WORDS;++w){Word aa=masks[q[4]*WORDS+w]&~s.resident[a][w],ab=masks[q[3]*WORDS+w]&~s.resident[b][w];
        h[6+w]=int64_t(aa);h[6+WORDS+w]=int64_t(ab);s.current_extra[a]+=__popcll(aa);s.current_extra[b]+=__popcll(ab);
        if(aa)s.changed_targets|=1u<<a;if(ab)s.changed_targets|=1u<<b;
        s.resident[a][w]|=aa;s.resident[b][w]|=ab;}++s.accepted;
    }
    __syncthreads();
  }
  for(int i=t;i<R*E;i+=THREADS){int r=i/E,e=i%E;support[i]=(s.resident[r][e/64]>>(e%64))&1;}
  if(t<R){extra[t]=s.current_extra[t];int n=0;
    for(int e=0;e<E;++e)if(e/HOME!=t&&((s.resident[t][e/64]>>(e%64))&1)){if(n<B)output_etc[t*B+n]=e;++n;}
    while(n<B)output_etc[t*B+n++]=-1;}
  if(t==0)*history_count=s.accepted;
}

// Exact compact-bit variant for the bounded planner. Package scores, global
// budget checks, BestScore tie-breaking, accepted history and expert-ID output
// ordering are unchanged. Configurations outside the bit/shared-memory bounds
// retain global_greedy_table_active (or the existing generic fallback).
template<int PackagesPerThread=PACKAGES_PER_THREAD>
__global__ void global_greedy_compact_active(const Word* masks,const int* freq,const int* classes,const int* groups,const int* records,
 const int* sizes,const int* packages,const int* package_count,const int* selected_ids,bool* support,int* extra,
 int* output_etc,int64_t* history,int* history_count,const int* status,int B,const int* active_keys,const int* active_rows,const int* active_counts){
  __shared__ CompactGreedyTableStorage storage;
  GreedyTableStorage& s=storage.core;
  const int t=threadIdx.x,n_packages=*package_count;
  const int n_active_keys=active_counts[0],n_active_rows=active_counts[1];
  for(int i=t;i<R*WORDS;i+=THREADS){int r=i/WORDS,w=i%WORDS;Word m=0;
    for(int bit=0;bit<64;++bit){int e=w*64+bit;if(e<E&&support[r*E+e])m|=Word(1)<<bit;}s.resident[r][w]=m;}
  if(t<R)s.current_extra[t]=extra[t];
  if(t==0){s.accepted=0;s.changed_targets=(1u<<R)-1;}

  // The diagnostic-only PackagesPerThread=1 variant requires cap<=THREADS:
  // retained_count is min(package_count,cap), so thread t owns at most package t.
  // Production retains the original MAXP-derived loop/storage after GPU tests
  // showed the smaller specialization was slower. Candidate IDs are unchanged.
  int forward_slot[PackagesPerThread],backward_slot[PackagesPerThread];
  #pragma unroll
  for(int i=0;i<PackagesPerThread;++i){
    forward_slot[i]=backward_slot[i]=-1;
    int p=t+i*THREADS;
    // The generic path skips all package reads when status is nonzero. Keep
    // that property even though metadata is now loaded before the iterations.
    if(p<n_packages&&*status==0){
      const int* q=packages+selected_ids[p]*5;
      int ka=bucket(q[0],q[1],q[2]),kb=bucket(q[1],q[0],q[2]);
      for(int j=0;j<sizes[ka];++j)if(records[ka*RETAINED+j]==q[3])forward_slot[i]=ka*RETAINED+j;
      for(int j=0;j<sizes[kb];++j)if(records[kb*RETAINED+j]==q[4])backward_slot[i]=kb*RETAINED+j;
    }
  }
  // Static exact coverage is built once. A duplicated expert in two groups
  // occupies two bits; adding that expert sets BOTH. The own-group bit range
  // has no duplicates, so popcount still counts distinct added experts.
  for(int entry=t;entry<n_active_rows;entry+=THREADS){
    const int row=active_rows[entry],key=row/RETAINED,size=key%STRIDE;
    const int rec=records[row];unsigned coverage=0;
    for(int j=0;j<sizes[key];++j){const int other=records[key*RETAINED+j];
      for(int k=0;k<size;++k){const int expert=groups[other*K+k];bool included=false;
        for(int q=0;q<size;++q)included|=groups[rec*K+q]==expert;
        if(included)coverage|=1u<<(j*K+k);
      }
    }
    storage.coverage[row]=coverage;
  }
  __syncthreads();

  for(int iteration=0;iteration<R*B&&*status==0;++iteration){
    // All cached values for old->new depend only on resident[new]. Every
    // target is initialized once; after accepting a package, only targets
    // receiving at least one genuinely new expert need their tables rebuilt.
    // Keep full package rescoring below: both endpoints' global budgets still
    // participate in feasibility, and the original deterministic tie remains.
    const unsigned changed=s.changed_targets;
    // The baseline counts include every retained group that is already
    // executable, including groups with zero standalone communication gain.
    for(int entry=t;entry<n_active_keys;entry+=THREADS){
      const int key=active_keys[entry],target=(key/STRIDE)%R,size=key%STRIDE;
      if(!(changed&(1u<<target)))continue;
      const unsigned group_bits=(1u<<size)-1u;
      unsigned present=0;int x0=0,x1=0;
      for(int j=0;j<sizes[key];++j){const int rec=records[key*RETAINED+j];
        for(int k=0;k<size;++k){const int expert=groups[rec*K+k];
          if((s.resident[target][expert/64]>>(expert%64))&1ull)present|=1u<<(j*K+k);
        }
        const unsigned required=group_bits<<(j*K);
        if((required&~present)==0){if(classes[rec]&1)x1+=freq[rec];else x0+=freq[rec];}
      }
      storage.present[key]=present;s.e0[key]=x0;s.e1[key]=x1;
    }
    // Every active row's key is present in active_keys. All readers must see
    // its freshly updated present mask before scoring rows on other warps.
    __syncthreads();
    for(int entry=t;entry<n_active_rows;entry+=THREADS){
      const int row=active_rows[entry],key=row/RETAINED,j=row%RETAINED;
      const int target=(key/STRIDE)%R,size=key%STRIDE;
      if(!(changed&(1u<<target)))continue;
      const unsigned group_bits=(1u<<size)-1u,present=storage.present[key];
      s.missing[row]=__popc((group_bits<<(j*K))&~present);
      const unsigned after=present|storage.coverage[row];int x0=0,x1=0;
      for(int k=0;k<sizes[key];++k){
        if(((group_bits<<(k*K))&~after)==0){const int rec=records[key*RETAINED+k];
          if(classes[rec]&1)x1+=freq[rec];else x0+=freq[rec];}
      }
      s.after0[row]=x0;s.after1[row]=x1;
    }
    __syncthreads();
    Score best{0,0,MAXP};
    #pragma unroll
    for(int i=0;i<PackagesPerThread;++i){int p=t+i*THREADS;
      if(p<n_packages){
        int sa=forward_slot[i],sb=backward_slot[i],ka=sa/RETAINED,kb=sb/RETAINED;
        int a=ka/(R*STRIDE),b=(ka/STRIDE)%R;
        // a->b adds the forward group to b, while b->a adds the reverse
        // group to a. Budgets remain global across all 28 card pairs.
        int ca=s.missing[sb],cb=s.missing[sa],cost=ca+cb;
        if(cost==0||s.current_extra[a]+ca>B||s.current_extra[b]+cb>B)continue;
        int gain=match_gain(s.after0[sa],s.after1[sa],s.after0[sb],s.after1[sb])
          -match_gain(s.e0[ka],s.e1[ka],s.e0[kb],s.e1[kb]);
        best=BestScore()(best,Score{gain,cost,selected_ids[p]});
      }
    }
    Score found=GreedyReduce(s.reduction).Reduce(best,BestScore());
    if(t==0)s.winner=found;__syncthreads();
    if(s.winner.gain<=0)break;
    if(t==0){const int* q=packages+s.winner.idx*5;int a=q[0],b=q[1];int64_t* h=history+s.accepted*HISTORY_WIDTH;
      h[0]=s.winner.idx;h[1]=s.winner.gain;h[2]=a;h[3]=b;h[4]=q[2];h[5]=s.winner.cost;
      s.changed_targets=0;
      for(int w=0;w<WORDS;++w){Word aa=masks[q[4]*WORDS+w]&~s.resident[a][w],ab=masks[q[3]*WORDS+w]&~s.resident[b][w];
        h[6+w]=int64_t(aa);h[6+WORDS+w]=int64_t(ab);s.current_extra[a]+=__popcll(aa);s.current_extra[b]+=__popcll(ab);
        if(aa)s.changed_targets|=1u<<a;if(ab)s.changed_targets|=1u<<b;
        s.resident[a][w]|=aa;s.resident[b][w]|=ab;}++s.accepted;
    }
    __syncthreads();
  }
  for(int i=t;i<R*E;i+=THREADS){int r=i/E,e=i%E;support[i]=(s.resident[r][e/64]>>(e%64))&1;}
  if(t<R){extra[t]=s.current_extra[t];int n=0;
    for(int e=0;e<E;++e)if(e/HOME!=t&&((s.resident[t][e/64]>>(e%64))&1)){if(n<B)output_etc[t*B+n]=e;++n;}
    while(n<B)output_etc[t*B+n++]=-1;}
  if(t==0)*history_count=s.accepted;
}

// Candidate pruning ranks complete packages once at mandatory support.
// All original direction-group records remain in the score tables: a package's
// gain still measures all groups enabled by its additions, not only itself.
struct InitialBefore {
  __host__ __device__ bool operator()(const Score& a,const Score& b)const{
    const bool av=a.gain>0,bv=b.gain>0;
    if(av!=bv)return av;
    if(!av)return a.idx<b.idx;
    const int64_t lhs=int64_t(a.gain)*b.cost,rhs=int64_t(b.gain)*a.cost;
    if(lhs!=rhs)return lhs>rhs;
    if(a.gain!=b.gain)return a.gain>b.gain;
    if(a.cost!=b.cost)return a.cost<b.cost;
    return a.idx<b.idx;
  }
};
__global__ void initial_package_scores(const Word* masks,const int* freq,const int* classes,
 const int* records,const int* sizes,const int* packages,const int* package_count,
 const bool* support,const int* extra,const int* status,Score* scores,int* diagnostics,int B){
  __shared__ Word resident[R][WORDS];
  int t=threadIdx.x;
  for(int i=t;i<R*WORDS;i+=THREADS){int r=i/WORDS,w=i%WORDS;Word m=0;
    for(int bit=0;bit<64;++bit){int e=w*64+bit;if(e<E&&support[r*E+e])m|=Word(1)<<bit;}
    resident[r][w]=m;
  }
  __syncthreads();
  int p=blockIdx.x*THREADS+t;if(p>=MAXP)return;
  Score score{0,0,p};
  if(p<*package_count&&*status==0){
    const int* q=packages+p*5;int a=q[0],b=q[1],size=q[2],ra=q[3],rb=q[4];
    int ca=0,cb=0;
    for(int w=0;w<WORDS;++w){ca+=__popcll(masks[rb*WORDS+w]&~resident[a][w]);
      cb+=__popcll(masks[ra*WORDS+w]&~resident[b][w]);}
    int cost=ca+cb;
    if(cost>0&&extra[a]+ca<=B&&extra[b]+cb<=B){
      int ka=bucket(a,b,size),kb=bucket(b,a,size);
      int x0=0,x1=0,y0=0,y1=0,bx0=0,bx1=0,by0=0,by1=0;
      for(int j=0;j<sizes[ka];++j){int rec=records[ka*RETAINED+j];bool before=true,after=true;
        for(int w=0;w<WORDS;++w){before&=!(masks[rec*WORDS+w]&~resident[b][w]);
          after&=!(masks[rec*WORDS+w]&~(resident[b][w]|masks[ra*WORDS+w]));}
        int f=freq[rec];bool gain=classes[rec]&1;
        if(before){if(gain)bx1+=f;else bx0+=f;}if(after){if(gain)x1+=f;else x0+=f;}
      }
      for(int j=0;j<sizes[kb];++j){int rec=records[kb*RETAINED+j];bool before=true,after=true;
        for(int w=0;w<WORDS;++w){before&=!(masks[rec*WORDS+w]&~resident[a][w]);
          after&=!(masks[rec*WORDS+w]&~(resident[a][w]|masks[rb*WORDS+w]));}
        int f=freq[rec];bool gain=classes[rec]&1;
        if(before){if(gain)by1+=f;else by0+=f;}if(after){if(gain)y1+=f;else y0+=f;}
      }
      int gain=match_gain(x0,x1,y0,y1)-match_gain(bx0,bx1,by0,by1);
      if(gain>0){score={gain,cost,p};atomicAdd(diagnostics,1);}
    }
  }
  scores[p]=score;
}
__global__ void retain_package_ids(const Score* sorted,const int* package_count,
 int* selected_ids,int* retained_count,int* diagnostics,int cap){
  const int count=min(*package_count,cap),p=blockIdx.x*THREADS+threadIdx.x;
  if(p<count)selected_ids[p]=sorted[p].idx;
  if(p==0){*retained_count=count;diagnostics[1]=*package_count;diagnostics[2]=count;}
}

// Unique active metadata is rebuilt on the GPU after every candidate prune.
// Atomic append order may vary, but each key/row computes independent integer
// state and the greedy selection tie remains the original package ID.
__global__ void build_active_tables(const int* packages,const int* selected_ids,
 const int* retained_count,const int* records,const int* sizes,const int* status,
 int* active_keys,int* active_rows,int* active_flags,int* active_counts){
  int p=blockIdx.x*THREADS+threadIdx.x;
  if(p>=*retained_count||*status)return;
  const int* q=packages+selected_ids[p]*5;
  int ka=bucket(q[0],q[1],q[2]),kb=bucket(q[1],q[0],q[2]);
  const int keys[2]={ka,kb},recs[2]={q[3],q[4]};
  for(int direction=0;direction<2;++direction){
    int key=keys[direction];
    if(atomicCAS(active_flags+key,0,1)==0){
      int pos=atomicAdd(active_counts,1);active_keys[pos]=key;
    }
    for(int j=0;j<sizes[key];++j){
      int row=key*RETAINED+j;
      if(records[row]!=recs[direction])continue;
      if(atomicCAS(active_flags+BUCKETS+row,0,1)==0){
        int pos=atomicAdd(active_counts+1,1);active_rows[pos]=row;
      }
      break;
    }
  }
}

// Bounded execution uses the original logical records/tiles. Physical CTA order
// never determines candidate IDs, group frequency ties, or stable ordering.
__global__ void sample_records_bounded(const int* sampled,const int64_t* ids,const bool* support,const int* extra,
 int* classes,int* groups,Word* sig0,Word* sig1,Word* masks,const int* status,int S,int B){
  for(int linear=blockIdx.x*THREADS+threadIdx.x;linear<PAIRS*D;linear+=gridDim.x*THREADS){
    int p=linear/D,t=linear%D;
    int a,b;pair_ids(p,a,b);int idx=p*D+t,src=int(ids[t]/S);
    for(int w=0;w<WORDS;++w)masks[idx*WORDS+w]=0;
    for(int k=0;k<K;++k)groups[idx*K+k]=-1;
    classes[idx]=-1;sig0[idx]=sig1[idx]=0;
    if(*status)continue;
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
    if(dir<0||(dir?vb:va)<0)continue;
    int n=dir?nb:na,g[K];for(int k=0;k<n;++k)g[k]=dir?gb[k]:ga[k];
    for(int k=1;k<n;++k){int v=g[k],j=k-1;while(j>=0&&g[j]>v){g[j+1]=g[j];--j;}g[j+1]=v;}
    Word h=0xcbf29ce484222325ULL,other=0x9e3779b97f4a7c15ULL;
    for(int k=0;k<n;++k){int e=g[k];groups[idx*K+k]=e;
      h=(h^Word(e+1))*0x100000001b3ULL;other=mix64(other^Word(e+1));
      masks[idx*WORDS+e/64]|=Word(1)<<(e%64);}
    sig0[idx]=h;sig1[idx]=other;classes[idx]=cls(dir,n,dir?vb:va);
  }
}
__global__ void record_frequencies_bounded(const int* classes,const int* groups,const Word* sig0,const Word* sig1,int* freq){
  __shared__ int c[THREADS];__shared__ Word h[THREADS],l[THREADS];
  for(int tile=blockIdx.x;tile<PAIRS*((D+THREADS-1)/THREADS);tile+=gridDim.x){
    int lane=threadIdx.x,t=(tile%((D+THREADS-1)/THREADS))*THREADS+lane,base=(tile/((D+THREADS-1)/THREADS))*D;
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
    __syncthreads();
  }
}
// Each warp owns one class and scans all D records in 32-record segments.
// Warps select independently, so no CTA-wide barrier is needed when an empty
// class terminates early. The comparator is exactly the original total order:
// frequency descending, full expert-list lexicographic order, then record ID.
// Exact warp frequency implementation, shared by diagnostics and EP8 sharding.
// One warp owns one output record;
// lanes partition the D comparisons, then reduce count and first occurrence.
// Full exact expert lists are checked after BOTH hashes and class match.
// shards=1 measures the local warp variant. shards=8 emits only flat record
// IDs shard_rank + 8*i into packed[i]; all-gather + unshard restores freq.
__global__ void diagnostic_frequency_warp(const int* classes,const int* groups,
 const Word* sig0,const Word* sig1,int* packed,int shard_rank,int shards){
  constexpr int WARPS=THREADS/32;
  const int lane=threadIdx.x&31,warp=threadIdx.x/32;
  const int count=(PAIRS*D+shards-1)/shards;
  for(int out=blockIdx.x*WARPS+warp;out<count;out+=gridDim.x*WARPS){
    const int idx=out*shards+shard_rank;
    if(idx>=PAIRS*D){if(lane==0)packed[out]=0;continue;}
    const int base=(idx/D)*D,own=idx%D,own_class=classes[idx];
    const Word own_h=sig0[idx],own_l=sig1[idx];
    const int size=own_class<0?0:(own_class/2)%STRIDE;
    int matches=0,first=D;
    if(own_class>=0){
      for(int j=lane;j<D;j+=32){const int other=base+j;
        if(classes[other]==own_class&&sig0[other]==own_h&&sig1[other]==own_l&&
           equal_group(idx,other,size,groups)){++matches;first=min(first,j);}
      }
    }
    #pragma unroll
    for(int delta=16;delta>0;delta/=2){
      matches+=__shfl_down_sync(0xffffffffu,matches,delta);
      first=min(first,__shfl_down_sync(0xffffffffu,first,delta));
    }
    if(lane==0)packed[out]=first==own?matches:0;
  }
}
__global__ void diagnostic_frequency_unpack(const int* gathered,int* freq){
  constexpr int SHARD_WORDS=(PAIRS*D+R-1)/R;
  for(int i=blockIdx.x*THREADS+threadIdx.x;i<PAIRS*D;i+=gridDim.x*THREADS)
    freq[i]=gathered[(i%R)*SHARD_WORDS+i/R];
}

__global__ void class_topg_warp_bounded(const int* classes,const int* freq,const int* groups,int* tops,int* first){
  constexpr int WARPS=THREADS/32;
  const int lane=threadIdx.x&31,warp=threadIdx.x/32;
  BestGroup better{freq,groups};
  for(int cc=blockIdx.x*WARPS+warp;cc<PAIRS*CLASSES;cc+=gridDim.x*WARPS){
    const int c=cc%CLASSES,base=(cc/CLASSES)*D;
    int selected[G];
    #pragma unroll
    for(int j=0;j<G;++j)selected[j]=-1;
    // One writer initializes and later overwrites this row. An early break
    // must leave every unused top-G slot at -1, including poisoned replays.
    if(lane==0)for(int j=0;j<G;++j)tops[cc*G+j]=-1;
    for(int g=0;g<G;++g){
      GroupChoice local{-1,D};
      for(int t=lane;t<D;t+=32){
        const int rec=base+t;
        if(classes[rec]!=c)continue;
        // first is the first class occurrence, NOT the first positive or
        // unselected frequency. This also covers classes with only freq=0.
        local.first=min(local.first,t);
        if(freq[rec]<=0)continue;
        bool used=false;
        #pragma unroll
        for(int j=0;j<G;++j)if(j<g)used|=selected[j]==rec;
        if(!used&&better.better(rec,local.id))local.id=rec;
      }
      #pragma unroll
      for(int delta=16;delta>0;delta/=2){
        const GroupChoice other{__shfl_down_sync(0xffffffffu,local.id,delta),
                                __shfl_down_sync(0xffffffffu,local.first,delta)};
        if(lane+delta<32)local=better(local,other);
      }
      const int chosen=__shfl_sync(0xffffffffu,local.id,0);
      // Constant-index writes keep the small selected list register-friendly.
      #pragma unroll
      for(int j=0;j<G;++j)if(j==g)selected[j]=chosen;
      if(lane==0){tops[cc*G+g]=chosen;if(g==0)first[cc]=local.first;}
      // chosen is broadcast to every lane, hence this exit is warp-uniform.
      if(chosen<0)break;
    }
  }
}
__global__ void assemble_groups_bounded(const int* tops,const int* first,const int* classes,int* records,
 int* sizes,int* offsets){
  for(int task=blockIdx.x*THREADS+threadIdx.x;task<PAIR_SIZES;task+=gridDim.x*THREADS){
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
}
__global__ void fill_packages_bounded(const int* classes,const int* records,const int* sizes,const int* offsets,int* packages){
  for(int task=blockIdx.x*THREADS+threadIdx.x;task<PAIR_SIZES;task+=gridDim.x*THREADS){
    int p=task/K,size=task%K+1,a,b;pair_ids(p,a,b);int out=offsets[task];
    int ka=bucket(a,b,size),kb=bucket(b,a,size);
    for(int ia=0;ia<sizes[ka];++ia)for(int ib=0;ib<sizes[kb];++ib){
      int ra=records[ka*RETAINED+ia],rb=records[kb*RETAINED+ib];
      if((classes[ra]&1)+(classes[rb]&1)==0)continue;
      if(out<MAXP){int* q=packages+out*5;q[0]=a;q[1]=b;q[2]=size;q[3]=ra;q[4]=rb;}++out;}
  }
}
__global__ void initial_package_scores_bounded(const Word* masks,const int* freq,const int* classes,
 const int* records,const int* sizes,const int* packages,const int* package_count,
 const bool* support,const int* extra,const int* status,Score* scores,int* diagnostics,int B){
  __shared__ Word resident[R][WORDS];
  int t=threadIdx.x;
  for(int i=t;i<R*WORDS;i+=THREADS){int r=i/WORDS,w=i%WORDS;Word m=0;
    for(int bit=0;bit<64;++bit){int e=w*64+bit;if(e<E&&support[r*E+e])m|=Word(1)<<bit;}
    resident[r][w]=m;
  }
  __syncthreads();
  for(int p=blockIdx.x*THREADS+t;p<MAXP;p+=gridDim.x*THREADS){
    Score score{0,0,p};
    if(p<*package_count&&*status==0){
      const int* q=packages+p*5;int a=q[0],b=q[1],size=q[2],ra=q[3],rb=q[4];
      int ca=0,cb=0;
      for(int w=0;w<WORDS;++w){ca+=__popcll(masks[rb*WORDS+w]&~resident[a][w]);
        cb+=__popcll(masks[ra*WORDS+w]&~resident[b][w]);}
      int cost=ca+cb;
      if(cost>0&&extra[a]+ca<=B&&extra[b]+cb<=B){
        int ka=bucket(a,b,size),kb=bucket(b,a,size);
        int x0=0,x1=0,y0=0,y1=0,bx0=0,bx1=0,by0=0,by1=0;
        for(int j=0;j<sizes[ka];++j){int rec=records[ka*RETAINED+j];bool before=true,after=true;
          for(int w=0;w<WORDS;++w){before&=!(masks[rec*WORDS+w]&~resident[b][w]);
            after&=!(masks[rec*WORDS+w]&~(resident[b][w]|masks[ra*WORDS+w]));}
          int f=freq[rec];bool gain=classes[rec]&1;
          if(before){if(gain)bx1+=f;else bx0+=f;}if(after){if(gain)x1+=f;else x0+=f;}
        }
        for(int j=0;j<sizes[kb];++j){int rec=records[kb*RETAINED+j];bool before=true,after=true;
          for(int w=0;w<WORDS;++w){before&=!(masks[rec*WORDS+w]&~resident[a][w]);
            after&=!(masks[rec*WORDS+w]&~(resident[a][w]|masks[rb*WORDS+w]));}
          int f=freq[rec];bool gain=classes[rec]&1;
          if(before){if(gain)by1+=f;else by0+=f;}if(after){if(gain)y1+=f;else y0+=f;}
        }
        int gain=match_gain(x0,x1,y0,y1)-match_gain(bx0,bx1,by0,by1);
        if(gain>0){score={gain,cost,p};atomicAdd(diagnostics,1);}
      }
    }
    scores[p]=score;
  }
}
__global__ void retain_package_ids_bounded(const Score* sorted,const int* package_count,
 int* selected_ids,int* retained_count,int* diagnostics,int cap){
  const int count=min(*package_count,cap);
  for(int p=blockIdx.x*THREADS+threadIdx.x;p<MAXP;p+=gridDim.x*THREADS){
    if(p<count)selected_ids[p]=sorted[p].idx;
    if(p==0){*retained_count=count;diagnostics[1]=*package_count;diagnostics[2]=count;}
  }
}
__global__ void build_active_tables_bounded(const int* packages,const int* selected_ids,
 const int* retained_count,const int* records,const int* sizes,const int* status,
 int* active_keys,int* active_rows,int* active_flags,int* active_counts){
  if(*status)return;
  for(int p=blockIdx.x*THREADS+threadIdx.x;p<*retained_count;p+=gridDim.x*THREADS){
    const int* q=packages+selected_ids[p]*5;
    int ka=bucket(q[0],q[1],q[2]),kb=bucket(q[1],q[0],q[2]);
    const int keys[2]={ka,kb},recs[2]={q[3],q[4]};
    for(int direction=0;direction<2;++direction){
      int key=keys[direction];
      if(atomicCAS(active_flags+key,0,1)==0){
        int pos=atomicAdd(active_counts,1);active_keys[pos]=key;
      }
      for(int j=0;j<sizes[key];++j){
        int row=key*RETAINED+j;
        if(records[row]!=recs[direction])continue;
        if(atomicCAS(active_flags+BUCKETS+row,0,1)==0){
          int pos=atomicAdd(active_counts+1,1);active_rows[pos]=row;
        }
        break;
      }
    }
  }
}


// Exact total-order sorting. Each bounded CTA processes fixed 256-entry runs;
// logarithmic merge passes use original package IDs as the final tie-breaker.
// No CUB device kernel is launched on the SM-budget path.
__device__ void sort_score_tile(Score* values){
  const int t=threadIdx.x;InitialBefore before;
  __syncthreads();
  for(int width=2;width<=THREADS;width*=2){
    for(int distance=width/2;distance>0;distance/=2){
      const Score a=values[t],b=values[t^distance];
      const bool want_first=((t&distance)==0)==((t&width)==0);
      const Score result=want_first?(before(b,a)?b:a):(before(a,b)?b:a);
      __syncthreads();values[t]=result;__syncthreads();
    }
  }
}
__global__ void score_blocks_bounded(const Score* input,Score* output){
  __shared__ Score values[THREADS];const int t=threadIdx.x;
  for(int tile=blockIdx.x;tile<(MAXP+THREADS-1)/THREADS;tile+=gridDim.x){
    const int p=tile*THREADS+t;values[t]=p<MAXP?input[p]:Score{0,0,p};
    sort_score_tile(values);if(p<MAXP)output[p]=values[t];__syncthreads();
  }
}
__global__ void merge_scores_bounded(const Score* input,Score* output,int width){
  InitialBefore before;
  for(int p=blockIdx.x*THREADS+threadIdx.x;p<MAXP;p+=gridDim.x*THREADS){
    const int begin=(p/(2*width))*(2*width),mid=min(begin+width,MAXP),end=min(begin+2*width,MAXP);
    const bool left=p<mid;const int own_begin=left?begin:mid,other_begin=left?mid:begin,other_end=left?end:mid;
    const Score value=input[p];int lo=other_begin,hi=other_end;
    while(lo<hi){const int m=lo+(hi-lo)/2;if(before(input[m],value))lo=m+1;else hi=m;}
    output[begin+(p-own_begin)+(lo-other_begin)]=value;
  }
}
__global__ void reset_pruning_bounded(int* diagnostics,int* flags,int* counts){
  for(int i=blockIdx.x*THREADS+threadIdx.x;i<BUCKETS+GREEDY_ROWS;i+=gridDim.x*THREADS)flags[i]=0;
  if(blockIdx.x==0){if(threadIdx.x<3)diagnostics[threadIdx.x]=0;if(threadIdx.x<2)counts[threadIdx.x]=0;}
}
__global__ void copy_packet_bounded(const int* input,int* output){
  for(int i=blockIdx.x*THREADS+threadIdx.x;i<PACKET_SIZE;i+=gridDim.x*THREADS)output[i]=input[i];
}
__global__ void initial_score_blocks_bounded(const Word* masks,const int* freq,const int* classes,
 const int* records,const int* sizes,const int* packages,const int* package_count,
 const bool* support,const int* extra,const int* status,Score* scores,int* diagnostics,int B){
  __shared__ Word resident[R][WORDS];
  __shared__ Score tile_scores[THREADS];
  int t=threadIdx.x;
  for(int i=t;i<R*WORDS;i+=THREADS){int r=i/WORDS,w=i%WORDS;Word m=0;
    for(int bit=0;bit<64;++bit){int e=w*64+bit;if(e<E&&support[r*E+e])m|=Word(1)<<bit;}
    resident[r][w]=m;
  }
  __syncthreads();
  for(int tile=blockIdx.x;tile<(MAXP+THREADS-1)/THREADS;tile+=gridDim.x){
  const int p=tile*THREADS+t;
  Score score{0,0,p};
  if(p<*package_count&&*status==0){
    const int* q=packages+p*5;int a=q[0],b=q[1],size=q[2],ra=q[3],rb=q[4];
    int ca=0,cb=0;
    for(int w=0;w<WORDS;++w){ca+=__popcll(masks[rb*WORDS+w]&~resident[a][w]);
      cb+=__popcll(masks[ra*WORDS+w]&~resident[b][w]);}
    int cost=ca+cb;
    if(cost>0&&extra[a]+ca<=B&&extra[b]+cb<=B){
      int ka=bucket(a,b,size),kb=bucket(b,a,size);
      int x0=0,x1=0,y0=0,y1=0,bx0=0,bx1=0,by0=0,by1=0;
      for(int j=0;j<sizes[ka];++j){int rec=records[ka*RETAINED+j];bool before=true,after=true;
        for(int w=0;w<WORDS;++w){before&=!(masks[rec*WORDS+w]&~resident[b][w]);
          after&=!(masks[rec*WORDS+w]&~(resident[b][w]|masks[ra*WORDS+w]));}
        int f=freq[rec];bool gain=classes[rec]&1;
        if(before){if(gain)bx1+=f;else bx0+=f;}if(after){if(gain)x1+=f;else x0+=f;}
      }
      for(int j=0;j<sizes[kb];++j){int rec=records[kb*RETAINED+j];bool before=true,after=true;
        for(int w=0;w<WORDS;++w){before&=!(masks[rec*WORDS+w]&~resident[a][w]);
          after&=!(masks[rec*WORDS+w]&~(resident[a][w]|masks[rb*WORDS+w]));}
        int f=freq[rec];bool gain=classes[rec]&1;
        if(before){if(gain)by1+=f;else by0+=f;}if(after){if(gain)y1+=f;else y0+=f;}
      }
      int gain=match_gain(x0,x1,y0,y1)-match_gain(bx0,bx1,by0,by1);
      if(gain>0){score={gain,cost,p};atomicAdd(diagnostics,1);}
    }
  }
  tile_scores[t]=score;
  sort_score_tile(tile_scores);
  if(p<MAXP)scores[p]=tile_scores[t];
  __syncthreads();

  }
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
 at::Tensor& output_etc,at::Tensor& history,at::Tensor& history_count,at::Tensor& status,int64_t S,int64_t B,int64_t sm_limit,bool fuse){
  TORCH_CHECK(packet.is_cuda()&&B>=1&&B<=E-HOME&&S>=1&&int64_t(R)*S>=D,"invalid packet/B/S");
  TORCH_CHECK(sm_limit>=0&&sm_limit<=1024,"invalid SM budget");
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
  if(sm_limit>0){
    const int samples_grid=std::min<int64_t>((PAIRS*D+THREADS-1)/THREADS,sm_limit);
    const int frequency_grid=std::min<int64_t>(PAIRS*((D+THREADS-1)/THREADS),sm_limit);
    const int class_grid=std::min<int64_t>((PAIRS*CLASSES+THREADS/32-1)/(THREADS/32),sm_limit);
    const int package_grid=std::min<int64_t>((PAIR_SIZES+THREADS-1)/THREADS,sm_limit);
    sample_records_bounded<<<samples_grid,THREADS,0,stream>>>(samples.data_ptr<int>(),ids.data_ptr<int64_t>(),support.data_ptr<bool>(),extra.data_ptr<int>(),classes.data_ptr<int>(),groups.data_ptr<int>(),w(sig0),w(sig1),w(masks),status.data_ptr<int>(),S,B);
    record_frequencies_bounded<<<frequency_grid,THREADS,0,stream>>>(classes.data_ptr<int>(),groups.data_ptr<int>(),w(sig0),w(sig1),freq.data_ptr<int>());
    class_topg_warp_bounded<<<class_grid,THREADS,0,stream>>>(classes.data_ptr<int>(),freq.data_ptr<int>(),groups.data_ptr<int>(),tops.data_ptr<int>(),first.data_ptr<int>());
    assemble_groups_bounded<<<package_grid,THREADS,0,stream>>>(tops.data_ptr<int>(),first.data_ptr<int>(),classes.data_ptr<int>(),records.data_ptr<int>(),sizes.data_ptr<int>(),offsets.data_ptr<int>());
    prefix_packages<<<1,THREADS,0,stream>>>(records.data_ptr<int>(),sizes.data_ptr<int>(),offsets.data_ptr<int>(),package_count.data_ptr<int>(),status.data_ptr<int>());
    fill_packages_bounded<<<package_grid,THREADS,0,stream>>>(classes.data_ptr<int>(),records.data_ptr<int>(),sizes.data_ptr<int>(),offsets.data_ptr<int>(),packages.data_ptr<int>());
  }else{
  dim3 grid((D+THREADS-1)/THREADS,PAIRS);
  sample_records<<<grid,THREADS,0,stream>>>(samples.data_ptr<int>(),ids.data_ptr<int64_t>(),support.data_ptr<bool>(),extra.data_ptr<int>(),classes.data_ptr<int>(),groups.data_ptr<int>(),w(sig0),w(sig1),w(masks),status.data_ptr<int>(),S,B);
  record_frequencies<<<grid,THREADS,0,stream>>>(classes.data_ptr<int>(),groups.data_ptr<int>(),w(sig0),w(sig1),freq.data_ptr<int>());
  class_topg<<<PAIRS*CLASSES,THREADS,0,stream>>>(classes.data_ptr<int>(),freq.data_ptr<int>(),groups.data_ptr<int>(),tops.data_ptr<int>(),first.data_ptr<int>());
  assemble_groups<<<(PAIR_SIZES+THREADS-1)/THREADS,THREADS,0,stream>>>(tops.data_ptr<int>(),first.data_ptr<int>(),classes.data_ptr<int>(),records.data_ptr<int>(),sizes.data_ptr<int>(),offsets.data_ptr<int>());
  prefix_packages<<<1,THREADS,0,stream>>>(records.data_ptr<int>(),sizes.data_ptr<int>(),offsets.data_ptr<int>(),package_count.data_ptr<int>(),status.data_ptr<int>());
  fill_packages<<<(PAIR_SIZES+THREADS-1)/THREADS,THREADS,0,stream>>>(classes.data_ptr<int>(),records.data_ptr<int>(),sizes.data_ptr<int>(),offsets.data_ptr<int>(),packages.data_ptr<int>());
  }
  if constexpr(USE_GREEDY_TABLE){
    global_greedy_table<><<<1,THREADS,0,stream>>>(w(masks),freq.data_ptr<int>(),classes.data_ptr<int>(),records.data_ptr<int>(),sizes.data_ptr<int>(),packages.data_ptr<int>(),package_count.data_ptr<int>(),support.data_ptr<bool>(),extra.data_ptr<int>(),output_etc.data_ptr<int>(),history.data_ptr<int64_t>(),history_count.data_ptr<int>(),status.data_ptr<int>(),B);
  }else{
    global_greedy<<<1,THREADS,0,stream>>>(w(masks),freq.data_ptr<int>(),classes.data_ptr<int>(),records.data_ptr<int>(),sizes.data_ptr<int>(),packages.data_ptr<int>(),package_count.data_ptr<int>(),support.data_ptr<bool>(),extra.data_ptr<int>(),output_etc.data_ptr<int>(),history.data_ptr<int64_t>(),history_count.data_ptr<int>(),status.data_ptr<int>(),B);
  }
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
void select_capped_active_phase(int64_t phase,const at::Tensor& packet,const at::Tensor& ids,const at::Tensor& etc,
 at::Tensor& samples,at::Tensor& support,at::Tensor& extra,at::Tensor& seen,at::Tensor& classes,at::Tensor& groups,
 at::Tensor& sig0,at::Tensor& sig1,at::Tensor& masks,at::Tensor& freq,at::Tensor& tops,at::Tensor& first,
 at::Tensor& records,at::Tensor& sizes,at::Tensor& offsets,at::Tensor& packages,at::Tensor& package_count,
 at::Tensor& output_etc,at::Tensor& history,at::Tensor& history_count,at::Tensor& status,int64_t S,int64_t B,at::Tensor& scores_in,at::Tensor& scores_out,at::Tensor& sort_scratch,at::Tensor& selected_ids,at::Tensor& retained_count,at::Tensor& diagnostics,int64_t cap,at::Tensor& active_keys,at::Tensor& active_rows,at::Tensor& active_flags,at::Tensor& active_counts,int64_t sm_limit,bool fuse){
  TORCH_CHECK(packet.is_cuda()&&B>=1&&B<=E-HOME&&S>=1&&int64_t(R)*S>=D,"invalid packet/B/S");
  TORCH_CHECK(sm_limit>=0&&sm_limit<=1024,"invalid SM budget");
  TORCH_CHECK(phase>=0&&phase<=2&&(phase==0||sm_limit>0),"invalid capped selector phase/budget");
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
  TORCH_CHECK(cap>0&&cap<=MAXP,"cap must be 1..MAXP; use select for unlimited");
  check(scores_in,packet,at::kInt,MAXP*3,"scores in");check(scores_out,packet,at::kInt,MAXP*3,"scores out");
  check(selected_ids,packet,at::kInt,MAXP,"selected IDs");check(retained_count,packet,at::kInt,1,"retained count");
  check(diagnostics,packet,at::kInt,3,"pruning diagnostics");
  TORCH_CHECK(sort_scratch.is_cuda()&&sort_scratch.device()==packet.device()&&sort_scratch.scalar_type()==at::kByte&&sort_scratch.is_contiguous(),"invalid sort scratch");
  check(active_keys,packet,at::kInt,BUCKETS,"active keys");
  check(active_rows,packet,at::kInt,GREEDY_ROWS,"active rows");
  check(active_flags,packet,at::kInt,BUCKETS+GREEDY_ROWS,"active flags");
  check(active_counts,packet,at::kInt,2,"active counts");
  c10::cuda::CUDAGuard guard(packet.device());auto stream=c10::cuda::getCurrentCUDAStream();
  auto w=[](at::Tensor& x){return reinterpret_cast<Word*>(x.data_ptr<int64_t>());};
  // Phase 0 is the original complete local path. Phase 1 only builds
  // records; phase 2 consumes externally reconstructed exact frequencies.
  // Phase 1 must still run on invalid input, and callers must execute the
  // same intervening collective on every rank before phase 2.
  if(phase!=2){
    mandatory_validate<<<1,THREADS,0,stream>>>(packet.data_ptr<int>(),etc.data_ptr<int>(),samples.data_ptr<int>(),support.data_ptr<bool>(),extra.data_ptr<int>(),seen.data_ptr<int>(),status.data_ptr<int>(),B);
    if(sm_limit>0){
      const int samples_grid=std::min<int64_t>((PAIRS*D+THREADS-1)/THREADS,sm_limit);
      sample_records_bounded<<<samples_grid,THREADS,0,stream>>>(samples.data_ptr<int>(),ids.data_ptr<int64_t>(),support.data_ptr<bool>(),extra.data_ptr<int>(),classes.data_ptr<int>(),groups.data_ptr<int>(),w(sig0),w(sig1),w(masks),status.data_ptr<int>(),S,B);
    }else{
      dim3 grid((D+THREADS-1)/THREADS,PAIRS);
      sample_records<<<grid,THREADS,0,stream>>>(samples.data_ptr<int>(),ids.data_ptr<int64_t>(),support.data_ptr<bool>(),extra.data_ptr<int>(),classes.data_ptr<int>(),groups.data_ptr<int>(),w(sig0),w(sig1),w(masks),status.data_ptr<int>(),S,B);
    }
    if(phase==1){C10_CUDA_KERNEL_LAUNCH_CHECK();return;}
    if(sm_limit>0){
      const int frequency_grid=std::min<int64_t>(PAIRS*((D+THREADS-1)/THREADS),sm_limit);
      record_frequencies_bounded<<<frequency_grid,THREADS,0,stream>>>(classes.data_ptr<int>(),groups.data_ptr<int>(),w(sig0),w(sig1),freq.data_ptr<int>());
    }else{
      dim3 grid((D+THREADS-1)/THREADS,PAIRS);
      record_frequencies<<<grid,THREADS,0,stream>>>(classes.data_ptr<int>(),groups.data_ptr<int>(),w(sig0),w(sig1),freq.data_ptr<int>());
    }
  }
  if(sm_limit>0){
    const int class_grid=std::min<int64_t>((PAIRS*CLASSES+THREADS/32-1)/(THREADS/32),sm_limit);
    const int package_grid=std::min<int64_t>((PAIR_SIZES+THREADS-1)/THREADS,sm_limit);
    class_topg_warp_bounded<<<class_grid,THREADS,0,stream>>>(classes.data_ptr<int>(),freq.data_ptr<int>(),groups.data_ptr<int>(),tops.data_ptr<int>(),first.data_ptr<int>());
    assemble_groups_bounded<<<package_grid,THREADS,0,stream>>>(tops.data_ptr<int>(),first.data_ptr<int>(),classes.data_ptr<int>(),records.data_ptr<int>(),sizes.data_ptr<int>(),offsets.data_ptr<int>());
    prefix_packages<<<1,THREADS,0,stream>>>(records.data_ptr<int>(),sizes.data_ptr<int>(),offsets.data_ptr<int>(),package_count.data_ptr<int>(),status.data_ptr<int>());
    fill_packages_bounded<<<package_grid,THREADS,0,stream>>>(classes.data_ptr<int>(),records.data_ptr<int>(),sizes.data_ptr<int>(),offsets.data_ptr<int>(),packages.data_ptr<int>());
  }else{
    class_topg<<<PAIRS*CLASSES,THREADS,0,stream>>>(classes.data_ptr<int>(),freq.data_ptr<int>(),groups.data_ptr<int>(),tops.data_ptr<int>(),first.data_ptr<int>());
    assemble_groups<<<(PAIR_SIZES+THREADS-1)/THREADS,THREADS,0,stream>>>(tops.data_ptr<int>(),first.data_ptr<int>(),classes.data_ptr<int>(),records.data_ptr<int>(),sizes.data_ptr<int>(),offsets.data_ptr<int>());
    prefix_packages<<<1,THREADS,0,stream>>>(records.data_ptr<int>(),sizes.data_ptr<int>(),offsets.data_ptr<int>(),package_count.data_ptr<int>(),status.data_ptr<int>());
    fill_packages<<<(PAIR_SIZES+THREADS-1)/THREADS,THREADS,0,stream>>>(classes.data_ptr<int>(),records.data_ptr<int>(),sizes.data_ptr<int>(),offsets.data_ptr<int>(),packages.data_ptr<int>());
  }
  auto* in=reinterpret_cast<Score*>(scores_in.data_ptr<int>());
  auto* out=reinterpret_cast<Score*>(scores_out.data_ptr<int>());
  const Score* sorted=out;
  const int score_grid=sm_limit>0?std::min<int64_t>((MAXP+THREADS-1)/THREADS,sm_limit):(MAXP+THREADS-1)/THREADS;
  if(sm_limit>0){
    reset_pruning_bounded<<<std::min<int64_t>((BUCKETS+GREEDY_ROWS+THREADS-1)/THREADS,sm_limit),THREADS,0,stream>>>(diagnostics.data_ptr<int>(),active_flags.data_ptr<int>(),active_counts.data_ptr<int>());
    if(fuse){
      initial_score_blocks_bounded<<<score_grid,THREADS,0,stream>>>(w(masks),freq.data_ptr<int>(),classes.data_ptr<int>(),records.data_ptr<int>(),sizes.data_ptr<int>(),packages.data_ptr<int>(),package_count.data_ptr<int>(),support.data_ptr<bool>(),extra.data_ptr<int>(),status.data_ptr<int>(),out,diagnostics.data_ptr<int>(),B);
    }else{
      initial_package_scores_bounded<<<score_grid,THREADS,0,stream>>>(w(masks),freq.data_ptr<int>(),classes.data_ptr<int>(),records.data_ptr<int>(),sizes.data_ptr<int>(),packages.data_ptr<int>(),package_count.data_ptr<int>(),support.data_ptr<bool>(),extra.data_ptr<int>(),status.data_ptr<int>(),in,diagnostics.data_ptr<int>(),B);
      score_blocks_bounded<<<score_grid,THREADS,0,stream>>>(in,out);
    }
    Score* current=out;Score* next=in;
    for(int width=THREADS;width<MAXP;width*=2){
      merge_scores_bounded<<<score_grid,THREADS,0,stream>>>(current,next,width);
      std::swap(current,next);
    }
    sorted=current;
    retain_package_ids_bounded<<<score_grid,THREADS,0,stream>>>(sorted,package_count.data_ptr<int>(),selected_ids.data_ptr<int>(),retained_count.data_ptr<int>(),diagnostics.data_ptr<int>(),cap);
  }else{
  C10_CUDA_CHECK(cudaMemsetAsync(diagnostics.data_ptr<int>(),0,3*sizeof(int),stream));
  initial_package_scores<<<(MAXP+THREADS-1)/THREADS,THREADS,0,stream>>>(w(masks),freq.data_ptr<int>(),classes.data_ptr<int>(),records.data_ptr<int>(),sizes.data_ptr<int>(),packages.data_ptr<int>(),package_count.data_ptr<int>(),support.data_ptr<bool>(),extra.data_ptr<int>(),status.data_ptr<int>(),in,diagnostics.data_ptr<int>(),B);
  size_t scratch_bytes=sort_scratch.numel();
  C10_CUDA_CHECK(cub::DeviceMergeSort::SortKeysCopy(sort_scratch.data_ptr(),scratch_bytes,in,out,MAXP,InitialBefore{},stream));
  retain_package_ids<<<(MAXP+THREADS-1)/THREADS,THREADS,0,stream>>>(out,package_count.data_ptr<int>(),selected_ids.data_ptr<int>(),retained_count.data_ptr<int>(),diagnostics.data_ptr<int>(),cap);
  }
  if constexpr(USE_GREEDY_TABLE){
    if(sm_limit>0){
      build_active_tables_bounded<<<score_grid,THREADS,0,stream>>>(packages.data_ptr<int>(),selected_ids.data_ptr<int>(),retained_count.data_ptr<int>(),records.data_ptr<int>(),sizes.data_ptr<int>(),status.data_ptr<int>(),active_keys.data_ptr<int>(),active_rows.data_ptr<int>(),active_flags.data_ptr<int>(),active_counts.data_ptr<int>());
    }else{
    C10_CUDA_CHECK(cudaMemsetAsync(active_flags.data_ptr<int>(),0,(BUCKETS+GREEDY_ROWS)*sizeof(int),stream));
    C10_CUDA_CHECK(cudaMemsetAsync(active_counts.data_ptr<int>(),0,2*sizeof(int),stream));
    build_active_tables<<<(MAXP+THREADS-1)/THREADS,THREADS,0,stream>>>(packages.data_ptr<int>(),selected_ids.data_ptr<int>(),retained_count.data_ptr<int>(),records.data_ptr<int>(),sizes.data_ptr<int>(),status.data_ptr<int>(),active_keys.data_ptr<int>(),active_rows.data_ptr<int>(),active_flags.data_ptr<int>(),active_counts.data_ptr<int>());
    }
    if constexpr(USE_COMPACT_GREEDY_TABLE){
      if(sm_limit>0){
        global_greedy_compact_active<><<<1,THREADS,0,stream>>>(w(masks),freq.data_ptr<int>(),classes.data_ptr<int>(),groups.data_ptr<int>(),records.data_ptr<int>(),sizes.data_ptr<int>(),packages.data_ptr<int>(),retained_count.data_ptr<int>(),selected_ids.data_ptr<int>(),support.data_ptr<bool>(),extra.data_ptr<int>(),output_etc.data_ptr<int>(),history.data_ptr<int64_t>(),history_count.data_ptr<int>(),status.data_ptr<int>(),B,active_keys.data_ptr<int>(),active_rows.data_ptr<int>(),active_counts.data_ptr<int>());
      }else{
    global_greedy_table_active<><<<1,THREADS,0,stream>>>(w(masks),freq.data_ptr<int>(),classes.data_ptr<int>(),records.data_ptr<int>(),sizes.data_ptr<int>(),packages.data_ptr<int>(),retained_count.data_ptr<int>(),selected_ids.data_ptr<int>(),support.data_ptr<bool>(),extra.data_ptr<int>(),output_etc.data_ptr<int>(),history.data_ptr<int64_t>(),history_count.data_ptr<int>(),status.data_ptr<int>(),B,active_keys.data_ptr<int>(),active_rows.data_ptr<int>(),active_counts.data_ptr<int>());
      }
    }else{
    global_greedy_table_active<><<<1,THREADS,0,stream>>>(w(masks),freq.data_ptr<int>(),classes.data_ptr<int>(),records.data_ptr<int>(),sizes.data_ptr<int>(),packages.data_ptr<int>(),retained_count.data_ptr<int>(),selected_ids.data_ptr<int>(),support.data_ptr<bool>(),extra.data_ptr<int>(),output_etc.data_ptr<int>(),history.data_ptr<int64_t>(),history_count.data_ptr<int>(),status.data_ptr<int>(),B,active_keys.data_ptr<int>(),active_rows.data_ptr<int>(),active_counts.data_ptr<int>());
    }
  }else{
    global_greedy_capped<<<1,THREADS,0,stream>>>(w(masks),freq.data_ptr<int>(),classes.data_ptr<int>(),records.data_ptr<int>(),sizes.data_ptr<int>(),packages.data_ptr<int>(),retained_count.data_ptr<int>(),selected_ids.data_ptr<int>(),support.data_ptr<bool>(),extra.data_ptr<int>(),output_etc.data_ptr<int>(),history.data_ptr<int64_t>(),history_count.data_ptr<int>(),status.data_ptr<int>(),B);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Preserve the existing low-level entry point and its complete local path.
void select_capped_active(const at::Tensor& packet,const at::Tensor& ids,const at::Tensor& etc,
 at::Tensor& samples,at::Tensor& support,at::Tensor& extra,at::Tensor& seen,at::Tensor& classes,at::Tensor& groups,
 at::Tensor& sig0,at::Tensor& sig1,at::Tensor& masks,at::Tensor& freq,at::Tensor& tops,at::Tensor& first,
 at::Tensor& records,at::Tensor& sizes,at::Tensor& offsets,at::Tensor& packages,at::Tensor& package_count,
 at::Tensor& output_etc,at::Tensor& history,at::Tensor& history_count,at::Tensor& status,int64_t S,int64_t B,at::Tensor& scores_in,at::Tensor& scores_out,at::Tensor& sort_scratch,at::Tensor& selected_ids,at::Tensor& retained_count,at::Tensor& diagnostics,int64_t cap,at::Tensor& active_keys,at::Tensor& active_rows,at::Tensor& active_flags,at::Tensor& active_counts,int64_t sm_limit,bool fuse){
  select_capped_active_phase(0,packet,ids,etc,samples,support,extra,seen,classes,groups,sig0,sig1,masks,freq,tops,first,records,sizes,offsets,packages,package_count,output_etc,history,history_count,status,S,B,scores_in,scores_out,sort_scratch,selected_ids,retained_count,diagnostics,cap,active_keys,active_rows,active_flags,active_counts,sm_limit,fuse);
}

// Frequency entries share exact kernels between diagnostics and the bounded
// production path. The caller supplies preallocated outputs and explicitly
// performs one peer all-gather between shard and unshard. The diagnostic local
// entry retains both baselines; production enables sharding only for the
// measured EP8/E352/K8/D256/G2/32-SM configuration.
void diagnostic_frequency_local(const at::Tensor& classes,const at::Tensor& groups,
 const at::Tensor& sig0,const at::Tensor& sig1,at::Tensor& freq,int64_t sm_limit,bool warp){
  TORCH_CHECK(classes.is_cuda()&&sm_limit>0&&sm_limit<=1024,"invalid frequency diagnostic device/budget");
  check(classes,classes,at::kInt,PAIRS*D,"classes");check(groups,classes,at::kInt,PAIRS*D*K,"groups");
  check(sig0,classes,at::kLong,PAIRS*D,"sig0");check(sig1,classes,at::kLong,PAIRS*D,"sig1");
  check(freq,classes,at::kInt,PAIRS*D,"frequency output");
  c10::cuda::CUDAGuard guard(classes.device());auto stream=c10::cuda::getCurrentCUDAStream();
  auto h0=reinterpret_cast<const Word*>(sig0.data_ptr<int64_t>()),h1=reinterpret_cast<const Word*>(sig1.data_ptr<int64_t>());
  if(warp){
    const int grid=std::min<int64_t>((PAIRS*D+THREADS/32-1)/(THREADS/32),sm_limit);
    diagnostic_frequency_warp<<<grid,THREADS,0,stream>>>(classes.data_ptr<int>(),groups.data_ptr<int>(),h0,h1,freq.data_ptr<int>(),0,1);
  }else{
    const int grid=std::min<int64_t>(PAIRS*((D+THREADS-1)/THREADS),sm_limit);
    record_frequencies_bounded<<<grid,THREADS,0,stream>>>(classes.data_ptr<int>(),groups.data_ptr<int>(),h0,h1,freq.data_ptr<int>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void diagnostic_frequency_shard(const at::Tensor& classes,const at::Tensor& groups,
 const at::Tensor& sig0,const at::Tensor& sig1,at::Tensor& packed,int64_t shard_rank,int64_t sm_limit){
  TORCH_CHECK(classes.is_cuda()&&sm_limit>0&&sm_limit<=1024&&shard_rank>=0&&shard_rank<R,
    "invalid frequency shard device/budget/rank");
  check(classes,classes,at::kInt,PAIRS*D,"classes");check(groups,classes,at::kInt,PAIRS*D*K,"groups");
  check(sig0,classes,at::kLong,PAIRS*D,"sig0");check(sig1,classes,at::kLong,PAIRS*D,"sig1");
  constexpr int SHARD_WORDS=(PAIRS*D+R-1)/R;
  check(packed,classes,at::kInt,SHARD_WORDS,"packed frequency shard");
  c10::cuda::CUDAGuard guard(classes.device());auto stream=c10::cuda::getCurrentCUDAStream();
  const int grid=std::min<int64_t>((SHARD_WORDS+THREADS/32-1)/(THREADS/32),sm_limit);
  diagnostic_frequency_warp<<<grid,THREADS,0,stream>>>(classes.data_ptr<int>(),groups.data_ptr<int>(),
    reinterpret_cast<const Word*>(sig0.data_ptr<int64_t>()),reinterpret_cast<const Word*>(sig1.data_ptr<int64_t>()),
    packed.data_ptr<int>(),shard_rank,R);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void diagnostic_frequency_unshard(const at::Tensor& gathered,at::Tensor& freq,int64_t sm_limit){
  TORCH_CHECK(gathered.is_cuda()&&sm_limit>0&&sm_limit<=1024,"invalid frequency unshard device/budget");
  check(gathered,gathered,at::kInt,R*((PAIRS*D+R-1)/R),"gathered frequency shards");
  check(freq,gathered,at::kInt,PAIRS*D,"frequency output");
  c10::cuda::CUDAGuard guard(gathered.device());auto stream=c10::cuda::getCurrentCUDAStream();
  const int grid=std::min<int64_t>((PAIRS*D+THREADS-1)/THREADS,sm_limit);
  diagnostic_frequency_unpack<<<grid,THREADS,0,stream>>>(gathered.data_ptr<int>(),freq.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Diagnostic-only replay of already-built candidates. This is deliberately
// not a planner configuration option: it compares old/compact greedy kernels
// with identical mandatory initialization, candidate IDs and active tables.
// Call select first and retain the same packet/native copies and metadata.
void diagnostic_greedy_replay(const at::Tensor& packet,const at::Tensor& etc,
 at::Tensor& samples,at::Tensor& support,at::Tensor& extra,at::Tensor& seen,
 const at::Tensor& masks,const at::Tensor& freq,const at::Tensor& classes,const at::Tensor& groups,
 const at::Tensor& records,const at::Tensor& sizes,const at::Tensor& packages,
 const at::Tensor& retained_count,const at::Tensor& selected_ids,
 at::Tensor& output_etc,at::Tensor& history,at::Tensor& history_count,at::Tensor& status,
 const at::Tensor& active_keys,const at::Tensor& active_rows,const at::Tensor& active_counts,
 int64_t B,int64_t cap,bool compact,bool single_package){
  TORCH_CHECK(packet.is_cuda()&&B>=1&&B<=E-HOME,"invalid diagnostic packet/B");
  TORCH_CHECK(cap>0&&cap<=MAXP,"diagnostic cap must match the prepared candidate limit");
  TORCH_CHECK(!single_package||(compact&&cap<=THREADS),
    "single-package diagnostic requires compact=true and prepared cap<=256");
  TORCH_CHECK(USE_GREEDY_TABLE,"diagnostic replay requires the active-table configuration");
  TORCH_CHECK(!compact||USE_COMPACT_GREEDY_TABLE,"compact greedy is unsupported for this K/G/shared-memory size");
  check(packet,packet,at::kInt,PACKET_SIZE,"packet");check(etc,packet,at::kInt,R*B,"native copies");
  check(samples,packet,at::kInt,D*2*K,"samples");check(support,packet,at::kBool,R*E,"support");
  check(extra,packet,at::kInt,R,"extra");check(seen,packet,at::kInt,R*E,"copy seen");
  check(masks,packet,at::kLong,PAIRS*D*WORDS,"masks");check(freq,packet,at::kInt,PAIRS*D,"frequencies");
  check(classes,packet,at::kInt,PAIRS*D,"classes");check(groups,packet,at::kInt,PAIRS*D*K,"groups");
  check(records,packet,at::kInt,BUCKETS*RETAINED,"records");check(sizes,packet,at::kInt,BUCKETS,"sizes");
  check(packages,packet,at::kInt,MAXP*5,"packages");check(retained_count,packet,at::kInt,1,"retained count");
  check(selected_ids,packet,at::kInt,MAXP,"selected IDs");check(output_etc,packet,at::kInt,R*B,"output copies");
  check(history,packet,at::kLong,R*B*HISTORY_WIDTH,"history");check(history_count,packet,at::kInt,1,"history count");
  check(status,packet,at::kInt,1,"status");check(active_keys,packet,at::kInt,BUCKETS,"active keys");
  check(active_rows,packet,at::kInt,GREEDY_ROWS,"active rows");check(active_counts,packet,at::kInt,2,"active counts");
  c10::cuda::CUDAGuard guard(packet.device());auto stream=c10::cuda::getCurrentCUDAStream();
  mandatory_validate<<<1,THREADS,0,stream>>>(packet.data_ptr<int>(),etc.data_ptr<int>(),samples.data_ptr<int>(),support.data_ptr<bool>(),extra.data_ptr<int>(),seen.data_ptr<int>(),status.data_ptr<int>(),B);
  const auto* m=reinterpret_cast<const Word*>(masks.data_ptr<int64_t>());
  if constexpr(USE_GREEDY_TABLE){
    if constexpr(USE_COMPACT_GREEDY_TABLE){
      if(compact){
        if(single_package){
          global_greedy_compact_active<1><<<1,THREADS,0,stream>>>(m,freq.data_ptr<int>(),classes.data_ptr<int>(),groups.data_ptr<int>(),records.data_ptr<int>(),sizes.data_ptr<int>(),packages.data_ptr<int>(),retained_count.data_ptr<int>(),selected_ids.data_ptr<int>(),support.data_ptr<bool>(),extra.data_ptr<int>(),output_etc.data_ptr<int>(),history.data_ptr<int64_t>(),history_count.data_ptr<int>(),status.data_ptr<int>(),B,active_keys.data_ptr<int>(),active_rows.data_ptr<int>(),active_counts.data_ptr<int>());
        }else{
          global_greedy_compact_active<><<<1,THREADS,0,stream>>>(m,freq.data_ptr<int>(),classes.data_ptr<int>(),groups.data_ptr<int>(),records.data_ptr<int>(),sizes.data_ptr<int>(),packages.data_ptr<int>(),retained_count.data_ptr<int>(),selected_ids.data_ptr<int>(),support.data_ptr<bool>(),extra.data_ptr<int>(),output_etc.data_ptr<int>(),history.data_ptr<int64_t>(),history_count.data_ptr<int>(),status.data_ptr<int>(),B,active_keys.data_ptr<int>(),active_rows.data_ptr<int>(),active_counts.data_ptr<int>());
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();return;
      }
    }
    global_greedy_table_active<><<<1,THREADS,0,stream>>>(m,freq.data_ptr<int>(),classes.data_ptr<int>(),records.data_ptr<int>(),sizes.data_ptr<int>(),packages.data_ptr<int>(),retained_count.data_ptr<int>(),selected_ids.data_ptr<int>(),support.data_ptr<bool>(),extra.data_ptr<int>(),output_etc.data_ptr<int>(),history.data_ptr<int64_t>(),history_count.data_ptr<int>(),status.data_ptr<int>(),B,active_keys.data_ptr<int>(),active_rows.data_ptr<int>(),active_counts.data_ptr<int>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int64_t pruning_workspace_bytes(const at::Tensor& scores_in,const at::Tensor& scores_out){
  check(scores_in,scores_in,at::kInt,MAXP*3,"scores in");check(scores_out,scores_in,at::kInt,MAXP*3,"scores out");
  c10::cuda::CUDAGuard guard(scores_in.device());auto stream=c10::cuda::getCurrentCUDAStream();
  auto* in=reinterpret_cast<const Score*>(scores_in.data_ptr<int>());
  auto* out=reinterpret_cast<Score*>(scores_out.data_ptr<int>());
  size_t bytes=0;C10_CUDA_CHECK(cub::DeviceMergeSort::SortKeysCopy(nullptr,bytes,in,out,MAXP,InitialBefore{},stream));
  return int64_t(bytes);
}

void copy_packet(const at::Tensor& input,at::Tensor& output,int64_t sm_limit){
  TORCH_CHECK(sm_limit>0&&sm_limit<=1024,"invalid SM budget");
  check(input,input,at::kInt,PACKET_SIZE,"input packet");check(output,input,at::kInt,PACKET_SIZE,"output packet");
  c10::cuda::CUDAGuard guard(input.device());
  copy_packet_bounded<<<std::min<int64_t>((PACKET_SIZE+THREADS-1)/THREADS,sm_limit),THREADS,0,c10::cuda::getCurrentCUDAStream()>>>(input.data_ptr<int>(),output.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
  m.def("select_capped_active",&select_capped_active);
  m.def("select_capped_active_phase",&select_capped_active_phase);
  m.def("pruning_workspace_bytes",&pruning_workspace_bytes);
  m.def("uses_active_table",[](){return USE_GREEDY_TABLE;});
  m.def("uses_compact_greedy_table",[](){return USE_COMPACT_GREEDY_TABLE;});
  m.def("diagnostic_greedy_replay",&diagnostic_greedy_replay);
  m.def("diagnostic_frequency_local",&diagnostic_frequency_local);
  m.def("diagnostic_frequency_shard",&diagnostic_frequency_shard);
  m.def("diagnostic_frequency_unshard",&diagnostic_frequency_unshard);
  m.def("frequency_shard",&diagnostic_frequency_shard);
  m.def("frequency_unshard",&diagnostic_frequency_unshard);
  m.def("gather",&gather);
  m.def("copy_packet",&copy_packet);
  m.def("select",&select_residency);
  m.def("count_frequencies",&count_frequencies);
}
