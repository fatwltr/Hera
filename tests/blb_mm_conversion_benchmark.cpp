#include "mm_conversion.h"
#include "nonlinear.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <thread>

using Clock = std::chrono::steady_clock;
using U128 = unsigned __int128;
namespace {
uint64_t Mix(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL; x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL; return x ^ (x >> 31);
}
struct Sample { double ms{}; uint64_t bytes{}; };
uint64_t Comm(NonLinear &nl) {
  uint64_t value=0; for (int i=0;i<MAX_THREADS;++i) value+=nl.iopackArr[i]->get_comm(); return value;
}
void Exchange(sci::NetIO &wire, int party, const void *local, void *peer, size_t bytes) {
  if (party==sci::ALICE) {wire.send_data(local,bytes);wire.flush();wire.recv_data(peer,bytes);}
  else {wire.recv_data(peer,bytes);wire.send_data(local,bytes);wire.flush();}
}
void Parallel(NonLinear &nl, size_t n, int workers,
              const std::function<void(FixOp &,size_t,size_t)> &call) {
  std::vector<std::thread> threads;
  for(int i=0;i<workers;++i) threads.emplace_back([&,i]{
    const auto begin=n*i/workers, end=n*(i+1)/workers;
    call(*nl.fpmath[i]->fix,begin,end);
  });
  for(auto &t:threads)t.join();
}
void Validate(sci::NetIO &wire,int party,const std::vector<uint64_t> &local,
              const std::vector<uint64_t> &expected,uint64_t modulus,int bits) {
  std::vector<uint64_t> other(local.size());
  Exchange(wire,party,local.data(),other.data(),local.size()*8);
  for(size_t i=0;i<local.size();++i) {
    const uint64_t value=modulus ? uint64_t((U128{local[i]}+other[i])%modulus)
                                 : (local[i]+other[i])&blb_mm::Mask(bits);
    if(value!=expected[i])throw std::runtime_error("BLB MM oracle mismatch at "+std::to_string(i));
  }
}
void Check(NonLinear &nl,sci::NetIO &wire,int party) {
  constexpr size_t n=257;
  for(uint64_t q:{uint64_t{97},uint64_t{1152921504606584833ULL}})
    for(int width:{13,40,43,51,64})for(bool bounded:{false,true}) {
      std::vector<uint64_t> input(n),output(n),expected(n);
      const int64_t edge=int64_t(q/(bounded?4:2))-1;
      const int64_t values[]={0,1,-1,edge,-edge,17,-17};
      for(size_t i=0;i<n;++i) {
        const auto x=values[i%7];const auto raw=x<0?q-uint64_t(-x):uint64_t(x);
        const auto a=Mix(i+111)%q;
        input[i]=party==sci::ALICE?a:(raw+q-a)%q;
        expected[i]=uint64_t(x)&blb_mm::Mask(width);
      }
      Parallel(nl,n,6,[&](FixOp &fix,size_t b,size_t e){
        blb_mm::PrimeToRing(fix,input.data()+b,output.data()+b,e-b,q,width,bounded);
      });
      Validate(wire,party,output,expected,0,width);
    }
  for(int width:{13,40,43,51,64})for(bool bounded:{false,true})for(int shift:{0,18,24}) {
    constexpr uint64_t q=1152921504606584833ULL;
    const auto mask=blb_mm::Mask(width);
    const int64_t edge=int64_t((uint64_t{1}<<(width-(bounded?2:1)))-1);
    const int64_t values[]={0,1,-1,edge,-edge,17,-17};
    std::vector<uint64_t> input(n),output(n),expected(n);
    for(size_t i=0;i<n;++i) {
      const auto x=values[i%7]; const auto a=Mix(i+211)&mask;
      input[i]=party==sci::ALICE?a:(uint64_t(x)-a)&mask;
      const auto value=x<0?(q-uint64_t(-x)%q)%q:uint64_t(x)%q;
      expected[i]=uint64_t((U128{value}<<shift)%q);
    }
    Parallel(nl,n,6,[&](FixOp &fix,size_t b,size_t e){
      blb_mm::RingToPrime(fix,input.data()+b,output.data()+b,e-b,width,q,bounded,shift);
    });
    Validate(wire,party,output,expected,q,width);
  }
  std::cout<<"BLB_MM_ORACLE=PASS,cases=50,elements_per_case="<<n<<std::endl;
}
void Benchmark(NonLinear &nl,sci::NetIO &wire,int party,size_t n,int workers,bool bounded,bool f2r) {
  constexpr uint64_t q=1152921504606584833ULL;
  constexpr int width=43;
  const auto mask=blb_mm::Mask(width);
  std::vector<uint64_t> input(n),output(n),expected(n);
  for(size_t i=0;i<n;++i) {
    const int64_t x=int64_t(i%8193)-4096;
    if(f2r) {
      const auto raw=x<0?q-uint64_t(-x):uint64_t(x),a=Mix(i+311)%q;
      input[i]=party==sci::ALICE?a:(raw+q-a)%q; expected[i]=uint64_t(x)&mask;
    } else {
      const auto a=Mix(i+411)&mask;
      input[i]=party==sci::ALICE?a:(uint64_t(x)-a)&mask;
      expected[i]=x<0?q-uint64_t(-x):uint64_t(x);
    }
  }
  double total_ms=0,total_bytes=0;
  for(int run=0;run<6;++run) {
    wire.sync();const auto before=Comm(nl);const auto start=Clock::now();
    Parallel(nl,n,workers,[&](FixOp &fix,size_t b,size_t e){
      if(f2r)blb_mm::PrimeToRing(fix,input.data()+b,output.data()+b,e-b,q,width,bounded);
      else blb_mm::RingToPrime(fix,input.data()+b,output.data()+b,e-b,width,q,bounded);
    });
    Sample local{std::chrono::duration<double,std::milli>(Clock::now()-start).count(),Comm(nl)-before},peer;
    Exchange(wire,party,&local,&peer,sizeof(local));
    // Native SCI counts sent bytes only; add both endpoints exactly once.
    if(run){total_ms+=std::max(local.ms,peer.ms);total_bytes+=local.bytes+peer.bytes;}
  }
  Validate(wire,party,output,expected,f2r?0:q,width);
  if(party==sci::ALICE)std::cout<<std::setprecision(10)
    <<"BLB_MM_KERNEL,direction="<<(f2r?"F2R":"R2F")<<",bounded="<<bounded
    <<",elements="<<n<<",workers="<<workers<<",mean_ms="<<total_ms/5
    <<",both_bytes="<<total_bytes/5<<",oracle=PASS"<<std::endl;
}
} // namespace
int main(int argc,char **argv) {
  const bool expect_mismatch=argc>3 && std::string(argv[3])=="--expect-contract-rejection";
  try {
    const int party=std::stoi(argv[1]);const int port=std::stoi(argv[2]);
    sci::NetIO wire(party==sci::ALICE?nullptr:"127.0.0.1",port);
    NonLinear nl(party,"127.0.0.1",port+1);
    if(expect_mismatch)throw std::runtime_error("mismatched public modes were accepted");
    Check(nl,wire,party);
    for(size_t n:{size_t{16384},size_t{196608}})for(int workers:{1,6})
      for(bool f2r:{true,false})for(bool bounded:{false,true})Benchmark(nl,wire,party,n,workers,bounded,f2r);
    std::cout<<"BLB_MM_STATUS=PASS"<<std::endl;
    return 0;
  } catch(const std::exception &e){
    if(expect_mismatch && std::string(e.what())=="BLB conversion public modes differ") {
      std::cout<<"BLB_MM_MODE_MISMATCH=PASS"<<std::endl;return 0;
    }
    std::cerr<<e.what()<<std::endl;return 1;
  }
}
