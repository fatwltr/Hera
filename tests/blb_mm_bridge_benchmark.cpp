#include "bridge_adapter.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>

using Clock=std::chrono::steady_clock;
namespace {
bool quick_audit=false;
uint64_t Mix(uint64_t x) {
  x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;
  x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);
}
void Exchange(sci::NetIO &wire,int party,const void *local,void *peer,size_t n) {
  if(party==1){wire.send_data(local,n);wire.flush();wire.recv_data(peer,n);}
  else{wire.recv_data(peer,n);wire.send_data(local,n);wire.flush();}
}
struct Sample{double ms;uint64_t bytes;std::array<double,6> stages{};};
uint64_t Comm(NonLinear &nl){uint64_t bytes=0;for(int i=0;i<MAX_THREADS;++i)bytes+=nl.iopackArr[i]->get_comm();return bytes;}
template<class F>Sample Measure(NonLinear &nl,sci::NetIO &wire,int party,F &&call) {
  wire.sync();const auto before=Comm(nl);const auto stages_before=nl.conversion_stage_ms;const auto start=Clock::now();
  call();Sample local{std::chrono::duration<double,std::milli>(Clock::now()-start).count(),
                     Comm(nl)-before},peer;
  for(size_t i=0;i<local.stages.size();++i)local.stages[i]=nl.conversion_stage_ms[i]-stages_before[i];
  Exchange(wire,party,&local,&peer,sizeof(local));
  return {std::max(local.ms,peer.ms),local.bytes+peer.bytes,local.stages};
}
void Run(NonLinear &nl,sci::NetIO &wire,int party,HE_CKKS &he,int count) {
  const auto terminal=he.context->last_context_data();
  const uint64_t q=terminal->parms().coeff_modulus()[0].value();
  const int degree=he.poly_modulus_degree,slots=he.num_slots,k=40,s=20;
  const uint64_t mask=(uint64_t{1}<<k)-1;
  NativeBridge bridge{nl};
  std::vector<std::vector<uint64_t>> input(count,std::vector<uint64_t>(slots));
  std::vector<double> values(slots);
  for(int ct=0;ct<count;++ct)for(int j=0;j<slots;++j) {
    const int64_t x=(int64_t((ct*slots+j)%15)-7)*4096;
    const auto a=Mix(size_t(ct)*slots+j+911)&mask;
    input[ct][j]=party==1?a:(uint64_t(x)-a)&mask;
  }
  // Independent valid C2M fixture, not the artifact M2C output. It guarantees
  // the public coefficient condition even if the original bridge is wrong.
  std::vector<seal::Plaintext> c2m_input(count);
  uint64_t c2m_violations=0,m2c_violations=0;
  for(int ct=0;ct<count;++ct) {
    for(int j=0;j<slots;++j)values[j]=(double((ct*slots+j)%15)-7)/256;
    seal::Plaintext plain;
    he.encoder->encode(values,terminal->parms_id(),std::ldexp(1.,he.he_scale),plain);
    seal::util::inverse_ntt_negacyclic_harvey(plain.data(),terminal->small_ntt_tables()[0]);
    auto &share=c2m_input[ct];share.resize(degree);
    for(int j=0;j<degree;++j) {
      c2m_violations+=std::min(plain[j],q-plain[j])>=q/4;
      const auto a=Mix(size_t(ct)*degree+j+1211)%q;
      share[j]=party==1?a:(plain[j]+q-a)%q;
    }
    seal::util::ntt_negacyclic_harvey(share.data(),terminal->small_ntt_tables()[0]);
    share.parms_id()=terminal->parms_id();share.scale()=std::ldexp(1.,he.he_scale);
    auto encoded=he.ckks_encode(input[ct],slots);
    std::vector<uint64_t> peer(degree);
    Exchange(wire,party,encoded.data(),peer.data(),degree*8);
    for(int j=0;j<degree;++j) {
      const auto raw=(encoded[j]+peer[j])&mask;
      m2c_violations+=std::min(raw,(uint64_t{1}<<k)-raw)>=(uint64_t{1}<<(k-2));
    }
  }
  if(c2m_violations||m2c_violations)throw std::runtime_error("native bridge fixture violates quarter-range");
  if(party==1)std::cout<<"BLB_BRIDGE_RANGE,degree="<<degree<<",ciphertexts="<<count
    <<",c2m_elements="<<size_t(count)*degree<<",m2c_coefficients="<<size_t(count)*degree
    <<",violations=0"<<std::endl;
  Sample forward{},reverse{};
  std::vector<seal::Plaintext> encrypted_shares;
  std::vector<std::vector<uint64_t>> output;
  const int warmup=quick_audit?0:1,repeats=quick_audit?1:5;
  for(int run=0;run<warmup+repeats;++run) {
    auto a=Measure(nl,wire,party,[&]{encrypted_shares=bridge.mpc_to_ckks(input,slots,q,k,s,he.he_scale,&he);});
    auto b=Measure(nl,wire,party,[&]{output=bridge.ckks_to_mpc(c2m_input,degree,q,k,s,&he);});
    if(run>=warmup){forward.ms+=a.ms;forward.bytes+=a.bytes;reverse.ms+=b.ms;reverse.bytes+=b.bytes;
      for(size_t i=0;i<a.stages.size();++i){forward.stages[i]+=a.stages[i];reverse.stages[i]+=b.stages[i];}}
  }
  if(party==1 && nl.correct_conversion)std::cout<<"BLB_BRIDGE_COMPONENT,bounded="<<nl.public_quarter_range
    <<",degree="<<degree<<",ciphertexts="<<count
    <<",F2R_ms="<<reverse.stages[0]/repeats<<",slot_extend_ms="<<forward.stages[1]/repeats
    <<",coefficient_extend_ms="<<forward.stages[2]/repeats<<",decode_ms="<<reverse.stages[3]/repeats
    <<",encode_ms="<<forward.stages[4]/repeats<<",rns_ms="<<forward.stages[5]/repeats<<std::endl;
  if(party==1)for(int dir=0;dir<2;++dir) {
    const auto &sample=dir?reverse:forward;
    std::cout<<std::setprecision(10)<<"BLB_BRIDGE_FUNCTION,bounded="<<nl.public_quarter_range
      <<",corrected="<<nl.correct_conversion
      <<",degree="<<degree<<",ciphertexts="<<count<<",values="<<size_t(count)*slots
      <<",workers="<<(nl.correct_conversion?nl.conversion_workers:1)<<",direction="<<(dir?"C2M":"M2C")<<",mean_ms="<<sample.ms/repeats
      <<",both_bytes="<<sample.bytes/repeats<<",scope=local_transforms_and_ring_conversion"<<std::endl;
  }
  // Untimed output oracle. The original artifact has independent fixed-point
  // and multi-prime reconstruction problems; report failure explicitly.
  double c2m_error=0,m2c_error=0;
  for(int ct=0;ct<count;++ct) {
    std::vector<uint64_t> peer(slots);
    Exchange(wire,party,output[ct].data(),peer.data(),slots*8);
    for(int j=0;j<slots;++j) {
      const auto raw=(output[ct][j]+peer[j])&mask;
      const int64_t signed_value=raw&(uint64_t{1}<<(k-1))? -int64_t((uint64_t{1}<<k)-raw):int64_t(raw);
      const auto expected=(double((ct*slots+j)%15)-7)/256;
      c2m_error=std::max(c2m_error,std::abs(signed_value/std::ldexp(1.,s)-expected));
    }
    auto &plain=encrypted_shares[ct];std::vector<uint64_t> other(plain.coeff_count());
    Exchange(wire,party,plain.data(),other.data(),other.size()*8);
    auto data=he.context->get_context_data(plain.parms_id());
    for(size_t prime=0;prime<data->parms().coeff_modulus().size();++prime) {
      const auto modulus=data->parms().coeff_modulus()[prime].value();
      for(int j=0;j<degree;++j) {
        const auto index=prime*degree+j;plain[index]=(plain[index]+other[index])%modulus;
      }
    }
    std::vector<double> decoded;he.encoder->decode(plain,decoded);
    for(int j=0;j<slots;++j) {
      const auto expected=(double((ct*slots+j)%15)-7)/256;
      const auto error=std::abs(decoded[j]-expected);
      if(!std::isfinite(error))m2c_error=INFINITY;else m2c_error=std::max(m2c_error,error);
    }
  }
  if(party==1)std::cout<<"BLB_BRIDGE_NUMERICAL,degree="<<degree<<",ciphertexts="<<count
    <<",bounded="<<nl.public_quarter_range<<",c2m_max_error="<<c2m_error
    <<",m2c_max_error="<<m2c_error<<",c2m="<<(c2m_error<=4/std::ldexp(1.,s)?"PASS":"FAIL")
    <<",m2c="<<(m2c_error<=4/std::ldexp(1.,s)?"PASS":"FAIL")<<std::endl;
  if(nl.correct_conversion && (c2m_error>4/std::ldexp(1.,s)||m2c_error>4/std::ldexp(1.,s)))
    throw std::runtime_error("corrected BLB bridge failed its four-ULP oracle");
}

void RoundTrip(NonLinear &nl,sci::NetIO &wire,int party,HE_CKKS &he,bool half) {
  const int k=40,s=20,slots=he.num_slots,degree=he.poly_modulus_degree;
  const uint64_t mask=(uint64_t{1}<<k)-1,q=he.context->last_context_data()->parms().coeff_modulus()[0].value();
  const int lanes=half?degree:slots;
  std::vector<std::vector<uint64_t>> input(2,std::vector<uint64_t>(lanes));
  std::vector<std::vector<int64_t>> expected(2,std::vector<int64_t>(lanes));
  for(int ct=0;ct<2;++ct)for(int i=0;i<lanes;++i){
    expected[ct][i]=(int64_t((ct*lanes+i)*7%17)-8)*1024;
    const auto a=Mix(size_t(ct)*lanes+i+6411)&mask;
    input[ct][i]=party==1?a:(uint64_t(expected[ct][i])-a)&mask;
  }
  // Exercise original BLB callers' multi-prime get_Q() interface and let
  // the corrected C2M perform its own terminal-level preparation.
  NativeBridge bridge{nl};auto plain=bridge.mpc_to_ckks(input,slots,he.get_Q(),k,s,he.he_scale,&he,half);
  auto output=bridge.ckks_to_mpc(plain,degree,he.get_Q(plain[0]),k,s,&he,half);
  uint64_t max_error=0;
  for(int ct=0;ct<2;++ct){
    std::vector<uint64_t> peer(lanes);Exchange(wire,party,output[ct].data(),peer.data(),lanes*8);
    for(int i=0;i<lanes;++i){
      const auto raw=(output[ct][i]+peer[i])&mask;
      const auto value=raw&(uint64_t{1}<<(k-1))?-int64_t((uint64_t{1}<<k)-raw):int64_t(raw);
      max_error=std::max(max_error,uint64_t(std::llabs(value-expected[ct][i])));
    }
  }
  if(max_error>8)throw std::runtime_error("BLB real/complex roundtrip exceeds eight ULP");
  if(party==1)std::cout<<"BLB_BRIDGE_ROUNDTRIP,degree="<<degree<<",half="<<half
    <<",bounded="<<nl.public_quarter_range<<",max_error_ulp="<<max_error<<",threshold=8,status=PASS"<<std::endl;
}
}
int main(int argc,char **argv) {
  try{
    const int party=std::stoi(argv[1]),port=std::stoi(argv[2]);
    quick_audit=argc>3 && std::string(argv[3])=="--audit";
    sci::NetIO wire(party==1?nullptr:"127.0.0.1",port);
    NonLinear nl(party,"127.0.0.1",port+1);
    {HE_CKKS he(party,nullptr,32768,{60,34,60,60,60,60,60},20,44);
     for(int count:{1,6})Run(nl,wire,party,he,count);
     if(nl.correct_conversion)for(bool half:{false,true})RoundTrip(nl,wire,party,he,half);}
    {HE_CKKS he(party,nullptr,16384,{60,34,60,60,60,60},20,38);
     for(int count:{1,12})Run(nl,wire,party,he,count);
     if(nl.correct_conversion)for(bool half:{false,true})RoundTrip(nl,wire,party,he,half);}
    std::cout<<"BLB_BRIDGE_EXECUTION=COMPLETE,numerical_status_reported_separately"<<std::endl;
    return 0;
  }catch(const std::exception &e){std::cerr<<e.what()<<std::endl;return 1;}
}
