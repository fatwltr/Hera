#include "hera/blb_reference.h"
#include "hera/seal_wire.h"
#include "blb_ring_protocol.h"
#include "OT/ot_pack.h"
#include "utils/io_pack.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <csignal>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using namespace hera;
using Clock=std::chrono::steady_clock;
namespace blb=conversion_internal::blb;
double Ms(Clock::time_point start)
{ return std::chrono::duration<double,std::milli>(Clock::now()-start).count(); }
void Check(bool ok,const std::string &what) { if(!ok) throw std::runtime_error(what); }
template<class T> T Take(Result<T> value)
{ Check(value.ok(),value.status().message());return value.take_value(); }
void Ok(Result<void> value) { Check(value.ok(),value.status().message()); }
void Exchange(int party,IProtocolTransport &wire,const void *local,void *peer,std::size_t n)
{
    if(party==sci::ALICE){wire.Send(local,n);wire.Receive(peer,n);}
    else {wire.Receive(peer,n);wire.Send(local,n);}
}
void Barrier(int party,IProtocolTransport &wire)
{ std::uint64_t token=1,peer=0;Exchange(party,wire,&token,&peer,sizeof(token));Check(peer==1,"bad barrier"); }
std::uint64_t Next(std::uint64_t &x)
{ x^=x<<13;x^=x>>7;x^=x<<17;return x; }
struct Input {
    std::vector<std::complex<double>> clear;
    ComplexMpcSharePair narrow;
    std::optional<SealCipherBatch> encrypted;
};
Input MakeInput(int party,const HeContext &profile,IProtocolTransport &wire)
{
    Input input;
    const auto slots=profile.slot_count();
    input.clear.resize(slots);
    input.narrow.real={std::vector<std::uint64_t>(slots),{{1,slots}},{43,13},ComplexComponent::kReal};
    input.narrow.imag={std::vector<std::uint64_t>(slots),{{1,slots}},{43,13},ComplexComponent::kImaginary};
    constexpr auto mask=(std::uint64_t{1}<<43)-1;
    std::uint64_t state=0x61a756deadbeefULL;
    for(std::size_t i=0;i<slots;++i) {
        input.clear[i]={double(int(i%17)-8)/32,double(int(i%13)-6)/32};
        for(bool imag:{false,true}) {
            const auto a=Next(state)&mask;
            const auto value=std::uint64_t(std::int64_t(std::llround(
                (imag?input.clear[i].imag():input.clear[i].real())*8192)));
            (imag?input.narrow.imag:input.narrow.real).local_share[i]=party==sci::ALICE?a:(value-a)&mask;
        }
    }
    if(party==sci::BOB) {
        seal::CKKSEncoder encoder(*profile.context);seal::Plaintext plain;
        encoder.encode(input.clear,std::ldexp(1.,profile.parameters.input_scale_bits),plain);
        seal::Encryptor encryptor(*profile.context,*profile.secret_key);seal::Ciphertext ct;
        encryptor.encrypt_symmetric(plain,ct);SendSealObject(wire,ct,seal::compr_mode_type::none);
    } else input.encrypted=SealCipherBatch{profile.profile,
        {ReceiveSealObject<seal::Ciphertext>(wire,*profile.context)},
        {{2*slots}},Layout::kSpatial,PhysicalSlotMap::DenseComplex(2*slots,slots)};
    return input;
}
struct Sample {
    double ms{},ring_ms{},decode_ms{},prepare_ms{},encode_ms{},extend_ms{},rns_ms{};
    std::uint64_t bytes{},ring_bytes{};
};
Sample Combine(Sample a,const Sample &b)
{
    a.ms=std::max(a.ms,b.ms);a.ring_ms=std::max(a.ring_ms,b.ring_ms);
    a.decode_ms=std::max(a.decode_ms,b.decode_ms);a.prepare_ms=std::max(a.prepare_ms,b.prepare_ms);
    a.encode_ms=std::max(a.encode_ms,b.encode_ms);a.extend_ms=std::max(a.extend_ms,b.extend_ms);
    a.rns_ms=std::max(a.rns_ms,b.rns_ms);a.bytes+=b.bytes;a.ring_bytes+=b.ring_bytes;
    return a;
}
void Verify(int party,IProtocolTransport &wire,const HeContext &profile,const Input &input,
            const ComplexMpcSharePair &shares,const std::optional<seal::Ciphertext> &ct,bool c2m)
{
    double error=0;
    if(c2m) {
        constexpr auto mask=(std::uint64_t{1}<<43)-1;
        for(bool imag:{false,true}) {
            const auto &local=(imag?shares.imag:shares.real).local_share;
            auto peer=local;Exchange(party,wire,local.data(),peer.data(),local.size()*sizeof(local[0]));
            for(std::size_t i=0;i<local.size();++i) {
                const auto raw=(local[i]+peer[i])&mask;
                const auto value=std::int64_t(raw&(std::uint64_t{1}<<42)?raw|~mask:raw);
                const auto target=std::llround((imag?input.clear[i].imag():input.clear[i].real())*8192);
                error=std::max(error,double(std::llabs(value-target))/8192);
            }
        }
        Check(error<=4./8192,"C2M oracle failed");
    } else if(party==sci::ALICE) {
        Check(ct.has_value(),"missing server ciphertext");SendSealObject(wire,*ct,seal::compr_mode_type::none);
    } else {
        auto output=ReceiveSealObject<seal::Ciphertext>(wire,*profile.context);
        seal::Plaintext plain;seal::Decryptor decryptor(*profile.context,*profile.secret_key);
        decryptor.decrypt(output,plain);seal::CKKSEncoder encoder(*profile.context);
        std::vector<std::complex<double>> decoded;encoder.decode(plain,decoded);
        for(std::size_t i=0;i<decoded.size();++i)error=std::max(error,std::abs(decoded[i]-input.clear[i]));
        Check(error<4./8192,"M2C oracle failed");
    }
    double peer;Exchange(party,wire,&error,&peer,sizeof(error));
    if(party==sci::ALICE)std::cout<<"BLB_COST_CHECK,error="<<std::max(error,peer)<<",oracle=PASS\n";
}
void Benchmark(int party,std::shared_ptr<IProtocolTransport> wire,HeContextRegistry &registry,
               const HeContext &profile,const Input &input,int operation,bool quick)
{
    const auto role=party==sci::ALICE?PartyRole::kServer:PartyRole::kClient;
    const bool bounded=true;
    const std::string op=operation==0?"C2M":operation==1?"M2C":"M2C_PREPARED";
    const auto n=profile.parameters.polynomial_degree;
    Barrier(party,*wire);
    const auto started=Clock::now();
    ConversionConfig cfg{43,13,40,20,true,0,true,true};
    BlbReferenceConverter converter(role,cfg,registry,wire,MakeProductionRandomSource(),
        "127.0.0.1",24400,1,seal::compr_mode_type::none);
    Sample setup;setup.ms=Ms(started);setup.bytes=converter.stats().communication_bytes-wire->bytes();
    Sample peer;Exchange(party,*wire,&setup,&peer,sizeof(setup));setup=Combine(setup,peer);
    const auto limit=static_cast<std::int64_t>(converter.stats().forward_ot_capacity);
    if(party==sci::ALICE)std::cout<<"BLB_COST_SETUP,n="<<n<<",bounded="<<bounded<<",op="<<op
        <<",ms="<<setup.ms<<",both_bytes="<<setup.bytes<<",ot_limit="<<limit<<'\n';
    const auto map=PhysicalSlotMap::DenseComplex(n,n/2);
    const MpcToCkksTarget target{profile.profile,0,profile.parameters.input_scale_bits,{{n}},Layout::kSpatial,map};
    PreparedBlbSlots prepared;
    if(operation==2) {
        // Secure preparation is actually performed, outside this API's timing.
        // M2C above measures it on EVERY call. This row is the prepared core only.
        auto lanes=input.narrow.real.local_share;
        lanes.insert(lanes.end(),input.narrow.imag.local_share.begin(),input.narrow.imag.local_share.end());
        sci::IOPack preparation_io(party,24600,"127.0.0.1");
        sci::OTPack preparation_ot(&preparation_io,party);
        blb::RingProtocol ring(party,*preparation_ot.iknp_straight);
        auto wide=ring.ExtendSigned(lanes,43,83);
        prepared={std::vector<WideRingShare>(wide.begin(),wide.begin()+n/2),
                  std::vector<WideRingShare>(wide.begin()+n/2,wide.end()),83,13};
    }
    ComplexMpcSharePair last_shares;std::optional<SealCipherBatch> last_ct;
    auto invoke=[&] {
        if(operation==0)last_shares=Take(converter.CkksToMpc(input.encrypted));
        else if(operation==1)last_ct=Take(converter.MpcToCkks(input.narrow,target));
        else last_ct=Take(converter.MpcToCkksPrepared(prepared,target));
    };
    invoke(); // Warm transform/allocators and OT facade, excluded.
    const auto before_warm=static_cast<std::int64_t>(converter.stats().forward_ot_used);
    invoke(); // Calibrate consumption to include >=3 complete OT buffers.
    const auto after_warm=static_cast<std::int64_t>(converter.stats().forward_ot_used);
    const auto consumption=(after_warm-before_warm+limit)%limit;
    Check(consumption>0,"cannot calibrate OT consumption");
    const int runs=quick?3:std::max<std::int64_t>(30,(3*limit+consumption-1)/consumption);
    std::vector<double> times;Sample total{};int refills=0;
    const auto pool_begin=static_cast<std::int64_t>(converter.stats().forward_ot_used);
    for(int i=0;i<runs;++i) {
        Barrier(party,*wire);
        const auto previous=converter.stats();const auto bytes=previous.communication_bytes;
        const auto used=static_cast<std::int64_t>(converter.stats().forward_ot_used);const auto start=Clock::now();
        invoke();
        Sample sample;sample.ms=Ms(start);sample.bytes=converter.stats().communication_bytes-bytes;
        const auto current=converter.stats();
        sample.ring_ms=current.ckks_to_mpc.field_to_ring-previous.ckks_to_mpc.field_to_ring;
        sample.decode_ms=current.ckks_to_mpc.slot_decode-previous.ckks_to_mpc.slot_decode;
        sample.prepare_ms=current.mpc_to_ckks.slot_prepare-previous.mpc_to_ckks.slot_prepare;
        sample.encode_ms=current.mpc_to_ckks.slot_encode-previous.mpc_to_ckks.slot_encode;
        sample.extend_ms=current.mpc_to_ckks.coefficient_extend-previous.mpc_to_ckks.coefficient_extend;
        sample.rns_ms=current.mpc_to_ckks.rns-previous.mpc_to_ckks.rns;
        sample.ring_bytes=(current.field_to_ring_communication_bytes-previous.field_to_ring_communication_bytes)+
            (current.signed_extension_communication_bytes-previous.signed_extension_communication_bytes);
        if(static_cast<std::int64_t>(converter.stats().forward_ot_used)<used)++refills;
        Exchange(party,*wire,&sample,&peer,sizeof(sample));sample=Combine(sample,peer);
        times.push_back(sample.ms);total.ms+=sample.ms;total.bytes+=sample.bytes;total.ring_bytes+=sample.ring_bytes;
        total.ring_ms+=sample.ring_ms;total.decode_ms+=sample.decode_ms;total.prepare_ms+=sample.prepare_ms;
        total.encode_ms+=sample.encode_ms;total.extend_ms+=sample.extend_ms;total.rns_ms+=sample.rns_ms;
        if(party==sci::ALICE)std::cout<<"BLB_COST_SAMPLE,n="<<n<<",bounded="<<bounded<<",op="<<op
            <<",run="<<i<<",ms="<<sample.ms<<",both_bytes="<<sample.bytes
            <<",ring_ms="<<sample.ring_ms<<",prepare_ms="<<sample.prepare_ms
            <<",extend_ms="<<sample.extend_ms<<",decode_ms="<<sample.decode_ms
            <<",encode_ms="<<sample.encode_ms<<",rns_ms="<<sample.rns_ms<<'\n';
    }
    if(!quick)Check(refills>=3,"measurement did not include three OT refills");
    std::optional<seal::Ciphertext> result;
    if(last_ct) {Check(last_ct->ciphertexts.size()==1,"unexpected batch output");result=last_ct->ciphertexts.front();}
    Verify(party,*wire,profile,input,last_shares,result,operation==0);
    std::sort(times.begin(),times.end());
    if(party==sci::ALICE)std::cout<<"BLB_COST_RESULT,n="<<n<<",bounded="<<bounded<<",op="<<op
        <<",runs="<<runs<<",mean_ms="<<total.ms/runs<<",median_ms="<<times[times.size()/2]
        <<",p95_ms="<<times[(times.size()-1)*95/100]<<",both_bytes="<<double(total.bytes)/runs
        <<",ring_both_bytes="<<double(total.ring_bytes)/runs<<",f2r_ms="<<total.ring_ms/runs
        <<",prepare_ms="<<total.prepare_ms/runs<<",coefficient_extend_ms="<<total.extend_ms/runs
        <<",decode_ms="<<total.decode_ms/runs<<",encode_ms="<<total.encode_ms/runs<<",rns_ms="<<total.rns_ms/runs
        <<",refills="<<refills<<",ots_per_call="<<consumption<<",pool_begin="<<pool_begin
        <<",pool_end="<<static_cast<std::int64_t>(converter.stats().forward_ot_used)<<std::endl;
}
void Run(int party,bool quick)
{
    std::cout<<std::setprecision(10);
    const auto role=party==sci::ALICE?PartyRole::kServer:PartyRole::kClient;
    auto wire=MakeNetIoTransport(role,"127.0.0.1",24300);
    auto config=Take(LoadHeraConfig(HERA_BENCH_CONFIG_PATH));
    // Match the FFN M2C output scale; the reference uses one fixed scale in both directions.
    config.he_parameter_profiles.at("bert_base_ffn").input_scale_bits=46;
    HeContextRegistry registry;const auto start=Clock::now();
    if(party==sci::BOB) {
        Ok(registry.CreateClientProfiles(config,false));
        Ok(SendPublicContextBundle(*wire,Take(registry.ExportSerializedPublicBundle(seal::compr_mode_type::none))));
    }else Ok(registry.CreateServerProfiles(config,Take(ReceivePublicContextBundle(*wire))));
    Sample local,peer;local.ms=Ms(start);local.bytes=wire->bytes();
    Exchange(party,*wire,&local,&peer,sizeof(local));local=Combine(local,peer);
    if(party==sci::ALICE)std::cout<<"BLB_COST_HE_SETUP,ms="<<local.ms<<",both_bytes="<<local.bytes<<'\n';
    for(const auto &name:{"bert_base_ffn","bert_base_attention_main"}) {
        const auto &profile=registry.Require(name,role);const auto input=MakeInput(party,profile,*wire);
        for(int op:{0,1,2})Benchmark(party,wire,registry,profile,input,op,quick);
    }
}
}
int main(int argc,char **argv)
{
    pid_t child=-1;
    try {
        const bool quick=argc==2&&std::string(argv[1])=="--quick";
        child=fork();Check(child>=0,"fork failed");
        if(child==0){try{Run(sci::BOB,quick);std::cout.flush();_exit(0);}catch(const std::exception &e){std::cerr<<e.what()<<'\n';_exit(2);}}
        Run(sci::ALICE,quick);int status;waitpid(child,&status,0);child=-1;
        Check(WIFEXITED(status)&&WEXITSTATUS(status)==0,"benchmark client failed");
        std::cout<<"HERA_BLB_COST_STATUS=PASS\n";
    }catch(const std::exception &e){if(child>0){kill(child,SIGTERM);waitpid(child,nullptr,0);}std::cerr<<e.what()<<'\n';return 1;}
}
