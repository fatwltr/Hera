#include "hera/blb_reference.h"
#include "hera/seal_wire.h"
#include "blb_encoding.h"
#include "blb_ring_protocol.h"
#include "OT/ot_pack.h"
#include "utils/io_pack.h"

#include <seal/util/ntt.h>
#include <algorithm>
#include <cmath>
#include <csignal>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using U128 = unsigned __int128;
using namespace hera;
namespace blb = conversion_internal::blb;
void Check(bool ok, const std::string &message)
{ if (!ok) throw std::runtime_error(message); }
template<class T> T Take(Result<T> result)
{
    Check(result.ok(), result.status().message());
    return result.take_value();
}
void Ok(Result<void> result) { Check(result.ok(), result.status().message()); }
U128 Raw(WideRingShare x) { return (U128{x.high} << 64) | x.low; }
WideRingShare Wide(U128 x) { return {static_cast<std::uint64_t>(x), static_cast<std::uint64_t>(x >> 64)}; }
std::uint64_t Next(std::uint64_t &state)
{ state ^= state << 13; state ^= state >> 7; state ^= state << 17; return state; }
void VerifyWide(int party, IProtocolTransport &wire,
    const std::vector<WideRingShare> &local, int bits, const std::vector<U128> &expected)
{
    auto peer = local;
    if (party == sci::BOB) wire.Send(peer.data(), peer.size()*sizeof(peer[0]));
    else {
        wire.Receive(peer.data(), peer.size()*sizeof(peer[0]));
        const auto mask = (U128{1} << bits)-1;
        for (std::size_t i=0; i<local.size(); ++i)
            Check(Raw(WideAdd(local[i],peer[i],bits)) == (expected[i]&mask),
                  "BLB integer protocol differs from independent oracle at " + std::to_string(i));
    }
}
void TestRing(int party, sci::OTPack &ot, IProtocolTransport &wire)
{
    blb::RingProtocol ring(party,*ot.iknp_straight);
    std::size_t checked = 0;
    {
        for (auto q : {std::uint64_t{13},std::uint64_t{97},std::uint64_t{1152921504606830593ULL}}) {
            for (int bits : {40,64,100,127}) {
                if (bits < 64 && q > (std::uint64_t{1} << bits)) continue;
                std::vector<U128> expected(257);
                std::vector<std::uint64_t> local(expected.size());
                std::uint64_t state = 0x8352736abcdefULL;
                const auto edge = static_cast<std::int64_t>(q/4-1);
                const std::int64_t values[] = {0,1,-1,edge,-edge};
                for (std::size_t i=0; i<local.size(); ++i) {
                    const auto x = values[i%5];
                    auto a = Next(state)%q;
                    if (i%11==0) a=0;
                    if (i%11==1) a=q-1;
                    if (i%11==2) a=q/2;
                    const auto clear = x<0 ? q-static_cast<std::uint64_t>(-x) : x;
                    local[i] = party==sci::ALICE ? a : (clear+q-a)%q;
                    expected[i] = static_cast<U128>(x);
                }
                const auto original = local;
                auto actual = ring.FieldToRing(local,q,bits);
                Check(local==original,"F2R mutated its input");
                VerifyWide(party,wire,actual,bits,expected); checked+=local.size();
            }
        }
        for (int source : {13,43,51,63}) {
            for (int extension : {1,40,63}) {
                const int bits=source+extension;
                if (bits>127) continue;
                const auto mask=(std::uint64_t{1}<<source)-1;
                const auto edge=(std::int64_t{1}<<(source-2))-1;
                const std::int64_t values[] = {0,1,-1,edge,-edge,
                    -edge};
                std::vector<U128> expected(257);
                std::vector<std::uint64_t> local(expected.size());
                std::uint64_t state=0x383733bb4cccULL;
                for (std::size_t i=0; i<local.size(); ++i) {
                    auto a=Next(state)&mask;
                    if (i%11==0) a=0;
                    if (i%11==1) a=mask;
                    const auto x=values[i%6];
                    local[i]=party==sci::ALICE ? a : (static_cast<std::uint64_t>(x)-a)&mask;
                    expected[i]=static_cast<U128>(x);
                }
                const auto original=local;
                VerifyWide(party,wire,ring.ExtendSigned(local,source,bits),bits,expected);
                Check(local==original,"signed extension mutated its input"); checked+=local.size();
            }
        }
        Check(ring.FieldToRing({},97,100).empty(),"empty F2R failed");
        Check(ring.ExtendSigned({},43,83).empty(),"empty extension failed");
    }
    if (party==sci::ALICE) std::cout<<"BLB_INTEGER_ORACLE_ELEMENTS="<<checked<<'\n';
}
HeraConfig Config()
{
    HeraConfig c;
    c.layers.push_back(TransformerLayerConfig{
        "test",2,4,1,1,4,4,8,NormPlacement::kPreNorm,ActivationKind::kGelu,
        AttentionMaskKind::kNone,false,false,false,true,true});
    c.mpc_numeric={43,13}; c.conversion={40,24,true}; c.threads=1;
    c.he_parameter_profiles.emplace("small",HeParameterConfig{
        2048,{40,30,40},24,24,24,1,1,true,SecurityLevel::kNone});
    c.he_parameter_profiles.emplace("large_rns",HeParameterConfig{
        4096,{50,45,45,45,50},24,24,24,1,1,true,SecurityLevel::kNone});
    LayerExecutionConfig execution;
    execution.layer_id="test";
    for (auto region : {HeRegionId::kRegion1,HeRegionId::kRegion2,HeRegionId::kRegion3,
                        HeRegionId::kRegion4,HeRegionId::kRegion5})
        execution.regions.emplace(region,HeRegionConfig{region,"small",Layout::kSpatial,Layout::kSpatial});
    c.layer_execution.push_back(std::move(execution));
    return c;
}
std::vector<std::complex<double>> Values(std::size_t slots)
{
    std::vector<std::complex<double>> x(slots);
    for (std::size_t i=0;i<slots;++i)
        x[i]={static_cast<double>(static_cast<int>(i%17)-8)/32,
              static_cast<double>(static_cast<int>(i%13)-6)/32};
    return x;
}
ComplexMpcSharePair Narrow(int party,const std::vector<std::complex<double>> &x)
{
    ComplexMpcSharePair out;
    const TensorShape shape{{1,x.size()}};
    out.real={std::vector<std::uint64_t>(x.size()),shape,{43,13},ComplexComponent::kReal};
    out.imag={std::vector<std::uint64_t>(x.size()),shape,{43,13},ComplexComponent::kImaginary};
    std::uint64_t state=0x7234367891ULL;
    const auto mask=(std::uint64_t{1}<<43)-1;
    for (std::size_t i=0;i<x.size();++i) {
        for (bool imag : {false,true}) {
            auto a=Next(state)&mask;
            const auto value=static_cast<std::uint64_t>(static_cast<std::int64_t>(
                std::llround((imag ? x[i].imag() : x[i].real())*8192)));
            (imag ? out.imag : out.real).local_share[i]=party==sci::ALICE ? a : (value-a)&mask;
        }
    }
    return out;
}
PreparedBlbSlots Prepared(int party,const std::vector<std::complex<double>> &x)
{
    // TEST FIXTURE ONLY: independently generate correct wide shares of public
    // known values. A production caller must securely obtain these shares.
    PreparedBlbSlots out{std::vector<WideRingShare>(x.size()),std::vector<WideRingShare>(x.size()),83,13};
    std::uint64_t state=0xabcdef123723ULL;
    const auto mask=(U128{1}<<83)-1;
    for(std::size_t i=0;i<x.size();++i) for(bool imag:{false,true}) {
        const auto a=((U128{Next(state)}<<64)|Next(state))&mask;
        const auto value=static_cast<U128>(static_cast<__int128>(std::llround(
            (imag ? x[i].imag() : x[i].real())*8192)));
        (imag ? out.imag : out.real)[i]=Wide((party==sci::ALICE ? a : value-a)&mask);
    }
    return out;
}
void VerifyC2m(int party,IProtocolTransport &wire,const ComplexMpcSharePair &local,
               const std::vector<std::complex<double>> &x)
{
    std::int64_t max_error=0;
    for(bool imag:{false,true}) {
        const auto &v=(imag ? local.imag : local.real).local_share;
        auto peer=v;
        if(party==sci::BOB) wire.Send(peer.data(),peer.size()*sizeof(peer[0]));
        else {
            wire.Receive(peer.data(),peer.size()*sizeof(peer[0]));
            const auto mask=(std::uint64_t{1}<<43)-1;
            for(std::size_t i=0;i<v.size();++i) {
                auto raw=(v[i]+peer[i])&mask;
                const auto actual=static_cast<std::int64_t>(raw&(std::uint64_t{1}<<42) ? raw|~mask : raw);
                const auto clear=static_cast<std::int64_t>(std::llround((imag ? x[i].imag() : x[i].real())*8192));
                max_error=std::max(max_error,static_cast<std::int64_t>(std::llabs(actual-clear)));
            }
        }
    }
    if(party==sci::ALICE) {
        std::cout<<"BLB_C2M_MAX_ULP="<<max_error<<'\n';
        Check(max_error<=4,"C2M exceeded four MPC ULPs");
    }
}
void VerifyM2c(int party,IProtocolTransport &wire,const HeContext &profile,
    const std::optional<seal::Ciphertext> &encrypted,const std::vector<std::complex<double>> &x,double tolerance)
{
    if(party==sci::ALICE) {
        Check(encrypted.has_value(),"server missing M2C output");
        SendSealObject(wire,*encrypted,seal::compr_mode_type::none);
    } else {
        Check(!encrypted.has_value(),"client unexpectedly got M2C output");
        auto ct=ReceiveSealObject<seal::Ciphertext>(wire,*profile.context);
        seal::Decryptor decryptor(*profile.context,*profile.secret_key);
        seal::CKKSEncoder encoder(*profile.context);
        seal::Plaintext plain; decryptor.decrypt(ct,plain);
        std::vector<std::complex<double>> actual; encoder.decode(plain,actual);
        double error=0;
        for(std::size_t i=0;i<x.size();++i) error=std::max(error,std::abs(actual[i]-x[i]));
        std::cout<<"BLB_M2C_MAX_ERROR="<<error<<std::endl;
        Check(error<tolerance,"M2C independent SEAL oracle exceeded tolerance");
    }
}
// Out-of-domain data are intentional here. Verify execution/format and
// preserved structural rejection, never numerical correctness of this mode.
void TestUncheckedSimulation(int party, const HeContext &profile,
    HeContextRegistry &registry, std::shared_ptr<IProtocolTransport> wire)
{
    const auto role=party==sci::ALICE ? PartyRole::kServer : PartyRole::kClient;
    const auto n=profile.parameters.polynomial_degree, slots=profile.slot_count();
    const auto terminal=profile.context->last_context_data();
    const auto q=terminal->parms().coeff_modulus()[0].value();
    const auto map=PhysicalSlotMap::DenseComplex(n,slots);
    std::optional<SealCipherBatch> input;
    if(party==sci::BOB) {
        const std::vector<std::complex<double>> large(slots,{double(q)/3/std::ldexp(1.,24),0});
        seal::CKKSEncoder encoder(*profile.context);seal::Plaintext plain;
        encoder.encode(large,std::ldexp(1.,24),plain);
        seal::Encryptor encryptor(*profile.context,*profile.secret_key);seal::Ciphertext ct;
        encryptor.encrypt_symmetric(plain,ct);
        auto terminal_ct=ct;seal::Evaluator evaluator(*profile.context);
        evaluator.mod_switch_to_inplace(terminal_ct,terminal->parms_id());
        seal::Decryptor decryptor(*profile.context,*profile.secret_key);
        decryptor.decrypt(terminal_ct,plain);
        seal::util::inverse_ntt_negacyclic_harvey(plain.data(),terminal->small_ntt_tables()[0]);
        bool outside=false;
        for(std::size_t i=0;i<n;++i)outside |= std::min(plain[i],q-plain[i])>=q/4;
        Check(outside,"simulation C2M fixture did not exceed quarter-range");
        SendSealObject(*wire,ct,seal::compr_mode_type::none);
    } else input=SealCipherBatch{profile.profile,
        {ReceiveSealObject<seal::Ciphertext>(*wire,*profile.context)},{{n}},Layout::kSpatial,map};
    ConversionConfig cfg{43,13,40,24,true};
    cfg.benchmark_unchecked_range=true;
    BlbReferenceConverter converter(role,cfg,registry,wire,
        MakeDeterministicRandomSourceForTesting({8,1,9}),"127.0.0.1",23860,2);
    for(const auto bound : {std::optional<int>{},std::optional<int>{62}}) {
        auto shares=Take(converter.CkksToMpc(input,std::nullopt,bound));
        Check(!shares.real.public_value_bound_bits && !shares.imag.public_value_bound_bits,
            "unchecked simulation invented a proved output range");
    }
    // 3*2^40 exceeds quarter of the 43-bit source ring. Its uniform
    // Encode coefficient also exceeds quarter of the 51-bit coefficient ring.
    const std::vector<std::complex<double>> huge(slots,{std::ldexp(3.,27),0});
    auto shares=Narrow(party,huge);
    shares.real.public_value_bound_bits=42;shares.imag.public_value_bound_bits=42;
    MpcToCkksTarget target{profile.profile,0,24,{{n}},Layout::kSpatial,map};
    target.public_value_bound_bits=15; // Intentionally untrusted/inconsistent range metadata.
    const auto check_output=[&](Result<std::optional<SealCipherBatch>> result) {
        auto output=Take(std::move(result));
        if(party==sci::ALICE)Check(output && output->ciphertexts.size()==1 &&
            output->ciphertexts.front().parms_id()==profile.context->first_parms_id(),
            "unchecked benchmark skipped or malformed its ciphertext output");
        else Check(!output,"benchmark client received server-only output");
    };
    check_output(converter.MpcToCkks(shares,target));
    check_output(converter.MpcToCkksPrepared(Prepared(party,huge),target));
    const auto before=converter.stats();
    auto bad=Prepared(party,huge);
    if(party==sci::BOB)bad.ring_bits=43;
    Check(!converter.MpcToCkksPrepared(bad,target).ok(),"simulation ignored invalid prepared width");
    auto bad_target=target;bad_target.public_zero_imaginary=true;
    Check(!converter.MpcToCkks(shares,bad_target).ok(),"simulation ignored an invalid lane layout");
    Check(converter.stats().signed_extension_batches==before.signed_extension_batches,
        "malformed simulation input entered OT");
    check_output(converter.MpcToCkks(shares,target));
    Check(converter.stats().bounded_field_to_ring_elements==2*n &&
        converter.stats().bounded_signed_extension_elements==5*n,
        "simulation skipped actual bounded conversion work");
    if(party==sci::ALICE)std::cout<<"BLB_OUT_OF_RANGE_SIMULATION=PASS\n";
}

void TestReference(int party,std::shared_ptr<IProtocolTransport> wire)
{
    const auto role=party==sci::ALICE ? PartyRole::kServer : PartyRole::kClient;
    HeContextRegistry registry;
    if(role==PartyRole::kClient) {
        Ok(registry.CreateClientProfiles(Config(),false));
        Ok(SendPublicContextBundle(*wire,Take(registry.ExportSerializedPublicBundle(seal::compr_mode_type::none))));
    } else Ok(registry.CreateServerProfiles(Config(),Take(ReceivePublicContextBundle(*wire))));
    for(const auto &name : {"small","large_rns"}) {
        const auto &profile=registry.Require(name,role);
        const auto x=Values(profile.slot_count());
        const auto n=profile.parameters.polynomial_degree;
        const auto map=PhysicalSlotMap::DenseComplex(n,profile.slot_count());
        MpcToCkksTarget target{name,0,24,{{n}},Layout::kSpatial,map};
        std::optional<SealCipherBatch> input;
        if(party==sci::BOB) {
            seal::CKKSEncoder encoder(*profile.context); seal::Plaintext plain;
            encoder.encode(x,std::ldexp(1.,24),plain);
            seal::Encryptor encryptor(*profile.context,*profile.secret_key);
            seal::Ciphertext ct; encryptor.encrypt_symmetric(plain,ct);
            SendSealObject(*wire,ct,seal::compr_mode_type::none);
        } else input=SealCipherBatch{name,{ReceiveSealObject<seal::Ciphertext>(*wire,*profile.context)},{{n}},Layout::kSpatial,map};
        const auto verify=[&](Result<std::optional<SealCipherBatch>> result,double tolerance) {
            auto batch=Take(std::move(result));
            std::optional<seal::Ciphertext> ct;
            if(batch) {Check(batch->ciphertexts.size()==1,"unexpected batch size");ct=batch->ciphertexts.front();}
            VerifyM2c(party,*wire,profile,ct,x,tolerance);
        };
        {
            ConversionConfig cfg{43,13,40,24,true,4096,true,true};
            BlbReferenceConverter converter(role,cfg,registry,wire,
                MakeDeterministicRandomSourceForTesting({17,3,9}),"127.0.0.1",23860,2);
            auto c2m=Take(converter.CkksToMpc(input));
            VerifyC2m(party,*wire,c2m,x);
            const auto before=converter.stats();
            verify(converter.MpcToCkks(Narrow(party,x),target),4./8192);
            const auto raw=converter.stats();
            Check(raw.bounded_signed_extension_elements-before.bounded_signed_extension_elements==2*n,
                "raw M2C hid slot or coefficient extension");
            verify(converter.MpcToCkksPrepared(Prepared(party,x),target),4./8192);
            const auto prepared=converter.stats();
            Check(prepared.bounded_signed_extension_elements-raw.bounded_signed_extension_elements==n &&
                  prepared.slot_extension_input_elements==raw.slot_extension_input_elements,
                "prepared M2C repeated the slot extension");
            verify(converter.MpcToCkks(c2m,target),8./8192);
            for(int bad=0;bad<3;++bad) {
                auto invalid=Prepared(party,x);
                if(party==sci::BOB) {
                    if(bad==0)invalid.ring_bits=43;
                    if(bad==1)invalid.imag.pop_back();
                    if(bad==2)invalid.real[0].high=std::uint64_t{1}<<60;
                }
                const auto before_reject=converter.stats();
                Check(!converter.MpcToCkksPrepared(invalid,target).ok(),"invalid prepared contract accepted");
                Check(converter.stats().signed_extension_batches==before_reject.signed_extension_batches,
                    "invalid prepared contract entered OT");
            }
            // Direction mismatch is rejected by the shared v19 frame before OT.
            if(party==sci::ALICE)Check(!converter.CkksToMpc(input).ok(),"direction mismatch accepted");
            else Check(!converter.MpcToCkks(Narrow(party,x),target).ok(),"direction mismatch accepted");
            verify(converter.MpcToCkksPrepared(Prepared(party,x),target),4./8192);
            Check(converter.stats().transform_cache_hits>0,"engine did not cache local transforms");
        }
        {
            // Without blanket assertions, prove each boundary using metadata.
            ConversionConfig cfg{43,13,40,24,true};
            BlbReferenceConverter converter(role,cfg,registry,wire,
                MakeDeterministicRandomSourceForTesting({5,2,7}),"127.0.0.1",23860,1);
            for(const auto bound : {std::optional<int>{},std::optional<int>{62}}) {
                Check(!converter.CkksToMpc(input,std::nullopt,bound).ok(),"unproved C2M range accepted");
                Check(converter.stats().field_to_ring_batches==0,"unproved C2M range entered OT");
            }
            for(const auto bound : {std::optional<int>{},std::optional<int>{42}}) {
                target.public_value_bound_bits=bound;
                Check(!converter.MpcToCkks(Narrow(party,x),target).ok(),"unproved M2C range accepted");
                Check(!converter.MpcToCkksPrepared(Prepared(party,x),target).ok(),"unproved prepared range accepted");
                Check(converter.stats().signed_extension_batches==0,"unproved M2C range entered OT");
            }
            target.public_value_bound_bits=15;
            VerifyC2m(party,*wire,Take(converter.CkksToMpc(input,std::nullopt,30)),x);
            verify(converter.MpcToCkks(Narrow(party,x),target),4./8192);
            verify(converter.MpcToCkksPrepared(Prepared(party,x),target),4./8192);
            target.public_value_bound_bits.reset();
        }
        TestUncheckedSimulation(party,profile,registry,wire);
    }
}
void TestPlan()
{
    const auto plan=Take(blb::PlanM2cEncoding(4096,43,13,46,40,20,false));
    Check(plan.slot_work_bits==83 && plan.guard_bits==20 &&
        plan.coefficient_ring_bits==51 && plan.coefficient_scale_bits==21 &&
        plan.coefficient_work_bits==91 && plan.rns_work_bits==116,"bit/scale ledger changed");
    auto transform=Take(conversion_internal::FixedCkksTransform::Create(4096,20,103));
    blb::PreparedSlotShares narrow{{std::vector<WideRingShare>(2048),std::vector<WideRingShare>(2048)},43,13};
    Check(!blb::EncodePreparedSlots(narrow,plan,transform,false).ok(),"local Encode accepted narrow shares");
    narrow.ring_bits=83;
    auto wrong_roots=Take(conversion_internal::FixedCkksTransform::Create(4096,19,103));
    Check(!blb::EncodePreparedSlots(narrow,plan,wrong_roots,false).ok(),"local Encode accepted wrong root precision");
    Check(!blb::PlanM2cEncoding(4096,63,13,60,63,24,false).ok(),"unsupported workspace accepted");
}
void Run(int party)
{
    auto wire=MakeNetIoTransport(party==sci::ALICE ? PartyRole::kServer : PartyRole::kClient,"127.0.0.1",23850);
    {
        sci::IOPack io(party,23660,"127.0.0.1"); sci::OTPack ot(&io,party);
        TestRing(party,ot,*wire);
    }
    TestReference(party,wire);
}
}
int main()
{
    pid_t child=-1;
    try {
        TestPlan();
        child=fork();Check(child>=0,"fork failed");
        if(child==0) {
            try {Run(sci::BOB); std::cout.flush();_exit(0);}
            catch(const std::exception &e){std::cerr<<"BLB client: "<<e.what()<<'\n';_exit(2);}
        }
        Run(sci::ALICE);
        int status{}; waitpid(child,&status,0);child=-1;
        Check(WIFEXITED(status)&&WEXITSTATUS(status)==0,"BLB reference peer failed");
        std::cout<<"HERA_BLB_REFERENCE_STATUS=PASS\n";
    }catch(const std::exception &e) {
        if(child>0){kill(child,SIGTERM);waitpid(child,nullptr,0);}
        std::cerr<<e.what()<<'\n';return 1;
    }
}
