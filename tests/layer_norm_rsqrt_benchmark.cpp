#include "hera/attention_ln2.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

#ifndef HERA_HIGH_PRECISION_RSQRT
#define HERA_HIGH_PRECISION_RSQRT 1
#endif

namespace {
using Clock = std::chrono::steady_clock;
void Check(bool ok, const std::string &error) { if (!ok) throw std::runtime_error(error); }
void Transfer(int fd, void *data, std::size_t bytes, bool write_data)
{
    auto *cursor = static_cast<char *>(data);
    while (bytes) {
        const auto n = write_data ? write(fd, cursor, bytes) : read(fd, cursor, bytes);
        Check(n > 0, "benchmark pipe failed"); cursor += n; bytes -= n;
    }
}
struct Options { int tokens{128}, degree{16384}, iterations{3}, warmup{1}, port{24700}; std::string fixture{"normal"}; };
struct Packet { int ok{}; double conversion_ms{}, rsqrt_ms{}; std::uint64_t bytes{}, rounds{}, share[4096]{}, variance[4096]{}; };
hera::HeraConfig Config(const Options &o)
{
    hera::HeraConfig c;
    hera::TransformerLayerConfig l;
    l.id="bench"; l.tokens=o.tokens; l.model_width=1024; l.ffn_width=4096;
    l.q_heads=l.kv_heads=16; l.qk_head_width=l.value_head_width=64;
    c.layers={l}; c.mpc_numeric={43,13,true}; c.conversion={40,20,true};
    hera::HeParameterConfig p;
    p.polynomial_degree=o.degree; p.coefficient_modulus_bits={60,60,46,46,46,46,60};
    p.input_scale_bits=p.default_scale_bits=45; p.output_scale_bits=46;
    p.declared_depth=6; p.terminal_prime_count=1; p.security_level=hera::SecurityLevel::kTc128;
    c.he_parameter_profiles.emplace("bench",p);
    hera::LayerExecutionConfig e; e.layer_id=l.id;
    for (auto r : {hera::HeRegionId::kRegion1,hera::HeRegionId::kRegion2,hera::HeRegionId::kRegion3,hera::HeRegionId::kRegion4,hera::HeRegionId::kRegion5})
        e.regions.emplace(r,hera::HeRegionConfig{r,"bench",hera::Layout::kSpatial,hera::Layout::kSpatial});
    c.layer_execution={e}; return c;
}
std::vector<double> Values(const Options &o)
{
    std::vector<double> v(o.tokens);
    const double edges[]={1e-5,1.1e-5,2e-5,1e-4,0.25,1.0,16.0,63.0};
    for (int i=0;i<o.tokens;++i) v[i]=o.fixture=="normal" ? 0.25+0.25*(i%16) :
        o.fixture=="quantum" ? std::ldexp(1.,i%5-24) : edges[i%8];
    return v;
}
Packet Run(hera::PartyRole role, hera::HeContextRegistry &registry,
           const Options &o, const std::optional<hera::SealCipherBatch> &input,
           int pipe_fd)
{
    const bool server=role==hera::PartyRole::kServer;
    auto transport=hera::MakeNetIoTransport(role,"127.0.0.1",o.port);
    hera::CkksMpcConverter converter(role,{43,13,40,20,true},registry,transport,
        hera::MakeDeterministicRandomSourceForTesting(server?std::vector<std::uint8_t>{1,2,3}:std::vector<std::uint8_t>{4,5,6}),
        "127.0.0.1",o.port+20,1);
    hera::MpcLayerNormRuntime runtime(role,"127.0.0.1",o.port+100);
    Packet total;
    const auto expected=Values(o);
    double worst=0;
    for (int run=0;run<o.warmup+o.iterations;++run) {
        const auto before=converter.stats();
        auto physical=converter.CkksToMpc(input
#if HERA_HIGH_PRECISION_RSQRT
            ,24
#endif
        );
        Check(physical.ok(),physical.status().message());
        std::vector<std::uint64_t> variance_values(o.tokens);
        for (int token = 0; token < o.tokens; ++token)
            variance_values[token] = token < (o.tokens + 1) / 2
                ? physical.value().real.local_share[token] : physical.value().imag.local_share[token];
        hera::MpcShareTensor variance{std::move(variance_values),
            {{static_cast<std::size_t>(o.tokens),1}},physical.value().real.numeric,hera::ComplexComponent::kReal};
        const bool invalid = o.fixture == "invalid-zero" || o.fixture == "invalid-range" || o.fixture == "invalid-negative";
        if (invalid) variance.local_share.front() = server && o.fixture == "invalid-range"
            ? std::uint64_t{64} << variance.numeric.scale_bits :
                server && o.fixture == "invalid-negative" ? (std::uint64_t{1} << 43) - 1 : 0;
        if (o.fixture=="quantum")
            for (int i=0;i<o.tokens;++i) variance.local_share[i]=server ? std::uint64_t{1}<<(i%5) : 0;
        hera::AttentionLn2ExecutionReport report;
        const auto started=Clock::now();
        auto inverse=runtime.ReciprocalSqrt(variance,20,&report
#if HERA_HIGH_PRECISION_RSQRT
            ,13
#endif
        );
        if (invalid) {
            Check(!inverse.ok(), "invalid variance was accepted");
            Check(inverse.status().message().find("positive and fit") != std::string::npos,
                "invalid variance failed for an unrelated reason");
            if (server) std::cout << "CHECK=PASS\nINVALID_VARIANCE_REJECTED=" << o.fixture << '\n';
            total.ok=1; return total;
        }
        Check(inverse.ok(),inverse.status().message());
        Packet packet; packet.ok=1;
        packet.conversion_ms=converter.stats().milliseconds-before.milliseconds;
        packet.rsqrt_ms=std::chrono::duration<double,std::milli>(Clock::now()-started).count();
        packet.bytes=converter.stats().communication_bytes-before.communication_bytes+report.mpc_communication_bytes;
        packet.rounds=converter.stats().rounds-before.rounds+report.mpc_rounds;
        std::copy(inverse.value().local_share.begin(),inverse.value().local_share.end(),packet.share);
        std::copy(variance.local_share.begin(),variance.local_share.end(),packet.variance);
        if (!server) Transfer(pipe_fd,&packet,sizeof(packet),true);
        else {
            Packet peer; Transfer(pipe_fd,&peer,sizeof(peer),false); Check(peer.ok,"client failed");
            const auto mask=(std::uint64_t{1}<<43)-1;
            for (int i=0;i<o.tokens;++i) {
                const auto actual=std::ldexp(static_cast<double>((packet.share[i]+peer.share[i])&mask),-13);
                const auto exact=1/std::sqrt(expected[i]);
                worst=std::max(worst,std::abs(actual-exact)/exact);
                if (run==o.warmup+o.iterations-1 && i<std::min(8,o.tokens)) {
                    const auto raw=(packet.variance[i]+peer.variance[i])&mask;
                    const auto signed_value=raw>=(std::uint64_t{1}<<42) ? static_cast<std::int64_t>(raw)-(std::int64_t{1}<<43) : static_cast<std::int64_t>(raw);
                    std::cout<<std::setprecision(10)<<"VALUE="<<i<<",VAR_EXPECTED="<<expected[i]
                        <<",VAR_ACTUAL="<<std::ldexp(static_cast<double>(signed_value),-(HERA_HIGH_PRECISION_RSQRT?24:13))
                        <<",RSQRT_ACTUAL="<<actual<<'\n';
                }
            }
            if (run>=o.warmup) {
                total.conversion_ms+=std::max(packet.conversion_ms,peer.conversion_ms);
                total.rsqrt_ms+=std::max(packet.rsqrt_ms,peer.rsqrt_ms);
                total.bytes+=packet.bytes+peer.bytes; total.rounds+=std::max(packet.rounds,peer.rounds);
            }
        }
    }
    if (server) {
        std::cout<<std::setprecision(10)<<"TOKENS="<<o.tokens<<"\nDEGREE="<<o.degree<<"\nFIXTURE="<<o.fixture
            <<"\nVARIANCE_SCALE_BITS="<<(HERA_HIGH_PRECISION_RSQRT?24:13)
            <<"\nCONVERSION_MS="<<total.conversion_ms/o.iterations<<"\nRSQRT_MS="<<total.rsqrt_ms/o.iterations
            <<"\nCOMMUNICATION_BYTES="<<total.bytes/o.iterations<<"\nROUNDS="<<total.rounds/o.iterations
            <<"\nMAX_RELATIVE_ERROR="<<worst<<"\nCHECK="<<(worst<0.02?"PASS":"FAIL")<<'\n';
        total.ok=worst<0.02;
    }
    return total;
}
}
int main(int argc,char **argv)
{
    try {
        Options o;
        for (int i=1;i<argc;++i) {
            const std::string arg(argv[i]); const auto eq=arg.find('='); Check(eq!=std::string::npos,"expected --name=value");
            const auto key=arg.substr(0,eq), val=arg.substr(eq+1);
            if (key=="--fixture") o.fixture=val;
            else if (key=="--tokens") o.tokens=std::stoi(val);
            else if (key=="--degree") o.degree=std::stoi(val);
            else if (key=="--iterations") o.iterations=std::stoi(val);
            else if (key=="--warmup") o.warmup=std::stoi(val);
            else if (key=="--port") o.port=std::stoi(val);
            else throw std::runtime_error("unknown option");
        }
        Check(o.tokens>0&&o.tokens<=4096&&o.iterations>0&&o.warmup>=0,"invalid size");
        auto config=Config(o); hera::HeContextRegistry client,server;
        Check(client.CreateClientProfiles(config).ok(),"client profiles failed");
        if (o.fixture == "statistics") {
            std::size_t stride=1; while (stride<static_cast<std::size_t>(o.tokens)) stride*=2;
            std::vector<int> steps;
            for (std::size_t step=1;step<static_cast<std::size_t>(o.degree/2)/stride;step*=2)
                steps.push_back(static_cast<int>(step*stride));
            Check(client.CompileRotationKeys("bench",steps).ok(),"statistics rotation keys failed");
        }
        auto bundle=client.ExportPublicBundle(); Check(bundle.ok(),bundle.status().message());
        Check(server.CreateServerProfiles(config,bundle.value()).ok(),"server profiles failed");
        const auto &context=client.Require("bench",hera::PartyRole::kClient);
        hera::PhysicalSlotMap map; map.ciphertext_count=1; map.slots_per_ciphertext=context.slot_count(); map.logical_element_count=o.tokens;
        map.assignments.resize(context.slot_count());
        const auto real_values=Values(o);
        std::vector<std::complex<double>> values(context.slot_count());
        for (int i=0;i<o.tokens;++i) {
            if (i < (o.tokens + 1) / 2) {
                map.assignments[i].real_logical_index=i;
                values[i]={real_values[i],0.};
            } else {
                map.assignments[i].imaginary_logical_index=i;
                values[i]={0.,real_values[i]};
            }
        }
        seal::CKKSEncoder encoder(*context.context); seal::Plaintext plain;
        encoder.encode(values,context.context->last_parms_id(),std::ldexp(1.,46),plain);
        seal::Encryptor encryptor(*context.context,*context.secret_key); seal::Ciphertext ciphertext; encryptor.encrypt_symmetric(plain,ciphertext);
        hera::SealCipherBatch batch{"bench",{std::move(ciphertext)},{{static_cast<std::size_t>(o.tokens),1}},hera::Layout::kReduction,map};
        if (o.fixture == "statistics") {
            const auto &layer=config.layers.front();
            auto packing=hera::AttentionLn2PackingPlanner::RealStrictColumns(
                {{layer.tokens,layer.model_width}},context.slot_count());
            Check(packing.ok(),packing.status().message());
            std::vector<std::vector<double>> planes(packing.value().ciphertext_count,
                std::vector<double>(context.slot_count()));
            const auto expected=Values(o);
            for (std::size_t token=0;token<layer.tokens;++token) {
                const auto amplitude=std::sqrt(std::max(0.,expected[token]-layer.attention_norm_epsilon));
                for (std::size_t f=0;f<layer.model_width;++f) {
                    const auto &loc=packing.value().Locations(token*layer.model_width+f).front();
                    planes[loc.ciphertext][loc.slot]=f%2 ? amplitude : -amplitude;
                }
            }
            auto data=context.context->first_context_data();
            for (int i=0;i<3;++i) data=data->next_context_data();
            hera::SealCipherTensor input; input.parameter_profile="bench";
            input.physical_packing=packing.take_value(); input.logical_spec.shape={{layer.tokens,layer.model_width}};
            input.logical_spec.domain=hera::Domain::kHeCiphertext; input.logical_spec.layout=hera::Layout::kStrictColumn;
            input.logical_spec.numeric={43,13,true}; input.logical_spec.he_parameter_profile="bench";
            input.logical_spec.he_region=hera::HeRegionId::kRegion1; input.logical_spec.planned_level=3;
            input.logical_spec.planned_scale_bits=46; input.ciphertexts.resize(planes.size());
            for (std::size_t i=0;i<planes.size();++i) {
                encoder.encode(planes[i],data->parms_id(),std::ldexp(1.,46),plain);
                encryptor.encrypt_symmetric(plain,input.ciphertexts[i]);
            }
            auto statistics=hera::LayerNormStatisticsKernel::Evaluate(input,layer,
                server.Require("bench",hera::PartyRole::kServer),nullptr,2,46);
            Check(statistics.ok(),statistics.status().message());
            batch.ciphertexts=std::move(statistics.value().variance.ciphertexts);
        }
        int pipes[2]; Check(pipe(pipes)==0,"pipe failed"); const auto pid=fork(); Check(pid>=0,"fork failed");
        if (pid==0) {
            close(pipes[0]); try { Run(hera::PartyRole::kClient,client,o,std::nullopt,pipes[1]); _exit(0); }
            catch (const std::exception &e) { std::cerr<<"CLIENT_FAIL="<<e.what()<<'\n'; Packet p; Transfer(pipes[1],&p,sizeof(p),true); _exit(2); }
        }
        close(pipes[1]); const auto result=Run(hera::PartyRole::kServer,server,o,batch,pipes[0]);
        int status=0; waitpid(pid,&status,0); Check(WIFEXITED(status)&&WEXITSTATUS(status)==0,"client exited unsuccessfully");
        return result.ok?0:1;
    } catch (const std::exception &e) { std::cerr<<"RSQRT_BENCH_FAIL="<<e.what()<<'\n'; return 2; }
}
