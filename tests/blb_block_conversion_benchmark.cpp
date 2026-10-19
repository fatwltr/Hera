// Conversion workload replay, not an execution of BLB's simulated BERT model.
// Built with the native BLB bridge adapter by benchmark_blb_block_conversion.py.
#include "bridge_adapter.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <complex>
#include <iomanip>
#include <iostream>
#include <memory>

namespace {
using Clock = std::chrono::steady_clock;
constexpr int kRing = 40, kScale = 20;
constexpr uint64_t kMask = (uint64_t{1} << kRing) - 1;
uint64_t Mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
void Exchange(sci::NetIO &wire, int party, const void *local, void *peer, size_t n) {
    if (party == 1) {
        wire.send_data(local, n); wire.flush(); wire.recv_data(peer, n);
    } else {
        wire.recv_data(peer, n); wire.send_data(local, n); wire.flush();
    }
}
uint64_t Comm(NonLinear &nl) {
    uint64_t n = 0;
    for (int i = 0; i < MAX_THREADS; ++i) n += nl.iopackArr[i]->get_comm();
    return n; // Native SCI counters count SENT bytes only.
}
struct Call {
    const char *name;
    int profile, ciphertexts;
    bool m2c, complex_lanes, supplemental;
    int source_scale; // C2M fixture precision; M2C uses the profile input scale.
    size_t active;
    std::vector<std::vector<uint64_t>> slots;
    std::vector<seal::Plaintext> fields;
    std::vector<std::vector<uint64_t>> result_slots;
    std::vector<seal::Plaintext> result_fields;
    std::vector<std::vector<std::complex<double>>> field_expected;
};
struct Sample {
    double milliseconds{};
    uint64_t bytes{};
    std::array<double, 6> stages{};
};
std::vector<Call> Plan(bool dual) {
    // Native block shapes: T=128, D=768, H=12, I=3072.
    // LN boundary ownership follows transformer_block_test, including its
    // final LN statistics. Native half=true C2M paths already use both lanes.
    std::vector<Call> calls{
        {"ln1_center",0,6,true,false,false,44,98304},
        {"ln1_inverse_broadcast",0,6,true,false,false,44,98304},
        {"qk_and_v",0,9,false,true,false,44,294912},
        {"softmax_shift",1,12,true,false,false,40,196608},
        {"softmax_exp",1,12,false,false,false,40,196608},
        {"softmax_probability",2,12,true,false,false,43,196608},
        {"ln2_center",2,3,false,true,false,35,98304},
        {"ln2_variance",2,1,false,false,false,36,128},
        {"ln2_center_to_ffn",3,6,true,false,false,46,98304},
        {"ln2_inverse_broadcast",3,6,true,false,false,46,98304},
        {"gelu_polynomial",3,12,false,true,false,30,393216},
        {"gelu_to_wd",4,48,true,false,false,38,393216},
        {"next_ln_center",4,6,false,true,false,20,98304},
        {"next_ln_variance",4,1,false,false,false,64,128},
        // Explicit alternative assumption: each missing branch is converted
        // separately. These are measured supplements, not a repaired model.
        {"supplement_v_to_av",2,6,true,false,true,43,98304},
        {"supplement_attention_residual",2,6,true,false,true,43,98304},
        {"supplement_ffn_residual",4,12,true,false,true,38,98304},
    };
    if (dual) for (auto &c : calls) {
        // Pack at the conversion boundary only. Never apply nonlinear HE
        // polynomials to a+ib as if their two components were independent.
        if (!c.complex_lanes && c.ciphertexts % 2 == 0) {
            c.ciphertexts /= 2;
            c.complex_lanes = true;
        }
    }
    return calls;
}
int64_t Expected(size_t i) { return (int64_t(i % 15) - 7) * 4096; }
void Prepare(Call &c, HE_CKKS &he, int party, size_t call_index) {
    const size_t n = he.poly_modulus_degree, slots = he.num_slots;
    const size_t lanes = c.complex_lanes ? n : slots;
    c.slots.resize(c.ciphertexts, std::vector<uint64_t>(lanes));
    for (int ct = 0; ct < c.ciphertexts; ++ct)
        for (size_t j = 0; j < lanes; ++j) {
            const size_t index = size_t(ct) * lanes + j;
            const int64_t value = index < c.active ? Expected(index) : 0;
            const auto a = Mix(index + call_index * 1000003 + 911) & kMask;
            c.slots[ct][j] = party == 1 ? a : (uint64_t(value) - a) & kMask;
        }
    if (c.m2c) return;
    const auto terminal = he.context->last_context_data();
    const auto q = terminal->parms().coeff_modulus()[0].value();
    for (int ct = 0; ct < c.ciphertexts; ++ct) {
        std::vector<std::complex<double>> values(slots);
        for (size_t j = 0; j < slots; ++j) {
            const auto real_index = size_t(ct) * lanes + j;
            const auto imag_index = real_index + slots;
            values[j] = {real_index < c.active ? Expected(real_index) / std::ldexp(1., kScale) : 0.,
                c.complex_lanes && imag_index < c.active ? Expected(imag_index) / std::ldexp(1., kScale) : 0.};
        }
        // SEAL's encoder rejects scale>=q bits. For the artifact's scale64
        // variance boundary, encode at scale50, then lift exact integer
        // coefficients modulo q. Small inputs keep the TRUE coefficients
        // within quarter-q; this is checked before producing random shares.
        const int encoding_scale = std::min(c.source_scale, 50);
        seal::Plaintext plain;
        he.encoder->encode(values, terminal->parms_id(), std::ldexp(1., encoding_scale), plain);
        seal::util::inverse_ntt_negacyclic_harvey(plain.data(), terminal->small_ntt_tables()[0]);
        for (size_t j = 0; j < n; ++j) {
            const auto lift = c.source_scale - encoding_scale;
            plain[j] = uint64_t((static_cast<unsigned __int128>(plain[j]) << lift) % q);
            if (std::min(plain[j], q - plain[j]) >= q / 4)
                throw std::runtime_error("C2M fixture violates public quarter range");
        }
        seal::util::ntt_negacyclic_harvey(plain.data(), terminal->small_ntt_tables()[0]);
        plain.scale() = std::ldexp(1., c.source_scale);
        std::vector<std::complex<double>> quantized;
        auto oracle_plain = plain;
        oracle_plain.scale() = std::ldexp(1., encoding_scale);
        he.encoder->decode(oracle_plain, quantized);
        for (auto &value : quantized) value /= std::ldexp(1., c.source_scale - encoding_scale);
        c.field_expected.push_back(std::move(quantized));
        seal::util::inverse_ntt_negacyclic_harvey(plain.data(), terminal->small_ntt_tables()[0]);
        for (size_t j = 0; j < n; ++j) {
            const auto a = Mix(j + size_t(ct) * n + call_index * 1000003 + 1211) % q;
            plain[j] = party == 1 ? a : (plain[j] + q - a) % q;
        }
        seal::util::ntt_negacyclic_harvey(plain.data(), terminal->small_ntt_tables()[0]);
        plain.scale() = std::ldexp(1., c.source_scale);
        c.fields.push_back(std::move(plain));
    }
}
void Convert(Call &c, NativeBridge &bridge, HE_CKKS &he) {
    if (c.m2c)
        c.result_fields = bridge.mpc_to_ckks(c.slots, he.num_slots, he.get_Q(), kRing,
            kScale, he.he_scale, &he, c.complex_lanes);
    else
        c.result_slots = bridge.ckks_to_mpc(c.fields, he.poly_modulus_degree,
            he.get_Q(c.fields[0]), kRing, kScale, &he, c.complex_lanes);
}
double Check(Call &c, HE_CKKS &he, sci::NetIO &wire, int party) {
    const size_t n = he.poly_modulus_degree, slots = he.num_slots;
    const size_t lanes = c.complex_lanes ? n : slots;
    double max_error = 0;
    for (int ct = 0; ct < c.ciphertexts; ++ct) {
        if (c.m2c) {
            auto plain = c.result_fields[ct];
            std::vector<uint64_t> peer(plain.coeff_count());
            Exchange(wire, party, plain.data(), peer.data(), peer.size() * 8);
            const auto data = he.context->get_context_data(plain.parms_id());
            const auto &moduli = data->parms().coeff_modulus();
            for (size_t p = 0; p < moduli.size(); ++p)
                for (size_t j = 0; j < n; ++j) {
                    const auto index = p * n + j;
                    plain[index] = (plain[index] + peer[index]) % moduli[p].value();
                }
            std::vector<std::complex<double>> decoded;
            he.encoder->decode(plain, decoded);
            for (size_t j = 0; j < lanes; ++j) {
                const size_t index = size_t(ct) * lanes + j;
                const double expected = index < c.active ? Expected(index) / std::ldexp(1., kScale) : 0.;
                const double actual = j < slots ? decoded[j].real() : decoded[j - slots].imag();
                if (!std::isfinite(actual)) throw std::runtime_error("nonfinite M2C result");
                max_error = std::max(max_error, std::abs(actual - expected));
            }
        } else {
            const auto &local = c.result_slots[ct];
            std::vector<uint64_t> peer(lanes);
            Exchange(wire, party, local.data(), peer.data(), lanes * 8);
            for (size_t j = 0; j < lanes; ++j) {
                const size_t index = size_t(ct) * lanes + j;
                const auto raw = (local[j] + peer[j]) & kMask;
                const auto value = raw & (uint64_t{1} << (kRing - 1))
                    ? -int64_t((uint64_t{1} << kRing) - raw) : int64_t(raw);
                // Compare against independent SEAL decoding of the actual
                // quantized plaintext, isolating conversion error from the
                // artifact's low-precision HE boundary (notably scale20).
                const double expected = j < slots ? c.field_expected[ct][j].real()
                    : c.field_expected[ct][j - slots].imag();
                max_error = std::max(max_error, std::abs(value / std::ldexp(1., kScale) - expected));
            }
        }
    }
    // The artifact's scale20 boundary is as coarse as the MPC output itself.
    // Its root20 fixed FFT adds a few ULP relative to SEAL's floating decode;
    // use the existing bridge roundtrip envelope (8 ULP) for this boundary.
    // All higher-precision C2M and every M2C retain the 4 ULP gate.
    const int tolerance = !c.m2c && c.source_scale == kScale ? 8 : 4;
    if (max_error > tolerance / std::ldexp(1., kScale))
        throw std::runtime_error(std::string("ULP oracle failed: ") + c.name + " error=" + std::to_string(max_error));
    return max_error;
}
} // namespace

int main(int argc, char **argv) {
    try {
        if (argc != 6) throw std::invalid_argument("party port dual warmup repeats required");
        const int party = std::stoi(argv[1]), port = std::stoi(argv[2]);
        const bool dual = std::stoi(argv[3]);
        const int warmup = std::stoi(argv[4]), repeats = std::stoi(argv[5]);
        if ((party != 1 && party != 2) || warmup < 0 || repeats < 1) throw std::invalid_argument("invalid run controls");
        sci::NetIO wire(party == 1 ? nullptr : "127.0.0.1", port);
        NonLinear nonlinear(party, "127.0.0.1", port + 1);
        if (!nonlinear.correct_conversion) throw std::runtime_error("correct conversion mode required");
        std::vector<std::unique_ptr<HE_CKKS>> profiles;
        profiles.emplace_back(new HE_CKKS(party,nullptr,32768,{60,34,60,60,60,60,60},20,44));
        profiles.emplace_back(new HE_CKKS(party,nullptr,32768,{60,40,40,40,40,40,40,40,60},20,40));
        profiles.emplace_back(new HE_CKKS(party,nullptr,32768,{60,60,60,60,60,60,60},20,43));
        profiles.emplace_back(new HE_CKKS(party,nullptr,32768,{60,32,60,60,60,60,60,60},20,46));
        profiles.emplace_back(new HE_CKKS(party,nullptr,16384,{60,34,60,60,60,60},20,38));
        auto calls = Plan(dual);
        for (size_t i = 0; i < calls.size(); ++i) Prepare(calls[i], *profiles[calls[i].profile], party, i);
        NativeBridge bridge{nonlinear};
        std::vector<Sample> sums(calls.size());
        for (int run = 0; run < warmup + repeats; ++run) {
            double block_ms = 0, supplement_ms = 0;
            uint64_t block_bytes = 0, supplement_bytes = 0;
            for (size_t i = 0; i < calls.size(); ++i) {
                auto &c = calls[i];
                wire.sync();
                const auto bytes_before = Comm(nonlinear);
                const auto stages_before = nonlinear.conversion_stage_ms;
                const auto started = Clock::now();
                Convert(c, bridge, *profiles[c.profile]);
                Sample local, peer;
                local.milliseconds = std::chrono::duration<double, std::milli>(Clock::now() - started).count();
                local.bytes = Comm(nonlinear) - bytes_before;
                for (size_t j = 0; j < local.stages.size(); ++j)
                    local.stages[j] = nonlinear.conversion_stage_ms[j] - stages_before[j];
                Exchange(wire, party, &local, &peer, sizeof(local));
                const double elapsed = std::max(local.milliseconds, peer.milliseconds);
                const auto bytes = local.bytes + peer.bytes;
                (c.supplemental ? supplement_ms : block_ms) += elapsed;
                (c.supplemental ? supplement_bytes : block_bytes) += bytes;
                if (run >= warmup) {
                    sums[i].milliseconds += elapsed; sums[i].bytes += bytes;
                    for (size_t j = 0; j < local.stages.size(); ++j) sums[i].stages[j] += local.stages[j];
                }
            }
            if (party == 1) std::cout << std::setprecision(12)
                << "BLB_BLOCK_SAMPLE,index=" << run << ",warmup=" << (run < warmup)
                << ",base_ms=" << block_ms << ",base_bytes=" << block_bytes
                << ",supplement_ms=" << supplement_ms << ",supplement_bytes=" << supplement_bytes << std::endl;
        }
        for (size_t i = 0; i < calls.size(); ++i) {
            auto &c = calls[i]; const auto &he = *profiles[c.profile];
            const double error = Check(c, *profiles[c.profile], wire, party);
            if (party == 1) {
                std::cout << std::setprecision(12) << "BLB_BLOCK_CALL,index=" << i << ",name=" << c.name
                    << ",direction=" << (c.m2c ? "M2C" : "C2M") << ",profile=" << c.profile
                    << ",degree=" << he.poly_modulus_degree << ",ciphertexts=" << c.ciphertexts
                    << ",complex=" << c.complex_lanes << ",active=" << c.active
                    << ",capacity=" << size_t(c.ciphertexts) * (c.complex_lanes ? he.poly_modulus_degree : he.num_slots)
                    << ",scale=" << (c.m2c ? he.he_scale : c.source_scale)
                    << ",supplemental=" << c.supplemental << ",mean_ms=" << sums[i].milliseconds / repeats
                    << ",both_bytes=" << sums[i].bytes / repeats << ",max_error_ulp=" << error * std::ldexp(1.,kScale)
                    << ",threshold_ulp=" << (!c.m2c && c.source_scale == kScale ? 8 : 4)
                    << ",F2R_ms=" << sums[i].stages[0] / repeats
                    << ",slot_extend_ms=" << sums[i].stages[1] / repeats
                    << ",coefficient_extend_ms=" << sums[i].stages[2] / repeats
                    << ",decode_ms=" << sums[i].stages[3] / repeats
                    << ",encode_ms=" << sums[i].stages[4] / repeats
                    << ",rns_ms=" << sums[i].stages[5] / repeats << ",status=PASS" << std::endl;
            }
        }
        std::cout << "BLB_BLOCK_REPLAY=PASS,model_inference=NOT_EXECUTED" << std::endl;
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "BLB_BLOCK_REPLAY=FAIL,error=" << e.what() << std::endl;
        return 1;
    }
}
