#include "hera/attention_ln2.h"
#include "hera/detail/parallel_ranges.h"
#include "hera/detail/packing_utils.h"
#include "hera/ffn_gelu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    std::string kernel;
    std::size_t tokens{2};
    std::size_t hidden{4};
    std::size_t heads{2};
    std::size_t kv_heads{1};
    std::size_t qk_width{2};
    std::size_t value_width{2};
    std::size_t ffn_width{5};
    std::size_t degree{8192};
    std::size_t warmup{0};
    std::size_t iterations{1};
    std::size_t he_threads{1};
    bool with_layer_norm{};
    bool shared_r15{};
    bool compact_r15{};
    bool virtual_weights{};
    bool hoisting{true};
    std::size_t baby_steps{};
    std::size_t plaintext_cache_mib{256};
};

void Require(bool ok, const std::string &message)
{
    if (!ok) throw std::runtime_error(message);
}

Options Parse(int argc, char **argv)
{
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        const auto split = arg.find('=');
        Require(split != std::string::npos, "expected --name=value: " + arg);
        const auto name = arg.substr(0, split);
        const auto value = arg.substr(split + 1);
        if (name == "--kernel") { options.kernel = value; continue; }
        const auto number = std::stoull(value);
        if (name == "--tokens") options.tokens = number;
        else if (name == "--hidden") options.hidden = number;
        else if (name == "--heads") options.heads = number;
        else if (name == "--kv-heads") options.kv_heads = number;
        else if (name == "--qk-width") options.qk_width = number;
        else if (name == "--value-width") options.value_width = number;
        else if (name == "--ffn-width") options.ffn_width = number;
        else if (name == "--degree") options.degree = number;
        else if (name == "--warmup") options.warmup = number;
        else if (name == "--iterations") options.iterations = number;
        else if (name == "--he-threads") options.he_threads = number;
        else if (name == "--compact-r15") options.compact_r15 = number != 0;
        else if (name == "--shared-r15") options.shared_r15 = number != 0;
        else if (name == "--plaintext-cache-mib") options.plaintext_cache_mib = number;
        else if (name == "--projection-baby-steps") options.baby_steps = number;
        else if (name == "--hoisting") options.hoisting = number != 0;
        else if (name == "--virtual-weights") options.virtual_weights = number != 0;
        else if (name == "--with-layer-norm") options.with_layer_norm = number != 0;
        else throw std::invalid_argument("unknown option: " + name);
    }
    Require(options.kernel == "qkv" || options.kernel == "wo" ||
                options.kernel == "wu" || options.kernel == "wd" || options.kernel == "ffn-he" ||
                options.kernel == "softmax-exp" ||
                options.kernel == "gelu-poly", "unsupported --kernel");
    Require(options.tokens && options.hidden && options.heads &&
                options.kv_heads && options.qk_width && options.value_width &&
                options.ffn_width && options.iterations && options.he_threads,
            "all dimensions, iterations, and thread counts must be positive");
    Require(options.heads % options.kv_heads == 0,
            "heads must be divisible by kv-heads");
    Require(!options.with_layer_norm || options.kernel == "wo" || (options.kernel == "wd" && options.shared_r15),
            "--with-layer-norm requires Wo or shared-R1/5 Wd");
    Require(!options.compact_r15 || options.shared_r15, "--compact-r15 requires --shared-r15");
    Require(!options.shared_r15 || options.kernel == "wd" || options.kernel == "ffn-he",
            "--shared-r15 requires Wd or ffn-he");
    Require(options.degree >= 4096 &&
                (options.degree & (options.degree - 1)) == 0,
            "degree must be a power of two >= 4096");
    return options;
}

hera::HeraConfig MakeConfig(const Options &o)
{
    hera::HeraConfig config;
    hera::TransformerLayerConfig layer;
    layer.id = "bench";
    layer.tokens = o.tokens;
    layer.model_width = o.hidden;
    layer.q_heads = o.heads;
    layer.kv_heads = o.kv_heads;
    layer.qk_head_width = o.qk_width;
    layer.value_head_width = o.value_width;
    layer.ffn_width = o.ffn_width;
    layer.norm = hera::NormPlacement::kPreNorm;
    layer.activation = hera::ActivationKind::kGelu;
    config.layers = {layer};
    hera::HeParameterConfig profile;
    profile.polynomial_degree = o.degree;
    profile.terminal_prime_count = 1;
    profile.security_level = hera::SecurityLevel::kNone;
    if (o.kernel == "softmax-exp") {
        profile.coefficient_modulus_bits = {60, 40, 40, 40,
                                            40, 40, 40, 60};
        profile.input_scale_bits = profile.default_scale_bits =
            profile.output_scale_bits = 40;
    } else if (o.kernel == "gelu-poly" || o.kernel == "wu" || o.kernel == "ffn-he" || o.shared_r15) {
        profile.coefficient_modulus_bits = {60, 60, 46, 46, 46, 60, 60};
        profile.input_scale_bits = profile.default_scale_bits = 45;
        profile.output_scale_bits = 46;
    } else if (o.kernel == "qkv" || o.kernel == "wo" || o.kernel == "wd") {
        profile.coefficient_modulus_bits = {60, 35, 60, 60, 60, 60, 60};
        profile.input_scale_bits = profile.default_scale_bits =
            profile.output_scale_bits = 44;
    } else {
        profile.coefficient_modulus_bits = {60, 35, 60, 60, 60,
                                            60, 60, 60, 60, 60};
        profile.input_scale_bits = profile.default_scale_bits =
            profile.output_scale_bits = 44;
    }
    if (o.compact_r15) {
        profile.coefficient_modulus_bits.pop_back();
        profile.security_level = hera::SecurityLevel::kTc128;
    }
    profile.declared_depth = profile.coefficient_modulus_bits.size() - 1;
    config.he_parameter_profiles.emplace("bench", profile);
    hera::LayerExecutionConfig execution;
    execution.layer_id = layer.id;
    for (const auto region : {hera::HeRegionId::kRegion1,
                              hera::HeRegionId::kRegion2,
                              hera::HeRegionId::kRegion3,
                              hera::HeRegionId::kRegion4,
                              hera::HeRegionId::kRegion5})
        execution.regions.emplace(region, hera::HeRegionConfig{
            region, "bench", hera::Layout::kSpatial,
            hera::Layout::kSpatial});
    config.layer_execution = {execution};
    config.threads = o.he_threads;
    return config;
}

seal::parms_id_type LevelId(const hera::HeContext &context,
                            std::size_t level)
{
    auto data = context.context->first_context_data();
    for (std::size_t i = 0; i < level && data; ++i)
        data = data->next_context_data();
    Require(static_cast<bool>(data), "requested input level is unavailable");
    return data->parms_id();
}

hera::SealCipherTensor EncryptPacked(
    const std::vector<double> &values, const hera::HePackingPlan &packing,
    const hera::HeContext &context, std::size_t level, int scale_bits,
    hera::HeRegionId region)
{
    Require(values.size() == packing.logical_locations.size(),
            "input values do not match packing");
    std::vector<std::vector<std::complex<double>>> slots(
        packing.ciphertext_count,
        std::vector<std::complex<double>>(packing.slot_count));
    for (std::size_t i = 0; i < values.size(); ++i)
        for (const auto &where : packing.Locations(i)) {
            auto &slot = slots.at(where.ciphertext).at(where.slot);
            if (where.component == hera::ComplexLane::kReal)
                slot.real(values[i]);
            else slot.imag((packing.layout == hera::Layout::kMultiHeadK ? -1.0 : 1.0) * values[i]);
        }
    seal::CKKSEncoder encoder(*context.context);
    seal::Encryptor encryptor(*context.context, *context.secret_key);
    std::vector<seal::Ciphertext> ciphertexts(packing.ciphertext_count);
    for (std::size_t i = 0; i < ciphertexts.size(); ++i) {
        seal::Plaintext plain;
        encoder.encode(slots[i], LevelId(context, level),
                       std::ldexp(1.0, scale_bits), plain);
        encryptor.encrypt_symmetric(plain, ciphertexts[i]);
    }
    hera::TensorSpec spec;
    spec.shape = packing.logical_shape;
    spec.domain = hera::Domain::kHeCiphertext;
    spec.layout = packing.layout;
    spec.numeric = {43, 13, true};
    spec.he_region = region;
    spec.he_parameter_profile = context.profile;
    spec.planned_level = level;
    spec.planned_scale_bits = scale_bits;
    return {context.profile, std::move(ciphertexts), spec, packing,
            packing.layout == hera::Layout::kMultiHeadK};
}

hera::SealCipherTensor Metadata(hera::SealCipherTensor tensor)
{
    for (auto &ct : tensor.ciphertexts) {
        const auto id = ct.parms_id();
        const auto scale = ct.scale();
        ct = seal::Ciphertext();
        ct.parms_id() = id;
        ct.scale() = scale;
    }
    return tensor;
}

std::vector<double> DecodePacked(const hera::SealCipherTensor &tensor,
                                 const hera::HeContext &context)
{
    seal::Decryptor decryptor(*context.context, *context.secret_key);
    seal::CKKSEncoder encoder(*context.context);
    std::vector<std::vector<std::complex<double>>> slots(
        tensor.ciphertexts.size());
    for (std::size_t i = 0; i < slots.size(); ++i) {
        seal::Plaintext plain;
        decryptor.decrypt(tensor.ciphertexts[i], plain);
        encoder.decode(plain, slots[i]);
    }
    std::vector<double> values(tensor.physical_packing.logical_locations.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        const auto &where = tensor.physical_packing.Locations(i).front();
        const auto z = slots.at(where.ciphertext).at(where.slot);
        values[i] = where.component == hera::ComplexLane::kReal
            ? z.real() : (tensor.conjugated_complex ? -z.imag() : z.imag());
    }
    return values;
}

std::vector<double> Values(std::size_t count)
{
    std::vector<double> values(count);
    for (std::size_t i = 0; i < count; ++i)
        values[i] = static_cast<double>(static_cast<int>(i % 11) - 5) * 0.004;
    return values;
}

std::vector<double> MatrixProduct(const std::vector<double> &input,
                                  const std::vector<double> &weights,
                                  std::size_t tokens, std::size_t width,
                                  std::size_t output_width)
{
    std::vector<double> result(tokens * output_width);
    for (std::size_t t = 0; t < tokens; ++t)
        for (std::size_t o = 0; o < output_width; ++o)
            for (std::size_t i = 0; i < width; ++i)
                result[t * output_width + o] +=
                    input[t * width + i] * weights[i * output_width + o];
    return result;
}

void Report(const Options &options, std::vector<double> elapsed,
            std::size_t input_ct, std::size_t output_ct,
            std::size_t rotations, std::size_t ct_plain,
            std::size_t ct_ct, std::size_t rescales, double max_error)
{
    if (options.virtual_weights) max_error = std::numeric_limits<double>::quiet_NaN();
    std::sort(elapsed.begin(), elapsed.end());
    const auto mean = std::accumulate(elapsed.begin(), elapsed.end(), 0.0) /
                      elapsed.size();
    std::cout << std::fixed << std::setprecision(3)
              << "HERA_BENCHMARK=" << (options.with_layer_norm ? (options.shared_r15 ? "wd-residual-ln-statistics" : "wo-residual-ln-statistics") : options.kernel) << '\n'
              << "BOUNDARY=prepared_inputs_to_kernel_output\n"
              << "TOKENS=" << options.tokens << '\n'
              << "HIDDEN=" << options.hidden << '\n'
              << "FFN_WIDTH=" << options.ffn_width << '\n'
              << "POLYNOMIAL_DEGREE=" << options.degree << '\n'
              << "INPUT_CIPHERTEXTS=" << input_ct << '\n'
              << "OUTPUT_CIPHERTEXTS=" << output_ct << '\n'
              << "HE_THREADS=" << options.he_threads << '\n'
              << "HE_POOL_THREADS=" << hera::detail::HeWorkerPool::Instance().WorkerThreadCount() << '\n'
              << "HE_MEMORY_BUDGET_BYTES=" << hera::detail::HeWorkerPool::Instance().MemoryBudget() << '\n'
              << "WARMUP=" << options.warmup << '\n'
              << "ITERATIONS=" << options.iterations << '\n'
              << "HERA_MEAN_MS=" << mean << '\n'
              << "HERA_MEDIAN_MS=" << elapsed[elapsed.size() / 2] << '\n'
              << "HERA_MIN_MS=" << elapsed.front() << '\n'
              << "HERA_ROTATIONS=" << rotations << '\n'
              << "HERA_CT_PLAIN_MULTIPLICATIONS=" << ct_plain << '\n'
              << "HERA_CT_CT_MULTIPLICATIONS=" << ct_ct << '\n'
              << "HERA_RESCALES=" << rescales << '\n'
              << "MAX_ABS_ERROR=" << std::scientific << std::setprecision(8) << max_error << '\n'
              << "HERA_BENCHMARK_STATUS=PASS\n";
}

} // namespace

int main(int argc, char **argv)
{
    try {
        const auto o = Parse(argc, argv);
        hera::ConfigureProjectionRotationHoisting(o.hoisting);
        hera::ConfigureProjectionBabySteps(o.baby_steps);
        const auto config = MakeConfig(o);
        const auto &layer = config.layers.front();
        const auto &params = config.he_parameter_profiles.at("bench");
        std::cout << "DATA_PRIMES=" << params.coefficient_modulus_bits.size() << '\n'
                  << "TC128_MAX_COEFF_BITS=" << seal::CoeffModulus::MaxBitCount(o.degree, seal::sec_level_type::tc128) << '\n'
                  << "COEFF_BITS_WITH_SPECIAL=" << std::accumulate(params.coefficient_modulus_bits.begin(), params.coefficient_modulus_bits.end(), 60) << '\n';
        const auto slots = o.degree / 2;
        hera::HePackingPlan input_plan, output_plan;
        std::size_t input_level = 0;
        int input_scale = params.input_scale_bits;
        hera::HeRegionId region = hera::HeRegionId::kRegion2;
        std::size_t in_width = 0, out_width = 0;
        std::vector<int> steps;
        if (o.kernel == "qkv") {
            auto plan = hera::MultiHeadPackingPlan::RealStrictColumns(
                {{o.tokens, o.hidden}}, slots);
            Require(plan.ok(), plan.status().message());
            input_plan = plan.take_value();
            std::set<int> projection_steps;
            for (const auto layout : {hera::Layout::kMultiHeadQ,
                                      hera::Layout::kMultiHeadK,
                                      hera::Layout::kMultiHeadV}) {
                auto required =
                    hera::ComplexDiagonalProjectionKernel::RequiredRotationSteps(
                        layer, input_plan, layout);
                Require(required.ok(), required.status().message());
                projection_steps.insert(required.value().begin(),
                                        required.value().end());
            }
            steps.assign(projection_steps.begin(), projection_steps.end());
            auto qk_decision = hera::MultiHeadPackingPlan::ProjectionDecision(
                layer, hera::Layout::kMultiHeadQ, slots);
            auto v_decision = hera::MultiHeadPackingPlan::ProjectionDecision(
                layer, hera::Layout::kMultiHeadV, slots);
            auto bsgs = hera::ComplexDiagonalProjectionPlanner::ColumnBsgs(layer, slots);
            Require(qk_decision.ok() && v_decision.ok() && bsgs.ok(),
                    "QKV preprocessing decision failed");
            std::cout << "QKV_QK_HEADS_PER_TILE=" << qk_decision.value().heads_per_tile << '\n'
                      << "QKV_V_HEADS_PER_TILE=" << v_decision.value().heads_per_tile << '\n'
                      << "QKV_BABY_STEPS=" << bsgs.value().baby_steps << '\n'
                      << "QKV_GIANT_STEPS=" << bsgs.value().giant_steps << '\n';
            std::cout << "QKV_QK_HEAD_GROUPS=";
            for (std::size_t i = 0; i < qk_decision.value().head_groups.size(); ++i)
                std::cout << (i == 0 ? "" : ",") << qk_decision.value().head_groups[i];
            std::cout << '\n';
            input_level = 1;
            in_width = o.hidden;
        } else if (o.kernel == "wo") {
            auto plan = hera::AttentionLn2PackingPlanner::AttentionValue(
                layer, slots);
            Require(plan.ok(), plan.status().message());
            input_plan = plan.value().output_packing;
            auto required = hera::AttentionOutputProjectionKernel::RequiredRealRotationSteps(
                layer, slots);
            Require(required.ok(), required.status().message());
            steps = required.take_value();
            if (o.with_layer_norm) {
                std::size_t stride = 1;
                while (stride < o.tokens) stride <<= 1U;
                std::set<int> combined(steps.begin(), steps.end());
                for (std::size_t step = 1; step < slots / stride; step <<= 1U)
                    combined.insert(static_cast<int>(step * stride));
                steps.assign(combined.begin(), combined.end());
            }
            auto split = hera::AttentionOutputProjectionKernel::RealColumnBsgs(layer, slots);
            Require(split.ok(), split.status().message());
            std::cout << "WO_BABY_STEPS=" << split.value().baby_steps << '\n'
                      << "WO_GIANT_STEPS=" << split.value().giant_steps << '\n';
            input_level = 2;
            in_width = hera::AttentionWidth(layer);
            out_width = o.hidden;
            region = hera::HeRegionId::kRegion4;
        } else if (o.kernel == "wu" || o.kernel == "wd") {
            in_width = o.kernel == "wu" ? o.hidden : o.ffn_width;
            out_width = o.kernel == "wu" ? o.ffn_width : o.hidden;
            auto input = hera::FfnGeluPackingPlanner::ComplexColumns(
                {{o.tokens, in_width}}, slots,
                hera::Layout::kComplexStrictColumn);
            auto output = hera::FfnGeluPackingPlanner::RealColumns(
                {{o.tokens, out_width}}, slots, hera::Layout::kStrictColumn);
            Require(input.ok() && output.ok(), "column packing failed");
            input_plan = input.take_value();
            output_plan = output.take_value();
            auto required = hera::ComplexColumnProjectionKernel::RequiredRotationSteps(
                input_plan, output_plan);
            Require(required.ok(), required.status().message());
            steps = required.take_value();
            input_level = o.kernel == "wu"
                ? params.coefficient_modulus_bits.size() - params.terminal_prime_count - hera::kFfnSuffixDepth + 1
                : o.shared_r15 ? params.coefficient_modulus_bits.size() - params.terminal_prime_count - 4 : 2;
            if (o.kernel == "wu") input_scale = params.default_scale_bits + 15;
            if (o.shared_r15) {
                input_scale = params.output_scale_bits;
                std::set<int> combined(steps.begin(), steps.end());
                if (o.with_layer_norm)
                    for (std::size_t step = 1; step < slots / hera::detail::TokenColumnStride(o.tokens); step <<= 1U)
                        combined.insert(static_cast<int>(step * hera::detail::TokenColumnStride(o.tokens)));
                steps.assign(combined.begin(), combined.end());
            }
            region = o.kernel == "wu" ? hera::HeRegionId::kRegion5
                                       : hera::HeRegionId::kRegion4;
        } else if (o.kernel == "ffn-he") {
            auto packing = hera::FfnGeluPackingPlanner::GetShared(layer, slots);
            Require(packing.ok(), packing.status().message());
            input_plan = packing.value()->centered;
            auto required = hera::SealSciFfnGeluExecutor::RequiredRotationSteps(layer, params);
            Require(required.ok(), required.status().message());
            steps = required.take_value();
            input_level = params.coefficient_modulus_bits.size() - params.terminal_prime_count - hera::kFfnSuffixDepth;
            region = hera::HeRegionId::kRegion5;
        } else if (o.kernel == "gelu-poly") {
            auto plan = hera::FfnGeluPackingPlanner::RealColumns(
                {{o.tokens, o.ffn_width}}, slots, hera::Layout::kStrictColumn);
            Require(plan.ok(), plan.status().message());
            input_plan = plan.take_value();
            input_level = params.coefficient_modulus_bits.size() - params.terminal_prime_count - 3;
            input_scale = 46;
            region = hera::HeRegionId::kRegion5;
        }

        hera::HeContextRegistry client;
        const bool require_relin = o.kernel == "softmax-exp" || o.kernel == "ffn-he" ||
                                   o.kernel == "gelu-poly" || o.with_layer_norm;
        auto created = client.CreateClientProfiles(config, require_relin);
        Require(created.ok(), created.status().message());
        if (!steps.empty()) {
            auto compiled = client.CompileRotationKeys("bench", steps);
            Require(compiled.ok(), compiled.status().message());
        }
        auto bundle = client.ExportPublicBundle();
        Require(bundle.ok(), bundle.status().message());
        hera::HeContextRegistry server;
        auto imported = server.CreateServerProfiles(config, bundle.value());
        Require(imported.ok(), imported.status().message());
        const auto &client_context = client.Require(
            "bench", hera::PartyRole::kClient);
        const auto &server_context = server.Require(
            "bench", hera::PartyRole::kServer);
        Require(server_context.galois_keys.size() == steps.size() &&
                    (server_context.relinearization_keys.size() != 0) ==
                        require_relin,
                "kernel received unused or missing evaluation keys");
        std::cout << "HERA_GALOIS_KEYS=" << server_context.galois_keys.size()
                  << '\n' << "HERA_RELIN_KEYS="
                  << server_context.relinearization_keys.size() << '\n';
        Require(!server_context.has_secret_key(),
                "HE evaluator unexpectedly owns a secret key");

        if (o.kernel == "softmax-exp") {
            auto input = Values(o.tokens * o.heads * o.tokens);
            for (auto &v : input) v = -std::abs(v) * 10.0;
            seal::CKKSEncoder encoder(*client_context.context);
            seal::Plaintext plain;
            seal::Encryptor encryptor(*client_context.context,
                                      *client_context.secret_key);
            const auto ciphertext_count = (input.size() + slots - 1) / slots;
            std::vector<seal::Ciphertext> ciphertexts(ciphertext_count);
            for (std::size_t group = 0; group < ciphertext_count; ++group) {
                std::vector<double> slot_values(slots);
                const auto first = group * slots;
                const auto count = std::min(slots, input.size() - first);
                std::copy_n(input.begin() + first, count, slot_values.begin());
                encoder.encode(slot_values, std::ldexp(1.0, 40), plain);
                encryptor.encrypt_symmetric(plain, ciphertexts[group]);
            }
            hera::SealCipherBatch batch{
                "bench", std::move(ciphertexts),
                {{o.tokens, o.heads, o.tokens}},
                hera::Layout::kSoftmaxRows,
                hera::PhysicalSlotMap::DenseReal(input.size(), slots)};
            auto run = [&] {
                hera::SoftmaxExecutionReport report;
                auto output = hera::SoftmaxExponentKernel::Evaluate(
                    batch, server_context, config.softmax, &report,
                    o.he_threads);
                Require(output.ok(), output.status().message());
                return std::make_pair(output.take_value(), report);
            };
            for (std::size_t i = 0; i < o.warmup; ++i) run();
            std::vector<double> elapsed;
            hera::SealCipherBatch output;
            hera::SoftmaxExecutionReport report;
            for (std::size_t i = 0; i < o.iterations; ++i) {
                const auto start = Clock::now();
                auto result = run();
                const auto stop = Clock::now();
                elapsed.push_back(std::chrono::duration<double, std::milli>(
                    stop - start).count());
                output = std::move(result.first);
                report = result.second;
            }
            seal::Decryptor decryptor(*client_context.context,
                                      *client_context.secret_key);
            double error = 0;
            for (std::size_t group = 0; group < output.ciphertexts.size();
                 ++group) {
                decryptor.decrypt(output.ciphertexts[group], plain);
                std::vector<double> decoded;
                encoder.decode(plain, decoded);
                const auto first = group * slots;
                for (std::size_t lane = 0;
                     lane < std::min(slots, input.size() - first); ++lane) {
                    const auto x = input[first + lane];
                    const auto reference = std::pow(
                        1.0 + std::ldexp(x, -config.softmax.exp_divisor_bits),
                        std::uint64_t{1} << config.softmax.exp_squarings);
                    error = std::max(error,
                                     std::abs(decoded[lane] - reference));
                }
            }
            Require(error < 0.01, "Softmax exponent numerical check failed");
            Report(o, elapsed, batch.ciphertexts.size(),
                   output.ciphertexts.size(), report.rotations,
                   report.ciphertext_plain_multiplications,
                   report.ciphertext_multiplications, report.rescales, error);
            return EXIT_SUCCESS;
        }

        const auto input_values = Values(input_plan.logical_locations.size());
        auto input = EncryptPacked(input_values, input_plan, client_context,
                                   input_level, input_scale, region);
        if (o.kernel == "ffn-he") {
            auto packing = hera::FfnGeluPackingPlanner::GetShared(layer, slots);
            Require(packing.ok(), packing.status().message());
            auto inverse = EncryptPacked(std::vector<double>(o.tokens, 1.0),
                packing.value()->inverse_broadcast, client_context, input_level, 45, region);
            hera::DenseVectorParameter gamma{{"gamma", o.hidden}, std::vector<double>(o.hidden, 1.0)};
            hera::DenseVectorParameter beta{{"beta", o.hidden}, std::vector<double>(o.hidden, 0.0)};
            hera::GammaFusedLayerNormParameters original{gamma, beta};
            hera::DenseMatrixParameter wu{{"Wu", o.hidden, o.ffn_width, false}, Values(o.hidden * o.ffn_width)};
            hera::SealPlainParameterStore prepared;
            Require(prepared.ConfigureVirtualWeightPlaintexts(o.virtual_weights ? 1 : 0).ok(), "virtual cache setup failed");
            Require(prepared.ConfigureProjectionPlaintextMemory(o.plaintext_cache_mib << 20).ok(), "plaintext budget setup failed");
        prepared.BeginPlaintextPreparation();
            auto run = [&](const hera::SealCipherTensor &centered, const hera::SealCipherTensor &inv) {
                hera::ModelPlaintextCacheScope scope(&prepared, layer.id);
                hera::FfnGeluExecutionReport report;
                hera::SealCipherTensor residual;
                auto normalized = hera::FfnLayerNormTailKernel::Evaluate(centered, inv, gamma, beta,
                    layer, server_context, &report, o.he_threads, &original, &residual);
                Require(normalized.ok(), normalized.status().message());
                auto projected = hera::FfnUpProjectionKernel::Evaluate(normalized.value(), wu, std::nullopt,
                    layer, server_context, &report, o.he_threads);
                Require(projected.ok(), projected.status().message());
                auto poly = hera::GeluPolynomialKernel::Evaluate(projected.value(), config.gelu, server_context, &report, o.he_threads);
                Require(poly.ok(), poly.status().message());
                return std::make_pair(poly.take_value(), report);
            };
            const auto preparation_start = Clock::now();
            run(Metadata(input), Metadata(inverse));
            prepared.FreezePlaintextCache();
            std::cout << "MODEL_PREPARE_MS=" << std::chrono::duration<double, std::milli>(Clock::now() - preparation_start).count() << '\n'
                      << "PREPARED_PLAINTEXT_BYTES=" << prepared.encoded_plaintext_bytes() << '\n'
                  << "PROJECTION_RESIDENT_BYTES=" << prepared.projection_resident_bytes() << '\n'
                  << "PROJECTION_SPILLED_BYTES=" << prepared.projection_spilled_bytes() << '\n';
            for (std::size_t i = 0; i < o.warmup; ++i) run(input, inverse);
            std::vector<double> elapsed;
            hera::GeluPolynomialOutputs output;
            hera::FfnGeluExecutionReport report;
            for (std::size_t i = 0; i < o.iterations; ++i) {
                const auto start = Clock::now();
                auto evaluated = run(input, inverse);
                elapsed.push_back(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
                output = std::move(evaluated.first); report = evaluated.second;
            }
            double error = 0;
            if (!o.virtual_weights) {
                const auto raw = MatrixProduct(input_values, wu.row_major, o.tokens, o.hidden, o.ffn_width);
                const auto decoded = DecodePacked(output.branches, client_context);
                for (std::size_t i = 0; i < decoded.size(); ++i) {
                    const double x = raw[i % raw.size()];
                    const auto &g = config.gelu;
                    const auto expected = g.a * std::pow(x, 4) + (i >= raw.size() ? g.b : -g.b) * std::pow(x, 3) +
                        g.c * x * x + (i >= raw.size() ? 0.5 + g.d : 0.5 - g.d) * x + g.e;
                    error = std::max(error, std::abs(decoded[i] - expected));
                }
                Require(error < 0.03, "FFN HE numerical check failed");
            }
            std::cout << "MODEL_NUMERICAL_VALIDATION=" << (o.virtual_weights ? "DISABLED_VIRTUAL_WEIGHTS" : "PASS") << '\n'
                      << "HERA_CONJUGATIONS=" << report.conjugations << '\n'
                      << "HERA_RELINEARIZATIONS=" << report.relinearizations << '\n'
                      << "HERA_KEY_SWITCHES=" << report.rotations + report.conjugations + report.relinearizations << '\n';
            Report(o, elapsed, input.ciphertexts.size(), output.input.ciphertexts.size() + output.branches.ciphertexts.size(),
                report.rotations, report.ciphertext_plain_multiplications, report.ciphertext_multiplications, report.rescales, error);
            return EXIT_SUCCESS;
        }
        if (o.kernel == "gelu-poly") {
            hera::SealPlainParameterStore prepared;
            const auto prepare_start = Clock::now();
            Require(prepared.ConfigureProjectionPlaintextMemory(o.plaintext_cache_mib << 20).ok(), "plaintext budget setup failed");
        prepared.BeginPlaintextPreparation();
            {
                hera::ModelPlaintextCacheScope scope(&prepared, layer.id);
                auto compiled = hera::GeluPolynomialKernel::Evaluate(Metadata(input), config.gelu,
                                                                       server_context);
                Require(compiled.ok(), compiled.status().message());
            }
            prepared.FreezePlaintextCache();
            std::cout << "MODEL_PREPARE_MS=" << std::chrono::duration<double, std::milli>(
                Clock::now() - prepare_start).count() << '\n'
                      << "PREPARED_PLAINTEXTS=" << prepared.encoded_plaintext_count() << '\n'
                      << "PREPARED_PLAINTEXT_BYTES=" << prepared.encoded_plaintext_bytes() << '\n'
                  << "PROJECTION_RESIDENT_BYTES=" << prepared.projection_resident_bytes() << '\n'
                  << "PROJECTION_SPILLED_BYTES=" << prepared.projection_spilled_bytes() << '\n';
            auto run = [&] {
                hera::ModelPlaintextCacheScope scope(&prepared, layer.id);
                hera::FfnGeluExecutionReport report;
                auto output = hera::GeluPolynomialKernel::Evaluate(
                    input, config.gelu, server_context, &report, o.he_threads);
                Require(output.ok(), output.status().message());
                return std::make_pair(output.take_value(), report);
            };
            for (std::size_t i = 0; i < o.warmup; ++i) run();
            std::vector<double> elapsed;
            hera::GeluPolynomialOutputs output;
            hera::FfnGeluExecutionReport report;
            for (std::size_t i = 0; i < o.iterations; ++i) {
                const auto start = Clock::now();
                auto result = run();
                const auto stop = Clock::now();
                elapsed.push_back(std::chrono::duration<double, std::milli>(
                    stop - start).count());
                output = std::move(result.first);
                report = result.second;
            }
            const auto decoded = DecodePacked(output.branches,
                                              client_context);
            double error = 0;
            for (std::size_t i = 0; i < decoded.size(); ++i) {
                const auto logical = i % input_values.size();
                const auto x = input_values[logical];
                const auto &c = config.gelu;
                const bool positive = i >= input_values.size();
                const auto expected = c.a * std::pow(x, 4) +
                    (positive ? c.b : -c.b) * std::pow(x, 3) + c.c * x * x +
                    (positive ? 0.5 + c.d : 0.5 - c.d) * x + c.e;
                error = std::max(error, std::abs(decoded[i] - expected));
            }
            Require(error < 0.02, "GeLU polynomial numerical check failed");
            std::cout << "C2M_BATCH_CIPHERTEXTS=" << output.input.ciphertexts.size() +
                output.branches.ciphertexts.size() << '\n';
            Report(o, elapsed, input.ciphertexts.size(),
                   output.branches.ciphertexts.size(),
                   report.rotations, report.ciphertext_plain_multiplications,
                   report.ciphertext_multiplications, report.rescales, error);
            return EXIT_SUCCESS;
        }

        std::vector<hera::DenseMatrixParameter> weights;
        if (o.kernel == "qkv") {
            for (const auto &item : std::vector<std::pair<std::string,
                                          std::size_t>>{
                     {"Wq", hera::QueryWidth(layer)},
                     {"Wk", hera::KeyWidth(layer)},
                     {"Wv", hera::ValueWidth(layer)}})
                weights.push_back({{item.first, o.hidden, item.second, false},
                                   Values(o.hidden * item.second)});
        } else {
            const auto name = o.kernel == "wo" ? "Wo" :
                              o.kernel == "wu" ? "Wu" : "Wd";
            weights.push_back({{name, in_width, out_width, false},
                               Values(in_width * out_width)});
        }
        hera::SealPlainParameterStore prepared;
        Require(prepared.ConfigureVirtualWeightPlaintexts(o.virtual_weights ? 1 : 0).ok(), "virtual cache setup failed");
        Require(prepared.ConfigureProjectionPlaintextMemory(o.plaintext_cache_mib << 20).ok(), "plaintext budget setup failed");
        prepared.BeginPlaintextPreparation();
        hera::AttentionLn2ExecutionReport wo_report;
        std::optional<hera::SealCipherTensor> residual;
        std::vector<double> residual_values;
        double wo_ms = 0, residual_ms = 0, statistics_ms = 0;
        if (o.with_layer_norm) {
            auto plan = hera::AttentionLn2PackingPlanner::RealStrictColumns(
                {{o.tokens, o.hidden}}, slots);
            Require(plan.ok(), plan.status().message());
            residual_values = Values(o.tokens * o.hidden);
            for (auto &value : residual_values) value *= 2.0;
            residual = EncryptPacked(residual_values, plan.value(), client_context,
                                     o.shared_r15 ? input_level + 1 : 0, input_scale, region);
        }
        std::size_t column_conjugations = 0;
        std::size_t stats_relinearizations = 0;
        const auto evaluate = [&](const hera::SealCipherTensor &source,
                                  std::size_t *rotations,
                                  std::size_t *ct_plain,
                                  std::size_t *rescales) {
            const auto kernel_start = Clock::now();
            std::vector<hera::SealCipherTensor> results;
            hera::ModelPlaintextCacheScope scope(&prepared, layer.id);
            if (o.kernel == "qkv") {
                hera::Block2ExecutionReport report;
                hera::ComplexDiagonalProjectionKernel projection(
                    layer, server_context, &report, o.he_threads);
                for (std::size_t i = 0; i < 3; ++i) {
                    const auto layout = i == 0 ? hera::Layout::kMultiHeadQ :
                                        i == 1 ? hera::Layout::kMultiHeadK :
                                                 hera::Layout::kMultiHeadV;
                    auto result = projection.Evaluate(source, weights[i],
                                                  std::nullopt, layout);
                    Require(result.ok(), result.status().message());
                    results.push_back(result.take_value());
                }
                *rotations = report.rotations;
                *ct_plain = report.ciphertext_plain_multiplications;
                *rescales = report.rescales;
            } else if (o.kernel == "wo") {
                hera::AttentionLn2ExecutionReport report;
                const auto projection_start = Clock::now();
                auto result = hera::AttentionOutputProjectionKernel::EvaluateReal(
                    source, weights[0], std::nullopt, layer, server_context,
                    &report, o.he_threads);
                Require(result.ok(), result.status().message());
                const auto projection_end = Clock::now();
                if (o.with_layer_norm && !prepared.preparing_plaintexts()) {
                    auto combined = hera::AttentionResidualKernel::Add(
                        result.take_value(), *residual, server_context, &report);
                    Require(combined.ok(), combined.status().message());
                    const auto residual_end = Clock::now();
                    auto statistics = hera::LayerNormStatisticsKernel::Evaluate(
                        combined.value(), layer, server_context, &report, o.he_threads);
                    Require(statistics.ok(), statistics.status().message());
                    const auto statistics_end = Clock::now();
                    wo_ms = std::chrono::duration<double, std::milli>(projection_end - projection_start).count();
                    residual_ms = std::chrono::duration<double, std::milli>(residual_end - projection_end).count();
                    statistics_ms = std::chrono::duration<double, std::milli>(statistics_end - residual_end).count();
                    results.push_back(std::move(statistics.value().centered));
                    results.push_back(std::move(statistics.value().variance));
                } else {
                    if (o.with_layer_norm) {
                        auto ready = hera::LayerNormStatisticsKernel::Prepare(
                            result.value(), layer, server_context);
                        Require(ready.ok(), ready.status().message());
                    }
                    results.push_back(result.take_value());
                }
                wo_report = report;
                *rotations = report.rotations;
                *ct_plain = report.ciphertext_plain_multiplications;
                *rescales = report.rescales;
            } else {
                hera::HeKernelOperationCounts report;
                auto result = hera::ComplexColumnProjectionKernel::Evaluate(
                    source, weights[0], std::nullopt, output_plan, region,
                    (o.shared_r15 || o.kernel == "wu") ? params.output_scale_bits : params.default_scale_bits, server_context, &report,
                    o.he_threads);
                Require(result.ok(), result.status().message());
                if (o.with_layer_norm) {
                    const auto projection_end = Clock::now();
                    wo_ms = std::chrono::duration<double, std::milli>(projection_end - kernel_start).count();
                    if (!prepared.preparing_plaintexts()) {
                        seal::Evaluator evaluator(*server_context.context);
                        for (std::size_t ct = 0; ct < result.value().ciphertexts.size(); ++ct)
                            evaluator.add_inplace(result.value().ciphertexts[ct], residual->ciphertexts[ct]);
                    }
                    const auto residual_end = Clock::now();
                    const auto statistics_level = params.coefficient_modulus_bits.size() - params.terminal_prime_count - 3;
                    if (!prepared.preparing_plaintexts()) {
                        seal::Evaluator evaluator(*server_context.context);
                        for (auto &ct : result.value().ciphertexts)
                            evaluator.mod_switch_to_inplace(ct, LevelId(server_context, statistics_level));
                    } else for (auto &ct : result.value().ciphertexts) ct.parms_id() = LevelId(server_context, statistics_level);
                    result.value().logical_spec.planned_level = statistics_level;
                    if (prepared.preparing_plaintexts()) {
                        auto compiled = hera::LayerNormStatisticsKernel::Prepare(result.value(), layer, server_context, nullptr, params.output_scale_bits);
                        Require(compiled.ok(), compiled.status().message());
                    } else {
                        hera::AttentionLn2ExecutionReport statistics_report;
                        auto stats = hera::LayerNormStatisticsKernel::Evaluate(result.value(), layer, server_context,
                            &statistics_report, o.he_threads, params.output_scale_bits);
                        Require(stats.ok(), stats.status().message());
                        statistics_ms = std::chrono::duration<double, std::milli>(Clock::now() - residual_end).count();
                        residual_ms = std::chrono::duration<double, std::milli>(residual_end - projection_end).count();
                        stats_relinearizations = statistics_report.relinearizations;
                        wo_report = statistics_report;
                        report.rotations += statistics_report.rotations;
                        report.rescales += statistics_report.rescales;
                        report.ciphertext_plain_multiplications += statistics_report.ciphertext_plain_multiplications;
                        results.push_back(std::move(stats.value().centered));
                        results.push_back(std::move(stats.value().variance));
                    }
                } else results.push_back(result.take_value());
                *rotations = report.rotations;
                *ct_plain = report.ciphertext_plain_multiplications;
                *rescales = report.rescales;
                column_conjugations = report.conjugations;
            }
            return results;
        };
        std::size_t rotations = 0, ct_plain = 0, rescales = 0;
        const auto prepare_started = Clock::now();
        evaluate(Metadata(input), &rotations, &ct_plain, &rescales);
        prepared.FreezePlaintextCache();
        std::cout << "MODEL_PREPARE_MS=" << std::chrono::duration<double, std::milli>(
            Clock::now() - prepare_started).count() << '\n'
                  << "PREPARED_PLAINTEXTS=" << prepared.encoded_plaintext_count() << '\n'
                  << "PREPARED_PLAINTEXT_BYTES=" << prepared.encoded_plaintext_bytes() << '\n'
                  << "PROJECTION_RESIDENT_BYTES=" << prepared.projection_resident_bytes() << '\n'
                  << "PROJECTION_SPILLED_BYTES=" << prepared.projection_spilled_bytes() << '\n';
        for (std::size_t i = 0; i < o.warmup; ++i)
            evaluate(input, &rotations, &ct_plain, &rescales);
        std::vector<double> elapsed;
        std::vector<hera::SealCipherTensor> output;
        double wo_total_ms = 0, residual_total_ms = 0, statistics_total_ms = 0;
        for (std::size_t i = 0; i < o.iterations; ++i) {
            const auto start = Clock::now();
            auto result = evaluate(input, &rotations, &ct_plain, &rescales);
            const auto stop = Clock::now();
            wo_total_ms += wo_ms;
            residual_total_ms += residual_ms;
            statistics_total_ms += statistics_ms;
            elapsed.push_back(std::chrono::duration<double, std::milli>(
                stop - start).count());
            output = std::move(result);
        }
        double error = 0;
        std::vector<double> expected_centered, expected_variance;
        if (o.with_layer_norm && !o.virtual_weights) {
            expected_centered = MatrixProduct(input_values, weights[0].row_major,
                                             o.tokens, in_width, out_width);
            expected_variance.resize(o.tokens);
            for (std::size_t t = 0; t < o.tokens; ++t) {
                double mean = 0;
                for (std::size_t f = 0; f < o.hidden; ++f) {
                    const auto index = t * o.hidden + f;
                    expected_centered[index] += residual_values[index];
                    mean += expected_centered[index];
                }
                mean /= o.hidden;
                for (std::size_t f = 0; f < o.hidden; ++f) {
                    auto &value = expected_centered[t * o.hidden + f];
                    value -= mean;
                    expected_variance[t] += value * value / o.hidden;
                }
                expected_variance[t] += layer.attention_norm_epsilon;
            }
        }
        for (std::size_t j = 0; j < output.size() && !o.virtual_weights; ++j) {
            const auto actual = DecodePacked(output[j], client_context);
            if (o.with_layer_norm) {
                const auto &reference = j == 0 ? expected_centered : expected_variance;
                Require(actual.size() == reference.size(), "LN output shape mismatch");
                double local_error = 0;
                for (std::size_t i = 0; i < actual.size(); ++i)
                    local_error = std::max(local_error, std::abs(actual[i] - reference[i]));
                error = std::max(error, local_error);
                Require(local_error < (o.shared_r15 ? 0.03 : (j == 0 ? 1e-5 : 1e-7)), "Wo/LN numerical check failed");
                Require(output[j].ciphertexts.front().parms_id() ==
                            LevelId(client_context, o.shared_r15 ? (params.coefficient_modulus_bits.size() - params.terminal_prime_count - (j == 0 ? 2 : 0)) : input_level + (j == 0 ? 2 : 4)),
                        "Wo/LN output level mismatch");
                std::cout << (j == 0 ? "CENTERED_MAX_ERROR=" : "VARIANCE_MAX_ERROR=")
                          << std::scientific << local_error << '\n';
                continue;
            }
            auto expected = MatrixProduct(
                input_values, weights[j].row_major, o.tokens, in_width,
                weights[j].spec.columns);
            if (o.kernel == "qkv" && j < 2) {
                const double factor = 1.0 / std::sqrt(2.0 *
                    (j == 0 ? static_cast<double>(o.qk_width) : 1.0));
                for (auto &value : expected) value *= factor;
            }
            Require(actual.size() == expected.size(),
                    "projection output shape mismatch");
            for (std::size_t i = 0; i < actual.size(); ++i)
                error = std::max(error, std::abs(actual[i] - expected[i]));
        }
        Require(error < (o.kernel == "qkv" ? 1e-4 : 0.03),
                "projection numerical check failed: " +
                                std::to_string(error));
        std::size_t output_ct = 0;
        for (const auto &tensor : output)
            output_ct += tensor.ciphertexts.size();
        std::cout << "MODEL_NUMERICAL_VALIDATION=" << (o.virtual_weights ? "DISABLED_VIRTUAL_WEIGHTS" : "PASS") << '\n';
        Report(o, elapsed, input.ciphertexts.size(), output_ct,
               rotations, ct_plain, wo_report.ciphertext_multiplications, rescales, error);
        if (o.kernel == "wu" || o.kernel == "wd")
            std::cout << "HERA_CONJUGATIONS=" << column_conjugations << '\n'
                      << "HERA_RELINEARIZATIONS=" << stats_relinearizations << '\n'
                      << "HERA_KEY_SWITCHES=" << rotations + column_conjugations + stats_relinearizations << '\n';
        if (o.kernel == "wo")
            std::cout << "HERA_CONJUGATIONS=" << wo_report.conjugations << '\n'
                      << "WO_BABY_ROTATIONS=" << wo_report.wo_baby_rotations << '\n'
                      << "WO_GIANT_ROTATIONS=" << wo_report.wo_giant_rotations << '\n'
                      << "WO_OUTPUT_CONJUGATIONS=" << wo_report.wo_output_conjugations << '\n'
                      << "WO_PHASE_MULTIPLICATIONS=" << wo_report.wo_phase_multiplications << '\n'
                      << "HERA_KS_OPERATIONS=" << rotations + wo_report.conjugations + wo_report.relinearizations << '\n'
                      << "HERA_RELINEARIZATIONS=" << wo_report.relinearizations << '\n';
        if (o.with_layer_norm)
            std::cout << (o.shared_r15 ? "WD_MEAN_MS=" : "WO_MEAN_MS=") << wo_total_ms / o.iterations << '\n'
                      << "RESIDUAL_MEAN_MS=" << residual_total_ms / o.iterations << '\n'
                      << "STATISTICS_MEAN_MS=" << statistics_total_ms / o.iterations << '\n'
                      << "STATISTICS_PLAINTEXT_ENCODINGS=" << wo_report.layer_norm_plaintext_encodings << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << "HERA_BENCHMARK_STATUS=FAIL " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
