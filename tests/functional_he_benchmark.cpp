#include "hera/attention_ln2.h"
#include "hera/ffn_gelu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
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
        else throw std::invalid_argument("unknown option: " + name);
    }
    Require(options.kernel == "qkv" || options.kernel == "wo" ||
                options.kernel == "wu" || options.kernel == "wd" ||
                options.kernel == "softmax-exp" ||
                options.kernel == "gelu-poly", "unsupported --kernel");
    Require(options.tokens && options.hidden && options.heads &&
                options.kv_heads && options.qk_width && options.value_width &&
                options.ffn_width && options.iterations && options.he_threads,
            "all dimensions, iterations, and thread counts must be positive");
    Require(options.heads % options.kv_heads == 0,
            "heads must be divisible by kv-heads");
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
    } else if (o.kernel == "gelu-poly" || o.kernel == "wu") {
        profile.coefficient_modulus_bits = {60, 60, 46, 46, 46, 46, 60, 60};
        profile.input_scale_bits = profile.default_scale_bits = 45;
        profile.output_scale_bits = 46;
    } else {
        profile.coefficient_modulus_bits = {60, 35, 60, 60, 60,
                                            60, 60, 60, 60, 60};
        profile.input_scale_bits = profile.default_scale_bits =
            profile.output_scale_bits = 44;
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
            else slot.imag(values[i]);
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
    return {context.profile, std::move(ciphertexts), spec, packing};
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
            ? z.real() : z.imag();
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
    std::sort(elapsed.begin(), elapsed.end());
    const auto mean = std::accumulate(elapsed.begin(), elapsed.end(), 0.0) /
                      elapsed.size();
    std::cout << std::fixed << std::setprecision(3)
              << "HERA_BENCHMARK=" << options.kernel << '\n'
              << "BOUNDARY=prepared_inputs_to_kernel_output\n"
              << "TOKENS=" << options.tokens << '\n'
              << "HIDDEN=" << options.hidden << '\n'
              << "FFN_WIDTH=" << options.ffn_width << '\n'
              << "POLYNOMIAL_DEGREE=" << options.degree << '\n'
              << "INPUT_CIPHERTEXTS=" << input_ct << '\n'
              << "OUTPUT_CIPHERTEXTS=" << output_ct << '\n'
              << "HE_THREADS=" << options.he_threads << '\n'
              << "WARMUP=" << options.warmup << '\n'
              << "ITERATIONS=" << options.iterations << '\n'
              << "HERA_MEAN_MS=" << mean << '\n'
              << "HERA_MEDIAN_MS=" << elapsed[elapsed.size() / 2] << '\n'
              << "HERA_MIN_MS=" << elapsed.front() << '\n'
              << "HERA_ROTATIONS=" << rotations << '\n'
              << "HERA_CT_PLAIN_MULTIPLICATIONS=" << ct_plain << '\n'
              << "HERA_CT_CT_MULTIPLICATIONS=" << ct_ct << '\n'
              << "HERA_RESCALES=" << rescales << '\n'
              << "MAX_ABS_ERROR=" << max_error << '\n'
              << "HERA_BENCHMARK_STATUS=PASS\n";
}

} // namespace

int main(int argc, char **argv)
{
    try {
        const auto o = Parse(argc, argv);
        const auto config = MakeConfig(o);
        const auto &layer = config.layers.front();
        const auto &params = config.he_parameter_profiles.at("bench");
        const auto slots = o.degree / 2;
        hera::HePackingPlan input_plan, output_plan;
        std::size_t input_level = 0;
        int input_scale = params.input_scale_bits;
        hera::HeRegionId region = hera::HeRegionId::kRegion2;
        std::size_t in_width = 0, out_width = 0;
        std::vector<int> steps;
        if (o.kernel == "qkv") {
            auto plan = hera::MultiHeadPackingPlan::DenseSpatial(
                {{o.tokens, o.hidden}}, slots);
            Require(plan.ok(), plan.status().message());
            input_plan = plan.take_value();
            auto required = hera::SealBlock2Executor::RequiredRotationSteps(
                layer, params, input_plan);
            Require(required.ok(), required.status().message());
            steps = required.take_value();
            input_level = 1;
            in_width = o.hidden;
        } else if (o.kernel == "wo") {
            auto plan = hera::AttentionLn2PackingPlanner::AttentionValue(
                layer, slots);
            Require(plan.ok(), plan.status().message());
            input_plan = plan.value().output_packing;
            auto required = hera::SealSciAttentionLn2Executor::RequiredRotationSteps(
                layer, params);
            Require(required.ok(), required.status().message());
            steps = required.take_value();
            input_level = 5;
            in_width = hera::AttentionWidth(layer);
            out_width = o.hidden;
            region = hera::HeRegionId::kRegion4;
        } else if (o.kernel == "wu" || o.kernel == "wd") {
            in_width = o.kernel == "wu" ? o.hidden : o.ffn_width;
            out_width = o.kernel == "wu" ? o.ffn_width : o.hidden;
            auto input = hera::FfnGeluPackingPlanner::ComplexColumns(
                {{o.tokens, in_width}}, slots,
                hera::Layout::kComplexStrictColumn);
            auto output = hera::FfnGeluPackingPlanner::ComplexColumns(
                {{o.tokens, out_width}}, slots,
                hera::Layout::kComplexStrictColumn);
            Require(input.ok() && output.ok(), "column packing failed");
            input_plan = input.take_value();
            output_plan = output.take_value();
            auto required = hera::ComplexColumnProjectionKernel::RequiredRotationSteps(
                input_plan, output_plan);
            Require(required.ok(), required.status().message());
            steps = required.take_value();
            input_level = o.kernel == "wu" ? 1 : 5;
            region = o.kernel == "wu" ? hera::HeRegionId::kRegion5
                                       : hera::HeRegionId::kRegion4;
        } else if (o.kernel == "gelu-poly") {
            auto plan = hera::FfnGeluPackingPlanner::RealColumns(
                {{o.tokens, o.ffn_width}}, slots, hera::Layout::kStrictColumn);
            Require(plan.ok(), plan.status().message());
            input_plan = plan.take_value();
            input_level = 4;
            input_scale = 46;
            region = hera::HeRegionId::kRegion5;
        }

        hera::HeContextRegistry client;
        auto created = client.CreateClientProfiles(config);
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
        if (o.kernel == "gelu-poly") {
            auto run = [&] {
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
            const auto decoded = DecodePacked(output.positive_branch,
                                              client_context);
            double error = 0;
            for (std::size_t i = 0; i < decoded.size(); ++i) {
                const auto x = input_values[i];
                const auto &c = config.gelu;
                const auto expected = c.a * std::pow(x, 4) +
                    c.b * std::pow(x, 3) + c.c * x * x +
                    (0.5 + c.d) * x + c.e;
                error = std::max(error, std::abs(decoded[i] - expected));
            }
            Require(error < 0.02, "GeLU polynomial numerical check failed");
            Report(o, elapsed, input.ciphertexts.size(),
                   output.positive_branch.ciphertexts.size(),
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
        prepared.BeginPlaintextPreparation();
        const auto evaluate = [&](const hera::SealCipherTensor &source,
                                  std::size_t *rotations,
                                  std::size_t *ct_plain,
                                  std::size_t *rescales) {
            std::vector<hera::SealCipherTensor> results;
            hera::ModelPlaintextCacheScope scope(&prepared, layer.id);
            if (o.kernel == "qkv") {
                hera::Block2ExecutionReport report;
                hera::ComplexDiagonalProjectionKernel qk(
                    layer, server_context, &report);
                hera::ComplexDiagonalProjectionKernel v(
                    layer, server_context, &report);
                for (std::size_t i = 0; i < 3; ++i) {
                    auto &kernel = i == 2 ? v : qk;
                    const auto layout = i == 0 ? hera::Layout::kMultiHeadQ :
                                        i == 1 ? hera::Layout::kMultiHeadK :
                                                 hera::Layout::kMultiHeadV;
                    auto result = kernel.Evaluate(source, weights[i],
                                                  std::nullopt, layout);
                    Require(result.ok(), result.status().message());
                    results.push_back(result.take_value());
                }
                *rotations = report.rotations;
                *ct_plain = report.ciphertext_plain_multiplications;
                *rescales = report.rescales;
            } else if (o.kernel == "wo") {
                hera::AttentionLn2ExecutionReport report;
                auto result = hera::AttentionOutputProjectionKernel::Evaluate(
                    source, weights[0], std::nullopt, layer, server_context,
                    &report, o.he_threads);
                Require(result.ok(), result.status().message());
                results.push_back(result.take_value());
                *rotations = report.rotations;
                *ct_plain = report.ciphertext_plain_multiplications;
                *rescales = report.rescales;
            } else {
                hera::HeKernelOperationCounts report;
                auto result = hera::ComplexColumnProjectionKernel::Evaluate(
                    source, weights[0], std::nullopt, output_plan, region,
                    params.default_scale_bits, server_context, &report,
                    o.he_threads);
                Require(result.ok(), result.status().message());
                results.push_back(result.take_value());
                *rotations = report.rotations;
                *ct_plain = report.ciphertext_plain_multiplications;
                *rescales = report.rescales;
            }
            return results;
        };
        std::size_t rotations = 0, ct_plain = 0, rescales = 0;
        evaluate(Metadata(input), &rotations, &ct_plain, &rescales);
        prepared.FreezePlaintextCache();
        for (std::size_t i = 0; i < o.warmup; ++i)
            evaluate(input, &rotations, &ct_plain, &rescales);
        std::vector<double> elapsed;
        std::vector<hera::SealCipherTensor> output;
        for (std::size_t i = 0; i < o.iterations; ++i) {
            const auto start = Clock::now();
            auto result = evaluate(input, &rotations, &ct_plain, &rescales);
            const auto stop = Clock::now();
            elapsed.push_back(std::chrono::duration<double, std::milli>(
                stop - start).count());
            output = std::move(result);
        }
        double error = 0;
        for (std::size_t j = 0; j < output.size(); ++j) {
            const auto actual = DecodePacked(output[j], client_context);
            const auto expected = MatrixProduct(
                input_values, weights[j].row_major, o.tokens, in_width,
                weights[j].spec.columns);
            Require(actual.size() == expected.size(),
                    "projection output shape mismatch");
            for (std::size_t i = 0; i < actual.size(); ++i)
                error = std::max(error, std::abs(actual[i] - expected[i]));
        }
        Require(error < 0.03, "projection numerical check failed: " +
                                std::to_string(error));
        std::size_t output_ct = 0;
        for (const auto &tensor : output)
            output_ct += tensor.ciphertexts.size();
        Report(o, elapsed, input.ciphertexts.size(), output_ct,
               rotations, ct_plain, 0, rescales, error);
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << "HERA_BENCHMARK_STATUS=FAIL " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
