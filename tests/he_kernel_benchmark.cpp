#include "hera/attention_ln2.h"
#include "hera/detail/parallel_ranges.h"
#include "hera/detail/level_schedule.h"

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
    std::size_t tokens{8};
    std::size_t heads{2};
    std::size_t width{8};
    std::size_t degree{8192};
    std::size_t iterations{3};
    std::size_t warmup{1};
    std::size_t he_threads{1};
    double blb_milliseconds{-1.0};
    bool block2_chain{};
};

void Require(bool condition, const std::string &message)
{
    if (!condition) throw std::runtime_error(message);
}

Options Parse(int argc, char **argv)
{
    Options result;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        const auto equals = arg.find('=');
        Require(equals != std::string::npos, "expected --name=value: " + arg);
        const auto name = arg.substr(0, equals);
        const auto value = arg.substr(equals + 1);
        if (name == "--blb-ms") {
            result.blb_milliseconds = std::stod(value);
            Require(std::isfinite(result.blb_milliseconds) &&
                        result.blb_milliseconds > 0,
                    "--blb-ms must be positive and finite");
            continue;
        }
        const auto number = std::stoull(value);
        Require(number > 0 || (name == "--warmup" && number == 0),
                name + " must be positive");
        if (name == "--tokens") result.tokens = number;
        else if (name == "--heads") result.heads = number;
        else if (name == "--width") result.width = number;
        else if (name == "--degree") result.degree = number;
        else if (name == "--iterations") result.iterations = number;
        else if (name == "--warmup") result.warmup = number;
        else if (name == "--he-threads") result.he_threads = number;
        else if (name == "--block2-chain") result.block2_chain = number != 0;
        else throw std::runtime_error("unknown option: " + name);
    }
    Require(result.degree >= 4096 &&
                (result.degree & (result.degree - 1)) == 0,
            "--degree must be a power of two >= 4096");
    return result;
}

hera::HeraConfig MakeConfig(const Options &options)
{
    hera::HeraConfig config;
    hera::TransformerLayerConfig layer;
    layer.id = "benchmark";
    layer.tokens = options.tokens;
    layer.q_heads = options.heads;
    layer.kv_heads = options.heads;
    layer.qk_head_width = options.width;
    layer.value_head_width = options.width;
    layer.model_width = options.heads * options.width;
    layer.ffn_width = 2 * layer.model_width;
    layer.norm = hera::NormPlacement::kPreNorm;
    config.layers = {layer};

    // These are kernel-only chains, not Figure-10 fused-block parameters.
    // QK consumes four levels, A*V consumes three; the last data prime is
    // retained as the output level. HeContextRegistry adds the separate
    // key-switch special prime.
    hera::HeParameterConfig profile;
    profile.polynomial_degree = options.degree;
#ifdef HERA_BENCHMARK_QK
    profile.coefficient_modulus_bits = {60, 60, 60, 60, 60, 60};
    profile.declared_depth = 5;
    if (options.block2_chain) {
        profile.coefficient_modulus_bits = {60, 35, 60, 60, 60, 60, 60};
        profile.declared_depth = 6;
    }
#else
    profile.coefficient_modulus_bits = {60, 60, 60, 60};
    profile.declared_depth = 3;
    if (options.block2_chain) {
        profile.coefficient_modulus_bits = {60, 35, 60, 60, 60, 60, 60};
        profile.declared_depth = 6;
    }
#endif
    profile.default_scale_bits = 44;
    profile.input_scale_bits = 44;
    profile.output_scale_bits = 44;
    profile.terminal_prime_count = 1;
    profile.security_level = hera::SecurityLevel::kNone;
    config.he_parameter_profiles.emplace("kernel_bench", profile);
    hera::LayerExecutionConfig execution;
    execution.layer_id = layer.id;
    for (const auto region : {hera::HeRegionId::kRegion1,
                              hera::HeRegionId::kRegion2,
                              hera::HeRegionId::kRegion3,
                              hera::HeRegionId::kRegion4,
                              hera::HeRegionId::kRegion5})
        execution.regions.emplace(
            region, hera::HeRegionConfig{region, "kernel_bench",
                                         hera::Layout::kSpatial,
                                         hera::Layout::kSpatial});
    config.layer_execution = {execution};
    config.threads = options.he_threads;
    return config;
}

hera::SealCipherTensor EncryptPacked(
    const std::vector<double> &values, const hera::HePackingPlan &packing,
    const hera::HeContext &context)
{
    Require(values.size() == packing.logical_locations.size(),
            "packed input size mismatch");
    std::vector<std::vector<std::complex<double>>> slots(
        packing.ciphertext_count,
        std::vector<std::complex<double>>(packing.slot_count));
    for (std::size_t logical = 0; logical < values.size(); ++logical)
        for (const auto &place : packing.Locations(logical)) {
            auto &slot = slots.at(place.ciphertext).at(place.slot);
            if (place.component == hera::ComplexLane::kReal)
                slot.real(values[logical]);
            else
                slot.imag((packing.layout == hera::Layout::kMultiHeadK ? -1.0 : 1.0) * values[logical]);
        }
    seal::CKKSEncoder encoder(*context.context);
    seal::Encryptor encryptor(*context.context, *context.secret_key);
    std::vector<seal::Ciphertext> ciphertexts(packing.ciphertext_count);
    const auto scale = std::ldexp(1.0, context.parameters.input_scale_bits);
    for (std::size_t i = 0; i < ciphertexts.size(); ++i) {
        seal::Plaintext plaintext;
        encoder.encode(slots[i], scale, plaintext);
        encryptor.encrypt_symmetric(plaintext, ciphertexts[i]);
    }
    hera::TensorSpec spec;
    spec.shape = packing.logical_shape;
    spec.domain = hera::Domain::kHeCiphertext;
    spec.layout = packing.layout;
    spec.numeric = {43, 13, true};
    spec.he_region = hera::HeRegionId::kRegion2;
    spec.he_parameter_profile = context.profile;
    spec.planned_level = 0;
    spec.planned_scale_bits = context.parameters.input_scale_bits;
    return {context.profile, std::move(ciphertexts), spec, packing,
            packing.layout == hera::Layout::kMultiHeadK};
}

std::vector<double> DecryptPacked(const hera::SealCipherTensor &tensor,
                                  const hera::HeContext &context)
{
    seal::Decryptor decryptor(*context.context, *context.secret_key);
    seal::CKKSEncoder encoder(*context.context);
    std::vector<std::vector<std::complex<double>>> decoded(
        tensor.ciphertexts.size());
    for (std::size_t i = 0; i < tensor.ciphertexts.size(); ++i) {
        seal::Plaintext plaintext;
        decryptor.decrypt(tensor.ciphertexts[i], plaintext);
        encoder.decode(plaintext, decoded[i]);
    }
    std::vector<double> result(tensor.physical_packing.logical_locations.size());
    for (std::size_t i = 0; i < result.size(); ++i) {
        const auto &place = tensor.physical_packing.Locations(i).front();
        const auto value = decoded[place.ciphertext][place.slot];
        result[i] = place.component == hera::ComplexLane::kReal
            ? value.real() : (tensor.conjugated_complex ? -value.imag() : value.imag());
    }
    return result;
}

std::size_t LevelOf(const hera::HeContext &context,
                    seal::parms_id_type parms_id)
{
    std::size_t level = 0;
    for (auto data = context.context->first_context_data(); data;
         data = data->next_context_data(), ++level)
        if (data->parms_id() == parms_id) return level;
    throw std::runtime_error("output parms_id is outside benchmark context");
}

void PrintTimings(std::vector<double> milliseconds,
                  const Options &options, std::size_t input_ciphertexts,
                  const hera::SealCipherTensor &output,
                  const hera::HeContext &context, const seal::Ciphertext &input,
                  std::size_t rotations, std::size_t conjugations,
                  std::size_t ct_ct,
                  std::size_t relins, std::size_t rescales,
                  double maximum_error)
{
    std::sort(milliseconds.begin(), milliseconds.end());
    const double mean = std::accumulate(
        milliseconds.begin(), milliseconds.end(), 0.0) /
        milliseconds.size();
    const double median = milliseconds[milliseconds.size() / 2];
#ifdef HERA_BENCHMARK_QK
    const char *name = "QK";
#else
    const char *name = "ATTENTION_V";
#endif
    std::cout << std::fixed << std::setprecision(3)
              << "HERA_BENCHMARK=" << name << '\n'
              << "BOUNDARY=prepacked_ciphertexts_to_kernel_output\n"
              << "TOKENS=" << options.tokens << '\n'
              << "HEADS=" << options.heads << '\n'
              << "HEAD_WIDTH=" << options.width << '\n'
              << "POLYNOMIAL_DEGREE=" << options.degree << '\n'
              << "DATA_PRIMES=" << context.context->first_context_data()->parms().coeff_modulus().size()
              << '\n'
              << "INPUT_DATA_PRIMES=" << context.context->get_context_data(input.parms_id())->parms().coeff_modulus().size() << '\n'
              << "OUTPUT_DATA_PRIMES=" << context.context->get_context_data(output.ciphertexts.front().parms_id())->parms().coeff_modulus().size() << '\n'
              << "INPUT_CIPHERTEXTS=" << input_ciphertexts << '\n'
              << "OUTPUT_CIPHERTEXTS=" << output.ciphertexts.size() << '\n'
              << "OUTPUT_LEVEL=" << output.logical_spec.planned_level.value()
              << '\n'
              << "HE_THREADS=" << options.he_threads << '\n'
              << "HE_POOL_THREADS=" << hera::detail::HeWorkerPool::Instance().WorkerThreadCount() << '\n'
              << "HE_MEMORY_BUDGET_BYTES=" << hera::detail::HeWorkerPool::Instance().MemoryBudget() << '\n'
              << "WARMUP=" << options.warmup << '\n'
              << "ITERATIONS=" << options.iterations << '\n'
              << "HERA_MEAN_MS=" << mean << '\n'
              << "HERA_MEDIAN_MS=" << median << '\n'
              << "HERA_MIN_MS=" << milliseconds.front() << '\n'
              << "HERA_ROTATIONS=" << rotations << '\n'
              << "HERA_CONJUGATIONS=" << conjugations << '\n'
              << "HERA_CT_CT_MULTIPLICATIONS=" << ct_ct << '\n'
              << "HERA_RELINEARIZATIONS=" << relins << '\n'
              << "HERA_RESCALES=" << rescales << '\n'
              << "MAX_ABS_ERROR=" << std::scientific
              << std::setprecision(8) << maximum_error << std::fixed
              << std::setprecision(3) << '\n';
    if (options.blb_milliseconds > 0) {
        std::cout << "BLB_EXTERNAL_MEAN_MS=" << options.blb_milliseconds
                  << '\n' << "HERA_OVER_BLB_TIME_RATIO="
                  << mean / options.blb_milliseconds << '\n';
    } else {
        std::cout << "BLB_EXTERNAL_MEAN_MS=NOT_MEASURED\n";
    }
    std::cout << "HERA_BENCHMARK_STATUS=PASS\n";
}

} // namespace

int main(int argc, char **argv)
{
    try {
        const auto options = Parse(argc, argv);
        const auto config = MakeConfig(options);
        auto valid = hera::ValidateHeraConfig(config);
        Require(valid.ok(), valid.status().message());
        const auto &layer = config.layers.front();
        const auto slots = options.degree / 2;
#ifdef HERA_BENCHMARK_QK
        auto q_plan = hera::MultiHeadPackingPlan::ProjectionOutput(
            layer, hera::Layout::kMultiHeadQ, slots);
        auto k_plan = hera::MultiHeadPackingPlan::ProjectionOutput(
            layer, hera::Layout::kMultiHeadK, slots);
        Require(q_plan.ok() && k_plan.ok(), "Q/K packing failed");
        auto steps = hera::MultiHeadQkKernel::RequiredRotationSteps(
            layer, slots);
#else
        auto av_plan = hera::AttentionLn2PackingPlanner::AttentionValue(
            layer, slots);
        auto v_plan = hera::MultiHeadPackingPlan::ProjectionOutput(
            layer, hera::Layout::kMultiHeadV, slots);
        Require(av_plan.ok() && v_plan.ok(), "A/V packing failed");
        auto steps = hera::Result<std::vector<int>>(
            av_plan.value().required_rotation_steps);
#endif
        Require(steps.ok(), steps.status().message());
        hera::HeContextRegistry client;
        Require(client.CreateClientProfiles(config).ok(),
                "client CKKS context setup failed");
        auto keys = client.CompileRotationKeys("kernel_bench", steps.value());
        Require(keys.ok(), keys.status().message());
        auto public_bundle = client.ExportPublicBundle();
        Require(public_bundle.ok(), public_bundle.status().message());
        hera::HeContextRegistry server;
        auto imported = server.CreateServerProfiles(config,
                                                     public_bundle.value());
        Require(imported.ok(), imported.status().message());
        const auto &client_context = client.Require(
            "kernel_bench", hera::PartyRole::kClient);
        Require(client_context.galois_keys.size() == steps.value().size(),
                "kernel generated more Galois keys than requested");
        std::cout << "HERA_GALOIS_KEYS=" << client_context.galois_keys.size()
                  << '\n' << "HERA_RELIN_KEYS="
                  << client_context.relinearization_keys.size() << '\n';
        const auto &server_context = server.Require(
            "kernel_bench", hera::PartyRole::kServer);
        Require(!server_context.has_secret_key(),
                "benchmark evaluator must not own a secret key");

        const std::size_t tokens = options.tokens;
        const std::size_t heads = options.heads;
        const std::size_t width = options.width;
        std::vector<double> left, right, expected;
        hera::SealCipherTensor first, second;
#ifdef HERA_BENCHMARK_QK
        left.resize(tokens * heads * width);
        right.resize(left.size());
        expected.resize(heads * tokens * tokens);
        for (std::size_t i = 0; i < left.size(); ++i) {
            left[i] = std::sin(static_cast<double>(i + 1) * 0.1) /
                      (10.0 * std::sqrt(static_cast<double>(width)));
            right[i] = std::cos(static_cast<double>(i + 2) * 0.07) / 10.0;
        }
        for (std::size_t h = 0; h < heads; ++h)
            for (std::size_t row = 0; row < tokens; ++row)
                for (std::size_t col = 0; col < tokens; ++col)
                    for (std::size_t d = 0; d < width; ++d)
                        expected[(h * tokens + row) * tokens + col] +=
                            left[(row * heads + h) * width + d] *
                            right[(col * heads + h) * width + d];
        for (auto &value : left) value /= std::sqrt(2.0);
        for (auto &value : right) value /= std::sqrt(2.0);
        first = EncryptPacked(left, q_plan.value(), client_context);
        second = EncryptPacked(right, k_plan.value(), client_context);
        const auto start_level = client_context.parameters
            .coefficient_modulus_bits.size() -
            client_context.parameters.terminal_prime_count -
            hera::MultiHeadQkKernel::kDepth;
        seal::Evaluator evaluator(*client_context.context);
        for (auto *tensor : {&first, &second}) {
            for (auto &ct : tensor->ciphertexts)
                for (std::size_t level = 0; level < start_level; ++level)
                    evaluator.mod_switch_to_next_inplace(ct);
            tensor->logical_spec.planned_level = start_level;
        }
#else
        left.resize(tokens * heads * tokens);
        right.resize(tokens * heads * width);
        expected.resize(right.size());
        for (std::size_t row = 0; row < tokens; ++row)
            for (std::size_t h = 0; h < heads; ++h) {
                const auto offset = (row * heads + h) * tokens;
                double sum = 0.0;
                for (std::size_t col = 0; col < tokens; ++col) {
                    left[offset + col] = 1.0 + ((row * 3 + h * 2 + col * 5) % 11);
                    sum += left[offset + col];
                }
                for (std::size_t col = 0; col < tokens; ++col)
                    left[offset + col] /= sum;
            }
        for (std::size_t i = 0; i < right.size(); ++i)
            right[i] = std::sin(static_cast<double>(i + 1) * 0.08) / 10.0;
        for (std::size_t row = 0; row < tokens; ++row)
            for (std::size_t h = 0; h < heads; ++h)
                for (std::size_t d = 0; d < width; ++d)
                    for (std::size_t col = 0; col < tokens; ++col)
                        expected[(row * heads + h) * width + d] +=
                            left[(row * heads + h) * tokens + col] *
                            right[(col * heads + h) * width + d];
        first = EncryptPacked(left, av_plan.value().probability_packing,
                              client_context);
        second = EncryptPacked(right, v_plan.value(), client_context);
        if (options.block2_chain) {
            const auto av_start = hera::detail::AttentionValueInputLevel(server_context.parameters);
            auto data = server_context.context->first_context_data();
            for (std::size_t level = 0; level < av_start; ++level)
                data = data->next_context_data();
            seal::Evaluator evaluator(*server_context.context);
            for (auto *tensor : {&first, &second}) {
                for (auto &ct : tensor->ciphertexts) evaluator.mod_switch_to_inplace(ct, data->parms_id());
                tensor->logical_spec.planned_level = av_start;
            }
        }
#endif
#ifdef HERA_BENCHMARK_QK
        auto bsgs = hera::MultiHeadQkKernel::Plan(layer, slots);
        Require(bsgs.ok(), bsgs.status().message());
        std::cout << "QK_HEAD_TILE_COUNT=" << bsgs.value().head_tiles.size() << '\n';
        for (std::size_t i = 0; i < bsgs.value().head_tiles.size(); ++i) {
            const auto &tile = bsgs.value().head_tiles[i];
            const auto prefix = "QK_TILE_" + std::to_string(i) + "_";
            std::cout << prefix << "HEAD_BEGIN=" << tile.head_begin << '\n'
                      << prefix << "HEADS=" << tile.heads_per_tile << '\n'
                      << prefix << "INPUT_GROUPS=" << tile.input_groups << '\n'
                      << prefix << "PREPROCESS_BABY_STEPS=" << tile.preprocessing_baby_steps << '\n'
                      << prefix << "PREPROCESS_GIANT_STEPS=" << tile.preprocessing_giant_steps << '\n'
                      << prefix << "BABY_STEPS=" << tile.baby_steps << '\n'
                      << prefix << "GIANT_STEPS=" << tile.giant_steps << '\n'
                      << prefix << "PLANNED_KS="
                      << tile.rotations + tile.conjugations + tile.relinearizations << '\n';
        }
#endif
#ifdef HERA_BENCHMARK_QK
        hera::SealPlainParameterStore prepared;
        prepared.BeginPlaintextPreparation();
        const auto prepare_started = std::chrono::steady_clock::now();
        {
            hera::ModelPlaintextCacheScope scope(&prepared, layer.id);
            auto compiled = hera::MultiHeadQkKernel::Prepare(layer, server_context);
            Require(compiled.ok(), compiled.status().message());
        }
        prepared.FreezePlaintextCache();
        std::cout << "QK_PREPARE_MS=" << std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - prepare_started).count() << '\n'
                  << "QK_PREPARED_PLAINTEXTS=" << prepared.encoded_plaintext_count() << '\n'
                  << "QK_PREPARED_PLAINTEXT_BYTES=" << prepared.encoded_plaintext_bytes() << '\n';
#else
        hera::SealPlainParameterStore prepared;
        prepared.BeginPlaintextPreparation();
        const auto prepare_started = std::chrono::steady_clock::now();
        {
            hera::ModelPlaintextCacheScope scope(&prepared, layer.id);
            auto compiled = hera::MultiHeadAttentionValueKernel::Prepare(layer, server_context,
                first.logical_spec.planned_level.value());
            Require(compiled.ok(), compiled.status().message());
        }
        prepared.FreezePlaintextCache();
        std::cout << "AV_PREPARE_MS=" << std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - prepare_started).count() << '\n'
                  << "AV_PREPARED_PLAINTEXTS=" << prepared.encoded_plaintext_count() << '\n'
                  << "AV_PREPARED_PLAINTEXT_BYTES=" << prepared.encoded_plaintext_bytes() << '\n';
#endif
        auto evaluate = [&](auto *report) {
#ifdef HERA_BENCHMARK_QK
            hera::ModelPlaintextCacheScope scope(&prepared, layer.id);
            return hera::MultiHeadQkKernel::Evaluate(
                first, second, layer, server_context, report,
                options.he_threads);
#else
            hera::ModelPlaintextCacheScope scope(&prepared, layer.id);
            return hera::MultiHeadAttentionValueKernel::Evaluate(
                first, second, layer, server_context, report,
                options.he_threads);
#endif
        };
        for (std::size_t i = 0; i < options.warmup; ++i) {
#ifdef HERA_BENCHMARK_QK
            hera::Block2ExecutionReport report;
#else
            hera::AttentionLn2ExecutionReport report;
#endif
            auto result = evaluate(&report);
            Require(result.ok(), result.status().message());
        }
        std::vector<double> elapsed;
        elapsed.reserve(options.iterations);
        hera::SealCipherTensor output;
#ifdef HERA_BENCHMARK_QK
        hera::Block2ExecutionReport operations;
#else
        hera::AttentionLn2ExecutionReport operations;
#endif
        for (std::size_t i = 0; i < options.iterations; ++i) {
            decltype(operations) report;
            const auto start = Clock::now();
            auto result = evaluate(&report);
            const auto stop = Clock::now();
            Require(result.ok(), result.status().message());
            elapsed.push_back(std::chrono::duration<double, std::milli>(
                                  stop - start).count());
            output = result.take_value();
            operations = report;
        }
#ifdef HERA_BENCHMARK_QK
        std::cout << "QK_PLANNED_KS="
                  << bsgs.value().rotations + bsgs.value().conjugations +
                         bsgs.value().relinearizations << '\n';
        std::cout << "HERA_KS_OPERATIONS="
                  << operations.rotations + operations.conjugations + operations.relinearizations << '\n'
                  << "QK_INPUT_LEVEL=" << first.logical_spec.planned_level.value() << '\n'
                  << "QK_RESCALE_DEPTH=" << hera::MultiHeadQkKernel::kDepth << '\n';
#endif
        const auto actual = DecryptPacked(output, client_context);
        Require(actual.size() == expected.size(), "output shape mismatch");
        double maximum_error = 0.0;
        for (std::size_t i = 0; i < actual.size(); ++i)
            maximum_error = std::max(maximum_error,
                                     std::abs(actual[i] - expected[i]));
        Require(maximum_error <= 0.025, "kernel numerical error is too large");
        Require(LevelOf(client_context, output.ciphertexts.front().parms_id()) ==
                    first.logical_spec.planned_level.value() +
#ifdef HERA_BENCHMARK_QK
                        hera::MultiHeadQkKernel::kDepth,
#else
                        3,
#endif
                "kernel did not consume its dedicated modulus chain");
#ifndef HERA_BENCHMARK_QK
        std::cout << "AV_LANE_RULE=largest_power_of_two\n"
                  << "A_INPUT_CIPHERTEXTS=" << first.ciphertexts.size() << '\n'
                  << "V_INPUT_CIPHERTEXTS=" << second.ciphertexts.size() << '\n'
                  << "A_PREPROCESS_ROTATIONS="
                  << operations.attention_probability_rotations << '\n'
                  << "V_PREPROCESS_ROTATIONS="
                  << operations.attention_value_rotations << '\n'
                  << "GIANT_RECOVERY_ROTATIONS="
                  << operations.attention_recovery_rotations << '\n'
                  << "HERA_KS_OPERATIONS="
                  << (operations.rotations + operations.conjugations +
                      operations.relinearizations) << '\n';
#endif
        PrintTimings(std::move(elapsed), options,
                     first.ciphertexts.size() + second.ciphertexts.size(),
                     output, server_context, first.ciphertexts.front(), operations.rotations,
                     operations.conjugations,
                     operations.ciphertext_multiplications,
                     operations.relinearizations, operations.rescales,
                     maximum_error);
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << "HERA_BENCHMARK_STATUS=FAIL " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
