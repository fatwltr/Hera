#include "hera/attention_ln2.h"
#include "hera/detail/parallel_ranges.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

void Require(bool condition, const std::string &message)
{
    if (!condition) throw std::runtime_error(message);
}

seal::parms_id_type LevelId(const hera::HeContext &context, std::size_t level)
{
    auto data = context.context->first_context_data();
    for (std::size_t i = 0; i < level && data; ++i)
        data = data->next_context_data();
    Require(static_cast<bool>(data), "requested level is unavailable");
    return data->parms_id();
}

std::size_t ActualLevel(const hera::HeContext &context,
                        const seal::Ciphertext &ciphertext)
{
    std::size_t level = 0;
    for (auto data = context.context->first_context_data(); data;
         data = data->next_context_data(), ++level)
        if (data->parms_id() == ciphertext.parms_id()) return level;
    throw std::runtime_error("unknown ciphertext level");
}

hera::SealCipherTensor Encrypt(const std::vector<double> &values,
                              const hera::HePackingPlan &plan,
                              const hera::HeContext &context,
                              std::size_t level)
{
    std::vector<std::vector<std::complex<double>>> slots(
        plan.ciphertext_count,
        std::vector<std::complex<double>>(plan.slot_count));
    for (std::size_t logical = 0; logical < values.size(); ++logical) {
        const auto &where = plan.Locations(logical).front();
        auto &slot = slots[where.ciphertext][where.slot];
        if (where.component == hera::ComplexLane::kReal)
            slot.real(values[logical]);
        else slot.imag(values[logical]);
    }
    seal::CKKSEncoder encoder(*context.context);
    seal::Encryptor encryptor(*context.context, *context.secret_key);
    hera::SealCipherTensor input;
    input.parameter_profile = context.profile;
    input.physical_packing = plan;
    input.ciphertexts.resize(plan.ciphertext_count);
    for (std::size_t i = 0; i < slots.size(); ++i) {
        seal::Plaintext plain;
        encoder.encode(slots[i], LevelId(context, level),
                       std::ldexp(1.0, context.parameters.default_scale_bits),
                       plain);
        encryptor.encrypt_symmetric(plain, input.ciphertexts[i]);
    }
    input.logical_spec.shape = plan.logical_shape;
    input.logical_spec.domain = hera::Domain::kHeCiphertext;
    input.logical_spec.layout = plan.layout;
    input.logical_spec.numeric = {43, 13, true};
    input.logical_spec.he_region = hera::HeRegionId::kRegion4;
    input.logical_spec.he_parameter_profile = context.profile;
    input.logical_spec.planned_level = level;
    input.logical_spec.planned_scale_bits = context.parameters.default_scale_bits;
    return input;
}

std::vector<double> Decode(const hera::SealCipherTensor &tensor,
                           const hera::HeContext &context)
{
    seal::Decryptor decryptor(*context.context, *context.secret_key);
    seal::CKKSEncoder encoder(*context.context);
    std::vector<std::vector<double>> decoded(tensor.ciphertexts.size());
    for (std::size_t i = 0; i < decoded.size(); ++i) {
        seal::Plaintext plain;
        decryptor.decrypt(tensor.ciphertexts[i], plain);
        encoder.decode(plain, decoded[i]);
    }
    std::vector<double> values(tensor.physical_packing.logical_locations.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        const auto &where = tensor.physical_packing.Locations(i).front();
        values[i] = decoded[where.ciphertext][where.slot];
    }
    return values;
}

double MaxError(const std::vector<double> &actual,
                const std::vector<double> &expected)
{
    Require(actual.size() == expected.size(), "output shape differs");
    double error = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        Require(std::isfinite(actual[i]), "non-finite output");
        error = std::max(error, std::abs(actual[i] - expected[i]));
    }
    return error;
}
} // namespace

int main(int argc, char **argv)
{
    try {
        std::string config_path = "configs/bert_base_single_layer_seq128.json";
        std::size_t tokens = 0, hidden = 0, warmup = 2, iterations = 20;
        std::size_t he_threads = 1, memory_mib = 128;
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            const auto split = arg.find('=');
            Require(split != std::string::npos, "expected --name=value");
            const auto key = arg.substr(0, split);
            const auto value = arg.substr(split + 1);
            if (key == "--config") config_path = value;
            else if (key == "--tokens") tokens = std::stoull(value);
            else if (key == "--hidden") hidden = std::stoull(value);
            else if (key == "--warmup") warmup = std::stoull(value);
            else if (key == "--iterations") iterations = std::stoull(value);
            else if (key == "--he-threads") he_threads = std::stoull(value);
            else if (key == "--he-memory-mib") memory_mib = std::stoull(value);
            else throw std::invalid_argument("unknown option: " + key);
        }
        Require(iterations != 0, "iterations must be positive");
        Require(he_threads != 0 && memory_mib != 0 && memory_mib <= 512,
                "thread count must be positive and memory budget must be 1..512 MiB");
        hera::detail::HeWorkerPool::Instance().SetMemoryBudget(memory_mib << 20);
        auto loaded = hera::LoadHeraConfig(config_path);
        Require(loaded.ok(), loaded.status().message());
        auto config = loaded.take_value();
        auto layer = config.layers.front();
        auto execution = *hera::FindLayerExecution(config, layer.id);
        const auto profile_name = execution.regions.at(
            config.placement.attention_output_ln2).parameter_profile;
        const auto params = config.he_parameter_profiles.at(profile_name);
        if (tokens) layer.tokens = tokens;
        if (hidden) layer.model_width = hidden;
        // Only instantiate the selected Region-4 profile for this HE-only run.
        config.layers = {layer};
        config.he_parameter_profiles = {{profile_name, params}};
        for (auto &entry : execution.regions)
            entry.second.parameter_profile = profile_name;
        config.layer_execution = {execution};
        config.threads = 1;
        auto checked = hera::AttentionLn2LevelPlanner::ValidateOutput(params);
        Require(checked.ok(), checked.status().message());
        const auto input_level =
            hera::AttentionLn2LevelPlanner::OutputInputLevel(params) + 1;
        const auto slots = params.polynomial_degree / 2;
        std::size_t stride = 1;
        while (stride < layer.tokens) stride <<= 1U;
        Require(stride <= slots, "token column does not fit in slots");
        const auto features = slots / stride;
        std::vector<int> steps{0}; // Conjugation is used only in untimed realify.
        for (std::size_t step = 1; step < features; step <<= 1U)
            steps.push_back(static_cast<int>(step * stride));

        hera::HeContextRegistry client;
        checked = client.CreateClientProfiles(config);
        Require(checked.ok(), checked.status().message());
        checked = client.CompileRotationKeys(profile_name, steps);
        Require(checked.ok(), checked.status().message());
        auto bundle = client.ExportPublicBundle();
        Require(bundle.ok(), bundle.status().message());
        hera::HeContextRegistry server;
        checked = server.CreateServerProfiles(config, bundle.value());
        Require(checked.ok(), checked.status().message());
        const auto &client_context = client.Require(profile_name,
                                                    hera::PartyRole::kClient);
        const auto &server_context = server.Require(profile_name,
                                                    hera::PartyRole::kServer);
        Require(!server_context.has_secret_key(), "server has a secret key");
        auto complex_plan = hera::AttentionLn2PackingPlanner::ComplexStrictColumns(
            {{layer.tokens, layer.model_width}}, slots);
        Require(complex_plan.ok(), complex_plan.status().message());
        std::vector<double> values(layer.tokens * layer.model_width);
        for (std::size_t t = 0; t < layer.tokens; ++t)
            for (std::size_t f = 0; f < layer.model_width; ++f)
                values[t * layer.model_width + f] =
                    0.5 * std::sin(0.13 * static_cast<double>(f) + 0.07 * t) +
                    0.2 * std::cos(0.03 * static_cast<double>(f) - 0.11 * t);
        auto complex_input = Encrypt(values, complex_plan.value(), client_context,
                                     input_level);
        auto realified = hera::ComplexRealifyKernel::Evaluate(
            complex_input, layer, server_context);
        Require(realified.ok(), realified.status().message());
        const auto input = realified.take_value();

        const auto run = [&](hera::AttentionLn2ExecutionReport *report) {
            return hera::LayerNormStatisticsKernel::Evaluate(
                input, layer, server_context, report, he_threads);
        };
        hera::AttentionLn2ExecutionReport cold_report;
        const auto cold_start = Clock::now();
        auto cold_result = run(&cold_report);
        const auto cold_stop = Clock::now();
        Require(cold_result.ok(), cold_result.status().message());
        cold_result.take_value();
        for (std::size_t i = 0; i < warmup; ++i) {
            auto result = run(nullptr);
            Require(result.ok(), result.status().message());
        }
        std::vector<double> wall_ms, cpu_ms;
        wall_ms.reserve(iterations);
        cpu_ms.reserve(iterations);
        hera::LayerNormStatisticsOutputs output;
        hera::AttentionLn2ExecutionReport last_report;
        for (std::size_t i = 0; i < iterations; ++i) {
            hera::AttentionLn2ExecutionReport report;
            const auto cpu_start = std::clock();
            const auto start = Clock::now();
            auto result = run(&report);
            const auto stop = Clock::now();
            const auto cpu_stop = std::clock();
            Require(result.ok(), result.status().message());
            wall_ms.push_back(std::chrono::duration<double, std::milli>(
                stop - start).count());
            cpu_ms.push_back(1000.0 * (cpu_stop - cpu_start) / CLOCKS_PER_SEC);
            output = result.take_value();
            last_report = report;
        }
        std::vector<double> centered(values.size()), variance(layer.tokens);
        for (std::size_t t = 0; t < layer.tokens; ++t) {
            const auto begin = values.begin() + t * layer.model_width;
            const auto mean = std::accumulate(begin, begin + layer.model_width,
                                               0.0) / layer.model_width;
            double square_sum = 0;
            for (std::size_t f = 0; f < layer.model_width; ++f) {
                const auto index = t * layer.model_width + f;
                centered[index] = values[index] - mean;
                square_sum += centered[index] * centered[index];
            }
            variance[t] = square_sum / layer.model_width +
                          layer.attention_norm_epsilon;
        }
        const auto centered_error = MaxError(Decode(output.centered, client_context),
                                             centered);
        const auto variance_error = MaxError(Decode(output.variance, client_context),
                                             variance);
        Require(centered_error < 0.01 && variance_error < 0.01,
                "LayerNorm numerical verification failed");
        const auto n = input.ciphertexts.size();
        const auto reduction_steps = steps.size() - 1;
        Require(last_report.rotations == 2 * reduction_steps &&
                    last_report.ciphertext_plain_multiplications == 3 * n + last_report.relinearizations &&
                    last_report.ciphertext_multiplications == n &&
                    last_report.relinearizations > 0 && last_report.relinearizations <= n &&
                    last_report.rescales == n + last_report.relinearizations + 1,
                "optimized operation counts differ from expected");
        Require(last_report.layer_norm_workers <= 2 &&
                    last_report.layer_norm_workers <= he_threads &&
                    hera::detail::HeWorkerPool::Instance().WorkerThreadCount() <= 1,
                "statistics exceeded its worker limit");
        Require(ActualLevel(client_context, output.centered.ciphertexts.front()) ==
                    input_level + 1 &&
                    ActualLevel(client_context, output.variance.ciphertexts.front()) ==
                    input_level + 3,
                "output levels differ from expected");
        const auto mean = std::accumulate(wall_ms.begin(), wall_ms.end(), 0.0) /
                          iterations;
        double squared_deviation = 0;
        for (auto value : wall_ms) squared_deviation += (value - mean) * (value - mean);
        auto sorted = wall_ms;
        std::sort(sorted.begin(), sorted.end());
        const auto median = (sorted[(iterations - 1) / 2] + sorted[iterations / 2]) / 2;
        const auto p95_index = static_cast<std::size_t>(std::ceil(0.95 * iterations)) - 1;
        std::cout << std::fixed << std::setprecision(6)
                  << "HERA_BENCHMARK=ln2-statistics\n"
                  << "BOUNDARY=LayerNormStatisticsKernel::Evaluate\n"
                  << "INPUT_PREPARATION=complex_encrypt_then_realify_untimed\n"
                  << "INCLUDES=validation_masks_encoding_HE_output_packaging\n"
                  << "CONFIG=" << config_path << '\n'
                  << "PROFILE=" << profile_name << '\n'
                  << "TOKENS=" << layer.tokens << '\n'
                  << "HIDDEN=" << layer.model_width << '\n'
                  << "POLYNOMIAL_DEGREE=" << params.polynomial_degree << '\n'
                  << "SLOT_COUNT=" << slots << '\n'
                  << "FEATURES_PER_CIPHERTEXT=" << features << '\n'
                  << "INPUT_CIPHERTEXTS=" << n << '\n'
                  << "INPUT_LEVEL=" << input_level << '\n'
                  << "TARGET_SCALE_BITS=" << params.default_scale_bits << '\n'
                  << "DATA_MODULUS_BITS=";
        for (auto bits : params.coefficient_modulus_bits) std::cout << bits << ',';
        std::cout << '\n' << "INPUT_SCALE_BITS=";
        for (const auto &ct : input.ciphertexts) std::cout << std::log2(ct.scale()) << ',';
        std::cout << '\n' << "SECURITY_LEVEL=" << hera::ToString(params.security_level)
                  << '\n' << "HE_THREADS_REQUESTED=" << he_threads << '\n'
                  << "HE_THREADS=" << last_report.layer_norm_workers << '\n'
                  << "BACKGROUND_POOL_THREADS="
                  << hera::detail::HeWorkerPool::Instance().WorkerThreadCount() << '\n'
                  << "HE_MEMORY_BUDGET_MIB=" << memory_mib << '\n'
                  << "COLD_MS=" << std::chrono::duration<double, std::milli>(
                         cold_stop - cold_start).count() << '\n'
                  << "COLD_PLAINTEXT_ENCODINGS=" << cold_report.layer_norm_plaintext_encodings
                  << '\n' << "STEADY_PLAINTEXT_ENCODINGS="
                  << last_report.layer_norm_plaintext_encodings << '\n'
                  << "WARMUP=" << warmup << '\n'
                  << "ITERATIONS=" << iterations << '\n'
                  << "MEAN_MS=" << mean << '\n'
                  << "MEDIAN_MS=" << median << '\n'
                  << "P95_MS=" << sorted[p95_index] << '\n'
                  << "MIN_MS=" << sorted.front() << '\n'
                  << "MAX_MS=" << sorted.back() << '\n'
                  << "STDDEV_MS=" << std::sqrt(squared_deviation / iterations) << '\n'
                  << "MEAN_CPU_MS=" << std::accumulate(cpu_ms.begin(), cpu_ms.end(), 0.0) /
                                             iterations << '\n'
                  << "ROTATIONS=" << last_report.rotations << '\n'
                  << "CT_PLAIN_MULTIPLICATIONS=" << last_report.ciphertext_plain_multiplications
                  << '\n' << "CT_CT_MULTIPLICATIONS=" << last_report.ciphertext_multiplications
                  << '\n' << "RELINEARIZATIONS=" << last_report.relinearizations
                  << '\n' << "RESCALES=" << last_report.rescales
                  << std::scientific
                  << '\n' << "CENTERED_MAX_ABS_ERROR=" << centered_error
                  << '\n' << "VARIANCE_MAX_ABS_ERROR=" << variance_error
                  << std::fixed
                  << '\n' << "SAMPLES_MS=";
        for (auto ms : wall_ms) std::cout << ms << ',';
        std::cout << "\nHERA_BENCHMARK_STATUS=PASS\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "HERA_BENCHMARK_STATUS=FAIL " << error.what() << '\n';
        return 1;
    }
}
