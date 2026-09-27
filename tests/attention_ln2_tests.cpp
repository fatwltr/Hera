#include "hera/attention_ln2.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <sys/wait.h>
#include <signal.h>
#include <unistd.h>

namespace {

void Require(bool condition, const std::string &message)
{
    if (!condition) throw std::runtime_error(message);
}

hera::HeraConfig MakeConfig()
{
    hera::HeraConfig config;
    hera::TransformerLayerConfig layer;
    layer.id = "layer0";
    layer.tokens = 2;
    layer.model_width = 4;
    layer.q_heads = 2;
    layer.kv_heads = 1;
    layer.qk_head_width = 2;
    layer.value_head_width = 2;
    layer.ffn_width = 8;
    layer.norm = hera::NormPlacement::kPreNorm;
    layer.output_bias = true;
    layer.attention_residual = true;
    layer.attention_norm_epsilon = 1.0e-5;
    config.layers = {layer};
    config.mpc_numeric = {43, 13, true};
    config.conversion = {40, 20, true};
    hera::HeParameterConfig profile;
    profile.polynomial_degree = 4096;
    profile.coefficient_modulus_bits = {
        60, 35, 60, 60, 60, 60, 60, 60, 60, 60};
    profile.default_scale_bits = 44;
    profile.input_scale_bits = 44;
    profile.output_scale_bits = 44;
    profile.declared_depth = 9;
    profile.terminal_prime_count = 1;
    profile.require_exact_terminal_level = true;
    profile.security_level = hera::SecurityLevel::kNone;
    config.he_parameter_profiles.emplace("shared", profile);
    hera::LayerExecutionConfig execution;
    execution.layer_id = layer.id;
    for (const auto region : {hera::HeRegionId::kRegion1,
                              hera::HeRegionId::kRegion2,
                              hera::HeRegionId::kRegion3,
                              hera::HeRegionId::kRegion4,
                              hera::HeRegionId::kRegion5})
        execution.regions.emplace(
            region, hera::HeRegionConfig{region, "shared",
                                         hera::Layout::kSpatial,
                                         hera::Layout::kSpatial});
    config.layer_execution = {execution};
    config.threads = 1;
    return config;
}

seal::parms_id_type LevelId(const hera::HeContext &context,
                            std::size_t target)
{
    auto data = context.context->first_context_data();
    for (std::size_t level = 0; level < target && data; ++level)
        data = data->next_context_data();
    if (!data) throw std::runtime_error("requested HE level is absent");
    return data->parms_id();
}

std::size_t ActualLevel(const hera::HeContext &context,
                        seal::parms_id_type id)
{
    std::size_t level = 0;
    for (auto data = context.context->first_context_data(); data;
         data = data->next_context_data(), ++level)
        if (data->parms_id() == id) return level;
    throw std::runtime_error("unknown parms_id");
}

hera::SealCipherTensor Encrypt(
    const std::vector<double> &values, const hera::HePackingPlan &packing,
    const hera::HeContext &context, std::size_t level,
    hera::HeRegionId region)
{
    std::vector<std::vector<std::complex<double>>> slots(
        packing.ciphertext_count,
        std::vector<std::complex<double>>(packing.slot_count));
    Require(values.size() == packing.logical_locations.size(),
            "plaintext and packing logical sizes differ");
    for (std::size_t logical = 0; logical < values.size(); ++logical) {
        for (const auto &location : packing.Locations(logical)) {
            auto &slot = slots[location.ciphertext][location.slot];
            if (location.component == hera::ComplexLane::kReal)
                slot.real(values[logical]);
            else
                slot.imag(values[logical]);
        }
    }
    seal::CKKSEncoder encoder(*context.context);
    seal::Encryptor encryptor(*context.context, *context.secret_key);
    std::vector<seal::Ciphertext> ciphertexts(packing.ciphertext_count);
    const double scale = std::ldexp(
        1.0, context.parameters.default_scale_bits);
    for (std::size_t i = 0; i < ciphertexts.size(); ++i) {
        seal::Plaintext plain;
        encoder.encode(slots[i], LevelId(context, level), scale, plain);
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
    spec.planned_scale_bits = context.parameters.default_scale_bits;
    return {context.profile, std::move(ciphertexts), spec, packing};
}

std::vector<double> Decrypt(const hera::SealCipherTensor &tensor,
                            const hera::HeContext &context)
{
    seal::Decryptor decryptor(*context.context, *context.secret_key);
    seal::CKKSEncoder encoder(*context.context);
    std::vector<std::vector<std::complex<double>>> decoded(
        tensor.ciphertexts.size());
    for (std::size_t i = 0; i < tensor.ciphertexts.size(); ++i) {
        seal::Plaintext plain;
        decryptor.decrypt(tensor.ciphertexts[i], plain);
        encoder.decode(plain, decoded[i]);
    }
    std::vector<double> result(tensor.physical_packing.logical_locations.size());
    for (std::size_t logical = 0; logical < result.size(); ++logical) {
        const auto &location = tensor.physical_packing.Locations(logical).front();
        const auto value = decoded[location.ciphertext][location.slot];
        result[logical] = location.component == hera::ComplexLane::kReal
            ? value.real() : value.imag();
    }
    return result;
}

double MaxError(const std::vector<double> &left,
                const std::vector<double> &right)
{
    Require(left.size() == right.size(), "error vector sizes differ");
    double result = 0.0;
    for (std::size_t i = 0; i < left.size(); ++i)
        result = std::max(result, std::abs(left[i] - right[i]));
    return result;
}

std::uint64_t EncodeShare(double value)
{
    const auto scaled = static_cast<std::int64_t>(
        std::llround(std::ldexp(value, 13)));
    return static_cast<std::uint64_t>(scaled) &
           ((std::uint64_t{1} << 43) - 1);
}

double DecodeShare(std::uint64_t value)
{
    constexpr std::uint64_t modulus = std::uint64_t{1} << 43;
    value &= modulus - 1;
    const auto signed_value = value >= modulus / 2
        ? static_cast<std::int64_t>(value - modulus)
        : static_cast<std::int64_t>(value);
    return std::ldexp(static_cast<double>(signed_value), -13);
}

void TestPackingAndLevels()
{
    auto config = MakeConfig();
    Require(hera::ValidateHeraConfig(config).ok(), "test config is invalid");
    const auto &layer = config.layers.front();
    const auto &profile = config.he_parameter_profiles.at("shared");
    Require(hera::AttentionLn2LevelPlanner::Validate(profile).ok(),
            "valid attention/LN2 level plan rejected");
    auto short_chain = profile;
    short_chain.coefficient_modulus_bits.pop_back();
    Require(!hera::AttentionLn2LevelPlanner::Validate(short_chain).ok(),
            "short attention/LN2 chain accepted");
    auto av = hera::AttentionLn2PackingPlanner::AttentionValue(
        layer, profile.polynomial_degree / 2);
    Require(av.ok(), av.status().message());
    Require(av.value().probability_packing.Validate().ok() &&
                av.value().output_packing.Validate().ok() &&
                av.value().probability_slot_map.Validate().ok(),
            "attention packing plan is invalid");
    for (const auto &locations : av.value().probability_packing.logical_locations)
        Require(locations.size() == 1,
                "Softmax A physical mapping is not bijective");
    for (const auto kv_heads : {std::size_t{2}, std::size_t{1}}) {
        auto variant = layer;
        variant.tokens = 3;
        variant.q_heads = 2;
        variant.kv_heads = kv_heads;
        variant.value_head_width = 12;
        auto variant_plan = hera::AttentionLn2PackingPlanner::AttentionValue(
            variant, 32);
        Require(variant_plan.ok(),
                "MHA/GQA non-power-of-two A*V packing rejected");
        Require(variant_plan.value().probability_packing.Validate().ok() &&
                    variant_plan.value().output_packing.Validate().ok(),
                "MHA/GQA tail A*V packing is invalid");
        // 32 slots / (3 tokens * 2 heads) gives five value lanes. Dimension
        // five must therefore be carried by the imaginary component paired
        // with real dimension zero, exactly as softmax_vh complex_pairs does.
        const auto &real = variant_plan.value().output_packing.Locations(0);
        const auto &imag = variant_plan.value().output_packing.Locations(5);
        Require(!real.empty() && !imag.empty() &&
                    real.front().ciphertext == imag.front().ciphertext &&
                    real.front().slot == imag.front().slot &&
                    real.front().component == hera::ComplexLane::kReal &&
                    imag.front().component ==
                        hera::ComplexLane::kImaginary,
                "A*V output does not use complex-pair value columns");
    }
    auto tiled = layer;
    tiled.tokens = 4;
    tiled.q_heads = 10;
    tiled.kv_heads = 1;
    tiled.value_head_width = 3;
    auto tiled_plan = hera::AttentionLn2PackingPlanner::AttentionValue(
        tiled, 32);
    Require(tiled_plan.ok() &&
                tiled_plan.value().probability_packing.ciphertext_count > 1,
            "head-tiled A*V packing was not generated");
}

void TestHePath()
{
    auto config = MakeConfig();
    const auto &layer = config.layers.front();
    const auto &profile = config.he_parameter_profiles.at("shared");
    hera::HeContextRegistry client;
    Require(client.CreateClientProfiles(config).ok(),
            "cannot create client profiles");
    auto steps = hera::SealSciAttentionLn2Executor::RequiredRotationSteps(
        layer, profile);
    Require(steps.ok(), steps.status().message());
    Require(client.CompileRotationKeys("shared", steps.value()).ok(),
            "cannot compile attention/LN2 rotation keys");
    auto bundle = client.ExportPublicBundle();
    Require(bundle.ok(), bundle.status().message());
    hera::HeContextRegistry server;
    Require(server.CreateServerProfiles(config, bundle.value()).ok(),
            "cannot create server profiles");
    const auto &client_context = client.Require(
        "shared", hera::PartyRole::kClient);
    const auto &server_context = server.Require(
        "shared", hera::PartyRole::kServer);
    Require(!server_context.has_secret_key(),
            "server unexpectedly owns the secret key");

    // Force one logical feature into the imaginary component and verify the
    // coefficient-domain (-i) realification path independently of Wo.
    auto imaginary_layer = layer;
    imaginary_layer.model_width = client_context.slot_count() / 2 + 1;
    auto imaginary_plan =
        hera::AttentionLn2PackingPlanner::ComplexStrictColumns(
            {{imaginary_layer.tokens, imaginary_layer.model_width}},
            client_context.slot_count());
    Require(imaginary_plan.ok(), imaginary_plan.status().message());
    std::vector<double> imaginary_values(
        imaginary_layer.tokens * imaginary_layer.model_width, 0.0);
    imaginary_values[imaginary_layer.model_width - 1] = 0.33;
    imaginary_values[2 * imaginary_layer.model_width - 1] = -0.22;
    auto imaginary_ct = Encrypt(
        imaginary_values, imaginary_plan.value(), client_context, 6,
        hera::HeRegionId::kRegion4);
    auto realified_imaginary = hera::ComplexRealifyKernel::Evaluate(
        imaginary_ct, imaginary_layer, server_context);
    Require(realified_imaginary.ok(),
            realified_imaginary.status().message());
    const auto decoded_imaginary = Decrypt(
        realified_imaginary.value(), client_context);
    Require(MaxError(decoded_imaginary, imaginary_values) < 0.01,
            "imaginary-component realification is incorrect: actual=" +
            std::to_string(decoded_imaginary[
                imaginary_layer.model_width - 1]) + "," +
            std::to_string(decoded_imaginary[
                2 * imaginary_layer.model_width - 1]));
    auto imaginary_statistics = hera::LayerNormStatisticsKernel::Evaluate(
        realified_imaginary.value(), imaginary_layer, server_context);
    Require(imaginary_statistics.ok(),
            imaginary_statistics.status().message());
    std::vector<double> imaginary_centered(imaginary_values.size());
    std::vector<double> imaginary_variance(imaginary_layer.tokens);
    for (std::size_t token = 0; token < imaginary_layer.tokens; ++token) {
        double mean = 0.0;
        for (std::size_t feature = 0;
             feature < imaginary_layer.model_width; ++feature)
            mean += imaginary_values[
                token * imaginary_layer.model_width + feature];
        mean /= static_cast<double>(imaginary_layer.model_width);
        for (std::size_t feature = 0;
             feature < imaginary_layer.model_width; ++feature) {
            const auto index = token * imaginary_layer.model_width + feature;
            imaginary_centered[index] = imaginary_values[index] - mean;
            imaginary_variance[token] +=
                imaginary_centered[index] * imaginary_centered[index];
        }
        imaginary_variance[token] /=
            static_cast<double>(imaginary_layer.model_width);
        imaginary_variance[token] +=
            imaginary_layer.attention_norm_epsilon;
    }
    Require(MaxError(Decrypt(
                imaginary_statistics.value().centered, client_context),
                imaginary_centered) < 0.02 &&
                MaxError(Decrypt(
                    imaginary_statistics.value().variance, client_context),
                    imaginary_variance) < 0.02,
            "mixed real/imaginary tail LayerNorm is incorrect");

    auto av_plan = hera::AttentionLn2PackingPlanner::AttentionValue(
        layer, client_context.slot_count());
    Require(av_plan.ok(), av_plan.status().message());
    auto v_plan = hera::MultiHeadPackingPlan::ProjectionOutput(
        layer, hera::Layout::kMultiHeadV, client_context.slot_count());
    Require(v_plan.ok(), v_plan.status().message());
    // [token, query-head, source-token]
    const std::vector<double> probabilities{
        0.75, 0.25, 0.20, 0.80,
        0.60, 0.40, 0.10, 0.90};
    // [source-token, kv-head, value-dimension]
    const std::vector<double> values{1.0, -0.5, 0.25, 2.0};
    auto probability_ct = Encrypt(
        probabilities, av_plan.value().probability_packing,
        client_context, 2, hera::HeRegionId::kRegion2);
    auto value_ct = Encrypt(values, v_plan.value(), client_context, 2,
                            hera::HeRegionId::kRegion2);
    hera::AttentionLn2ExecutionReport report;
    auto attention = hera::MultiHeadAttentionValueKernel::Evaluate(
        probability_ct, value_ct, layer, server_context, &report);
    Require(attention.ok(), attention.status().message());
    Require(ActualLevel(client_context,
                        attention.value().ciphertexts.front().parms_id()) == 5,
            "A*V did not end at level 5");
    const std::vector<double> expected_attention{
        0.8125, 0.125, 0.40, 1.50,
        0.70, 0.50, 0.325, 1.75};
    Require(MaxError(Decrypt(attention.value(), client_context),
                     expected_attention) < 0.025,
            "A*V numerical error exceeds tolerance");

    // Exercise the actual complex-pair arithmetic, rather than only checking
    // metadata. With 2048 slots, 3 tokens x 300 heads gives two lanes and two
    // giant groups (the second one is a one-diagonal tail). Value dimensions
    // 0/1 and 2/3 therefore occupy the real and imaginary components. MQA also
    // verifies that physical query-head replication survives all three BSGS
    // phases.
    auto complex_layer = layer;
    complex_layer.tokens = 3;
    complex_layer.q_heads = 300;
    complex_layer.kv_heads = 1;
    complex_layer.value_head_width = 4;
    auto complex_a_plan = hera::AttentionLn2PackingPlanner::AttentionValue(
        complex_layer, client_context.slot_count());
    auto complex_v_plan = hera::MultiHeadPackingPlan::ProjectionOutput(
        complex_layer, hera::Layout::kMultiHeadV,
        client_context.slot_count());
    Require(complex_a_plan.ok() && complex_v_plan.ok(),
            "cannot build complex-pair A*V test packing");
    std::vector<double> complex_probabilities(
        complex_layer.tokens * complex_layer.q_heads * complex_layer.tokens);
    for (std::size_t row = 0; row < complex_layer.tokens; ++row)
        for (std::size_t head = 0; head < complex_layer.q_heads; ++head) {
            const auto begin =
                (row * complex_layer.q_heads + head) * complex_layer.tokens;
            complex_probabilities[begin] = 0.20;
            complex_probabilities[begin + 1] = 0.30;
            complex_probabilities[begin + 2] = 0.50;
        }
    const std::vector<double> complex_values{
        1.0, 0.1, -1.0, -0.1,
        3.0, 0.3, -3.0, -0.3,
        5.0, 0.5, -5.0, -0.5};
    auto complex_a = Encrypt(
        complex_probabilities,
        complex_a_plan.value().probability_packing,
        client_context, 2, hera::HeRegionId::kRegion2);
    auto complex_v = Encrypt(
        complex_values, complex_v_plan.value(), client_context, 2,
        hera::HeRegionId::kRegion2);
    hera::AttentionLn2ExecutionReport complex_report;
    auto complex_attention = hera::MultiHeadAttentionValueKernel::Evaluate(
        complex_a, complex_v, complex_layer, server_context,
        &complex_report);
    Require(complex_attention.ok(), complex_attention.status().message());
    const auto complex_actual = Decrypt(
        complex_attention.value(), client_context);
    std::vector<double> complex_expected(complex_actual.size());
    for (std::size_t row = 0; row < complex_layer.tokens; ++row)
        for (std::size_t head = 0; head < complex_layer.q_heads; ++head) {
            const auto begin =
                (row * complex_layer.q_heads + head) * 4;
            complex_expected[begin] = 3.6;
            complex_expected[begin + 1] = 0.36;
            complex_expected[begin + 2] = -3.6;
            complex_expected[begin + 3] = -0.36;
        }
    Require(MaxError(complex_actual, complex_expected) < 0.025,
            "complex-pair A*V real/imaginary arithmetic is incorrect");
    Require(complex_report.ciphertext_multiplications == 3 &&
                complex_report.relinearizations == 2,
            "A*V did not lazily accumulate/relinearize once per giant group");

    auto parallel_layer = complex_layer;
    parallel_layer.value_head_width = 8;
    auto parallel_v_plan = hera::MultiHeadPackingPlan::ProjectionOutput(
        parallel_layer, hera::Layout::kMultiHeadV,
        client_context.slot_count());
    Require(parallel_v_plan.ok(), "cannot build multi-group V packing");
    std::vector<double> parallel_values(
        parallel_layer.tokens * parallel_layer.value_head_width);
    const std::vector<double> basis{1.0, 0.1, -1.0, -0.1,
                                    2.0, 0.2, -2.0, -0.2};
    for (std::size_t token = 0; token < parallel_layer.tokens; ++token)
        for (std::size_t feature = 0;
             feature < parallel_layer.value_head_width; ++feature)
            parallel_values[token * parallel_layer.value_head_width + feature] =
                basis[feature] * (2.0 * token + 1.0);
    auto parallel_v = Encrypt(
        parallel_values, parallel_v_plan.value(), client_context, 2,
        hera::HeRegionId::kRegion2);
    hera::AttentionLn2ExecutionReport parallel_report;
    auto parallel_attention = hera::MultiHeadAttentionValueKernel::Evaluate(
        complex_a, parallel_v, parallel_layer, server_context,
        &parallel_report, 2);
    Require(parallel_attention.ok(),
            parallel_attention.status().message());
    std::vector<double> parallel_expected(
        parallel_layer.tokens * parallel_layer.q_heads *
        parallel_layer.value_head_width);
    for (std::size_t row = 0; row < parallel_layer.tokens; ++row)
        for (std::size_t head = 0; head < parallel_layer.q_heads; ++head)
            for (std::size_t feature = 0;
                 feature < parallel_layer.value_head_width; ++feature)
                parallel_expected[
                    (row * parallel_layer.q_heads + head) *
                        parallel_layer.value_head_width + feature] =
                    3.6 * basis[feature];
    Require(MaxError(Decrypt(parallel_attention.value(), client_context),
                     parallel_expected) < 0.025,
            "parallel multi-group A*V differs from plaintext reference");

    hera::DenseMatrixParameter wo;
    wo.spec = {"Wo", 4, 4, true};
    wo.row_major.assign(16, 0.0);
    for (std::size_t i = 0; i < 4; ++i) wo.row_major[i * 4 + i] = 1.0;
    hera::DenseVectorParameter bo;
    bo.spec = {"bo", 4};
    bo.values = {0.01, -0.02, 0.03, -0.04};
    auto projected = hera::AttentionOutputProjectionKernel::Evaluate(
        attention.value(), wo, bo, layer, server_context, &report);
    Require(projected.ok(), projected.status().message());
    Require(ActualLevel(client_context,
                        projected.value().ciphertexts.front().parms_id()) == 6,
            "Wo did not end at level 6");

    auto residual_plan =
        hera::AttentionLn2PackingPlanner::ComplexStrictColumns(
            {{layer.tokens, layer.model_width}}, client_context.slot_count());
    Require(residual_plan.ok(), residual_plan.status().message());
    const std::vector<double> residual{
        0.10, -0.10, 0.20, -0.20,
        0.05, -0.05, 0.15, -0.15};
    auto residual_ct = Encrypt(residual, residual_plan.value(), client_context,
                               0, hera::HeRegionId::kRegion2);
    auto added = hera::AttentionResidualKernel::Add(
        projected.take_value(), residual_ct, server_context, &report);
    Require(added.ok(), added.status().message());
    Require(report.modulus_switches > 0,
            "higher-level residual was not mod-switched");
    auto real = hera::ComplexRealifyKernel::Evaluate(
        added.value(), layer, server_context, &report);
    Require(real.ok(), real.status().message());

    std::vector<double> expected = expected_attention;
    for (std::size_t token = 0; token < layer.tokens; ++token)
        for (std::size_t feature = 0; feature < layer.model_width; ++feature)
            expected[token * layer.model_width + feature] +=
                bo.values[feature] +
                residual[token * layer.model_width + feature];
    const auto actual_real = Decrypt(real.value(), client_context);
    Require(MaxError(actual_real, expected) < 0.05,
            "Wo/residual/realify numerical error exceeds tolerance: actual=" +
            std::to_string(actual_real.front()) + " expected=" +
            std::to_string(expected.front()));

    auto statistics = hera::LayerNormStatisticsKernel::Evaluate(
        real.value(), layer, server_context, &report);
    Require(statistics.ok(), statistics.status().message());
    Require(ActualLevel(client_context,
        statistics.value().centered.ciphertexts.front().parms_id()) == 7,
        "centered output did not end at level 7");
    Require(ActualLevel(client_context,
        statistics.value().variance.ciphertexts.front().parms_id()) == 9,
        "variance output did not end at terminal level 9");
    std::vector<double> expected_centered(expected.size());
    std::vector<double> expected_variance(layer.tokens);
    for (std::size_t token = 0; token < layer.tokens; ++token) {
        double mean = 0.0;
        for (std::size_t feature = 0; feature < layer.model_width; ++feature)
            mean += expected[token * layer.model_width + feature];
        mean /= static_cast<double>(layer.model_width);
        for (std::size_t feature = 0; feature < layer.model_width; ++feature) {
            const auto index = token * layer.model_width + feature;
            expected_centered[index] = expected[index] - mean;
            expected_variance[token] +=
                expected_centered[index] * expected_centered[index];
        }
        expected_variance[token] /= static_cast<double>(layer.model_width);
        expected_variance[token] += layer.attention_norm_epsilon;
    }
    const auto actual_centered = Decrypt(
        statistics.value().centered, client_context);
    Require(MaxError(actual_centered, expected_centered) < 0.05,
            "LayerNorm centered error exceeds tolerance");
    const auto actual_variance = Decrypt(
        statistics.value().variance, client_context);
    const auto variance_error = MaxError(actual_variance, expected_variance);
    Require(variance_error < 0.05,
            "LayerNorm variance error exceeds tolerance: actual=" +
            std::to_string(actual_variance.front()) + "," +
            std::to_string(actual_variance.back()) + " expected=" +
            std::to_string(expected_variance.front()) + "," +
            std::to_string(expected_variance.back()) + " error=" +
            std::to_string(variance_error) + " centered=" +
            std::to_string(actual_centered[0]) + "," +
            std::to_string(actual_centered[1]) + "," +
            std::to_string(actual_centered[2]) + "," +
            std::to_string(actual_centered[3]));
    Require(report.ciphertext_multiplications > 0 && report.rescales > 0 &&
                report.rotations > 0 && report.conjugations > 0,
            "attention/LN2 HE operation report is incomplete");
}

void TestParallelWoProjection()
{
    auto config = MakeConfig();
    auto &layer = config.layers.front();
    layer.tokens = 8;
    layer.model_width = 513;
    layer.kv_heads = layer.q_heads;
    const auto &profile = config.he_parameter_profiles.at("shared");
    auto av_plan = hera::AttentionLn2PackingPlanner::AttentionValue(
        layer, profile.polynomial_degree / 2);
    auto output_plan = hera::AttentionLn2PackingPlanner::ComplexStrictColumns(
        {{layer.tokens, layer.model_width}}, profile.polynomial_degree / 2);
    Require(av_plan.ok() && output_plan.ok(),
            "parallel Wo packing failed");
    Require(output_plan.value().ciphertext_count == 2,
            "parallel Wo test needs two output groups");
    hera::HeContextRegistry client;
    Require(client.CreateClientProfiles(config).ok(),
            "parallel Wo client context failed");
    // Only feature 0 and feature 512 have nonzero weights. Their source-to-
    // destination rotations for eight tokens are exactly 0 through 7.
    Require(client.CompileRotationKeys("shared", {0, 1, 2, 3, 4, 5, 6, 7}).ok(),
            "parallel Wo rotation keys failed");
    auto bundle = client.ExportPublicBundle();
    Require(bundle.ok(), bundle.status().message());
    hera::HeContextRegistry server;
    Require(server.CreateServerProfiles(config, bundle.value()).ok(),
            "parallel Wo server context failed");
    const auto &client_context = client.Require("shared", hera::PartyRole::kClient);
    const auto &server_context = server.Require("shared", hera::PartyRole::kServer);
    std::vector<double> values(
        layer.tokens * layer.q_heads * layer.value_head_width);
    for (std::size_t i = 0; i < values.size(); ++i)
        values[i] = (static_cast<int>(i % 7) - 3) * 0.01;
    auto input = Encrypt(values, av_plan.value().output_packing,
                         client_context, 5, hera::HeRegionId::kRegion4);
    std::vector<double> weights(
        layer.q_heads * layer.value_head_width * layer.model_width, 0.0);
    weights[0] = 0.1;
    weights[512] = -0.15;
    hera::DenseMatrixParameter wo{
        {"Wo", layer.q_heads * layer.value_head_width,
         layer.model_width, true}, weights};
    hera::AttentionLn2ExecutionReport serial_report, parallel_report;
    auto serial = hera::AttentionOutputProjectionKernel::Evaluate(
        input, wo, std::nullopt, layer, server_context, &serial_report, 1);
    auto parallel = hera::AttentionOutputProjectionKernel::Evaluate(
        input, wo, std::nullopt, layer, server_context, &parallel_report, 2);
    Require(serial.ok() && parallel.ok(),
            serial.ok() ? parallel.status().message() : serial.status().message());
    Require(MaxError(Decrypt(serial.value(), client_context),
                     Decrypt(parallel.value(), client_context)) < 1e-4,
            "parallel Wo projection differs from serial");
    Require(serial_report.rotations == parallel_report.rotations &&
                serial_report.ciphertext_plain_multiplications ==
                    parallel_report.ciphertext_plain_multiplications &&
                serial_report.rescales == parallel_report.rescales,
            "parallel Wo projection changed operation counts");

    hera::SealPlainParameterStore parameters;
    parameters.BeginPlaintextPreparation();
    {
        hera::ModelPlaintextCacheScope scope(&parameters, layer.id);
        auto metadata = input;
        for (auto &ct : metadata.ciphertexts) {
            const auto id = ct.parms_id();
            const auto scale = ct.scale();
            ct = seal::Ciphertext();
            ct.parms_id() = id;
            ct.scale() = scale;
        }
        auto prepared = hera::AttentionOutputProjectionKernel::Evaluate(
            metadata, wo, std::nullopt, layer, server_context);
        Require(prepared.ok(), prepared.status().message());
    }
    parameters.FreezePlaintextCache();
    Require(parameters.prepared_projection_count() == 1,
            "Prepare did not compile Wo projection");
    hera::AttentionLn2ExecutionReport cached_report;
    hera::Result<hera::SealCipherTensor> cached =
        hera::Status::Internal("cached Wo was not evaluated");
    {
        hera::ModelPlaintextCacheScope scope(&parameters, layer.id);
        cached = hera::AttentionOutputProjectionKernel::Evaluate(
            input, wo, std::nullopt, layer, server_context,
            &cached_report, 2);
    }
    Require(cached.ok(), cached.status().message());
    Require(parameters.prepared_projection_hits() == 1,
            "online Wo did not use its prepared projection plan");
    Require(MaxError(Decrypt(serial.value(), client_context),
                     Decrypt(cached.value(), client_context)) < 1e-4,
            "prepared Wo differs from the direct HE path");
    Require(cached_report.rotations == serial_report.rotations &&
                cached_report.ciphertext_plain_multiplications ==
                    serial_report.ciphertext_plain_multiplications &&
                cached_report.rescales == serial_report.rescales,
            "prepared Wo changed HE operation counts");
}

struct IntegrationPacket {
    int ok{};
    std::uint64_t centered[8]{};
    std::uint64_t inverse[2]{};
};

void RunIntegratedParty(
    hera::PartyRole role, const hera::HeraConfig &config,
    hera::HeContextRegistry &registry,
    const std::optional<hera::SealCipherTensor> &value,
    const std::optional<hera::SealCipherTensor> &residual,
    int control_port, int worker_port, int nonlinear_port,
    IntegrationPacket *packet, std::string *error)
{
    try {
        auto transport = hera::MakeNetIoTransport(
            role, role == hera::PartyRole::kClient ? "127.0.0.1" : "",
            control_port);
        hera::CkksMpcConverter converter(
            role, {43, 13, 40, 20, true}, registry, transport,
            hera::MakeDeterministicRandomSourceForTesting(
                role == hera::PartyRole::kClient
                    ? std::vector<std::uint8_t>{11, 13, 17}
                    : std::vector<std::uint8_t>{19, 23, 29}),
            "127.0.0.1", worker_port, 1);
        hera::MpcLayerNormRuntime nonlinear(
            role, "127.0.0.1", nonlinear_port);
        hera::SealCipherTensorStore he_store;
        hera::SealPlainParameterStore parameter_store;
        hera::CryptoTensorStore mpc_store;
        constexpr hera::TensorId probability_id = 8100000;
        constexpr hera::TensorId value_id = 8100001;
        constexpr hera::TensorId residual_id = 8100002;
        const auto &layer = config.layers.front();
        const std::vector<double> probabilities{
            0.75, 0.25, 0.20, 0.80,
            0.60, 0.40, 0.10, 0.90};
        std::vector<std::uint64_t> local_probability(probabilities.size());
        if (role == hera::PartyRole::kClient)
            std::transform(probabilities.begin(), probabilities.end(),
                           local_probability.begin(), EncodeShare);
        hera::MpcShareTensor probability_payload{
            std::move(local_probability),
            {{layer.tokens, layer.q_heads, layer.tokens}},
            {43, 13, false}, hera::ComplexComponent::kReal};
        Require(mpc_store.PutMpcShare(
                    probability_id, std::move(probability_payload)).ok(),
                "cannot register Softmax probability shares");

        const auto schema = hera::DeriveWeightSchema(layer).value();
        hera::LayerWeightHandles weights;
        weights.layer_id = layer.id;
        weights.wo = {9100001, schema.wo};
        const auto bo_spec = *std::find_if(
            schema.vectors.begin(), schema.vectors.end(),
            [](const hera::VectorSpec &spec) { return spec.name == "bo"; });
        hera::PlainParameterHandle bo_handle{9100002, bo_spec};
        weights.plain_parameters.push_back(bo_handle);
        if (role == hera::PartyRole::kServer) {
            Require(value && residual, "server HE inputs are absent");
            Require(he_store.Put(value_id, *value).ok() &&
                        he_store.Put(residual_id, *residual).ok(),
                    "cannot register attention/LN2 HE inputs");
            std::vector<double> wo(16, 0.0);
            for (std::size_t i = 0; i < 4; ++i) wo[i * 4 + i] = 1.0;
            Require(parameter_store.PutMatrix(weights.wo, wo).ok(),
                    "cannot register Wo");
            Require(parameter_store.PutVector(
                        bo_handle, {0.01, -0.02, 0.03, -0.04}).ok(),
                    "cannot register bo");
        }
        hera::TensorSpec probability_spec;
        probability_spec.shape = {{layer.tokens, layer.q_heads,
                                   layer.tokens}};
        probability_spec.domain = hera::Domain::kMpcArithmeticShare;
        probability_spec.layout = hera::Layout::kSoftmaxRows;
        probability_spec.numeric = {43, 13, false};
        hera::TensorSpec value_spec;
        value_spec.shape = {{layer.tokens, layer.kv_heads,
                             layer.value_head_width}};
        value_spec.domain = hera::Domain::kHeCiphertext;
        value_spec.layout = hera::Layout::kMultiHeadV;
        value_spec.numeric = {43, 13, true};
        value_spec.he_region = hera::HeRegionId::kRegion2;
        value_spec.he_parameter_profile = "shared";
        value_spec.planned_level = 2;
        value_spec.planned_scale_bits = 44;
        hera::TensorSpec residual_spec;
        residual_spec.shape = {{layer.tokens, layer.model_width}};
        residual_spec.domain = hera::Domain::kHeCiphertext;
        residual_spec.layout = hera::Layout::kComplexStrictColumn;
        residual_spec.numeric = {43, 13, true};
        residual_spec.he_region = hera::HeRegionId::kRegion2;
        residual_spec.he_parameter_profile = "shared";
        residual_spec.planned_level = 0;
        residual_spec.planned_scale_bits = 44;
        hera::SealSciAttentionLn2Executor executor(
            config, layer, registry, he_store, parameter_store, mpc_store,
            converter, nonlinear);
        hera::AttentionLn2ExecutionReport report;
        auto output = executor.Evaluate(
            {{probability_id, probability_spec}, {value_id, value_spec},
             {residual_id, residual_spec}}, weights, &report);
        Require(output.ok(), output.status().message());
        auto centered = mpc_store.GetMpcShare(
            output.value().centered_shares.id);
        auto inverse = mpc_store.GetMpcShare(
            output.value().inverse_stddev_shares.id);
        Require(centered.ok() && inverse.ok(),
                "integrated attention/LN2 outputs are absent");
        Require(centered.value().local_share.size() == 8 &&
                    inverse.value().local_share.size() == 2,
                "integrated attention/LN2 output shapes are wrong");
        std::copy(centered.value().local_share.begin(),
                  centered.value().local_share.end(), packet->centered);
        std::copy(inverse.value().local_share.begin(),
                  inverse.value().local_share.end(), packet->inverse);
        if (role == hera::PartyRole::kServer)
            Require(report.mpc_to_ckks_calls == 1 &&
                        report.ckks_to_mpc_calls == 2 &&
                        report.reciprocal_square_roots == 2,
                    "integrated operation report is incomplete");
        packet->ok = 1;
    } catch (const std::exception &exception) {
        *error = exception.what();
    }
}

void TestIntegratedExecutor()
{
    auto config = MakeConfig();
    const auto &layer = config.layers.front();
    const auto &profile = config.he_parameter_profiles.at("shared");
    hera::HeContextRegistry client_registry;
    Require(client_registry.CreateClientProfiles(config).ok(),
            "cannot create integrated client profiles");
    auto steps = hera::SealSciAttentionLn2Executor::RequiredRotationSteps(
        layer, profile);
    Require(steps.ok() && client_registry.CompileRotationKeys(
                "shared", steps.value()).ok(),
            "cannot compile integrated rotation keys");
    auto bundle = client_registry.ExportSerializedPublicBundle();
    Require(bundle.ok(), bundle.status().message());
    hera::HeContextRegistry server_registry;
    Require(server_registry.CreateServerProfiles(
                config, bundle.value()).ok(),
            "cannot create integrated server profiles");
    const auto &client_context = client_registry.Require(
        "shared", hera::PartyRole::kClient);
    auto value_plan = hera::MultiHeadPackingPlan::ProjectionOutput(
        layer, hera::Layout::kMultiHeadV, client_context.slot_count());
    auto residual_plan =
        hera::AttentionLn2PackingPlanner::ComplexStrictColumns(
            {{layer.tokens, layer.model_width}}, client_context.slot_count());
    Require(value_plan.ok() && residual_plan.ok(),
            "cannot compile integrated HE packing");
    auto value = Encrypt({1.0, -0.5, 0.25, 2.0}, value_plan.value(),
                         client_context, 2, hera::HeRegionId::kRegion2);
    const std::vector<double> residual_values{
        0.10, -0.10, 0.20, -0.20,
        0.05, -0.05, 0.15, -0.15};
    auto residual = Encrypt(residual_values, residual_plan.value(),
                            client_context, 0,
                            hera::HeRegionId::kRegion2);

    int pipe_fd[2];
    Require(pipe(pipe_fd) == 0, "cannot create integration result pipe");
    const pid_t child = fork();
    Require(child >= 0, "cannot fork integrated attention/LN2 client");
    if (child == 0) {
        close(pipe_fd[0]);
        IntegrationPacket packet;
        std::string error;
        RunIntegratedParty(
            hera::PartyRole::kClient, config, client_registry,
            std::nullopt, std::nullopt, 24520, 24530, 24540,
            &packet, &error);
        if (!error.empty())
            std::cerr << "integrated client: " << error << '\n';
        const auto written = write(pipe_fd[1], &packet, sizeof(packet));
        close(pipe_fd[1]);
        _exit(error.empty() && written == static_cast<ssize_t>(sizeof(packet))
                  ? 0 : 2);
    }
    close(pipe_fd[1]);
    IntegrationPacket server_packet;
    std::string server_error;
    RunIntegratedParty(
        hera::PartyRole::kServer, config, server_registry, value, residual,
        24520, 24530, 24540, &server_packet, &server_error);
    if (!server_error.empty()) {
        kill(child, SIGTERM);
        int failed_status = 0;
        waitpid(child, &failed_status, 0);
        close(pipe_fd[0]);
        throw std::runtime_error(server_error);
    }
    IntegrationPacket client_packet;
    const auto received = read(pipe_fd[0], &client_packet,
                               sizeof(client_packet));
    close(pipe_fd[0]);
    int status = 0;
    waitpid(child, &status, 0);
    Require(received == static_cast<ssize_t>(sizeof(client_packet)) &&
                client_packet.ok && WIFEXITED(status) &&
                WEXITSTATUS(status) == 0,
            "integrated attention/LN2 client failed");

    const std::vector<double> attention{
        0.8125, 0.125, 0.40, 1.50,
        0.70, 0.50, 0.325, 1.75};
    const std::vector<double> bias{0.01, -0.02, 0.03, -0.04};
    std::vector<double> expected_centered(8);
    std::vector<double> expected_inverse(2);
    for (std::size_t token = 0; token < 2; ++token) {
        std::vector<double> row(4);
        double mean = 0.0;
        for (std::size_t feature = 0; feature < 4; ++feature) {
            const auto index = token * 4 + feature;
            row[feature] = attention[index] + bias[feature] +
                           residual_values[index];
            mean += row[feature];
        }
        mean /= 4.0;
        double variance = 0.0;
        for (std::size_t feature = 0; feature < 4; ++feature) {
            const auto index = token * 4 + feature;
            expected_centered[index] = row[feature] - mean;
            variance += expected_centered[index] * expected_centered[index];
        }
        expected_inverse[token] = 1.0 /
            std::sqrt(variance / 4.0 + layer.attention_norm_epsilon);
    }
    constexpr std::uint64_t ring_mask = (std::uint64_t{1} << 43) - 1;
    for (std::size_t i = 0; i < expected_centered.size(); ++i) {
        const auto reconstructed =
            (server_packet.centered[i] + client_packet.centered[i]) &
            ring_mask;
        Require(std::abs(DecodeShare(reconstructed) - expected_centered[i]) <
                    0.06,
                "integrated centered share mismatch");
    }
    for (std::size_t i = 0; i < expected_inverse.size(); ++i) {
        const auto reconstructed =
            (server_packet.inverse[i] + client_packet.inverse[i]) & ring_mask;
        const auto actual = DecodeShare(reconstructed);
        Require(std::abs(actual - expected_inverse[i]) < 0.08,
                "integrated reciprocal-sqrt share mismatch: actual=" +
                std::to_string(actual) + " expected=" +
                std::to_string(expected_inverse[i]));
    }
}

} // namespace

int main()
{
    try {
        TestPackingAndLevels();
        TestHePath();
        TestParallelWoProjection();
        TestIntegratedExecutor();
        std::cout << "ATTENTION_LN2_HE_READY=YES\n"
                  << "ATTENTION_LN2_STATUS=PASS\n";
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << "ATTENTION_LN2_STATUS=FAIL\n"
                  << "ATTENTION_LN2_ERROR=" << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
