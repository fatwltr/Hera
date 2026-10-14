#include "hera/attention_ln2.h"
#include "hera/detail/packing_utils.h"
#include "hera/detail/parallel_ranges.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <set>
#include <stdexcept>
#include <sys/wait.h>
#include <signal.h>
#include <unistd.h>
#include <utility>

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
    const auto &a_real = av.value().probability_packing.Locations(0).front();
    const auto &a_imag = av.value().probability_packing.Locations(1).front();
    Require(a_real.ciphertext == a_imag.ciphertext &&
                a_real.slot == a_imag.slot &&
                a_real.component == hera::ComplexLane::kReal &&
                a_imag.component == hera::ComplexLane::kImaginary,
            "consecutive A diagonals are not complex-paired");
    const auto &assignment = av.value().probability_slot_map.assignments[
        a_real.ciphertext * profile.polynomial_degree / 2 + a_real.slot];
    Require(assignment.real_logical_index == 0 &&
                assignment.imaginary_logical_index == 1 &&
                std::find(av.value().required_rotation_steps.begin(),
                          av.value().required_rotation_steps.end(), 0) !=
                    av.value().required_rotation_steps.end(),
            "A conversion map or conjugation key is missing");
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
        // Padded token stride is four, giving four physical lanes. Dimension
        // four must therefore be carried by the imaginary component paired
        // with real dimension zero, exactly as softmax_vh complex_pairs does.
        const auto &real = variant_plan.value().output_packing.Locations(0);
        const auto &imag = variant_plan.value().output_packing.Locations(4);
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

    auto bert = layer;
    bert.tokens = 128;
    bert.model_width = 768;
    bert.q_heads = bert.kv_heads = 12;
    bert.qk_head_width = bert.value_head_width = 64;
    bert.ffn_width = 3072;
    auto bert_av = hera::AttentionLn2PackingPlanner::AttentionValue(bert, 16384);
    auto bert_v = hera::MultiHeadPackingPlan::ProjectionOutput(
        bert, hera::Layout::kMultiHeadV, 16384);
    auto bert_q = hera::MultiHeadPackingPlan::ProjectionOutput(
        bert, hera::Layout::kMultiHeadQ, 16384);
    Require(bert_av.ok() && bert_v.ok() && bert_q.ok(),
            "BERT-base power-of-two packing failed");
    Require(bert_av.value().probability_packing.ciphertext_count == 8 &&
                bert_v.value().ciphertext_count == 4 &&
                bert_av.value().output_packing.ciphertext_count == 4,
            "BERT-base did not use eight A and four V/output ciphertexts");
    const auto &v8 = bert_v.value().Locations(8).front();
    const auto &o8 = bert_av.value().output_packing.Locations(8).front();
    Require(v8.slot == 0 && v8.component == hera::ComplexLane::kImaginary &&
                o8.slot == 0 && o8.component == hera::ComplexLane::kImaginary &&
                bert_v.value().Locations(16).front().ciphertext == 1 &&
                bert_av.value().probability_packing.Locations(16)
                    .front().ciphertext == 1,
            "BERT-base A/V boundaries do not match eight lanes");
    Require(bert_q.value().ciphertext_count == 3 &&
                bert_q.value().Locations(8).front().slot == 8 * 128 * 8 &&
                bert_q.value().Locations(16).front().component == hera::ComplexLane::kImaginary &&
                bert_q.value().Locations(32).front().ciphertext == 1 &&
                bert_q.value().Locations(8 * 64).front().ciphertext == 2,
            "Q/K 8+4 policy drifted from the independent twelve-head V policy");
    const auto &bert_rotations = bert_av.value().required_rotation_steps;
    Require(bert_rotations.size() == 37,
            "BERT-base A*V did not request exactly 36 rotations and conjugation");
    for (const int step : {-6144, -3072, 3072, 6144})
        Require(std::find(bert_rotations.begin(), bert_rotations.end(), step)
                    != bert_rotations.end(),
                "BERT-base is missing a directional replication key");
    // The old first=6 alignment requested +6B, canonicalized to -7168.
    Require(std::find(bert_rotations.begin(), bert_rotations.end(), -7168)
                == bert_rotations.end(),
            "BERT-base still requests the obsolete seed-alignment key");
    for (const auto *packing : {&bert_av.value().probability_packing,
                               &bert_v.value(), &bert_av.value().output_packing})
        for (const auto &locations : packing->logical_locations)
            for (const auto &location : locations)
                Require(location.slot < 8 * 128 * 12,
                        "BERT-base A/V packing used a lane beyond capacity eight");
}

void TestComplexProbabilityInputs(bool terminal_suffix = false)
{
    auto config = MakeConfig();
    auto &profile = config.he_parameter_profiles.at("shared");
    profile.coefficient_modulus_bits = {60, 60, 60, 60};
    profile.declared_depth = 3;
    const std::size_t av_input_level = terminal_suffix ? 3 : 0;
    if (terminal_suffix) {
        profile.coefficient_modulus_bits = {60, 35, 60, 60, 60, 60, 60};
        profile.declared_depth = 6;
    }
    hera::HeContextRegistry client;
    Require(client.CreateClientProfiles(config).ok(),
            "complex A client context failed");
    // Exercise paired replication, one-lane fallback, partial physical blocks,
    // unpaired diagonal tails, multiple giant groups, and head tiles.
    for (const auto shape : {std::pair<std::size_t, std::size_t>{1, 2}, {3, 2},
                             {8, 4},
                             {3, 300}, {5, 300}, {5, 600},
                             {13, 31}, {16, 12}, {40, 5}, {128, 1}}) {
        auto layer = config.layers.front();
        layer.tokens = shape.first;
        layer.q_heads = shape.second;
        layer.kv_heads = layer.q_heads == 2 ? 2 :
                         (layer.q_heads == 4 ? 2 : 1);
        layer.value_head_width = (layer.tokens == 128 || layer.q_heads == 12) ? 64 :
                                 (layer.tokens == 40 ? 24 : 4);
        auto a_plan = hera::AttentionLn2PackingPlanner::AttentionValue(
            layer, profile.polynomial_degree / 2);
        auto v_plan = hera::MultiHeadPackingPlan::ProjectionOutput(
            layer, hera::Layout::kMultiHeadV, profile.polynomial_degree / 2);
        Require(a_plan.ok() && v_plan.ok(), "complex A packing failed");
        Require(client.CompileRotationKeys(
                    "shared", a_plan.value().required_rotation_steps).ok(),
                "complex A Galois keys failed");
        auto bundle = client.ExportPublicBundle();
        Require(bundle.ok(), bundle.status().message());
        hera::HeContextRegistry server;
        Require(server.CreateServerProfiles(config, bundle.value()).ok(),
                "complex A server context failed");
        const auto &client_context = client.Require(
            "shared", hera::PartyRole::kClient);
        const auto &server_context = server.Require(
            "shared", hera::PartyRole::kServer);
        std::vector<double> a(layer.tokens * layer.q_heads * layer.tokens);
        std::vector<double> v(
            layer.tokens * layer.kv_heads * layer.value_head_width);
        std::vector<double> expected(
            layer.tokens * layer.q_heads * layer.value_head_width, 0.0);
        for (std::size_t i = 0; i < v.size(); ++i)
            v[i] = std::sin(static_cast<double>(i + 1) * 0.7) * 0.2;
        for (std::size_t row = 0; row < layer.tokens; ++row)
            for (std::size_t head = 0; head < layer.q_heads; ++head) {
                const auto offset = (row * layer.q_heads + head) * layer.tokens;
                double sum = 0.0;
                for (std::size_t column = 0; column < layer.tokens; ++column) {
                    a[offset + column] = 1.0 +
                        ((row * 3 + head * 2 + column * 5) % 11);
                    sum += a[offset + column];
                }
                for (std::size_t column = 0; column < layer.tokens; ++column) {
                    a[offset + column] /= sum;
                    for (std::size_t d = 0; d < layer.value_head_width; ++d)
                        expected[(row * layer.q_heads + head) *
                                     layer.value_head_width + d] +=
                            a[offset + column] *
                            v[(column * layer.kv_heads +
                               head / (layer.q_heads / layer.kv_heads)) *
                                  layer.value_head_width + d];
                }
            }
        auto a_ct = Encrypt(a, a_plan.value().probability_packing,
                            client_context, av_input_level, hera::HeRegionId::kRegion2);
        auto v_ct = Encrypt(v, v_plan.value(), client_context, av_input_level,
                            hera::HeRegionId::kRegion2);
        Require(MaxError(Decrypt(a_ct, client_context), a) < 1e-5,
                "complex A input packing lost a probability component");
        hera::AttentionLn2ExecutionReport report;
        auto output = hera::MultiHeadAttentionValueKernel::Evaluate(
            a_ct, v_ct, layer, server_context, &report, 2);
        Require(output.ok(), output.status().message());
        Require(MaxError(Decrypt(output.value(), client_context), expected) <
                    1e-4,
                "complex A extraction/replication differs from plain A*V");
        Require(ActualLevel(client_context,
                    output.value().ciphertexts.front().parms_id()) == av_input_level + 3,
                "complex A extraction consumed an extra CKKS level");
        hera::SealPlainParameterStore prepared;
        prepared.BeginPlaintextPreparation();
        {
            hera::ModelPlaintextCacheScope scope(&prepared, layer.id);
            Require(hera::MultiHeadAttentionValueKernel::Prepare(layer, server_context, av_input_level).ok(),
                "AV public mask preparation failed");
        }
        prepared.FreezePlaintextCache();
        const auto mask_count = prepared.encoded_plaintext_count();
        {
            hera::ModelPlaintextCacheScope scope(&prepared, layer.id);
            hera::AttentionLn2ExecutionReport warm;
            auto cached = hera::MultiHeadAttentionValueKernel::Evaluate(a_ct, v_ct, layer, server_context, &warm, 2);
            Require(cached.ok(), cached.status().message());
            Require(MaxError(Decrypt(cached.value(), client_context), expected) < 1e-4 &&
                        prepared.encoded_plaintext_count() == mask_count &&
                        warm.attention_value_plaintext_encodings == 0,
                "prepared AV changed numerics or encoded masks online");
        }
        const auto tiles = (layer.q_heads +
                            client_context.slot_count() / hera::detail::TokenColumnStride(layer.tokens) - 1) /
                           (client_context.slot_count() / hera::detail::TokenColumnStride(layer.tokens));
        Require(report.conjugations == ((layer.tokens + 1) / 2) * tiles,
                "prepared A did not share conjugation across diagonal pairs");
        Require(report.rotations == report.attention_probability_rotations +
                    report.attention_value_rotations +
                    report.attention_recovery_rotations,
                "A*V stage rotation counts do not sum to the kernel total");
        if (layer.tokens == 40) {
            // T=40 pads to stride=64. Capacity 2048/(64*5)=6
            // rounds down to four lanes, with five eight-diagonal groups.
            Require(a_ct.ciphertexts.size() == 5 &&
                        v_ct.ciphertexts.size() == 3 &&
                        report.attention_probability_rotations == 38 &&
                        report.attention_value_rotations == 24 &&
                        report.attention_recovery_rotations == 24 &&
                        report.conjugations == 20 &&
                        report.relinearizations == 15 &&
                        report.ciphertext_multiplications == 120,
                    "padded four-lane A*V no longer matches the 121-KS plan");
        }
        if (layer.tokens == 128) {
            Require(report.attention_probability_rotations == 166 &&
                        report.attention_value_rotations == 64 &&
                        report.attention_recovery_rotations == 12 &&
                        report.conjugations == 64 &&
                        report.relinearizations == 8 &&
                        report.ciphertext_multiplications == 256,
                    "16-lane THOR schedule no longer matches the 314-KS plan");
        }
        if (layer.tokens == 5) {
            std::size_t real_only_count = 0;
            const auto max_heads = client_context.slot_count() / hera::detail::TokenColumnStride(layer.tokens);
            for (std::size_t begin = 0; begin < layer.q_heads;) {
                const auto heads = std::min(max_heads, layer.q_heads - begin);
                const auto lanes = hera::detail::FloorPowerOfTwo(
                    client_context.slot_count() / (hera::detail::TokenColumnStride(layer.tokens) * heads));
                real_only_count += (layer.tokens + lanes - 1) / lanes;
                begin += heads;
            }
            Require(a_ct.ciphertexts.size() < real_only_count,
                    "complex A pairing did not reduce ciphertext count");
        }
    }
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
    std::set<int> all_steps(steps.value().begin(), steps.value().end());
    // This fixture also runs a different attention geometry. Compile its
    // exact keys instead of depending on SEAL's rotation decomposition.
    auto complex_key_layer = layer;
    complex_key_layer.tokens = 3;
    complex_key_layer.q_heads = 200;
    complex_key_layer.kv_heads = 1;
    complex_key_layer.value_head_width = 8;
    auto complex_keys = hera::AttentionLn2PackingPlanner::AttentionValue(
        complex_key_layer, profile.polynomial_degree / 2);
    Require(complex_keys.ok(), complex_keys.status().message());
    all_steps.insert(complex_keys.value().required_rotation_steps.begin(),
                     complex_keys.value().required_rotation_steps.end());
    Require(client.CompileRotationKeys("shared", {all_steps.begin(), all_steps.end()}).ok(),
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
    // metadata. With 2048 slots, stride=4 and 200 heads give two lanes and two
    // real-only giant groups; complex BSGS combines them into
    // one group. Value dimensions
    // 0/1 and 2/3 therefore occupy the real and imaginary components. MQA also
    // verifies that physical query-head replication survives all three BSGS
    // phases.
    auto complex_layer = layer;
    complex_layer.tokens = 3;
    complex_layer.q_heads = 200;
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
                complex_report.relinearizations == 1 &&
                complex_report.conjugations == 2,
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

void TestStatisticsReductionAndCache()
{
    auto config = MakeConfig();
    auto &layer = config.layers.front();
    layer.tokens = 3;
    layer.model_width = 1025; // Three real ciphertexts; the final one is a tail.
    hera::HeContextRegistry client;
    Require(client.CreateClientProfiles(config).ok(), "cannot create statistics context");
    const auto &client_context = client.Require("shared", hera::PartyRole::kClient);
    const std::size_t stride = 4;
    const auto features = client_context.slot_count() / stride;
    std::vector<int> steps{0};
    for (std::size_t step = 1; step < features; step <<= 1U)
        steps.push_back(static_cast<int>(step * stride));
    Require(client.CompileRotationKeys("shared", steps).ok(),
            "cannot compile statistics keys");
    auto bundle = client.ExportPublicBundle();
    Require(bundle.ok(), bundle.status().message());
    hera::HeContextRegistry server;
    Require(server.CreateServerProfiles(config, bundle.value()).ok(),
            "cannot create statistics server");
    const auto &server_context = server.Require("shared", hera::PartyRole::kServer);
    auto plan = hera::AttentionLn2PackingPlanner::ComplexStrictColumns(
        {{layer.tokens, layer.model_width}}, client_context.slot_count());
    Require(plan.ok(), plan.status().message());
    std::vector<double> values(layer.tokens * layer.model_width);
    for (std::size_t f = 0; f < layer.model_width; ++f) {
        values[f] = -2.0 + 0.4 * std::sin(0.1 * static_cast<double>(f));
        values[layer.model_width + f] = 2.5; // Zero variance before epsilon.
        values[2 * layer.model_width + f] = 10.0 + (f % 2 ? 0.001 : -0.001);
    }
    auto encrypted = Encrypt(values, plan.value(), client_context, 6,
                             hera::HeRegionId::kRegion4);
    auto realified = hera::ComplexRealifyKernel::Evaluate(
        encrypted, layer, server_context);
    Require(realified.ok(), realified.status().message());
    auto input = realified.take_value();
    Require(input.ciphertexts.size() == 3 &&
                input.ciphertexts[0].scale() != input.ciphertexts[1].scale(),
            "statistics fixture does not cover mixed scales and a tail");
    // Deliberately dirty every padding slot. Both reductions must mask it.
    seal::CKKSEncoder encoder(*client_context.context);
    seal::Evaluator evaluator(*client_context.context);
    for (std::size_t i = 0; i < input.ciphertexts.size(); ++i) {
        std::vector<double> padding(client_context.slot_count(), 9.0);
        for (const auto &locations : input.physical_packing.logical_locations)
            if (locations.front().ciphertext == i) padding[locations.front().slot] = 0.0;
        seal::Plaintext plain;
        encoder.encode(padding, input.ciphertexts[i].parms_id(),
                       input.ciphertexts[i].scale(), plain);
        evaluator.add_plain_inplace(input.ciphertexts[i], plain);
    }
    const auto check = [&](const hera::LayerNormStatisticsOutputs &output,
                           const hera::TransformerLayerConfig &settings,
                           double input_factor) {
        std::vector<double> centered(values.size()), variance(layer.tokens);
        for (std::size_t token = 0; token < layer.tokens; ++token) {
            double mean = 0;
            for (std::size_t f = 0; f < layer.model_width; ++f)
                mean += values[token * layer.model_width + f] * input_factor;
            mean /= layer.model_width;
            for (std::size_t f = 0; f < layer.model_width; ++f) {
                const auto index = token * layer.model_width + f;
                centered[index] = values[index] * input_factor - mean;
                variance[token] += centered[index] * centered[index];
            }
            variance[token] = variance[token] / layer.model_width +
                              settings.attention_norm_epsilon;
        }
        const auto actual_variance = Decrypt(output.variance, client_context);
        Require(MaxError(Decrypt(output.centered, client_context), centered) < 1.0e-3 &&
                    MaxError(actual_variance, variance) < 1.0e-5 &&
                    std::abs(actual_variance[1] - settings.attention_norm_epsilon) < 1.0e-7,
                "statistics reduction differs from plaintext reference");
    };
    hera::AttentionLn2ExecutionReport cold;
    auto serial = hera::LayerNormStatisticsKernel::Evaluate(
        input, layer, server_context, &cold, 1);
    Require(serial.ok(), serial.status().message());
    check(serial.value(), layer, 1.0);
    Require(cold.layer_norm_plaintext_encodings > 0, "cold plaintext plan was not encoded");
    hera::AttentionLn2ExecutionReport warm;
    auto parallel = hera::LayerNormStatisticsKernel::Evaluate(
        input, layer, server_context, &warm, 32);
    Require(parallel.ok(), parallel.status().message());
    check(parallel.value(), layer, 1.0);
    Require(warm.layer_norm_plaintext_encodings == 0 && warm.layer_norm_workers == 2,
            "statistics cache or worker cap failed");
    Require(cold.relinearizations < input.ciphertexts.size() &&
                cold.relinearizations == warm.relinearizations &&
                cold.rescales == input.ciphertexts.size() + cold.relinearizations + 1 &&
                cold.ciphertext_plain_multiplications ==
                    3 * input.ciphertexts.size() + cold.relinearizations,
            "lazy statistics did not group full blocks separately from the tail");
    // LN1 and LN2 alternate profiles. Metadata preparation must preserve both
    // plans, with no HE operation or encoding on the subsequent original hit.
    auto alternate_config = MakeConfig();
    auto alternate_profile = alternate_config.he_parameter_profiles.at("shared");
    alternate_profile.polynomial_degree = 2048;
    alternate_profile.coefficient_modulus_bits = {60, 60, 60, 60, 60};
    alternate_profile.declared_depth = 4;
    alternate_profile.input_scale_bits = alternate_profile.default_scale_bits =
        alternate_profile.output_scale_bits = 38;
    alternate_config.he_parameter_profiles = {{"alternate", alternate_profile}};
    for (auto &placement : alternate_config.layer_execution.front().regions)
        placement.second.parameter_profile = "alternate";
    hera::HeContextRegistry alternate;
    Require(alternate.CreateClientProfiles(alternate_config, false).ok(),
            "cannot create alternate statistics context");
    const auto &alternate_context = alternate.Require("alternate", hera::PartyRole::kClient);
    auto alternate_plan = hera::AttentionLn2PackingPlanner::RealStrictColumns(
        input.logical_spec.shape, alternate_context.slot_count());
    Require(alternate_plan.ok(), alternate_plan.status().message());
    hera::SealCipherTensor alternate_metadata;
    alternate_metadata.parameter_profile = "alternate";
    alternate_metadata.logical_spec = input.logical_spec;
    alternate_metadata.logical_spec.he_parameter_profile = "alternate";
    alternate_metadata.logical_spec.planned_level = 1;
    alternate_metadata.logical_spec.planned_scale_bits = 38;
    alternate_metadata.physical_packing = alternate_plan.take_value();
    alternate_metadata.ciphertexts.resize(alternate_metadata.physical_packing.ciphertext_count);
    for (auto &ct : alternate_metadata.ciphertexts) {
        ct.parms_id() = LevelId(alternate_context, 1);
        ct.scale() = std::ldexp(1.0, 38);
    }
    hera::AttentionLn2ExecutionReport alternate_cold, original_hit, alternate_hit;
    Require(hera::LayerNormStatisticsKernel::Prepare(
                alternate_metadata, layer, alternate_context, &alternate_cold).ok() &&
                alternate_cold.layer_norm_plaintext_encodings > 0,
            "alternate statistics metadata preparation failed");
    Require(hera::LayerNormStatisticsKernel::Prepare(input, layer, server_context, &original_hit).ok() &&
                hera::LayerNormStatisticsKernel::Prepare(
                    alternate_metadata, layer, alternate_context, &alternate_hit).ok() &&
                original_hit.layer_norm_plaintext_encodings == 0 &&
                alternate_hit.layer_norm_plaintext_encodings == 0,
            "alternating statistics profiles evicted each other");
    Require(MaxError(Decrypt(serial.value().centered, client_context),
                     Decrypt(parallel.value().centered, client_context)) < 1.0e-5 &&
                MaxError(Decrypt(serial.value().variance, client_context),
                         Decrypt(parallel.value().variance, client_context)) < 1.0e-7,
            "serial and parallel statistics differ");
    auto &pool = hera::detail::HeWorkerPool::Instance();
    const auto previous_budget = pool.MemoryBudget();
    pool.SetMemoryBudget(1ULL << 20);
    hera::AttentionLn2ExecutionReport constrained;
    auto limited = hera::LayerNormStatisticsKernel::Evaluate(
        input, layer, server_context, &constrained, 32);
    pool.SetMemoryBudget(previous_budget);
    Require(limited.ok(), limited.status().message());
    Require(constrained.layer_norm_workers == 1, "small memory budget did not disable parallelism");
    check(limited.value(), layer, 1.0);
    auto settings = layer;
    settings.attention_norm_epsilon = 1.0e-3;
    hera::AttentionLn2ExecutionReport epsilon_changed;
    auto new_epsilon = hera::LayerNormStatisticsKernel::Evaluate(
        input, settings, server_context, &epsilon_changed);
    Require(new_epsilon.ok(), new_epsilon.status().message());
    Require(epsilon_changed.layer_norm_plaintext_encodings > 0,
            "epsilon change did not invalidate cached plaintexts");
    check(new_epsilon.value(), settings, 1.0);
    auto scaled_input = input;
    for (auto &ct : scaled_input.ciphertexts) ct.scale() *= 2.0;
    hera::AttentionLn2ExecutionReport scale_changed;
    auto new_scale = hera::LayerNormStatisticsKernel::Evaluate(
        scaled_input, settings, server_context, &scale_changed);
    Require(new_scale.ok(), new_scale.status().message());
    Require(scale_changed.layer_norm_plaintext_encodings > 0,
            "physical scale change did not invalidate cached plaintexts");
    check(new_scale.value(), settings, 0.5);
    auto wrong_packing = input;
    std::swap(wrong_packing.physical_packing.logical_locations[0],
              wrong_packing.physical_packing.logical_locations[1]);
    Require(!hera::LayerNormStatisticsKernel::Evaluate(
                wrong_packing, layer, server_context).ok(),
            "statistics accepted nonstandard column coordinates");
    Require(!hera::LayerNormStatisticsKernel::Prepare(
                wrong_packing, layer, server_context).ok(),
            "statistics preparation accepted nonstandard column coordinates");
    for (std::size_t i = 0; i < 9; ++i) {
        auto unique = layer;
        unique.attention_norm_epsilon = 0.02 + 0.01 * i;
        Require(hera::LayerNormStatisticsKernel::Prepare(input, unique, server_context).ok(),
                "statistics cache eviction preparation failed");
    }
    hera::AttentionLn2ExecutionReport evicted;
    Require(hera::LayerNormStatisticsKernel::Prepare(input, layer, server_context, &evicted).ok() &&
                evicted.layer_norm_plaintext_encodings > 0,
            "statistics cache did not enforce its entry limit");
    Require(MaxError(Decrypt(input, client_context), values) < 1.0e-3 &&
                ActualLevel(client_context, input.ciphertexts.front().parms_id()) == 6,
            "statistics modified its input");
}


void TestSevenPrimeDirectResidual()
{
    auto config = MakeConfig();
    auto &layer = config.layers.front();
    layer.tokens = 16;
    layer.q_heads = layer.kv_heads = 16;
    layer.qk_head_width = layer.value_head_width = 8;
    layer.model_width = 71; // Two real ciphertexts, with a partial final column block.
    auto &profile = config.he_parameter_profiles.at("shared");
    profile.polynomial_degree = 2048;
    profile.coefficient_modulus_bits = {60, 35, 60, 60, 60, 60, 60};
    profile.declared_depth = 6;
    hera::HeContextRegistry client;
    Require(client.CreateClientProfiles(config).ok(), "direct-residual context failed");
    auto rotations = hera::SealSciAttentionLn2Executor::RequiredOutputRotationSteps(layer, profile);
    Require(rotations.ok(), rotations.status().message());
    Require(client.CompileRotationKeys("shared", rotations.value()).ok(), "direct-residual keys failed");
    auto bundle = client.ExportPublicBundle();
    Require(bundle.ok(), bundle.status().message());
    hera::HeContextRegistry server;
    Require(server.CreateServerProfiles(config, bundle.value()).ok(), "direct-residual server failed");
    const auto &cc = client.Require("shared", hera::PartyRole::kClient);
    const auto &sc = server.Require("shared", hera::PartyRole::kServer);
    auto av = hera::AttentionLn2PackingPlanner::AttentionValue(layer, cc.slot_count());
    auto real = hera::AttentionLn2PackingPlanner::RealStrictColumns(
        {{layer.tokens, layer.model_width}}, cc.slot_count());
    Require(av.ok() && real.ok(), "direct-residual packing failed");
    bool imaginary_source = false;
    for (const auto &locations : av.value().output_packing.logical_locations)
        imaginary_source |= locations.front().component == hera::ComplexLane::kImaginary;
    Require(imaginary_source && real.value().ciphertext_count == 2,
            "direct-residual fixture must cover complex source and real tail");
    const auto input_width = layer.q_heads * layer.value_head_width;
    std::vector<double> values(layer.tokens * input_width);
    std::vector<double> hidden(layer.tokens * layer.model_width);
    for (std::size_t i = 0; i < values.size(); ++i)
        values[i] = 0.1 * std::sin(0.07 * static_cast<double>(i));
    for (std::size_t i = 0; i < hidden.size(); ++i)
        hidden[i] = 0.2 * std::cos(0.13 * static_cast<double>(i));
    hera::DenseMatrixParameter wo{{"Wo", input_width, layer.model_width, true}, {}};
    wo.row_major.resize(input_width * layer.model_width);
    for (std::size_t i = 0; i < wo.row_major.size(); ++i)
        wo.row_major[i] = 0.01 * std::sin(0.11 * static_cast<double>(i));
    hera::DenseVectorParameter bias{{"bo", layer.model_width}, {}};
    bias.values.resize(layer.model_width);
    for (std::size_t f = 0; f < layer.model_width; ++f)
        bias.values[f] = 0.001 * static_cast<double>(f % 9);
    auto input = Encrypt(values, av.value().output_packing, cc, 2, hera::HeRegionId::kRegion4);
    auto residual = Encrypt(hidden, real.value(), cc, 1, hera::HeRegionId::kRegion2);
    std::vector<double> expected(hidden);
    for (std::size_t t = 0; t < layer.tokens; ++t)
        for (std::size_t f = 0; f < layer.model_width; ++f) {
            auto &y = expected[t * layer.model_width + f];
            y += bias.values[f];
            for (std::size_t j = 0; j < input_width; ++j)
                y += values[t * input_width + j] * wo.row_major[j * layer.model_width + f];
        }
    // Compile the real projection through the production preparation/cache path.
    hera::SealPlainParameterStore parameters;
    parameters.BeginPlaintextPreparation();
    {
        hera::ModelPlaintextCacheScope cache(&parameters, layer.id);
        auto prepared = hera::AttentionOutputProjectionKernel::EvaluateReal(input, wo, bias, layer, sc);
        Require(prepared.ok(), prepared.status().message());
    }
    parameters.FreezePlaintextCache();
    hera::AttentionLn2ExecutionReport report;
    auto projected = [&] {
        hera::ModelPlaintextCacheScope cache(&parameters, layer.id);
        return hera::AttentionOutputProjectionKernel::EvaluateReal(input, wo, bias, layer, sc, &report, 2);
    }();
    Require(projected.ok(), projected.status().message());
    Require(projected.value().logical_spec.layout == hera::Layout::kStrictColumn &&
                ActualLevel(cc, projected.value().ciphertexts.front().parms_id()) == 3,
            "real Wo output contract mismatch");
    auto late = Encrypt(hidden, real.value(), cc, 4, hera::HeRegionId::kRegion4);
    Require(!hera::AttentionResidualKernel::Add(projected.value(), late, sc).ok(),
            "direct residual accepted a lower-modulus input");
    auto wrong_scale = residual;
    for (auto &ct : wrong_scale.ciphertexts) ct.scale() *= 2.0;
    Require(!hera::AttentionResidualKernel::Add(projected.value(), wrong_scale, sc).ok(),
            "direct residual accepted mismatched scales");
    auto combined = hera::AttentionResidualKernel::Add(projected.take_value(), residual, sc, &report);
    Require(combined.ok(), combined.status().message());
    Require(MaxError(Decrypt(combined.value(), cc), expected) < 1e-5,
            "real Wo plus original X differs from plaintext reference");
    Require(ActualLevel(cc, residual.ciphertexts.front().parms_id()) == 1 &&
                MaxError(Decrypt(residual, cc), hidden) < 1e-6,
            "direct residual modified original X");
    auto statistics = hera::LayerNormStatisticsKernel::Evaluate(combined.value(), layer, sc, &report, 2);
    Require(statistics.ok(), statistics.status().message());
    Require(ActualLevel(cc, statistics.value().centered.ciphertexts.front().parms_id()) == 4 &&
                ActualLevel(cc, statistics.value().variance.ciphertexts.front().parms_id()) == 6,
            "seven-prime LN statistics did not reach L4/L6");
    std::vector<double> centered(expected.size()), variance(layer.tokens);
    for (std::size_t t = 0; t < layer.tokens; ++t) {
        double mean = 0.0;
        for (std::size_t f = 0; f < layer.model_width; ++f)
            mean += expected[t * layer.model_width + f];
        mean /= layer.model_width;
        for (std::size_t f = 0; f < layer.model_width; ++f) {
            const auto i = t * layer.model_width + f;
            centered[i] = expected[i] - mean;
            variance[t] += centered[i] * centered[i];
        }
        variance[t] = variance[t] / layer.model_width + layer.attention_norm_epsilon;
    }
    const auto centered_error = MaxError(Decrypt(statistics.value().centered, cc), centered);
    const auto variance_error = MaxError(Decrypt(statistics.value().variance, cc), variance);
    Require(centered_error < 1e-5 && variance_error < 1e-6,
            "direct-residual LN statistics differs from plaintext reference");
    std::cout << "DIRECT_RESIDUAL_CENTERED_MAX_ERROR=" << centered_error << '\n'
              << "DIRECT_RESIDUAL_VARIANCE_MAX_ERROR=" << variance_error << '\n';
}


void TestRealWoBsgsPackingVariants()
{
    struct Case { std::size_t heads, kv_heads, width, hidden; };
    // GQA + partial complex lane/real output, then MQA + a physical head tile tail.
    for (const auto &settings : {Case{12, 4, 37, 263}, Case{259, 1, 3, 19}}) {
        auto config = MakeConfig();
        auto &layer = config.layers.front();
        layer.tokens = 3;
        layer.q_heads = settings.heads;
        layer.kv_heads = settings.kv_heads;
        layer.value_head_width = settings.width;
        layer.model_width = settings.hidden;
        auto &profile = config.he_parameter_profiles.at("shared");
        profile.polynomial_degree = 2048;
        profile.coefficient_modulus_bits = {60, 35, 60, 60, 60, 60, 60};
        profile.declared_depth = 6;
        auto packing = hera::AttentionLn2PackingPlanner::AttentionValue(layer, 1024);
        auto split = hera::AttentionOutputProjectionKernel::RealColumnBsgs(layer, 1024);
        auto keys = hera::AttentionOutputProjectionKernel::RequiredRealRotationSteps(layer, 1024);
        Require(packing.ok() && split.ok() && keys.ok(), "variant Wo planning failed");
        Require(split.value().baby_steps * split.value().giant_steps ==
                    split.value().columns_per_ciphertext,
                "Wo split does not cover the physical column cycle");
        hera::HeContextRegistry client;
        Require(client.CreateClientProfiles(config, false).ok(), "variant Wo context failed");
        Require(client.CompileRotationKeys("shared", keys.value()).ok(), "variant Wo keys failed");
        auto bundle = client.ExportPublicBundle();
        Require(bundle.ok(), bundle.status().message());
        hera::HeContextRegistry server;
        Require(server.CreateServerProfiles(config, bundle.value()).ok(), "variant Wo server failed");
        const auto &cc = client.Require("shared", hera::PartyRole::kClient);
        const auto &sc = server.Require("shared", hera::PartyRole::kServer);
        const auto width = layer.q_heads * layer.value_head_width;
        std::vector<double> values(layer.tokens * width);
        for (std::size_t i = 0; i < values.size(); ++i)
            values[i] = 0.07 * std::sin(0.031 * static_cast<double>(i));
        auto input = Encrypt(values, packing.value().output_packing, cc, 2, hera::HeRegionId::kRegion4);
        // Dirty all padding tokens, unused columns and absent complex components.
        std::vector<std::vector<std::complex<double>>> dirt(input.ciphertexts.size(),
            std::vector<std::complex<double>>(cc.slot_count(), {0.31, -0.27}));
        for (const auto &locations : input.physical_packing.logical_locations) {
            const auto &where = locations.front();
            auto &slot = dirt[where.ciphertext][where.slot];
            if (where.component == hera::ComplexLane::kReal) slot.real(0.0);
            else slot.imag(0.0);
        }
        seal::CKKSEncoder encoder(*cc.context);
        seal::Evaluator evaluator(*cc.context);
        for (std::size_t ct = 0; ct < input.ciphertexts.size(); ++ct) {
            seal::Plaintext plain;
            encoder.encode(dirt[ct], input.ciphertexts[ct].parms_id(), input.ciphertexts[ct].scale(), plain);
            evaluator.add_plain_inplace(input.ciphertexts[ct], plain);
        }
        hera::DenseMatrixParameter wo{{"Wo", width, layer.model_width, true}, {}};
        wo.row_major.resize(width * layer.model_width);
        for (std::size_t i = 0; i < wo.row_major.size(); ++i)
            wo.row_major[i] = 0.013 * std::cos(0.019 * static_cast<double>(i));
        hera::DenseVectorParameter bias{{"bo", layer.model_width}, {}};
        bias.values.resize(layer.model_width);
        for (std::size_t f = 0; f < layer.model_width; ++f)
            bias.values[f] = 0.001 * static_cast<double>(static_cast<int>(f % 13) - 6);
        std::vector<double> expected(layer.tokens * layer.model_width);
        for (std::size_t t = 0; t < layer.tokens; ++t)
            for (std::size_t f = 0; f < layer.model_width; ++f) {
                auto &value = expected[t * layer.model_width + f];
                value = bias.values[f];
                for (std::size_t row = 0; row < width; ++row)
                    value += values[t * width + row] * wo.row_major[row * layer.model_width + f];
            }
        hera::AttentionLn2ExecutionReport cold;
        auto serial = hera::AttentionOutputProjectionKernel::EvaluateReal(input, wo, bias, layer, sc, &cold, 1);
        Require(serial.ok(), serial.status().message());
        Require(MaxError(Decrypt(serial.value(), cc), expected) < 1e-5,
                "cold Wo BSGS differs from logical matrix product");
        hera::SealPlainParameterStore parameters;
        parameters.BeginPlaintextPreparation();
        {
            hera::ModelPlaintextCacheScope scope(&parameters, layer.id);
            auto metadata = input;
            for (auto &ct : metadata.ciphertexts) {
                const auto id = ct.parms_id(); const auto scale = ct.scale();
                ct = seal::Ciphertext(); ct.parms_id() = id; ct.scale() = scale;
            }
            auto prepared = hera::AttentionOutputProjectionKernel::EvaluateReal(metadata, wo, bias, layer, sc);
            Require(prepared.ok(), prepared.status().message());
        }
        parameters.FreezePlaintextCache();
        auto program = parameters.GetPreparedProjection(
            layer.id + "/attention/Wo/real.weights.bsgs");
        Require(program.ok() && program.value()->fixed_phase,
                "real-weight Wo program has no fixed phase");
        const auto &phase = *program.value()->fixed_phase;
        Require(phase.scale() == 1.0 && phase.parms_id() == LevelId(cc, 2),
                "Wo fixed phase consumes scale or uses the wrong level");
        std::vector<std::complex<double>> phase_slots;
        encoder.decode(phase, phase_slots);
        for (std::size_t slot = 0; slot < phase_slots.size(); ++slot)
            Require(std::abs(phase_slots[slot] - std::complex<double>{
                        0.0, slot % 2 == 0 ? 1.0 : -1.0}) < 1e-12,
                    "Wo fixed phase has the wrong slot signs");
        std::size_t phase_groups = 0, weight_products = 0;
        for (const auto &group : program.value()->groups) {
            phase_groups += !group.phase_terms.empty();
            for (const auto *terms : {&group.terms, &group.phase_terms})
                for (const auto &term : *terms) {
                    ++weight_products;
                    std::vector<std::complex<double>> slots;
                    encoder.decode(*term.plaintext, slots);
                    for (const auto &slot : slots)
                        Require(std::abs(slot.imag()) < 1e-10,
                                "Wo weight plaintext contains imaginary data");
                }
        }
        const auto plaintext_count = parameters.encoded_plaintext_count();
        hera::AttentionLn2ExecutionReport warm;
        auto parallel = [&] {
            hera::ModelPlaintextCacheScope scope(&parameters, layer.id);
            return hera::AttentionOutputProjectionKernel::EvaluateReal(input, wo, bias, layer, sc, &warm, 2);
        }();
        Require(parallel.ok(), parallel.status().message());
        Require(MaxError(Decrypt(parallel.value(), cc), expected) < 1e-5 &&
                    MaxError(Decrypt(serial.value(), cc), Decrypt(parallel.value(), cc)) < 1e-6,
                "cached parallel Wo differs from the logical matrix product");
        Require(parameters.encoded_plaintext_count() == plaintext_count,
                "frozen Wo BSGS encoded an online model plaintext");
        Require(warm.rotations == cold.rotations && warm.conjugations == cold.conjugations &&
                    warm.wo_phase_multiplications == phase_groups &&
                    warm.ciphertext_plain_multiplications == weight_products + phase_groups &&
                    warm.wo_baby_rotations + warm.wo_giant_rotations == warm.rotations &&
                    warm.wo_output_conjugations == warm.conjugations &&
                    warm.conjugations == parallel.value().ciphertexts.size() &&
                    warm.rescales == parallel.value().ciphertexts.size() &&
                    warm.ciphertext_multiplications == 0 && warm.relinearizations == 0 &&
                    warm.rotations <= split.value().rotation_upper_bound,
                "Wo BSGS operation accounting/rotation reuse is inconsistent");
        seal::Decryptor decryptor(*cc.context, *cc.secret_key);
        for (const auto &ct : parallel.value().ciphertexts) {
            Require(ct.parms_id() == LevelId(cc, 3) &&
                        std::abs(ct.scale() / std::ldexp(1.0, 44) - 1.0) < 1e-6,
                    "variant Wo output level/scale mismatch");
            seal::Plaintext plain; decryptor.decrypt(ct, plain);
            std::vector<std::complex<double>> decoded; encoder.decode(plain, decoded);
            for (const auto &slot : decoded)
                Require(std::abs(slot.imag()) < 1e-5, "Wo output has a significant imaginary component");
        }
        if (layer.q_heads == 12) {
            // Either channel can be absent in a group; bias remains complete.
            for (const auto keep : {hera::ComplexLane::kReal, hera::ComplexLane::kImaginary}) {
                auto single_channel = wo;
                for (std::size_t row = 0; row < width; ++row)
                    if (input.physical_packing.Locations(row).front().component != keep)
                        std::fill_n(single_channel.row_major.begin() + row * layer.model_width,
                                    layer.model_width, 0.0);
                auto reference = expected;
                for (std::size_t t = 0; t < layer.tokens; ++t)
                    for (std::size_t f = 0; f < layer.model_width; ++f) {
                        auto &value = reference[t * layer.model_width + f];
                        value = bias.values[f];
                        for (std::size_t row = 0; row < width; ++row)
                            value += values[t * width + row] *
                                single_channel.row_major[row * layer.model_width + f];
                    }
                hera::AttentionLn2ExecutionReport single_report;
                auto actual = hera::AttentionOutputProjectionKernel::EvaluateReal(
                    input, single_channel, bias, layer, sc, &single_report, 2);
                Require(actual.ok(), actual.status().message());
                Require(MaxError(Decrypt(actual.value(), cc), reference) < 1e-5,
                        "Wo with one weight channel differs from the logical matrix product");
                Require((single_report.wo_phase_multiplications == 0) ==
                            (keep == hera::ComplexLane::kReal),
                        "Wo applied fixed phase to the wrong channel");
            }
        }
        auto invalid = input;
        std::swap(invalid.physical_packing.logical_locations[0], invalid.physical_packing.logical_locations[1]);
        {
            hera::ModelPlaintextCacheScope scope(&parameters, layer.id);
            Require(!hera::AttentionOutputProjectionKernel::EvaluateReal(invalid, wo, bias, layer, sc).ok(),
                    "cached Wo accepted a different physical input mapping");
        }
        Require(ActualLevel(cc, input.ciphertexts.front().parms_id()) == 2 &&
                    MaxError(Decrypt(input, cc), values) < 1e-6,
                "Wo modified the input ciphertexts");
        std::cout << "WO_VARIANT_HEADS=" << layer.q_heads
                  << " MAX_ERROR=" << MaxError(Decrypt(parallel.value(), cc), expected)
                  << " KS=" << warm.rotations + warm.conjugations << '\n';
    }
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
    Require(client.CreateClientProfiles(config, false).ok(),
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
        residual_spec.he_region = hera::HeRegionId::kRegion4;
        residual_spec.he_parameter_profile = "shared";
        residual_spec.planned_level = 1;
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
            Require(report.mpc_to_ckks_calls == (config.he_parameter_profiles.at("shared").declared_depth >= 9 ? 1U : 2U) &&
                        report.ckks_to_mpc_calls == (config.he_parameter_profiles.at("shared").declared_depth >= 9 ? 2U : 3U) &&
                        report.reciprocal_square_roots == 2,
                    "integrated operation report is incomplete");
        packet->ok = 1;
    } catch (const std::exception &exception) {
        *error = exception.what();
    }
}

void TestIntegratedExecutor(bool fused = false)
{
    auto config = MakeConfig();
    auto &input_profile = config.he_parameter_profiles.at("shared");
    input_profile.coefficient_modulus_bits = {60, 35, 60, 60, 60, 60, 60};
    input_profile.declared_depth = 6;
    if (fused) {
        input_profile.coefficient_modulus_bits.insert(input_profile.coefficient_modulus_bits.end(), 3, 60);
        input_profile.declared_depth = 9;
    }
    const auto &layer = config.layers.front();
    const auto &profile = config.he_parameter_profiles.at("shared");
    hera::HeContextRegistry client_registry;
    Require(client_registry.CreateClientProfiles(config).ok(),
            "cannot create integrated client profiles");
    auto av_keys = hera::AttentionLn2PackingPlanner::AttentionValue(
        layer, profile.polynomial_degree / 2);
    Require(av_keys.ok(), av_keys.status().message());
    auto steps = hera::SealSciAttentionLn2Executor::RequiredOutputRotationSteps(layer, profile);
    Require(steps.ok(), steps.status().message());
    std::set<int> combined(steps.value().begin(), steps.value().end());
    combined.insert(av_keys.value().required_rotation_steps.begin(),
                    av_keys.value().required_rotation_steps.end());
    Require(client_registry.CompileRotationKeys("shared",
                std::vector<int>(combined.begin(), combined.end())).ok(),
            "cannot compile integrated AV/Wo/statistics rotation keys");
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
    const auto &output_context = client_context;
    auto residual_plan =
        hera::AttentionLn2PackingPlanner::RealStrictColumns(
            {{layer.tokens, layer.model_width}}, output_context.slot_count());
    Require(value_plan.ok() && residual_plan.ok(),
            "cannot compile integrated HE packing");
    auto value = Encrypt({1.0, -0.5, 0.25, 2.0}, value_plan.value(),
                         client_context, 2, hera::HeRegionId::kRegion2);
    const std::vector<double> residual_values{
        0.10, -0.10, 0.20, -0.20,
        0.05, -0.05, 0.15, -0.15};
    auto residual = Encrypt(residual_values, residual_plan.value(),
                            output_context, 1,
                            hera::HeRegionId::kRegion4);

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

void TestSharedProfileStatistics(bool compact)
{
    auto config = MakeConfig();
    auto &profile = config.he_parameter_profiles.at("shared");
    profile.coefficient_modulus_bits = {60, 60, 46, 46, 46, 46, 60, 60};
    profile.input_scale_bits = profile.default_scale_bits = 45;
    profile.output_scale_bits = 46;
    profile.declared_depth = 7;
    if (compact) { profile.coefficient_modulus_bits.pop_back(); profile.declared_depth = 6; }
    const auto input_level = profile.coefficient_modulus_bits.size() - 4;
    auto &layer = config.layers.front();
    layer.tokens = 3;
    hera::HeContextRegistry client;
    Require(client.CreateClientProfiles(config).ok(), "shared statistics client failed");
    std::vector<int> steps;
    for (std::size_t step = 1; step < profile.polynomial_degree / 2 / 4; step <<= 1U)
        steps.push_back(static_cast<int>(step * 4));
    Require(client.CompileRotationKeys("shared", steps).ok(), "statistics keys failed");
    auto bundle = client.ExportPublicBundle();
    Require(bundle.ok(), bundle.status().message());
    hera::HeContextRegistry server;
    Require(server.CreateServerProfiles(config, bundle.value()).ok(), "shared statistics server failed");
    const auto &cc = client.Require("shared", hera::PartyRole::kClient);
    const auto &sc = server.Require("shared", hera::PartyRole::kServer);
    auto packing = hera::AttentionLn2PackingPlanner::RealStrictColumns({{3, 4}}, cc.slot_count());
    Require(packing.ok(), packing.status().message());
    const std::vector<double> values{1.0, 2.0, 3.0, 4.0, 2.5, 2.5, 2.5, 2.5, 1.001, 0.999, 1.001, 0.999};
    auto input = Encrypt(values, packing.value(), cc, input_level, hera::HeRegionId::kRegion1);
    // Encrypt's existing fixture uses scale 45. Doubling the encoded value
    // preserves the value when promoting both physical and declared scale.
    seal::Evaluator evaluator(*cc.context);
    for (auto &ct : input.ciphertexts) { evaluator.add_inplace(ct, ct); ct.scale() *= 2.0; }
    input.logical_spec.planned_scale_bits = 46;
    Require(hera::LayerNormStatisticsKernel::Prepare(input, layer, sc, nullptr, 46).ok(),
            "shared statistics preparation failed");
    hera::AttentionLn2ExecutionReport report;
    auto output = hera::LayerNormStatisticsKernel::Evaluate(input, layer, sc, &report, 2, 46);
    Require(output.ok(), output.status().message());
    const auto variance = Decrypt(output.value().variance, cc);
    const std::vector<double> expected{1.25 + layer.attention_norm_epsilon,
        layer.attention_norm_epsilon, 1e-6 + layer.attention_norm_epsilon};
    Require(MaxError(variance, expected) < 1e-7 && report.layer_norm_plaintext_encodings == 0,
            "shared scale-46 statistics lost variance precision or missed prepared cache");
    Require(output.value().centered.logical_spec.planned_level == input_level + 1 &&
            output.value().variance.logical_spec.planned_level == input_level + 3 &&
            output.value().variance.logical_spec.planned_scale_bits == 46,
            "shared statistics output contract is incorrect");
    Require(!hera::LayerNormStatisticsKernel::Evaluate(input, layer, sc).ok(),
            "low precision square lift was accepted silently");
    Require(!hera::LayerNormStatisticsKernel::Prepare(input, layer, sc, nullptr, 40).ok(),
            "unsupported output scale was accepted");
}

} // namespace

int main(int argc, char **argv)
{
    try {
        TestPackingAndLevels();
        TestComplexProbabilityInputs();
        TestComplexProbabilityInputs(true);
        TestHePath();
        TestStatisticsReductionAndCache();
        TestSharedProfileStatistics(false);
        TestSharedProfileStatistics(true);
        TestSevenPrimeDirectResidual();
        TestRealWoBsgsPackingVariants();
        TestParallelWoProjection();
        TestIntegratedExecutor(argc == 2 && std::string(argv[1]) == "--fused-attention");
        std::cout << "ATTENTION_LN2_HE_READY=YES\n"
                  << "ATTENTION_LN2_STATUS=PASS\n";
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << "ATTENTION_LN2_STATUS=FAIL\n"
                  << "ATTENTION_LN2_ERROR=" << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
