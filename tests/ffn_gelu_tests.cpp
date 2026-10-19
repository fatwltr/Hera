#include "hera/blb_reference.h"
#include "hera/ffn_gelu.h"
#include "hera/mpc_session.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

constexpr int kMpcPort = 19780;
constexpr std::uint64_t kRingMask = (std::uint64_t{1} << 43) - 1;

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
    layer.ffn_width = 5;
    layer.activation = hera::ActivationKind::kGelu;
    layer.ffn_bias = true;
    config.layers = {layer};
    config.mpc_numeric = {43, 13, true};
    config.conversion = {40, 20, true};
    hera::HeParameterConfig profile;
    profile.polynomial_degree = 8192;
    profile.coefficient_modulus_bits = {60, 60, 46, 46, 46, 60, 60};
    profile.input_scale_bits = 45;
    profile.default_scale_bits = 45;
    profile.output_scale_bits = 46;
    profile.declared_depth = 6;
    profile.terminal_prime_count = 1;
    profile.require_exact_terminal_level = true;
    profile.security_level = hera::SecurityLevel::kNone;
    config.he_parameter_profiles.emplace("ffn", profile);
    hera::LayerExecutionConfig execution;
    execution.layer_id = layer.id;
    for (const auto region : {hera::HeRegionId::kRegion1,
                              hera::HeRegionId::kRegion2,
                              hera::HeRegionId::kRegion3,
                              hera::HeRegionId::kRegion4,
                              hera::HeRegionId::kRegion5})
        execution.regions.emplace(
            region, hera::HeRegionConfig{region, "ffn",
                                         hera::Layout::kSpatial,
                                         hera::Layout::kSpatial});
    config.layer_execution = {execution};
    return config;
}

seal::parms_id_type LevelId(const hera::HeContext &context,
                            std::size_t target)
{
    auto data = context.context->first_context_data();
    for (std::size_t level = 0; level < target && data; ++level)
        data = data->next_context_data();
    if (!data) throw std::runtime_error("requested level is absent");
    return data->parms_id();
}

std::size_t ActualLevel(const hera::HeContext &context,
                        seal::parms_id_type id)
{
    std::size_t level = 0;
    for (auto data = context.context->first_context_data(); data;
         data = data->next_context_data(), ++level)
        if (data->parms_id() == id) return level;
    throw std::runtime_error("ciphertext level is outside context");
}

hera::SealCipherTensor Encrypt(
    const std::vector<double> &values, const hera::HePackingPlan &packing,
    const hera::HeContext &context, hera::Layout layout,
    std::size_t level = 0)
{
    Require(values.size() == packing.logical_locations.size(),
            "packing/value size mismatch");
    std::vector<std::vector<std::complex<double>>> slots(
        packing.ciphertext_count,
        std::vector<std::complex<double>>(packing.slot_count));
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
    const double scale = std::ldexp(1.0, context.parameters.input_scale_bits);
    for (std::size_t i = 0; i < ciphertexts.size(); ++i) {
        seal::Plaintext plain;
        encoder.encode(slots[i], LevelId(context, level), scale, plain);
        encryptor.encrypt_symmetric(plain, ciphertexts[i]);
    }
    hera::TensorSpec spec;
    spec.shape = packing.logical_shape;
    spec.domain = hera::Domain::kHeCiphertext;
    spec.layout = layout;
    spec.numeric = {43, 13, true};
    spec.he_region = hera::HeRegionId::kRegion5;
    spec.he_parameter_profile = context.profile;
    spec.planned_level = level;
    spec.planned_scale_bits = context.parameters.input_scale_bits;
    return {context.profile, std::move(ciphertexts), spec, packing};
}

std::vector<double> Decrypt(const hera::SealCipherTensor &tensor,
                            const hera::HeContext &context)
{
    seal::Decryptor decryptor(*context.context, *context.secret_key);
    seal::CKKSEncoder encoder(*context.context);
    std::vector<std::vector<std::complex<double>>> slots(
        tensor.ciphertexts.size());
    for (std::size_t i = 0; i < tensor.ciphertexts.size(); ++i) {
        seal::Plaintext plain;
        decryptor.decrypt(tensor.ciphertexts[i], plain);
        encoder.decode(plain, slots[i]);
    }
    std::vector<double> output(
        tensor.physical_packing.logical_locations.size());
    for (std::size_t logical = 0; logical < output.size(); ++logical) {
        const auto &location =
            tensor.physical_packing.Locations(logical).front();
        const auto value = slots[location.ciphertext][location.slot];
        output[logical] = location.component == hera::ComplexLane::kReal
            ? value.real() : value.imag();
    }
    return output;
}

double MaxError(const std::vector<double> &left,
                const std::vector<double> &right)
{
    Require(left.size() == right.size(), "error vector size mismatch");
    double error = 0.0;
    for (std::size_t i = 0; i < left.size(); ++i)
        error = std::max(error, std::abs(left[i] - right[i]));
    return error;
}

double F0(double x, const hera::GeluConfig &g)
{
    return g.a * x * x * x * x - g.b * x * x * x + g.c * x * x +
           (0.5 - g.d) * x + g.e;
}

double F1(double x, const hera::GeluConfig &g)
{
    return g.a * x * x * x * x + g.b * x * x * x + g.c * x * x +
           (0.5 + g.d) * x + g.e;
}

void TestPacking()
{
    hera::TransformerLayerConfig layer;
    layer.tokens = 3;
    layer.model_width = 7;
    layer.ffn_width = 9;
    auto plan = hera::FfnGeluPackingPlanner::Build(layer, 16);
    Require(plan.ok(), plan.status().message());
    Require(plan.value().token_stride == 4 &&
                plan.value().columns_per_component == 4,
            "column-first packing geometry is wrong");
    const auto &real = plan.value().centered.Locations(0).front();
    const auto &imag = plan.value().centered.Locations(4).front();
    Require(real.component == hera::ComplexLane::kReal && real.slot == 0,
            "first feature is not in the real lane");
    Require(imag.component == hera::ComplexLane::kImaginary && imag.slot == 0,
            "imaginary lane did not start after real columns");
    Require(plan.value().centered.Validate().ok() &&
                plan.value().projected.Validate().ok(),
            "FFN packing plan is invalid");
    Require(plan.value().projected.ciphertext_count == 3 &&
                plan.value().projected.layout == hera::Layout::kStrictColumn &&
                std::all_of(plan.value().projected.logical_locations.begin(),
                    plan.value().projected.logical_locations.end(), [](const auto &locations) {
                        return locations.front().component == hera::ComplexLane::kReal;
                    }),
            "Wu packing still pairs output features into complex lanes");
}

void TestImmutableParameterViews()
{
    hera::SealPlainParameterStore store;
    hera::WeightHandle matrix{4101, {"shared-Wu", 2, 3, true}};
    hera::PlainParameterHandle vector{4102, {"shared-gamma", 3}};
    Require(store.PutMatrix(matrix, {1, 2, 3, 4, 5, 6}).ok() &&
                store.PutVector(vector, {0.5, 1.0, 1.5}).ok(),
            "cannot register immutable parameter views");
    auto first_matrix = store.GetMatrixShared(matrix.id);
    auto second_matrix = store.GetMatrixShared(matrix.id);
    auto first_vector = store.GetVectorShared(vector.id);
    auto second_vector = store.GetVectorShared(vector.id);
    Require(first_matrix.ok() && second_matrix.ok() &&
                first_vector.ok() && second_vector.ok(),
            "cannot load immutable parameter views");
    Require(first_matrix.value().get() == second_matrix.value().get() &&
                first_vector.value().get() == second_vector.value().get(),
            "online parameter lookup copied a registered payload");
    Require(first_matrix.value()->row_major[5] == 6.0 &&
                first_vector.value()->values[2] == 1.5,
            "immutable parameter view changed its values");
    Require(!store.GetMatrixShared(999999).ok() &&
                !store.GetVectorShared(999999).ok(),
            "missing immutable parameter was accepted");
}

void TestGammaFusion()
{
    hera::SealPlainParameterStore store;
    hera::LayerWeightHandles handles;
    handles.wu = {5101, {"Wu", 6, 2, true}};
    handles.plain_parameters = {{5102, {"ln2.gamma", 6}}, {5103, {"ln2.beta", 6}}};
    const std::vector<double> gamma{0.0, 1e-12, -2.0, 1.5, 0.03125, 0.125}, beta{0.4, -0.3, 0.2, -0.1, 0.01, 10};
    const std::vector<double> weights{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    Require(store.PutMatrix(handles.wu, weights).ok() &&
                store.PutVector(handles.plain_parameters[0], gamma).ok() &&
                store.PutVector(handles.plain_parameters[1], beta).ok(), "fusion setup failed");
    auto original = store.GetMatrixShared(handles.wu.id);
    Require(store.FuseLayerNormGammaIntoWu(handles).ok(), "Wu gamma fusion failed");
    auto fused = store.GetMatrixShared(handles.wu.id);
    auto affine = store.GetGammaFusedLayerNorm(handles.wu.id);
    Require(fused.ok() && affine.ok(), "fused Wu/tail absent");
    Require(affine.value()->gamma.values[0] == 0 && affine.value()->gamma.values[1] == 1e-12 &&
                affine.value()->gamma.values[4] == gamma[4] && affine.value()->gamma.values[5] == gamma[5],
            "zero/tiny gamma must remain in the LN tail");
    Require(original.value()->row_major == weights, "fusion mutated an existing immutable view");
    for (double z : {-1.7, 0.0, 2.3})
        for (std::size_t col = 0; col < 2; ++col) {
            double expected = 0, actual = 0;
            for (std::size_t row = 0; row < gamma.size(); ++row) {
                expected += (z * gamma[row] + beta[row]) * weights[row * 2 + col];
                actual += (z * affine.value()->gamma.values[row] + affine.value()->beta.values[row]) *
                          fused.value()->row_major[row * 2 + col];
            }
            Require(std::abs(expected - actual) < 1e-12, "gamma fusion changed LN/Wu algebra");
        }
    Require(store.FuseLayerNormGammaIntoWu(handles).ok(), "gamma fusion is not idempotent");
    Require(store.GetMatrixShared(handles.wu.id).value().get() == fused.value().get(),
            "second gamma fusion replaced already-fused weights");
    hera::LayerWeightHandles qkv;
    qkv.wq = {5201, {"Wq", 6, 2, true}};
    qkv.wk = {5202, {"Wk", 6, 2, true}};
    qkv.wv = {5203, {"Wv", 6, 2, true}};
    qkv.plain_parameters = {{5204, {"ln1.gamma", 6}}, {5205, {"ln1.beta", 6}}};
    for (const auto *handle : {&qkv.wq, &qkv.wk, &qkv.wv})
        Require(store.PutMatrix(*handle, weights).ok(), "QKV stability fixture failed");
    Require(store.PutVector(qkv.plain_parameters[0], gamma).ok() &&
                store.PutVector(qkv.plain_parameters[1], beta).ok() &&
                store.FuseLayerNormGammaIntoQkv(qkv).ok(), "QKV stable fusion failed");
    for (const auto *handle : {&qkv.wq, &qkv.wk, &qkv.wv}) {
        const auto qkv_affine = store.GetGammaFusedLayerNorm(handle->id).value();
        Require(qkv_affine->gamma.values == affine.value()->gamma.values &&
                    qkv_affine->beta.values == affine.value()->beta.values &&
                    store.GetMatrixShared(handle->id).value()->row_major == fused.value()->row_major,
                "LN1 and LN2 applied different gamma fusion policies");
    }
    hera::SealPlainParameterStore overflowing;
    for (const auto *handle : {&qkv.wq, &qkv.wk, &qkv.wv})
        Require(overflowing.PutMatrix(*handle,
            std::vector<double>(weights.size(), std::numeric_limits<double>::max())).ok(), "overflow fixture failed");
    Require(overflowing.PutVector(qkv.plain_parameters[0], gamma).ok() &&
                overflowing.PutVector(qkv.plain_parameters[1], beta).ok(), "overflow affine fixture failed");
    const auto original_q = overflowing.GetMatrixShared(qkv.wq.id).value();
    Require(!overflowing.FuseLayerNormGammaIntoQkv(qkv).ok() &&
                overflowing.GetMatrixShared(qkv.wq.id).value() == original_q &&
                !overflowing.GetGammaFusedLayerNorm(qkv.wq.id).ok(),
            "overflowing QKV fusion published partial parameters");
}

void TestHeKernels()
{
    auto config = MakeConfig();
    Require(hera::ValidateHeraConfig(config).ok(), "FFN config is invalid");
    const auto &layer = config.layers.front();
    const auto &profile = config.he_parameter_profiles.at("ffn");
    Require(hera::FfnGeluLevelPlanner::Validate(profile).ok(),
            "valid FFN level plan was rejected");
    auto removed_complex_profile = profile;
    removed_complex_profile.coefficient_modulus_bits = {60, 60, 46, 46, 46, 46, 60};
    Require(!hera::FfnGeluLevelPlanner::Validate(removed_complex_profile).ok(),
            "old complex-Wu modulus suffix was accepted by the real-Wu planner");
    auto short_profile = profile;
    short_profile.coefficient_modulus_bits.pop_back();
    Require(hera::FfnGeluLevelPlanner::Validate(short_profile).ok(),
            "valid six-prime FFN suffix was rejected");
    short_profile.coefficient_modulus_bits.pop_back();
    Require(!hera::FfnGeluLevelPlanner::Validate(short_profile).ok(),
            "short FFN level plan was accepted");

    hera::HeContextRegistry client;
    Require(client.CreateClientProfiles(config).ok(),
            "cannot create FFN client profile");
    auto steps = hera::SealSciFfnGeluExecutor::RequiredRotationSteps(
        layer, profile);
    Require(steps.ok(), steps.status().message());
    auto wd_input_plan = hera::FfnGeluPackingPlanner::ComplexColumns(
        {{layer.tokens, layer.ffn_width}}, profile.polynomial_degree / 2,
        hera::Layout::kFfn);
    auto wd_output_plan = hera::FfnGeluPackingPlanner::ComplexColumns(
        {{layer.tokens, layer.model_width}}, profile.polynomial_degree / 2,
        hera::Layout::kComplexStrictColumn);
    Require(wd_input_plan.ok() && wd_output_plan.ok(),
            "cannot build Wd packing plans");
    auto wd_steps = hera::ComplexColumnProjectionKernel::RequiredRotationSteps(
        wd_input_plan.value(), wd_output_plan.value());
    Require(wd_steps.ok(), wd_steps.status().message());
    steps.value().insert(steps.value().end(), wd_steps.value().begin(),
                         wd_steps.value().end());
    std::sort(steps.value().begin(), steps.value().end());
    steps.value().erase(std::unique(steps.value().begin(), steps.value().end()),
                        steps.value().end());
    Require(client.CompileRotationKeys("ffn", steps.value()).ok(),
            "cannot create FFN Galois keys");
    auto bundle = client.ExportPublicBundle();
    Require(bundle.ok(), bundle.status().message());
    hera::HeContextRegistry server;
    Require(server.CreateServerProfiles(config, bundle.value()).ok(),
            "cannot create FFN server profile");
    const auto &client_context = client.Require(
        "ffn", hera::PartyRole::kClient);
    const auto &server_context = server.Require(
        "ffn", hera::PartyRole::kServer);
    Require(!server_context.has_secret_key(),
            "FFN server unexpectedly owns the secret key");

    // Force a feature into the imaginary half and verify that realification
    // creates the second real-only ciphertext without decrypting at runtime.
    auto imaginary_layer = layer;
    imaginary_layer.ffn_width =
        client_context.slot_count() / 2 + 1;
    auto imaginary_plan = hera::FfnGeluPackingPlanner::Build(
        imaginary_layer, client_context.slot_count());
    Require(imaginary_plan.ok(), imaginary_plan.status().message());
    std::vector<double> imaginary_values(
        imaginary_layer.tokens * imaginary_layer.ffn_width, 0.0);
    imaginary_values[imaginary_layer.ffn_width - 1] = 0.375;
    imaginary_values[2 * imaginary_layer.ffn_width - 1] = -0.625;
    auto imaginary_input_plan = hera::FfnGeluPackingPlanner::ComplexColumns(
        {{imaginary_layer.tokens, imaginary_layer.ffn_width}}, client_context.slot_count(),
        hera::Layout::kComplexStrictColumn);
    Require(imaginary_input_plan.ok(), imaginary_input_plan.status().message());
    auto imaginary_cipher = Encrypt(
        imaginary_values, imaginary_input_plan.value(), client_context,
        hera::Layout::kComplexStrictColumn, 2);
    auto imaginary_real = hera::ComplexColumnRealifyKernel::Evaluate(
        imaginary_cipher, imaginary_plan.value().projected, profile.output_scale_bits, server_context);
    Require(imaginary_real.ok(), imaginary_real.status().message());
    const auto imaginary_decoded = Decrypt(
        imaginary_real.value(), client_context);
    const auto imaginary_error = MaxError(imaginary_decoded,
                                           imaginary_values);
    Require(imaginary_error < 5.0e-3,
            "imaginary FFN lane was not realified correctly: error=" +
                std::to_string(imaginary_error) + " actual=" +
                std::to_string(imaginary_decoded[
                    imaginary_layer.ffn_width - 1]));

    auto packing = hera::FfnGeluPackingPlanner::Build(
        layer, client_context.slot_count());
    Require(packing.ok(), packing.status().message());
    const std::vector<double> centered{
        -0.8, -0.2, 0.3, 0.7, 0.5, -0.6, 0.1, 0.0};
    const std::vector<double> inverse{1.15, 0.85};
    const std::vector<double> gamma{1.0, 0.9, 1.1, 0.8};
    const std::vector<double> beta{0.05, -0.04, 0.02, 0.01};
    auto centered_ct = Encrypt(centered, packing.value().centered,
                               client_context,
                               hera::Layout::kComplexStrictColumn, 1);
    auto inverse_ct = Encrypt(inverse, packing.value().inverse_broadcast,
                              client_context, hera::Layout::kReduction, 1);
    hera::DenseVectorParameter gamma_parameter{{"ln2.gamma", 4}, gamma};
    hera::DenseVectorParameter beta_parameter{{"ln2.beta", 4}, beta};
    hera::FfnGeluExecutionReport report;
    auto normalized = hera::FfnLayerNormTailKernel::Evaluate(
        centered_ct, inverse_ct, gamma_parameter, beta_parameter,
        layer, server_context, &report);
    Require(normalized.ok(), normalized.status().message());
    std::vector<double> expected_normalized(centered.size());
    for (std::size_t token = 0; token < layer.tokens; ++token)
        for (std::size_t feature = 0; feature < layer.model_width; ++feature) {
            const auto index = token * layer.model_width + feature;
            expected_normalized[index] =
                centered[index] * inverse[token] * gamma[feature] +
                beta[feature];
        }
    const auto normalized_values = Decrypt(normalized.value(), client_context);
    const auto normalized_error = MaxError(
        normalized_values, expected_normalized);
    Require(normalized_error < 1.0e-3,
            "LayerNorm tail numerical error exceeds 1e-3: " +
                std::to_string(normalized_error) + " actual0=" +
                std::to_string(normalized_values.front()) + " expected0=" +
                std::to_string(expected_normalized.front()));
    Require(ActualLevel(server_context,
                        normalized.value().ciphertexts.front().parms_id()) == 2,
            "LayerNorm tail did not finish at suffix level 2");

    std::vector<double> wu(layer.model_width * layer.ffn_width);
    for (std::size_t i = 0; i < wu.size(); ++i)
        wu[i] = (static_cast<int>(i % 7) - 3) * 0.035;
    const std::vector<double> bu{0.02, -0.01, 0.03, 0.0, -0.02};
    hera::DenseMatrixParameter wu_parameter{
        {"Wu", layer.model_width, layer.ffn_width, true}, wu};
    hera::DenseVectorParameter bu_parameter{{"bu", layer.ffn_width}, bu};
    {
        auto zero = wu_parameter;
        std::fill(zero.row_major.begin(), zero.row_major.end(), 0.0);
        hera::FfnGeluExecutionReport counts;
        auto constant = hera::FfnUpProjectionKernel::Evaluate(
            normalized.value(), zero, bu_parameter, layer, server_context, &counts);
        Require(constant.ok(), constant.status().message());
        std::vector<double> expected(layer.tokens * layer.ffn_width);
        for (std::size_t i = 0; i < expected.size(); ++i) expected[i] = bu[i % layer.ffn_width];
        Require(MaxError(Decrypt(constant.value(), client_context), expected) < 1e-7 &&
                    constant.value().logical_spec.planned_level == 3 && counts.rotations == 0 && counts.rescales == 0,
                "zero projection did not preserve bias, level or zero rotation budget");
        auto real = hera::FfnGeluPackingPlanner::RealColumns(
            {{layer.tokens, layer.ffn_width}}, server_context.slot_count(), hera::Layout::kStrictColumn);
        auto real_constant = hera::ComplexColumnProjectionKernel::Evaluate(normalized.value(),
            zero, bu_parameter, real.value(), hera::HeRegionId::kRegion5, 45, server_context);
        Require(real_constant.ok() && MaxError(Decrypt(real_constant.value(), client_context), expected) < 1e-7,
                "zero real-output projection changed bias");
        hera::SealPlainParameterStore prepared;
        prepared.BeginPlaintextPreparation();
        {
            hera::ModelPlaintextCacheScope scope(&prepared, "constant");
            Require(hera::FfnUpProjectionKernel::Evaluate(normalized.value(), zero,
                bu_parameter, layer, server_context).ok(), "constant projection preparation failed");
        }
        prepared.FreezePlaintextCache();
        {
            hera::ModelPlaintextCacheScope scope(&prepared, "constant");
            auto cached = hera::FfnUpProjectionKernel::Evaluate(normalized.value(), zero,
                bu_parameter, layer, server_context);
            Require(cached.ok() && MaxError(Decrypt(cached.value(), client_context), expected) < 1e-7,
                "frozen constant projection plan failed");
        }
    }
    auto projected = hera::FfnUpProjectionKernel::Evaluate(
        normalized.value(), wu_parameter, bu_parameter,
        layer, server_context, &report);
    Require(projected.ok(), projected.status().message());
    std::vector<double> expected_projected(layer.tokens * layer.ffn_width);
    for (std::size_t token = 0; token < layer.tokens; ++token)
        for (std::size_t output = 0; output < layer.ffn_width; ++output) {
            double sum = bu[output];
            for (std::size_t feature = 0; feature < layer.model_width;
                 ++feature)
                sum += expected_normalized[token * layer.model_width + feature] *
                       wu[feature * layer.ffn_width + output];
            expected_projected[token * layer.ffn_width + output] = sum;
        }
    Require(MaxError(Decrypt(projected.value(), client_context),
                     expected_projected) < 5.0e-3,
            "Wu numerical error exceeds 5e-3");

    {
        hera::SealPlainParameterStore resident;
        resident.BeginPlaintextPreparation();
        {
            hera::ModelPlaintextCacheScope scope(&resident, "resident-wu");
            auto ready = hera::FfnUpProjectionKernel::Evaluate(normalized.value(), wu_parameter,
                bu_parameter, layer, server_context);
            Require(ready.ok(), ready.status().message());
        }
        resident.FreezePlaintextCache();
        const auto encoded_count = resident.encoded_plaintext_count();
        auto plan = resident.GetPreparedProjection("resident-wu/ffn/Wu");
        Require(encoded_count > 0 && plan.ok(), "projection plaintexts were not prepared");
        {
            hera::ModelPlaintextCacheScope scope(&resident, "resident-wu");
            auto result = hera::FfnUpProjectionKernel::Evaluate(normalized.value(), wu_parameter,
                bu_parameter, layer, server_context, nullptr, 2);
            Require(result.ok(), result.status().message());
            Require(MaxError(Decrypt(result.value(), client_context), expected_projected) < 5e-3 &&
                        resident.encoded_plaintext_count() == encoded_count,
                "resident projection changed Wu values or encoded plaintexts online");
        }
        resident.BeginPlaintextPreparation();
        resident.AbortPlaintextPreparation();
        auto restored = resident.GetPreparedProjection("resident-wu/ffn/Wu");
        Require(restored.ok() && restored.value() == plan.value() &&
                    resident.encoded_plaintext_count() == encoded_count,
            "aborting Prepare invalidated the resident projection plan");

        // A retained immutable plan owns its plaintexts even after the store
        // publishes a replacement cache and drops its old plaintext entries.
        resident.BeginPlaintextPreparation();
        resident.FreezePlaintextCache();
        Require(resident.encoded_plaintext_count() == 0,
            "replacement preparation retained the previous plaintext cache");
        auto retained = hera::EvaluatePreparedProjection(
            *plan.value(), normalized.value(), server_context, nullptr, nullptr, 2);
        Require(retained.ok(), retained.status().message());
        auto retained_output = projected.value();
        retained_output.ciphertexts = retained.take_value();
        Require(MaxError(Decrypt(retained_output, client_context), expected_projected) < 5e-3,
            "retained projection plan lost its resident plaintexts");
    }

    Require(projected.value().logical_spec.layout == hera::Layout::kStrictColumn &&
                projected.value().logical_spec.planned_scale_bits == 46,
            "Wu did not directly produce real columns at GeLU scale");
    auto &real = projected;

    // The same rectangular kernel supports the down projection at an exact
    // suffix level. Its output is directly add-compatible with the retained
    // residual, and accurate realification consumes exactly one more level.
    auto activation = projected.value();
    seal::Evaluator suffix_evaluator(*server_context.context);
    for (auto &ct : activation.ciphertexts)
        suffix_evaluator.mod_switch_to_inplace(ct, LevelId(server_context, 4));
    activation.logical_spec.planned_level = 4;
    std::vector<double> wd(layer.ffn_width * layer.model_width);
    for (std::size_t i = 0; i < wd.size(); ++i)
        wd[i] = (static_cast<int>(i % 5) - 2) * 0.025;
    const std::vector<double> bd{0.01, -0.02, 0.03, -0.01};
    hera::DenseMatrixParameter wd_parameter{
        {"Wd", layer.ffn_width, layer.model_width, true}, wd};
    hera::DenseVectorParameter bd_parameter{{"bd", layer.model_width}, bd};
    hera::HeKernelOperationCounts suffix_counts;
    auto down = hera::ComplexColumnProjectionKernel::Evaluate(
        activation, wd_parameter, bd_parameter, wd_output_plan.value(),
        hera::HeRegionId::kRegion4, profile.default_scale_bits,
        server_context, &suffix_counts);
    Require(down.ok(), down.status().message());
    Require(ActualLevel(server_context,
                        down.value().ciphertexts.front().parms_id()) == 5,
            "Wd did not land at the retained residual level");
    std::vector<double> residual(layer.tokens * layer.model_width);
    for (std::size_t i = 0; i < residual.size(); ++i)
        residual[i] = (static_cast<int>(i) - 3) * 0.04;
    auto residual_ct = Encrypt(residual, wd_output_plan.value(),
                               client_context,
                               hera::Layout::kComplexStrictColumn, 5);
    for (std::size_t i = 0; i < down.value().ciphertexts.size(); ++i) {
        Require(down.value().ciphertexts[i].parms_id() ==
                    residual_ct.ciphertexts[i].parms_id(),
                "Wd and residual parms_id differ");
        suffix_evaluator.add_inplace(down.value().ciphertexts[i],
                                     residual_ct.ciphertexts[i]);
    }
    auto wd_real_plan = hera::FfnGeluPackingPlanner::RealColumns(
        {{layer.tokens, layer.model_width}}, client_context.slot_count(),
        hera::Layout::kStrictColumn);
    Require(wd_real_plan.ok(), wd_real_plan.status().message());
    auto hidden = hera::ComplexColumnRealifyKernel::Evaluate(
        down.value(), wd_real_plan.value(), profile.default_scale_bits,
        server_context, &suffix_counts);
    Require(hidden.ok(), hidden.status().message());
    Require(ActualLevel(server_context,
                        hidden.value().ciphertexts.front().parms_id()) == 6,
            "accurate Block-5 realify did not consume one level");
    std::vector<double> expected_hidden(layer.tokens * layer.model_width);
    for (std::size_t token = 0; token < layer.tokens; ++token)
        for (std::size_t output = 0; output < layer.model_width; ++output) {
            double sum = bd[output] + residual[token * layer.model_width + output];
            for (std::size_t feature = 0; feature < layer.ffn_width; ++feature)
                sum += expected_projected[token * layer.ffn_width + feature] *
                       wd[feature * layer.model_width + output];
            expected_hidden[token * layer.model_width + output] = sum;
        }
    Require(MaxError(Decrypt(hidden.value(), client_context), expected_hidden) <
                5.0e-3,
            "Wd/residual/realify numerical error exceeds 5e-3");
    auto polynomials = hera::GeluPolynomialKernel::Evaluate(
        real.value(), config.gelu, server_context, &report);
    Require(polynomials.ok(), polynomials.status().message());
    std::vector<double> expected_f0(expected_projected.size());
    std::vector<double> expected_f1(expected_projected.size());
    for (std::size_t i = 0; i < expected_projected.size(); ++i) {
        expected_f0[i] = F0(expected_projected[i], config.gelu);
        expected_f1[i] = F1(expected_projected[i], config.gelu);
    }
    auto expected_branches = expected_f0;
    expected_branches.insert(expected_branches.end(), expected_f1.begin(), expected_f1.end());
    Require(MaxError(Decrypt(polynomials.value().branches, client_context), expected_branches) < 1.0e-2,
            "packed HE GeLU branch error exceeds 1e-2");
    Require(polynomials.value().branches.ciphertexts.size() == real.value().ciphertexts.size(),
            "GeLU packed branches must use one ciphertext per input");
    auto parallel_input = real.value();
    parallel_input.ciphertexts.push_back(
        parallel_input.ciphertexts.front());
    parallel_input.logical_spec.shape =
        {{layer.tokens, 2 * layer.ffn_width}};
    auto &parallel_packing = parallel_input.physical_packing;
    const auto original_packing = real.value().physical_packing;
    parallel_packing.logical_shape = parallel_input.logical_spec.shape;
    parallel_packing.ciphertext_count = 2;
    parallel_packing.logical_locations.resize(
        layer.tokens * 2 * layer.ffn_width);
    std::vector<double> parallel_f0(
        layer.tokens * 2 * layer.ffn_width);
    std::vector<double> parallel_f1(parallel_f0.size());
    for (std::size_t token = 0; token < layer.tokens; ++token)
        for (std::size_t feature = 0; feature < layer.ffn_width; ++feature)
            for (std::size_t copy = 0; copy < 2; ++copy) {
                const auto old_index = token * layer.ffn_width + feature;
                const auto new_index = token * 2 * layer.ffn_width +
                    copy * layer.ffn_width + feature;
                auto location = original_packing.Locations(old_index).front();
                location.ciphertext = copy;
                parallel_packing.logical_locations[new_index] = {location};
                parallel_f0[new_index] = expected_f0[old_index];
                parallel_f1[new_index] = expected_f1[old_index];
            }
    hera::FfnGeluExecutionReport parallel_report;
    auto parallel_polynomials = hera::GeluPolynomialKernel::Evaluate(
        parallel_input, config.gelu, server_context, &parallel_report, 2);
    Require(parallel_polynomials.ok(),
            parallel_polynomials.status().message());
    Require(parallel_report.ciphertext_multiplications == 6 &&
                parallel_report.ciphertext_plain_multiplications == 8 &&
                parallel_report.rescales == 8,
            "parallel GeLU operation accounting is wrong");
    auto parallel_branches = parallel_f0;
    parallel_branches.insert(parallel_branches.end(), parallel_f1.begin(), parallel_f1.end());
    Require(MaxError(Decrypt(parallel_polynomials.value().branches, client_context), parallel_branches) < 1.0e-2,
            "parallel packed GeLU branch error exceeds 1e-2");
    for (const auto *tensor : {&polynomials.value().input, &polynomials.value().branches})
        for (const auto &ct : tensor->ciphertexts)
            Require(ActualLevel(server_context, ct.parms_id()) ==
                        profile.coefficient_modulus_bits.size() - 1,
                    "GeLU output is not at the single-prime terminal level");
    Require(report.ciphertext_multiplications != 0 &&
                report.relinearizations != 0 && report.rescales != 0,
            "FFN HE operation report is empty");
    // Compile the complete suffix on metadata, then execute it with fused Wu.
    // This verifies prepared constants, packed branches and all five levels.
    hera::SealPlainParameterStore store;
    hera::LayerWeightHandles handles;
    handles.wu = {5201, wu_parameter.spec};
    handles.plain_parameters = {{5202, gamma_parameter.spec}, {5203, beta_parameter.spec}};
    Require(store.PutMatrix(handles.wu, wu).ok() &&
                store.PutVector(handles.plain_parameters[0], gamma).ok() &&
                store.PutVector(handles.plain_parameters[1], beta).ok() &&
                store.FuseLayerNormGammaIntoWu(handles).ok(), "prepared suffix fusion failed");
    const auto fused = store.GetMatrixShared(handles.wu.id).value();
    const auto affine = store.GetGammaFusedLayerNorm(handles.wu.id).value();
    auto metadata = [](const hera::SealCipherTensor &source) {
        auto output = source;
        for (auto &ct : output.ciphertexts) {
            const auto id = ct.parms_id(); const auto scale = ct.scale();
            ct = seal::Ciphertext(); ct.parms_id() = id; ct.scale() = scale;
        }
        return output;
    };
    auto suffix = [&](const hera::SealCipherTensor &c, const hera::SealCipherTensor &r,
                      hera::FfnGeluExecutionReport *tail_report) {
        auto ln = hera::FfnLayerNormTailKernel::Evaluate(c, r, affine->gamma, affine->beta,
                                                       layer, server_context, tail_report, 2);
        Require(ln.ok(), ln.status().message());
        auto up = hera::FfnUpProjectionKernel::Evaluate(ln.value(), *fused, bu_parameter,
                                                        layer, server_context, nullptr, 2);
        Require(up.ok(), up.status().message());
        auto poly = hera::GeluPolynomialKernel::Evaluate(up.value(), config.gelu, server_context,
                                                         nullptr, 2);
        Require(poly.ok(), poly.status().message());
        return poly.take_value();
    };
    store.BeginPlaintextPreparation();
    {
        hera::ModelPlaintextCacheScope scope(&store, "prepared-suffix");
        suffix(metadata(centered_ct), metadata(inverse_ct), nullptr);
    }
    store.FreezePlaintextCache();
    {
        hera::ModelPlaintextCacheScope scope(&store, "prepared-suffix");
        hera::FfnGeluExecutionReport tail_report;
        auto result = suffix(centered_ct, inverse_ct, &tail_report);
        Require(tail_report.conjugations == 0 && tail_report.ciphertext_plain_multiplications == 1,
                "gamma-fused LN tail still splits its complex components");
        Require(MaxError(Decrypt(result.branches, client_context), expected_branches) < 1e-2,
                "prepared gamma-fused suffix differs from original LN/Wu oracle");
        auto changed = config.gelu; changed.b += 0.01;
        auto rejected = hera::GeluPolynomialKernel::Evaluate(real.value(), changed, server_context);
        Require(!rejected.ok(), "frozen polynomial accepted different coefficients");
    }

}

void TestNormalizedFfnTail()
{
    auto config = MakeConfig();
    auto &layer = config.layers.front();
    layer.tokens = 128;
    layer.model_width = 65;
    hera::HeContextRegistry client, server;
    Require(client.CreateClientProfiles(config).ok(), "dual LN2 client context failed");
    Require(client.CompileRotationKeys("ffn", {0}).ok(), "dual LN2 conjugation key failed");
    auto bundle = client.ExportPublicBundle();
    Require(bundle.ok() && server.CreateServerProfiles(config, bundle.value()).ok(), "dual LN2 server failed");
    const auto &cc = client.Require("ffn", hera::PartyRole::kClient);
    const auto &sc = server.Require("ffn", hera::PartyRole::kServer);
    auto packing = hera::FfnGeluPackingPlanner::Build(layer, cc.slot_count());
    Require(packing.ok() && packing.value().centered.ciphertext_count == 2,
            "dual LN2 fixture must cover both complex lanes and a tail");
    std::vector<double> centered(layer.tokens * layer.model_width), inverse(layer.tokens);
    hera::GammaFusedLayerNormParameters original{
        {{"ln2.gamma", layer.model_width}, std::vector<double>(layer.model_width)},
        {{"ln2.beta", layer.model_width}, std::vector<double>(layer.model_width)}};
    auto fused = original;
    for (std::size_t f = 0; f < layer.model_width; ++f) {
        original.gamma.values[f] = f < 32 ? 0.0 : (f % 3 ? -1.3 : 1e-12);
        if (f == 64) original.gamma.values[f] = 1.25;
        original.beta.values[f] = (static_cast<int>(f % 7) - 3) * 0.02;
        const auto g = original.gamma.values[f];
        fused.gamma.values[f] = std::abs(g) < 0.0625 ? g : 1.0;
        fused.beta.values[f] = std::abs(g) < 0.0625 ? original.beta.values[f] : original.beta.values[f] / g;
    }
    std::vector<double> expected(centered.size()), projection_expected(centered.size());
    for (std::size_t t = 0; t < layer.tokens; ++t) {
        inverse[t] = 0.8 + (t % 5) * 0.03;
        for (std::size_t f = 0; f < layer.model_width; ++f) {
            const auto i = t * layer.model_width + f;
            centered[i] = (static_cast<int>((t + f) % 11) - 5) * 0.04;
            expected[i] = centered[i] * inverse[t] * original.gamma.values[f] + original.beta.values[f];
            projection_expected[i] = centered[i] * inverse[t] * fused.gamma.values[f] + fused.beta.values[f];
        }
    }
    auto c = Encrypt(centered, packing.value().centered, cc, hera::Layout::kComplexStrictColumn, 1);
    auto r = Encrypt(inverse, packing.value().inverse_broadcast, cc, hera::Layout::kReduction, 1);
    // Dirty unused complex lanes; only registered logical features may survive.
    seal::CKKSEncoder encoder(*cc.context);
    seal::Evaluator evaluator(*cc.context);
    std::vector<std::vector<std::complex<double>>> padding(c.ciphertexts.size(),
        std::vector<std::complex<double>>(cc.slot_count(), {9.0, -9.0}));
    for (const auto &locations : c.physical_packing.logical_locations) {
        const auto &loc = locations.front();
        if (loc.component == hera::ComplexLane::kReal) padding[loc.ciphertext][loc.slot].real(0.0);
        else padding[loc.ciphertext][loc.slot].imag(0.0);
    }
    for (std::size_t ct = 0; ct < c.ciphertexts.size(); ++ct) {
        seal::Plaintext plain;
        encoder.encode(padding[ct], c.ciphertexts[ct].parms_id(), c.ciphertexts[ct].scale(), plain);
        evaluator.add_plain_inplace(c.ciphertexts[ct], plain);
    }
    hera::SealCipherTensor residual;
    hera::FfnGeluExecutionReport report;
    auto projected = hera::FfnLayerNormTailKernel::Evaluate(c, r, fused.gamma, fused.beta,
        layer, sc, &report, 2, &original, &residual);
    Require(projected.ok(), projected.status().message());
    const auto error = MaxError(Decrypt(residual, cc), expected);
    Require(error < 2e-3 && MaxError(Decrypt(projected.value(), cc), projection_expected) < 2e-3 &&
                residual.logical_spec.layout == hera::Layout::kStrictColumn &&
                residual.ciphertexts.size() == 3 && residual.logical_spec.planned_level == 2 &&
                projected.value().logical_spec.planned_level == 2 &&
                report.ciphertext_multiplications == c.ciphertexts.size() &&
                report.relinearizations == c.ciphertexts.size(),
            "dual LN2 residual differs from original affine, padding, shared product, or depth contract");
    hera::SealPlainParameterStore store;
    store.BeginPlaintextPreparation();
    {
        hera::ModelPlaintextCacheScope scope(&store, "dual-ln2");
        auto ready = hera::FfnLayerNormTailKernel::Evaluate(c, r, fused.gamma, fused.beta,
            layer, sc, nullptr, 2, &original, &residual);
        Require(ready.ok(), ready.status().message());
    }
    store.FreezePlaintextCache();
    {
        hera::ModelPlaintextCacheScope scope(&store, "dual-ln2");
        auto ready = hera::FfnLayerNormTailKernel::Evaluate(c, r, fused.gamma, fused.beta,
            layer, sc, nullptr, 2, &original, &residual);
        Require(ready.ok() && MaxError(Decrypt(residual, cc), expected) < 2e-3,
                "prepared LN2 full residual differs from oracle");
        auto changed = original;
        changed.beta.values.back() += 0.1;
        Require(!hera::FfnLayerNormTailKernel::Evaluate(c, r, fused.gamma, fused.beta,
            layer, sc, nullptr, 2, &changed, &residual).ok(),
            "frozen LN2 program accepted changed original affine");
    }
    std::cout << "LN2_NORMALIZED_RESIDUAL_MAX_ERROR=" << error << '\n';
}

void TestParallelColumnProjection()
{
    auto config = MakeConfig();
    const auto &profile = config.he_parameter_profiles.at("ffn");
    constexpr std::size_t tokens = 128;
    // Exercise both source lanes, multiple ciphertexts and a partial tail.
    constexpr std::size_t input_width = 65;
    constexpr std::size_t output_width = 65;
    auto input_plan = hera::FfnGeluPackingPlanner::ComplexColumns(
        {{tokens, input_width}}, profile.polynomial_degree / 2,
        hera::Layout::kComplexStrictColumn);
    auto output_plan = hera::FfnGeluPackingPlanner::RealColumns(
        {{tokens, output_width}}, profile.polynomial_degree / 2,
        hera::Layout::kStrictColumn);
    Require(input_plan.ok() && output_plan.ok(),
            "parallel projection packing failed");
    Require(output_plan.value().ciphertext_count > 1,
            "parallel projection test needs multiple output groups");
    auto steps = hera::ComplexColumnProjectionKernel::RequiredRotationSteps(
        input_plan.value(), output_plan.value());
    Require(steps.ok(), steps.status().message());
    auto thin_plan = hera::FfnGeluPackingPlanner::ComplexColumns(
        {{tokens, 4}}, profile.polynomial_degree / 2, hera::Layout::kComplexStrictColumn);
    Require(thin_plan.ok(), thin_plan.status().message());
    auto thin_steps = hera::ComplexColumnProjectionKernel::RequiredRotationSteps(
        thin_plan.value(), output_plan.value());
    Require(thin_steps.ok(), thin_steps.status().message());
    steps.value().insert(steps.value().end(), thin_steps.value().begin(), thin_steps.value().end());
    std::sort(steps.value().begin(), steps.value().end());
    steps.value().erase(std::unique(steps.value().begin(), steps.value().end()), steps.value().end());
    hera::HeContextRegistry client;
    Require(client.CreateClientProfiles(config, false).ok(),
            "parallel projection client setup failed");
    Require(client.CompileRotationKeys("ffn", steps.value()).ok(),
            "parallel projection rotation keys failed");
    auto bundle = client.ExportPublicBundle();
    Require(bundle.ok(), bundle.status().message());
    hera::HeContextRegistry server;
    Require(server.CreateServerProfiles(config, bundle.value()).ok(),
            "parallel projection server setup failed");
    const auto &client_context = client.Require("ffn", hera::PartyRole::kClient);
    const auto &server_context = server.Require("ffn", hera::PartyRole::kServer);
    std::vector<double> values(tokens * input_width);
    for (std::size_t i = 0; i < values.size(); ++i)
        values[i] = (static_cast<int>(i % 9) - 4) * 0.004;
    auto input = Encrypt(values, input_plan.value(), client_context,
                         hera::Layout::kComplexStrictColumn, 2);
    std::vector<double> weight_values(input_width * output_width);
    for (std::size_t i = 0; i < weight_values.size(); ++i)
        weight_values[i] = (static_cast<int>(i % 7) - 3) * 0.006;
    hera::DenseMatrixParameter weights{
        {"parallel", input_width, output_width, true}, weight_values};
    auto wu_layer = config.layers.front();
    wu_layer.tokens = tokens; wu_layer.model_width = input_width; wu_layer.ffn_width = output_width;
    hera::FfnGeluExecutionReport serial_counts, parallel_counts;
    auto serial = hera::FfnUpProjectionKernel::Evaluate(
        input, weights, std::nullopt, wu_layer, server_context, &serial_counts, 1);
    auto parallel = hera::FfnUpProjectionKernel::Evaluate(
        input, weights, std::nullopt, wu_layer, server_context, &parallel_counts, 4);
    Require(serial.ok() && parallel.ok(),
            serial.ok() ? parallel.status().message() : serial.status().message());
    std::vector<double> expected(tokens * output_width);
    for (std::size_t token = 0; token < tokens; ++token)
        for (std::size_t output = 0; output < output_width; ++output)
            for (std::size_t feature = 0; feature < input_width; ++feature)
                expected[token * output_width + output] += values[token * input_width + feature] *
                    weight_values[feature * output_width + output];
    Require(MaxError(Decrypt(serial.value(), client_context), expected) < 1e-4,
            "complex-column BSGS projection differs from plaintext matrix product");
    Require(MaxError(Decrypt(serial.value(), client_context),
                     Decrypt(parallel.value(), client_context)) < 1e-4,
            "parallel column projection differs from serial");
    Require(serial_counts.rotations == parallel_counts.rotations &&
                serial_counts.ciphertext_plain_multiplications ==
                    parallel_counts.ciphertext_plain_multiplications &&
                serial_counts.rescales == parallel_counts.rescales,
            "parallel column projection changed operation counts");

    // Unused imaginary lanes are not logical zeros by contract. A thin
    // complex input must extract its real component even if every feature
    // happens to fit in the real half of one ciphertext.
    std::vector<double> thin_values(tokens * 4, 0.02);
    auto thin = Encrypt(thin_values, thin_plan.value(), client_context,
                        hera::Layout::kComplexStrictColumn, 2);
    seal::CKKSEncoder encoder(*client_context.context);
    seal::Plaintext padding;
    encoder.encode(std::complex<double>{0, 0.375}, thin.ciphertexts.front().parms_id(),
                   thin.ciphertexts.front().scale(), padding);
    seal::Evaluator evaluator(*client_context.context);
    evaluator.add_plain_inplace(thin.ciphertexts.front(), padding);
    hera::DenseMatrixParameter thin_weights{{"thin", 4, output_width, true},
        std::vector<double>(weight_values.begin(), weight_values.begin() + 4 * output_width)};
    auto thin_result = hera::ComplexColumnProjectionKernel::Evaluate(thin, thin_weights,
        std::nullopt, output_plan.value(), hera::HeRegionId::kRegion5,
        profile.output_scale_bits, server_context, nullptr, 2);
    Require(thin_result.ok(), thin_result.status().message());
    std::vector<double> thin_expected(tokens * output_width);
    for (std::size_t token = 0; token < tokens; ++token)
        for (std::size_t output = 0; output < output_width; ++output)
            for (std::size_t feature = 0; feature < 4; ++feature)
                thin_expected[token * output_width + output] += thin_values[token * 4 + feature] *
                    thin_weights.row_major[feature * output_width + output];
    Require(MaxError(Decrypt(thin_result.value(), client_context), thin_expected) < 1e-4,
            "unused imaginary padding contaminated the thin projection");

    hera::SealPlainParameterStore prepared_store;
    prepared_store.BeginPlaintextPreparation();
    {
        hera::ModelPlaintextCacheScope scope(&prepared_store, "layer0");
        auto metadata = input;
        for (auto &ct : metadata.ciphertexts) {
            const auto id = ct.parms_id();
            const auto scale = ct.scale();
            ct = seal::Ciphertext();
            ct.parms_id() = id;
            ct.scale() = scale;
        }
        auto compiled = hera::ComplexColumnProjectionKernel::Evaluate(
            metadata, weights, std::nullopt, output_plan.value(),
            hera::HeRegionId::kRegion5, profile.output_scale_bits,
            server_context);
        Require(compiled.ok(), compiled.status().message());
    }
    Require(prepared_store.prepared_projection_count() == 1,
            "Prepare did not compile the column projection plan");
    prepared_store.FreezePlaintextCache();
    hera::HeKernelOperationCounts cached_counts;
    {
        hera::ModelPlaintextCacheScope scope(&prepared_store, "layer0");
        auto cached = hera::ComplexColumnProjectionKernel::Evaluate(
            input, weights, std::nullopt, output_plan.value(),
            hera::HeRegionId::kRegion5, profile.output_scale_bits,
            server_context, &cached_counts, 2);
        Require(cached.ok(), cached.status().message());
        Require(MaxError(Decrypt(serial.value(), client_context),
                         Decrypt(cached.value(), client_context)) < 1e-4,
                "cached projection differs from uncached projection");
        auto wrong_level = input;
        seal::Evaluator evaluator(*server_context.context);
        for (auto &ciphertext : wrong_level.ciphertexts)
            evaluator.mod_switch_to_next_inplace(ciphertext);
        auto rejected = hera::ComplexColumnProjectionKernel::Evaluate(
            wrong_level, weights, std::nullopt, output_plan.value(),
            hera::HeRegionId::kRegion5, profile.output_scale_bits,
            server_context);
        Require(!rejected.ok(), "cached projection accepted a wrong level");
    }
    Require(prepared_store.prepared_projection_hits() == 2,
            "cached projection lookup count is wrong");
    Require(cached_counts.rotations == serial_counts.rotations &&
                cached_counts.ciphertext_plain_multiplications ==
                    serial_counts.ciphertext_plain_multiplications,
            "cached projection changed HE operation counts");
}

std::uint64_t Encode(double value)
{
    return static_cast<std::uint64_t>(static_cast<std::int64_t>(
        std::llround(std::ldexp(value, 13)))) & kRingMask;
}

double Decode(std::uint64_t value)
{
    value &= kRingMask;
    const auto signed_value = value >= (std::uint64_t{1} << 42)
        ? static_cast<std::int64_t>(value - (std::uint64_t{1} << 43))
        : static_cast<std::int64_t>(value);
    return std::ldexp(static_cast<double>(signed_value), -13);
}

void WriteAll(int fd, const std::vector<std::uint64_t> &values)
{
    const auto *data = reinterpret_cast<const char *>(values.data());
    std::size_t bytes = values.size() * sizeof(std::uint64_t);
    while (bytes != 0) {
        const auto written = write(fd, data, bytes);
        if (written <= 0) _exit(91);
        data += written;
        bytes -= static_cast<std::size_t>(written);
    }
}

std::vector<std::uint64_t> ReadAll(int fd, std::size_t count)
{
    std::vector<std::uint64_t> values(count);
    auto *data = reinterpret_cast<char *>(values.data());
    std::size_t bytes = count * sizeof(std::uint64_t);
    while (bytes != 0) {
        const auto read_count = read(fd, data, bytes);
        Require(read_count > 0, "cannot read MPC child result");
        data += read_count;
        bytes -= static_cast<std::size_t>(read_count);
    }
    return values;
}

void RunMpcChild(hera::PartyRole role,
                 const std::vector<std::uint64_t> &x,
                 const std::vector<std::uint64_t> &f0,
                 const std::vector<std::uint64_t> &f1,
                 int output_fd, std::size_t workers)
{
    hera::MpcShareTensor tx{x, {{1, x.size()}}, {43, 13, true},
                            hera::ComplexComponent::kReal};
    hera::MpcShareTensor t0{f0, tx.physical_shape, tx.numeric,
                            hera::ComplexComponent::kReal};
    hera::MpcShareTensor t1{f1, tx.physical_shape, tx.numeric,
                            hera::ComplexComponent::kReal};
    std::vector<std::uint64_t> results;
    {
        auto session = std::make_shared<hera::SharedMpcSession>(role, "127.0.0.1", kMpcPort, workers);
        hera::MpcGeluRuntime runtime(session);
        for (int repeat = 0; repeat < 2; ++repeat) {
            hera::FfnGeluExecutionReport report;
            auto result = runtime.Evaluate(tx, t0, t1, 2.7, &report);
            if (!result.ok()) _exit(92);
            if (report.mpc_workers != std::min(workers, x.size())) _exit(93);
            results.insert(results.end(), result.value().local_share.begin(), result.value().local_share.end());
            runtime.ReleaseSession();
        }
    }
    WriteAll(output_fd, results);
    _exit(0);
}

void TestMpcGelu(std::size_t workers = 1, bool small = false)
{
    const hera::GeluConfig gelu;
    const double ulp = std::ldexp(1., -13);
    const std::vector<double> cases{-40., -4., -2.7 - ulp, -2.7,
        -2.7 + ulp, -1., -ulp, 0., ulp, 1., 2.7 - ulp, 2.7, 2.7 + ulp, 4., 40.,
        -std::ldexp(1., 29), -std::ldexp(1., 29) + ulp, std::ldexp(1., 29) - ulp};
    // Cover both a large uneven batch and a small batch that previously
    // failed to use requested workers because of the fixed 32768 grain.
    std::vector<double> x(workers == 1 ? cases.size() : small ? cases.size() + 1 : 65539);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = cases[i % cases.size()];
    std::vector<std::uint64_t> encoded_x, encoded_f0, encoded_f1;
    for (const auto value : x) {
        encoded_x.push_back(Encode(value));
        // Out-of-domain polynomial values are unused; finite sentinels
        // let the selector cover the complete signed ring, including INT_MIN.
        encoded_f0.push_back(Encode(std::abs(value) > 40 ? -0.75 : F0(value, gelu)));
        encoded_f1.push_back(Encode(std::abs(value) > 40 ? 0.5 : F1(value, gelu)));
    }
    auto split = [](const std::vector<std::uint64_t> &values,
                    std::uint64_t salt) {
        std::pair<std::vector<std::uint64_t>,
                  std::vector<std::uint64_t>> shares;
        shares.first.resize(values.size());
        shares.second.resize(values.size());
        for (std::size_t i = 0; i < values.size(); ++i) {
            shares.first[i] = (salt + 7919 * i) & kRingMask;
            shares.second[i] = (values[i] - shares.first[i]) & kRingMask;
        }
        return shares;
    };
    const auto sx = split(encoded_x, 11731);
    const auto s0 = split(encoded_f0, 27143);
    const auto s1 = split(encoded_f1, 39133);
    int server_pipe[2], client_pipe[2];
    Require(pipe(server_pipe) == 0 && pipe(client_pipe) == 0,
            "cannot create MPC result pipes");
    const auto server = fork();
    Require(server >= 0, "cannot fork MPC server");
    if (server == 0) {
        close(server_pipe[0]);
        close(client_pipe[0]); close(client_pipe[1]);
        RunMpcChild(hera::PartyRole::kServer, sx.first, s0.first, s1.first,
                    server_pipe[1], workers);
    }
    const auto client = fork();
    Require(client >= 0, "cannot fork MPC client");
    if (client == 0) {
        close(client_pipe[0]);
        close(server_pipe[0]); close(server_pipe[1]);
        RunMpcChild(hera::PartyRole::kClient, sx.second, s0.second, s1.second,
                    client_pipe[1], workers);
    }
    close(server_pipe[1]); close(client_pipe[1]);
    const auto server_result = ReadAll(server_pipe[0], 2 * x.size());
    const auto client_result = ReadAll(client_pipe[0], 2 * x.size());
    close(server_pipe[0]); close(client_pipe[0]);
    int server_status = 0, client_status = 0;
    waitpid(server, &server_status, 0);
    waitpid(client, &client_status, 0);
    Require(WIFEXITED(server_status) && WEXITSTATUS(server_status) == 0 &&
                WIFEXITED(client_status) && WEXITSTATUS(client_status) == 0,
            "MPC GeLU child failed");
    for (std::size_t i = 0; i < server_result.size(); ++i) {
        const auto j = i % x.size();
        const auto value = Decode(encoded_x[j]);
        const auto threshold = Decode(Encode(gelu.threshold));
        const auto expected = value < -threshold ? std::uint64_t{0}
            : value < 0 ? encoded_f0[j] : value <= threshold ? encoded_f1[j] : encoded_x[j];
        Require(((server_result[i] + client_result[i]) & kRingMask) == expected,
                "MPC GeLU selection is incorrect at index " +
                    std::to_string(i));
    }
}

struct IntegratedPacket {
    int ok{};
    std::uint64_t shares[10]{};
};

hera::LayerWeightHandles MakeIntegratedWeights(
    const hera::TransformerLayerConfig &layer)
{
    hera::LayerWeightHandles weights;
    weights.layer_id = layer.id;
    weights.wu = {101, {"Wu", layer.model_width, layer.ffn_width, true}};
    weights.plain_parameters = {
        {201, {"ln2.gamma", layer.model_width}},
        {202, {"ln2.beta", layer.model_width}},
        {203, {"bu", layer.ffn_width}}};
    return weights;
}

void RunIntegratedParty(
    hera::PartyRole role, const hera::HeraConfig &config,
    hera::HeContextRegistry &registry,
    const std::vector<std::uint64_t> &centered,
    const std::vector<std::uint64_t> &inverse,
    IntegratedPacket *packet, std::string *error)
{
    try {
        auto transport = hera::MakeNetIoTransport(
            role, role == hera::PartyRole::kClient ? "127.0.0.1" : "",
            19800);
        auto conversion = hera::ConversionConfig{43, 13, 40, 20, true};
        conversion.public_c2m_quarter_range = true;
        conversion.public_m2c_quarter_range = true;
        hera::BlbReferenceConverter converter(
            role, conversion, registry, transport,
            hera::MakeDeterministicRandomSourceForTesting(
                role == hera::PartyRole::kClient
                    ? std::vector<std::uint8_t>{11, 13, 17}
                    : std::vector<std::uint8_t>{19, 23, 29}),
            "127.0.0.1", 19810, 1);
        hera::MpcGeluRuntime mpc(role, "127.0.0.1", 19820);
        hera::SealPlainParameterStore parameters;
        hera::CryptoTensorStore tensors;
        const auto &layer = config.layers.front();
        const auto weights = MakeIntegratedWeights(layer);
        constexpr hera::TensorId centered_id = 8100001;
        constexpr hera::TensorId inverse_id = 8100002;
        Require(tensors.PutMpcShare(
                    centered_id,
                    {centered, {{layer.tokens, layer.model_width}},
                     config.mpc_numeric, hera::ComplexComponent::kReal}).ok(),
                "cannot store integrated centered shares");
        Require(tensors.PutMpcShare(
                    inverse_id,
                    {inverse, {{layer.tokens, 1}}, config.mpc_numeric,
                     hera::ComplexComponent::kReal}).ok(),
                "cannot store integrated inverse shares");
        if (role == hera::PartyRole::kServer) {
            const std::vector<double> gamma{1.0, 0.9, 1.1, 0.8};
            const std::vector<double> beta{0.05, -0.04, 0.02, 0.01};
            std::vector<double> wu(layer.model_width * layer.ffn_width);
            for (std::size_t i = 0; i < wu.size(); ++i)
                wu[i] = (static_cast<int>(i % 7) - 3) * 0.035;
            const std::vector<double> bu{0.02, -0.01, 0.03, 0.0, -0.02};
            Require(parameters.PutMatrix(weights.wu, wu).ok() &&
                        parameters.PutVector(weights.plain_parameters[0],
                                             gamma).ok() &&
                        parameters.PutVector(weights.plain_parameters[1],
                                             beta).ok() &&
                        parameters.PutVector(weights.plain_parameters[2],
                                             bu).ok(),
                    "cannot register integrated FFN parameters");
        }
        hera::SealSciFfnGeluExecutor executor(
            config, layer, registry, parameters, tensors, converter, mpc);
        hera::TensorSpec centered_spec;
        centered_spec.shape = {{layer.tokens, layer.model_width}};
        centered_spec.domain = hera::Domain::kMpcArithmeticShare;
        centered_spec.layout = hera::Layout::kStrictColumn;
        centered_spec.numeric = config.mpc_numeric;
        hera::TensorSpec inverse_spec = centered_spec;
        inverse_spec.shape = {{layer.tokens, 1}};
        inverse_spec.layout = hera::Layout::kReduction;
        hera::FfnGeluExecutionReport report;
        auto output = executor.Evaluate(
            {{centered_id, centered_spec}, {inverse_id, inverse_spec}},
            weights, &report);
        Require(output.ok(), output.status().message());
        auto shares = tensors.GetMpcShare(output.value().activation_shares.id);
        Require(shares.ok() && shares.value().local_share.size() == 10,
                "integrated FFN output shares are absent");
        Require(shares.value().public_value_bound_bits == std::optional<int>{39} &&
                    converter.stats().bounded_field_to_ring_elements != 0,
                "quarter-q range was not propagated through the GeLU selection");
        std::copy(shares.value().local_share.begin(),
                  shares.value().local_share.end(), packet->shares);
        Require(report.mpc_to_ckks_calls == 1 &&
                    report.ckks_to_mpc_calls == 1 &&
                    report.comparisons == 20 && report.muxes == 30,
                "integrated FFN operation report is inconsistent");
        if (role == hera::PartyRole::kServer)
            Require(report.ciphertext_multiplications != 0 &&
                        report.rescales != 0,
                    "server did not execute FFN HE kernels");
        else
            Require(report.ciphertext_multiplications == 0 &&
                        report.rescales == 0,
                    "client unexpectedly executed FFN HE kernels");
        packet->ok = 1;
    } catch (const std::exception &exception) {
        *error = exception.what();
    }
}

void TestIntegratedExecutor()
{
    auto config = MakeConfig();
    auto &layer = config.layers.front();
    hera::HeContextRegistry client_registry;
    Require(client_registry.CreateClientProfiles(config).ok(),
            "cannot create integrated FFN client profiles");
    auto steps = hera::SealSciFfnGeluExecutor::RequiredRotationSteps(
        layer, config.he_parameter_profiles.at("ffn"));
    Require(steps.ok() &&
                client_registry.CompileRotationKeys("ffn", steps.value()).ok(),
            "cannot compile integrated FFN rotation keys");
    auto bundle = client_registry.ExportSerializedPublicBundle();
    Require(bundle.ok(), bundle.status().message());
    hera::HeContextRegistry server_registry;
    Require(server_registry.CreateServerProfiles(config, bundle.value()).ok(),
            "cannot create integrated FFN server profiles");

    const std::vector<double> centered_values{
        -0.8, -0.2, 0.3, 0.7, 0.5, -0.6, 0.1, 0.0};
    const std::vector<double> inverse_values{1.15, 0.85};
    std::vector<std::uint64_t> centered_encoded, inverse_encoded;
    for (const auto value : centered_values) centered_encoded.push_back(Encode(value));
    for (const auto value : inverse_values) inverse_encoded.push_back(Encode(value));
    auto split = [](const std::vector<std::uint64_t> &values,
                    std::uint64_t salt) {
        std::pair<std::vector<std::uint64_t>,
                  std::vector<std::uint64_t>> shares;
        shares.first.resize(values.size());
        shares.second.resize(values.size());
        for (std::size_t i = 0; i < values.size(); ++i) {
            shares.first[i] = (salt + 104729 * i) & kRingMask;
            shares.second[i] = (values[i] - shares.first[i]) & kRingMask;
        }
        return shares;
    };
    const auto centered_shares = split(centered_encoded, 70001);
    const auto inverse_shares = split(inverse_encoded, 90001);

    int client_pipe[2];
    Require(pipe(client_pipe) == 0, "cannot create integrated FFN pipe");
    const auto client = fork();
    Require(client >= 0, "cannot fork integrated FFN client");
    if (client == 0) {
        close(client_pipe[0]);
        IntegratedPacket packet;
        std::string error;
        RunIntegratedParty(
            hera::PartyRole::kClient, config, client_registry,
            centered_shares.second, inverse_shares.second, &packet, &error);
        if (!error.empty())
            std::cerr << "integrated FFN client: " << error << '\n';
        const auto written = write(client_pipe[1], &packet, sizeof(packet));
        close(client_pipe[1]);
        _exit(packet.ok && written == static_cast<ssize_t>(sizeof(packet))
                  ? 0 : 93);
    }
    close(client_pipe[1]);
    IntegratedPacket server_packet;
    std::string server_error;
    RunIntegratedParty(
        hera::PartyRole::kServer, config, server_registry,
        centered_shares.first, inverse_shares.first,
        &server_packet, &server_error);
    if (!server_error.empty()) {
        kill(client, SIGTERM);
        int failed = 0;
        waitpid(client, &failed, 0);
        close(client_pipe[0]);
        throw std::runtime_error(server_error);
    }
    IntegratedPacket client_packet;
    const auto received = read(client_pipe[0], &client_packet,
                               sizeof(client_packet));
    close(client_pipe[0]);
    int status = 0;
    waitpid(client, &status, 0);
    Require(received == static_cast<ssize_t>(sizeof(client_packet)) &&
                server_packet.ok && client_packet.ok &&
                WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "integrated FFN client failed");

    const std::vector<double> gamma{1.0, 0.9, 1.1, 0.8};
    const std::vector<double> beta{0.05, -0.04, 0.02, 0.01};
    std::vector<double> wu(layer.model_width * layer.ffn_width);
    for (std::size_t i = 0; i < wu.size(); ++i)
        wu[i] = (static_cast<int>(i % 7) - 3) * 0.035;
    const std::vector<double> bu{0.02, -0.01, 0.03, 0.0, -0.02};
    std::vector<double> expected(layer.tokens * layer.ffn_width);
    for (std::size_t token = 0; token < layer.tokens; ++token) {
        for (std::size_t output = 0; output < layer.ffn_width; ++output) {
            double x = bu[output];
            for (std::size_t feature = 0; feature < layer.model_width;
                 ++feature) {
                const auto index = token * layer.model_width + feature;
                const auto normalized = centered_values[index] *
                    inverse_values[token] * gamma[feature] + beta[feature];
                x += normalized * wu[feature * layer.ffn_width + output];
            }
            expected[token * layer.ffn_width + output] =
                x < -config.gelu.threshold ? 0.0
                : (x < 0.0 ? F0(x, config.gelu)
                : (x <= config.gelu.threshold ? F1(x, config.gelu) : x));
        }
    }
    std::vector<double> actual(expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i)
        actual[i] = Decode(server_packet.shares[i] +
                           client_packet.shares[i]);
    const auto error = MaxError(actual, expected);
    Require(error < 0.02,
            "integrated FFN/GeLU error exceeds 0.02: " +
                std::to_string(error));
}

} // namespace

int main()
{
    try {
        TestPacking();
        TestImmutableParameterViews();
        TestGammaFusion();
        TestHeKernels();
        TestNormalizedFfnTail();
        TestParallelColumnProjection();
        TestMpcGelu();
        TestMpcGelu(3);
        TestMpcGelu(3, true);
        TestIntegratedExecutor();
        std::cout << "FFN_GELU_STATUS=PASS\n";
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << "FFN_GELU_STATUS=FAIL\n"
                  << "FFN_GELU_ERROR=" << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
