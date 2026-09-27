#include "hera/block2_he.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

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
    layer.value_head_width = 3;
    layer.ffn_width = 8;
    layer.norm = hera::NormPlacement::kPreNorm;
    layer.qkv_bias = true;
    config.layers = {layer};
    hera::HeParameterConfig profile;
    profile.polynomial_degree = 4096;
    profile.coefficient_modulus_bits = {60, 35, 60, 60, 60,
                                        60, 60, 60, 60, 60};
    profile.default_scale_bits = 44;
    profile.input_scale_bits = 44;
    profile.output_scale_bits = 44;
    profile.declared_depth = 7;
    profile.terminal_prime_count = 1;
    profile.require_exact_terminal_level = true;
    profile.security_level = hera::SecurityLevel::kNone;
    config.he_parameter_profiles.emplace("block2", profile);
    hera::LayerExecutionConfig execution;
    execution.layer_id = layer.id;
    for (const auto region : {hera::HeRegionId::kRegion1,
                              hera::HeRegionId::kRegion2,
                              hera::HeRegionId::kRegion3,
                              hera::HeRegionId::kRegion4,
                              hera::HeRegionId::kRegion5})
        execution.regions.emplace(
            region, hera::HeRegionConfig{region, "block2",
                                         hera::Layout::kSpatial,
                                         hera::Layout::kSpatial});
    config.layer_execution = {execution};
    config.threads = 1;
    return config;
}

hera::HePackingPlan MakeInterleavedSpatial(
    const hera::TensorShape &shape, std::size_t slots)
{
    const auto count = shape.ElementCount().value();
    hera::HePackingPlan plan;
    plan.layout = hera::Layout::kSpatial;
    plan.logical_shape = shape;
    plan.slot_count = slots;
    plan.ciphertext_count = (count + 2 * slots - 1) / (2 * slots);
    plan.logical_locations.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
        const auto ct = i / (2 * slots);
        const auto local = i % (2 * slots);
        plan.logical_locations[i].push_back(
            {ct, (local / 2) % slots,
             (local & 1U) == 0 ? hera::ComplexLane::kReal
                               : hera::ComplexLane::kImaginary,
             std::nullopt});
    }
    Require(plan.Validate().ok(), "interleaved spatial plan is invalid");
    return plan;
}

hera::SealCipherTensor Encrypt(
    const std::vector<double> &values, const hera::HePackingPlan &packing,
    const hera::HeContext &context, std::size_t level = 0)
{
    std::vector<std::vector<std::complex<double>>> slots(
        packing.ciphertext_count,
        std::vector<std::complex<double>>(packing.slot_count));
    for (std::size_t logical = 0; logical < values.size(); ++logical) {
        for (const auto &location : packing.Locations(logical)) {
            if (location.component == hera::ComplexLane::kReal)
                slots[location.ciphertext][location.slot].real(values[logical]);
            else
                slots[location.ciphertext][location.slot].imag(values[logical]);
        }
    }
    seal::CKKSEncoder encoder(*context.context);
    seal::Encryptor encryptor(*context.context, *context.secret_key);
    std::vector<seal::Ciphertext> ciphertexts(packing.ciphertext_count);
    const double scale = std::ldexp(1.0, context.parameters.input_scale_bits);
    for (std::size_t i = 0; i < ciphertexts.size(); ++i) {
        seal::Plaintext plain;
        encoder.encode(slots[i], scale, plain);
        encryptor.encrypt_symmetric(plain, ciphertexts[i]);
        if (level != 0) {
            auto data = context.context->first_context_data();
            for (std::size_t step = 0; step < level && data; ++step)
                data = data->next_context_data();
            Require(static_cast<bool>(data), "invalid requested encryption level");
            seal::Evaluator evaluator(*context.context);
            evaluator.mod_switch_to_inplace(ciphertexts[i], data->parms_id());
        }
    }
    hera::TensorSpec spec;
    spec.shape = packing.logical_shape;
    spec.domain = hera::Domain::kHeCiphertext;
    spec.layout = packing.layout;
    spec.numeric = {43, 13};
    spec.he_region = hera::HeRegionId::kRegion2;
    spec.he_parameter_profile = context.profile;
    spec.planned_level = level;
    spec.planned_scale_bits = context.parameters.input_scale_bits;
    return {context.profile, std::move(ciphertexts), spec, packing};
}

std::vector<double> Decrypt(
    const hera::SealCipherTensor &tensor, const hera::HeContext &context);
double MaxError(const std::vector<double> &a,
                const std::vector<double> &b);

void TestParallelQkGroups()
{
    auto config = MakeConfig();
    auto &layer = config.layers.front();
    layer.tokens = 32;
    layer.q_heads = 8;
    layer.kv_heads = 8;
    layer.qk_head_width = 2;
    layer.qkv_bias = false;
    const auto &profile = config.he_parameter_profiles.at("block2");
    const auto slots = profile.polynomial_degree / 2;
    auto q_plan = hera::MultiHeadPackingPlan::ProjectionOutput(
        layer, hera::Layout::kMultiHeadQ, slots);
    auto k_plan = hera::MultiHeadPackingPlan::ProjectionOutput(
        layer, hera::Layout::kMultiHeadK, slots);
    Require(q_plan.ok() && k_plan.ok(), "parallel QK packing failed");
    auto projection_input = MakeInterleavedSpatial(
        {{layer.tokens, layer.model_width}}, slots);
    auto steps = hera::SealBlock2Executor::RequiredRotationSteps(
        layer, profile, projection_input);
    Require(steps.ok(), steps.status().message());
    hera::HeContextRegistry client;
    Require(client.CreateClientProfiles(config).ok(), "QK client setup failed");
    Require(client.CompileRotationKeys("block2", steps.value()).ok(),
            "QK rotation keys failed");
    auto bundle = client.ExportPublicBundle();
    Require(bundle.ok(), bundle.status().message());
    hera::HeContextRegistry server;
    Require(server.CreateServerProfiles(config, bundle.value()).ok(),
            "QK server setup failed");
    const auto &client_context = client.Require("block2", hera::PartyRole::kClient);
    const auto &server_context = server.Require("block2", hera::PartyRole::kServer);
    std::vector<double> q(layer.tokens * hera::QueryWidth(layer));
    std::vector<double> k(layer.tokens * hera::KeyWidth(layer));
    for (std::size_t i = 0; i < q.size(); ++i) {
        q[i] = (static_cast<int>(i % 11) - 5) * 0.003;
        k[i] = (static_cast<int>(i % 13) - 6) * 0.002;
    }
    constexpr std::size_t kQkStartLevel = 4;
    auto q_cipher = Encrypt(q, q_plan.value(), client_context, kQkStartLevel);
    auto k_cipher = Encrypt(k, k_plan.value(), client_context, kQkStartLevel);
    hera::Block2ExecutionReport serial_report, parallel_report;
    auto serial = hera::MultiHeadQkKernel::Evaluate(
        q_cipher, k_cipher, layer, server_context, &serial_report, 1);
    auto parallel = hera::MultiHeadQkKernel::Evaluate(
        q_cipher, k_cipher, layer, server_context, &parallel_report, 2);
    Require(serial.ok() && parallel.ok(),
            serial.ok() ? parallel.status().message() : serial.status().message());
    auto serial_values = Decrypt(serial.value(), client_context);
    auto parallel_values = Decrypt(parallel.value(), client_context);
    Require(MaxError(serial_values, parallel_values) < 1e-4,
            "parallel QK differs from serial QK");
    Require(serial_report.rotations == parallel_report.rotations &&
                serial_report.ciphertext_multiplications ==
                    parallel_report.ciphertext_multiplications &&
                serial_report.rescales == parallel_report.rescales,
            "parallel QK changed operation counts");
}

std::vector<double> Decrypt(
    const hera::SealCipherTensor &tensor, const hera::HeContext &context)
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

hera::LayerWeightHandles RegisterWeights(
    const hera::TransformerLayerConfig &layer,
    hera::SealPlainParameterStore &store,
    std::vector<double> &wq, std::vector<double> &wk,
    std::vector<double> &wv)
{
    const auto schema = hera::DeriveWeightSchema(layer).value();
    hera::PlainParameterId id = 1;
    hera::LayerWeightHandles handles;
    handles.layer_id = layer.id;
    handles.wq = {id++, schema.wq};
    handles.wk = {id++, schema.wk};
    handles.wv = {id++, schema.wv};
    handles.wo = {id++, schema.wo};
    handles.wu = {id++, schema.wu};
    handles.wd = {id++, schema.wd};
    wq.resize(schema.wq.rows * schema.wq.columns);
    wk.resize(schema.wk.rows * schema.wk.columns);
    wv.resize(schema.wv.rows * schema.wv.columns);
    for (std::size_t i = 0; i < wq.size(); ++i)
        wq[i] = (static_cast<int>(i % 7) - 3) * 0.015;
    for (std::size_t i = 0; i < wk.size(); ++i)
        wk[i] = (static_cast<int>(i % 5) - 2) * 0.02;
    for (std::size_t i = 0; i < wv.size(); ++i)
        wv[i] = (static_cast<int>(i % 9) - 4) * 0.01;
    Require(store.PutMatrix(handles.wq, wq).ok(), "cannot register Wq");
    Require(store.PutMatrix(handles.wk, wk).ok(), "cannot register Wk");
    Require(store.PutMatrix(handles.wv, wv).ok(), "cannot register Wv");
    for (const auto &spec : schema.vectors) {
        hera::PlainParameterHandle handle{id++, spec};
        std::vector<double> values(spec.length, 0.0);
        if (spec.name == "ln1.gamma")
            for (std::size_t i = 0; i < values.size(); ++i)
                values[i] = 0.9 + 0.05 * static_cast<double>(i);
        else if (spec.name == "ln1.beta")
            for (std::size_t i = 0; i < values.size(); ++i)
                values[i] = -0.02 + 0.01 * static_cast<double>(i);
        else if (spec.name == "bq" || spec.name == "bk" ||
                 spec.name == "bv")
            for (std::size_t i = 0; i < values.size(); ++i)
                values[i] = (static_cast<int>(i % 3) - 1) * 0.004;
        Require(store.PutVector(handle, values).ok(),
                "cannot register vector " + spec.name);
        handles.plain_parameters.push_back(handle);
    }
    return handles;
}

std::vector<double> MatMul(const std::vector<double> &x,
                           std::size_t rows, std::size_t inner,
                           const std::vector<double> &w,
                           std::size_t columns)
{
    std::vector<double> result(rows * columns);
    for (std::size_t row = 0; row < rows; ++row)
        for (std::size_t column = 0; column < columns; ++column)
            for (std::size_t k = 0; k < inner; ++k)
                result[row * columns + column] +=
                    x[row * inner + k] * w[k * columns + column];
    return result;
}

double MaxError(const std::vector<double> &a,
                const std::vector<double> &b)
{
    Require(a.size() == b.size(), "error vectors have different sizes");
    double result = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        result = std::max(result, std::abs(a[i] - b[i]));
    return result;
}

std::size_t ActualLevel(const hera::HeContext &context,
                        seal::parms_id_type id)
{
    std::size_t level = 0;
    for (auto data = context.context->first_context_data(); data;
         data = data->next_context_data(), ++level)
        if (data->parms_id() == id) return level;
    throw std::runtime_error("ciphertext has an unknown parms_id");
}

void TestPackingVariants(const hera::TransformerLayerConfig &base,
                         std::size_t slots)
{
    for (const std::size_t kv_heads : {std::size_t{4}, std::size_t{2},
                                       std::size_t{1}}) {
        auto layer = base;
        layer.q_heads = 4;
        layer.kv_heads = kv_heads;
        auto k = hera::MultiHeadPackingPlan::ProjectionOutput(
            layer, hera::Layout::kMultiHeadK, slots);
        Require(k.ok(), "MHA/GQA/MQA K packing rejected");
        for (const auto &locations : k.value().logical_locations) {
            Require(locations.size() == layer.q_heads / layer.kv_heads,
                    "K physical head replication count is wrong");
        }
    }

    auto smoke = base;
    smoke.tokens = 32;
    smoke.model_width = 512;
    smoke.q_heads = 8;
    smoke.kv_heads = 2;
    smoke.qk_head_width = 70;
    smoke.value_head_width = 63;
    auto input = hera::MultiHeadPackingPlan::ProjectionInput(
        smoke, 16384);
    Require(input.ok(), "smoke projection-input packing rejected");
    Require(input.value().ciphertext_count == 4,
            "smoke projection input should use four ciphertexts");
    Require(input.value().Locations(0).size() == smoke.q_heads,
            "spatial input was not replicated across query heads");
    const auto imag_feature = 64;
    const auto imag_logical = imag_feature;
    Require(input.value().Locations(imag_logical).front().component ==
                hera::ComplexLane::kImaginary,
            "projection input did not fill imaginary lanes after real lanes");
    auto q = hera::MultiHeadPackingPlan::ProjectionOutput(
        smoke, hera::Layout::kMultiHeadQ, 16384);
    auto v = hera::MultiHeadPackingPlan::ProjectionOutput(
        smoke, hera::Layout::kMultiHeadV, 16384);
    Require(q.ok() && v.ok(), "tail multihead projection packing rejected");
    Require(q.value().Validate().ok() && v.value().Validate().ok(),
            "tail multihead projection packing is not bijective");
}

void TestBlock2()
{
    auto config = MakeConfig();
    Require(hera::ValidateHeraConfig(config).ok(), "test config is invalid");
    const auto &layer = config.layers.front();
    const auto &profile = config.he_parameter_profiles.at("block2");
    auto short_profile = profile;
    short_profile.coefficient_modulus_bits.resize(7);
    short_profile.declared_depth = 6;
    Require(!hera::Block2LevelPlanner::Validate(short_profile).ok(),
            "short Block-2 modulus chain was accepted");
    TestPackingVariants(layer, profile.polynomial_degree / 2);

    hera::HeContextRegistry client;
    Require(client.CreateClientProfiles(config).ok(),
            "cannot create client HE context");
    const auto packing = MakeInterleavedSpatial(
        {{layer.tokens, layer.model_width}}, profile.polynomial_degree / 2);
    auto steps = hera::SealBlock2Executor::RequiredRotationSteps(
        layer, profile, packing);
    Require(steps.ok(), steps.status().message());
    Require(std::find(steps.value().begin(), steps.value().end(), -3) !=
                steps.value().end(),
            "exact non-power-of-two projection rotation key is absent");
    Require(client.CompileRotationKeys("block2", steps.value()).ok(),
            "cannot compile Block 2 Galois keys");
    auto bundle = client.ExportPublicBundle();
    Require(bundle.ok(), bundle.status().message());
    hera::HeContextRegistry server;
    Require(server.CreateServerProfiles(config, bundle.value()).ok(),
            "cannot create server HE context");
    Require(!server.Require("block2", hera::PartyRole::kServer).has_secret_key(),
            "server unexpectedly owns the secret key");

    const auto &client_context =
        client.Require("block2", hera::PartyRole::kClient);
    const std::vector<double> centered{-0.35, 0.20, 0.10, -0.15,
                                        0.30, -0.25, 0.05, 0.40};
    const std::vector<double> inverse{1.20, 1.20, 1.20, 1.20,
                                      0.85, 0.85, 0.85, 0.85};
    auto centered_ct = Encrypt(centered, packing, client_context);
    auto inverse_ct = Encrypt(inverse, packing, client_context);

    hera::SealCipherTensorStore tensors;
    auto centered_handle = tensors.Allocate(std::move(centered_ct));
    auto inverse_handle = tensors.Allocate(std::move(inverse_ct));
    Require(centered_handle.ok() && inverse_handle.ok(),
            "cannot register Block 2 inputs");
    hera::SealPlainParameterStore parameters;
    std::vector<double> wq, wk, wv;
    auto weights = RegisterWeights(layer, parameters, wq, wk, wv);
    hera::SealBlock2Executor executor(config, layer, server, tensors,
                                       parameters);
    hera::Block2ExecutionReport report;
    auto output = executor.Evaluate(
        {centered_handle.value(), inverse_handle.value()}, weights, &report);
    Require(output.ok(), output.status().message());

    auto value_payload = tensors.Get(output.value().value.id);
    auto score_payload = tensors.Get(output.value().scaled_attention_scores.id);
    Require(value_payload.ok() && score_payload.ok(),
            "Block 2 outputs are absent from the tensor store");
    auto actual_v = Decrypt(value_payload.value(), client_context);
    auto actual_scores = Decrypt(score_payload.value(), client_context);

    std::vector<double> normalized(centered.size());
    for (std::size_t i = 0; i < normalized.size(); ++i) {
        const auto feature = i % layer.model_width;
        const double gamma = 0.9 + 0.05 * static_cast<double>(feature);
        const double beta = -0.02 + 0.01 * static_cast<double>(feature);
        normalized[i] = centered[i] * inverse[i] * gamma + beta;
    }
    auto expected_q = MatMul(normalized, layer.tokens, layer.model_width,
                            wq, hera::QueryWidth(layer));
    for (std::size_t row = 0; row < layer.tokens; ++row)
        for (std::size_t column = 0; column < hera::QueryWidth(layer);
             ++column) {
            expected_q[row * hera::QueryWidth(layer) + column] +=
                (static_cast<int>(column % 3) - 1) * 0.004;
            expected_q[row * hera::QueryWidth(layer) + column] /=
                std::sqrt(static_cast<double>(layer.qk_head_width));
        }
    auto expected_k = MatMul(normalized, layer.tokens, layer.model_width,
                            wk, hera::KeyWidth(layer));
    auto expected_v = MatMul(normalized, layer.tokens, layer.model_width,
                            wv, hera::ValueWidth(layer));
    for (std::size_t row = 0; row < layer.tokens; ++row) {
        for (std::size_t column = 0; column < hera::KeyWidth(layer); ++column)
            expected_k[row * hera::KeyWidth(layer) + column] +=
                (static_cast<int>(column % 3) - 1) * 0.004;
        for (std::size_t column = 0; column < hera::ValueWidth(layer); ++column)
            expected_v[row * hera::ValueWidth(layer) + column] +=
                (static_cast<int>(column % 3) - 1) * 0.004;
    }
    std::vector<double> expected_scores(layer.q_heads * layer.tokens *
                                        layer.tokens);
    for (std::size_t head = 0; head < layer.q_heads; ++head)
        for (std::size_t row = 0; row < layer.tokens; ++row)
            for (std::size_t column = 0; column < layer.tokens; ++column)
                for (std::size_t d = 0; d < layer.qk_head_width; ++d) {
                    const auto q_index = row * hera::QueryWidth(layer) +
                                         head * layer.qk_head_width + d;
                    const auto kv_head = head /
                        (layer.q_heads / layer.kv_heads);
                    const auto k_index = column * hera::KeyWidth(layer) +
                                         kv_head * layer.qk_head_width + d;
                    expected_scores[(head * layer.tokens + row) *
                                    layer.tokens + column] +=
                        expected_q[q_index] * expected_k[k_index];
                }
    Require(MaxError(actual_v, expected_v) < 5e-3,
            "V projection exceeds CKKS error budget");
    Require(MaxError(actual_scores, expected_scores) < 2e-2,
            "QK scores exceed CKKS error budget");
    Require(output.value().value.spec.planned_level == 2,
            "V output has the wrong planned level");
    Require(output.value().scaled_attention_scores.spec.planned_level == 9,
            "attention scores have the wrong planned level");
    const auto &server_context =
        server.Require("block2", hera::PartyRole::kServer);
    Require(ActualLevel(server_context,
                        value_payload.value().ciphertexts.front().parms_id()) == 2,
            "V ciphertext is not actually at level 2");
    Require(ActualLevel(server_context,
                        score_payload.value().ciphertexts.front().parms_id()) == 9,
            "score ciphertext is not actually at terminal level 9");
    Require(std::abs(std::log2(
                value_payload.value().ciphertexts.front().scale()) - 44.0) < 0.01 &&
            std::abs(std::log2(
                score_payload.value().ciphertexts.front().scale()) - 44.0) < 0.01,
            "Block-2 ciphertext scale does not match the profile");
    Require(report.ciphertext_multiplications != 0 &&
            report.ciphertext_plain_multiplications != 0 &&
            report.relinearizations != 0 && report.rescales != 0,
            "Block 2 operation report is incomplete");

    // Compile from metadata-only ciphertexts: Prepare must not encrypt zeros
    // or execute QK, rotations, or any other HE arithmetic.
    parameters.BeginPlaintextPreparation();
    {
        hera::ModelPlaintextCacheScope scope(&parameters, layer.id);
        auto meta_centered = tensors.Get(centered_handle.value().id);
        auto meta_inverse = tensors.Get(inverse_handle.value().id);
        Require(meta_centered.ok() && meta_inverse.ok(),
                "Block 2 metadata inputs are absent");
        for (auto *tensor : {&meta_centered.value(), &meta_inverse.value()})
            for (auto &ct : tensor->ciphertexts) {
                const auto id = ct.parms_id();
                const auto scale = ct.scale();
                ct = seal::Ciphertext();
                ct.parms_id() = id;
                ct.scale() = scale;
            }
        const auto get_vector = [&](const std::string &name) {
            for (const auto &handle : weights.plain_parameters)
                if (handle.spec.name == name)
                    return parameters.GetVector(handle.id).value();
            throw std::runtime_error("missing Block 2 test vector: " + name);
        };
        auto normalized_meta = hera::LayerNormTailKernel::Evaluate(
            meta_centered.value(), meta_inverse.value(),
            get_vector("ln1.gamma"), get_vector("ln1.beta"),
            layer, server_context);
        Require(normalized_meta.ok(), normalized_meta.status().message());
        auto qk_meta = normalized_meta.value();
        const auto terminal = profile.coefficient_modulus_bits.size() -
                              profile.terminal_prime_count;
        const auto qk_input_level = terminal - 5 - 1;
        auto qk_context_data = server_context.context->first_context_data();
        for (std::size_t level = 0; level < qk_input_level; ++level)
            qk_context_data = qk_context_data->next_context_data();
        Require(static_cast<bool>(qk_context_data),
                "Block 2 test QK level is absent");
        for (auto &ct : qk_meta.ciphertexts)
            ct.parms_id() = qk_context_data->parms_id();
        qk_meta.logical_spec.planned_level = qk_input_level;
        hera::ComplexDiagonalProjectionKernel qk_projection(
            layer, server_context);
        hera::ComplexDiagonalProjectionKernel value_projection(
            layer, server_context);
        auto q = qk_projection.Evaluate(
            qk_meta, parameters.GetMatrix(weights.wq.id).value(),
            get_vector("bq"), hera::Layout::kMultiHeadQ);
        auto k = qk_projection.Evaluate(
            qk_meta, parameters.GetMatrix(weights.wk.id).value(),
            get_vector("bk"), hera::Layout::kMultiHeadK);
        auto v = value_projection.Evaluate(
            normalized_meta.value(),
            parameters.GetMatrix(weights.wv.id).value(),
            get_vector("bv"), hera::Layout::kMultiHeadV);
        Require(q.ok() && k.ok() && v.ok(),
                q.ok() ? (k.ok() ? v.status().message()
                                 : k.status().message())
                       : q.status().message());
    }
    parameters.FreezePlaintextCache();
    Require(parameters.prepared_projection_count() == 3,
            "Prepare did not compile Wq, Wk and Wv projection plans");
    hera::Block2ExecutionReport cached_report;
    hera::Result<hera::Block2HeOutputs> cached =
        hera::Status::Internal("cached Block 2 was not evaluated");
    {
        hera::ModelPlaintextCacheScope scope(&parameters, layer.id);
        cached = executor.Evaluate(
            {centered_handle.value(), inverse_handle.value()}, weights,
            &cached_report);
    }
    Require(cached.ok(), cached.status().message());
    Require(parameters.prepared_projection_hits() == 3,
            "online Block 2 did not use all prepared projection plans");
    auto cached_value = tensors.Get(cached.value().value.id);
    auto cached_scores = tensors.Get(cached.value().scaled_attention_scores.id);
    Require(cached_value.ok() && cached_scores.ok(),
            "cached Block 2 outputs are absent");
    Require(MaxError(Decrypt(cached_value.value(), client_context), actual_v) <
                1e-4 &&
            MaxError(Decrypt(cached_scores.value(), client_context),
                     actual_scores) < 1e-4,
            "prepared Block 2 differs from the direct HE path");
    Require(cached_report.rotations == report.rotations &&
                cached_report.ciphertext_plain_multiplications ==
                    report.ciphertext_plain_multiplications &&
                cached_report.rescales == report.rescales,
            "prepared Block 2 changed HE operation counts");
}

} // namespace

int main()
{
    try {
        TestBlock2();
        TestParallelQkGroups();
        std::cout << "HERA_BLOCK2_HE_STATUS=PASS\n";
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << "HERA_BLOCK2_HE_STATUS=FAIL\n"
                  << "HERA_BLOCK2_HE_ERROR=" << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
