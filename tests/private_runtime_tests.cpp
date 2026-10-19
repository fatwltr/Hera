#include "hera/blb_reference.h"
#include "hera/private_runtime.h"
#include "hera/mpc_session.h"

#include <seal/seal.h>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <complex>
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
    layer.value_head_width = 2;
    layer.ffn_width = 5;
    layer.norm = hera::NormPlacement::kPreNorm;
    layer.activation = hera::ActivationKind::kGelu;
    layer.attention_mask = hera::AttentionMaskKind::kNone;
    layer.qkv_bias = true;
    layer.output_bias = true;
    layer.ffn_bias = true;
    layer.attention_residual = true;
    layer.ffn_residual = true;
    layer.attention_norm_rsqrt_bits = 20;
    config.layers = {layer};
    config.mpc_numeric = {43, 13, true};
    config.he_parameter_profiles = {
        {"r1", {2048, {60, 60, 60, 60, 60}, 38, 38, 38, 4, 1, true,
                 hera::SecurityLevel::kNone}},
        {"r2", {2048, {60, 35, 60, 60, 60, 60, 60},
                    44, 44, 44, 6, 1, true,
                    hera::SecurityLevel::kNone}},
        {"r3", {2048, {60, 40, 40, 40, 40, 40, 40, 60},
                40, 40, 40, 7, 1, true, hera::SecurityLevel::kNone}},
        {"r5", {2048, {60, 60, 46, 46, 46, 60, 60},
                45, 45, 46, 6, 1, true, hera::SecurityLevel::kNone}},
    };
    hera::LayerExecutionConfig execution;
    execution.layer_id = layer.id;
    execution.regions = {
        {hera::HeRegionId::kRegion1,
         {hera::HeRegionId::kRegion1, "r1", hera::Layout::kStrictColumn,
          hera::Layout::kReduction}},
        {hera::HeRegionId::kRegion2,
         {hera::HeRegionId::kRegion2, "r2", hera::Layout::kSpatial,
          hera::Layout::kAttentionOutput}},
        {hera::HeRegionId::kRegion3,
         {hera::HeRegionId::kRegion3, "r3", hera::Layout::kSoftmaxRows,
          hera::Layout::kSoftmaxRows}},
        {hera::HeRegionId::kRegion4,
         {hera::HeRegionId::kRegion4, "r2",
          hera::Layout::kAttentionOutput, hera::Layout::kReduction}},
        {hera::HeRegionId::kRegion5,
         {hera::HeRegionId::kRegion5, "r5", hera::Layout::kSpatial,
          hera::Layout::kSpatial}},
    };
    config.layer_execution = {execution};
    return config;
}

seal::parms_id_type LevelId(const hera::HeContext &context,
                            std::size_t wanted)
{
    auto data = context.context->first_context_data();
    for (std::size_t level = 0; data && level < wanted; ++level)
        data = data->next_context_data();
    if (!data) throw std::runtime_error("requested level is absent");
    return data->parms_id();
}

hera::SealCipherTensor EncryptInput(const hera::HeraConfig &config,
                                    const hera::HeContext &context)
{
    const auto &layer = config.layers.front();
    auto packing = hera::AttentionLn2PackingPlanner::RealStrictColumns(
        {{layer.tokens, layer.model_width}}, context.slot_count());
    Require(packing.ok(), packing.status().message());
    std::vector<std::vector<std::complex<double>>> slots(
        packing.value().ciphertext_count,
        std::vector<std::complex<double>>(context.slot_count()));
    const std::vector<double> input{0.10, -0.20, 0.30, 0.05,
                                    -0.10, 0.25, -0.15, 0.20};
    for (std::size_t i = 0; i < input.size(); ++i) {
        const auto &where = packing.value().Locations(i).front();
        slots[where.ciphertext][where.slot] = input[i];
    }
    seal::CKKSEncoder encoder(*context.context);
    seal::Encryptor encryptor(*context.context, *context.secret_key);
    std::vector<seal::Ciphertext> ciphertexts(slots.size());
    for (std::size_t i = 0; i < slots.size(); ++i) {
        seal::Plaintext plain;
        encoder.encode(slots[i], std::ldexp(1.0,
            context.parameters.input_scale_bits), plain);
        encryptor.encrypt_symmetric(plain, ciphertexts[i]);
    }
    hera::TensorSpec spec;
    spec.shape = {{layer.tokens, layer.model_width}};
    spec.domain = hera::Domain::kHeCiphertext;
    spec.layout = hera::Layout::kStrictColumn;
    spec.numeric = config.mpc_numeric;
    spec.he_region = hera::HeRegionId::kRegion2;
    spec.he_parameter_profile = context.profile;
    spec.planned_level = 0;
    spec.planned_scale_bits = context.parameters.input_scale_bits;
    return {context.profile, std::move(ciphertexts), spec,
            packing.take_value()};
}

struct PartyResult {
    bool ok{};
    hera::TensorHandle output;
    std::string error;
};

void TestVirtualWeightPlaintextPool()
{
    const auto config = MakeConfig();
    hera::HeContextRegistry contexts;
    Require(contexts.CreateClientProfiles(config).ok(),
            "virtual-cache contexts cannot be created");
    const auto &context = contexts.Require("r5", hera::PartyRole::kClient);
    seal::CKKSEncoder encoder(*context.context);
    const auto parms_id = LevelId(context, 2);
    const auto scale = std::ldexp(1.0, 45);
    std::vector<std::complex<double>> first(context.slot_count(), {0.001, 0.0});
    std::vector<std::complex<double>> second(context.slot_count(), {-0.002, 0.0});

    hera::SealPlainParameterStore store;
    Require(store.ConfigureVirtualWeightPlaintexts(1).ok(),
            "virtual weight cache cannot be configured");
    store.BeginPlaintextPreparation();
    auto one = store.EncodeModelPlaintext(
        encoder, first, parms_id, scale,
        "layer0/ffn.projection.diagonal");
    auto two = store.EncodeModelPlaintext(
        encoder, second, parms_id, scale,
        "layer1/ffn.projection.diagonal");
    Require(one.ok() && two.ok(), "virtual weight templates cannot be encoded");
    Require(store.encoded_plaintext_count() == 1 &&
                store.virtual_weight_requests() == 2 &&
                store.virtual_weight_cache_hits() == 1,
            "compatible diagonals did not share one physical plaintext");

    auto constant_one = store.EncodeModelPlaintext(
        encoder, first, parms_id, scale, "layer0/ffn.ln2.beta");
    auto constant_two = store.EncodeModelPlaintext(
        encoder, second, parms_id, scale, "layer1/ffn.ln2.beta");
    Require(constant_one.ok() && constant_two.ok() &&
                store.encoded_plaintext_count() == 3,
            "non-weight constants were incorrectly virtualized");
    store.FreezePlaintextCache();

    std::vector<std::complex<double>> third(context.slot_count(), {0.003, 0.0});
    auto online_hit = store.EncodeModelPlaintext(
        encoder, third, parms_id, scale,
        "layer2/attention.wo.bsgs.diagonal");
    Require(online_hit.ok() && store.virtual_weight_cache_hits() == 2,
            "frozen virtual cache did not reuse its prepared template");

    store.BeginPlaintextPreparation();
    auto staged = store.EncodeModelPlaintext(
        encoder, third, parms_id, scale, "layer3/ffn.ln2.beta");
    Require(staged.ok() && store.encoded_plaintext_count() == 1,
            "replacement preparation did not use an isolated cache");
    store.AbortPlaintextPreparation();
    Require(store.plaintext_cache_frozen() &&
                store.encoded_plaintext_count() == 3,
            "aborted preparation destroyed the active model cache");
    Require(store.EncodeModelPlaintext(
        encoder, first, parms_id, scale, "layer0/ffn.ln2.beta").ok(),
        "aborted preparation lost an existing plaintext");
    Require(!store.EncodeModelPlaintext(
        encoder, third, parms_id, scale, "layer3/ffn.ln2.beta").ok(),
        "aborted preparation allowed online encoding");
}

void TestCompiledExecutionPlan()
{
    auto config = MakeConfig();
    auto second_layer = config.layers.front();
    second_layer.id = "layer1";
    config.layers.push_back(second_layer);
    auto second_execution = config.layer_execution.front();
    second_execution.layer_id = second_layer.id;
    config.layer_execution.push_back(second_execution);

    auto compiled = hera::CompiledTransformerExecutionPlan::Compile(config);
    Require(compiled.ok(), compiled.status().message());
    Require(compiled.value().layers.size() == 2,
            "compiled plan omitted a Transformer layer");
    for (const auto &layer : compiled.value().layers) {
        Require(layer.statistics_input_level == 1 &&
                    layer.qk_projection_input_level == 1 &&
                    layer.attention_scores_level == 6 &&
                    layer.retained_value_level == 2 &&
                    layer.attention_value_input_level == 3 &&
                    layer.attention_output_profile == "r2" &&
                    layer.attention_output_input_level == 2 &&
                    layer.attention_residual_level == 3 &&
                    layer.attention_residual_layout == hera::Layout::kStrictColumn,
                "compiled plan has an incorrect level boundary");
        Require(!layer.block2_rotation_steps.empty(),
                "compiled Block 2 rotation plan is empty");
    }
    auto shared_ffn = config;
    for (auto &layer : shared_ffn.layers) layer.norm = hera::NormPlacement::kNormalizedResidual;
    for (auto &execution : shared_ffn.layer_execution)
        execution.regions.at(hera::HeRegionId::kRegion1).parameter_profile = "r5";
    auto shared_plan = hera::CompiledTransformerExecutionPlan::Compile(shared_ffn);
    Require(shared_plan.ok(), shared_plan.status().message());
    for (const auto &layer : shared_plan.value().layers)
        Require(layer.statistics_input_level == 3 && layer.statistics_input_scale_bits == 46 &&
                layer.ffn_output_profile == "r5" && layer.ffn_output_level == 3 && layer.ffn_output_scale_bits == 46,
                "shared R1/R5 compiled level and scale contracts differ");
    auto runtime_steps =
        hera::SealSciTransformerRuntime::RequiredRotationSteps(config);
    Require(runtime_steps.ok() &&
                runtime_steps.value() ==
                    compiled.value().rotation_steps_by_profile,
            "runtime key generation diverged from the compiled plan");
    auto insufficient = config;
    insufficient.he_parameter_profiles.at("r2").declared_depth = 3;
    Require(!hera::CompiledTransformerExecutionPlan::Compile(insufficient).ok(),
            "Region-4 profile with insufficient depth unexpectedly compiled");
    auto independent = config;
    independent.he_parameter_profiles.emplace("other", config.he_parameter_profiles.at("r2"));
    independent.layer_execution.front().regions.at(hera::HeRegionId::kRegion4)
        .parameter_profile = "other";
    Require(!hera::CompiledTransformerExecutionPlan::Compile(independent).ok(),
            "independent Region-4 profile unexpectedly compiled");
    auto legacy = config;
    legacy.he_parameter_profiles.at("r2") =
        {2048, {60, 35, 60, 60, 60, 60, 60, 60, 60, 60},
         44, 44, 44, 9, 1, true, hera::SecurityLevel::kNone};
    for (auto &execution : legacy.layer_execution)
        execution.regions.at(hera::HeRegionId::kRegion4).parameter_profile = "r2";
    auto legacy_plan = hera::CompiledTransformerExecutionPlan::Compile(legacy);
    Require(legacy_plan.ok() &&
                legacy_plan.value().layers.front().attention_residual_level == 6,
            "shared-profile compatibility plan failed");
    config.layers.front().norm = hera::NormPlacement::kPostNorm;
    Require(!hera::CompiledTransformerExecutionPlan::Compile(config).ok(),
            "unsupported Post-LN config unexpectedly compiled");
}

PartyResult RunParty(
    hera::PartyRole role, const hera::HeraConfig &config,
    hera::HeContextRegistry &contexts, hera::RuntimeTensorArena &arena,
    hera::SealPlainParameterStore &parameters,
    const hera::TransformerWeightHandles &weights, int base)
{
    try {
        auto transport = hera::MakeNetIoTransport(
            role, role == hera::PartyRole::kClient ? "127.0.0.1" : "",
            base + 10);
        hera::BlbReferenceConverter converter(
            role, {43, 13, 40, 20, true,0,true,true}, contexts, transport,
            hera::MakeDeterministicRandomSourceForTesting(
                role == hera::PartyRole::kClient
                    ? std::vector<std::uint8_t>{2, 3, 5, 7}
                    : std::vector<std::uint8_t>{11, 13, 17, 19}),
            "127.0.0.1", base + 100, 1);
        auto nonlinear_session = std::make_shared<hera::SharedMpcSession>(
            role, "127.0.0.1", base + 300, 3);
        hera::MpcSoftmaxRuntime softmax(nonlinear_session);
        hera::MpcLayerNormRuntime norm(nonlinear_session);
        hera::MpcGeluRuntime gelu(nonlinear_session);
        auto session_ready = nonlinear_session->Prepare();
        if (!session_ready.ok())
            return {false, {}, session_ready.status().message()};
        auto runtime = std::make_shared<hera::SealSciTransformerRuntime>(
            config, contexts, arena, parameters, converter, softmax, norm,
            gelu);
        auto framework = hera::HeraContext::Create(config);
        if (!framework.ok()) return {false, {}, framework.status().message()};
        auto bound = framework.value()->BindExecutionRuntime(runtime);
        if (!bound.ok()) return {false, {}, bound.status().message()};
        auto transformer = hera::PrivateTransformer::Compile(framework.value());
        if (!transformer.ok())
            return {false, {}, transformer.status().message()};
        hera::PreparationReport preparation;
        auto prepared = transformer.value().Prepare(weights, &preparation);
        if (!prepared.ok())
            return {false, {}, prepared.status().message()};
        hera::PreparationReport repeated;
        prepared = transformer.value().Prepare(weights, &repeated);
        if (!prepared.ok())
            return {false, {}, prepared.status().message()};
        if (role == hera::PartyRole::kServer &&
            (preparation.encoded_plaintext_count == 0 ||
             repeated.cache_hits != preparation.encoded_plaintext_count))
            return {false, {}, "prepared plaintext cache was not reused"};
        hera::EncryptedTransformerInput input;
        input.hidden_states.tensor = {11000001,
            contexts.role() == hera::PartyRole::kServer
                ? arena.he.Get(11000001).value().logical_spec
                : hera::TensorSpec{}};
        if (role == hera::PartyRole::kClient) {
            const auto &layer = config.layers.front();
            input.hidden_states.tensor.spec.shape =
                {{layer.tokens, layer.model_width}};
            input.hidden_states.tensor.spec.domain =
                hera::Domain::kHeCiphertext;
            input.hidden_states.tensor.spec.layout = hera::Layout::kStrictColumn;
            input.hidden_states.tensor.spec.numeric = config.mpc_numeric;
            input.hidden_states.tensor.spec.he_region =
                hera::HeRegionId::kRegion2;
            const auto &regions = config.layer_execution.front().regions;
            const auto &profile = regions.at(hera::HeRegionId::kRegion2).parameter_profile;
            input.hidden_states.tensor.spec.he_parameter_profile = profile;
            input.hidden_states.tensor.spec.planned_level = 0;
            input.hidden_states.tensor.spec.planned_scale_bits =
                config.he_parameter_profiles.at(profile).input_scale_bits;
        }
        input.attention_mask.kind = hera::AttentionMaskKind::kNone;
        hera::ExecutionReport report;
        const auto conversion_bytes_before = converter.stats().communication_bytes;
        const auto mpc_bytes_before = nonlinear_session->communication_bytes();
        auto output = transformer.value().Evaluate(input, weights, &report);
        if (!output.ok()) return {false, {}, output.status().message()};
        if (role == hera::PartyRole::kServer &&
            parameters.encoded_plaintext_count() !=
                preparation.encoded_plaintext_count)
            return {false, {},
                    "online inference encoded a new model plaintext"};
        Require(report.stages.size() == config.layers.size() * 5,
                "runtime did not report five fused stages per layer");
        std::uint64_t stage_bytes = 0, stage_rounds = 0;
        std::uint64_t conversion_bytes = 0, mpc_bytes = 0;
        double stage_milliseconds = 0.0;
        for (const auto &stage : report.stages) {
            stage_bytes += stage.metrics.communication_bytes;
            conversion_bytes += stage.metrics.conversion_communication_bytes;
            mpc_bytes += stage.metrics.mpc_communication_bytes;
            Require(stage.metrics.communication_bytes ==
                        stage.metrics.conversion_communication_bytes + stage.metrics.mpc_communication_bytes,
                    "stage communication categories do not sum to its online traffic");
            stage_rounds += stage.metrics.rounds;
            stage_milliseconds += stage.milliseconds;
            const auto &costs = stage.metrics.costs;
            Require(std::isfinite(costs.he_computation_milliseconds) &&
                        costs.he_computation_milliseconds >= 0.0 &&
                        costs.he_mpc_conversion_milliseconds > 0.0 &&
                        costs.other_nonlinear_milliseconds >= 0.0 &&
                        costs.local_overhead_milliseconds >= 0.0,
                    "stage cost breakdown contains invalid durations");
            Require(std::abs(costs.stage_milliseconds() - stage.milliseconds) < 1e-6,
                    "stage cost breakdown double counted or lost wall time");
            Require((costs.he_computation_milliseconds > 0.0) ==
                        (role == hera::PartyRole::kServer),
                    "HE timing lost a server interval or counted client bookkeeping");
            Require((costs.other_nonlinear_milliseconds == 0.0) ==
                        (stage.name == "block5_bridge"),
                    "nonlinear MPC time was attributed to the wrong stage");
        }
        Require(std::abs(report.costs.stage_milliseconds() - stage_milliseconds) < 1e-6 &&
                    report.costs.stage_milliseconds() <= report.online_milliseconds,
                "runtime cost totals diverged from the fused-stage wall times");
        Require(report.communication_bytes == stage_bytes && report.rounds == stage_rounds,
                "runtime totals lost communication recorded by a fused stage");
        Require(conversion_bytes == converter.stats().communication_bytes - conversion_bytes_before &&
                    mpc_bytes == nonlinear_session->communication_bytes() - mpc_bytes_before,
                "communication categories include setup or lose online protocol traffic");
        for (const auto &stage : report.stages)
            if (stage.name == "pre_ln_block2")
                Require(stage.metrics.conversions == (stage.layer == 0 ? 5 : 3),
                        "direct residual performed an extra MPC-to-CKKS conversion");
        const auto expected_he = role == hera::PartyRole::kServer
            ? std::size_t{2} : std::size_t{0};
        if (arena.he.Size() != expected_he || arena.mpc.Size() != 0)
            return {false, {},
                    "runtime arena retained dead intermediate tensors"};
        return {true, output.value().tensor, {}};
    } catch (const std::exception &error) {
        return {false, {}, error.what()};
    }
}

std::vector<double> PlainNormalizedResidualModel(
    const hera::HeraConfig &config, const hera::TransformerWeightData &weights,
    bool normalized_attention = true, bool normalized_ffn = true)
{
    std::vector<double> hidden{0.10, -0.20, 0.30, 0.05, -0.10, 0.25, -0.15, 0.20};
    for (std::size_t index = 0; index < config.layers.size(); ++index) {
        const auto &layer = config.layers[index];
        const auto &w = weights.layers[index];
        const auto norm = [&](const std::vector<double> &x, const std::string &name) {
            std::vector<double> y(x.size());
            for (std::size_t t = 0; t < layer.tokens; ++t) {
                double mean = 0, variance = 0;
                for (std::size_t f = 0; f < layer.model_width; ++f) mean += x[t * layer.model_width + f];
                mean /= layer.model_width;
                for (std::size_t f = 0; f < layer.model_width; ++f) {
                    const double c = x[t * layer.model_width + f] - mean;
                    variance += c * c;
                }
                const auto inverse = 1.0 / std::sqrt(variance / layer.model_width + layer.attention_norm_epsilon);
                for (std::size_t f = 0; f < layer.model_width; ++f)
                    y[t * layer.model_width + f] = (x[t * layer.model_width + f] - mean) * inverse *
                        w.vectors.at(name + ".gamma")[f] + w.vectors.at(name + ".beta")[f];
            }
            return y;
        };
        const auto project = [&](const std::vector<double> &x, std::size_t in, std::size_t out,
                                 const std::vector<double> &matrix, const std::string &bias, bool use_bias) {
            std::vector<double> y(layer.tokens * out);
            for (std::size_t t = 0; t < layer.tokens; ++t)
                for (std::size_t o = 0; o < out; ++o) {
                    double value = use_bias ? w.vectors.at(bias)[o] : 0.0;
                    for (std::size_t f = 0; f < in; ++f) value += x[t * in + f] * matrix[f * out + o];
                    y[t * out + o] = value;
                }
            return y;
        };
        const auto x = norm(hidden, "ln1");
        const auto q_width = layer.q_heads * layer.qk_head_width;
        const auto kv_width = layer.kv_heads * layer.qk_head_width;
        const auto v_width = layer.kv_heads * layer.value_head_width;
        const auto q = project(x, layer.model_width, q_width, w.wq, "bq", layer.qkv_bias);
        const auto k = project(x, layer.model_width, kv_width, w.wk, "bk", layer.qkv_bias);
        const auto v = project(x, layer.model_width, v_width, w.wv, "bv", layer.qkv_bias);
        const auto attention_width = layer.q_heads * layer.value_head_width;
        std::vector<double> av(layer.tokens * attention_width);
        for (std::size_t t = 0; t < layer.tokens; ++t)
            for (std::size_t head = 0; head < layer.q_heads; ++head) {
                const auto kv_head = head / (layer.q_heads / layer.kv_heads);
                std::vector<double> scores(layer.tokens);
                for (std::size_t column = 0; column < layer.tokens; ++column)
                    for (std::size_t f = 0; f < layer.qk_head_width; ++f)
                        scores[column] += q[t * q_width + head * layer.qk_head_width + f] *
                            k[column * kv_width + kv_head * layer.qk_head_width + f] / std::sqrt(layer.qk_head_width);
                const auto maximum = *std::max_element(scores.begin(), scores.end());
                double sum = 0;
                for (auto &score : scores) {
                    const auto shifted = score - maximum;
                    score = shifted < config.softmax.clip_threshold ? 0.0 : std::pow(
                        1.0 + shifted / std::ldexp(1.0, config.softmax.exp_divisor_bits),
                        std::ldexp(1.0, config.softmax.exp_squarings));
                    sum += score;
                }
                for (std::size_t f = 0; f < layer.value_head_width; ++f)
                    for (std::size_t column = 0; column < layer.tokens; ++column)
                        av[t * attention_width + head * layer.value_head_width + f] +=
                            scores[column] / sum * v[column * v_width + kv_head * layer.value_head_width + f];
            }
        auto h = project(av, attention_width, layer.model_width, w.wo, "bo", layer.output_bias);
        if (layer.attention_residual)
            for (std::size_t i = 0; i < h.size(); ++i) h[i] += normalized_attention ? x[i] : hidden[i];
        const auto u = norm(h, "ln2");
        auto activation = project(u, layer.model_width, layer.ffn_width, w.wu, "bu", layer.ffn_bias);
        for (auto &value : activation) {
            const auto &g = config.gelu;
            if (value < -g.threshold) value = 0.0;
            else if (value <= g.threshold) {
                const double sign = value < 0.0 ? -1.0 : 1.0;
                value = g.a * std::pow(value, 4) + sign * g.b * std::pow(value, 3) +
                    g.c * value * value + (0.5 + sign * g.d) * value + g.e;
            }
        }
        hidden = project(activation, layer.ffn_width, layer.model_width, w.wd, "bd", layer.ffn_bias);
        if (layer.ffn_residual)
            for (std::size_t i = 0; i < hidden.size(); ++i) hidden[i] += normalized_ffn ? u[i] : h[i];
    }
    return hidden;
}

void TestRuntime(hera::HeraConfig config, int base)
{
    auto &profile = config.he_parameter_profiles.at("r5");
    profile.coefficient_modulus_bits.pop_back();
    profile.declared_depth = profile.coefficient_modulus_bits.size() - 1;
    const auto terminal = config.he_parameter_profiles.at("r5").coefficient_modulus_bits.size() - 1;
    const auto output_level = terminal - 3;
    for (auto &execution : config.layer_execution)
        execution.regions.at(hera::HeRegionId::kRegion1).parameter_profile = "r5";
    Require(hera::ValidateHeraConfig(config).ok(), "test config is invalid");
    hera::HeContextRegistry client;
    Require(client.CreateClientProfiles(config).ok(),
            "client contexts cannot be created");
    auto rotations = hera::SealSciTransformerRuntime::RequiredRotationSteps(
        config);
    Require(rotations.ok(), rotations.status().message());
    for (const auto &entry : rotations.value())
        Require(client.CompileRotationKeys(entry.first, entry.second).ok(),
                "rotation keys cannot be compiled");
    auto bundle = client.ExportSerializedPublicBundle();
    Require(bundle.ok(), bundle.status().message());
    hera::HeContextRegistry server;
    Require(server.CreateServerProfiles(config, bundle.value()).ok(),
            "server contexts cannot be created");
    Require(!server.Require("r2", hera::PartyRole::kServer).has_secret_key(),
            "server unexpectedly owns a secret key");
    std::vector<seal::MemoryPoolHandle> evaluation_pools;
    for (const auto &entry : config.he_parameter_profiles) {
        const auto &context = client.Require(entry.first, hera::PartyRole::kClient);
        if (context.galois_keys.size()) evaluation_pools.push_back(context.galois_keys.pool());
        if (context.relinearization_keys.size()) evaluation_pools.push_back(context.relinearization_keys.pool());
    }
    for (auto &pool : evaluation_pools)
        Require(!(pool == seal::MemoryPoolHandle::Global()),
                "client evaluation keys retained global-pool storage");
    Require(!server.ReleaseClientEvaluationKeys().ok(),
            "server accepted the client-only evaluation-key release");
    Require(client.ReleaseClientEvaluationKeys().ok(),
            "client evaluation keys cannot be released");
    for (const auto &pool : evaluation_pools)
        Require(pool.use_count() == 1,
                "released client evaluation pool still has an owning key");
    evaluation_pools.clear();
    for (const auto &entry : config.he_parameter_profiles) {
        const auto &context = client.Require(entry.first, hera::PartyRole::kClient);
        Require(context.has_secret_key() && context.public_key.data().size() != 0 &&
                    context.galois_keys.size() == 0 && context.relinearization_keys.size() == 0,
                "client release removed encryption keys or retained evaluation keys");
    }

    hera::RuntimeTensorArena server_arena;
    hera::RuntimeTensorArena client_arena;
    auto input = EncryptInput(config,
        client.Require("r2", hera::PartyRole::kClient));
    Require(server_arena.he.Put(11000001, input).ok(),
            "server input cannot be registered");
    hera::SealPlainParameterStore server_parameters;
    hera::SealPlainParameterStore client_parameters;
    auto data = hera::MakeDeterministicTransformerWeights(config, 1234);
    if (config.layers.front().norm == hera::NormPlacement::kNormalizedResidual) {
        for (auto &layer : data.layers) {
            layer.vectors["ln1.gamma"] = {1.2, 0.0, -0.8, 0.6};
            layer.vectors["ln1.beta"] = {0.3, -0.2, 0.1, -0.15};
            layer.vectors["ln2.gamma"] = {1.1, -0.7, 0.0, 0.9};
            layer.vectors["ln2.beta"] = {-0.1, 0.2, 0.25, -0.15};
        }
    }
    auto server_weights = hera::RegisterTransformerWeights(
        config, data, server_parameters);
    auto client_weights = hera::RegisterTransformerWeights(
        config, std::nullopt, client_parameters);
    Require(server_weights.ok() && client_weights.ok(),
            "runtime weights cannot be registered");

    const auto child = fork();
    Require(child >= 0, "fork failed");
    if (child == 0) {
        auto result = RunParty(hera::PartyRole::kClient, config, client,
                               client_arena, client_parameters,
                               client_weights.value(), base);
        if (!result.ok) std::cerr << "client runtime: " << result.error << '\n';
        _exit(result.ok ? 0 : 2);
    }
    auto server_result = RunParty(
        hera::PartyRole::kServer, config, server, server_arena,
        server_parameters, server_weights.value(), base);
    int child_status = 0;
    waitpid(child, &child_status, 0);
    Require(server_result.ok, server_result.error);
    Require(WIFEXITED(child_status) && WEXITSTATUS(child_status) == 0,
            "client runtime failed");
    Require(server_result.output.spec.layout == hera::Layout::kStrictColumn &&
                server_result.output.spec.planned_level ==
                    std::optional<std::size_t>{output_level} &&
                server_result.output.spec.planned_scale_bits == std::optional<int>{46},
            "final HE output contract is wrong");
    Require(server_result.output.spec.he_parameter_profile ==
                std::optional<std::string>{"r5"},
            "final output did not retain the shared Region-1/5 profile");
    auto output = server_arena.he.Get(server_result.output.id);
    Require(output.ok(), output.status().message());
    const auto &client_shared = client.Require(
        "r5", hera::PartyRole::kClient);
    Require(output.value().ciphertexts.front().parms_id() ==
                LevelId(client_shared, output_level),
            "final ciphertext is not at the shared-profile output level");
    seal::Decryptor decryptor(*client_shared.context,
                              *client_shared.secret_key);
    seal::CKKSEncoder encoder(*client_shared.context);
    std::vector<std::vector<std::complex<double>>> planes;
    for (const auto &ciphertext : output.value().ciphertexts) {
        seal::Plaintext plain;
        decryptor.decrypt(ciphertext, plain);
        std::vector<std::complex<double>> decoded;
        encoder.decode(plain, decoded);
        for (const auto &value : decoded)
            Require(std::isfinite(value.real()) && std::isfinite(value.imag()),
                    "runtime output contains a non-finite value");
        planes.push_back(std::move(decoded));
    }
    if (config.layers.front().norm == hera::NormPlacement::kNormalizedResidual) {
        const auto expected = PlainNormalizedResidualModel(config, data);
        if (std::all_of(config.layers.begin(), config.layers.end(), [](const auto &layer) {
            return layer.attention_residual && layer.ffn_residual;
        })) for (const auto &wrong : {PlainNormalizedResidualModel(config, data, false, true),
                                  PlainNormalizedResidualModel(config, data, true, false)}) {
            double gap = 0;
            for (std::size_t i = 0; i < expected.size(); ++i)
                gap = std::max(gap, std::abs(wrong[i] - expected[i]));
            Require(gap > 0.05, "model oracle cannot distinguish the wrong residual source");
        }
        double error = 0;
        for (std::size_t i = 0; i < expected.size(); ++i) {
            const auto &loc = output.value().physical_packing.Locations(i).front();
            error = std::max(error, std::abs(planes[loc.ciphertext][loc.slot].real() - expected[i]));
        }
        Require(error < 0.02, "normalized residual two-layer model differs from unfused plaintext oracle: " +
                std::to_string(error));
        std::cout << "NORMALIZED_RESIDUAL_TWO_LAYER_MAX_ERROR=" << error << '\n';
    }
}

} // namespace

int main()
{
    try {
        TestVirtualWeightPlaintextPool();
        TestCompiledExecutionPlan();
        auto two_layers = MakeConfig();
        two_layers.layers.front().norm = hera::NormPlacement::kNormalizedResidual;
        two_layers.threads = 2;
        auto second_layer = two_layers.layers.front();
        second_layer.id = "layer1";
        two_layers.layers.push_back(second_layer);
        auto second_execution = two_layers.layer_execution.front();
        second_execution.layer_id = second_layer.id;
        two_layers.layer_execution.push_back(second_execution);
        // A two-layer run subsumes the one-layer path and, unlike two
        // back-to-back SCI process pairs in one test executable, does not
        // inherit fork-unsafe global OT state from an earlier session.
        TestRuntime(std::move(two_layers), 21200);
        std::cout << "HERA_PRIVATE_RUNTIME_STATUS=PASS\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "HERA_PRIVATE_RUNTIME_STATUS=FAIL: " << error.what()
                  << '\n';
        return 1;
    }
}
