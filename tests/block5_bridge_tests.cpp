#include "hera/block5_bridge.h"
#include "hera/private_transformer.h"

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <complex>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

void Require(bool condition, const std::string &message)
{
    if (!condition) throw std::runtime_error(message);
}

void TestPlannerAndGraph()
{
    auto config = hera::LoadHeraConfig(HERA_TEST_CONFIG_PATH);
    Require(config.ok(), config.status().message());
    auto steps = hera::SealSciBlock5BridgeExecutor::RequiredRotationSteps(
        config.value(), 0);
    Require(steps.ok(), steps.status().message());
    const auto *execution = hera::FindLayerExecution(
        config.value(), config.value().layers.front().id);
    Require(execution != nullptr, "layer placement is absent");
    const auto shared = execution->regions.at(hera::HeRegionId::kRegion4)
                            .parameter_profile;
    Require(steps.value().count(shared) == 1 &&
                !steps.value().at(shared).empty(),
            "Wd rotation schedule was not compiled for shared profile");

    auto graph = hera::BuildTransformerGraph(config.value());
    Require(graph.ok(), graph.status().message());
    bool wd = false;
    bool residual_add = false;
    bool realify = false;
    for (const auto &node : graph.value().nodes()) {
        if (node.name.find(".ffn.Wd") != std::string::npos) {
            wd = true;
            Require(node.kind == hera::OpKind::kMatMulCipherPlain,
                    "Wd is not a cipher/plain projection");
            const auto *attributes = std::get_if<hera::MatMulAttributes>(
                &node.attributes);
            Require(attributes != nullptr &&
                        attributes->kernel ==
                            hera::HeKernelKind::kFfnDownProjection,
                    "Wd does not use the typed down-projection kernel");
            Require(graph.value().tensors().at(node.output).layout ==
                        hera::Layout::kComplexStrictColumn,
                    "Wd output is not complex strict-column packed");
        }
        if (node.name.find(".ffn.residual_add") != std::string::npos) {
            residual_add = true;
            Require(node.kind == hera::OpKind::kElementwiseAdd,
                    "FFN residual is not a direct HE add");
            for (const auto input : node.inputs)
                Require(graph.value().tensors().at(input).domain ==
                            hera::Domain::kHeCiphertext,
                        "FFN residual add contains a non-HE input");
        }
        if (node.name.find(".ffn.realify_complex") != std::string::npos)
            realify = true;
    }
    Require(wd && residual_add && realify,
            "Block-5 graph nodes are incomplete");
}

constexpr std::uint64_t kMask = (std::uint64_t{1} << 43) - 1;

hera::HeraConfig MakeRuntimeConfig()
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
    config.layers = {layer};
    config.mpc_numeric = {43, 13, true};
    config.conversion = {40, 20, true};
    hera::HeParameterConfig profile;
    profile.polynomial_degree = 8192;
    profile.coefficient_modulus_bits = {
        60, 44, 44, 44, 44, 44, 44, 44, 44, 60};
    profile.input_scale_bits = 44;
    profile.default_scale_bits = 44;
    profile.output_scale_bits = 44;
    profile.declared_depth = 7;
    profile.terminal_prime_count = 1;
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
                hera::Layout::kSpatial, hera::Layout::kSpatial});
    config.layer_execution = {execution};
    return config;
}

void TestCrossLayerPlanner()
{
    auto config = MakeRuntimeConfig();
    auto next = config.layers.front();
    next.id = "layer1";
    config.layers.push_back(next);
    auto execution = config.layer_execution.front();
    execution.layer_id = next.id;
    config.layer_execution.push_back(execution);
    Require(hera::ValidateHeraConfig(config).ok(),
            "two-layer bridge config is invalid");
    auto steps = hera::SealSciBlock5BridgeExecutor::RequiredRotationSteps(
        config, 0);
    Require(steps.ok() && steps.value().count("shared") == 1 &&
                !steps.value().at("shared").empty(),
            "cross-layer rotations were not aggregated");
    config.layers[1].norm = hera::NormPlacement::kPostNorm;
    auto rejected = hera::SealSciBlock5BridgeExecutor::RequiredRotationSteps(
        config, 0);
    Require(!rejected.ok() &&
                rejected.status().code() ==
                    hera::StatusCode::kKernelUnavailable,
            "automatic Post-LN bridge was not rejected explicitly");
}

seal::parms_id_type LevelId(const hera::HeContext &context,
                            std::size_t target)
{
    auto data = context.context->first_context_data();
    for (std::size_t level = 0; level < target && data; ++level)
        data = data->next_context_data();
    if (!data) throw std::runtime_error("test HE level is absent");
    return data->parms_id();
}

hera::SealCipherTensor EncryptResidual(
    const std::vector<double> &values, const hera::HePackingPlan &packing,
    const hera::HeContext &context)
{
    std::vector<std::vector<std::complex<double>>> slots(
        packing.ciphertext_count,
        std::vector<std::complex<double>>(packing.slot_count));
    for (std::size_t logical = 0; logical < values.size(); ++logical) {
        const auto &location = packing.Locations(logical).front();
        auto &slot = slots[location.ciphertext][location.slot];
        if (location.component == hera::ComplexLane::kReal)
            slot.real(values[logical]);
        else slot.imag(values[logical]);
    }
    seal::CKKSEncoder encoder(*context.context);
    seal::Encryptor encryptor(*context.context, *context.secret_key);
    std::vector<seal::Ciphertext> ciphertexts(packing.ciphertext_count);
    for (std::size_t i = 0; i < ciphertexts.size(); ++i) {
        seal::Plaintext plain;
        encoder.encode(slots[i], LevelId(context, 6),
                       std::ldexp(1.0, 44), plain);
        encryptor.encrypt_symmetric(plain, ciphertexts[i]);
    }
    hera::TensorSpec spec;
    spec.shape = packing.logical_shape;
    spec.domain = hera::Domain::kHeCiphertext;
    spec.layout = hera::Layout::kComplexStrictColumn;
    spec.numeric = {43, 13, true};
    spec.he_region = hera::HeRegionId::kRegion4;
    spec.he_parameter_profile = context.profile;
    spec.planned_level = 6;
    spec.planned_scale_bits = 44;
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
    std::vector<double> result(tensor.physical_packing.logical_locations.size());
    for (std::size_t logical = 0; logical < result.size(); ++logical) {
        const auto &location = tensor.physical_packing.Locations(logical).front();
        const auto decoded = slots[location.ciphertext][location.slot];
        result[logical] = location.component == hera::ComplexLane::kReal
            ? decoded.real() : decoded.imag();
    }
    return result;
}

std::uint64_t EncodeFixed(double value)
{
    return static_cast<std::uint64_t>(static_cast<std::int64_t>(
        std::llround(std::ldexp(value, 13)))) & kMask;
}

struct PartyResult {
    bool ok{};
    hera::TensorHandle hidden;
    std::string error;
    hera::TensorHandle scores;
    hera::TensorHandle value;
};

PartyResult RunParty(
    hera::PartyRole role, const hera::HeraConfig &config,
    hera::HeContextRegistry &registry,
    const std::vector<std::uint64_t> &activation_share,
    hera::SealCipherTensorStore &he_store,
    hera::SealPlainParameterStore &parameters)
{
    try {
        const auto failed = [](std::string message) {
            PartyResult result;
            result.error = std::move(message);
            return result;
        };
        const int base_port = config.layers.size() == 1 ? 19900 : 20000;
        auto transport = hera::MakeNetIoTransport(
            role, role == hera::PartyRole::kClient ? "127.0.0.1" : "",
            base_port);
        hera::CkksMpcConverter converter(
            role, {43, 13, 40, 20, true}, registry, transport,
            hera::MakeDeterministicRandomSourceForTesting(
                role == hera::PartyRole::kClient
                    ? std::vector<std::uint8_t>{31, 37, 41}
                    : std::vector<std::uint8_t>{43, 47, 53}),
            "127.0.0.1", base_port + 100, 1);
        // Conversion and SCI sessions advance their worker ports between
        // calls. Keep their ranges disjoint so the cross-layer case cannot
        // make a conversion worker collide with a later rsqrt session.
        hera::MpcLayerNormRuntime mpc(
            role, "127.0.0.1", base_port + 500);
        hera::CryptoTensorStore mpc_store;
        const auto &layer = config.layers.front();
        constexpr hera::TensorId activation_id = 9200001;
        constexpr hera::TensorId residual_id = 9200002;
        auto stored = mpc_store.PutMpcShare(
            activation_id,
            {activation_share, {{layer.tokens, layer.ffn_width}},
             config.mpc_numeric, hera::ComplexComponent::kReal});
        if (!stored.ok()) return failed(stored.status().message());
        hera::TensorSpec activation_spec;
        activation_spec.shape = {{layer.tokens, layer.ffn_width}};
        activation_spec.domain = hera::Domain::kMpcArithmeticShare;
        activation_spec.layout = hera::Layout::kFfn;
        activation_spec.numeric = config.mpc_numeric;
        hera::TensorSpec residual_spec;
        residual_spec.shape = {{layer.tokens, layer.model_width}};
        residual_spec.domain = hera::Domain::kHeCiphertext;
        residual_spec.layout = hera::Layout::kComplexStrictColumn;
        residual_spec.numeric = config.mpc_numeric;
        residual_spec.he_region = hera::HeRegionId::kRegion4;
        residual_spec.he_parameter_profile = "shared";
        residual_spec.planned_level = 6;
        residual_spec.planned_scale_bits = 44;
        hera::LayerWeightHandles weights;
        weights.layer_id = layer.id;
        weights.wd = {601, {"Wd", layer.ffn_width, layer.model_width, true}};
        weights.plain_parameters = {{701, {"bd", layer.model_width}}};
        std::optional<hera::LayerWeightHandles> next_weights;
        if (config.layers.size() > 1) {
            const auto &next = config.layers[1];
            hera::LayerWeightHandles handles;
            handles.layer_id = next.id;
            handles.wq = {611, {"Wq", next.model_width,
                                hera::QueryWidth(next), false}};
            handles.wk = {612, {"Wk", next.model_width,
                                hera::KeyWidth(next), false}};
            handles.wv = {613, {"Wv", next.model_width,
                                hera::ValueWidth(next), false}};
            handles.plain_parameters = {
                {711, {"ln1.gamma", next.model_width}},
                {712, {"ln1.beta", next.model_width}}};
            next_weights = std::move(handles);
        }
        hera::SealSciBlock5BridgeExecutor executor(
            config, 0, registry, he_store, parameters, mpc_store,
            converter, mpc);
        hera::Block5BridgeExecutionReport report;
        auto output = executor.Evaluate(
            {{activation_id, activation_spec},
             hera::TensorHandle{residual_id, residual_spec}},
            weights, next_weights, &report);
        if (!output.ok()) return failed(output.status().message());
        PartyResult party_result;
        party_result.ok = true;
        party_result.hidden = output.value().hidden_states;
        const auto expected_m2c = config.layers.size() == 1 ? 1U : 4U;
        const auto expected_c2m = config.layers.size() == 1 ? 0U : 3U;
        if (report.mpc_to_ckks_calls != expected_m2c ||
            report.ckks_to_mpc_calls != expected_c2m)
            return failed("bridge performed an unexpected conversion count");
        if (role == hera::PartyRole::kServer &&
            report.direct_residual_additions != 1)
            return failed("server did not directly add FFN residual");
        if (config.layers.size() > 1) {
            if (output.value().hidden_states.spec.planned_level !=
                    std::optional<std::size_t>{9})
                return failed(
                    "cross-layer hidden state did not land at terminal");
            if (!output.value().next_layer)
                return failed("cross-layer output is absent");
            if (output.value().next_layer->block2.scaled_attention_scores.spec
                    .planned_level != std::optional<std::size_t>{9} ||
                output.value().next_layer->block2.value.spec.planned_level !=
                    std::optional<std::size_t>{2})
                return failed("next Block-2 level contract is wrong");
            if (role == hera::PartyRole::kServer) {
                auto scores = he_store.Get(
                    output.value().next_layer->block2.scaled_attention_scores.id);
                auto value = he_store.Get(
                    output.value().next_layer->block2.value.id);
                if (!scores.ok() || !value.ok())
                    return failed("next Block-2 payload is absent");
                const auto &context = registry.Require(
                    "shared", hera::PartyRole::kServer);
                if (scores.value().ciphertexts.front().parms_id() !=
                        LevelId(context, 9) ||
                    value.value().ciphertexts.front().parms_id() !=
                        LevelId(context, 2))
                    return failed("next Block-2 actual levels are wrong");
            }
            party_result.scores =
                output.value().next_layer->block2.scaled_attention_scores;
            party_result.value = output.value().next_layer->block2.value;
        }
        return party_result;
    } catch (const std::exception &error) {
        PartyResult result;
        result.error = error.what();
        return result;
    }
}

void TestLastLayerRuntime()
{
    auto config = MakeRuntimeConfig();
    Require(hera::ValidateHeraConfig(config).ok(),
            "runtime Block-5 config is invalid");
    hera::HeContextRegistry client;
    Require(client.CreateClientProfiles(config).ok(),
            "cannot create Block-5 client context");
    auto rotations = hera::SealSciBlock5BridgeExecutor::RequiredRotationSteps(
        config, 0);
    Require(rotations.ok(), rotations.status().message());
    Require(client.CompileRotationKeys("shared",
                rotations.value().at("shared")).ok(),
            "cannot compile Block-5 rotations");
    auto bundle = client.ExportSerializedPublicBundle();
    Require(bundle.ok(), bundle.status().message());
    hera::HeContextRegistry server;
    Require(server.CreateServerProfiles(config, bundle.value()).ok(),
            "cannot create Block-5 server context");
    Require(!server.Require("shared", hera::PartyRole::kServer)
                 .has_secret_key(),
            "Block-5 server owns a secret key");

    const auto &layer = config.layers.front();
    auto residual_plan = hera::FfnGeluPackingPlanner::ComplexColumns(
        {{layer.tokens, layer.model_width}},
        client.Require("shared", hera::PartyRole::kClient).slot_count(),
        hera::Layout::kComplexStrictColumn);
    Require(residual_plan.ok(), residual_plan.status().message());
    const std::vector<double> activation{
        -0.5, -0.2, 0.1, 0.4, 0.7,
         0.3, -0.1, 0.2, -0.4, 0.6};
    const std::vector<double> residual{
        0.2, -0.1, 0.05, 0.3,
       -0.2,  0.4, 0.15, -0.05};
    std::vector<double> wd(layer.ffn_width * layer.model_width);
    for (std::size_t i = 0; i < wd.size(); ++i)
        wd[i] = (static_cast<int>(i % 7) - 3) * 0.02;
    const std::vector<double> bd{0.01, -0.02, 0.015, 0.005};
    hera::SealPlainParameterStore server_parameters;
    hera::LayerWeightHandles weights;
    weights.layer_id = layer.id;
    weights.wd = {601, {"Wd", layer.ffn_width, layer.model_width, true}};
    weights.plain_parameters = {{701, {"bd", layer.model_width}}};
    Require(server_parameters.PutMatrix(weights.wd, wd).ok() &&
                server_parameters.PutVector(weights.plain_parameters[0], bd).ok(),
            "cannot register Wd parameters");
    hera::SealPlainParameterStore client_parameters;
    hera::SealCipherTensorStore server_he, client_he;
    Require(server_he.Put(9200002, EncryptResidual(
                residual, residual_plan.value(),
                client.Require("shared", hera::PartyRole::kClient))).ok(),
            "cannot store retained residual");

    std::vector<std::uint64_t> server_share(activation.size());
    std::vector<std::uint64_t> client_share(activation.size());
    for (std::size_t i = 0; i < activation.size(); ++i) {
        server_share[i] = (70001 + i * 104729) & kMask;
        client_share[i] = (EncodeFixed(activation[i]) - server_share[i]) & kMask;
    }
    int pipe_fd[2];
    Require(pipe(pipe_fd) == 0, "cannot create Block-5 test pipe");
    const auto child = fork();
    Require(child >= 0, "cannot fork Block-5 client");
    if (child == 0) {
        close(pipe_fd[0]);
        auto result = RunParty(hera::PartyRole::kClient, config, client,
                               client_share, client_he, client_parameters);
        if (!result.ok) std::cerr << "Block-5 client: " << result.error << '\n';
        const int value = result.ok ? 1 : 0;
        const auto written = write(pipe_fd[1], &value, sizeof(value));
        close(pipe_fd[1]);
        _exit(result.ok && written == static_cast<ssize_t>(sizeof(value))
                  ? 0 : 91);
    }
    close(pipe_fd[1]);
    auto server_result = RunParty(hera::PartyRole::kServer, config, server,
                                  server_share, server_he, server_parameters);
    if (!server_result.ok) {
        kill(child, SIGTERM);
        int ignored = 0;
        waitpid(child, &ignored, 0);
        close(pipe_fd[0]);
        throw std::runtime_error(server_result.error);
    }
    int client_ok = 0;
    const auto received = read(pipe_fd[0], &client_ok, sizeof(client_ok));
    close(pipe_fd[0]);
    int status = 0;
    waitpid(child, &status, 0);
    Require(received == sizeof(client_ok) && client_ok == 1 &&
                WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "Block-5 client process failed");
    auto hidden = server_he.Get(server_result.hidden.id);
    Require(hidden.ok(), hidden.status().message());
    auto actual = Decrypt(hidden.value(),
        client.Require("shared", hera::PartyRole::kClient));
    std::vector<double> expected(layer.tokens * layer.model_width);
    for (std::size_t token = 0; token < layer.tokens; ++token)
        for (std::size_t output = 0; output < layer.model_width; ++output) {
            double value = bd[output] +
                residual[token * layer.model_width + output];
            for (std::size_t feature = 0; feature < layer.ffn_width; ++feature)
                value += activation[token * layer.ffn_width + feature] *
                    wd[feature * layer.model_width + output];
            expected[token * layer.model_width + output] = value;
        }
    double maximum = 0.0;
    for (std::size_t i = 0; i < expected.size(); ++i)
        maximum = std::max(maximum, std::abs(actual[i] - expected[i]));
    Require(maximum < 0.03,
            "Block-5 runtime error exceeds 0.03: " +
                std::to_string(maximum));
}

void TestCrossLayerRuntime()
{
    auto config = MakeRuntimeConfig();
    auto next = config.layers.front();
    next.id = "layer1";
    next.qkv_bias = false;
    config.layers.push_back(next);
    auto execution = config.layer_execution.front();
    execution.layer_id = next.id;
    config.layer_execution.push_back(execution);
    Require(hera::ValidateHeraConfig(config).ok(),
            "cross-layer runtime config is invalid");

    hera::HeContextRegistry client;
    Require(client.CreateClientProfiles(config).ok(),
            "cannot create cross-layer client context");
    auto rotations = hera::SealSciBlock5BridgeExecutor::RequiredRotationSteps(
        config, 0);
    Require(rotations.ok() && client.CompileRotationKeys(
                "shared", rotations.value().at("shared")).ok(),
            "cannot compile cross-layer rotations");
    auto bundle = client.ExportSerializedPublicBundle();
    Require(bundle.ok(), bundle.status().message());
    hera::HeContextRegistry server;
    Require(server.CreateServerProfiles(config, bundle.value()).ok(),
            "cannot create cross-layer server context");

    const auto &layer = config.layers.front();
    const auto &client_context = client.Require(
        "shared", hera::PartyRole::kClient);
    auto residual_plan = hera::FfnGeluPackingPlanner::ComplexColumns(
        {{layer.tokens, layer.model_width}}, client_context.slot_count(),
        hera::Layout::kComplexStrictColumn);
    Require(residual_plan.ok(), residual_plan.status().message());
    const std::vector<double> activation{
        -0.45, -0.15, 0.12, 0.38, 0.64,
         0.28, -0.08, 0.19, -0.34, 0.52};
    const std::vector<double> residual{
        0.25, -0.12, 0.08, 0.31,
       -0.18,  0.36, 0.11, -0.02};
    std::vector<double> wd(layer.ffn_width * layer.model_width);
    for (std::size_t i = 0; i < wd.size(); ++i)
        wd[i] = (static_cast<int>(i % 7) - 3) * 0.018;
    const std::vector<double> bd{0.008, -0.015, 0.012, 0.004};

    hera::LayerWeightHandles current_weights;
    current_weights.layer_id = layer.id;
    current_weights.wd = {
        601, {"Wd", layer.ffn_width, layer.model_width, true}};
    current_weights.plain_parameters = {{701, {"bd", layer.model_width}}};
    hera::LayerWeightHandles next_weights;
    next_weights.layer_id = next.id;
    next_weights.wq = {611, {"Wq", next.model_width,
                             hera::QueryWidth(next), false}};
    next_weights.wk = {612, {"Wk", next.model_width,
                             hera::KeyWidth(next), false}};
    next_weights.wv = {613, {"Wv", next.model_width,
                             hera::ValueWidth(next), false}};
    next_weights.plain_parameters = {
        {711, {"ln1.gamma", next.model_width}},
        {712, {"ln1.beta", next.model_width}}};
    hera::SealPlainParameterStore server_parameters, client_parameters;
    Require(server_parameters.PutMatrix(current_weights.wd, wd).ok() &&
                server_parameters.PutVector(
                    current_weights.plain_parameters[0], bd).ok(),
            "cannot register cross-layer Wd parameters");
    std::vector<double> wq(next.model_width * hera::QueryWidth(next));
    std::vector<double> wk(next.model_width * hera::KeyWidth(next));
    std::vector<double> wv(next.model_width * hera::ValueWidth(next));
    for (std::size_t i = 0; i < wq.size(); ++i)
        wq[i] = (static_cast<int>(i % 5) - 2) * 0.025;
    for (std::size_t i = 0; i < wk.size(); ++i)
        wk[i] = (static_cast<int>(i % 3) - 1) * 0.03;
    for (std::size_t i = 0; i < wv.size(); ++i)
        wv[i] = (static_cast<int>(i % 4) - 1) * 0.02;
    Require(server_parameters.PutMatrix(next_weights.wq, wq).ok() &&
                server_parameters.PutMatrix(next_weights.wk, wk).ok() &&
                server_parameters.PutMatrix(next_weights.wv, wv).ok() &&
                server_parameters.PutVector(next_weights.plain_parameters[0],
                    std::vector<double>(next.model_width, 1.0)).ok() &&
                server_parameters.PutVector(next_weights.plain_parameters[1],
                    std::vector<double>(next.model_width, 0.0)).ok(),
            "cannot register next Block-2 parameters");

    hera::SealCipherTensorStore server_he, client_he;
    Require(server_he.Put(9200002, EncryptResidual(
                residual, residual_plan.value(), client_context)).ok(),
            "cannot store cross-layer residual");
    std::vector<std::uint64_t> server_share(activation.size());
    std::vector<std::uint64_t> client_share(activation.size());
    for (std::size_t i = 0; i < activation.size(); ++i) {
        server_share[i] = (170011 + i * 65537) & kMask;
        client_share[i] = (EncodeFixed(activation[i]) - server_share[i]) & kMask;
    }
    int pipe_fd[2];
    Require(pipe(pipe_fd) == 0, "cannot create cross-layer pipe");
    const auto child = fork();
    Require(child >= 0, "cannot fork cross-layer client");
    if (child == 0) {
        close(pipe_fd[0]);
        auto result = RunParty(hera::PartyRole::kClient, config, client,
                               client_share, client_he, client_parameters);
        if (!result.ok) std::cerr << "cross-layer client: " << result.error << '\n';
        const int value = result.ok ? 1 : 0;
        const auto written = write(pipe_fd[1], &value, sizeof(value));
        close(pipe_fd[1]);
        _exit(result.ok && written == static_cast<ssize_t>(sizeof(value))
                  ? 0 : 92);
    }
    close(pipe_fd[1]);
    auto server_result = RunParty(hera::PartyRole::kServer, config, server,
                                  server_share, server_he, server_parameters);
    if (!server_result.ok) {
        kill(child, SIGTERM);
        int ignored = 0;
        waitpid(child, &ignored, 0);
        close(pipe_fd[0]);
        throw std::runtime_error(server_result.error);
    }
    int client_ok = 0;
    const auto received = read(pipe_fd[0], &client_ok, sizeof(client_ok));
    close(pipe_fd[0]);
    int status = 0;
    waitpid(child, &status, 0);
    Require(received == sizeof(client_ok) && client_ok == 1 &&
                WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "cross-layer client process failed");
    auto hidden = server_he.Get(server_result.hidden.id);
    Require(hidden.ok(), "cross-layer current hidden state is absent");
    auto actual = Decrypt(hidden.value(), client_context);
    std::vector<double> expected(layer.tokens * layer.model_width);
    for (std::size_t token = 0; token < layer.tokens; ++token)
        for (std::size_t output = 0; output < layer.model_width; ++output) {
            double value = bd[output] +
                residual[token * layer.model_width + output];
            for (std::size_t feature = 0; feature < layer.ffn_width; ++feature)
                value += activation[token * layer.ffn_width + feature] *
                    wd[feature * layer.model_width + output];
            expected[token * layer.model_width + output] = value;
        }
    double maximum = 0.0;
    for (std::size_t i = 0; i < expected.size(); ++i)
        maximum = std::max(maximum, std::abs(actual[i] - expected[i]));
    Require(maximum < 0.03,
            "cross-layer current hidden error exceeds 0.03");

    auto score_tensor = server_he.Get(server_result.scores.id);
    auto value_tensor = server_he.Get(server_result.value.id);
    Require(score_tensor.ok() && value_tensor.ok(),
            "cross-layer Block-2 tensors cannot be loaded");
    const auto actual_scores = Decrypt(score_tensor.value(), client_context);
    const auto actual_value = Decrypt(value_tensor.value(), client_context);
    std::vector<double> normalized(expected.size());
    for (std::size_t token = 0; token < next.tokens; ++token) {
        double mean = 0.0;
        for (std::size_t feature = 0; feature < next.model_width; ++feature)
            mean += expected[token * next.model_width + feature];
        mean /= static_cast<double>(next.model_width);
        double variance = 0.0;
        for (std::size_t feature = 0; feature < next.model_width; ++feature) {
            const auto centered =
                expected[token * next.model_width + feature] - mean;
            variance += centered * centered;
        }
        variance /= static_cast<double>(next.model_width);
        const auto inverse = 1.0 / std::sqrt(
            variance + next.attention_norm_epsilon);
        for (std::size_t feature = 0; feature < next.model_width; ++feature)
            normalized[token * next.model_width + feature] =
                (expected[token * next.model_width + feature] - mean) * inverse;
    }
    std::vector<double> q(next.tokens * hera::QueryWidth(next));
    std::vector<double> k(next.tokens * hera::KeyWidth(next));
    std::vector<double> expected_value(next.tokens * hera::ValueWidth(next));
    for (std::size_t token = 0; token < next.tokens; ++token) {
        for (std::size_t output = 0; output < hera::QueryWidth(next); ++output)
            for (std::size_t feature = 0; feature < next.model_width; ++feature)
                q[token * hera::QueryWidth(next) + output] +=
                    normalized[token * next.model_width + feature] *
                    wq[feature * hera::QueryWidth(next) + output] /
                    std::sqrt(static_cast<double>(next.qk_head_width));
        for (std::size_t output = 0; output < hera::KeyWidth(next); ++output)
            for (std::size_t feature = 0; feature < next.model_width; ++feature)
                k[token * hera::KeyWidth(next) + output] +=
                    normalized[token * next.model_width + feature] *
                    wk[feature * hera::KeyWidth(next) + output];
        for (std::size_t output = 0; output < hera::ValueWidth(next); ++output)
            for (std::size_t feature = 0; feature < next.model_width; ++feature)
                expected_value[token * hera::ValueWidth(next) + output] +=
                    normalized[token * next.model_width + feature] *
                    wv[feature * hera::ValueWidth(next) + output];
    }
    std::vector<double> expected_scores(
        next.q_heads * next.tokens * next.tokens);
    const auto heads_per_kv = next.q_heads / next.kv_heads;
    for (std::size_t head = 0; head < next.q_heads; ++head)
        for (std::size_t row = 0; row < next.tokens; ++row)
            for (std::size_t column = 0; column < next.tokens; ++column)
                for (std::size_t dimension = 0;
                     dimension < next.qk_head_width; ++dimension) {
                    const auto q_column = head * next.qk_head_width + dimension;
                    const auto kv_head = head / heads_per_kv;
                    const auto k_column = kv_head * next.qk_head_width + dimension;
                    expected_scores[(head * next.tokens + row) * next.tokens +
                                    column] +=
                        q[row * hera::QueryWidth(next) + q_column] *
                        k[column * hera::KeyWidth(next) + k_column];
                }
    double value_error = 0.0, score_error = 0.0;
    for (std::size_t i = 0; i < expected_value.size(); ++i)
        value_error = std::max(value_error,
                               std::abs(actual_value[i] - expected_value[i]));
    for (std::size_t i = 0; i < expected_scores.size(); ++i)
        score_error = std::max(score_error,
                               std::abs(actual_scores[i] - expected_scores[i]));
    Require(value_error < 0.05 && score_error < 0.05,
            "cross-layer Block-2 accumulated error exceeds 0.05: V=" +
                std::to_string(value_error) + " QK=" +
                std::to_string(score_error));
}

} // namespace

int main()
{
    try {
        TestPlannerAndGraph();
        TestCrossLayerPlanner();
        TestLastLayerRuntime();
        TestCrossLayerRuntime();
        std::cout << "BLOCK5_BRIDGE_STATUS=PASS\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "BLOCK5_BRIDGE_STATUS=FAIL: " << error.what() << '\n';
        return 1;
    }
}
