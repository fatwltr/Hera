#include "hera/private_runtime.h"
#include "hera/mpc_session.h"

#include <seal/seal.h>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

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
        {"shared", {2048, {60, 35, 60, 60, 60, 60, 60, 60, 60, 60},
                    44, 44, 44, 9, 1, true,
                    hera::SecurityLevel::kNone}},
        {"r3", {2048, {60, 40, 40, 40, 40, 40, 40, 60},
                40, 40, 40, 7, 1, true, hera::SecurityLevel::kNone}},
        {"r5", {2048, {60, 60, 46, 46, 46, 46, 60, 60},
                45, 45, 46, 6, 1, true, hera::SecurityLevel::kNone}},
    };
    hera::LayerExecutionConfig execution;
    execution.layer_id = layer.id;
    execution.regions = {
        {hera::HeRegionId::kRegion1,
         {hera::HeRegionId::kRegion1, "r1", hera::Layout::kStrictColumn,
          hera::Layout::kReduction}},
        {hera::HeRegionId::kRegion2,
         {hera::HeRegionId::kRegion2, "shared", hera::Layout::kSpatial,
          hera::Layout::kAttentionOutput}},
        {hera::HeRegionId::kRegion3,
         {hera::HeRegionId::kRegion3, "r3", hera::Layout::kSoftmaxRows,
          hera::Layout::kSoftmaxRows}},
        {hera::HeRegionId::kRegion4,
         {hera::HeRegionId::kRegion4, "shared",
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
    spec.he_region = hera::HeRegionId::kRegion1;
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
        hera::CkksMpcConverter converter(
            role, {43, 13, 40, 20, true}, contexts, transport,
            hera::MakeDeterministicRandomSourceForTesting(
                role == hera::PartyRole::kClient
                    ? std::vector<std::uint8_t>{2, 3, 5, 7}
                    : std::vector<std::uint8_t>{11, 13, 17, 19}),
            "127.0.0.1", base + 100, 1);
        auto nonlinear_session = std::make_shared<hera::SharedMpcSession>(
            role, "127.0.0.1", base + 300);
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
                hera::HeRegionId::kRegion1;
            input.hidden_states.tensor.spec.he_parameter_profile = "r1";
            input.hidden_states.tensor.spec.planned_level = 0;
            input.hidden_states.tensor.spec.planned_scale_bits = 38;
        }
        input.attention_mask.kind = hera::AttentionMaskKind::kNone;
        hera::ExecutionReport report;
        auto output = transformer.value().Evaluate(input, weights, &report);
        if (!output.ok()) return {false, {}, output.status().message()};
        Require(report.stages.size() == config.layers.size() * 5,
                "runtime did not report five fused stages per layer");
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

void TestRuntime(hera::HeraConfig config, int base)
{
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
    Require(!server.Require("shared", hera::PartyRole::kServer).has_secret_key(),
            "server unexpectedly owns a secret key");

    hera::RuntimeTensorArena server_arena;
    hera::RuntimeTensorArena client_arena;
    auto input = EncryptInput(config,
        client.Require("r1", hera::PartyRole::kClient));
    Require(server_arena.he.Put(11000001, input).ok(),
            "server input cannot be registered");
    hera::SealPlainParameterStore server_parameters;
    hera::SealPlainParameterStore client_parameters;
    auto data = hera::MakeDeterministicTransformerWeights(config, 1234);
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
                    std::optional<std::size_t>{7},
            "final HE output contract is wrong");
    auto output = server_arena.he.Get(server_result.output.id);
    Require(output.ok(), output.status().message());
    const auto &client_shared = client.Require(
        "shared", hera::PartyRole::kClient);
    Require(output.value().ciphertexts.front().parms_id() ==
                LevelId(client_shared, 7),
            "final ciphertext is not at shared level 7");
    seal::Decryptor decryptor(*client_shared.context,
                              *client_shared.secret_key);
    seal::CKKSEncoder encoder(*client_shared.context);
    for (const auto &ciphertext : output.value().ciphertexts) {
        seal::Plaintext plain;
        decryptor.decrypt(ciphertext, plain);
        std::vector<std::complex<double>> decoded;
        encoder.decode(plain, decoded);
        for (const auto &value : decoded)
            Require(std::isfinite(value.real()) && std::isfinite(value.imag()),
                    "runtime output contains a non-finite value");
    }
}

} // namespace

int main()
{
    try {
        auto two_layers = MakeConfig();
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
