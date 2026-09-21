#include "hera/private_runtime.h"
#include "hera/mpc_session.h"

#include <seal/seal.h>
#include <seal/version.h>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <complex>
#include <cstring>
#include <iostream>
#include <sstream>

namespace hera {
namespace {

using Clock = std::chrono::steady_clock;

// Each SCI IOPack owns three sockets at p, p+50, and p+100. Keep complete
// ranges disjoint. All sequential nonlinear kernels share the Softmax range;
// the former LayerNorm/GeLU ranges remain reserved for wire compatibility.
constexpr int kConversionWorkerOffset = 100;
constexpr int kSoftmaxOffset = 300;
// Keep the historical +500/+700 ranges reserved so deployments can upgrade
// without changing their firewall allowance.
constexpr int kRuntimePortExtent = 800;

void Check(const Result<void> &result)
{
    if (!result.ok()) throw std::runtime_error(result.status().message());
}

std::uint64_t RuntimeFingerprint(const HeraConfig &config,
                                 const PrivateRuntimeCliOptions &options)
{
    std::uint64_t hash = 1469598103934665603ULL;
    auto mix = [&](const void *data, std::size_t size) {
        const auto *bytes = static_cast<const unsigned char *>(data);
        for (std::size_t i = 0; i < size; ++i) {
            hash ^= bytes[i];
            hash *= 1099511628211ULL;
        }
    };
    auto scalar = [&](const auto &value) { mix(&value, sizeof(value)); };
    scalar(config.schema_version);
    scalar(config.mpc_numeric.ring_bits);
    scalar(config.mpc_numeric.scale_bits);
    scalar(config.mpc_numeric.is_signed);
    for (const auto &layer : config.layers) {
        mix(layer.id.data(), layer.id.size());
        scalar(layer.tokens); scalar(layer.model_width);
        scalar(layer.q_heads); scalar(layer.kv_heads);
        scalar(layer.qk_head_width); scalar(layer.value_head_width);
        scalar(layer.ffn_width);
        const auto norm = static_cast<int>(layer.norm);
        const auto activation = static_cast<int>(layer.activation);
        const auto mask = static_cast<int>(layer.attention_mask);
        scalar(norm); scalar(activation); scalar(mask);
        scalar(layer.qkv_bias); scalar(layer.output_bias);
        scalar(layer.ffn_bias); scalar(layer.attention_residual);
        scalar(layer.ffn_residual); scalar(layer.attention_norm_epsilon);
        scalar(layer.attention_norm_rsqrt_bits);
    }
    for (const auto &entry : config.he_parameter_profiles) {
        mix(entry.first.data(), entry.first.size());
        scalar(entry.second.polynomial_degree);
        for (const auto bit : entry.second.coefficient_modulus_bits) scalar(bit);
        scalar(entry.second.input_scale_bits);
        scalar(entry.second.default_scale_bits);
        scalar(entry.second.output_scale_bits);
        scalar(entry.second.declared_depth);
        scalar(entry.second.terminal_prime_count);
        scalar(entry.second.require_exact_terminal_level);
        const auto security = static_cast<int>(entry.second.security_level);
        scalar(security);
    }
    for (const auto &execution : config.layer_execution) {
        mix(execution.layer_id.data(), execution.layer_id.size());
        for (const auto &entry : execution.regions) {
            const auto key = static_cast<int>(entry.first);
            const auto id = static_cast<int>(entry.second.id);
            const auto input = static_cast<int>(entry.second.input_layout);
            const auto output = static_cast<int>(entry.second.output_layout);
            scalar(key); scalar(id); scalar(input); scalar(output);
            mix(entry.second.parameter_profile.data(),
                entry.second.parameter_profile.size());
        }
    }
    const int placements[] = {
        static_cast<int>(config.placement.ln_statistics),
        static_cast<int>(config.placement.attention_main),
        static_cast<int>(config.placement.softmax_polynomial),
        static_cast<int>(config.placement.attention_output_ln2),
        static_cast<int>(config.placement.ffn_activation)};
    for (const auto placement : placements) scalar(placement);
    scalar(config.conversion.statistical_security_bits);
    scalar(config.conversion.fft_scale_bits);
    scalar(config.conversion.allow_internal_mod_switch);
    scalar(config.softmax.clip_threshold);
    scalar(config.softmax.masked_shift);
    scalar(config.softmax.exp_divisor_bits);
    scalar(config.softmax.exp_squarings);
    scalar(config.softmax.pad_rows_to_power_of_two);
    scalar(config.gelu.threshold); scalar(config.gelu.a);
    scalar(config.gelu.b); scalar(config.gelu.c);
    scalar(config.gelu.d); scalar(config.gelu.e);
    scalar(config.threads);
    scalar(options.virtual_weights);
    const auto effective_templates = options.virtual_weights
        ? options.weight_template_count : std::size_t{0};
    scalar(effective_templates);
    scalar(options.threads);
    return hash;
}

void SendU64(IProtocolTransport &transport, std::uint64_t value)
{
    transport.Send(&value, sizeof(value));
}

std::uint64_t ReceiveU64(IProtocolTransport &transport)
{
    std::uint64_t value{};
    transport.Receive(&value, sizeof(value));
    return value;
}

void SendCiphertexts(IProtocolTransport &transport,
                     const std::vector<seal::Ciphertext> &ciphertexts)
{
    SendU64(transport, ciphertexts.size());
    for (const auto &ciphertext : ciphertexts) {
        std::stringstream stream;
        ciphertext.save(stream, seal::compr_mode_type::none);
        const auto bytes = stream.str();
        SendU64(transport, bytes.size());
        if (!bytes.empty()) transport.Send(bytes.data(), bytes.size());
    }
}

std::vector<seal::Ciphertext> ReceiveCiphertexts(
    IProtocolTransport &transport, const seal::SEALContext &context)
{
    const auto count = ReceiveU64(transport);
    if (count == 0 || count > (std::uint64_t{1} << 20))
        throw std::runtime_error("invalid ciphertext batch size");
    std::vector<seal::Ciphertext> result(count);
    for (auto &ciphertext : result) {
        const auto size = ReceiveU64(transport);
        if (size == 0 || size > (std::uint64_t{1} << 34))
            throw std::runtime_error("invalid serialized ciphertext size");
        std::string bytes(size, '\0');
        transport.Receive(bytes.data(), bytes.size());
        std::stringstream stream(bytes);
        ciphertext.load(context, stream);
    }
    return result;
}

const HeRegionConfig &Region(const HeraConfig &config,
                             const TransformerLayerConfig &layer,
                             HeRegionId id)
{
    const auto *execution = FindLayerExecution(config, layer.id);
    if (!execution) throw std::runtime_error("layer placement is absent");
    const auto found = execution->regions.find(id);
    if (found == execution->regions.end())
        throw std::runtime_error("HE region is absent");
    return found->second;
}

TensorSpec InputSpec(const HeraConfig &config, const HeContext &context)
{
    const auto &layer = config.layers.front();
    TensorSpec spec;
    spec.shape = {{layer.tokens, layer.model_width}};
    spec.domain = Domain::kHeCiphertext;
    spec.layout = Layout::kStrictColumn;
    spec.numeric = config.mpc_numeric;
    spec.he_region = HeRegionId::kRegion1;
    spec.he_parameter_profile = context.profile;
    spec.planned_level = 0;
    spec.planned_scale_bits = context.parameters.input_scale_bits;
    return spec;
}

SealCipherTensor EncryptSyntheticInput(const HeraConfig &config,
                                       const HeContext &context,
                                       std::uint64_t seed)
{
    const auto &layer = config.layers.front();
    const TensorShape shape{{layer.tokens, layer.model_width}};
    auto packing = AttentionLn2PackingPlanner::RealStrictColumns(
        shape, context.slot_count());
    if (!packing.ok()) throw std::runtime_error(packing.status().message());
    std::vector<std::vector<std::complex<double>>> slots(
        packing.value().ciphertext_count,
        std::vector<std::complex<double>>(context.slot_count()));
    auto next = [&seed]() {
        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
        return (static_cast<double>(seed % 2001) / 1000.0 - 1.0) * 0.25;
    };
    for (std::size_t logical = 0;
         logical < packing.value().logical_locations.size(); ++logical) {
        const auto &where = packing.value().Locations(logical).front();
        slots[where.ciphertext][where.slot] = next();
    }
    seal::CKKSEncoder encoder(*context.context);
    seal::Encryptor encryptor(*context.context, *context.secret_key);
    std::vector<seal::Ciphertext> ciphertexts(slots.size());
    for (std::size_t i = 0; i < slots.size(); ++i) {
        seal::Plaintext plain;
        encoder.encode(slots[i], std::ldexp(
            1.0, context.parameters.input_scale_bits), plain);
        encryptor.encrypt_symmetric(plain, ciphertexts[i]);
    }
    return {context.profile, std::move(ciphertexts), InputSpec(config, context),
            packing.take_value()};
}

ConversionConfig ProtocolConfig(const HeraConfig &config)
{
    return {config.mpc_numeric.ring_bits, config.mpc_numeric.scale_bits,
            config.conversion.statistical_security_bits,
            config.conversion.fft_scale_bits,
            config.conversion.allow_internal_mod_switch};
}

PublicAttentionMask MakeMask(const TransformerLayerConfig &layer)
{
    PublicAttentionMask mask;
    mask.kind = layer.attention_mask;
    if (mask.kind != AttentionMaskKind::kNone)
        mask.shape = {{layer.tokens, layer.tokens}};
    if (mask.kind == AttentionMaskKind::kPublicPadding)
        mask.additive_values.assign(layer.tokens * layer.tokens, 0.0);
    return mask;
}

int RunOneParty(const PrivateRuntimeCliOptions &options,
                const HeraConfig &config, PartyRole role)
{
    const auto setup_started = Clock::now();
    auto control = MakeNetIoTransport(
        role, role == PartyRole::kClient ? options.ip : "", options.port);
    HeContextRegistry contexts;
    if (role == PartyRole::kClient) {
        Check(contexts.CreateClientProfiles(config));
        auto rotations = SealSciTransformerRuntime::RequiredRotationSteps(config);
        if (!rotations.ok())
            throw std::runtime_error(rotations.status().message());
        for (const auto &entry : rotations.value())
            Check(contexts.CompileRotationKeys(entry.first, entry.second));
        SendU64(*control, RuntimeFingerprint(config, options));
        auto bundle = contexts.ExportSerializedPublicBundle();
        if (!bundle.ok()) throw std::runtime_error(bundle.status().message());
        Check(SendPublicContextBundle(*control, bundle.value()));
    } else {
        if (ReceiveU64(*control) != RuntimeFingerprint(config, options))
            throw std::runtime_error(
                "client/server configs or benchmark modes differ");
        auto bundle = ReceivePublicContextBundle(*control);
        if (!bundle.ok()) throw std::runtime_error(bundle.status().message());
        Check(contexts.CreateServerProfiles(config, bundle.value()));
        for (const auto &entry : config.he_parameter_profiles)
            if (contexts.Require(entry.first, PartyRole::kServer).has_secret_key())
                throw std::runtime_error("server received a secret key");
    }

    const auto &first_layer = config.layers.front();
    const auto &input_context = contexts.Require(
        Region(config, first_layer, HeRegionId::kRegion1).parameter_profile,
        role);
    constexpr TensorId input_id = 11000001;
    RuntimeTensorArena arena;
    SealCipherTensor client_input;
    if (role == PartyRole::kClient) {
        client_input = EncryptSyntheticInput(config, input_context,
                                             options.input_seed);
        SendCiphertexts(*control, client_input.ciphertexts);
    } else {
        auto ciphertexts = ReceiveCiphertexts(*control, *input_context.context);
        auto packing = AttentionLn2PackingPlanner::RealStrictColumns(
            {{first_layer.tokens, first_layer.model_width}},
            input_context.slot_count());
        if (!packing.ok()) throw std::runtime_error(packing.status().message());
        Check(arena.he.Put(input_id,
            {input_context.profile, std::move(ciphertexts),
             InputSpec(config, input_context), packing.take_value()}));
    }

    SealPlainParameterStore parameters;
    Check(parameters.ConfigureVirtualWeightPlaintexts(
        options.virtual_weights ? options.weight_template_count : 0));
    std::optional<TransformerWeightData> weight_data;
    if (role == PartyRole::kServer)
        weight_data = MakeDeterministicTransformerWeights(
            config, options.weight_seed);
    auto weights = RegisterTransformerWeights(config, weight_data, parameters);
    if (!weights.ok()) throw std::runtime_error(weights.status().message());

    auto conversion_transport = MakeNetIoTransport(
        role, role == PartyRole::kClient ? options.ip : "", options.port + 10);
    CkksMpcConverter converter(
        role, ProtocolConfig(config), contexts, conversion_transport,
        MakeProductionRandomSource(), options.ip,
        options.port + kConversionWorkerOffset,
        options.threads);
    auto nonlinear_session = std::make_shared<SharedMpcSession>(
        role, options.ip, options.port + kSoftmaxOffset);
    MpcSoftmaxRuntime softmax(nonlinear_session);
    MpcLayerNormRuntime norm(nonlinear_session);
    MpcGeluRuntime gelu(nonlinear_session);
    auto coordinator = RuntimeCoordinator::Connect(
        role, role == PartyRole::kClient ? options.ip : "",
        options.port + 1);
    if (!coordinator.ok())
        throw std::runtime_error(coordinator.status().message());
    coordinator.value()->SetCancellationCallback([&] {
        converter.Cancel();
        softmax.Cancel();
        norm.Cancel();
        gelu.Cancel();
        control->Cancel();
    });
    Check(coordinator.value()->Start());
    // Establish the large VOLE/Silent-OT state once in setup.  All nonlinear
    // kernels consume the same ordered stream during online evaluation.
    Check(nonlinear_session->Prepare());
    auto runtime = std::make_shared<SealSciTransformerRuntime>(
        config, contexts, arena, parameters, converter, softmax, norm, gelu,
        coordinator.value().get());
    auto framework = HeraContext::Create(config);
    if (!framework.ok()) throw std::runtime_error(framework.status().message());
    Check(framework.value()->BindExecutionRuntime(runtime));
    auto transformer = PrivateTransformer::Compile(framework.value());
    if (!transformer.ok())
        throw std::runtime_error(transformer.status().message());
    PreparationReport preparation;
    Check(transformer.value().Prepare(weights.value(), &preparation));
    const auto setup_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - setup_started).count();

    EncryptedTransformerInput input;
    input.hidden_states.tensor = {input_id, InputSpec(config, input_context)};
    input.attention_mask = MakeMask(first_layer);
    const auto total_runs = options.warmup + options.repeat;
    double measured_online = 0.0, measured_transfer = 0.0;
    TensorHandle last_output;
    std::optional<SealCipherTensor> received_output;
    ExecutionReport last_report;
    for (std::size_t run = 0; run < total_runs; ++run) {
        const auto online_started = Clock::now();
        ExecutionReport report;
        auto output = transformer.value().Evaluate(
            input, weights.value(), &report);
        if (!output.ok()) throw std::runtime_error(output.status().message());
        const auto online_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - online_started).count();
        const auto transfer_started = Clock::now();
        if (role == PartyRole::kServer) {
            auto payload = arena.he.Get(output.value().tensor.id);
            if (!payload.ok())
                throw std::runtime_error(payload.status().message());
            SendCiphertexts(*control, payload.value().ciphertexts);
        } else {
            if (!output.value().tensor.spec.he_parameter_profile)
                throw std::runtime_error("output HE profile is absent");
            const auto &output_context = contexts.Require(
                *output.value().tensor.spec.he_parameter_profile, role);
            auto ciphertexts = ReceiveCiphertexts(
                *control, *output_context.context);
            auto packing = AttentionLn2PackingPlanner::RealStrictColumns(
                output.value().tensor.spec.shape, output_context.slot_count());
            if (!packing.ok())
                throw std::runtime_error(packing.status().message());
            received_output = SealCipherTensor{
                output_context.profile, std::move(ciphertexts),
                output.value().tensor.spec, packing.take_value()};
        }
        const auto transfer_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - transfer_started).count();
        if (run >= options.warmup) {
            measured_online += online_ms;
            measured_transfer += transfer_ms;
        }
        last_output = output.value().tensor;
        last_report = std::move(report);
    }

    if (role == PartyRole::kClient && options.verify) {
        const auto &output_context = contexts.Require(
            *last_output.spec.he_parameter_profile, role);
        seal::Decryptor decryptor(*output_context.context,
                                  *output_context.secret_key);
        seal::CKKSEncoder encoder(*output_context.context);
        for (const auto &ciphertext : received_output->ciphertexts) {
            seal::Plaintext plain;
            decryptor.decrypt(ciphertext, plain);
            std::vector<std::complex<double>> decoded;
            encoder.decode(plain, decoded);
            for (const auto &value : decoded)
                if (!std::isfinite(value.real()) ||
                    !std::isfinite(value.imag()))
                    throw std::runtime_error(
                        "final decrypted output is not finite");
        }
    }
    Check(coordinator.value()->Finish());
    const auto denominator = static_cast<double>(options.repeat);
    for (const auto &stage : last_report.stages) {
        const auto &metrics = stage.metrics;
        std::cout << "HERA_STAGE=layer" << stage.layer << '.' << stage.name
                  << ",MS=" << stage.milliseconds
                  << ",COMM=" << metrics.communication_bytes
                  << ",ROUNDS=" << metrics.rounds
                  << ",CONVERSIONS=" << metrics.conversions
                  << ",ROTATIONS=" << metrics.rotations
                  << ",CT_PT_MUL="
                  << metrics.ciphertext_plain_multiplications
                  << ",CT_CT_MUL=" << metrics.ciphertext_multiplications
                  << ",RELIN=" << metrics.relinearizations
                  << ",RESCALE=" << metrics.rescales
                  << ",MPC_OPS=" << metrics.mpc_operations << '\n';
    }
    std::cout << "HERA_PARTY="
              << (role == PartyRole::kClient ? "client" : "server")
              << "\nHERA_SEAL_VERSION=" << SEAL_VERSION_MAJOR << '.'
              << SEAL_VERSION_MINOR << '.' << SEAL_VERSION_PATCH
              << "\nHERA_SETUP_MS=" << setup_ms
              << "\nHERA_PREPARE_MS=" << preparation.milliseconds
              << "\nHERA_PREPARED_PLAINTEXTS="
              << preparation.encoded_plaintext_count
              << "\nHERA_PREPARED_BYTES=" << preparation.encoded_bytes
              << "\nHERA_PREPARE_CACHE_HITS=" << preparation.cache_hits
              << "\nHERA_WEIGHT_MODE="
              << (options.virtual_weights ? "VIRTUAL" : "MATERIALIZED")
              << "\nHERA_WEIGHT_TEMPLATE_COUNT="
              << (options.virtual_weights ? options.weight_template_count : 0)
              << "\nHERA_VIRTUAL_WEIGHT_REQUESTS="
              << preparation.virtual_weight_requests
              << "\nHERA_VIRTUAL_WEIGHT_CACHE_HITS="
              << preparation.virtual_weight_cache_hits
              << "\nHERA_MODEL_NUMERICAL_VALIDATION="
              << (options.virtual_weights ? "DISABLED" : "FINAL_FINITE_ONLY")
              << "\nHERA_ONLINE_MODEL_ENCODES=0"
              << "\nHERA_ONLINE_MS=" << measured_online / denominator
              << "\nHERA_OUTPUT_TRANSFER_MS="
              << measured_transfer / denominator
              << "\nHERA_END_TO_END_ONLINE_MS="
              << (measured_online + measured_transfer) / denominator
              << "\nHERA_COMMUNICATION_BYTES="
              << last_report.communication_bytes
              << "\nHERA_ROUNDS=" << last_report.rounds
              << "\nHERA_FINAL_LEVEL="
              << last_output.spec.planned_level.value_or(9999)
              << "\nHERA_PRIVATE_INFERENCE_STATUS=PASS\n";
    return 0;
}

} // namespace

int RunPrivateRuntimeCli(const PrivateRuntimeCliOptions &options)
{
    try {
        if (options.party != "local" && options.party != "client" &&
            options.party != "server")
            throw std::invalid_argument(
                "--party must be local, client, or server");
        if (options.verify && options.party != "local")
            throw std::invalid_argument(
                "--verify is restricted to --party=local test runs");
        if (options.port < 1024 ||
            options.port + kRuntimePortExtent >= 32768)
            throw std::invalid_argument("runtime port range is invalid");
        if (options.threads == 0 || options.repeat == 0)
            throw std::invalid_argument("threads/repeat must be positive");
        if (options.weight_template_count == 0)
            throw std::invalid_argument(
                "--weight-template-count must be positive");
        auto loaded = LoadHeraConfig(options.config);
        if (!loaded.ok()) throw std::runtime_error(loaded.status().message());
        const auto config = loaded.take_value();
        if (options.party == "client")
            return RunOneParty(options, config, PartyRole::kClient);
        if (options.party == "server")
            return RunOneParty(options, config, PartyRole::kServer);
        // main() prints the validated configuration before entering the
        // runtime. Flush it so the forked client cannot replay the parent's
        // buffered banner.
        std::cout.flush();
        std::cerr.flush();
        const auto child = fork();
        if (child < 0) throw std::runtime_error("fork failed");
        if (child == 0) {
            try {
                const auto status = RunOneParty(
                    options, config, PartyRole::kClient);
                std::cout.flush();
                std::cerr.flush();
                _exit(status);
            } catch (const std::exception &error) {
                std::cerr << "client: " << error.what() << '\n';
                std::cerr.flush();
                _exit(2);
            }
        }
        const int server_result = RunOneParty(
            options, config, PartyRole::kServer);
        int child_status = 0;
        waitpid(child, &child_status, 0);
        if (!WIFEXITED(child_status) || WEXITSTATUS(child_status) != 0)
            throw std::runtime_error("local client failed");
        return server_result;
    } catch (const std::exception &error) {
        std::cerr << "HERA_PRIVATE_INFERENCE_STATUS=FAIL: " << error.what()
                  << '\n';
        return 1;
    }
}

} // namespace hera
