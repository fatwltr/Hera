#include "hera/blb_reference.h"
#include "hera/private_runtime.h"
#include "hera/mpc_session.h"
#include "hera/detail/parallel_ranges.h"
#include "hera/detail/worker_ports.h"
#include "hera/seal_wire.h"

#include <seal/seal.h>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <complex>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <limits>
#include <map>

namespace hera {
namespace {

// Each SCI IOPack owns three sockets at p, p+50, and p+100. Expand the
// nonlinear range after conversion sockets when more workers are requested.
constexpr int kConversionWorkerOffset = 100;

seal::compr_mode_type CompressionMode(const std::string &name)
{
    seal::compr_mode_type mode;
    if (name == "none") mode = seal::compr_mode_type::none;
    else if (name == "zstd") mode = static_cast<seal::compr_mode_type>(2);
    else if (name == "zlib") mode = static_cast<seal::compr_mode_type>(1);
    else throw std::invalid_argument(
        "--seal-compression must be none, zstd or zlib");
    if (!seal::Serialization::IsSupportedComprMode(mode))
        throw std::invalid_argument(
            "requested SEAL compression is unavailable in this build");
    return mode;
}

void Check(const Result<void> &result)
{
    if (!result.ok()) throw std::runtime_error(result.status().message());
}

ConversionConfig ProtocolConfig(const HeraConfig &config,
                                const PrivateRuntimeCliOptions &options)
{
    ConversionConfig protocol{
        config.mpc_numeric.ring_bits, config.mpc_numeric.scale_bits,
        config.conversion.statistical_security_bits,
        config.conversion.fft_scale_bits,
        config.conversion.allow_internal_mod_switch};
    // Model runs simulate the conversion directly, with either weight mode.
    protocol.benchmark_unchecked_range = true;
    protocol.batch_elements = options.conversion_batch_elements
        ? options.conversion_batch_elements
        : ConversionConfig::DefaultBatchElements(options.conversion_workers);
    return protocol;
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
    constexpr std::uint32_t wire_protocol_version = 12;
    scalar(wire_protocol_version);
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
        scalar(layer.attention_norm_variance_scale_bits);
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
#ifdef HERA_SEAL_ROTATION_HOISTING
    scalar(true);
#else
    scalar(false);
#endif
    scalar(options.projection_baby_steps);
    scalar(options.he_memory_mib); // The automatic BSGS split depends on this budget.
    scalar(options.he_threads);
    scalar(options.conversion_workers);
    const auto conversion_protocol = ProtocolConfig(config, options);
    scalar(conversion_protocol.batch_elements);
    scalar(conversion_protocol.benchmark_unchecked_range);
    scalar(options.mpc_workers ? options.mpc_workers : options.he_threads);
    scalar(options.stage_timeout_seconds);
    mix(options.seal_compression.data(), options.seal_compression.size());
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
                     const std::vector<seal::Ciphertext> &ciphertexts,
                     seal::compr_mode_type compression)
{
    SendU64(transport, ciphertexts.size());
    for (const auto &ciphertext : ciphertexts)
        SendSealObject(transport, ciphertext, compression);
}

std::vector<seal::Ciphertext> ReceiveCiphertexts(
    IProtocolTransport &transport, const seal::SEALContext &context)
{
    const auto count = ReceiveU64(transport);
    if (count == 0 || count > (std::uint64_t{1} << 20))
        throw std::runtime_error("invalid ciphertext batch size");
    std::vector<seal::Ciphertext> result(count);
    for (auto &ciphertext : result) {
        ciphertext = ReceiveSealObject<seal::Ciphertext>(transport, context);
    }
    return result;
}

struct InitialInputBoundary {
    std::string profile;
    HeRegionId region;
    std::size_t level;
    int scale_bits;
};

InitialInputBoundary SelectInitialInputBoundary(
    const HeraConfig &config, const CompiledLayerExecutionPlan &plan)
{
    const auto &layer = config.layers.front();
    if (layer.norm == NormPlacement::kPreNorm && layer.attention_residual) {
        // Standard Pre-LN retains raw hidden for the attention residual. It
        // must remain in the shared attention profile, even if the statistics
        // profile differs. Normalized-residual mode can enter statistics
        // directly because its skip is produced by the LN1 tail.
        return {plan.attention_output_profile, HeRegionId::kRegion2, 0,
            config.he_parameter_profiles.at(plan.attention_output_profile).default_scale_bits};
    }
    return {plan.statistics_profile, HeRegionId::kRegion1,
        plan.statistics_input_level, plan.statistics_input_scale_bits};
}

TensorSpec InputSpec(const HeraConfig &config, const HeContext &context,
                     const InitialInputBoundary &boundary)
{
    const auto &layer = config.layers.front();
    TensorSpec spec;
    spec.shape = {{layer.tokens, layer.model_width}};
    spec.domain = Domain::kHeCiphertext;
    spec.layout = Layout::kStrictColumn;
    spec.numeric = config.mpc_numeric;
    spec.he_region = boundary.region;
    spec.he_parameter_profile = context.profile;
    spec.planned_level = boundary.level;
    spec.planned_scale_bits = boundary.scale_bits;
    return spec;
}

SealCipherTensor EncryptSyntheticInput(const HeraConfig &config,
                                       const HeContext &context,
                                       const InitialInputBoundary &boundary,
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
        auto data = context.context->first_context_data();
        for (std::size_t level = 0; level < boundary.level; ++level)
            data = data->next_context_data();
        encoder.encode(slots[i], data->parms_id(), std::ldexp(
            1.0, boundary.scale_bits), plain);
        encryptor.encrypt_symmetric(plain, ciphertexts[i]);
    }
    return {context.profile, std::move(ciphertexts), InputSpec(config, context, boundary),
            packing.take_value()};
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
    const auto compression = CompressionMode(options.seal_compression);
    auto compiled = CompiledTransformerExecutionPlan::Compile(config);
    if (!compiled.ok())
        throw std::runtime_error(compiled.status().message());
    std::shared_ptr<const CompiledTransformerExecutionPlan> plan =
        std::make_shared<CompiledTransformerExecutionPlan>(
            compiled.take_value());
    auto control = MakeNetIoTransport(
        role, role == PartyRole::kClient ? options.ip : "", options.port);
    HeContextRegistry contexts;
    if (role == PartyRole::kClient) {
        Check(contexts.CreateClientProfiles(config));
        for (const auto &entry : plan->rotation_steps_by_profile)
            Check(contexts.CompileRotationKeys(entry.first, entry.second));
        SendU64(*control, RuntimeFingerprint(config, options));
        auto bundle = contexts.ExportSerializedPublicBundle(compression);
        if (!bundle.ok()) throw std::runtime_error(bundle.status().message());
        Check(SendPublicContextBundle(*control, bundle.value()));
        Check(contexts.ReleaseClientEvaluationKeys());
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
    const auto input_plan = SelectInitialInputBoundary(config, plan->layers.front());
    const auto &input_context = contexts.Require(input_plan.profile, role);
    constexpr TensorId input_id = 11000001;
    RuntimeTensorArena arena;
    SealCipherTensor client_input;
    if (role == PartyRole::kClient) {
        client_input = EncryptSyntheticInput(config, input_context, input_plan,
                                             options.input_seed);
        SendCiphertexts(*control, client_input.ciphertexts, compression);
    } else {
        auto ciphertexts = ReceiveCiphertexts(*control, *input_context.context);
        auto packing = AttentionLn2PackingPlanner::RealStrictColumns(
            {{first_layer.tokens, first_layer.model_width}},
            input_context.slot_count());
        if (!packing.ok()) throw std::runtime_error(packing.status().message());
        Check(arena.he.Put(input_id,
            {input_context.profile, std::move(ciphertexts),
             InputSpec(config, input_context, input_plan), packing.take_value()}));
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
    const auto conversion_protocol = ProtocolConfig(config, options);
    BlbReferenceConverter converter(
        role, conversion_protocol, contexts, conversion_transport,
        MakeProductionRandomSource(), options.ip,
        options.port + kConversionWorkerOffset,
        options.conversion_workers, compression);
    auto nonlinear_session = std::make_shared<SharedMpcSession>(
        role, options.ip, detail::RuntimeMpcPort(options.port, options.conversion_workers,
            options.mpc_workers ? options.mpc_workers : options.he_threads),
        options.mpc_workers ? options.mpc_workers : options.he_threads);
    MpcSoftmaxRuntime softmax(nonlinear_session);
    MpcLayerNormRuntime norm(nonlinear_session);
    MpcGeluRuntime gelu(nonlinear_session);
    auto coordinator = RuntimeCoordinator::Connect(
        role, role == PartyRole::kClient ? options.ip : "",
        options.port + 1, std::chrono::seconds(2), std::chrono::seconds(30),
        std::chrono::seconds(options.stage_timeout_seconds));
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
    // Establish independent VOLE/Silent-OT streams once in setup. Sequential
    // nonlinear stages reuse them; row/element shards run concurrently.
    Check(nonlinear_session->Prepare());
    auto runtime = std::make_shared<SealSciTransformerRuntime>(
        config, contexts, arena, parameters, converter, softmax, norm, gelu,
        coordinator.value().get(), plan);
    auto framework = HeraContext::Create(config);
    if (!framework.ok()) throw std::runtime_error(framework.status().message());
    Check(framework.value()->BindExecutionRuntime(runtime));
    auto transformer = PrivateTransformer::Compile(framework.value());
    if (!transformer.ok())
        throw std::runtime_error(transformer.status().message());
    Check(transformer.value().Prepare(weights.value()));

    EncryptedTransformerInput input;
    input.hidden_states.tensor = {input_id, InputSpec(config, input_context, input_plan)};
    input.attention_mask = MakeMask(first_layer);
    const auto total_runs = options.warmup + options.repeat;
    struct BlockMeasurement {
        ExecutionReport::CostBreakdown costs;
        std::uint64_t mpc_bytes{};
        std::uint64_t conversion_bytes{};
    };
    std::map<std::size_t, BlockMeasurement> measured_blocks;
    TensorHandle last_output;
    std::optional<SealCipherTensor> received_output;
    for (std::size_t run = 0; run < total_runs; ++run) {
        // Both parties finish Prepare and the preceding output transfer before
        // entering measured stages. Otherwise a cold client call includes the
        // server's offline plaintext compilation in its online measurement.
        constexpr std::uint64_t benchmark_ready = 0x4845524100000000ULL;
        const auto ready = benchmark_ready + run;
        if (role == PartyRole::kServer) {
            SendU64(*control, ready);
            if (ReceiveU64(*control) != ready)
                throw std::runtime_error("online benchmark readiness mismatch");
        } else {
            if (ReceiveU64(*control) != ready)
                throw std::runtime_error("online benchmark readiness mismatch");
            SendU64(*control, ready);
        }
        ExecutionReport report;
        auto output = transformer.value().Evaluate(
            input, weights.value(), &report);
        if (!output.ok()) throw std::runtime_error(output.status().message());
        if (role == PartyRole::kServer) {
            auto payload = arena.he.Get(output.value().tensor.id);
            if (!payload.ok())
                throw std::runtime_error(payload.status().message());
            SendCiphertexts(*control, payload.value().ciphertexts,
                            compression);
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
        if (run >= options.warmup) {
            for (const auto &stage : report.stages) {
                auto &block = measured_blocks[stage.layer];
                block.costs += stage.metrics.costs;
                block.mpc_bytes += stage.metrics.mpc_communication_bytes;
                block.conversion_bytes += stage.metrics.conversion_communication_bytes;
            }
        }
        last_output = output.value().tensor;
    }

    if (role == PartyRole::kClient && options.verify && !options.virtual_weights) {
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
    // The local parties share stdout. Serialize their reports after all timed
    // work so buffered writes cannot split metrics or the final PASS marker.
    constexpr std::uint64_t report_ready = 0x4845524152505431ULL;
    if (options.party == "local" && role == PartyRole::kClient &&
        ReceiveU64(*control) != report_ready)
        throw std::runtime_error("invalid local report synchronization");
    const auto print_measurement = [&](const std::string &label,
                                       const BlockMeasurement &value) {
        std::cout << label
                  << ",HE_MS=" << value.costs.he_computation_milliseconds / denominator
                  << ",MPC_MS=" << value.costs.other_nonlinear_milliseconds / denominator
                  << ",CONVERSION_MS=" << value.costs.he_mpc_conversion_milliseconds / denominator
                  << ",HE_COMM_BYTES=0"
                  << ",MPC_COMM_BYTES=" << value.mpc_bytes / denominator
                  << ",CONVERSION_COMM_BYTES=" << value.conversion_bytes / denominator
                  << '\n';
    };
    std::cout << std::fixed << std::setprecision(3)
              << "HERA_PARTY="
              << (role == PartyRole::kClient ? "client" : "server") << '\n';
    BlockMeasurement total;
    for (const auto &[layer, block] : measured_blocks) {
        print_measurement("HERA_BLOCK=layer" + std::to_string(layer), block);
        total.costs += block.costs;
        total.mpc_bytes += block.mpc_bytes;
        total.conversion_bytes += block.conversion_bytes;
    }
    if (measured_blocks.size() > 1)
        print_measurement("HERA_TOTAL", total);
    std::cout << "HERA_PRIVATE_INFERENCE_STATUS=PASS\n";
    std::cout.flush();
    if (options.party == "local" && role == PartyRole::kServer)
        SendU64(*control, report_ready);
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
        if (options.he_threads == 0 || options.conversion_workers == 0 ||
            options.conversion_batch_elements > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            options.he_memory_mib == 0 || options.he_memory_mib > (std::numeric_limits<std::size_t>::max() >> 20) ||
            options.stage_timeout_seconds == 0 || options.stage_timeout_seconds > 86400 ||
            options.repeat == 0)
            throw std::invalid_argument("invalid thread, memory or repeat budget");
        (void)detail::RuntimeMpcPort(options.port, options.conversion_workers,
            options.mpc_workers ? options.mpc_workers : options.he_threads);
        if (options.weight_template_count == 0)
            throw std::invalid_argument(
                "--weight-template-count must be positive");
        auto loaded = LoadHeraConfig(options.config);
        if (!loaded.ok()) throw std::runtime_error(loaded.status().message());
        auto config = loaded.take_value();
        ConfigureProjectionBabySteps(options.projection_baby_steps);
        config.threads = options.he_threads;
        detail::HeWorkerPool::Instance().SetMemoryBudget(
            options.he_memory_mib * (std::size_t{1} << 20));
        if (options.party == "client")
            return RunOneParty(options, config, PartyRole::kClient);
        if (options.party == "server")
            return RunOneParty(options, config, PartyRole::kServer);
        // Flush any explicitly requested graph output before forking.
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
