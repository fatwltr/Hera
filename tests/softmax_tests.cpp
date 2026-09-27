#include "hera/softmax.h"

#include <seal/seal.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <sys/wait.h>
#include <signal.h>
#include <unistd.h>

#ifndef HERA_TEST_CONFIG_PATH
#define HERA_TEST_CONFIG_PATH "configs/figure10_smoke.json"
#endif

namespace {

void Require(bool condition, const std::string &message)
{
    if (!condition) throw std::runtime_error(message);
}

std::uint64_t Encode(double value, int scale, int ring)
{
    const auto integer = static_cast<std::int64_t>(
        std::llround(std::ldexp(value, scale)));
    return static_cast<std::uint64_t>(integer) &
           ((std::uint64_t{1} << ring) - 1);
}

void TestGraph(const hera::HeraConfig &config)
{
    auto graph = hera::BuildTransformerGraph(config);
    Require(graph.ok(), graph.status().message());
    bool row_max = false;
    bool block3 = false;
    bool mpc_sum = false;
    bool mpc_normalize = false;
    bool old_he_sum = false;
    bool final_bridge = false;
    for (const auto &node : graph.value().nodes()) {
        row_max = row_max || node.kind == hera::OpKind::kMpcReduceMax;
        block3 = block3 || node.name.find("softmax.he_square_5") !=
                               std::string::npos;
        mpc_sum = mpc_sum || node.kind == hera::OpKind::kMpcReduceSum;
        mpc_normalize = mpc_normalize ||
                        node.kind == hera::OpKind::kMpcMultiplyFixed;
        old_he_sum = old_he_sum || node.name.find("softmax.row_sum") !=
                                     std::string::npos;
        final_bridge = final_bridge ||
            node.name.find("attention_output.probabilities.mpc_to_he") !=
                std::string::npos;
    }
    Require(row_max && block3 && mpc_sum && mpc_normalize,
            "staged Softmax graph nodes are incomplete");
    Require(!old_he_sum, "obsolete HE Softmax row sum remains in graph");
    Require(final_bridge,
            "MPC-to-AxV level-2 boundary is not explicit in full graph");
}

void TestExponent(const hera::HeraConfig &config)
{
    hera::HeContextRegistry contexts;
    auto created = contexts.CreateClientProfiles(config);
    Require(created.ok(), created.status().message());
    const auto *execution = hera::FindLayerExecution(
        config, config.layers.front().id);
    Require(execution != nullptr, "layer execution is absent");
    const auto profile = execution->regions.at(
        hera::HeRegionId::kRegion3).parameter_profile;
    const auto &context = contexts.Require(profile, hera::PartyRole::kClient);
    auto planner = hera::Block3LevelPlanner::Validate(
        context.parameters, config.softmax);
    Require(planner.ok(), planner.status().message());

    const std::vector<double> input{-1.0, 0.0, -2.0, -12.5};
    std::vector<double> slots(context.slot_count(), 0.0);
    std::copy(input.begin(), input.end(), slots.begin());
    seal::CKKSEncoder encoder(*context.context);
    seal::Plaintext plain;
    encoder.encode(slots, std::ldexp(
        1.0, context.parameters.input_scale_bits), plain);
    seal::Encryptor encryptor(*context.context, context.secret_key.value());
    seal::Ciphertext ciphertext;
    encryptor.encrypt_symmetric(plain, ciphertext);
    const auto map = hera::PhysicalSlotMap::DenseReal(
        input.size(), context.slot_count());
    hera::SealCipherBatch batch{
        profile, {ciphertext}, {{1, 1, input.size()}},
        hera::Layout::kSoftmaxRows, map};
    hera::SoftmaxExecutionReport report;
    auto output = hera::SoftmaxExponentKernel::Evaluate(
        batch, context, config.softmax, &report);
    Require(output.ok(), output.status().message());
    Require(report.ciphertext_plain_multiplications == 1 &&
                report.ciphertext_multiplications == 6 &&
                report.relinearizations == 6 && report.rescales == 7 &&
                report.rotations == 0,
            "Block-3 operation accounting is wrong");
    Require(output.value().ciphertexts.front().parms_id() ==
                context.context->last_parms_id(),
            "Block-3 output is not terminal-level");

    seal::Decryptor decryptor(*context.context, context.secret_key.value());
    decryptor.decrypt(output.value().ciphertexts.front(), plain);
    std::vector<double> decoded;
    encoder.decode(plain, decoded);
    for (std::size_t i = 0; i < input.size(); ++i) {
        const auto expected = std::pow(
            1.0 + std::ldexp(input[i], -config.softmax.exp_divisor_bits),
            std::uint64_t{1} << config.softmax.exp_squarings);
        Require(std::abs(decoded[i] - expected) < 2e-3,
                "Block-3 exponent approximation mismatch");
    }

    // Force two independent HE ciphertexts through the bounded parallel
    // path; both outputs must retain the same numerical and level contract.
    hera::SealCipherBatch parallel_batch{
        profile, {ciphertext, ciphertext},
        {{1, 1, context.slot_count() + input.size()}},
        hera::Layout::kSoftmaxRows,
        hera::PhysicalSlotMap::DenseReal(
            context.slot_count() + input.size(), context.slot_count())};
    hera::SoftmaxExecutionReport parallel_report;
    auto parallel_output = hera::SoftmaxExponentKernel::Evaluate(
        parallel_batch, context, config.softmax, &parallel_report, 2);
    Require(parallel_output.ok(), parallel_output.status().message());
    Require(parallel_report.ciphertext_multiplications == 12 &&
                parallel_report.rescales == 14,
            "parallel Block-3 operation accounting is wrong");
    for (const auto &computed : parallel_output.value().ciphertexts) {
        Require(computed.parms_id() == context.context->last_parms_id(),
                "parallel Block-3 output is not terminal-level");
        decryptor.decrypt(computed, plain);
        encoder.decode(plain, decoded);
        for (std::size_t i = 0; i < input.size(); ++i) {
            const auto expected = std::pow(
                1.0 + std::ldexp(input[i], -config.softmax.exp_divisor_bits),
                std::uint64_t{1} << config.softmax.exp_squarings);
            Require(std::abs(decoded[i] - expected) < 2e-3,
                    "parallel Block-3 exponent mismatch");
        }
    }
}

void TestRowPacking(const hera::HeraConfig &root_config)
{
    auto layer = root_config.layers.front();
    layer.tokens = 3;
    layer.q_heads = 2;
    auto map = hera::SoftmaxPackingPlanner::RealRows(layer, 8, true);
    Require(map.ok(), map.status().message());
    Require(map.value().ciphertext_count == 3 &&
                map.value().logical_element_count == 18,
            "non-power-of-two Softmax packing size is wrong");
    std::vector<bool> seen(18, false);
    std::size_t padding = 0;
    for (const auto &assignment : map.value().assignments) {
        Require(assignment.imaginary_logical_index ==
                    hera::kUnusedLogicalIndex,
                "Softmax row packing used an imaginary lane");
        if (assignment.real_logical_index == hera::kUnusedLogicalIndex) {
            ++padding;
        } else {
            Require(!seen[assignment.real_logical_index],
                    "Softmax row packing duplicated a logical value");
            seen[assignment.real_logical_index] = true;
        }
    }
    Require(std::all_of(seen.begin(), seen.end(), [](bool value) {
                return value;
            }) && padding == 6,
            "Softmax row packing did not preserve padded rows");
}

struct PartyResult {
    std::unique_ptr<hera::MpcShareTensor> shifted;
    std::unique_ptr<hera::MpcShareTensor> probabilities;
    std::string error;
};

void RunMpcParty(hera::PartyRole role, int port,
                 const hera::TransformerLayerConfig &layer,
                 const hera::SoftmaxConfig &config,
                 const hera::PublicAttentionMask &mask,
                 const hera::MpcShareTensor &scores,
                 const hera::MpcShareTensor &exponentials,
                 PartyResult *result)
{
    try {
        hera::MpcSoftmaxRuntime runtime(role, "127.0.0.1", port);
        auto shifted = runtime.CenterAndMask(scores, layer, mask, config);
        if (!shifted.ok()) {
            result->error = shifted.status().message();
            return;
        }
        result->shifted = std::make_unique<hera::MpcShareTensor>(
            shifted.value());
        auto probability = runtime.ClipAndNormalize(
            shifted.value(), exponentials, layer, mask, config);
        if (!probability.ok()) {
            result->error = probability.status().message();
            return;
        }
        result->probabilities = std::make_unique<hera::MpcShareTensor>(
            probability.take_value());
    } catch (const std::exception &error) {
        result->error = error.what();
    }
}

void TestMpc(const hera::HeraConfig &root_config)
{
    auto layer = root_config.layers.front();
    layer.tokens = 2;
    layer.q_heads = 1;
    layer.kv_heads = 1;
    layer.attention_mask = hera::AttentionMaskKind::kPublicPadding;
    hera::PublicAttentionMask mask;
    mask.kind = hera::AttentionMaskKind::kPublicPadding;
    mask.shape = {{2, 2}};
    mask.additive_values = {
        0.0, -std::numeric_limits<double>::infinity(), 0.0, 0.0};
    const hera::NumericFormat signed_format{43, 13, true};
    const hera::NumericFormat unsigned_format{43, 13, false};
    const std::vector<double> clear_scores{1.0, 2.0, -1.0, 1.0};
    const std::vector<double> clear_shifted{0.0, -14.0, -2.0, 0.0};
    std::vector<double> clear_exponentials(4);
    std::vector<std::uint64_t> clear_score_bits(4), clear_exp_bits(4);
    for (std::size_t i = 0; i < 4; ++i) {
        clear_score_bits[i] = Encode(clear_scores[i], 13, 43);
        clear_exponentials[i] = std::pow(
            1.0 + clear_shifted[i] / 64.0, 64.0);
        clear_exp_bits[i] = Encode(clear_exponentials[i], 13, 43);
    }
    const auto ring_mask = (std::uint64_t{1} << 43) - 1;
    std::vector<std::uint64_t> server_score(4), client_score(4);
    std::vector<std::uint64_t> server_exp(4), client_exp(4);
    for (std::size_t i = 0; i < 4; ++i) {
        server_score[i] = (0x12345 + 97 * i) & ring_mask;
        client_score[i] = (clear_score_bits[i] - server_score[i]) & ring_mask;
        server_exp[i] = (0x54321 + 53 * i) & ring_mask;
        client_exp[i] = (clear_exp_bits[i] - server_exp[i]) & ring_mask;
    }
    const hera::TensorShape shape{{2, 1, 2}};
    const hera::MpcShareTensor server_scores{
        server_score, shape, signed_format, hera::ComplexComponent::kReal};
    const hera::MpcShareTensor client_scores{
        client_score, shape, signed_format, hera::ComplexComponent::kReal};
    const hera::MpcShareTensor server_exps{
        server_exp, shape, unsigned_format, hera::ComplexComponent::kReal};
    const hera::MpcShareTensor client_exps{
        client_exp, shape, unsigned_format, hera::ComplexComponent::kReal};

    struct ChildPacket {
        int ok{};
        std::uint64_t shifted[4]{};
        std::uint64_t probabilities[4]{};
    };
    int result_pipe[2];
    Require(pipe(result_pipe) == 0, "cannot create MPC result pipe");
    constexpr int port = 19870;
    const pid_t child = fork();
    Require(child >= 0, "cannot fork MPC client");
    if (child == 0) {
        close(result_pipe[0]);
        PartyResult client_result;
        RunMpcParty(hera::PartyRole::kClient, port, layer,
                    root_config.softmax, mask, client_scores, client_exps,
                    &client_result);
        ChildPacket packet;
        packet.ok = client_result.error.empty() &&
                    client_result.probabilities != nullptr;
        if (packet.ok) {
            std::copy(client_result.shifted->local_share.begin(),
                      client_result.shifted->local_share.end(),
                      packet.shifted);
            std::copy(client_result.probabilities->local_share.begin(),
                      client_result.probabilities->local_share.end(),
                      packet.probabilities);
        }
        const auto written = write(result_pipe[1], &packet, sizeof(packet));
        close(result_pipe[1]);
        _exit(written == static_cast<ssize_t>(sizeof(packet)) && packet.ok
                  ? 0 : 2);
    }
    close(result_pipe[1]);
    PartyResult server_result;
    RunMpcParty(hera::PartyRole::kServer, port, layer,
                root_config.softmax, mask, server_scores, server_exps,
                &server_result);
    ChildPacket packet;
    const auto received = read(result_pipe[0], &packet, sizeof(packet));
    close(result_pipe[0]);
    int child_status = 0;
    waitpid(child, &child_status, 0);
    Require(server_result.error.empty(), server_result.error);
    Require(received == static_cast<ssize_t>(sizeof(packet)) && packet.ok &&
                WIFEXITED(child_status) && WEXITSTATUS(child_status) == 0,
            "MPC client failed");
    Require(server_result.probabilities != nullptr,
            "MPC Softmax did not return probability shares");
    for (std::size_t index = 0; index < 4; ++index) {
        const auto bits = (server_result.shifted->local_share[index] +
                           packet.shifted[index]) & ring_mask;
        const auto signed_bits = bits >= (std::uint64_t{1} << 42)
            ? static_cast<std::int64_t>(bits - (std::uint64_t{1} << 43))
            : static_cast<std::int64_t>(bits);
        const double shifted = std::ldexp(
            static_cast<double>(signed_bits), -13);
        Require(std::abs(shifted - clear_shifted[index]) < 1e-6,
                "MPC Softmax centered value mismatch at index " +
                    std::to_string(index) + ": actual=" +
                    std::to_string(shifted) + ", expected=" +
                    std::to_string(clear_shifted[index]));
    }
    for (std::size_t row = 0; row < 2; ++row) {
        double sum = 0.0;
        const double expected_sum = row == 0
            ? clear_exponentials[0]
            : clear_exponentials[2] + clear_exponentials[3];
        for (std::size_t column = 0; column < 2; ++column) {
            const auto index = row * 2 + column;
            const auto bits =
                (server_result.probabilities->local_share[index] +
                 packet.probabilities[index]) & ring_mask;
            const double probability = std::ldexp(
                static_cast<double>(bits), -13);
            const double expected = index == 1 ? 0.0
                : clear_exponentials[index] / expected_sum;
            Require(probability >= 0.0 && probability <= 1.0 &&
                        std::abs(probability - expected) < 0.02,
                    "MPC Softmax probability mismatch at index " +
                        std::to_string(index) + ": actual=" +
                        std::to_string(probability) + ", expected=" +
                        std::to_string(expected));
            sum += probability;
        }
        Require(std::abs(sum - 1.0) < 0.02,
                "MPC Softmax probability row sum=" + std::to_string(sum));
    }
}

hera::HeraConfig IntegrationConfig()
{
    hera::HeraConfig config;
    config.layers.push_back(hera::TransformerLayerConfig{
        "softmax_integration", 2, 4, 1, 1, 4, 4, 8,
        hera::NormPlacement::kPreNorm, hera::ActivationKind::kGelu,
        hera::AttentionMaskKind::kNone, false, false, false, true, true});
    config.mpc_numeric = {43, 13, true};
    config.conversion = {40, 20, true};
    // Keep the orchestration test small; TestExponent separately validates
    // the production six-square Block-3 schedule.
    config.softmax = {-13.0, -14.0, 6, 1, true};
    config.he_parameter_profiles.emplace(
        "scores", hera::HeParameterConfig{
            2048, {40, 30, 40}, 20, 20, 20, 1, 1, true,
            hera::SecurityLevel::kNone});
    config.he_parameter_profiles.emplace(
        "block3", hera::HeParameterConfig{
            2048, {40, 30, 40}, 30, 30, 30, 2, 1, true,
            hera::SecurityLevel::kNone});
    hera::LayerExecutionConfig execution;
    execution.layer_id = config.layers.front().id;
    for (const auto region : {hera::HeRegionId::kRegion1,
                              hera::HeRegionId::kRegion2,
                              hera::HeRegionId::kRegion3,
                              hera::HeRegionId::kRegion4,
                              hera::HeRegionId::kRegion5}) {
        const auto profile = region == hera::HeRegionId::kRegion3
            ? "block3" : "scores";
        execution.regions.emplace(region, hera::HeRegionConfig{
            region, profile, hera::Layout::kSpatial,
            hera::Layout::kSpatial});
    }
    config.layer_execution.push_back(std::move(execution));
    config.threads = 1;
    return config;
}

hera::SealCipherTensor MakeEncryptedScores(
    const hera::HeraConfig &config,
    const hera::HeContext &client_context)
{
    const auto &layer = config.layers.front();
    auto packing = hera::MultiHeadPackingPlan::AttentionDiagonal(
        layer, client_context.slot_count());
    Require(packing.ok(), packing.status().message());
    const std::vector<double> scores{1.0, 2.0, -1.0, 1.0};
    std::vector<std::vector<std::complex<double>>> slots(
        packing.value().ciphertext_count,
        std::vector<std::complex<double>>(client_context.slot_count()));
    for (std::size_t logical = 0; logical < scores.size(); ++logical) {
        const auto &location = packing.value().Locations(logical).front();
        auto &slot = slots[location.ciphertext][location.slot];
        if (location.component == hera::ComplexLane::kReal)
            slot.real(scores[logical]);
        else
            slot.imag(scores[logical]);
    }
    seal::CKKSEncoder encoder(*client_context.context);
    seal::Encryptor encryptor(*client_context.context,
                              client_context.secret_key.value());
    std::vector<seal::Ciphertext> ciphertexts(slots.size());
    for (std::size_t i = 0; i < slots.size(); ++i) {
        seal::Plaintext plain;
        encoder.encode(slots[i], client_context.context->last_parms_id(),
                       std::ldexp(1.0, 20), plain);
        encryptor.encrypt_symmetric(plain, ciphertexts[i]);
    }
    hera::TensorSpec spec;
    spec.shape = {{1, 2, 2}};
    spec.domain = hera::Domain::kHeCiphertext;
    spec.layout = hera::Layout::kAttentionDiagonal;
    spec.numeric = config.mpc_numeric;
    spec.he_region = hera::HeRegionId::kRegion2;
    spec.he_parameter_profile = "scores";
    spec.planned_level = 2;
    spec.planned_scale_bits = 20;
    return {"scores", std::move(ciphertexts), spec,
            packing.take_value()};
}

void RunExecutorParty(
    hera::PartyRole role, const hera::HeraConfig &config,
    hera::HeContextRegistry &registry,
    const std::optional<hera::SealCipherTensor> &server_scores,
    int control_port, int worker_port, int nonlinear_port,
    std::uint64_t *output_shares, std::string *error)
{
    try {
        auto transport = hera::MakeNetIoTransport(
            role, role == hera::PartyRole::kClient ? "127.0.0.1" : "",
            control_port);
        hera::CkksMpcConverter converter(
            role, {43, 13, 40, 20, true}, registry, transport,
            hera::MakeDeterministicRandomSourceForTesting(
                role == hera::PartyRole::kClient
                    ? std::vector<std::uint8_t>{1, 3, 5}
                    : std::vector<std::uint8_t>{2, 4, 6}),
            "127.0.0.1", worker_port, 1);
        hera::MpcSoftmaxRuntime nonlinear(
            role, "127.0.0.1", nonlinear_port);
        hera::SealCipherTensorStore he_store;
        hera::CryptoTensorStore mpc_store;
        constexpr hera::TensorId input_id = 7100000;
        if (role == hera::PartyRole::kServer) {
            Require(server_scores.has_value(), "server scores are absent");
            Require(he_store.Put(input_id, *server_scores).ok(),
                    "cannot register server score ciphertext");
        }
        const auto &layer = config.layers.front();
        hera::SealSciSoftmaxExecutor executor(
            config, layer, registry, he_store, mpc_store, converter,
            nonlinear);
        hera::PublicAttentionMask mask;
        mask.kind = hera::AttentionMaskKind::kNone;
        mask.shape = {{2, 2}};
        hera::SoftmaxExecutionReport report;
        auto result = executor.Evaluate(
            {{input_id, server_scores
                ? server_scores->logical_spec
                : hera::TensorSpec{{{1, 2, 2}},
                    hera::Domain::kHeCiphertext,
                    hera::Layout::kAttentionDiagonal,
                    config.mpc_numeric, hera::HeRegionId::kRegion2,
                    std::string("scores"), std::size_t{2}, 20}}, mask},
            &report);
        Require(result.ok(), result.status().message());
        Require(report.ckks_to_mpc_calls == 2 &&
                    report.mpc_to_ckks_calls == 1,
                "integrated Softmax operation report mismatch");
        if (role == hera::PartyRole::kServer)
            Require(report.ciphertext_multiplications ==
                        config.softmax.exp_squarings &&
                    report.rescales == 1 + config.softmax.exp_squarings,
                    "server Block-3 operation report mismatch");
        else
            Require(report.ciphertext_multiplications == 0 &&
                        report.rescales == 0,
                    "client unexpectedly executed Block-3 HE operations");
        auto shares = mpc_store.GetMpcShare(
            result.value().probability_shares.id);
        Require(shares.ok() && shares.value().local_share.size() == 4,
                "integrated Softmax output shares are absent");
        std::copy(shares.value().local_share.begin(),
                  shares.value().local_share.end(), output_shares);
    } catch (const std::exception &exception) {
        *error = exception.what();
    }
}

void TestIntegratedExecutor()
{
    const auto config = IntegrationConfig();
    Require(hera::ValidateHeraConfig(config).ok(),
            "Softmax integration config is invalid");
    hera::HeContextRegistry client_registry;
    Require(client_registry.CreateClientProfiles(config).ok(),
            "cannot create integration client profiles");
    auto public_bundle = client_registry.ExportSerializedPublicBundle();
    Require(public_bundle.ok(), public_bundle.status().message());
    hera::HeContextRegistry server_registry;
    Require(server_registry.CreateServerProfiles(
                config, public_bundle.value()).ok(),
            "cannot create integration server profiles");
    auto encrypted_scores = MakeEncryptedScores(
        config, client_registry.Require("scores", hera::PartyRole::kClient));

    struct Packet {
        int ok{};
        std::uint64_t shares[4]{};
    };
    int pipe_fd[2];
    Require(pipe(pipe_fd) == 0, "cannot create integration result pipe");
    const pid_t child = fork();
    Require(child >= 0, "cannot fork integrated Softmax client");
    if (child == 0) {
        close(pipe_fd[0]);
        Packet packet;
        std::string error;
        RunExecutorParty(
            hera::PartyRole::kClient, config, client_registry, std::nullopt,
            19920, 19930, 19940, packet.shares, &error);
        packet.ok = error.empty();
        if (!packet.ok)
            std::cerr << "integrated client: " << error << '\n';
        const auto written = write(pipe_fd[1], &packet, sizeof(packet));
        close(pipe_fd[1]);
        _exit(packet.ok && written == static_cast<ssize_t>(sizeof(packet))
                  ? 0 : 2);
    }
    close(pipe_fd[1]);
    std::uint64_t server_shares[4]{};
    std::string server_error;
    RunExecutorParty(
        hera::PartyRole::kServer, config, server_registry, encrypted_scores,
        19920, 19930, 19940, server_shares, &server_error);
    if (!server_error.empty()) {
        kill(child, SIGTERM);
        int failed_status = 0;
        waitpid(child, &failed_status, 0);
        close(pipe_fd[0]);
        throw std::runtime_error(server_error);
    }
    Packet packet;
    const auto received = read(pipe_fd[0], &packet, sizeof(packet));
    close(pipe_fd[0]);
    int status = 0;
    waitpid(child, &status, 0);
    Require(received == static_cast<ssize_t>(sizeof(packet)) && packet.ok &&
                WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "integrated Softmax client failed");
    const auto mask = (std::uint64_t{1} << 43) - 1;
    const std::vector<double> shifted{-1.0, 0.0, -2.0, 0.0};
    for (std::size_t row = 0; row < 2; ++row) {
        double expected_sum = 0.0;
        for (std::size_t column = 0; column < 2; ++column)
            expected_sum += std::pow(
                1.0 + shifted[row * 2 + column] / 64.0, 2.0);
        for (std::size_t column = 0; column < 2; ++column) {
            const auto index = row * 2 + column;
            const auto bits = (server_shares[index] +
                               packet.shares[index]) & mask;
            const double actual = std::ldexp(
                static_cast<double>(bits), -13);
            const double expected = std::pow(
                1.0 + shifted[index] / 64.0, 2.0) / expected_sum;
            Require(std::abs(actual - expected) < 0.03,
                    "integrated Softmax probability mismatch");
        }
    }
}

} // namespace

int main()
{
    try {
        auto config = hera::LoadHeraConfig(HERA_TEST_CONFIG_PATH);
        Require(config.ok(), config.status().message());
        if (std::getenv("HERA_SOFTMAX_MPC_ONLY") != nullptr) {
            TestMpc(config.value());
            std::cout << "HERA_SOFTMAX_MPC_STATUS=PASS\n";
            return EXIT_SUCCESS;
        }
        TestGraph(config.value());
        TestRowPacking(config.value());
        TestExponent(config.value());
        TestIntegratedExecutor();
        std::cout << "SOFTMAX_TO_MPC_READY=YES\n"
                  << "SOFTMAX_MPC_TO_HE_READY=NO\n"
                  << "SOFTMAX_AV_INTEGRATION_READY=NO\n"
                  << "HERA_SOFTMAX_STATUS=PASS\n";
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << "HERA_SOFTMAX_STATUS=FAIL\n"
                  << "HERA_SOFTMAX_ERROR=" << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
