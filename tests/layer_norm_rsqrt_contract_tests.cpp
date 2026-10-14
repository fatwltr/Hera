#include "hera/attention_ln2.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <sys/wait.h>
#include <signal.h>
#include <unistd.h>

namespace {
void Check(bool ok, const std::string &message)
{
    if (!ok) throw std::runtime_error(message);
}

constexpr std::array<int, 3> scales{13, 20, 28};
constexpr std::array<int, 3> widths{20, 27, 40};
// Entry zero denotes one input quantum at the selected variance precision.
constexpr std::array<double, 7> values{0.0, 1e-5, 1e-4, 0.25, 1.0, 16.0, 63.0};
constexpr auto mask = (std::uint64_t{1} << 63) - 1;
struct Packet { std::uint64_t shares[3][7]{}; };

void Transfer(int fd, Packet &packet, bool sending)
{
    auto *data = reinterpret_cast<char *>(&packet);
    std::size_t left = sizeof(packet);
    while (left) {
        const auto n = sending ? write(fd, data, left) : read(fd, data, left);
        Check(n > 0, "rsqrt contract test pipe failed");
        data += n;
        left -= n;
    }
}

Packet Run(hera::PartyRole role)
{
    hera::MpcLayerNormRuntime runtime(role, "127.0.0.1", 26580);
    Packet packet;
    for (std::size_t variant = 0; variant < scales.size(); ++variant) {
        const int variance_scale = std::max(24, scales[variant]);
        std::vector<std::uint64_t> shares(values.size());
        for (std::size_t i = 0; i < values.size(); ++i) {
            const auto encoded = i == 0 ? std::uint64_t{1}
                : static_cast<std::uint64_t>(std::llround(std::ldexp(values[i], variance_scale)));
            // Reconstructed positive values use genuinely wrapped random shares.
            const auto server = (0x7382b81314b6fc1dULL + i * 0x100000001b3ULL) & mask;
            shares[i] = role == hera::PartyRole::kServer ? server : (encoded - server) & mask;
        }
        hera::MpcShareTensor variance{std::move(shares), {{values.size(), 1}},
            {63, variance_scale, true}, hera::ComplexComponent::kReal};
        hera::AttentionLn2ExecutionReport report;
        auto output = runtime.ReciprocalSqrt(variance, widths[variant], &report, scales[variant]);
        Check(output.ok(), output.status().message());
        Check(output.value().numeric.ring_bits == 63 &&
            output.value().numeric.scale_bits == scales[variant] && output.value().numeric.is_signed,
            "rsqrt changed its public output format");
        Check(report.rsqrt_input_ring_bits == (variant == 2 ? 40 : 32) &&
            report.rsqrt_output_ring_bits == (variant == 0 ? 27 : variant == 1 ? 34 : 44),
            "rsqrt did not use the checked native plan");
        std::copy(output.value().local_share.begin(), output.value().local_share.end(), packet.shares[variant]);
    }
    return packet;
}
}

int main()
{
    pid_t child = -1;
    try {
        // No peer listens here: invalid contracts must fail before acquiring
        // a SCI session or entering any protocol, including direct API calls.
        hera::MpcLayerNormRuntime unsupported(hera::PartyRole::kServer, "", 26590);
        for (int scale : {29, 30, 31}) {
            hera::MpcShareTensor input{{std::uint64_t{1} << scale}, {{1, 1}},
                {63, scale, true}, hera::ComplexComponent::kReal};
            auto result = unsupported.ReciprocalSqrt(input, 40, nullptr, scale);
            Check(!result.ok() && result.status().message().find("LUT capacity") != std::string::npos,
                "direct rsqrt API accepted an unsupported LUT");
        }
        hera::MpcSoftmaxRuntime softmax(hera::PartyRole::kServer, "", 26590);
        hera::TransformerLayerConfig layer;
        layer.tokens = 2;
        layer.q_heads = 1;
        layer.attention_mask = hera::AttentionMaskKind::kNone;
        hera::MpcShareTensor shifted{{0, 0, 0, 0}, {{2, 1, 2}},
            {63, 31, true}, hera::ComplexComponent::kReal};
        auto exponent = shifted;
        exponent.numeric.is_signed = false;
        auto normalization = softmax.ClipAndNormalize(shifted, exponent, layer, {}, {});
        Check(!normalization.ok() && normalization.status().message().find("workspace") != std::string::npos,
            "direct Softmax API did not reject its oversized workspace before acquiring SCI");
        int fd[2];
        Check(pipe(fd) == 0, "cannot create rsqrt pipe");
        child = fork();
        Check(child >= 0, "cannot fork rsqrt client");
        if (child == 0) {
            close(fd[0]);
            try {
                auto packet = Run(hera::PartyRole::kClient);
                Transfer(fd[1], packet, true);
                _exit(0);
            } catch (const std::exception &e) {
                std::cerr << "rsqrt client: " << e.what() << '\n';
                _exit(2);
            }
        }
        close(fd[1]);
        auto server = Run(hera::PartyRole::kServer);
        Packet client;
        Transfer(fd[0], client, false);
        close(fd[0]);
        int status{};
        waitpid(child, &status, 0);
        child = -1;
        Check(WIFEXITED(status) && WEXITSTATUS(status) == 0, "rsqrt client failed");
        for (std::size_t v = 0; v < scales.size(); ++v) {
            const int variance_scale = std::max(24, scales[v]);
            double error = 0.0;
            for (std::size_t i = 0; i < values.size(); ++i) {
                const auto encoded = i == 0 ? 1LL : std::llround(std::ldexp(values[i], variance_scale));
                const double expected = 1.0 / std::sqrt(std::ldexp(static_cast<double>(encoded), -variance_scale));
                const double actual = std::ldexp(static_cast<double>((server.shares[v][i] + client.shares[v][i]) & mask), -scales[v]);
                error = std::max(error, std::abs(actual / expected - 1.0));
            }
            Check(error < 0.01, "rsqrt differs from the independent quantized-variance oracle");
            std::cout << "RSQRT_OUTPUT_SCALE=" << scales[v] << ",MAX_RELATIVE_ERROR=" << error << '\n';
        }
        std::cout << "RSQRT_CONTRACT_STATUS=PASS\n";
        return 0;
    } catch (const std::exception &e) {
        if (child > 0) { kill(child, SIGTERM); waitpid(child, nullptr, 0); }
        std::cerr << "RSQRT_CONTRACT_STATUS=FAIL: " << e.what() << '\n';
        return 1;
    }
}
