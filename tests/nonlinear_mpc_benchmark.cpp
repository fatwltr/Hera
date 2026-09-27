#include "hera/ffn_gelu.h"

#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
constexpr std::uint64_t kMask = (std::uint64_t{1} << 43) - 1;

struct Options {
    std::string kernel;
    std::size_t tokens{2};
    std::size_t heads{1};
    std::size_t width{8};
    std::size_t warmup{1};
    std::size_t iterations{1};
    int port{24900};
};

struct Metrics {
    double mean_ms{};
    double min_ms{};
    std::uint64_t communication_bytes{};
    std::uint64_t rounds{};
};

void Require(bool ok, const std::string &message)
{
    if (!ok) throw std::runtime_error(message);
}

Options Parse(int argc, char **argv)
{
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        const auto split = arg.find('=');
        Require(split != std::string::npos, "expected --name=value: " + arg);
        const auto name = arg.substr(0, split);
        const auto value = arg.substr(split + 1);
        if (name == "--kernel") { options.kernel = value; continue; }
        const auto number = std::stoull(value);
        if (name == "--tokens") options.tokens = number;
        else if (name == "--heads") options.heads = number;
        else if (name == "--width") options.width = number;
        else if (name == "--warmup") options.warmup = number;
        else if (name == "--iterations") options.iterations = number;
        else if (name == "--port") options.port = static_cast<int>(number);
        else throw std::invalid_argument("unknown option: " + name);
    }
    Require(options.kernel == "softmax-max" ||
                options.kernel == "softmax-normalize" ||
                options.kernel == "gelu-select", "unsupported --kernel");
    Require(options.tokens && options.heads && options.width &&
                options.iterations, "dimensions and iterations must be positive");
    Require(options.port >= 1024 && options.port < 32700,
            "port must be in [1024,32700)");
    return options;
}

std::uint64_t Encode(double value)
{
    const auto scaled = static_cast<std::int64_t>(
        std::llround(std::ldexp(value, 13)));
    return static_cast<std::uint64_t>(scaled) & kMask;
}

double Decode(std::uint64_t bits)
{
    bits &= kMask;
    const auto signed_value = bits >= (std::uint64_t{1} << 42)
        ? static_cast<std::int64_t>(bits) -
              static_cast<std::int64_t>(std::uint64_t{1} << 43)
        : static_cast<std::int64_t>(bits);
    return std::ldexp(static_cast<double>(signed_value), -13);
}

std::vector<double> Input(const Options &o)
{
    const auto size = o.kernel == "gelu-select"
        ? o.tokens * o.width : o.tokens * o.heads * o.tokens;
    std::vector<double> values(size);
    if (o.kernel == "gelu-select") {
        for (std::size_t i = 0; i < size; ++i)
            values[i] = static_cast<double>(static_cast<int>(i % 9) - 4);
    } else {
        for (std::size_t i = 0; i < size; ++i) {
            const auto column = i % o.tokens;
            const auto row = i / o.tokens;
            const double baseline = o.kernel == "softmax-max"
                ? 0.25 * static_cast<double>(row % 3) : 0.0;
            values[i] = baseline - 0.125 * static_cast<double>(column);
        }
    }
    return values;
}

std::vector<double> Expected(const Options &o,
                             const std::vector<double> &input)
{
    std::vector<double> expected(input.size());
    if (o.kernel == "softmax-max") {
        for (std::size_t i = 0; i < input.size(); ++i)
            expected[i] = -0.125 * static_cast<double>(i % o.tokens);
    } else if (o.kernel == "softmax-normalize") {
        const auto exponent = [&](double x) {
            return std::pow(1.0 + x / 64.0, 64.0);
        };
        for (std::size_t row = 0; row < o.tokens * o.heads; ++row) {
            double sum = 0;
            for (std::size_t col = 0; col < o.tokens; ++col)
                sum += exponent(input[row * o.tokens + col]);
            for (std::size_t col = 0; col < o.tokens; ++col)
                expected[row * o.tokens + col] =
                    exponent(input[row * o.tokens + col]) / sum;
        }
    } else {
        const hera::GeluConfig c;
        for (std::size_t i = 0; i < input.size(); ++i) {
            const auto x = input[i];
            const auto common = c.a * std::pow(x, 4) + c.c * x * x + c.e;
            const auto f0 = common - c.b * std::pow(x, 3) + (0.5 - c.d) * x;
            const auto f1 = common + c.b * std::pow(x, 3) + (0.5 + c.d) * x;
            expected[i] = x < -c.threshold ? 0.0
                : x < 0.0 ? f0 : x <= c.threshold ? f1 : x;
        }
    }
    return expected;
}

std::vector<std::uint64_t> Split(const std::vector<double> &values,
                                 bool server, std::uint64_t salt)
{
    std::vector<std::uint64_t> shares(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        const auto first = (salt + i * 7919) & kMask;
        shares[i] = server ? first : (Encode(values[i]) - first) & kMask;
    }
    return shares;
}

void WriteAll(int fd, const void *data, std::size_t count)
{
    const auto *bytes = static_cast<const char *>(data);
    while (count) {
        const auto written = write(fd, bytes, count);
        Require(written > 0, "cannot write client benchmark result");
        bytes += written;
        count -= static_cast<std::size_t>(written);
    }
}

void ReadAll(int fd, void *data, std::size_t count)
{
    auto *bytes = static_cast<char *>(data);
    while (count) {
        const auto received = read(fd, bytes, count);
        Require(received > 0, "cannot read client benchmark result");
        bytes += received;
        count -= static_cast<std::size_t>(received);
    }
}

struct PartyOutput {
    Metrics metrics;
    std::vector<std::uint64_t> shares;
};

PartyOutput RunParty(const Options &o, hera::PartyRole role)
{
    const bool server = role == hera::PartyRole::kServer;
    const auto values = Input(o);
    const auto shape = o.kernel == "gelu-select"
        ? hera::TensorShape{{o.tokens, o.width}}
        : hera::TensorShape{{o.tokens, o.heads, o.tokens}};
    auto make_share = [&](const std::vector<double> &clear,
                          std::uint64_t salt, bool signed_values) {
        return hera::MpcShareTensor{
            Split(clear, server, salt), shape,
            {43, 13, signed_values}, hera::ComplexComponent::kReal};
    };
    auto x = make_share(values, 11939, true);
    const auto count = values.size();
    std::vector<double> f0(count), f1(count), exponentials(count);
    const hera::GeluConfig gelu;
    for (std::size_t i = 0; i < count; ++i) {
        const auto v = values[i];
        const auto common = gelu.a * std::pow(v, 4) +
                            gelu.c * v * v + gelu.e;
        f0[i] = common - gelu.b * std::pow(v, 3) + (0.5 - gelu.d) * v;
        f1[i] = common + gelu.b * std::pow(v, 3) + (0.5 + gelu.d) * v;
        exponentials[i] = std::pow(1.0 + v / 64.0, 64.0);
    }
    auto branch0 = make_share(f0, 20431, true);
    auto branch1 = make_share(f1, 31033, true);
    auto exp = make_share(exponentials, 40739, false);
    hera::TransformerLayerConfig layer;
    layer.tokens = o.tokens;
    layer.q_heads = o.heads;
    layer.kv_heads = o.heads;
    layer.qk_head_width = o.width;
    layer.value_head_width = o.width;
    layer.attention_mask = hera::AttentionMaskKind::kNone;
    hera::PublicAttentionMask mask;
    mask.kind = hera::AttentionMaskKind::kNone;
    mask.shape = {{o.tokens, o.tokens}};
    hera::SoftmaxConfig softmax;
    std::vector<double> timings;
    std::vector<std::uint64_t> output;
    std::uint64_t bytes = 0, rounds = 0;
    if (o.kernel == "gelu-select") {
        hera::MpcGeluRuntime runtime(role, "127.0.0.1", o.port);
        for (std::size_t i = 0; i < o.warmup + o.iterations; ++i) {
            hera::FfnGeluExecutionReport report;
            const auto start = Clock::now();
            auto result = runtime.Evaluate(x, branch0, branch1,
                                           gelu.threshold, &report);
            const auto stop = Clock::now();
            Require(result.ok(), result.status().message());
            if (i >= o.warmup) {
                timings.push_back(std::chrono::duration<double, std::milli>(
                    stop - start).count());
                bytes += report.mpc_communication_bytes;
                rounds += report.mpc_rounds;
            }
            output = result.value().local_share;
        }
    } else {
        hera::MpcSoftmaxRuntime runtime(role, "127.0.0.1", o.port);
        for (std::size_t i = 0; i < o.warmup + o.iterations; ++i) {
            hera::SoftmaxExecutionReport report;
            const auto start = Clock::now();
            auto result = o.kernel == "softmax-max"
                ? runtime.CenterAndMask(x, layer, mask, softmax, &report)
                : runtime.ClipAndNormalize(x, exp, layer, mask,
                                           softmax, &report);
            const auto stop = Clock::now();
            Require(result.ok(), result.status().message());
            if (i >= o.warmup) {
                timings.push_back(std::chrono::duration<double, std::milli>(
                    stop - start).count());
                bytes += report.mpc_communication_bytes;
                rounds += report.mpc_rounds;
            }
            output = result.value().local_share;
        }
    }
    PartyOutput result;
    result.metrics.mean_ms = std::accumulate(
        timings.begin(), timings.end(), 0.0) / timings.size();
    result.metrics.min_ms = *std::min_element(timings.begin(), timings.end());
    result.metrics.communication_bytes = bytes;
    result.metrics.rounds = rounds;
    result.shares = std::move(output);
    return result;
}

} // namespace

int main(int argc, char **argv)
{
    try {
        const auto options = Parse(argc, argv);
        int fds[2];
        Require(pipe(fds) == 0, "cannot create benchmark pipe");
        const auto child = fork();
        Require(child >= 0, "cannot fork client");
        if (child == 0) {
            close(fds[0]);
            try {
                const auto result = RunParty(options, hera::PartyRole::kClient);
                WriteAll(fds[1], &result.metrics, sizeof(result.metrics));
                WriteAll(fds[1], result.shares.data(),
                         result.shares.size() * sizeof(std::uint64_t));
                close(fds[1]);
                _exit(0);
            } catch (const std::exception &error) {
                std::cerr << "client: " << error.what() << '\n';
                _exit(2);
            }
        }
        close(fds[1]);
        const auto server = RunParty(options, hera::PartyRole::kServer);
        Metrics client_metrics;
        std::vector<std::uint64_t> client_shares(server.shares.size());
        ReadAll(fds[0], &client_metrics, sizeof(client_metrics));
        ReadAll(fds[0], client_shares.data(),
                client_shares.size() * sizeof(std::uint64_t));
        close(fds[0]);
        int status = 0;
        waitpid(child, &status, 0);
        Require(WIFEXITED(status) && WEXITSTATUS(status) == 0,
                "MPC client failed");
        const auto expected = Expected(options, Input(options));
        double error = 0;
        for (std::size_t i = 0; i < expected.size(); ++i) {
            const auto actual = Decode(server.shares[i] + client_shares[i]);
            error = std::max(error, std::abs(actual - expected[i]));
        }
        Require(error < 0.03, "MPC numerical check failed: " +
                               std::to_string(error));
        std::cout << std::fixed << std::setprecision(3)
                  << "HERA_BENCHMARK=" << options.kernel << '\n'
                  << "BOUNDARY=two_party_shares_to_two_party_shares\n"
                  << "ELEMENTS=" << expected.size() << '\n'
                  << "WARMUP=" << options.warmup << '\n'
                  << "ITERATIONS=" << options.iterations << '\n'
                  << "SERVER_MEAN_MS=" << server.metrics.mean_ms << '\n'
                  << "CLIENT_MEAN_MS=" << client_metrics.mean_ms << '\n'
                  << "ONLINE_WALL_MEAN_MS=" << std::max(
                         server.metrics.mean_ms, client_metrics.mean_ms) << '\n'
                  << "SERVER_COMMUNICATION_BYTES="
                  << server.metrics.communication_bytes << '\n'
                  << "CLIENT_COMMUNICATION_BYTES="
                  << client_metrics.communication_bytes << '\n'
                  << "SERVER_ROUNDS=" << server.metrics.rounds << '\n'
                  << "CLIENT_ROUNDS=" << client_metrics.rounds << '\n'
                  << "MAX_ABS_ERROR=" << error << '\n'
                  << "HERA_BENCHMARK_STATUS=PASS\n";
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << "HERA_BENCHMARK_STATUS=FAIL " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
