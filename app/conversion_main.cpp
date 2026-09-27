#include "hera/conversion.h"
#include "hera/seal_wire.h"

#include <seal/seal.h>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cmath>
#include <algorithm>
#include <chrono>
#include <complex>
#include <filesystem>
#include <iostream>
#include <optional>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string party{"local"};
    std::string ip{"127.0.0.1"};
    std::filesystem::path config;
    std::string profile;
    int port{19100};
    std::size_t threads{1};
    std::string seal_compression{"none"};
#ifdef HERA_CONVERSION_BENCHMARK
    bool benchmark{true};
    std::size_t warmup{1};
#else
    bool benchmark{false};
    std::size_t warmup{0};
#endif
    std::size_t iterations{1};
};

Options Parse(int argc, char **argv)
{
    Options result;
    for (int i = 1; i < argc; ++i) {
        const std::string argument(argv[i]);
        const auto value = [&](const std::string &prefix)
            -> std::optional<std::string> {
            if (argument.rfind(prefix, 0) == 0)
                return argument.substr(prefix.size());
            return std::nullopt;
        };
        if (auto text = value("--party=")) result.party = *text;
        else if (auto text = value("--ip=")) result.ip = *text;
        else if (auto text = value("--config=")) result.config = *text;
        else if (auto text = value("--profile=")) result.profile = *text;
        else if (auto text = value("--port=")) result.port = std::stoi(*text);
        else if (auto text = value("--threads="))
            result.threads = static_cast<std::size_t>(std::stoul(*text));
        else if (auto text = value("--seal-compression="))
            result.seal_compression = *text;
        else if (auto text = value("--iterations="))
            result.iterations = std::stoull(*text);
        else if (auto text = value("--warmup="))
            result.warmup = std::stoull(*text);
        else throw std::invalid_argument("unknown option: " + argument);
    }
    if (result.config.empty())
        throw std::invalid_argument("--config is required");
    if (result.party != "local" && result.party != "client" &&
        result.party != "server")
        throw std::invalid_argument("--party must be local, client, or server");
    if (result.port < 1024 || result.port + 400 >= 32768)
        throw std::invalid_argument("conversion port range is invalid");
    if (result.threads < 1 || result.threads > 16)
        throw std::invalid_argument("--threads must be in [1,16]");
    if (result.iterations == 0 || result.iterations > 10000 ||
        result.warmup > 10000)
        throw std::invalid_argument("iterations/warmup count is invalid");
    if (result.seal_compression != "none" &&
        result.seal_compression != "zstd" &&
        result.seal_compression != "zlib")
        throw std::invalid_argument("unsupported SEAL compression option");
    return result;
}

void Check(const hera::Result<void> &result)
{
    if (!result.ok()) throw std::runtime_error(result.status().message());
}

seal::compr_mode_type CompressionMode(const Options &options)
{
    auto mode = seal::compr_mode_type::none;
    if (options.seal_compression == "zstd")
        mode = static_cast<seal::compr_mode_type>(2);
    else if (options.seal_compression == "zlib")
        mode = static_cast<seal::compr_mode_type>(1);
    if (!seal::Serialization::IsSupportedComprMode(mode))
        throw std::invalid_argument(
            "requested SEAL compression is unavailable in this build");
    return mode;
}

void SendString(hera::IProtocolTransport &transport, const std::string &value)
{
    const std::uint32_t size = static_cast<std::uint32_t>(value.size());
    transport.Send(&size, sizeof(size));
    if (size) transport.Send(value.data(), value.size());
}

std::string ReceiveString(hera::IProtocolTransport &transport)
{
    std::uint32_t size = 0;
    transport.Receive(&size, sizeof(size));
    if (size == 0 || size > 4096)
        throw std::runtime_error("invalid profile name on control channel");
    std::string result(size, '\0');
    transport.Receive(result.data(), result.size());
    return result;
}

void SendCiphertext(hera::IProtocolTransport &transport,
                    const seal::Ciphertext &ciphertext,
                    seal::compr_mode_type compression)
{
    hera::SendSealObject(transport, ciphertext, compression);
}

seal::Ciphertext ReceiveCiphertext(hera::IProtocolTransport &transport,
                                   const seal::SEALContext &context)
{
    return hera::ReceiveSealObject<seal::Ciphertext>(transport, context);
}

hera::PhysicalSlotMap DemoMap(std::size_t slots)
{
    hera::PhysicalSlotMap map;
    map.ciphertext_count = 1;
    map.slots_per_ciphertext = slots;
    map.logical_element_count = 8;
    map.assignments.resize(slots);
    for (std::size_t i = 0; i < 4; ++i) {
        map.assignments[i].real_logical_index = 2 * i;
        map.assignments[i].imaginary_logical_index = 2 * i + 1;
    }
    return map;
}

std::string SelectedProfile(const Options &options,
                            const hera::HeraConfig &config)
{
    const std::string profile = options.profile.empty()
        ? config.he_parameter_profiles.begin()->first : options.profile;
    if (config.he_parameter_profiles.count(profile) == 0)
        throw std::invalid_argument("unknown --profile: " + profile);
    return profile;
}

hera::ConversionConfig ProtocolConfig(const hera::HeraConfig &config)
{
    return {config.mpc_numeric.ring_bits, config.mpc_numeric.scale_bits,
            config.conversion.statistical_security_bits,
            config.conversion.fft_scale_bits,
            config.conversion.allow_internal_mod_switch};
}

using BenchmarkClock = std::chrono::steady_clock;

struct ConversionBenchmarkTimes {
    std::vector<double> ckks_to_mpc;
    std::vector<double> mpc_to_ckks;
    std::uint64_t ckks_to_mpc_bytes{};
    std::uint64_t mpc_to_ckks_bytes{};
    std::uint64_t ckks_to_mpc_rounds{};
    std::uint64_t mpc_to_ckks_rounds{};
};

void PrintBenchmark(const Options &options, const std::string &party,
                    const ConversionBenchmarkTimes &times)
{
    if (!options.benchmark) return;
    const auto mean = [](const std::vector<double> &values) {
        return std::accumulate(values.begin(), values.end(), 0.0) /
               values.size();
    };
    std::cout << "HERA_CONVERSION_BENCHMARK_PARTY=" << party << '\n'
              << "BOUNDARY=established_two_party_session_to_conversion_output\n"
              << "WARMUP=" << options.warmup << '\n'
              << "ITERATIONS=" << options.iterations << '\n'
              << "CKKS_TO_MPC_MEAN_MS=" << mean(times.ckks_to_mpc) << '\n'
              << "MPC_TO_CKKS_MEAN_MS=" << mean(times.mpc_to_ckks) << '\n'
              << "CKKS_TO_MPC_COMMUNICATION_BYTES="
              << times.ckks_to_mpc_bytes << '\n'
              << "MPC_TO_CKKS_COMMUNICATION_BYTES="
              << times.mpc_to_ckks_bytes << '\n'
              << "CKKS_TO_MPC_ROUNDS=" << times.ckks_to_mpc_rounds << '\n'
              << "MPC_TO_CKKS_ROUNDS=" << times.mpc_to_ckks_rounds << '\n'
              << std::flush;
}

int RunClient(const Options &options, const hera::HeraConfig &config)
{
    hera::HeContextRegistry contexts;
    Check(contexts.CreateClientProfiles(config));
    auto serialized = contexts.ExportSerializedPublicBundle(
        CompressionMode(options));
    if (!serialized.ok()) throw std::runtime_error(serialized.status().message());
    auto control = hera::MakeNetIoTransport(hera::PartyRole::kClient,
                                            options.ip, options.port);
    Check(hera::SendPublicContextBundle(*control, serialized.value()));

    const std::string profile_name = SelectedProfile(options, config);
    const auto &profile = contexts.Require(profile_name,
                                           hera::PartyRole::kClient);
    std::vector<std::complex<double>> clear(profile.slot_count());
    clear[0] = {0.125, -0.25};
    clear[1] = {-1.5, 0.75};
    clear[2] = {2.0, -1.0};
    clear[3] = {-0.0625, 0.03125};
    seal::CKKSEncoder encoder(*profile.context);
    seal::Plaintext plain;
    encoder.encode(clear, std::ldexp(1.0, profile.parameters.output_scale_bits),
                   plain);
    seal::Encryptor encryptor(*profile.context, profile.secret_key.value());
    seal::Ciphertext input;
    encryptor.encrypt_symmetric(plain, input);
    SendString(*control, profile_name);
    SendCiphertext(*control, input, CompressionMode(options));

    hera::CkksMpcConverter converter(
        hera::PartyRole::kClient, ProtocolConfig(config), contexts, control,
        hera::MakeProductionRandomSource(), options.ip, options.port + 200,
        options.threads, CompressionMode(options));
    const auto map = DemoMap(profile.slot_count());
    hera::MpcToCkksTarget target{
        profile_name, 0, profile.parameters.input_scale_bits,
        hera::TensorShape{{8}}, hera::Layout::kSpatial, map};
    ConversionBenchmarkTimes times;
    for (std::size_t iteration = 0;
         iteration < options.warmup + options.iterations; ++iteration) {
        const auto before_c2m = converter.stats();
        const auto c2m_start = BenchmarkClock::now();
        auto shares = converter.CkksToMpc(std::nullopt);
        const auto c2m_stop = BenchmarkClock::now();
        if (!shares.ok()) throw std::runtime_error(shares.status().message());
        const auto after_c2m = converter.stats();
        const auto m2c_start = BenchmarkClock::now();
        auto converted = converter.MpcToCkks(shares.value(), target);
        const auto m2c_stop = BenchmarkClock::now();
        if (!converted.ok())
            throw std::runtime_error(converted.status().message());
        if (converted.value())
            throw std::runtime_error("client received server-only ciphertext");
        const auto after_m2c = converter.stats();
        if (iteration >= options.warmup) {
            times.ckks_to_mpc.push_back(
                std::chrono::duration<double, std::milli>(
                    c2m_stop - c2m_start).count());
            times.mpc_to_ckks.push_back(
                std::chrono::duration<double, std::milli>(
                    m2c_stop - m2c_start).count());
            times.ckks_to_mpc_bytes += after_c2m.communication_bytes -
                                       before_c2m.communication_bytes;
            times.mpc_to_ckks_bytes += after_m2c.communication_bytes -
                                       after_c2m.communication_bytes;
            times.ckks_to_mpc_rounds += after_c2m.rounds - before_c2m.rounds;
            times.mpc_to_ckks_rounds += after_m2c.rounds - after_c2m.rounds;
        }
    }
    auto result = ReceiveCiphertext(*control, *profile.context);
    seal::Decryptor decryptor(*profile.context, profile.secret_key.value());
    seal::Plaintext result_plain;
    decryptor.decrypt(result, result_plain);
    std::vector<std::complex<double>> decoded;
    encoder.decode(result_plain, decoded);
    double maximum_error = 0;
    for (std::size_t i = 0; i < 4; ++i) {
        maximum_error = std::max(maximum_error,
            std::abs(decoded[i].real() - clear[i].real()));
        maximum_error = std::max(maximum_error,
            std::abs(decoded[i].imag() - clear[i].imag()));
    }
    const auto stats = converter.stats();
    if (maximum_error > 48.0 / 8192.0)
        throw std::runtime_error(
            "conversion round-trip exceeds current 48-ULP smoke bound");
    std::cout << "party=client profile=" << profile_name
              << " max_error=" << maximum_error
              << " communication_bytes=" << stats.communication_bytes
              << " rounds=" << stats.rounds
              << " transform_cache_hits=" << stats.transform_cache_hits
              << " transform_cache_misses=" << stats.transform_cache_misses
              << " local_peak_inflight="
              << stats.local_transform_peak_inflight
              << '\n' << std::flush;
    PrintBenchmark(options, "client", times);
    return 0;
}

int RunServer(const Options &options, const hera::HeraConfig &config)
{
    auto control = hera::MakeNetIoTransport(hera::PartyRole::kServer, "",
                                            options.port);
    auto bundle = hera::ReceivePublicContextBundle(*control);
    if (!bundle.ok()) throw std::runtime_error(bundle.status().message());
    hera::HeContextRegistry contexts;
    Check(contexts.CreateServerProfiles(config, bundle.value()));
    const std::string profile_name = ReceiveString(*control);
    if (profile_name != SelectedProfile(options, config))
        throw std::runtime_error("client/server selected profiles differ");
    const auto &profile = contexts.Require(profile_name,
                                           hera::PartyRole::kServer);
    auto input = ReceiveCiphertext(*control, *profile.context);
    const auto map = DemoMap(profile.slot_count());
    hera::SealCipherBatch batch{
        profile_name, {std::move(input)}, hera::TensorShape{{8}},
        hera::Layout::kSpatial, map};
    hera::CkksMpcConverter converter(
        hera::PartyRole::kServer, ProtocolConfig(config), contexts, control,
        hera::MakeProductionRandomSource(), "", options.port + 200,
        options.threads, CompressionMode(options));
    hera::MpcToCkksTarget target{
        profile_name, 0, profile.parameters.input_scale_bits,
        hera::TensorShape{{8}}, hera::Layout::kSpatial, map};
    ConversionBenchmarkTimes times;
    std::optional<hera::SealCipherBatch> output;
    for (std::size_t iteration = 0;
         iteration < options.warmup + options.iterations; ++iteration) {
        const auto before_c2m = converter.stats();
        const auto c2m_start = BenchmarkClock::now();
        auto shares = converter.CkksToMpc(batch);
        const auto c2m_stop = BenchmarkClock::now();
        if (!shares.ok()) throw std::runtime_error(shares.status().message());
        const auto after_c2m = converter.stats();
        const auto m2c_start = BenchmarkClock::now();
        auto converted = converter.MpcToCkks(shares.value(), target);
        const auto m2c_stop = BenchmarkClock::now();
        if (!converted.ok() || !converted.value())
            throw std::runtime_error(converted.ok()
                ? "server did not receive output ciphertext"
                : converted.status().message());
        output = std::move(*converted.value());
        const auto after_m2c = converter.stats();
        if (iteration >= options.warmup) {
            times.ckks_to_mpc.push_back(
                std::chrono::duration<double, std::milli>(
                    c2m_stop - c2m_start).count());
            times.mpc_to_ckks.push_back(
                std::chrono::duration<double, std::milli>(
                    m2c_stop - m2c_start).count());
            times.ckks_to_mpc_bytes += after_c2m.communication_bytes -
                                       before_c2m.communication_bytes;
            times.mpc_to_ckks_bytes += after_m2c.communication_bytes -
                                       after_c2m.communication_bytes;
            times.ckks_to_mpc_rounds += after_c2m.rounds - before_c2m.rounds;
            times.mpc_to_ckks_rounds += after_m2c.rounds - after_c2m.rounds;
        }
    }
    SendCiphertext(*control, output->ciphertexts.front(),
                   CompressionMode(options));
    const auto stats = converter.stats();
    std::cout << "party=server profile=" << profile_name
              << " communication_bytes=" << stats.communication_bytes
              << " rounds=" << stats.rounds
              << " a_mask_bytes=" << stats.a_mask_bytes
              << " z_mask_bytes=" << stats.z_mask_bytes
              << " internal_mod_switches=" << stats.internal_mod_switches
              << " transform_cache_hits=" << stats.transform_cache_hits
              << " transform_cache_misses=" << stats.transform_cache_misses
              << " local_peak_inflight="
              << stats.local_transform_peak_inflight
              << '\n';
    PrintBenchmark(options, "server", times);
    return 0;
}

} // namespace

int main(int argc, char **argv)
{
    try {
        const Options options = Parse(argc, argv);
        auto config_result = hera::LoadHeraConfig(options.config);
        if (!config_result.ok())
            throw std::runtime_error(config_result.status().message());
        const auto config = config_result.take_value();
        if (options.party == "client") return RunClient(options, config);
        if (options.party == "server") return RunServer(options, config);
        const pid_t child = fork();
        if (child < 0) throw std::runtime_error("fork failed");
        if (child == 0) {
            try { _exit(RunClient(options, config)); }
            catch (const std::exception &error) {
                std::cerr << "client: " << error.what() << '\n';
                _exit(2);
            }
        }
        const int server_result = RunServer(options, config);
        int child_status = 0;
        waitpid(child, &child_status, 0);
        if (!WIFEXITED(child_status) || WEXITSTATUS(child_status) != 0)
            throw std::runtime_error("local client failed");
        std::cout << "HERA_COMPLEX_CONVERSION_STATUS=PASS\n";
        if (options.benchmark)
            std::cout << "HERA_BENCHMARK_STATUS=PASS\n";
        return server_result;
    } catch (const std::exception &error) {
        std::cerr << "HERA_COMPLEX_CONVERSION_STATUS=FAIL: " << error.what()
                  << '\n';
        return 1;
    }
}
