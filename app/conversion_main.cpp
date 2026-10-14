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
#include <iomanip>
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
    std::string seal_compression{"zstd"};
#ifdef HERA_CONVERSION_BENCHMARK
    bool benchmark{true};
    std::size_t warmup{1};
#else
    bool benchmark{false};
    std::size_t warmup{0};
#endif
    std::size_t iterations{1};
    std::size_t ciphertexts{1};
    std::size_t batch_elements{};
    bool dense{};
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
        else if (auto text = value("--ciphertexts=")) result.ciphertexts = std::stoull(*text);
        else if (auto text = value("--batch-elements=")) result.batch_elements = std::stoull(*text);
        else if (argument == "--dense") result.dense = true;
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
    if (!result.ciphertexts || result.ciphertexts > 256 || result.batch_elements > (1U << 20))
        throw std::invalid_argument("invalid ciphertext count or coefficient batch budget");
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

hera::PhysicalSlotMap DemoMap(std::size_t slots, const Options &options)
{
    if (options.dense)
        return hera::PhysicalSlotMap::DenseComplex(2 * slots * options.ciphertexts, slots);
    hera::PhysicalSlotMap map;
    map.ciphertext_count = options.ciphertexts;
    map.slots_per_ciphertext = slots;
    map.logical_element_count = 8 * options.ciphertexts;
    map.assignments.resize(slots * options.ciphertexts);
    for (std::size_t ct = 0; ct < options.ciphertexts; ++ct)
        for (std::size_t i = 0; i < 4; ++i) {
            map.assignments[ct * slots + i].real_logical_index = 8 * ct + 2 * i;
            map.assignments[ct * slots + i].imaginary_logical_index = 8 * ct + 2 * i + 1;
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

hera::ConversionConfig ProtocolConfig(const hera::HeraConfig &config, const Options &options)
{
    return {config.mpc_numeric.ring_bits, config.mpc_numeric.scale_bits,
            config.conversion.statistical_security_bits,
            config.conversion.fft_scale_bits,
            config.conversion.allow_internal_mod_switch, options.batch_elements};
}

using BenchmarkClock = std::chrono::steady_clock;

struct ConversionBenchmarkTimes {
    std::vector<double> ckks_to_mpc;
    std::vector<double> mpc_to_ckks;
    std::uint64_t ckks_to_mpc_bytes{};
    std::uint64_t mpc_to_ckks_bytes{};
    std::uint64_t ckks_to_mpc_rounds{};
    std::uint64_t mpc_to_ckks_rounds{};
    hera::ConversionStageTimes ckks_to_mpc_stages;
    hera::ConversionStageTimes mpc_to_ckks_stages;
};

using StageMember = double hera::ConversionStageTimes::*;
const std::pair<const char *, StageMember> kStageMembers[]{
    {"PREPARATION", &hera::ConversionStageTimes::preparation},
    {"HANDSHAKE", &hera::ConversionStageTimes::handshake},
    {"SAMPLING", &hera::ConversionStageTimes::sampling},
    {"MASK_ENCODE", &hera::ConversionStageTimes::mask_encode},
    {"FIELD_TO_RING", &hera::ConversionStageTimes::field_to_ring},
    {"SIGNED_EXTEND", &hera::ConversionStageTimes::signed_extend},
    {"SLOT_ENCODE", &hera::ConversionStageTimes::slot_encode},
    {"SLOT_DECODE", &hera::ConversionStageTimes::slot_decode},
    {"RNS", &hera::ConversionStageTimes::rns},
    {"HE", &hera::ConversionStageTimes::he},
    {"CIPHERTEXT_TRANSPORT", &hera::ConversionStageTimes::ciphertext_transport},
};

void AccumulateStages(hera::ConversionStageTimes &total,
                      const hera::ConversionStageTimes &before,
                      const hera::ConversionStageTimes &after)
{
    for (const auto &[name, member] : kStageMembers)
        total.*member += after.*member - before.*member;
}

void PrintBenchmark(const Options &options, const std::string &party,
                    const ConversionBenchmarkTimes &times)
{
    if (!options.benchmark) return;
    const auto mean = [](const std::vector<double> &values) {
        return std::accumulate(values.begin(), values.end(), 0.0) /
               values.size();
    };
    std::cout << std::setprecision(9)
              << "HERA_CONVERSION_BENCHMARK_PARTY=" << party << '\n'
              << "BOUNDARY=established_two_party_session_to_conversion_output\n"
              << "WARMUP=" << options.warmup << '\n'
              << "ITERATIONS=" << options.iterations << '\n'
              << "CIPHERTEXTS=" << options.ciphertexts << '\n'
              << "CONVERSION_WORKERS=" << options.threads << '\n'
              << "BATCH_ELEMENTS=" << (options.batch_elements ? options.batch_elements : hera::ConversionConfig::DefaultBatchElements(options.threads)) << '\n'
              << "CKKS_TO_MPC_MEAN_MS=" << mean(times.ckks_to_mpc) << '\n'
              << "MPC_TO_CKKS_MEAN_MS=" << mean(times.mpc_to_ckks) << '\n'
              << "CKKS_TO_MPC_COMMUNICATION_BYTES="
              << times.ckks_to_mpc_bytes << '\n'
              << "MPC_TO_CKKS_COMMUNICATION_BYTES="
              << times.mpc_to_ckks_bytes << '\n'
              << "CKKS_TO_MPC_ROUNDS=" << times.ckks_to_mpc_rounds << '\n'
              << "MPC_TO_CKKS_ROUNDS=" << times.mpc_to_ckks_rounds << '\n';
    for (const auto &[name, member] : kStageMembers) {
        std::cout << "CKKS_TO_MPC_STAGE_" << name << "_MEAN_MS="
                  << times.ckks_to_mpc_stages.*member / options.iterations << '\n'
                  << "MPC_TO_CKKS_STAGE_" << name << "_MEAN_MS="
                  << times.mpc_to_ckks_stages.*member / options.iterations << '\n';
    }
    const auto print_samples = [](const char *name, const std::vector<double> &values) {
        std::cout << name << '=';
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (i != 0) std::cout << ',';
            std::cout << values[i];
        }
        std::cout << '\n';
    };
    print_samples("CKKS_TO_MPC_SAMPLES_MS", times.ckks_to_mpc);
    print_samples("MPC_TO_CKKS_SAMPLES_MS", times.mpc_to_ckks);
    std::cout << std::flush;
}

int RunClient(const Options &options, const hera::HeraConfig &config)
{
    hera::HeContextRegistry contexts;
    Check(contexts.CreateClientProfiles(config, false));
    auto serialized = contexts.ExportSerializedPublicBundle(
        CompressionMode(options));
    if (!serialized.ok()) throw std::runtime_error(serialized.status().message());
    auto control = hera::MakeNetIoTransport(hera::PartyRole::kClient,
                                            options.ip, options.port);
    Check(hera::SendPublicContextBundle(*control, serialized.value()));

    const std::string profile_name = SelectedProfile(options, config);
    const auto &profile = contexts.Require(profile_name,
                                           hera::PartyRole::kClient);
    std::vector<std::vector<std::complex<double>>> clear(options.ciphertexts,
        std::vector<std::complex<double>>(profile.slot_count()));
    seal::CKKSEncoder encoder(*profile.context);
    seal::Plaintext plain;
    seal::Encryptor encryptor(*profile.context, profile.secret_key.value());
    SendString(*control, profile_name);
    const std::uint64_t wire_count = options.ciphertexts;
    control->Send(&wire_count, sizeof(wire_count));
    for (std::size_t ct = 0; ct < options.ciphertexts; ++ct) {
        auto &values = clear[ct];
        if (options.dense) {
            for (std::size_t i = 0; i < values.size(); ++i)
                values[i] = {static_cast<double>(static_cast<int>((i + 3 * ct) % 33) - 16) / 8.,
                    static_cast<double>(static_cast<int>((7 * i + ct) % 25) - 12) / 8.};
        } else {
            values[0] = {0.125, -0.25}; values[1] = {-1.5, 0.75};
            values[2] = {2.0, -1.0}; values[3] = {-0.0625, 0.03125};
        }
        encoder.encode(values, options.dense ? profile.context->last_parms_id() : profile.context->first_parms_id(),
            std::ldexp(1., profile.parameters.output_scale_bits), plain);
        seal::Ciphertext input;
        encryptor.encrypt_symmetric(plain, input);
        if (options.dense) {
            // Isolate conversion error from source encryption/encoding noise.
            seal::Decryptor reference_decryptor(*profile.context, profile.secret_key.value());
            seal::Plaintext reference_plain;
            reference_decryptor.decrypt(input, reference_plain);
            encoder.decode(reference_plain, values);
        }
        SendCiphertext(*control, input, CompressionMode(options));
    }

    hera::CkksMpcConverter converter(
        hera::PartyRole::kClient, ProtocolConfig(config, options), contexts, control,
        hera::MakeProductionRandomSource(), options.ip, options.port + 200,
        options.threads, CompressionMode(options));
    const auto map = DemoMap(profile.slot_count(), options);
    hera::MpcToCkksTarget target{
        profile_name, 0, profile.parameters.input_scale_bits,
        hera::TensorShape{{map.logical_element_count}}, hera::Layout::kSpatial, map};
    ConversionBenchmarkTimes times;
    auto validation = options.dense ? hera::MakeNetIoTransport(hera::PartyRole::kClient, options.ip, options.port + 1) : nullptr;
    double maximum_c2m_error = 0;
    for (std::size_t iteration = 0;
         iteration < options.warmup + options.iterations; ++iteration) {
        const auto before_c2m = converter.stats();
        const auto c2m_start = BenchmarkClock::now();
        auto shares = converter.CkksToMpc(std::nullopt);
        const auto c2m_stop = BenchmarkClock::now();
        if (!shares.ok()) throw std::runtime_error(shares.status().message());
        const auto after_c2m = converter.stats();
        if (validation) {
            std::vector<std::uint64_t> real(shares.value().real.local_share.size()), imag(real.size());
            validation->Receive(real.data(), real.size() * sizeof(real[0]));
            validation->Receive(imag.data(), imag.size() * sizeof(imag[0]));
            const int bits = config.mpc_numeric.ring_bits;
            const auto mask = (std::uint64_t{1} << bits) - 1;
            const auto decode = [&](std::uint64_t value) {
                value &= mask;
                const auto signed_value = static_cast<std::int64_t>(value & (std::uint64_t{1} << (bits - 1)) ? value | ~mask : value);
                return std::ldexp(static_cast<double>(signed_value), -config.mpc_numeric.scale_bits);
            };
            for (std::size_t i = 0; i < real.size(); ++i) {
                const auto reference = clear[i / profile.slot_count()][i % profile.slot_count()];
                maximum_c2m_error = std::max({maximum_c2m_error,
                    std::abs(decode(real[i] + shares.value().real.local_share[i]) - reference.real()),
                    std::abs(decode(imag[i] + shares.value().imag.local_share[i]) - reference.imag())});
            }
            const std::uint8_t ok = maximum_c2m_error <= std::ldexp(2., -config.mpc_numeric.scale_bits);
            validation->Send(&ok, sizeof(ok));
            if (!ok) throw std::runtime_error("dense C2M exceeds 2 ULP");
        }
        const auto m2c_start = BenchmarkClock::now();
        auto converted = converter.MpcToCkks(shares.value(), target);
        const auto m2c_stop = BenchmarkClock::now();
        if (!converted.ok())
            throw std::runtime_error(converted.status().message());
        if (converted.value())
            throw std::runtime_error("client received server-only ciphertext");
        const auto after_m2c = converter.stats();
        if (iteration >= options.warmup) {
            AccumulateStages(times.ckks_to_mpc_stages, before_c2m.ckks_to_mpc,
                             after_c2m.ckks_to_mpc);
            AccumulateStages(times.mpc_to_ckks_stages, after_c2m.mpc_to_ckks,
                             after_m2c.mpc_to_ckks);
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
    seal::Decryptor decryptor(*profile.context, profile.secret_key.value());
    seal::Plaintext result_plain;
    double maximum_error = 0;
    for (std::size_t ct = 0; ct < options.ciphertexts; ++ct) {
        auto result = ReceiveCiphertext(*control, *profile.context);
        decryptor.decrypt(result, result_plain);
        std::vector<std::complex<double>> decoded;
        encoder.decode(result_plain, decoded);
        for (std::size_t i = 0; i < (options.dense ? decoded.size() : 4); ++i)
            maximum_error = std::max({maximum_error,
                std::abs(decoded[i].real() - clear[ct][i].real()),
                std::abs(decoded[i].imag() - clear[ct][i].imag())});
    }
    const auto stats = converter.stats();
    if (maximum_error > std::ldexp(options.dense ? 4. : 48., -config.mpc_numeric.scale_bits))
        throw std::runtime_error(
            "conversion round-trip exceeds " + std::to_string(options.dense ? 4 : 48) + "-ULP bound");
    std::cout << "MAX_C2M_ULP=" << std::ldexp(maximum_c2m_error, config.mpc_numeric.scale_bits)
              << "\nMAX_ROUNDTRIP_ULP=" << std::ldexp(maximum_error, config.mpc_numeric.scale_bits) << '\n';
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
    std::uint64_t wire_count{};
    control->Receive(&wire_count, sizeof(wire_count));
    if (wire_count != options.ciphertexts) throw std::invalid_argument("client/server ciphertext counts differ");
    std::vector<seal::Ciphertext> inputs;
    for (std::size_t ct = 0; ct < options.ciphertexts; ++ct)
        inputs.push_back(ReceiveCiphertext(*control, *profile.context));
    const auto map = DemoMap(profile.slot_count(), options);
    hera::SealCipherBatch batch{
        profile_name, std::move(inputs), hera::TensorShape{{map.logical_element_count}},
        hera::Layout::kSpatial, map};
    hera::CkksMpcConverter converter(
        hera::PartyRole::kServer, ProtocolConfig(config, options), contexts, control,
        hera::MakeProductionRandomSource(), "", options.port + 200,
        options.threads, CompressionMode(options));
    hera::MpcToCkksTarget target{
        profile_name, 0, profile.parameters.input_scale_bits,
        hera::TensorShape{{map.logical_element_count}}, hera::Layout::kSpatial, map};
    ConversionBenchmarkTimes times;
    auto validation = options.dense ? hera::MakeNetIoTransport(hera::PartyRole::kServer, "", options.port + 1) : nullptr;
    std::optional<hera::SealCipherBatch> output;
    for (std::size_t iteration = 0;
         iteration < options.warmup + options.iterations; ++iteration) {
        const auto before_c2m = converter.stats();
        const auto c2m_start = BenchmarkClock::now();
        auto shares = converter.CkksToMpc(batch);
        const auto c2m_stop = BenchmarkClock::now();
        if (!shares.ok()) throw std::runtime_error(shares.status().message());
        const auto after_c2m = converter.stats();
        if (validation) {
            validation->Send(shares.value().real.local_share.data(), shares.value().real.local_share.size() * sizeof(std::uint64_t));
            validation->Send(shares.value().imag.local_share.data(), shares.value().imag.local_share.size() * sizeof(std::uint64_t));
            std::uint8_t ok{};
            validation->Receive(&ok, sizeof(ok));
            if (!ok) throw std::runtime_error("dense C2M validation failed");
        }
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
            AccumulateStages(times.ckks_to_mpc_stages, before_c2m.ckks_to_mpc,
                             after_c2m.ckks_to_mpc);
            AccumulateStages(times.mpc_to_ckks_stages, after_c2m.mpc_to_ckks,
                             after_m2c.mpc_to_ckks);
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
    for (const auto &ciphertext : output->ciphertexts)
        SendCiphertext(*control, ciphertext, CompressionMode(options));
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
              << " field_batches=" << stats.field_to_ring_batches
              << " max_field_elements=" << stats.max_field_to_ring_elements
              << " extension_batches=" << stats.signed_extension_batches
              << " max_extension_elements=" << stats.max_signed_extension_elements
              << '\n';
    PrintBenchmark(options, "server", times);
    return 0;
}

} // namespace

int main(int argc, char **argv)
{
    try {
        const Options options = Parse(argc, argv);
        (void)CompressionMode(options);
        std::cout << "HERA_SEAL_COMPRESSION=" << options.seal_compression
                  << '\n' << std::flush;
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
