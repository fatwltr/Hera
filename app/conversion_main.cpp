#include "hera/conversion.h"

#include <seal/seal.h>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cmath>
#include <algorithm>
#include <complex>
#include <filesystem>
#include <iostream>
#include <optional>
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
    return result;
}

void Check(const hera::Result<void> &result)
{
    if (!result.ok()) throw std::runtime_error(result.status().message());
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
                    const seal::Ciphertext &ciphertext)
{
    std::stringstream stream;
    ciphertext.save(stream, seal::compr_mode_type::none);
    const std::string bytes = stream.str();
    const std::uint64_t size = bytes.size();
    transport.Send(&size, sizeof(size));
    transport.Send(bytes.data(), bytes.size());
}

seal::Ciphertext ReceiveCiphertext(hera::IProtocolTransport &transport,
                                   const seal::SEALContext &context)
{
    std::uint64_t size = 0;
    transport.Receive(&size, sizeof(size));
    if (size == 0 || size > (std::uint64_t{1} << 34))
        throw std::runtime_error("invalid ciphertext size on control channel");
    std::string bytes(size, '\0');
    transport.Receive(bytes.data(), bytes.size());
    std::stringstream stream(bytes);
    seal::Ciphertext result;
    result.load(context, stream);
    return result;
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

int RunClient(const Options &options, const hera::HeraConfig &config)
{
    hera::HeContextRegistry contexts;
    Check(contexts.CreateClientProfiles(config));
    auto serialized = contexts.ExportSerializedPublicBundle();
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
    SendCiphertext(*control, input);

    hera::CkksMpcConverter converter(
        hera::PartyRole::kClient, ProtocolConfig(config), contexts, control,
        hera::MakeProductionRandomSource(), options.ip, options.port + 200,
        options.threads);
    auto shares = converter.CkksToMpc(std::nullopt);
    if (!shares.ok()) throw std::runtime_error(shares.status().message());
    const auto map = DemoMap(profile.slot_count());
    hera::MpcToCkksTarget target{
        profile_name, 0, profile.parameters.input_scale_bits,
        hera::TensorShape{{8}}, hera::Layout::kSpatial, map};
    auto converted = converter.MpcToCkks(shares.value(), target);
    if (!converted.ok()) throw std::runtime_error(converted.status().message());
    if (converted.value())
        throw std::runtime_error("client received server-only ciphertext");
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
              << " rounds=" << stats.rounds << '\n' << std::flush;
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
        options.threads);
    auto shares = converter.CkksToMpc(batch);
    if (!shares.ok()) throw std::runtime_error(shares.status().message());
    hera::MpcToCkksTarget target{
        profile_name, 0, profile.parameters.input_scale_bits,
        hera::TensorShape{{8}}, hera::Layout::kSpatial, map};
    auto converted = converter.MpcToCkks(shares.value(), target);
    if (!converted.ok() || !converted.value())
        throw std::runtime_error(converted.ok()
            ? "server did not receive output ciphertext"
            : converted.status().message());
    SendCiphertext(*control, converted.value()->ciphertexts.front());
    const auto stats = converter.stats();
    std::cout << "party=server profile=" << profile_name
              << " communication_bytes=" << stats.communication_bytes
              << " rounds=" << stats.rounds
              << " a_mask_bytes=" << stats.a_mask_bytes
              << " z_mask_bytes=" << stats.z_mask_bytes
              << " internal_mod_switches=" << stats.internal_mod_switches
              << '\n';
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
        return server_result;
    } catch (const std::exception &error) {
        std::cerr << "HERA_COMPLEX_CONVERSION_STATUS=FAIL: " << error.what()
                  << '\n';
        return 1;
    }
}
