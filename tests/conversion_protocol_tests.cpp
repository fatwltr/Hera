#include "hera/conversion.h"
#include "hera/conversion_backend.h"
#include "hera/seal_wire.h"
#include "hera/detail/he_packing.h"

#include "fixed_ckks_transform.h"
#include "random_sampling.h"

#include <seal/util/ntt.h>
#include <seal/util/uintcore.h>

#include <seal/seal.h>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cmath>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

void Require(bool condition, const std::string &message)
{
    if (!condition) throw std::runtime_error(message);
}

struct CaptureState {
    std::mutex mutex;
    std::map<std::string, std::vector<std::uint8_t>> bytes;
};

class CapturingStream final : public hera::IRandomStream {
public:
    CapturingStream(std::unique_ptr<hera::IRandomStream> inner,
                    std::shared_ptr<CaptureState> state, std::string label)
        : inner_(std::move(inner)), state_(std::move(state)),
          label_(std::move(label)) {}

    void Fill(void *data, std::size_t bytes) override
    {
        inner_->Fill(data, bytes);
        const auto *begin = static_cast<const std::uint8_t *>(data);
        std::lock_guard<std::mutex> lock(state_->mutex);
        auto &output = state_->bytes[label_];
        output.insert(output.end(), begin, begin + bytes);
    }

private:
    std::unique_ptr<hera::IRandomStream> inner_;
    std::shared_ptr<CaptureState> state_;
    std::string label_;
};

class CapturingRandomSource final : public hera::IRandomSource {
public:
    CapturingRandomSource(std::shared_ptr<hera::IRandomSource> inner,
                          std::shared_ptr<CaptureState> state)
        : inner_(std::move(inner)), state_(std::move(state)) {}

    hera::Result<std::unique_ptr<hera::IRandomStream>> OpenStream(
        const std::string &label) override
    {
        auto inner = inner_->OpenStream(label);
        if (!inner.ok()) return inner.status();
        return std::unique_ptr<hera::IRandomStream>(new CapturingStream(
            inner.take_value(), state_, label));
    }

private:
    std::shared_ptr<hera::IRandomSource> inner_;
    std::shared_ptr<CaptureState> state_;
};

class CapturingTransport final : public hera::IProtocolTransport {
public:
    explicit CapturingTransport(std::shared_ptr<hera::IProtocolTransport> inner)
        : inner_(std::move(inner)) {}

    void Send(const void *data, std::size_t bytes) override
    {
        if (capture) {
            const auto *first = static_cast<const std::uint8_t *>(data);
            sent.emplace_back(first, first + bytes);
        }
        if (obsolete_frame && bytes >= 12) {
            std::uint64_t magic;
            std::memcpy(&magic, data, sizeof(magic));
            if (magic == 0x48455241434f4e56ULL) {
                const auto *first = static_cast<const std::uint8_t *>(data);
                std::vector<std::uint8_t> changed(first, first + bytes);
                const std::uint32_t obsolete_version = 13;
                std::memcpy(changed.data() + sizeof(magic), &obsolete_version, sizeof(obsolete_version));
                inner_->Send(changed.data(), changed.size());
                return;
            }
        }
        inner_->Send(data, bytes);
    }
    void Receive(void *data, std::size_t bytes) override
    { inner_->Receive(data, bytes); }
    void Cancel() noexcept override { inner_->Cancel(); }
    std::uint64_t bytes() const noexcept override { return inner_->bytes(); }
    std::uint64_t rounds() const noexcept override { return inner_->rounds(); }

    bool capture{true};
    bool obsolete_frame{false};
    std::vector<std::vector<std::uint8_t>> sent;

private:
    std::shared_ptr<hera::IProtocolTransport> inner_;
};

class ReplayTransport final : public hera::IProtocolTransport {
public:
    std::vector<std::uint8_t> data;
    std::size_t position{};

    void Send(const void *, std::size_t) override
    { throw std::runtime_error("unexpected replay send"); }
    void Receive(void *output, std::size_t bytes) override
    {
        Require(bytes <= data.size() - position, "incomplete captured wire data");
        std::memcpy(output, data.data() + position, bytes);
        position += bytes;
    }
    void Cancel() noexcept override {}
    std::uint64_t bytes() const noexcept override { return position; }
    std::uint64_t rounds() const noexcept override { return 0; }
};

class ReplayRandomStream final : public hera::IRandomStream {
public:
    explicit ReplayRandomStream(const std::vector<std::uint8_t> &data)
        : data_(data) {}
    void Fill(void *output, std::size_t bytes) override
    {
        Require(bytes <= data_.size() - position_, "incomplete captured randomness");
        std::memcpy(output, data_.data() + position_, bytes);
        position_ += bytes;
    }
private:
    const std::vector<std::uint8_t> &data_;
    std::size_t position_{};
};

void VerifyCombinedCiphertextMask(
    const CapturingTransport &wire, const CaptureState &random,
    const hera::HeContext &client_profile, const seal::Ciphertext &input)
{
    // The first two sends are the protocol frame and profile name. Everything
    // after them must be the masked ciphertext, with no post-decrypt Z deltas.
    Require(wire.sent.size() > 2, "masked ciphertext was not captured");
    ReplayTransport replay;
    for (std::size_t i = 2; i < wire.sent.size(); ++i)
        replay.data.insert(replay.data.end(), wire.sent[i].begin(),
                           wire.sent[i].end());
    auto masked = hera::ReceiveSealObject<seal::Ciphertext>(
        replay, *client_profile.context);
    Require(replay.position == replay.data.size(),
            "CkksToMpc still sends post-decrypt output-share refreshes");

    const auto find_stream = [&](const std::string &prefix)
        -> const std::vector<std::uint8_t> & {
        const std::vector<std::uint8_t> *found = nullptr;
        for (const auto &entry : random.bytes) {
            if (entry.first.rfind(prefix, 0) == 0) {
                Require(found == nullptr, "unexpected duplicate mask stream");
                found = &entry.second;
            }
        }
        Require(found != nullptr, "mask stream was not sampled");
        return *found;
    };
    Require(random.bytes.size() == 4,
            "single-ciphertext C2M must sample session and independent A/Z masks; wide carry uses OT pads");
    const auto terminal = client_profile.context->last_context_data();
    const auto modulus = terminal->parms().coeff_modulus().front().value();
    const auto degree = client_profile.parameters.polynomial_degree;
    const auto slots = client_profile.slot_count();
    ReplayRandomStream a_stream(find_stream("hera/c2m/A/"));
    ReplayRandomStream real_stream(find_stream("hera/c2m/Z-real/"));
    ReplayRandomStream imag_stream(find_stream("hera/c2m/Z-imag/"));
    const auto a = hera::conversion_internal::SampleUniformPrime(
        a_stream, degree, modulus);
    const auto real = hera::conversion_internal::SampleUniformRing(
        real_stream, slots, 43);
    const auto imag = hera::conversion_internal::SampleUniformRing(
        imag_stream, slots, 43);
    constexpr int root_bits = 24;
    constexpr int scale_shift = 20 - 13;
    const int wide_bits = seal::util::get_significant_bit_count(modulus) + 40;
    const int decode_guard = std::min(20, 127 - wide_bits - root_bits - 2);
    const int mask_value_bits = 43 + scale_shift +
        seal::util::get_power_of_two(degree) + 2;
    const int mask_bits = std::max(wide_bits + decode_guard, mask_value_bits);
    const int mask_guard = std::min(20, mask_bits - mask_value_bits);
    const auto transform =
        hera::conversion_internal::FixedCkksTransform::Create(
            degree, root_bits, mask_bits).take_value();
    const auto encoded_z = hera::conversion_internal::EncodeRingMask(
        real, imag, 43, scale_shift, mask_guard, transform, modulus).take_value();

    seal::Evaluator evaluator(*client_profile.context);
    auto terminal_input = input;
    evaluator.mod_switch_to_inplace(terminal_input, terminal->parms_id());
    seal::Decryptor decryptor(*client_profile.context,
                              client_profile.secret_key.value());
    seal::Plaintext original_plain, masked_plain;
    decryptor.decrypt(terminal_input, original_plain);
    decryptor.decrypt(masked, masked_plain);
    seal::util::inverse_ntt_negacyclic_harvey(
        original_plain.data(), terminal->small_ntt_tables()[0]);
    seal::util::inverse_ntt_negacyclic_harvey(
        masked_plain.data(), terminal->small_ntt_tables()[0]);
    bool a_contributed = false, z_contributed = false, mask_wrapped = false;
    for (std::size_t i = 0; i < degree; ++i) {
        const auto combined = (a[i] + encoded_z[i]) % modulus;
        const auto actual = (masked_plain[i] + modulus - original_plain[i]) % modulus;
        Require(actual == combined,
                "client ciphertext is not masked with A + Encode(Z)");
        a_contributed |= actual != encoded_z[i];
        z_contributed |= actual != a[i];
        mask_wrapped |= a[i] + encoded_z[i] >= modulus;
    }
    Require(a_contributed && z_contributed && mask_wrapped,
            "combined-mask regression did not exercise both domains and wrapping");
}

hera::HeraConfig TestConfig()
{
    hera::HeraConfig config;
    config.layers.push_back(hera::TransformerLayerConfig{
        "test", 2, 4, 1, 1, 4, 4, 8, hera::NormPlacement::kPreNorm,
        hera::ActivationKind::kGelu, hera::AttentionMaskKind::kNone,
        false, false, false, true, true});
    config.mpc_numeric = {43, 13};
    config.conversion = {40, 24, true};
    config.he_parameter_profiles.emplace(
        "test_profile",
        hera::HeParameterConfig{2048, {40, 30, 40}, 20, 20, 20, 1, 1,
                                true, hera::SecurityLevel::kNone});
    config.he_parameter_profiles.emplace(
        "target_profile",
        hera::HeParameterConfig{4096, {45, 35, 45}, 24, 24, 24, 1, 1,
                                true, hera::SecurityLevel::kNone});
    hera::LayerExecutionConfig execution;
    execution.layer_id = "test";
    for (const auto region : {hera::HeRegionId::kRegion1,
                              hera::HeRegionId::kRegion2,
                              hera::HeRegionId::kRegion3,
                              hera::HeRegionId::kRegion4,
                              hera::HeRegionId::kRegion5})
        execution.regions.emplace(
            region, hera::HeRegionConfig{region, "test_profile",
                                         hera::Layout::kSpatial,
                                         hera::Layout::kSpatial});
    config.layer_execution.push_back(std::move(execution));
    config.threads = 1;
    return config;
}

hera::PhysicalSlotMap ComplexTestMap(std::size_t slots,
                                     std::size_t ciphertext_count = 1)
{
    hera::PhysicalSlotMap map;
    map.ciphertext_count = ciphertext_count;
    map.slots_per_ciphertext = slots;
    map.logical_element_count = 8 * ciphertext_count;
    map.assignments.resize(slots * ciphertext_count);
    for (std::size_t ct = 0; ct < ciphertext_count; ++ct) {
        for (std::size_t i = 0; i < 4; ++i) {
            auto &assignment = map.assignments[ct * slots + i];
            assignment.real_logical_index = 8 * ct + 2 * i;
            assignment.imaginary_logical_index = 8 * ct + 2 * i + 1;
        }
    }
    return map;
}

hera::ComplexMpcSharePair RetargetShares(
    const hera::ComplexMpcSharePair &source,
    const hera::PhysicalSlotMap &source_map,
    const hera::PhysicalSlotMap &target_map)
{
    std::vector<std::uint64_t> logical(source_map.logical_element_count);
    for (std::size_t physical = 0; physical < source_map.assignments.size();
         ++physical) {
        const auto assignment = source_map.assignments[physical];
        if (assignment.real_logical_index != hera::kUnusedLogicalIndex)
            logical[assignment.real_logical_index] =
                source.real.local_share[physical];
        if (assignment.imaginary_logical_index != hera::kUnusedLogicalIndex)
            logical[assignment.imaginary_logical_index] =
                source.imag.local_share[physical];
    }
    const auto count = target_map.assignments.size();
    const auto shape = hera::TensorShape{{target_map.ciphertext_count,
                                          target_map.slots_per_ciphertext}};
    hera::ComplexMpcSharePair target{
        {std::vector<std::uint64_t>(count), shape, source.real.numeric,
         hera::ComplexComponent::kReal},
        {std::vector<std::uint64_t>(count), shape, source.imag.numeric,
         hera::ComplexComponent::kImaginary}};
    for (std::size_t physical = 0; physical < count; ++physical) {
        const auto assignment = target_map.assignments[physical];
        if (assignment.real_logical_index != hera::kUnusedLogicalIndex)
            target.real.local_share[physical] =
                logical[assignment.real_logical_index];
        if (assignment.imaginary_logical_index != hera::kUnusedLogicalIndex)
            target.imag.local_share[physical] =
                logical[assignment.imaginary_logical_index];
    }
    return target;
}

void WriteAll(int fd, const void *data, std::size_t bytes)
{
    const auto *cursor = static_cast<const std::uint8_t *>(data);
    while (bytes != 0) {
        const ssize_t written = write(fd, cursor, bytes);
        if (written <= 0) throw std::runtime_error("pipe write failed");
        cursor += written;
        bytes -= static_cast<std::size_t>(written);
    }
}

void ReadAll(int fd, void *data, std::size_t bytes)
{
    auto *cursor = static_cast<std::uint8_t *>(data);
    while (bytes != 0) {
        const ssize_t read_bytes = read(fd, cursor, bytes);
        if (read_bytes <= 0) throw std::runtime_error("pipe read failed");
        cursor += read_bytes;
        bytes -= static_cast<std::size_t>(read_bytes);
    }
}


void TestPublicZeroImaginary(hera::CkksMpcConverter &converter,
                             hera::PartyRole role,
                             const hera::HeContext &profile, int pipe_fd)
{
    const bool server = role == hera::PartyRole::kServer;
    const auto slots = profile.slot_count();
    const std::vector<double> values{-1.25, 0.0, 0.5, 2.0};
    const auto ring_mask = (std::uint64_t{1} << 43) - 1;
    const hera::TensorShape shape{{1, slots}};
    hera::ComplexMpcSharePair pair{
        {std::vector<std::uint64_t>(slots), shape, {43, 13, true}, hera::ComplexComponent::kReal},
        {std::vector<std::uint64_t>(slots), shape, {43, 13, true}, hera::ComplexComponent::kImaginary}};
    for (std::size_t i = 0; i < values.size(); ++i) {
        const auto clear = static_cast<std::uint64_t>(
            static_cast<std::int64_t>(std::llround(values[i] * 8192.0))) & ring_mask;
        const auto alice = (0x12345 + 97 * i) & ring_mask;
        pair.real.local_share[i] = server ? alice : (clear - alice) & ring_mask;
    }
    hera::MpcToCkksTarget target{
        "target_profile", 0, 24, {{4}}, hera::Layout::kSpatial,
        hera::PhysicalSlotMap::DenseReal(4, slots), true};
    const auto before = converter.stats();
    // Exercise both extension paths with negative values, zero, and padding,
    // including the public-zero coefficient that cannot skip re-extension.
    for (int mode : {0, 1, 2}) for (bool real_only : {true, false}) {
        target.public_zero_imaginary = real_only;
        target.public_value_bound_bits = mode ? std::optional<int>{mode == 1 ? 15 : 42} : std::nullopt;
        auto converted = converter.MpcToCkks(pair, target);
        Require(converted.ok(), converted.status().message());
        if (server) {
            Require(converted.value() && converted.value()->ciphertexts.size() == 1,
                    "real-only conversion output count mismatch");
            std::stringstream stream;
            converted.value()->ciphertexts.front().save(stream, seal::compr_mode_type::none);
            const auto bytes = stream.str();
            const std::uint64_t size = bytes.size();
            WriteAll(pipe_fd, &size, sizeof(size));
            WriteAll(pipe_fd, bytes.data(), bytes.size());
        } else {
            Require(!converted.value(), "client received real-only HE output");
            std::uint64_t size = 0;
            ReadAll(pipe_fd, &size, sizeof(size));
            Require(size && size < (std::uint64_t{1} << 30), "invalid real-only CT size");
            std::string bytes(size, '\0');
            ReadAll(pipe_fd, bytes.data(), bytes.size());
            std::stringstream stream(bytes);
            seal::Ciphertext ciphertext;
            ciphertext.load(*profile.context, stream);
            seal::Decryptor decryptor(*profile.context, profile.secret_key.value());
            seal::CKKSEncoder encoder(*profile.context);
            seal::Plaintext plain;
            decryptor.decrypt(ciphertext, plain);
            std::vector<std::complex<double>> decoded;
            encoder.decode(plain, decoded);
            for (std::size_t i = 0; i < values.size(); ++i)
                Require(std::abs(decoded[i].real() - values[i]) <= 4.0 / 8192.0 &&
                            std::abs(decoded[i].imag()) <= 4.0 / 8192.0,
                        "public-zero conversion changed real values or imaginary zero");
        }
    }
    Require(converter.stats().public_zero_imaginary_ciphertexts -
                before.public_zero_imaginary_ciphertexts == 3,
            "general conversion inferred a public-zero contract");
    Require(converter.stats().bounded_signed_extension_elements -
                before.bounded_signed_extension_elements == 7 * slots &&
            converter.stats().generic_signed_extension_elements -
                before.generic_signed_extension_elements == 14 * slots,
            "public-bound slot/coefficient extensions did not use the expected paths");
    target.public_zero_imaginary = true;
    target.public_value_bound_bits.reset();
    for (int invalid = 0; invalid < 10; ++invalid) {
        auto bad_pair = pair;
        auto bad_target = target;
        if ((invalid == 0 && server) || (invalid == 1 && !server))
            bad_pair.imag.local_share[0] = 1;
        if (invalid == 2 && !server) bad_target.public_zero_imaginary = false;
        if (invalid == 3) {
            bad_target.slot_map.assignments[3].real_logical_index = hera::kUnusedLogicalIndex;
            bad_target.slot_map.assignments[0].imaginary_logical_index = 3;
        }
        if (invalid == 4 && server) bad_target.public_value_bound_bits = 15;
        if (invalid == 5) bad_target.public_value_bound_bits = 43;
        if (invalid == 6) bad_target.public_value_bound_bits = -1;
        if (invalid == 7) bad_target.public_ciphertext_bound_bits = {15, 15};
        if (invalid == 8) bad_target.public_ciphertext_bound_bits = {server ? 14 : 15};
        if (invalid == 9) bad_target.public_ciphertext_bound_bits = {43};
        auto rejected = converter.MpcToCkks(bad_pair, bad_target);
        Require(!rejected.ok() && rejected.status().message().find("handshake") != std::string::npos,
                "public-zero handshake rejection failed for case " + std::to_string(invalid));
    }
}

void TestPairedRealBoundary(hera::CkksMpcConverter &converter,
    hera::PartyRole role, const hera::HeContext &profile,
    const seal::Ciphertext &real_input, int pipe_fd)
{
    hera::HePackingPlan original;
    original.layout = hera::Layout::kStrictColumn;
    original.logical_shape = {{12}};
    original.slot_count = profile.slot_count();
    original.ciphertext_count = 3;
    original.logical_locations.resize(12);
    for (std::size_t i = 0; i < 12; ++i)
        original.logical_locations[i].push_back(
            {i / 4, i % 4, hera::ComplexLane::kReal, std::nullopt});
    const auto packing = hera::detail::PairedRealPacking(original);
    Require(packing.ciphertext_count == 2, "odd real boundary did not pair 3->2");
    hera::PhysicalSlotMap map;
    map.ciphertext_count = 2;
    map.slots_per_ciphertext = profile.slot_count();
    map.logical_element_count = 12;
    map.assignments.resize(2 * profile.slot_count());
    for (std::size_t i = 0; i < 12; ++i) {
        const auto &location = packing.logical_locations[i].front();
        auto &entry = map.assignments[location.ciphertext * profile.slot_count() + location.slot];
        (location.component == hera::ComplexLane::kReal
            ? entry.real_logical_index : entry.imaginary_logical_index) = i;
    }
    std::optional<hera::SealCipherBatch> batch;
    const bool server = role == hera::PartyRole::kServer;
    if (server) {
        auto paired = hera::detail::PairRealCiphertexts(
            {real_input, real_input, real_input}, profile, 2);
        Require(paired.ok(), paired.status().message());
        for (const auto &ciphertext : paired.value())
            Require(ciphertext.parms_id() == real_input.parms_id() &&
                    ciphertext.scale() == real_input.scale(),
                    "real boundary pairing consumed level or changed scale");
        batch = hera::SealCipherBatch{profile.profile, paired.take_value(),
            {{12}}, hera::Layout::kStrictColumn, map};
    }
    auto converted = converter.CkksToMpc(batch);
    Require(converted.ok(), converted.status().message());
    const auto count = 2 * profile.slot_count();
    if (server) {
        WriteAll(pipe_fd, converted.value().real.local_share.data(), count * sizeof(std::uint64_t));
        WriteAll(pipe_fd, converted.value().imag.local_share.data(), count * sizeof(std::uint64_t));
    } else {
        std::vector<std::uint64_t> real(count), imag(count);
        ReadAll(pipe_fd, real.data(), count * sizeof(std::uint64_t));
        ReadAll(pipe_fd, imag.data(), count * sizeof(std::uint64_t));
        seal::Decryptor decryptor(*profile.context, profile.secret_key.value());
        seal::CKKSEncoder encoder(*profile.context);
        seal::Plaintext plain;
        decryptor.decrypt(real_input, plain);
        std::vector<std::complex<double>> reference;
        encoder.decode(plain, reference);
        const auto mask = (std::uint64_t{1} << 43) - 1;
        for (std::size_t i = 0; i < 12; ++i) {
            const auto &location = packing.logical_locations[i].front();
            const auto offset = location.ciphertext * profile.slot_count() + location.slot;
            auto value = location.component == hera::ComplexLane::kReal
                ? real[offset] + converted.value().real.local_share[offset]
                : imag[offset] + converted.value().imag.local_share[offset];
            if (location.component == hera::ComplexLane::kImaginary)
                value = hera::detail::UndoPairedRealPhase(value, location.slot, 43);
            value &= mask;
            const auto actual = (value & (std::uint64_t{1} << 42))
                ? static_cast<std::int64_t>(value | ~mask) : static_cast<std::int64_t>(value);
            const auto phase = (location.slot & 1) ? -1.0 : 1.0;
            const auto expected = reference[location.slot].real() +
                (location.component == hera::ComplexLane::kImaginary ? phase : -phase) * reference[location.slot].imag();
            Require(std::abs(actual / 8192.0 - expected) <= 2.0 / 8192.0,
                    "paired real boundary phase, tail or C2M oracle failed");
        }
    }
}

void TestWorkerBatchNegotiation(hera::HeContextRegistry &client_registry,
                                 hera::HeContextRegistry &server_registry)
{
    for (int mismatch = 0; mismatch < 2; ++mismatch) {
        const int port = 17500 + 200 * mismatch;
        const auto child = fork();
        Require(child >= 0, "cannot fork converter contract test");
        const auto role = child == 0 ? hera::PartyRole::kClient : hera::PartyRole::kServer;
        bool rejected = false;
        try {
            auto config = hera::ConversionConfig{};
            config.batch_elements = child == 0 && mismatch == 0 ? 32768 : 65536;
            auto transport = hera::MakeNetIoTransport(role, "127.0.0.1", port);
            hera::CkksMpcConverter converter(role, config,
                child == 0 ? client_registry : server_registry, transport,
                hera::MakeDeterministicRandomSourceForTesting({1, 3, 5}),
                "127.0.0.1", port + 20, child == 0 && mismatch == 1 ? 3 : 2);
        } catch (const std::invalid_argument &error) {
            rejected = std::string(error.what()).find("contracts differ") != std::string::npos;
        }
        if (child == 0) _exit(rejected ? 0 : 2);
        int status{};
        waitpid(child, &status, 0);
        Require(rejected && WIFEXITED(status) && WEXITSTATUS(status) == 0,
                "converter accepted a mismatched worker/batch contract");
    }
}

} // namespace

int main(int argc, char **argv)
{
    int result_pipe[2];
    if (pipe(result_pipe) != 0) return 1;
    try {
        const auto compression = argc == 2 &&
                std::string(argv[1]) == "--uncompressed"
            ? seal::compr_mode_type::none
            : seal::Serialization::compr_mode_default;
        const bool small_batches = argc == 2 && std::string(argv[1]) == "--small-batches";
        Require(argc == 1 || (argc == 2 &&
                    (std::string(argv[1]) == "--uncompressed" || small_batches)),
                "expected no arguments, --uncompressed or --small-batches");
        hera::ConversionConfig protocol{43, 13, 40, 24, true};
        if (small_batches) protocol.batch_elements = 2048;
        const auto config = TestConfig();
        Require(hera::ValidateHeraConfig(config).ok(), "test config invalid");
        hera::HeContextRegistry client_registry;
        auto client_created = client_registry.CreateClientProfiles(config, false);
        Require(client_created.ok(), client_created.status().message());
        auto direct_bundle = client_registry.ExportPublicBundle();
        Require(direct_bundle.ok(), direct_bundle.status().message());
        hera::HeContextRegistry direct_server;
        Require(direct_server.CreateServerProfiles(config, direct_bundle.value()).ok(),
                "cannot import a public bundle without evaluation keys");
        auto bundle = client_registry.ExportSerializedPublicBundle(compression);
        Require(bundle.ok(), bundle.status().message());
        for (const auto &profile : bundle.value().profiles)
            Require(profile.relinearization_keys.empty() &&
                        profile.galois_keys.empty() && !profile.public_key.empty(),
                    "conversion-only bundle contains unused evaluation keys");
        hera::HeContextRegistry server_registry;
        auto server_created =
            server_registry.CreateServerProfiles(config, bundle.value());
        Require(server_created.ok(), server_created.status().message());
        TestWorkerBatchNegotiation(client_registry, server_registry);
        for (const auto &entry : config.he_parameter_profiles) {
            const auto &direct = direct_server.Require(
                entry.first, hera::PartyRole::kServer);
            const auto &serialized = server_registry.Require(
                entry.first, hera::PartyRole::kServer);
            Require(direct.relinearization_keys.size() == 0 &&
                        direct.galois_keys.size() == 0 &&
                        serialized.relinearization_keys.size() == 0 &&
                        serialized.galois_keys.size() == 0,
                    "importing conversion profiles recreated evaluation keys");
        }
        Require(client_registry.Require("test_profile",
                                        hera::PartyRole::kClient)
                    .has_secret_key(),
                "client profile lost its secret key");
        Require(!server_registry.Require("test_profile",
                                         hera::PartyRole::kServer)
                     .has_secret_key(),
                "server profile obtained a secret key");

        const auto &client_profile =
            client_registry.Require("test_profile", hera::PartyRole::kClient);
        const std::size_t slots = client_profile.slot_count();
        const auto &target_profile =
            client_registry.Require("target_profile", hera::PartyRole::kClient);
        std::vector<std::complex<double>> clear(slots);
        clear[0] = {0.125, -0.25};
        clear[1] = {-1.5, 0.75};
        clear[2] = {2.0, -1.0};
        clear[3] = {-0.0625, 0.03125};
        seal::CKKSEncoder encoder(*client_profile.context);
        seal::Plaintext plaintext;
        encoder.encode(clear, std::ldexp(1.0, 20), plaintext);
        seal::Encryptor encryptor(*client_profile.context,
                                  client_profile.secret_key.value());
        seal::Ciphertext input;
        encryptor.encrypt_symmetric(plaintext, input);
        // Conversion must preserve the actual CKKS input. The low scale20
        // fixture's encryption noise can itself exceed two MPC ULP, so use
        // its decrypted value as the conversion oracle, keeping the existing
        // two-ULP C2M and four-ULP round-trip thresholds.
        seal::Decryptor input_decryptor(*client_profile.context,
                                       client_profile.secret_key.value());
        seal::Plaintext input_plain;
        input_decryptor.decrypt(input, input_plain);
        std::vector<std::complex<double>> reference;
        encoder.decode(input_plain, reference);
        auto real_values = clear;
        for (auto &value : real_values) value = {value.real(), 0};
        seal::Plaintext real_plain;
        encoder.encode(real_values, std::ldexp(1.0, 20), real_plain);
        seal::Ciphertext real_input;
        encryptor.encrypt_symmetric(real_plain, real_input);
        const auto map = ComplexTestMap(slots);
        const auto target_map = ComplexTestMap(target_profile.slot_count());
        hera::SealCipherBatch batch{
            "test_profile", {input}, hera::TensorShape{{8}},
            hera::Layout::kSpatial, map};
        hera::TensorSpec input_he_spec{
            hera::TensorShape{{8}}, hera::Domain::kHeCiphertext,
            hera::Layout::kSpatial, {43, 13}, hera::HeRegionId::kRegion1,
            std::string("test_profile"), 0, 20};
        hera::TensorSpec output_he_spec{
            hera::TensorShape{{8}}, hera::Domain::kHeCiphertext,
            hera::Layout::kSpatial, {43, 13}, hera::HeRegionId::kRegion2,
            std::string("target_profile"), 0, 24};
        hera::TensorSpec mpc_spec{
            hera::TensorShape{{8}}, hera::Domain::kMpcArithmeticShare,
            hera::Layout::kSpatial, {43, 13}, std::nullopt, std::nullopt,
            std::nullopt, std::nullopt};
        hera::ConversionAttributes c2m{
            hera::Domain::kHeCiphertext,
            hera::Domain::kMpcArithmeticShare, std::nullopt,
            hera::Layout::kSpatial, {43, 13}, std::nullopt};
        hera::ConversionAttributes m2c{
            hera::Domain::kMpcArithmeticShare,
            hera::Domain::kHeCiphertext, hera::HeRegionId::kRegion2,
            hera::Layout::kSpatial, {43, 13}, std::size_t{0}};

        const pid_t child = fork();
        Require(child >= 0, "fork failed");
        if (child == 0) {
            close(result_pipe[1]);
            try {
                auto transport = hera::MakeNetIoTransport(
                    hera::PartyRole::kClient, "127.0.0.1", 18150);
                hera::CkksMpcConverter converter(
                    hera::PartyRole::kClient, protocol,
                    client_registry, transport,
                    hera::MakeDeterministicRandomSourceForTesting({9, 8, 7}),
                    "127.0.0.1", 18300, 2, compression);
                hera::CryptoTensorStore store;
                hera::ConversionPackingRegistry packing;
                Require(packing.Register(input_he_spec, map).ok(),
                        "client slot map registration failed");
                Require(packing.Register(output_he_spec, target_map).ok(),
                        "client target slot map registration failed");
                Require(store.PutCiphertext(42, std::nullopt).ok(),
                        "client input marker registration failed");
                hera::SealSciConversionBackend backend(
                    hera::PartyRole::kClient, client_registry, converter,
                    store, packing);
                auto mpc_handle = backend.HeToMpc(
                    hera::TensorHandle{42, input_he_spec}, c2m, mpc_spec);
                Require(mpc_handle.ok(), mpc_handle.status().message());
                auto local_share = store.GetMpcShare(mpc_handle.value().id);
                Require(local_share.ok(), local_share.status().message());
                std::vector<std::uint64_t> server_share(8);
                ReadAll(result_pipe[0], server_share.data(),
                        server_share.size() * sizeof(std::uint64_t));
                const std::uint64_t ring_mask = (std::uint64_t{1} << 43) - 1;
                const auto signed_share = [ring_mask](std::uint64_t value) {
                    value &= ring_mask;
                    return (value & (std::uint64_t{1} << 42))
                        ? static_cast<std::int64_t>(value | ~ring_mask)
                        : static_cast<std::int64_t>(value);
                };
                double maximum_error = 0.0;
                for (std::size_t i = 0; i < 4; ++i) {
                    const auto real = signed_share(
                        local_share.value().local_share[2 * i] +
                        server_share[2 * i]);
                    const auto imag = signed_share(
                        local_share.value().local_share[2 * i + 1] +
                        server_share[2 * i + 1]);
                    maximum_error = std::max(
                        maximum_error,
                        std::abs(static_cast<double>(real) / 8192.0 -
                                 reference[i].real()));
                    maximum_error = std::max(
                        maximum_error,
                        std::abs(static_cast<double>(imag) / 8192.0 -
                                 reference[i].imag()));
                }
                std::cerr << "HERA_C2M_MAX_ULP=" << maximum_error * 8192.0 << '\n';
                Require(maximum_error <= 2.0 / 8192.0,
                        "CkksToMpc exceeds two scale-13 ULP");
                auto output_handle = backend.MpcToHe(
                    mpc_handle.value(), m2c, output_he_spec);
                Require(output_handle.ok(), output_handle.status().message());
                auto client_payload =
                    store.GetCiphertext(output_handle.value().id);
                Require(client_payload.ok() && !client_payload.value(),
                        "client unexpectedly stored MpcToCkks ciphertext");
                std::uint64_t size = 0;
                ReadAll(result_pipe[0], &size, sizeof(size));
                Require(size != 0 && size < (std::uint64_t{1} << 30),
                        "invalid result ciphertext size");
                std::string bytes(size, '\0');
                ReadAll(result_pipe[0], bytes.data(), bytes.size());
                std::stringstream stream(bytes);
                seal::Ciphertext result;
                result.load(*target_profile.context, stream);
                Require(target_profile.context->get_context_data(result.parms_id())
                            ->parms().coeff_modulus().size() == 3,
                        "MpcToCkks did not restore the full target RNS level");
                seal::Decryptor decryptor(*target_profile.context,
                                          target_profile.secret_key.value());
                seal::Plaintext decoded_plain;
                decryptor.decrypt(result, decoded_plain);
                std::vector<std::complex<double>> decoded;
                seal::CKKSEncoder target_encoder(*target_profile.context);
                target_encoder.decode(decoded_plain, decoded);
                maximum_error = 0.0;
                for (std::size_t i = 0; i < 4; ++i) {
                    maximum_error = std::max(
                        maximum_error,
                        std::abs(decoded[i].real() - reference[i].real()));
                    maximum_error = std::max(
                        maximum_error,
                        std::abs(decoded[i].imag() - reference[i].imag()));
                }
                std::cerr << "HERA_CROSS_PROFILE_MAX_ULP="
                          << maximum_error * 8192.0 << std::endl;
                // Retained fractional coefficient bits keep the cross-degree
                // round trip within four scale-13 ULP. One-way C2M retains
                // its independent two-ULP bound above.
                Require(maximum_error <= 4.0 / 8192.0,
                        "complex round-trip exceeds four scale-13 ULP");
                auto multi_shares = converter.CkksToMpc(std::nullopt);
                Require(multi_shares.ok(), multi_shares.status().message());
                Require(multi_shares.value().real.physical_shape ==
                            hera::TensorShape{{3, slots}},
                        "multi-ciphertext conversion lost physical shape");
                hera::MpcToCkksTarget multi_target{
                    "target_profile", 0, 24, hera::TensorShape{{24}},
                    hera::Layout::kSpatial,
                    ComplexTestMap(target_profile.slot_count(), 3)};
                auto multi_output = converter.MpcToCkks(
                    RetargetShares(multi_shares.value(),
                                   ComplexTestMap(slots, 3),
                                   multi_target.slot_map), multi_target);
                Require(multi_output.ok(), multi_output.status().message());
                Require(!multi_output.value().has_value(),
                        "client received multi-ciphertext HE output");
                for (std::size_t ct = 0; ct < 3; ++ct) {
                    std::uint64_t multi_size = 0;
                    ReadAll(result_pipe[0], &multi_size, sizeof(multi_size));
                    Require(multi_size != 0 && multi_size <
                                (std::uint64_t{1} << 30),
                            "invalid multi-ciphertext result size");
                    std::string multi_bytes(multi_size, '\0');
                    ReadAll(result_pipe[0], multi_bytes.data(),
                            multi_bytes.size());
                    std::stringstream multi_stream(multi_bytes);
                    seal::Ciphertext multi_result;
                    multi_result.load(*target_profile.context, multi_stream);
                    seal::Plaintext multi_plain;
                    decryptor.decrypt(multi_result, multi_plain);
                    target_encoder.decode(multi_plain, decoded);
                    double multi_error = 0;
                    for (std::size_t i = 0; i < 4; ++i)
                        multi_error = std::max({multi_error,
                            std::abs(decoded[i].real() - reference[i].real()),
                            std::abs(decoded[i].imag() - reference[i].imag())});
                    std::cerr << "HERA_MULTI_ROUND_TRIP_MAX_ULP="
                              << multi_error * 8192.0 << '\n';
                    Require(multi_error <= 4.0 / 8192.0,
                            "multi-ciphertext round-trip error: " +
                                std::to_string(multi_error * 8192.0) + " ULP");
                }
                // A terminal target has Q smaller than the 94-bit workspace.
                // Exercise general reduction and a different cached correction,
                // complementing the full-RNS target's division-free fast path.
                auto terminal_target = multi_target;
                terminal_target.target_level = 2;
                terminal_target.public_ciphertext_bound_bits = {15, std::nullopt, 15};
                const auto before_mixed = converter.stats();
                auto terminal_output = converter.MpcToCkks(
                    RetargetShares(multi_shares.value(), ComplexTestMap(slots, 3),
                                   terminal_target.slot_map), terminal_target);
                Require(terminal_output.ok(), terminal_output.status().message());
                Require(converter.stats().signed_extension_batches - before_mixed.signed_extension_batches == 6,
                        "mixed public ranges were merged across an intervening ciphertext");
                Require(!terminal_output.value(), "client received terminal HE output");
                double terminal_error = 0.0;
                bool terminal_finite = true;
                for (std::size_t ct = 0; ct < 3; ++ct) {
                    std::uint64_t terminal_size = 0;
                    ReadAll(result_pipe[0], &terminal_size, sizeof(terminal_size));
                    Require(terminal_size != 0 && terminal_size <
                                (std::uint64_t{1} << 30), "invalid terminal output size");
                    std::string terminal_bytes(terminal_size, '\0');
                    ReadAll(result_pipe[0], terminal_bytes.data(), terminal_bytes.size());
                    std::stringstream terminal_stream(terminal_bytes);
                    seal::Ciphertext terminal_result;
                    terminal_result.load(*target_profile.context, terminal_stream);
                    Require(target_profile.context->get_context_data(
                                terminal_result.parms_id())->parms().coeff_modulus().size() == 1,
                            "terminal target did not use a single prime");
                    seal::Plaintext terminal_plain;
                    decryptor.decrypt(terminal_result, terminal_plain);
                    target_encoder.decode(terminal_plain, decoded);
                    for (std::size_t i = 0; i < 4; ++i) {
                        terminal_finite = terminal_finite &&
                            std::isfinite(decoded[i].real()) &&
                            std::isfinite(decoded[i].imag());
                        terminal_error = std::max(terminal_error,
                            std::abs(decoded[i].real() - reference[i].real()));
                        terminal_error = std::max(terminal_error,
                            std::abs(decoded[i].imag() - reference[i].imag()));
                    }
                }
                std::cerr << "HERA_TERMINAL_MAX_ULP="
                          << terminal_error * 8192.0 << std::endl;
                Require(terminal_finite && terminal_error <= 4.0 / 8192.0,
                        "terminal ring-to-field round-trip error");
                TestPublicZeroImaginary(converter, hera::PartyRole::kClient,
                                        target_profile, result_pipe[0]);
                TestPairedRealBoundary(converter, hera::PartyRole::kClient,
                                       client_profile, real_input, result_pipe[0]);
                auto obsolete = converter.CkksToMpc(std::nullopt);
                Require(!obsolete.ok(), "obsolete conversion peer was accepted");
                const auto client_stats = converter.stats();
                Require(client_stats.transform_cache_hits >= 2 &&
                            client_stats.transform_cache_misses == 2 &&
                            client_stats.local_transform_tasks == 19 &&
                            client_stats.local_transform_peak_inflight == 2,
                        "client transform cache or pipeline was not exercised");
                close(result_pipe[0]);
                _exit(0);
            } catch (const std::exception &error) {
                std::cerr << "client: " << error.what() << '\n';
                _exit(2);
            }
        }

        close(result_pipe[0]);
        auto transport = std::make_shared<CapturingTransport>(
            hera::MakeNetIoTransport(hera::PartyRole::kServer, "", 18150));
        auto capture = std::make_shared<CaptureState>();
        auto server_random = std::make_shared<CapturingRandomSource>(
            hera::MakeDeterministicRandomSourceForTesting({5, 4, 3}),
            capture);
        hera::CkksMpcConverter converter(
            hera::PartyRole::kServer, protocol,
            server_registry, transport, server_random, "", 18300, 2, compression);
        hera::CryptoTensorStore store;
        hera::ConversionPackingRegistry packing;
        Require(packing.Register(input_he_spec, map).ok(),
                "server slot map registration failed");
        Require(packing.Register(output_he_spec, target_map).ok(),
                "server target slot map registration failed");
        Require(store.PutCiphertext(42, batch).ok(),
                "server input ciphertext registration failed");
        hera::SealSciConversionBackend backend(
            hera::PartyRole::kServer, server_registry, converter, store,
            packing);
        auto mpc_handle = backend.HeToMpc(
            hera::TensorHandle{42, input_he_spec}, c2m, mpc_spec);
        Require(mpc_handle.ok(), mpc_handle.status().message());
        auto local_share = store.GetMpcShare(mpc_handle.value().id);
        Require(local_share.ok(), local_share.status().message());
        VerifyCombinedCiphertextMask(*transport, *capture, client_profile, input);
        transport->capture = false;
        transport->sent.clear();
        WriteAll(result_pipe[1], local_share.value().local_share.data(),
                 8 * sizeof(std::uint64_t));
        auto output_handle = backend.MpcToHe(
            mpc_handle.value(), m2c, output_he_spec);
        Require(output_handle.ok(), output_handle.status().message());
        auto output = store.GetCiphertext(output_handle.value().id);
        Require(output.ok() && output.value().has_value(),
                "server did not store MpcToCkks ciphertext");
        std::stringstream stream;
        output.value()->ciphertexts.front().save(
            stream, seal::compr_mode_type::none);
        const std::string bytes = stream.str();
        const std::uint64_t size = bytes.size();
        WriteAll(result_pipe[1], &size, sizeof(size));
        WriteAll(result_pipe[1], bytes.data(), bytes.size());
        hera::SealCipherBatch multi_batch{
            "test_profile", {input, input, input},
            hera::TensorShape{{24}}, hera::Layout::kSpatial,
            ComplexTestMap(slots, 3)};
        auto multi_shares = converter.CkksToMpc(multi_batch);
        Require(multi_shares.ok(), multi_shares.status().message());
        hera::MpcToCkksTarget multi_target{
            "target_profile", 0, 24, hera::TensorShape{{24}},
            hera::Layout::kSpatial,
            ComplexTestMap(target_profile.slot_count(), 3)};
        auto multi_output = converter.MpcToCkks(
            RetargetShares(multi_shares.value(),
                           multi_batch.slot_map, multi_target.slot_map),
            multi_target);
        Require(multi_output.ok(), multi_output.status().message());
        Require(multi_output.value().has_value() &&
                    multi_output.value()->ciphertexts.size() == 3,
                "server multi-ciphertext output count mismatch");
        for (const auto &ciphertext : multi_output.value()->ciphertexts) {
            std::stringstream multi_stream;
            ciphertext.save(multi_stream, seal::compr_mode_type::none);
            const std::string multi_bytes = multi_stream.str();
            const std::uint64_t multi_size = multi_bytes.size();
            WriteAll(result_pipe[1], &multi_size, sizeof(multi_size));
            WriteAll(result_pipe[1], multi_bytes.data(), multi_bytes.size());
        }
        auto terminal_target = multi_target;
        terminal_target.target_level = 2;
        terminal_target.public_ciphertext_bound_bits = {15, std::nullopt, 15};
        const auto before_mixed = converter.stats();
        auto terminal_output = converter.MpcToCkks(
            RetargetShares(multi_shares.value(), multi_batch.slot_map,
                           terminal_target.slot_map), terminal_target);
        Require(terminal_output.ok(), terminal_output.status().message());
        Require(converter.stats().signed_extension_batches - before_mixed.signed_extension_batches == 6,
                "mixed public ranges were merged across an intervening ciphertext");
        Require(terminal_output.value() &&
                    terminal_output.value()->ciphertexts.size() == 3,
                "server terminal output count mismatch");
        for (const auto &ciphertext : terminal_output.value()->ciphertexts) {
            std::stringstream terminal_stream;
            ciphertext.save(terminal_stream, seal::compr_mode_type::none);
            const auto terminal_bytes = terminal_stream.str();
            const std::uint64_t terminal_size = terminal_bytes.size();
            WriteAll(result_pipe[1], &terminal_size, sizeof(terminal_size));
            WriteAll(result_pipe[1], terminal_bytes.data(), terminal_bytes.size());
        }
        TestPublicZeroImaginary(converter, hera::PartyRole::kServer,
                                target_profile, result_pipe[1]);
        TestPairedRealBoundary(converter, hera::PartyRole::kServer,
            server_registry.Require("test_profile", hera::PartyRole::kServer), real_input, result_pipe[1]);
        transport->obsolete_frame = true;
        auto obsolete = converter.CkksToMpc(batch);
        Require(!obsolete.ok(), "obsolete conversion frame entered secure work");
        close(result_pipe[1]);
        int status = 0;
        waitpid(child, &status, 0);
        Require(WIFEXITED(status) && WEXITSTATUS(status) == 0,
                "client conversion verification failed");
        const auto stats = converter.stats();
        Require(stats.a_mask_bytes == 6 * 2048 * sizeof(std::uint64_t),
                "A-domain mask accounting mismatch");
        Require(stats.z_mask_bytes == 6 * 2 * slots * sizeof(std::uint64_t),
                "Z-domain mask accounting mismatch");
        Require(stats.transform_cache_hits >= 2 &&
                    stats.transform_cache_misses == 2 &&
                    stats.local_transform_tasks == 19 &&
                    stats.local_transform_peak_inflight == 2,
                "server transform cache or pipeline was not exercised");
        std::cout << "HERA_CONVERSION_PROTOCOL_STATUS=PASS\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "HERA_CONVERSION_PROTOCOL_STATUS=FAIL: " << error.what()
                  << '\n';
        return 1;
    }
}
