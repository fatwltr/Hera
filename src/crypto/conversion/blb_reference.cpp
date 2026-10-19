#include "hera/blb_reference.h"
#include "hera/detail/protocol_worker.h"
#include "hera/detail/he_packing.h"
#include "hera/detail/conversion_lanes.h"
#include "hera/seal_wire.h"
#include "hera/detail/parallel_ranges.h"
#include "hera/detail/worker_ports.h"

#include "fixed_ckks_transform.h"
#include "blb_encoding.h"
#include "blb_ring_protocol.h"
#include "random_sampling.h"

#include "bounded_ot.h"
#include "utils/io_pack.h"

#ifdef I
#undef I
#endif
#ifdef J
#undef J
#endif

#include <seal/util/ntt.h>
#include <seal/util/uintarith.h>
#include <seal/util/uintarithmod.h>
#include <seal/util/uintarithsmallmod.h>
#include <seal/util/uintcore.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstring>
#include <deque>
#include <future>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <tuple>

namespace hera {
namespace {

using Clock = std::chrono::steady_clock;

double MillisecondsSince(Clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

using conversion_internal::FixedCkksTransform;
using conversion_internal::SampleUniformPrime;

constexpr std::uint64_t kFrameMagic = 0x48455241434f4e56ULL;
// Dual-lane real boundaries negotiate lane maps and half-encoding factors.
constexpr std::uint32_t kProtocolVersion = 19;
constexpr std::uint32_t kCkksToMpcDirection = 1;
constexpr std::uint32_t kMpcToCkksDirection = 2;
// Reserved layout bits negotiate public contracts without changing frame size.
constexpr std::uint32_t kPublicZeroImaginary = std::uint32_t{1} << 31;
constexpr std::uint32_t kInvalidPublicContract = std::uint32_t{1} << 30;
constexpr std::uint32_t kCanonicalLanePacking = std::uint32_t{1} << 23;
constexpr int kPublicBoundShift = 24;
constexpr std::uint32_t kC2mQuarterRange = std::uint32_t{1} << 22;
constexpr std::uint32_t kM2cQuarterRange = std::uint32_t{1} << 21;
constexpr std::uint32_t kPreparedSlots = std::uint32_t{1} << 20;
constexpr std::uint32_t kBenchmarkUncheckedRange = std::uint32_t{1} << 19;

struct ProtocolFrame {
    std::uint64_t magic{kFrameMagic};
    std::uint32_t version{kProtocolVersion};
    std::uint32_t direction{};
    std::uint64_t call_sequence{};
    std::uint64_t session_nonce_low{};
    std::uint64_t session_nonce_high{};
    std::uint64_t ciphertext_count{};
    std::uint64_t polynomial_degree{};
    std::uint64_t slots_per_ciphertext{};
    std::uint64_t logical_elements{};
    std::uint64_t shape_fingerprint{};
    std::uint32_t logical_layout{};
    std::uint64_t parameter_fingerprint{};
    std::uint64_t terminal_modulus{};
    std::uint64_t parms_id[4]{};
    std::uint64_t target_level{std::numeric_limits<std::uint64_t>::max()};
    std::int32_t ring_bits{};
    std::int32_t share_scale_bits{};
    std::int32_t he_scale_bits{};
    std::int32_t statistical_bits{};
    std::int32_t fft_scale_bits{};
    std::int32_t coefficient_bound_bits{-1};
    std::uint32_t profile_bytes{};
};

std::uint64_t LowMask(int bits)
{
    return bits == 64 ? std::numeric_limits<std::uint64_t>::max()
                      : (std::uint64_t{1} << bits) - 1;
}

int ScaleBits(double scale)
{
    if (!std::isfinite(scale) || scale <= 0.0)
        throw std::invalid_argument("ciphertext scale is not finite and positive");
    const double bits = std::log2(scale);
    const int rounded = static_cast<int>(std::llround(bits));
    if (std::abs(bits - rounded) > 1e-6)
        throw std::invalid_argument(
            "conversion requires a ciphertext scale close to a power of two");
    return rounded;
}

bool ProfileAllowsScale(const HeParameterConfig &profile, int bits)
{
    return bits == profile.default_scale_bits ||
           bits == profile.input_scale_bits || bits == profile.output_scale_bits;
}

using detail::ProtocolWorkerThread;

bool SameFrame(const ProtocolFrame &left, const ProtocolFrame &right)
{
    return left.magic == right.magic && left.version == right.version &&
           left.direction == right.direction &&
           left.call_sequence == right.call_sequence &&
           left.session_nonce_low == right.session_nonce_low &&
           left.session_nonce_high == right.session_nonce_high &&
           left.ciphertext_count == right.ciphertext_count &&
           left.polynomial_degree == right.polynomial_degree &&
           left.slots_per_ciphertext == right.slots_per_ciphertext &&
           left.logical_elements == right.logical_elements &&
           left.shape_fingerprint == right.shape_fingerprint &&
           left.logical_layout == right.logical_layout &&
           left.parameter_fingerprint == right.parameter_fingerprint &&
           left.terminal_modulus == right.terminal_modulus &&
           std::equal(std::begin(left.parms_id), std::end(left.parms_id),
                      std::begin(right.parms_id)) &&
           left.target_level == right.target_level &&
           left.ring_bits == right.ring_bits &&
           left.share_scale_bits == right.share_scale_bits &&
           left.he_scale_bits == right.he_scale_bits &&
           left.statistical_bits == right.statistical_bits &&
           left.fft_scale_bits == right.fft_scale_bits &&
           left.coefficient_bound_bits == right.coefficient_bound_bits &&
           left.profile_bytes == right.profile_bytes;
}

std::uint64_t ShapeFingerprint(const TensorShape &shape)
{
    std::uint64_t hash = 1469598103934665603ULL;
    for (const auto dimension : shape.dimensions) {
        hash ^= dimension;
        hash *= 1099511628211ULL;
    }
    hash ^= shape.dimensions.size();
    return hash * 1099511628211ULL;
}

std::uint64_t ParameterFingerprint(const seal::SEALContext &context)
{
    const auto &parms = context.key_context_data()->parms();
    std::uint64_t hash = 1469598103934665603ULL;
    const auto mix = [&hash](std::uint64_t value) {
        hash ^= value;
        hash *= 1099511628211ULL;
    };
    mix(parms.poly_modulus_degree());
    for (const auto &modulus : parms.coeff_modulus()) mix(modulus.value());
    return hash;
}

void SetParmsId(ProtocolFrame &frame, seal::parms_id_type id)
{
    std::copy(id.begin(), id.end(), std::begin(frame.parms_id));
}

std::string SessionLabel(const ProtocolFrame &frame)
{
    std::ostringstream stream;
    stream << std::hex << std::setfill('0') << std::setw(16)
           << frame.session_nonce_high << std::setw(16)
           << frame.session_nonce_low;
    return stream.str();
}

void SendString(IProtocolTransport &transport, const std::string &value)
{
    if (!value.empty()) transport.Send(value.data(), value.size());
}

std::string ReceiveString(IProtocolTransport &transport, std::size_t bytes)
{
    if (bytes > 4096) throw std::runtime_error("protocol profile name is too long");
    std::string result(bytes, '\0');
    if (bytes != 0) transport.Receive(result.data(), bytes);
    return result;
}

seal::parms_id_type ResolveTargetParmsId(const HeContext &profile,
                                         std::optional<std::size_t> level)
{
    auto data = profile.context->first_context_data();
    std::size_t current = 0;
    const std::size_t wanted = level.value_or(0);
    while (data && current < wanted) {
        data = data->next_context_data();
        ++current;
    }
    if (!data) throw std::invalid_argument("target CKKS level does not exist");
    return data->parms_id();
}

} // namespace

class BlbReferenceConverter::Impl {
public:
    struct Worker {
        int effective_party{};
        std::unique_ptr<sci::IOPack> io;
        std::array<sci::NetIO *,1> ot_channels{};
        std::unique_ptr<conversion_internal::ConversionOt> ot;
        // Destroy/join the executor before releasing the SCI objects it uses.
        std::unique_ptr<ProtocolWorkerThread> executor;
    };

    using TransformKey = std::tuple<std::size_t, int, int, bool>;

    Result<std::shared_ptr<const FixedCkksTransform>> GetTransform(
        std::size_t degree, int root_scale_bits, int working_bits)
    {
        const TransformKey key{degree, root_scale_bits, working_bits,
                               role_ == PartyRole::kClient};
        const auto found = transforms_.find(key);
        if (found != transforms_.end()) {
            ++stats_.transform_cache_hits;
            return found->second;
        }
        auto created = FixedCkksTransform::Create(
            degree, root_scale_bits, working_bits,
            role_ == PartyRole::kClient);
        if (!created.ok()) return created.status();
        auto shared = std::make_shared<const FixedCkksTransform>(
            created.take_value());
        transforms_.emplace(key, shared);
        ++stats_.transform_cache_misses;
        return shared;
    }

    Impl(PartyRole role, ConversionConfig config, HeContextRegistry &contexts,
         std::shared_ptr<IProtocolTransport> transport,
         std::shared_ptr<IRandomSource> randomness,
         std::string peer_address, int worker_port, std::size_t threads,
         seal::compr_mode_type wire_compression)
        : role_(role), config_(config), contexts_(contexts),
          transport_(std::move(transport)), randomness_(std::move(randomness)),
          wire_compression_(wire_compression)
    {
        if (!transport_ || !randomness_)
            throw std::invalid_argument("converter transport/random source is null");
        if (!seal::Serialization::IsSupportedComprMode(wire_compression_))
            throw std::invalid_argument(
                "requested SEAL compression is unavailable in this build");
        if (!contexts_.initialized() || contexts_.role() != role_)
            throw std::invalid_argument("converter context registry role mismatch");
        if (config_.ring_bits < 2 || config_.ring_bits > 63)
            throw std::invalid_argument("conversion ring bits must be in [2,63]");
        if (config_.statistical_security_bits < 1 ||
            config_.statistical_security_bits > 63)
            throw std::invalid_argument("statistical bits must be in [1,63]");
        if (config_.ring_bits + config_.statistical_security_bits > 127)
            throw std::invalid_argument("wide conversion ring exceeds 127 bits");
        detail::ValidateWorkerPortRange(worker_port,
            detail::ConversionWorkerPortSpan(threads));
        if (config_.batch_elements == 0)
            config_.batch_elements = ConversionConfig::DefaultBatchElements(threads);
        if (config_.batch_elements > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::invalid_argument("conversion batch exceeds SCI's integer size range");
        const int physical_party = role_ == PartyRole::kServer ? sci::ALICE
                                                               : sci::BOB;
        workers_.reserve(threads);
        for (std::size_t i = 0; i < threads; ++i) {
            auto worker = std::make_unique<Worker>();
            worker->effective_party = (i & 1) ? 3 - physical_party
                                               : physical_party;
            const int port = worker_port + static_cast<int>(3 * i);
            worker->io = std::make_unique<sci::IOPack>(
                physical_party, port,
                role_ == PartyRole::kServer ? std::string{} : peer_address);
            if (i == 0) {
                // Reject different pool sizes/batch boundaries before opening
                // the extra sockets or consuming stateful OT streams.
                const std::uint64_t contract[] = {kFrameMagic, kProtocolVersion,
                    threads, config_.batch_elements,
                    static_cast<std::uint64_t>(config_.public_c2m_quarter_range) |
                        (static_cast<std::uint64_t>(config_.benchmark_unchecked_range) << 1),
                    config_.public_m2c_quarter_range};
                std::uint64_t peer[6]{};
                if (role_ == PartyRole::kServer) {
                    worker->io->io->send_data(contract, sizeof(contract));
                    worker->io->io->recv_data(peer, sizeof(peer));
                } else {
                    worker->io->io->recv_data(peer, sizeof(peer));
                    worker->io->io->send_data(contract, sizeof(contract));
                }
                if (!std::equal(std::begin(contract), std::end(contract), std::begin(peer)))
                    throw std::invalid_argument("conversion worker/batch contracts differ");
            }
            worker->ot_channels[0] = worker->io->io;
            worker->ot = std::make_unique<conversion_internal::ConversionOt>(
                worker->effective_party, 1, worker->ot_channels.data(), false, true, "");
            workers_.push_back(std::move(worker));
        }
    }

    const ConversionConfig &config() const noexcept { return config_; }

    ConversionStats stats() const noexcept
    {
        ConversionStats result = stats_;
        result.communication_bytes = transport_->bytes();
        result.rounds = transport_->rounds();
        for (const auto &worker : workers_) {
            result.communication_bytes += worker->io->get_comm();
            result.rounds += worker->io->get_rounds();
            result.forward_ot_capacity += worker->ot->ferret->ot_limit;
            result.forward_ot_used += worker->ot->ferret->ot_used;
        }
        return result;
    }

    std::uint64_t RingBytes() const noexcept
    {
        std::uint64_t bytes = 0;
        for (const auto &worker : workers_) bytes += worker->io->get_comm();
        return bytes;
    }

    std::size_t CiphertextBatchSize(std::size_t degree, std::size_t count) const
    {
        return std::min(count, std::max(std::size_t{1}, config_.batch_elements / degree));
    }

    void Cancel() noexcept
    {
        if (transport_) transport_->Cancel();
        for (auto &worker : workers_)
            if (worker && worker->io) worker->io->cancel();
    }

    template <typename Operation>
    void Parallel(std::size_t size, Operation operation)
    {
        if (size == 0) return;
        std::vector<std::future<void>> futures;
        futures.reserve(workers_.size());
        std::exception_ptr first_error;
        try {
            for (std::size_t id = 0; id < workers_.size(); ++id) {
                const std::size_t begin = size * id / workers_.size();
                const std::size_t end = size * (id + 1) / workers_.size();
                if (begin == end) continue;
                auto &worker = *workers_[id];
                if (!worker.executor)
                    worker.executor = std::make_unique<ProtocolWorkerThread>();
                futures.push_back(worker.executor->Submit([&, id, begin, end] {
                    try {
                        operation(*workers_[id], begin, end);
                    } catch (...) {
                        // Wake peers blocked in OT before awaiting any futures.
                        Cancel();
                        throw;
                    }
                }));
            }
        } catch (...) {
            first_error = std::current_exception();
            Cancel();
        }
        for (auto &future : futures) {
            try { future.get(); }
            catch (...) { if (!first_error) first_error = std::current_exception(); }
        }
        if (first_error) std::rethrow_exception(first_error);
    }

    // The wrapper selects a PUBLIC range mode and schedules workers. The
    // actual field/ring protocols live in blb_ring_protocol.cpp, independently
    // of model, HE framing, packing and performance counters.
    std::vector<WideRingShare> CenteredFieldToRing(
        const std::vector<std::uint64_t> &field_share, std::uint64_t modulus,
        int output_bits, std::optional<int> public_bound_bits, bool public_quarter_range)
    {
        const int field_bits = seal::util::get_significant_bit_count(modulus);
        if (output_bits < field_bits || output_bits > 127)
            throw std::invalid_argument("invalid centered field-to-ring width");
        ++stats_.field_to_ring_batches;
        stats_.max_field_to_ring_elements = std::max(stats_.max_field_to_ring_elements, field_share.size());
        const bool bounded = config_.benchmark_unchecked_range || public_quarter_range || (public_bound_bits &&
            *public_bound_bits >= 0 && *public_bound_bits <= 62 &&
            (std::uint64_t{1} << *public_bound_bits) <= modulus / 4);
        if (!bounded) throw std::invalid_argument("BLB C2M requires a public quarter-range contract");
        stats_.bounded_field_to_ring_elements += field_share.size();
        const auto before_bytes = RingBytes();
        std::vector<WideRingShare> output(field_share.size());
        Parallel(field_share.size(), [&](Worker &worker, std::size_t begin, std::size_t end) {
            conversion_internal::blb::RingProtocol protocol(worker.effective_party, *worker.ot);
            std::vector<std::uint64_t> local(field_share.begin() + begin, field_share.begin() + end);
            auto converted = protocol.FieldToRing(local, modulus, output_bits);
            std::copy(converted.begin(), converted.end(), output.begin() + begin);
        });
        stats_.field_to_ring_communication_bytes += RingBytes() - before_bytes;
        return output;
    }

    std::vector<WideRingShare> SignedExtend(
        const std::vector<std::uint64_t> &shares, int input_bits,
        int output_bits, std::optional<int> public_bound_bits = std::nullopt)
    {
        if (input_bits < 2 || input_bits > 63 || output_bits <= input_bits ||
            output_bits > 127 || output_bits - input_bits > 63)
            throw std::invalid_argument("unsupported secure signed extension");
        ++stats_.signed_extension_batches;
        stats_.max_signed_extension_elements = std::max(stats_.max_signed_extension_elements, shares.size());
        const bool bounded = config_.benchmark_unchecked_range || config_.public_m2c_quarter_range || (public_bound_bits &&
            *public_bound_bits >= 0 && *public_bound_bits <= input_bits - 2);
        if (!bounded) throw std::invalid_argument("BLB M2C requires a public quarter-range contract");
        stats_.bounded_signed_extension_elements += shares.size();
        const auto before_bytes = RingBytes();
        std::vector<WideRingShare> output(shares.size());
        Parallel(shares.size(), [&](Worker &worker, std::size_t begin, std::size_t end) {
            conversion_internal::blb::RingProtocol protocol(worker.effective_party, *worker.ot);
            std::vector<std::uint64_t> local(shares.begin() + begin, shares.begin() + end);
            auto extended = protocol.ExtendSigned(local, input_bits, output_bits);
            std::copy(extended.begin(), extended.end(), output.begin() + begin);
        });
        stats_.signed_extension_communication_bytes += RingBytes() - before_bytes;
        return output;
    }

    std::shared_ptr<const std::vector<std::uint64_t>> GetRnsWrapCorrection(
        int bits, const seal::SEALContext::ContextData &context_data)
    {
        const auto key = std::make_pair(context_data.parms_id(), bits);
        const auto found = rns_wrap_corrections_.find(key);
        if (found != rns_wrap_corrections_.end()) return found->second;
        auto correction = std::make_shared<const std::vector<std::uint64_t>>(
            conversion_internal::blb::RnsWrapCorrection(bits, context_data));
        rns_wrap_corrections_.emplace(key, correction);
        return correction;
    }

    seal::Plaintext BuildBlbRnsPlaintextShare(
        const std::vector<WideRingShare> &coefficients, int bits,
        const seal::SEALContext::ContextData &context_data,
        const std::vector<std::uint64_t> &wrap_correction)
    {
        return conversion_internal::blb::ProjectExtendedRingToRns(
            coefficients, bits, context_data, wrap_correction, role_ == PartyRole::kServer);
    }

    Result<ComplexMpcSharePair> CkksToMpc(
        const std::optional<SealCipherBatch> &server_input,
        std::optional<int> output_scale_bits,
        std::optional<int> public_coefficient_bound_bits)
    {
        const auto started = Clock::now();
        try {
            const int share_scale_bits = output_scale_bits.value_or(config_.scale_bits);
            if (share_scale_bits < 0 || share_scale_bits >= config_.ring_bits - 1)
                return Status::Invalid("CkksToMpc output precision does not fit the share ring");
            // Higher-precision variance needs finer FFT roots too. Keep the
            // 127-bit/statistical-security bound; ordinary calls stay unchanged.
            const auto fft_bits_for = [&](const HeContext &context) {
                const auto modulus = context.context->last_context_data()->parms().coeff_modulus().front().value();
                const int available = 127 - seal::util::get_significant_bit_count(modulus) - config_.statistical_security_bits - 2;
                const int wanted = share_scale_bits > config_.scale_bits
                    ? std::max(config_.fft_scale_bits, share_scale_bits + 8) : config_.fft_scale_bits;
                return std::min(available, wanted);
            };
            if ((role_ == PartyRole::kServer) != server_input.has_value())
                return Status::Invalid(
                    "only the server supplies CkksToMpc ciphertexts");

            ProtocolFrame frame;
            std::string profile_name;
            const HeContext *profile = nullptr;
            std::vector<seal::Ciphertext> ciphertexts;
            auto handshake_started = started;
            if (role_ == PartyRole::kServer) {
                const auto map_status = server_input->slot_map.Validate();
                if (!map_status.ok()) return map_status.status();
                const auto logical_count =
                    server_input->logical_shape.ElementCount();
                if (!logical_count.ok()) return logical_count.status();
                if (logical_count.value() !=
                    server_input->slot_map.logical_element_count)
                    return Status::Invalid(
                        "ciphertext logical shape/slot map mismatch");
                profile_name = server_input->parameter_profile;
                profile = &contexts_.Require(profile_name, role_);
                if (server_input->ciphertexts.empty() ||
                    server_input->ciphertexts.size() !=
                        server_input->slot_map.ciphertext_count)
                    return Status::Invalid("ciphertext batch/map count mismatch");
                ciphertexts = server_input->ciphertexts;
                const auto terminal = profile->context->last_context_data();
                if (!terminal ||
                    terminal->parms().coeff_modulus().size() != 1)
                    return Status::FailedPrecondition(
                        "profile has no single-prime terminal CKKS level");
                seal::Evaluator evaluator(*profile->context);
                for (auto &ciphertext : ciphertexts) {
                    if (!profile->context->get_context_data(ciphertext.parms_id()))
                        return Status::Invalid("ciphertext has the wrong SEAL context");
                    if (ciphertext.parms_id() != terminal->parms_id()) {
                        if (!config_.allow_internal_mod_switch)
                            return Status::FailedPrecondition(
                                "CkksToMpc input is not at terminal level");
                        evaluator.mod_switch_to_inplace(ciphertext,
                                                        terminal->parms_id());
                        ++stats_.internal_mod_switches;
                    }
                    if (!ciphertext.is_ntt_form())
                        return Status::Invalid(
                            "CkksToMpc requires an NTT-form CKKS ciphertext");
                }
                const int he_scale = ScaleBits(ciphertexts.front().scale());
                if (!ProfileAllowsScale(profile->parameters, he_scale))
                    return Status::FailedPrecondition(
                        "CkksToMpc ciphertext scale is outside its HE profile");
                for (const auto &ciphertext : ciphertexts)
                    if (ScaleBits(ciphertext.scale()) != he_scale)
                        return Status::Invalid(
                            "CkksToMpc ciphertext scales are inconsistent");
                frame.direction = kCkksToMpcDirection;
                frame.call_sequence = call_sequence_++;
                auto session_stream = randomness_->OpenStream(
                    "hera/session/" + std::to_string(frame.call_sequence));
                if (!session_stream.ok()) return session_stream.status();
                session_stream.value()->Fill(&frame.session_nonce_low,
                                              sizeof(frame.session_nonce_low));
                session_stream.value()->Fill(&frame.session_nonce_high,
                                              sizeof(frame.session_nonce_high));
                frame.ciphertext_count = ciphertexts.size();
                frame.polynomial_degree = profile->parameters.polynomial_degree;
                frame.slots_per_ciphertext = profile->slot_count();
                frame.logical_elements =
                    server_input->slot_map.logical_element_count;
                frame.shape_fingerprint =
                    ShapeFingerprint(server_input->logical_shape);
                frame.logical_layout = static_cast<std::uint32_t>(
                    server_input->logical_layout);
                if (config_.public_c2m_quarter_range) frame.logical_layout |= kC2mQuarterRange;
                if (config_.benchmark_unchecked_range) frame.logical_layout |= kBenchmarkUncheckedRange;
                frame.parameter_fingerprint =
                    ParameterFingerprint(*profile->context);
                frame.terminal_modulus =
                    terminal->parms().coeff_modulus().front().value();
                SetParmsId(frame, terminal->parms_id());
                frame.ring_bits = config_.ring_bits;
                frame.share_scale_bits = share_scale_bits;
                frame.he_scale_bits = he_scale;
                frame.statistical_bits = config_.statistical_security_bits;
                frame.fft_scale_bits = fft_bits_for(*profile);
                // -1 denotes ABSENT only. Encode an explicitly invalid -1
                // differently so a peer with no contract cannot accept it.
                frame.coefficient_bound_bits = public_coefficient_bound_bits
                    ? (*public_coefficient_bound_bits >= 0 && *public_coefficient_bound_bits <= 62
                        ? *public_coefficient_bound_bits : -2) : -1;
                if (!config_.benchmark_unchecked_range && !config_.public_c2m_quarter_range &&
                    (!public_coefficient_bound_bits || *public_coefficient_bound_bits < 0 ||
                     *public_coefficient_bound_bits > 62 ||
                     (std::uint64_t{1} << *public_coefficient_bound_bits) > frame.terminal_modulus / 4))
                    frame.logical_layout |= kInvalidPublicContract;
                frame.profile_bytes = static_cast<std::uint32_t>(profile_name.size());
                handshake_started = Clock::now();
                stats_.ckks_to_mpc.preparation +=
                    std::chrono::duration<double, std::milli>(
                        handshake_started - started).count();
                transport_->Send(&frame, sizeof(frame));
                SendString(*transport_, profile_name);
                std::uint32_t acknowledged = 0;
                transport_->Receive(&acknowledged, sizeof(acknowledged));
                if (acknowledged != 1)
                    return Status::FailedPrecondition(
                        "client rejected the CkksToMpc handshake");
            } else {
                transport_->Receive(&frame, sizeof(frame));
                if (frame.magic != kFrameMagic ||
                    frame.version != kProtocolVersion ||
                    frame.direction != kCkksToMpcDirection) {
                    const std::uint32_t rejected = 0;
                    transport_->Send(&rejected, sizeof(rejected));
                    return Status::FailedPrecondition("invalid CkksToMpc frame");
                }
                profile_name = ReceiveString(*transport_, frame.profile_bytes);
                if (frame.call_sequence != call_sequence_++) {
                    const std::uint32_t rejected = 0;
                    transport_->Send(&rejected, sizeof(rejected));
                    return Status::FailedPrecondition(
                        "CkksToMpc call sequence mismatch");
                }
                try {
                    profile = &contexts_.Require(profile_name, role_);
                } catch (...) {
                    const std::uint32_t rejected = 0;
                    transport_->Send(&rejected, sizeof(rejected));
                    throw;
                }
                if (frame.polynomial_degree !=
                        profile->parameters.polynomial_degree ||
                    frame.slots_per_ciphertext != profile->slot_count() ||
                    frame.logical_elements == 0 ||
                    frame.logical_elements >
                        2 * frame.ciphertext_count *
                            frame.slots_per_ciphertext ||
                    frame.parameter_fingerprint !=
                        ParameterFingerprint(*profile->context) ||
                    frame.terminal_modulus !=
                        profile->context->last_context_data()
                            ->parms().coeff_modulus().front().value() ||
                    !std::equal(std::begin(frame.parms_id),
                                std::end(frame.parms_id),
                                profile->context->last_parms_id().begin()) ||
                    ((frame.logical_layout & kC2mQuarterRange) != 0) != config_.public_c2m_quarter_range ||
                    ((frame.logical_layout & kBenchmarkUncheckedRange) != 0) != config_.benchmark_unchecked_range ||
                    (frame.logical_layout & kInvalidPublicContract) != 0 ||
                    (!config_.benchmark_unchecked_range && !config_.public_c2m_quarter_range &&
                     (!public_coefficient_bound_bits || *public_coefficient_bound_bits < 0 ||
                      *public_coefficient_bound_bits > 62 ||
                      (std::uint64_t{1} << *public_coefficient_bound_bits) > frame.terminal_modulus / 4)) ||
                    frame.ring_bits != config_.ring_bits ||
                    frame.share_scale_bits != share_scale_bits ||
                    frame.coefficient_bound_bits != public_coefficient_bound_bits.value_or(-1) ||
                    (public_coefficient_bound_bits &&
                        (*public_coefficient_bound_bits < 0 || *public_coefficient_bound_bits > 62)) ||
                    frame.statistical_bits !=
                        config_.statistical_security_bits ||
                    frame.fft_scale_bits != fft_bits_for(*profile)) {
                    const std::uint32_t rejected = 0;
                    transport_->Send(&rejected, sizeof(rejected));
                    return Status::FailedPrecondition(
                        "CkksToMpc profile/protocol handshake mismatch");
                }
                const std::uint32_t acknowledged = 1;
                transport_->Send(&acknowledged, sizeof(acknowledged));
            }
            stats_.ckks_to_mpc.handshake += MillisecondsSince(handshake_started);

            const std::size_t degree = frame.polynomial_degree;
            const std::size_t slots = frame.slots_per_ciphertext;
            const std::size_t count = frame.ciphertext_count;
            const auto terminal = profile->context->last_context_data();
            const std::uint64_t modulus =
                terminal->parms().coeff_modulus().front().value();
            const int field_bits = seal::util::get_significant_bit_count(modulus);
            const int fft_bits = frame.fft_scale_bits;
            const int wide_bits = field_bits + config_.statistical_security_bits;
            if (wide_bits + fft_bits + 2 > 127)
                return Status::FailedPrecondition(
                    "CkksToMpc field/FFT workspace exceeds 127 bits");
            const int guard_bits = std::min(
                20, 127 - wide_bits - fft_bits - 2);
            auto transform_result = GetTransform(
                degree, fft_bits, wide_bits + guard_bits);
            if (!transform_result.ok()) return transform_result.status();
            const auto transform = transform_result.take_value();

            const int scale_shift = frame.he_scale_bits - share_scale_bits;
            if (scale_shift < 0)
                return Status::FailedPrecondition(
                    "CkksToMpc cannot increase the target share scale");
            auto coefficient_bound = public_coefficient_bound_bits;
            if (config_.public_c2m_quarter_range)
                coefficient_bound = coefficient_bound
                    ? std::min(*coefficient_bound, field_bits - 2) : field_bits - 2;
            // Simulation output is not a proved producer bound, even when
            // the caller happened to supply calibration metadata.
            if (config_.benchmark_unchecked_range) coefficient_bound.reset();
            auto slot_bound = coefficient_bound ? transform->DecodedSlotBoundBits(
                *coefficient_bound, guard_bits, scale_shift) : std::nullopt;
            if (slot_bound && *slot_bound >= config_.ring_bits) slot_bound.reset();
            std::vector<std::uint64_t> output_real(count * slots);
            std::vector<std::uint64_t> output_imag(count * slots);
            struct LocalDecoded {
                std::vector<std::uint64_t> real;
                std::vector<std::uint64_t> imag;
                double milliseconds{};
            };
            std::deque<std::pair<std::size_t,
                                 std::future<Result<LocalDecoded>>>> pending;
            const std::size_t max_inflight =
                std::max<std::size_t>(1, std::min({count, workers_.size(),
                    detail::HeWorkerPool::Instance().MemoryBudget() /
                        detail::EstimatedHeWorkerBytes(degree, 1, 16)}));
            const auto collect_decoded = [&]() -> Result<void> {
                auto item = std::move(pending.front());
                pending.pop_front();
                auto decoded = item.second.get();
                if (!decoded.ok()) return decoded.status();
                stats_.ckks_to_mpc.slot_decode += decoded.value().milliseconds;
                const std::size_t offset = item.first * slots;
                std::copy(decoded.value().real.begin(),
                          decoded.value().real.end(),
                          output_real.begin() + offset);
                std::copy(decoded.value().imag.begin(),
                          decoded.value().imag.end(),
                          output_imag.begin() + offset);
                return {};
            };
            seal::Evaluator evaluator(*profile->context);
            std::unique_ptr<seal::Decryptor> decryptor;
            if (role_ == PartyRole::kClient)
                decryptor = std::make_unique<seal::Decryptor>(
                    *profile->context, profile->secret_key.value());

            // Fixed public batches bound memory independently of tensor size.
            const auto kCiphertextBatch = CiphertextBatchSize(degree, count);
            for (std::size_t base = 0; base < count; base += kCiphertextBatch) {
                const auto batch = std::min(kCiphertextBatch, count - base);
                std::vector<std::uint64_t> fields(batch * degree);
                for (std::size_t ct_index = base; ct_index < base + batch; ++ct_index) {
                    std::vector<std::uint64_t> field_share(degree);
                    if (role_ == PartyRole::kServer) {
                        const auto sampling_started = Clock::now();
                        const std::string domain = "hera/c2m/A/" +
                            SessionLabel(frame) + "/" +
                            std::to_string(ct_index);
                        auto stream_result = randomness_->OpenStream(domain);
                        if (!stream_result.ok()) return stream_result.status();
                        auto mask = SampleUniformPrime(*stream_result.value(), degree,
                                                       modulus);
                        stats_.a_mask_bytes += degree * sizeof(std::uint64_t);

                        // A single uniform coefficient mask is sufficient:
                        // adding an independently encoded Z mask leaves its
                        // distribution unchanged, and Z has no later use.
                        stats_.ckks_to_mpc.sampling += MillisecondsSince(sampling_started);
                        const auto he_started = Clock::now();
                        seal::Plaintext mask_plain(degree);
                        std::copy(mask.begin(), mask.end(), mask_plain.data());
                        seal::util::ntt_negacyclic_harvey(
                            mask_plain.data(), terminal->small_ntt_tables()[0]);
                        mask_plain.parms_id() = terminal->parms_id();
                        mask_plain.scale() = ciphertexts[ct_index].scale();
                        auto &masked = ciphertexts[ct_index];
                        evaluator.add_plain_inplace(masked, mask_plain);
                        stats_.ckks_to_mpc.he += MillisecondsSince(he_started);
                        const auto send_started = Clock::now();
                        SendSealObject(*transport_, masked, wire_compression_);
                        stats_.ckks_to_mpc.ciphertext_transport +=
                            MillisecondsSince(send_started);
                        for (std::size_t i = 0; i < degree; ++i)
                            field_share[i] = mask[i] == 0 ? 0 : modulus - mask[i];
                    } else {
                        // This is the only decrypt call in the protocol. The
                        // received ciphertext is hidden by the uniform coefficient mask.
                        const auto receive_started = Clock::now();
                        auto masked = ReceiveSealObject<seal::Ciphertext>(
                            *transport_, *profile->context);
                        stats_.ckks_to_mpc.ciphertext_transport +=
                            MillisecondsSince(receive_started);
                        if (masked.parms_id() != terminal->parms_id())
                            return Status::FailedPrecondition(
                                "received masked ciphertext is not terminal-level");
                        seal::Plaintext decrypted;
                        const auto he_started = Clock::now();
                        decryptor->decrypt(masked, decrypted);
                        seal::util::inverse_ntt_negacyclic_harvey(
                            decrypted.data(), terminal->small_ntt_tables()[0]);
                        std::copy(decrypted.data(), decrypted.data() + degree,
                                  field_share.begin());
                        stats_.ckks_to_mpc.he += MillisecondsSince(he_started);
                    }

                    std::copy(field_share.begin(), field_share.end(),
                              fields.begin() + (ct_index - base) * degree);
                }
                const auto field_started = Clock::now();
                auto batch_wide = CenteredFieldToRing(fields, modulus, wide_bits,
                                                       public_coefficient_bound_bits,
                                                       config_.public_c2m_quarter_range);
                stats_.ckks_to_mpc.field_to_ring += MillisecondsSince(field_started);
                for (std::size_t ct_index = base; ct_index < base + batch; ++ct_index) {
                    const auto offset = (ct_index - base) * degree;
                    std::vector<WideRingShare> wide(batch_wide.begin() + offset,
                                                    batch_wide.begin() + offset + degree);
                    const auto local_work =
                        [transform, wide = std::move(wide), slots,
                         guard_bits, wide_bits, scale_shift,
                         ring_bits = config_.ring_bits,
                         negative = role_ == PartyRole::kClient]() mutable
                        -> Result<LocalDecoded> {
                            const auto decode_started = Clock::now();
                            for (auto &value : wide)
                                value = WideShiftLeft(
                                    value, guard_bits, wide_bits + guard_bits);
                            auto decoded = transform->Decode(wide);
                            if (!decoded.ok()) return decoded.status();
                            LocalDecoded result;
                            result.real.resize(slots);
                            result.imag.resize(slots);
                            const auto mask = LowMask(ring_bits);
                            for (std::size_t slot = 0; slot < slots; ++slot) {
                                result.real[slot] = WideLocalTruncate(
                                    decoded.value().real[slot],
                                    scale_shift + guard_bits,
                                    wide_bits + guard_bits, negative).low & mask;
                                result.imag[slot] = WideLocalTruncate(
                                    decoded.value().imag[slot],
                                    scale_shift + guard_bits,
                                    wide_bits + guard_bits, negative).low & mask;
                            }
                            result.milliseconds = MillisecondsSince(decode_started);
                            return result;
                        };
                    pending.emplace_back(
                        ct_index,
                        detail::SubmitLocalTask(max_inflight, std::move(local_work)));
                    ++stats_.local_transform_tasks;
                    stats_.local_transform_peak_inflight = std::max(
                        stats_.local_transform_peak_inflight, pending.size());
                    if (pending.size() >= max_inflight) {
                        auto collected = collect_decoded();
                        if (!collected.ok()) return collected.status();
                    }
                }
            }
            while (!pending.empty()) {
                auto collected = collect_decoded();
                if (!collected.ok()) return collected.status();
            }

            const TensorShape physical_shape{{count, slots}};
            const NumericFormat numeric{config_.ring_bits, share_scale_bits};
            ComplexMpcSharePair result{
                {std::move(output_real), physical_shape, numeric,
                 ComplexComponent::kReal, slot_bound},
                {std::move(output_imag), physical_shape, numeric,
                 ComplexComponent::kImaginary, slot_bound}};
            stats_.milliseconds += std::chrono::duration<double, std::milli>(
                Clock::now() - started).count();
            return result;
        } catch (const std::exception &error) {
            return Status::Internal(std::string("CkksToMpc failed: ") +
                                    error.what());
        }
    }

    Result<std::optional<SealCipherBatch>> MpcToCkks(
        const ComplexMpcSharePair &local_shares,
        const MpcToCkksTarget &target,
        const std::vector<bool> &half_values = {}, bool canonical_padding = false,
        const PreparedBlbSlots *wide_input = nullptr)
    {
        const auto started = Clock::now();
        try {
            auto shares_status = local_shares.Validate();
            if (!shares_status.ok()) return shares_status.status();
            auto map_status = target.slot_map.Validate();
            if (!map_status.ok()) return map_status.status();
            auto target_count = target.logical_shape.ElementCount();
            if (!target_count.ok()) return target_count.status();
            if (target_count.value() != target.slot_map.logical_element_count)
                return Status::Invalid(
                    "MpcToCkks logical shape/slot map mismatch");
            if (local_shares.real.numeric.ring_bits != config_.ring_bits ||
                local_shares.real.numeric.scale_bits != config_.scale_bits)
                return Status::Invalid("MpcToCkks input numeric format mismatch");

            const HeContext &profile = contexts_.Require(target.parameter_profile,
                                                         role_);
            const auto target_id = ResolveTargetParmsId(profile,
                                                        target.target_level);
            const auto context_data = profile.context->get_context_data(target_id);
            if (!context_data)
                return Status::Invalid("MpcToCkks target parms_id is invalid");
            const std::size_t degree = profile.parameters.polynomial_degree;
            const std::size_t slots = profile.slot_count();
            const std::size_t count = target.slot_map.ciphertext_count;
            if (target.slot_map.slots_per_ciphertext != slots ||
                local_shares.real.physical_shape != TensorShape{{count, slots}})
                return Status::Invalid(
                    "MpcToCkks physical shares do not match target slot map");
            if (target.scale_bits < config_.scale_bits)
                return Status::FailedPrecondition(
                    "MpcToCkks target scale may not discard input precision");
            if (!ProfileAllowsScale(profile.parameters, target.scale_bits))
                return Status::FailedPrecondition(
                    "MpcToCkks target scale is outside its HE profile");
            const bool any_half = std::any_of(half_values.begin(), half_values.end(), [](bool value) { return value; });
            if ((!half_values.empty() && half_values.size() != count) ||
                (any_half && target.scale_bits <= config_.scale_bits))
                return Status::Invalid("half-encoded conversion needs one coefficient precision bit");
            const int shift = target.scale_bits - config_.scale_bits;
            const int base_bits = config_.ring_bits +
                                  config_.statistical_security_bits;
            const int work_bits = base_bits + shift;
            if (base_bits + config_.fft_scale_bits + 2 > 127 ||
                work_bits > 127)
                return Status::FailedPrecondition(
                    "MpcToCkks scale/FFT workspace exceeds 127 bits");

            auto encoding_plan_result = conversion_internal::blb::PlanM2cEncoding(
                degree, config_.ring_bits, config_.scale_bits, target.scale_bits,
                config_.statistical_security_bits, config_.fft_scale_bits, any_half);
            if (!encoding_plan_result.ok()) return encoding_plan_result.status();
            const auto encoding_plan = encoding_plan_result.take_value();
            const int guard_bits = encoding_plan.guard_bits;
            const int coefficient_precision_bits = encoding_plan.coefficient_precision_bits;
            const int coefficient_ring_bits = encoding_plan.coefficient_ring_bits;
            auto transform_result = GetTransform(
                degree, config_.fft_scale_bits, base_bits + guard_bits);
            if (!transform_result.ok()) return transform_result.status();
            const auto transform = transform_result.take_value();
            const int coefficient_work_bits = encoding_plan.coefficient_work_bits;

            ProtocolFrame frame;
            frame.direction = kMpcToCkksDirection;
            frame.call_sequence = call_sequence_++;
            frame.ciphertext_count = count;
            frame.polynomial_degree = degree;
            frame.slots_per_ciphertext = slots;
            frame.logical_elements = target.slot_map.logical_element_count;
            frame.shape_fingerprint = ShapeFingerprint(target.logical_shape);
            frame.logical_layout =
                static_cast<std::uint32_t>(target.logical_layout);
            if (config_.public_m2c_quarter_range) frame.logical_layout |= kM2cQuarterRange;
            if (config_.benchmark_unchecked_range) frame.logical_layout |= kBenchmarkUncheckedRange;
            if (wide_input) frame.logical_layout |= kPreparedSlots;
            const auto expected_size = count * slots;
            const bool prepared_valid = !wide_input ||
                (wide_input->ring_bits == base_bits && wide_input->scale_bits == config_.scale_bits &&
                 wide_input->real.size() == expected_size && wide_input->imag.size() == expected_size &&
                 std::all_of(wide_input->real.begin(), wide_input->real.end(), [base_bits](auto x) {
                     const auto reduced = WideAdd(x, {}, base_bits);
                     return reduced.low == x.low && reduced.high == x.high;
                 }) && std::all_of(wide_input->imag.begin(), wide_input->imag.end(), [base_bits](auto x) {
                     const auto reduced = WideAdd(x, {}, base_bits);
                     return reduced.low == x.low && reduced.high == x.high;
                 }));
            const bool imaginary_contract_valid = !target.public_zero_imaginary ||
                (std::all_of(target.slot_map.assignments.begin(),
                             target.slot_map.assignments.end(),
                             [](const PhysicalSlotAssignment &entry) {
                                 return entry.imaginary_logical_index == kUnusedLogicalIndex;
                             }) &&
                 std::all_of(local_shares.imag.local_share.begin(),
                             local_shares.imag.local_share.end(),
                             [](std::uint64_t share) { return share == 0; }));
            if (target.public_zero_imaginary)
                frame.logical_layout |= kPublicZeroImaginary;
            bool bound_contract_valid = !target.public_value_bound_bits ||
                (*target.public_value_bound_bits >= 0 &&
                 *target.public_value_bound_bits < config_.ring_bits &&
                 (config_.benchmark_unchecked_range || !local_shares.real.public_value_bound_bits ||
                  local_shares.real.public_value_bound_bits == target.public_value_bound_bits));
            if (!target.public_ciphertext_bound_bits.empty()) {
                bound_contract_valid &= target.public_ciphertext_bound_bits.size() == count;
                // Bind the ordered public contracts into the existing
                // fingerprint without adding a transport round trip.
                auto fingerprint = std::uint64_t{1469598103934665603ULL};
                for (const auto bound : target.public_ciphertext_bound_bits) {
                    fingerprint ^= bound ? static_cast<std::uint64_t>(*bound) + 1 : 0;
                    fingerprint *= 1099511628211ULL;
                    if (bound && (*bound < 0 || *bound >= config_.ring_bits ||
                        (!config_.benchmark_unchecked_range && local_shares.real.public_value_bound_bits &&
                         *bound < *local_shares.real.public_value_bound_bits)))
                        bound_contract_valid = false;
                }
                fingerprint ^= target.public_ciphertext_bound_bits.size();
                frame.shape_fingerprint ^= fingerprint;
            }
            bound_contract_valid &= prepared_valid;
            // Source and post-Encode contracts must both be established before OT.
            for (std::size_t ct = 0; ct < count; ++ct) {
                auto bound = target.public_value_bound_bits;
                if (!target.public_ciphertext_bound_bits.empty() &&
                    ct < target.public_ciphertext_bound_bits.size() && target.public_ciphertext_bound_bits[ct])
                    bound = target.public_ciphertext_bound_bits[ct];
                if (!config_.benchmark_unchecked_range && !config_.public_m2c_quarter_range) {
                    const auto coefficient_bound = bound ? transform->EncodedCoefficientBoundBits(
                        *bound, guard_bits, coefficient_precision_bits) : std::nullopt;
                    bound_contract_valid &= bound && *bound >= 0 && *bound <= config_.ring_bits - 2 &&
                        coefficient_bound && *coefficient_bound <= coefficient_ring_bits - 2;
                }
            }
            if (wide_input && target.public_zero_imaginary && prepared_valid)
                bound_contract_valid &= std::all_of(wide_input->imag.begin(), wide_input->imag.end(),
                    [](auto x) { return x.low == 0 && x.high == 0; });
            bool periods_valid = target.public_slot_periods.empty() || target.public_slot_periods.size() == count;
            for (const auto &periods : target.public_slot_periods)
                for (const auto period : periods) {
                    frame.shape_fingerprint ^= period;
                    frame.shape_fingerprint *= 1099511628211ULL;
                    periods_valid &= period == 0 || (period <= slots && (period & (period - 1)) == 0);
                }
            frame.shape_fingerprint ^= target.public_slot_periods.size();
            if (periods_valid && prepared_valid && !target.public_slot_periods.empty())
                for (std::size_t ct = 0; ct < count; ++ct)
                    for (std::size_t plane = 0; plane < 2; ++plane) {
                        const auto period = target.public_slot_periods[ct][plane];
                        if (!period || period == slots) continue;
                        std::vector<std::size_t> first(period, kUnusedLogicalIndex);
                        const auto &values = plane ? local_shares.imag.local_share : local_shares.real.local_share;
                        for (std::size_t slot = 0; slot < slots; ++slot) {
                            const auto &entry = target.slot_map.assignments[ct * slots + slot];
                            if ((plane ? entry.imaginary_logical_index : entry.real_logical_index) == kUnusedLogicalIndex) continue;
                            auto &representative = first[slot % period];
                            if (representative == kUnusedLogicalIndex) representative = slot;
                            else if (wide_input) {
                                const auto &wide_values = plane ? wide_input->imag : wide_input->real;
                                const auto x = wide_values[ct * slots + slot], y = wide_values[ct * slots + representative];
                                periods_valid &= x.low == y.low && x.high == y.high;
                            } else periods_valid &= values[ct * slots + slot] == values[ct * slots + representative];
                        }
                    }
            bound_contract_valid &= periods_valid;
            if (canonical_padding || !target.public_slot_periods.empty()) frame.logical_layout |= kCanonicalLanePacking;
            if (target.public_value_bound_bits && bound_contract_valid)
                frame.logical_layout |= static_cast<std::uint32_t>(
                    *target.public_value_bound_bits + 1) << kPublicBoundShift;
            // Coordinate rejection even if only the server has invalid shares:
            // neither peer may enter SignedExtend while the other has returned.
            if (!imaginary_contract_valid || !bound_contract_valid)
                frame.logical_layout |= kInvalidPublicContract;
            // Bind physical lane assignments as well as the pre-encoding factors.
            for (const auto &entry : target.slot_map.assignments)
                for (auto index : {entry.real_logical_index, entry.imaginary_logical_index}) {
                    frame.shape_fingerprint ^= index;
                    frame.shape_fingerprint *= 1099511628211ULL;
                }
            // Bind the public pre-encoding factors as part of the wire contract.
            for (bool half : half_values) {
                frame.shape_fingerprint ^= static_cast<std::uint64_t>(half) + 17;
                frame.shape_fingerprint *= 1099511628211ULL;
            }
            frame.parameter_fingerprint = ParameterFingerprint(*profile.context);
            frame.terminal_modulus = profile.context->last_context_data()
                                         ->parms().coeff_modulus().front().value();
            SetParmsId(frame, target_id);
            frame.target_level = target.target_level.value_or(0);
            frame.ring_bits = config_.ring_bits;
            frame.share_scale_bits = config_.scale_bits;
            frame.he_scale_bits = target.scale_bits;
            frame.statistical_bits = config_.statistical_security_bits;
            frame.fft_scale_bits = config_.fft_scale_bits;
            frame.profile_bytes =
                static_cast<std::uint32_t>(target.parameter_profile.size());
            const auto handshake_started = Clock::now();
            stats_.mpc_to_ckks.preparation +=
                std::chrono::duration<double, std::milli>(
                    handshake_started - started).count();
            if (role_ == PartyRole::kServer) {
                transport_->Send(&frame, sizeof(frame));
                SendString(*transport_, target.parameter_profile);
                std::uint32_t acknowledged = 0;
                transport_->Receive(&acknowledged, sizeof(acknowledged));
                if (acknowledged != 1)
                    return Status::FailedPrecondition(
                        "client rejected the MpcToCkks handshake");
            } else {
                ProtocolFrame peer;
                transport_->Receive(&peer, sizeof(peer));
                const std::string peer_profile =
                    ReceiveString(*transport_, peer.profile_bytes);
                if (!SameFrame(peer, frame) ||
                    peer_profile != target.parameter_profile ||
                    !imaginary_contract_valid || !bound_contract_valid) {
                    const std::uint32_t rejected = 0;
                    transport_->Send(&rejected, sizeof(rejected));
                    return Status::FailedPrecondition(
                        "MpcToCkks profile/protocol handshake mismatch");
                }
                const std::uint32_t acknowledged = 1;
                transport_->Send(&acknowledged, sizeof(acknowledged));
            }
            stats_.mpc_to_ckks.handshake += MillisecondsSince(handshake_started);

            const auto wrap_correction = GetRnsWrapCorrection(work_bits, *context_data);
            const auto half_wrap_correction = any_half
                ? GetRnsWrapCorrection(work_bits - 1, *context_data) : wrap_correction;
            const auto public_bound = [&](std::size_t ct) {
                if (!target.public_ciphertext_bound_bits.empty() &&
                    target.public_ciphertext_bound_bits[ct])
                    return target.public_ciphertext_bound_bits[ct];
                return target.public_value_bound_bits;
            };
            seal::Evaluator evaluator(*profile.context);
            std::unique_ptr<seal::Encryptor> encryptor;
            if (role_ == PartyRole::kClient)
                encryptor = std::make_unique<seal::Encryptor>(
                    *profile.context, profile.secret_key.value());
            std::vector<seal::Ciphertext> outputs;
            if (role_ == PartyRole::kServer) outputs.reserve(count);
            struct LocalRns {
                seal::Plaintext plaintext;
                double milliseconds{};
            };
            std::deque<std::pair<std::size_t, std::future<LocalRns>>> pending_rns;
            const std::size_t max_inflight =
                std::max<std::size_t>(1, std::min({count, workers_.size(),
                    detail::HeWorkerPool::Instance().MemoryBudget() /
                        detail::EstimatedHeWorkerBytes(degree,
                            context_data->parms().coeff_modulus().size(), 8)}));
            const auto consume_rns = [&]() -> Result<void> {
                auto item = std::move(pending_rns.front());
                pending_rns.pop_front();
                auto rns = item.second.get();
                stats_.mpc_to_ckks.rns += rns.milliseconds;
                auto plaintext = std::move(rns.plaintext);
                plaintext.parms_id() = target_id;
                plaintext.scale() = std::ldexp(1.0, target.scale_bits);
                if (role_ == PartyRole::kClient) {
                    seal::Ciphertext encrypted;
                    const auto he_started = Clock::now();
                    encryptor->encrypt_symmetric(plaintext, encrypted);
                    stats_.mpc_to_ckks.he += MillisecondsSince(he_started);
                    const auto send_started = Clock::now();
                    SendSealObject(*transport_, encrypted, wire_compression_);
                    stats_.mpc_to_ckks.ciphertext_transport +=
                        MillisecondsSince(send_started);
                } else {
                    const auto receive_started = Clock::now();
                    auto encrypted = ReceiveSealObject<seal::Ciphertext>(
                        *transport_, *profile.context);
                    stats_.mpc_to_ckks.ciphertext_transport +=
                        MillisecondsSince(receive_started);
                    if (encrypted.parms_id() != target_id)
                        return Status::FailedPrecondition(
                            "MpcToCkks client ciphertext level mismatch");
                    const auto he_started = Clock::now();
                    evaluator.add_plain_inplace(encrypted, plaintext);
                    stats_.mpc_to_ckks.he += MillisecondsSince(he_started);
                    outputs.push_back(std::move(encrypted));
                }
                return {};
            };

            const auto kCiphertextBatch = CiphertextBatchSize(degree, count);
            for (std::size_t base = 0; base < count;) {
                const auto slot_bound = public_bound(base);
                // Only merge a contiguous run with the same public range.
                // Matching first/last entries do not establish the bounds
                // of intervening ciphertexts.
                std::size_t batch = 1;
                const auto limit = std::min(kCiphertextBatch, count - base);
                while (batch < limit && public_bound(base + batch) == slot_bound) ++batch;
                const auto coefficient_bound_bits = slot_bound
                    ? transform->EncodedCoefficientBoundBits(*slot_bound,
                        guard_bits, coefficient_precision_bits)
                    : std::nullopt;
                const auto real_count = batch * slots;
                const auto begin = base * slots;
                std::vector<std::uint64_t> lanes(
                    local_shares.real.local_share.begin() + begin,
                    local_shares.real.local_share.begin() + begin + real_count);
                if (!target.public_zero_imaginary)
                    lanes.insert(lanes.end(), local_shares.imag.local_share.begin() + begin,
                                 local_shares.imag.local_share.begin() + begin + real_count);
                else stats_.public_zero_imaginary_ciphertexts += batch;
                const auto extend_started = Clock::now();
                const auto lane_plan = detail::PlanSlotExtensions(target, base, batch, canonical_padding);
                if (!wide_input) {
                    stats_.slot_extension_input_elements += lanes.size();
                    stats_.slot_extension_padding_elements += lane_plan.padding;
                }
                std::vector<WideRingShare> wide_lanes;
                if (wide_input) {
                    wide_lanes.assign(wide_input->real.begin() + begin,
                        wide_input->real.begin() + begin + real_count);
                    if (!target.public_zero_imaginary)
                        wide_lanes.insert(wide_lanes.end(), wide_input->imag.begin() + begin,
                            wide_input->imag.begin() + begin + real_count);
                } else if (lane_plan.identity) {
                    stats_.slot_extension_unique_elements += lanes.size();
                    wide_lanes = SignedExtend(lanes, config_.ring_bits, base_bits, slot_bound);
                } else {
                    std::vector<std::uint64_t> unique;
                    unique.reserve(lane_plan.representatives.size());
                    for (auto index : lane_plan.representatives) unique.push_back(lanes[index]);
                    stats_.slot_extension_unique_elements += unique.size();
                    auto extended = SignedExtend(unique, config_.ring_bits, base_bits, slot_bound);
                    wide_lanes.resize(lanes.size());
                    for (std::size_t i = 0; i < lanes.size(); ++i)
                        if (lane_plan.inverse[i] != kUnusedLogicalIndex)
                            wide_lanes[i] = extended[lane_plan.inverse[i]];
                }
                const auto prepare_ms = MillisecondsSince(extend_started);
                stats_.mpc_to_ckks.slot_prepare += prepare_ms;
                stats_.mpc_to_ckks.signed_extend += prepare_ms;
                std::vector<std::uint64_t> coefficient_ring(batch * degree);
                const auto encode_started = Clock::now();
                for (std::size_t part = 0; part < batch; ++part) {
                    conversion_internal::blb::PreparedSlotShares prepared;
                    prepared.ring_bits = base_bits;
                    prepared.scale_bits = config_.scale_bits;
                    prepared.values.real.resize(slots);
                    prepared.values.imag.resize(slots);
                    for (std::size_t slot = 0; slot < slots; ++slot) {
                        prepared.values.real[slot] = wide_lanes[part * slots + slot];
                        if (!target.public_zero_imaginary)
                            prepared.values.imag[slot] = wide_lanes[real_count + part * slots + slot];
                    }
                    auto encoded = conversion_internal::blb::EncodePreparedSlots(
                        prepared, encoding_plan, *transform, role_ == PartyRole::kClient);
                    if (!encoded.ok()) return encoded.status();
                    std::copy(encoded.value().values.begin(), encoded.value().values.end(),
                              coefficient_ring.begin() + part * degree);
                }
                stats_.mpc_to_ckks.slot_encode += MillisecondsSince(encode_started);
                // Re-extend coefficients before the fixed R2F correction.
                // A proved public FFT bound permits the small MSB-to-wrap
                // protocol here, without removing the statistical extension.
                const auto coefficient_extend_started = Clock::now();
                auto wide_coefficients = SignedExtend(coefficient_ring,
                    coefficient_ring_bits, coefficient_work_bits,
                    coefficient_bound_bits);
                stats_.coefficient_extension_elements += coefficient_ring.size();
                if (config_.benchmark_unchecked_range || config_.public_m2c_quarter_range ||
                    (coefficient_bound_bits && *coefficient_bound_bits <= coefficient_ring_bits - 2))
                    stats_.bounded_coefficient_extension_elements += coefficient_ring.size();
                const auto coefficient_ms = MillisecondsSince(coefficient_extend_started);
                stats_.mpc_to_ckks.coefficient_extend += coefficient_ms;
                stats_.mpc_to_ckks.signed_extend += coefficient_ms;
                for (std::size_t ct = base; ct < base + batch; ++ct) {
                    const auto offset = (ct - base) * degree;
                    std::vector<WideRingShare> coefficients(
                        wide_coefficients.begin() + offset,
                        wide_coefficients.begin() + offset + degree);
                    const bool half = !half_values.empty() && half_values[ct];
                    const int encoded_work_bits = work_bits - static_cast<int>(half);
                    const auto correction = half ? half_wrap_correction : wrap_correction;
                    for (auto &coefficient : coefficients)
                        coefficient = WideShiftLeft(coefficient,
                            shift - coefficient_precision_bits - static_cast<int>(half), encoded_work_bits);
                    pending_rns.emplace_back(
                        ct,
                        detail::SubmitLocalTask(max_inflight,
                                   [this, coefficients = std::move(coefficients),
                                    encoded_work_bits, context_data, correction]() {
                                       const auto rns_started = Clock::now();
                                       auto plaintext = BuildBlbRnsPlaintextShare(
                                           coefficients, encoded_work_bits, *context_data,
                                           *correction);
                                       return LocalRns{std::move(plaintext),
                                                       MillisecondsSince(rns_started)};
                                   }));
                    ++stats_.local_transform_tasks;
                    stats_.local_transform_peak_inflight = std::max(
                        stats_.local_transform_peak_inflight, pending_rns.size());
                    if (pending_rns.size() >= max_inflight) {
                        auto consumed = consume_rns();
                        if (!consumed.ok()) return consumed.status();
                    }
                }
                base += batch;
            }
            while (!pending_rns.empty()) {
                auto consumed = consume_rns();
                if (!consumed.ok()) return consumed.status();
            }

            stats_.milliseconds += std::chrono::duration<double, std::milli>(
                Clock::now() - started).count();
            if (role_ == PartyRole::kClient)
                return std::optional<SealCipherBatch>{};
            return std::optional<SealCipherBatch>(SealCipherBatch{
                target.parameter_profile, std::move(outputs),
                target.logical_shape, target.logical_layout, target.slot_map});
        } catch (const std::exception &error) {
            return Status::Internal(std::string("MpcToCkks failed: ") +
                                    error.what());
        }
    }


    Result<std::optional<SealCipherBatch>> MpcToCkksPacked(
        const ComplexMpcSharePair &local, const MpcToCkksTarget &target)
    {
        const auto started = Clock::now();
        try {
            auto valid = local.Validate();
            if (!valid.ok()) return valid.status();
            auto planned = detail::PlanRealConversion(target.slot_map);
            if (!planned.ok()) return planned.status();
            const auto &plan = planned.value();
            const auto slots = target.slot_map.slots_per_ciphertext;
            if (local.real.physical_shape != TensorShape{{target.slot_map.ciphertext_count, slots}})
                return Status::Invalid("packed conversion share dimensions differ");
            if (!target.public_ciphertext_bound_bits.empty() &&
                target.public_ciphertext_bound_bits.size() != target.slot_map.ciphertext_count)
                return Status::Invalid("packed conversion bound count differs");
            const bool has_real = std::any_of(plan.groups.begin(), plan.groups.end(),
                [](const auto &group) { return group.real; });
            if (!has_real) return MpcToCkks(local, target, {}, true);
            // Preserve coordinated validation of the original zero-imaginary contract.
            if (target.public_zero_imaginary &&
                (std::any_of(target.slot_map.assignments.begin(), target.slot_map.assignments.end(),
                    [](const auto &entry) { return entry.imaginary_logical_index != kUnusedLogicalIndex; }) ||
                 std::any_of(local.imag.local_share.begin(), local.imag.local_share.end(),
                    [](auto value) { return value != 0; })))
                return MpcToCkks(local, target);
            if (!config_.benchmark_unchecked_range && local.real.public_value_bound_bits &&
                ((target.public_value_bound_bits && *target.public_value_bound_bits < *local.real.public_value_bound_bits) ||
                 std::any_of(target.public_ciphertext_bound_bits.begin(), target.public_ciphertext_bound_bits.end(),
                    [&](const auto &range) { return range && *range < *local.real.public_value_bound_bits; })))
                return MpcToCkks(local, target);
            if (!target.public_slot_periods.empty() && target.public_slot_periods.size() != target.slot_map.ciphertext_count)
                return MpcToCkks(local, target);
            const auto &context = contexts_.Require(target.parameter_profile, role_);
            ComplexMpcSharePair packed;
            packed.real.numeric = local.real.numeric;
            packed.imag.numeric = local.imag.numeric;
            packed.real.component = ComplexComponent::kReal;
            packed.imag.component = ComplexComponent::kImaginary;
            for (auto member : {&ComplexMpcSharePair::real, &ComplexMpcSharePair::imag}) {
                (packed.*member).local_share.assign(plan.groups.size() * slots, 0);
                (packed.*member).physical_shape = {{plan.groups.size(), slots}};
            }
            auto destination = target;
            destination.slot_map = plan.map;
            destination.public_zero_imaginary = false;
            destination.public_value_bound_bits.reset();
            destination.public_ciphertext_bound_bits.clear();
            destination.public_slot_periods.clear();
            const auto period = [&](std::size_t ct, std::size_t plane) {
                return target.public_slot_periods.empty() ? std::size_t{0} : target.public_slot_periods[ct][plane];
            };
            const auto bound = [&](std::size_t ct) {
                return !target.public_ciphertext_bound_bits.empty() && target.public_ciphertext_bound_bits[ct]
                    ? target.public_ciphertext_bound_bits[ct] : target.public_value_bound_bits;
            };
            std::vector<bool> half(plan.groups.size());
            for (std::size_t g = 0; g < plan.groups.size(); ++g) {
                const auto &group = plan.groups[g];
                half[g] = group.real;
                auto real_period = period(group.first, 0);
                auto imag_period = !group.real ? period(group.first, 1)
                    : group.second != kUnusedLogicalIndex ? period(group.second, 0)
                    : group.split ? real_period : std::size_t{0};
                // The scale-1 imaginary phase alternates with slot parity.
                if (group.real && imag_period == 1) imag_period = 2;
                destination.public_slot_periods.push_back({real_period, imag_period});
                auto range = bound(group.first);
                if (group.second != kUnusedLogicalIndex) {
                    const auto other = bound(group.second);
                    range = range && other ? std::optional<int>{std::max(*range, *other)} : std::nullopt;
                }
                destination.public_ciphertext_bound_bits.push_back(range);
                for (std::size_t slot = 0; slot < slots; ++slot) {
                    const auto source = group.first * slots + slot;
                    const auto dest = g * slots + slot;
                    if (!group.real) {
                        packed.real.local_share[dest] = local.real.local_share[source];
                        packed.imag.local_share[dest] = local.imag.local_share[source];
                    } else if (target.slot_map.assignments[source].real_logical_index != kUnusedLogicalIndex) {
                        if (group.split && slot >= group.split)
                            packed.imag.local_share[dest - group.split] = detail::UndoPairedRealPhase(
                                local.real.local_share[source], slot - group.split, config_.ring_bits);
                        else packed.real.local_share[dest] = local.real.local_share[source];
                    }
                    if (group.second != kUnusedLogicalIndex &&
                        target.slot_map.assignments[group.second * slots + slot].real_logical_index != kUnusedLogicalIndex)
                        packed.imag.local_share[dest] = detail::UndoPairedRealPhase(
                            local.real.local_share[group.second * slots + slot], slot, config_.ring_bits);
                }
            }
            stats_.mpc_to_ckks.preparation += MillisecondsSince(started);
            stats_.milliseconds += MillisecondsSince(started);
            if (target.public_slot_periods.empty()) destination.public_slot_periods.clear();
            auto result = MpcToCkks(packed, destination, half, true);
            if (!result.ok()) return result.status();
            for (const auto &group : plan.groups) if (group.real) {
                stats_.packed_real_input_ciphertexts += group.second == kUnusedLogicalIndex ? 1 : 2;
                ++stats_.packed_real_wire_ciphertexts;
            }
            if (role_ == PartyRole::kClient) return result;
            const auto split_started = Clock::now();
            auto input = std::move(result.value()->ciphertexts);
            std::vector<seal::Ciphertext> output(target.slot_map.ciphertext_count);
            seal::Evaluator evaluator(*context.context);
            auto phase = detail::MakeRealPairPhase(context, input.front().parms_id());
            if (!phase.ok()) return phase.status();
            for (std::size_t g = 0; g < plan.groups.size(); ++g) {
                const auto &group = plan.groups[g];
                if (!group.real) { output[group.first] = std::move(input[g]); continue; }
                seal::Ciphertext conjugated, imaginary;
                evaluator.complex_conjugate(input[g], context.galois_keys, conjugated);
                evaluator.add(input[g], conjugated, output[group.first]);
                if (group.second != kUnusedLogicalIndex || group.split) {
                    evaluator.sub(conjugated, input[g], imaginary);
                    evaluator.multiply_plain_inplace(imaginary, phase.value());
                    if (group.split) {
                        const int rotation = group.split == slots / 2
                            ? static_cast<int>(group.split) : -static_cast<int>(group.split);
                        evaluator.rotate_vector_inplace(imaginary, rotation, context.galois_keys);
                        evaluator.add_inplace(output[group.first], imaginary);
                    } else output[group.second] = std::move(imaginary);
                }
                ++stats_.real_output_split_ciphertexts;
            }
            stats_.mpc_to_ckks.he += MillisecondsSince(split_started);
            stats_.milliseconds += MillisecondsSince(split_started);
            return std::optional<SealCipherBatch>{SealCipherBatch{target.parameter_profile,
                std::move(output), target.logical_shape, target.logical_layout, target.slot_map}};
        } catch (const std::exception &error) {
            return Status::Internal(std::string("packed MpcToCkks [") + target.parameter_profile + "] failed: " + error.what());
        }
    }

private:
    PartyRole role_;
    ConversionConfig config_;
    HeContextRegistry &contexts_;
    std::shared_ptr<IProtocolTransport> transport_;
    std::shared_ptr<IRandomSource> randomness_;
    seal::compr_mode_type wire_compression_;
    std::vector<std::unique_ptr<Worker>> workers_;
    std::map<TransformKey, std::shared_ptr<const FixedCkksTransform>>
        transforms_;
    std::map<std::pair<seal::parms_id_type, int>,
             std::shared_ptr<const std::vector<std::uint64_t>>> rns_wrap_corrections_;
    std::uint64_t call_sequence_{};
    ConversionStats stats_;
};

BlbReferenceConverter::BlbReferenceConverter(
    PartyRole role, ConversionConfig config, HeContextRegistry &contexts,
    std::shared_ptr<IProtocolTransport> transport,
    std::shared_ptr<IRandomSource> randomness, std::string peer_address,
    int worker_port, std::size_t threads,
    seal::compr_mode_type wire_compression)
    : impl_(std::make_unique<Impl>(
          role, config, contexts, std::move(transport), std::move(randomness),
          std::move(peer_address), worker_port, threads, wire_compression))
{}

BlbReferenceConverter::~BlbReferenceConverter() = default;
BlbReferenceConverter::BlbReferenceConverter(BlbReferenceConverter &&) noexcept = default;
BlbReferenceConverter &BlbReferenceConverter::operator=(BlbReferenceConverter &&) noexcept =
    default;

Result<ComplexMpcSharePair> BlbReferenceConverter::CkksToMpc(
    const std::optional<SealCipherBatch> &server_input,
    std::optional<int> output_scale_bits,
    std::optional<int> public_coefficient_bound_bits)
{
    return impl_->CkksToMpc(server_input, output_scale_bits, public_coefficient_bound_bits);
}

Result<std::optional<SealCipherBatch>> BlbReferenceConverter::MpcToCkks(
    const ComplexMpcSharePair &local_shares,
    const MpcToCkksTarget &target)
{
    return impl_->MpcToCkks(local_shares, target);
}

Result<std::optional<SealCipherBatch>> BlbReferenceConverter::MpcToCkksPacked(
    const ComplexMpcSharePair &local_shares, const MpcToCkksTarget &target)
{
    return impl_->MpcToCkksPacked(local_shares, target);
}

Result<std::optional<SealCipherBatch>> BlbReferenceConverter::MpcToCkksPrepared(
    const PreparedBlbSlots &local, const MpcToCkksTarget &target)
{
    // Markers supply public shape/numeric metadata only; the engine reads
    // actual slot values exclusively from local in the prepared branch.
    const auto count = target.slot_map.assignments.size();
    const TensorShape shape{{target.slot_map.ciphertext_count, target.slot_map.slots_per_ciphertext}};
    ComplexMpcSharePair markers{
        {std::vector<std::uint64_t>(count), shape, {impl_->config().ring_bits, impl_->config().scale_bits}, ComplexComponent::kReal},
        {std::vector<std::uint64_t>(count), shape, {impl_->config().ring_bits, impl_->config().scale_bits}, ComplexComponent::kImaginary}};
    return impl_->MpcToCkks(markers, target, {}, false, &local);
}

ConversionStats BlbReferenceConverter::stats() const noexcept
{
    return impl_->stats();
}

void BlbReferenceConverter::Cancel() noexcept
{
    if (impl_) impl_->Cancel();
}

} // namespace hera
