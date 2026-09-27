#include "hera/conversion.h"
#include "hera/seal_wire.h"

#include "fixed_ckks_transform.h"
#include "random_sampling.h"

#include "OT/split-kkot.h"
#include "FloatingPoint/fixed-point.h"
#include "OT/ot_pack.h"
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
#include <seal/util/uintcore.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <future>
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
using conversion_internal::FixedCkksTransform;
using conversion_internal::SampleUniformPrime;
using conversion_internal::SampleUniformRing;
using conversion_internal::SlotSharePair;

constexpr std::uint64_t kFrameMagic = 0x48455241434f4e56ULL;
constexpr std::uint32_t kProtocolVersion = 2;
constexpr std::uint32_t kCkksToMpcDirection = 1;
constexpr std::uint32_t kMpcToCkksDirection = 2;

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

std::vector<std::uint64_t> ReduceUnsignedModulo(
    const std::uint64_t *value, std::size_t value_limbs,
    const std::uint64_t *modulus, std::size_t modulus_limbs,
    seal::MemoryPoolHandle pool)
{
    if (modulus_limbs == 0 ||
        seal::util::is_zero_uint(modulus, modulus_limbs))
        throw std::invalid_argument("RNS bridge modulus is zero");

    const std::size_t limbs = std::max(value_limbs, modulus_limbs);
    std::vector<std::uint64_t> numerator(limbs, 0);
    std::vector<std::uint64_t> denominator(limbs, 0);
    std::vector<std::uint64_t> quotient(limbs, 0);
    std::copy_n(value, value_limbs, numerator.begin());
    std::copy_n(modulus, modulus_limbs, denominator.begin());
    seal::util::divide_uint_inplace(numerator.data(), denominator.data(), limbs,
                                    quotient.data(), pool);
    numerator.resize(modulus_limbs);
    return numerator;
}

std::vector<std::uint64_t> WideModuloComposite(
    WideRingShare value, int bits, const std::uint64_t *modulus,
    std::size_t modulus_limbs, seal::MemoryPoolHandle pool)
{
    std::uint64_t raw[2]{value.low, value.high};
    if (bits < 64)
        raw[0] &= LowMask(bits);
    else if (bits < 128)
        raw[1] &= LowMask(bits - 64);
    return ReduceUnsignedModulo(raw, bits <= 64 ? 1 : 2, modulus,
                                modulus_limbs, std::move(pool));
}

std::vector<std::uint64_t> PowerOfTwoModuloComposite(
    int bits, const std::uint64_t *modulus, std::size_t modulus_limbs,
    seal::MemoryPoolHandle pool)
{
    if (bits < 0 || bits > 127)
        throw std::invalid_argument("RNS bridge exponent must be in [0,127]");
    std::uint64_t power[2]{};
    power[static_cast<std::size_t>(bits) / 64] =
        std::uint64_t{1} << (bits % 64);
    return ReduceUnsignedModulo(power, bits < 64 ? 1 : 2, modulus,
                                modulus_limbs, std::move(pool));
}

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

class CkksMpcConverter::Impl {
public:
    struct Worker {
        int effective_party{};
        std::unique_ptr<sci::IOPack> io;
        std::unique_ptr<sci::OTPack> ot;
        std::unique_ptr<FixOp> fix;
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
        if (!contexts_.initialized() || contexts_.role() != role_)
            throw std::invalid_argument("converter context registry role mismatch");
        if (config_.ring_bits < 2 || config_.ring_bits > 63)
            throw std::invalid_argument("conversion ring bits must be in [2,63]");
        if (config_.statistical_security_bits < 1 ||
            config_.statistical_security_bits > 63)
            throw std::invalid_argument("statistical bits must be in [1,63]");
        if (config_.ring_bits + config_.statistical_security_bits > 127)
            throw std::invalid_argument("wide conversion ring exceeds 127 bits");
        if (threads < 1 || threads > 16)
            throw std::invalid_argument("conversion threads must be in [1,16]");
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
            worker->ot = std::make_unique<sci::OTPack>(
                worker->io.get(), worker->effective_party);
            worker->fix = std::make_unique<FixOp>(
                worker->effective_party, worker->io.get(), worker->ot.get());
            workers_.push_back(std::move(worker));
        }
    }

    ConversionStats stats() const noexcept
    {
        ConversionStats result = stats_;
        result.communication_bytes = transport_->bytes();
        result.rounds = transport_->rounds();
        for (const auto &worker : workers_) {
            result.communication_bytes += worker->io->get_comm();
            result.rounds += worker->io->get_rounds();
        }
        return result;
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
        std::vector<std::thread> threads;
        std::exception_ptr first_error;
        std::mutex error_mutex;
        for (std::size_t id = 0; id < workers_.size(); ++id) {
            const std::size_t begin = size * id / workers_.size();
            const std::size_t end = size * (id + 1) / workers_.size();
            if (begin == end) continue;
            threads.emplace_back([&, id, begin, end] {
                try {
                    operation(*workers_[id], begin, end);
                } catch (...) {
                    {
                        std::lock_guard<std::mutex> lock(error_mutex);
                        if (!first_error)
                            first_error = std::current_exception();
                    }
                    // A failed worker must wake peers blocked in OT before
                    // this thread group is joined.
                    Cancel();
                }
            });
        }
        for (auto &thread : threads) thread.join();
        if (first_error) std::rethrow_exception(first_error);
    }

    std::vector<std::uint64_t> CenteredFieldToRing(
        const std::vector<std::uint64_t> &field_share, std::uint64_t modulus,
        int output_bits)
    {
        const int field_bits = seal::util::get_significant_bit_count(modulus);
        if (output_bits < field_bits || output_bits > 64)
            throw std::invalid_argument("invalid centered field-to-ring width");
        const std::uint64_t center = modulus / 2;
        const std::uint64_t mask = LowMask(output_bits);
        std::vector<std::uint64_t> output(field_share.size());
        Parallel(field_share.size(), [&](Worker &worker, std::size_t begin,
                                         std::size_t end) {
            const int size = static_cast<int>(end - begin);
            std::vector<std::uint64_t> adjusted(size);
            for (int i = 0; i < size; ++i) {
                adjusted[i] = field_share[begin + i] % modulus;
                if (worker.effective_party == sci::ALICE)
                    adjusted[i] = static_cast<std::uint64_t>(
                        (static_cast<unsigned __int128>(adjusted[i]) + center) %
                        modulus);
            }
            std::vector<std::uint8_t> wrap(size);
            worker.fix->aux->wrap_computation_prime(
                adjusted.data(), wrap.data(), size, field_bits, modulus);
            std::vector<std::uint64_t> arithmetic_wrap(size);
            worker.fix->aux->B2A(wrap.data(), arithmetic_wrap.data(), size,
                                 output_bits);
            for (int i = 0; i < size; ++i) {
                unsigned __int128 value = adjusted[i];
                value -= static_cast<unsigned __int128>(modulus) *
                         arithmetic_wrap[i];
                if (worker.effective_party == sci::ALICE) value -= center;
                output[begin + i] = static_cast<std::uint64_t>(value) & mask;
            }
        });
        return output;
    }

    std::vector<WideRingShare> SignedExtend(
        const std::vector<std::uint64_t> &shares, int input_bits,
        int output_bits)
    {
        if (input_bits < 2 || input_bits > 63 || output_bits <= input_bits ||
            output_bits > 127 || output_bits - input_bits > 63)
            throw std::invalid_argument("unsupported secure signed extension");
        const int extension_bits = output_bits - input_bits;
        const std::uint64_t input_mask = LowMask(input_bits);
        std::vector<WideRingShare> result(shares.size());
        Parallel(shares.size(), [&](Worker &worker, std::size_t begin,
                                    std::size_t end) {
            const int size = static_cast<int>(end - begin);
            std::vector<std::uint64_t> local(size);
            for (int i = 0; i < size; ++i)
                local[i] = shares[begin + i] & input_mask;
            std::vector<std::uint8_t> wrap(size), sign(size);
            worker.fix->aux->wrap_computation(local.data(), wrap.data(), size,
                                               input_bits);
            worker.fix->aux->MSB(local.data(), sign.data(), size, input_bits);
            std::vector<std::uint64_t> wrap_a(size), sign_a(size);
            worker.fix->aux->B2A(wrap.data(), wrap_a.data(), size,
                                 extension_bits);
            worker.fix->aux->B2A(sign.data(), sign_a.data(), size,
                                 extension_bits);
            for (int i = 0; i < size; ++i) {
                const auto base = WideFromLow(local[i], output_bits);
                const auto correction = WideShiftLeft(
                    WideFromLow((wrap_a[i] + sign_a[i]) &
                                    LowMask(extension_bits),
                                extension_bits),
                    input_bits, output_bits);
                result[begin + i] =
                    WideSubtract(base, correction, output_bits);
            }
        });
        return result;
    }

    seal::Plaintext BuildBlbRnsPlaintextShare(
        const std::vector<WideRingShare> &coefficients, int bits,
        const seal::SEALContext::ContextData &context_data)
    {
        const auto &parms = context_data.parms();
        const auto &moduli = parms.coeff_modulus();
        const std::size_t degree = parms.poly_modulus_degree();
        const std::size_t modulus_limbs = moduli.size();
        if (coefficients.size() != degree || modulus_limbs == 0)
            throw std::invalid_argument(
                "BLB RNS bridge coefficient/profile size mismatch");

        // BLB converts the encoded coefficient share to one field modulo the
        // target coefficient modulus Q, and only then expands that field
        // element into the target RNS base. Unlike the artifact's uint64_t Q,
        // this implementation retains every limb of Q = product(q_j), which
        // is required when a target level contains more than one data prime.
        const auto *composite_modulus = context_data.total_coeff_modulus();
        auto pool = seal::MemoryManager::GetPool();
        const auto wrap_correction = PowerOfTwoModuloComposite(
            bits, composite_modulus, modulus_limbs, pool);

        seal::Plaintext plaintext(degree * modulus_limbs);
        std::vector<std::uint64_t> bridge_share(modulus_limbs);
        std::vector<std::uint64_t> adjusted_share(modulus_limbs);
        for (std::size_t coefficient = 0; coefficient < degree;
             ++coefficient) {
            bridge_share = WideModuloComposite(
                coefficients[coefficient], bits, composite_modulus,
                modulus_limbs, pool);
            if (role_ == PartyRole::kServer) {
                seal::util::sub_uint_uint_mod(
                    bridge_share.data(), wrap_correction.data(),
                    composite_modulus, modulus_limbs,
                    adjusted_share.data());
                bridge_share.swap(adjusted_share);
            }

            // SEAL replaces the full-Q integer in-place by its residues in
            // q_0,...,q_{L-1}. This is the same logical boundary as BLB's
            // ckks_ntt, without repeating ring-to-field for every prime.
            context_data.rns_tool()->base_q()->decompose(
                bridge_share.data(), pool);
            for (std::size_t prime = 0; prime < modulus_limbs; ++prime)
                plaintext[coefficient + prime * degree] =
                    bridge_share[prime];
        }

        for (std::size_t prime = 0; prime < modulus_limbs; ++prime)
            seal::util::ntt_negacyclic_harvey(
                plaintext.data() + prime * degree,
                context_data.small_ntt_tables()[prime]);
        return plaintext;
    }

    Result<ComplexMpcSharePair> CkksToMpc(
        const std::optional<SealCipherBatch> &server_input)
    {
        const auto started = Clock::now();
        try {
            if ((role_ == PartyRole::kServer) != server_input.has_value())
                return Status::Invalid(
                    "only the server supplies CkksToMpc ciphertexts");

            ProtocolFrame frame;
            std::string profile_name;
            const HeContext *profile = nullptr;
            std::vector<seal::Ciphertext> ciphertexts;
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
                frame.parameter_fingerprint =
                    ParameterFingerprint(*profile->context);
                frame.terminal_modulus =
                    terminal->parms().coeff_modulus().front().value();
                SetParmsId(frame, terminal->parms_id());
                frame.ring_bits = config_.ring_bits;
                frame.share_scale_bits = config_.scale_bits;
                frame.he_scale_bits = he_scale;
                frame.statistical_bits = config_.statistical_security_bits;
                frame.fft_scale_bits = config_.fft_scale_bits;
                frame.profile_bytes = static_cast<std::uint32_t>(profile_name.size());
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
                    frame.ring_bits != config_.ring_bits ||
                    frame.share_scale_bits != config_.scale_bits ||
                    frame.statistical_bits !=
                        config_.statistical_security_bits ||
                    frame.fft_scale_bits != config_.fft_scale_bits) {
                    const std::uint32_t rejected = 0;
                    transport_->Send(&rejected, sizeof(rejected));
                    return Status::FailedPrecondition(
                        "CkksToMpc profile/protocol handshake mismatch");
                }
                const std::uint32_t acknowledged = 1;
                transport_->Send(&acknowledged, sizeof(acknowledged));
            }

            const std::size_t degree = frame.polynomial_degree;
            const std::size_t slots = frame.slots_per_ciphertext;
            const std::size_t count = frame.ciphertext_count;
            const auto terminal = profile->context->last_context_data();
            const std::uint64_t modulus =
                terminal->parms().coeff_modulus().front().value();
            const int field_bits = seal::util::get_significant_bit_count(modulus);
            const int wide_bits = field_bits + config_.statistical_security_bits;
            if (wide_bits + config_.fft_scale_bits + 2 > 127)
                return Status::FailedPrecondition(
                    "CkksToMpc field/FFT workspace exceeds 127 bits");
            const int guard_bits = std::min(
                20, 127 - wide_bits - config_.fft_scale_bits - 2);
            auto transform_result = GetTransform(
                degree, config_.fft_scale_bits, wide_bits + guard_bits);
            if (!transform_result.ok()) return transform_result.status();
            const auto transform = transform_result.take_value();

            std::vector<std::uint64_t> provisional_real(count * slots);
            std::vector<std::uint64_t> provisional_imag(count * slots);
            struct LocalDecoded {
                std::vector<std::uint64_t> real;
                std::vector<std::uint64_t> imag;
            };
            std::deque<std::pair<std::size_t,
                                 std::future<Result<LocalDecoded>>>> pending;
            const std::size_t max_inflight =
                count > 1 && workers_.size() > 1 ? std::size_t{2}
                                                  : std::size_t{1};
            const auto collect_decoded = [&]() -> Result<void> {
                auto item = std::move(pending.front());
                pending.pop_front();
                auto decoded = item.second.get();
                if (!decoded.ok()) return decoded.status();
                const std::size_t offset = item.first * slots;
                std::copy(decoded.value().real.begin(),
                          decoded.value().real.end(),
                          provisional_real.begin() + offset);
                std::copy(decoded.value().imag.begin(),
                          decoded.value().imag.end(),
                          provisional_imag.begin() + offset);
                return {};
            };
            seal::Evaluator evaluator(*profile->context);
            std::unique_ptr<seal::Decryptor> decryptor;
            if (role_ == PartyRole::kClient)
                decryptor = std::make_unique<seal::Decryptor>(
                    *profile->context, profile->secret_key.value());

            for (std::size_t ct_index = 0; ct_index < count; ++ct_index) {
                std::vector<std::uint64_t> field_share(degree);
                if (role_ == PartyRole::kServer) {
                    const std::string domain = "hera/c2m/A/" +
                        SessionLabel(frame) + "/" +
                        std::to_string(ct_index);
                    auto stream_result = randomness_->OpenStream(domain);
                    if (!stream_result.ok()) return stream_result.status();
                    auto mask = SampleUniformPrime(*stream_result.value(), degree,
                                                   modulus);
                    stats_.a_mask_bytes += degree * sizeof(std::uint64_t);
                    seal::Plaintext mask_plain(degree);
                    std::copy(mask.begin(), mask.end(), mask_plain.data());
                    seal::util::ntt_negacyclic_harvey(
                        mask_plain.data(), terminal->small_ntt_tables()[0]);
                    mask_plain.parms_id() = terminal->parms_id();
                    mask_plain.scale() = ciphertexts[ct_index].scale();
                    seal::Ciphertext masked = ciphertexts[ct_index];
                    evaluator.add_plain_inplace(masked, mask_plain);
                    SendSealObject(*transport_, masked, wire_compression_);
                    for (std::size_t i = 0; i < degree; ++i)
                        field_share[i] = mask[i] == 0 ? 0 : modulus - mask[i];
                } else {
                    // This is the only decrypt call in the protocol. The
                    // received ciphertext has already been hidden by r_A.
                    auto masked = ReceiveSealObject<seal::Ciphertext>(
                        *transport_, *profile->context);
                    if (masked.parms_id() != terminal->parms_id())
                        return Status::FailedPrecondition(
                            "received masked ciphertext is not terminal-level");
                    seal::Plaintext decrypted;
                    decryptor->decrypt(masked, decrypted);
                    seal::util::inverse_ntt_negacyclic_harvey(
                        decrypted.data(), terminal->small_ntt_tables()[0]);
                    std::copy(decrypted.data(), decrypted.data() + degree,
                              field_share.begin());
                }

                auto ring = CenteredFieldToRing(field_share, modulus, field_bits);
                auto wide = SignedExtend(ring, field_bits, wide_bits);
                const int scale_shift =
                    frame.he_scale_bits - config_.scale_bits;
                if (scale_shift < 0)
                    return Status::FailedPrecondition(
                        "CkksToMpc cannot increase the target share scale");
                const auto local_work =
                    [transform, wide = std::move(wide), slots,
                     guard_bits, wide_bits, scale_shift,
                     ring_bits = config_.ring_bits,
                     negative = role_ == PartyRole::kClient]() mutable
                    -> Result<LocalDecoded> {
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
                        return result;
                    };
                pending.emplace_back(
                    ct_index,
                    std::async(max_inflight > 1 ? std::launch::async
                                                : std::launch::deferred,
                               std::move(local_work)));
                ++stats_.local_transform_tasks;
                stats_.local_transform_peak_inflight = std::max(
                    stats_.local_transform_peak_inflight, pending.size());
                if (pending.size() >= max_inflight) {
                    auto collected = collect_decoded();
                    if (!collected.ok()) return collected.status();
                }
            }
            while (!pending.empty()) {
                auto collected = collect_decoded();
                if (!collected.ok()) return collected.status();
            }

            std::vector<std::uint64_t> output_real(count * slots);
            std::vector<std::uint64_t> output_imag(count * slots);
            const std::uint64_t ring_mask = LowMask(config_.ring_bits);
            if (role_ == PartyRole::kServer) {
                for (std::size_t ct = 0; ct < count; ++ct) {
                    auto real_stream = randomness_->OpenStream(
                        "hera/c2m/Z-real/" + SessionLabel(frame) +
                        "/" + std::to_string(ct));
                    auto imag_stream = randomness_->OpenStream(
                        "hera/c2m/Z-imag/" + SessionLabel(frame) +
                        "/" + std::to_string(ct));
                    if (!real_stream.ok()) return real_stream.status();
                    if (!imag_stream.ok()) return imag_stream.status();
                    auto z_real = SampleUniformRing(*real_stream.value(), slots,
                                                    config_.ring_bits);
                    auto z_imag = SampleUniformRing(*imag_stream.value(), slots,
                                                    config_.ring_bits);
                    const std::size_t offset = ct * slots;
                    std::vector<std::uint64_t> delta_real(slots), delta_imag(slots);
                    for (std::size_t slot = 0; slot < slots; ++slot) {
                        output_real[offset + slot] = z_real[slot];
                        output_imag[offset + slot] = z_imag[slot];
                        delta_real[slot] =
                            (provisional_real[offset + slot] - z_real[slot]) &
                            ring_mask;
                        delta_imag[slot] =
                            (provisional_imag[offset + slot] - z_imag[slot]) &
                            ring_mask;
                    }
                    transport_->Send(delta_real.data(),
                                     delta_real.size() * sizeof(std::uint64_t));
                    transport_->Send(delta_imag.data(),
                                     delta_imag.size() * sizeof(std::uint64_t));
                    stats_.z_mask_bytes +=
                        2 * slots * sizeof(std::uint64_t);
                }
            } else {
                for (std::size_t ct = 0; ct < count; ++ct) {
                    const std::size_t offset = ct * slots;
                    std::vector<std::uint64_t> delta_real(slots), delta_imag(slots);
                    transport_->Receive(delta_real.data(),
                                        delta_real.size() * sizeof(std::uint64_t));
                    transport_->Receive(delta_imag.data(),
                                        delta_imag.size() * sizeof(std::uint64_t));
                    for (std::size_t slot = 0; slot < slots; ++slot) {
                        output_real[offset + slot] =
                            (provisional_real[offset + slot] + delta_real[slot]) &
                            ring_mask;
                        output_imag[offset + slot] =
                            (provisional_imag[offset + slot] + delta_imag[slot]) &
                            ring_mask;
                    }
                }
            }

            const TensorShape physical_shape{{count, slots}};
            const NumericFormat numeric{config_.ring_bits, config_.scale_bits};
            ComplexMpcSharePair result{
                {std::move(output_real), physical_shape, numeric,
                 ComplexComponent::kReal},
                {std::move(output_imag), physical_shape, numeric,
                 ComplexComponent::kImaginary}};
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
        const MpcToCkksTarget &target)
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
            const int shift = target.scale_bits - config_.scale_bits;
            const int base_bits = config_.ring_bits +
                                  config_.statistical_security_bits;
            const int work_bits = base_bits + shift;
            if (base_bits + config_.fft_scale_bits + 2 > 127 ||
                work_bits > 127)
                return Status::FailedPrecondition(
                    "MpcToCkks scale/FFT workspace exceeds 127 bits");

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
                    peer_profile != target.parameter_profile) {
                    const std::uint32_t rejected = 0;
                    transport_->Send(&rejected, sizeof(rejected));
                    return Status::FailedPrecondition(
                        "MpcToCkks profile/protocol handshake mismatch");
                }
                const std::uint32_t acknowledged = 1;
                transport_->Send(&acknowledged, sizeof(acknowledged));
            }

            const int guard_bits = std::min(
                20, 127 - base_bits - config_.fft_scale_bits - 2);
            auto transform_result = GetTransform(
                degree, config_.fft_scale_bits, base_bits + guard_bits);
            if (!transform_result.ok()) return transform_result.status();
            const auto transform = transform_result.take_value();
            seal::Evaluator evaluator(*profile.context);
            std::unique_ptr<seal::Encryptor> encryptor;
            if (role_ == PartyRole::kClient)
                encryptor = std::make_unique<seal::Encryptor>(
                    *profile.context, profile.secret_key.value());
            std::vector<seal::Ciphertext> outputs;
            if (role_ == PartyRole::kServer) outputs.reserve(count);
            std::deque<std::pair<std::size_t,
                                 std::future<seal::Plaintext>>> pending_rns;
            const std::size_t max_inflight =
                count > 1 && workers_.size() > 1 ? std::size_t{2}
                                                  : std::size_t{1};
            const auto consume_rns = [&]() -> Result<void> {
                auto item = std::move(pending_rns.front());
                pending_rns.pop_front();
                auto plaintext = item.second.get();
                plaintext.parms_id() = target_id;
                plaintext.scale() = std::ldexp(1.0, target.scale_bits);
                if (role_ == PartyRole::kClient) {
                    seal::Ciphertext encrypted;
                    encryptor->encrypt_symmetric(plaintext, encrypted);
                    SendSealObject(*transport_, encrypted, wire_compression_);
                } else {
                    auto encrypted = ReceiveSealObject<seal::Ciphertext>(
                        *transport_, *profile.context);
                    if (encrypted.parms_id() != target_id)
                        return Status::FailedPrecondition(
                            "MpcToCkks client ciphertext level mismatch");
                    evaluator.add_plain_inplace(encrypted, plaintext);
                    outputs.push_back(std::move(encrypted));
                }
                return {};
            };

            for (std::size_t ct = 0; ct < count; ++ct) {
                const std::size_t offset = ct * slots;
                std::vector<std::uint64_t> local_real(
                    local_shares.real.local_share.begin() + offset,
                    local_shares.real.local_share.begin() + offset + slots);
                std::vector<std::uint64_t> local_imag(
                    local_shares.imag.local_share.begin() + offset,
                    local_shares.imag.local_share.begin() + offset + slots);
                auto wide_real = SignedExtend(local_real, config_.ring_bits,
                                              base_bits);
                auto wide_imag = SignedExtend(local_imag, config_.ring_bits,
                                              base_bits);
                SlotSharePair pair;
                pair.real.resize(slots);
                pair.imag.resize(slots);
                for (std::size_t slot = 0; slot < slots; ++slot) {
                    pair.real[slot] = WideShiftLeft(
                        wide_real[slot], guard_bits, base_bits + guard_bits);
                    pair.imag[slot] = WideShiftLeft(
                        wide_imag[slot], guard_bits, base_bits + guard_bits);
                }
                auto coefficients_result = transform->Encode(pair);
                if (!coefficients_result.ok()) return coefficients_result.status();
                auto coefficients = coefficients_result.take_value();
                std::vector<std::uint64_t> coefficient_ring(degree);
                for (std::size_t i = 0; i < coefficients.size(); ++i) {
                    auto coefficient = WideLocalTruncate(
                        coefficients[i], guard_bits, base_bits + guard_bits,
                        role_ == PartyRole::kClient);
                    coefficient_ring[i] =
                        coefficient.low & LowMask(config_.ring_bits);
                }
                // The local FFT truncations do not preserve the structured
                // statistical extension required by probabilistic
                // ring-to-field. Return to the semantic 2^ell ring and
                // securely re-extend immediately before RNS conversion.
                coefficients = SignedExtend(coefficient_ring,
                                            config_.ring_bits, base_bits);
                for (auto &coefficient : coefficients)
                    coefficient = WideShiftLeft(coefficient, shift, work_bits);
                pending_rns.emplace_back(
                    ct,
                    std::async(max_inflight > 1 ? std::launch::async
                                                : std::launch::deferred,
                               [this, coefficients = std::move(coefficients),
                                work_bits, context_data]() {
                                   return BuildBlbRnsPlaintextShare(
                                       coefficients, work_bits, *context_data);
                               }));
                ++stats_.local_transform_tasks;
                stats_.local_transform_peak_inflight = std::max(
                    stats_.local_transform_peak_inflight, pending_rns.size());
                if (pending_rns.size() >= max_inflight) {
                    auto consumed = consume_rns();
                    if (!consumed.ok()) return consumed.status();
                }
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
    std::uint64_t call_sequence_{};
    ConversionStats stats_;
};

CkksMpcConverter::CkksMpcConverter(
    PartyRole role, ConversionConfig config, HeContextRegistry &contexts,
    std::shared_ptr<IProtocolTransport> transport,
    std::shared_ptr<IRandomSource> randomness, std::string peer_address,
    int worker_port, std::size_t threads,
    seal::compr_mode_type wire_compression)
    : impl_(std::make_unique<Impl>(
          role, config, contexts, std::move(transport), std::move(randomness),
          std::move(peer_address), worker_port, threads, wire_compression))
{}

CkksMpcConverter::~CkksMpcConverter() = default;
CkksMpcConverter::CkksMpcConverter(CkksMpcConverter &&) noexcept = default;
CkksMpcConverter &CkksMpcConverter::operator=(CkksMpcConverter &&) noexcept =
    default;

Result<ComplexMpcSharePair> CkksMpcConverter::CkksToMpc(
    const std::optional<SealCipherBatch> &server_input)
{
    return impl_->CkksToMpc(server_input);
}

Result<std::optional<SealCipherBatch>> CkksMpcConverter::MpcToCkks(
    const ComplexMpcSharePair &local_shares,
    const MpcToCkksTarget &target)
{
    return impl_->MpcToCkks(local_shares, target);
}

ConversionStats CkksMpcConverter::stats() const noexcept
{
    return impl_->stats();
}

void CkksMpcConverter::Cancel() noexcept
{
    if (impl_) impl_->Cancel();
}

} // namespace hera
