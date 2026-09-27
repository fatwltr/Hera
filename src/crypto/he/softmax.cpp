#include "hera/softmax.h"
#include "hera/detail/mpc_session_internal.h"
#include "hera/detail/parallel_ranges.h"

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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>

namespace hera {
namespace {

using Clock = std::chrono::steady_clock;

constexpr double kScaleTolerance = 1.0e-3;

std::uint64_t RingMask(int bits)
{
    return bits == 64 ? std::numeric_limits<std::uint64_t>::max()
                      : (std::uint64_t{1} << bits) - 1;
}

std::uint64_t EncodeFixed(double value, const NumericFormat &format)
{
    const long double scaled = std::ldexp(
        static_cast<long double>(value), format.scale_bits);
    const auto integer = static_cast<__int128>(std::llround(scaled));
    return static_cast<std::uint64_t>(integer) & RingMask(format.ring_bits);
}

bool CloseScale(double actual, double expected)
{
    const double largest = std::max(std::abs(actual), std::abs(expected));
    return std::isfinite(largest) && largest != 0.0 &&
           std::abs(actual - expected) <= largest * kScaleTolerance;
}

Result<std::size_t> LevelOf(const HeContext &context,
                            seal::parms_id_type parms_id)
{
    std::size_t level = 0;
    for (auto data = context.context->first_context_data(); data;
         data = data->next_context_data(), ++level)
        if (data->parms_id() == parms_id) return level;
    return Status::Invalid("ciphertext is outside the Block-3 context");
}

double DroppedPrime(const HeContext &context, seal::parms_id_type parms_id)
{
    const auto data = context.context->get_context_data(parms_id);
    if (!data || !data->next_context_data())
        throw std::invalid_argument("Block 3 has no remaining rescale prime");
    return static_cast<double>(data->parms().coeff_modulus().back().value());
}

Result<void> NormalizeScale(seal::Ciphertext &ciphertext, double target)
{
    if (!CloseScale(ciphertext.scale(), target))
        return Status::FailedPrecondition(
            "Block-3 CKKS scale drift exceeds configured tolerance");
    // This canonicalizes only the small floating representation drift after
    // a mathematically valid rescale; it is never used to repair a bad scale.
    ciphertext.scale() = target;
    return {};
}

Result<std::vector<bool>> ValidMask(const TransformerLayerConfig &layer,
                                    const PublicAttentionMask &mask)
{
    if (mask.kind != layer.attention_mask)
        return Status::Invalid("Softmax mask kind differs from layer config");
    const auto tokens = layer.tokens;
    std::vector<bool> valid(tokens * tokens, true);
    if (mask.kind == AttentionMaskKind::kPublicCausal) {
        for (std::size_t row = 0; row < tokens; ++row)
            for (std::size_t column = row + 1; column < tokens; ++column)
                valid[row * tokens + column] = false;
    } else if (mask.kind == AttentionMaskKind::kPublicPadding) {
        if (mask.shape != TensorShape{{tokens, tokens}} ||
            mask.additive_values.size() != tokens * tokens)
            return Status::Invalid(
                "public padding mask must contain tokens*tokens values");
        for (std::size_t i = 0; i < valid.size(); ++i) {
            if (!std::isfinite(mask.additive_values[i])) {
                valid[i] = mask.additive_values[i] > 0.0;
            } else {
                valid[i] = mask.additive_values[i] >= 0.0;
            }
        }
    } else if (mask.kind != AttentionMaskKind::kNone) {
        return Status::Invalid("unsupported Softmax attention mask");
    }
    for (std::size_t row = 0; row < tokens; ++row) {
        bool any = false;
        for (std::size_t column = 0; column < tokens; ++column)
            any = any || valid[row * tokens + column];
        if (!any)
            return Status::FailedPrecondition(
                "Softmax public mask contains an all-masked row");
    }
    return valid;
}

std::size_t RowMajorIndex(std::size_t row, std::size_t head,
                          std::size_t column,
                          const TransformerLayerConfig &layer)
{
    return (row * layer.q_heads + head) * layer.tokens + column;
}

std::size_t Block2Index(std::size_t head, std::size_t row,
                        std::size_t column,
                        const TransformerLayerConfig &layer)
{
    return (head * layer.tokens + row) * layer.tokens + column;
}

Result<PhysicalSlotMap> ToConversionMap(const HePackingPlan &packing)
{
    auto logical_count = packing.logical_shape.ElementCount();
    if (!logical_count.ok()) return logical_count.status();
    PhysicalSlotMap map;
    map.ciphertext_count = packing.ciphertext_count;
    map.slots_per_ciphertext = packing.slot_count;
    map.logical_element_count = logical_count.value();
    map.assignments.resize(map.ciphertext_count * map.slots_per_ciphertext);
    for (std::size_t logical = 0;
         logical < packing.logical_locations.size(); ++logical) {
        if (packing.logical_locations[logical].size() != 1)
            return Status::FailedPrecondition(
                "Softmax score conversion requires one physical score lane");
        const auto &location = packing.logical_locations[logical].front();
        auto &assignment = map.assignments[
            location.ciphertext * map.slots_per_ciphertext + location.slot];
        if (location.component == ComplexLane::kReal)
            assignment.real_logical_index = logical;
        else
            assignment.imaginary_logical_index = logical;
    }
    auto valid = map.Validate();
    if (!valid.ok()) return valid.status();
    return map;
}

bool SamePacking(const HePackingPlan &left, const HePackingPlan &right)
{
    if (left.layout != right.layout ||
        left.logical_shape != right.logical_shape ||
        left.slot_count != right.slot_count ||
        left.ciphertext_count != right.ciphertext_count ||
        left.logical_locations.size() != right.logical_locations.size())
        return false;
    for (std::size_t logical = 0;
         logical < left.logical_locations.size(); ++logical) {
        const auto &a = left.logical_locations[logical];
        const auto &b = right.logical_locations[logical];
        if (a.size() != b.size()) return false;
        for (std::size_t i = 0; i < a.size(); ++i)
            if (a[i].ciphertext != b[i].ciphertext ||
                a[i].slot != b[i].slot ||
                a[i].component != b[i].component ||
                a[i].packed_query_head != b[i].packed_query_head)
                return false;
    }
    return true;
}

Result<MpcShareTensor> CompactScores(
    const ComplexMpcSharePair &physical, const HePackingPlan &packing,
    const TransformerLayerConfig &layer)
{
    const auto pair_valid = physical.Validate();
    if (!pair_valid.ok()) return pair_valid.status();
    const std::size_t expected = layer.q_heads * layer.tokens * layer.tokens;
    if (packing.logical_locations.size() != expected)
        return Status::Invalid("Block-2 score packing has the wrong size");
    std::vector<std::uint64_t> rows(expected);
    for (std::size_t head = 0; head < layer.q_heads; ++head) {
        for (std::size_t row = 0; row < layer.tokens; ++row) {
            for (std::size_t column = 0; column < layer.tokens; ++column) {
                const auto source = Block2Index(head, row, column, layer);
                const auto &locations = packing.Locations(source);
                if (locations.size() != 1)
                    return Status::FailedPrecondition(
                        "attention score has replicated physical locations");
                const auto &location = locations.front();
                const auto physical_index =
                    location.ciphertext * packing.slot_count + location.slot;
                rows[RowMajorIndex(row, head, column, layer)] =
                    location.component == ComplexLane::kReal
                        ? physical.real.local_share[physical_index]
                        : physical.imag.local_share[physical_index];
            }
        }
    }
    NumericFormat numeric = physical.real.numeric;
    numeric.is_signed = true;
    return MpcShareTensor{std::move(rows),
                          {{layer.tokens, layer.q_heads, layer.tokens}},
                          numeric, ComplexComponent::kReal};
}

ComplexMpcSharePair ScatterDenseReal(const MpcShareTensor &logical,
                                     const PhysicalSlotMap &map)
{
    const std::size_t count = map.ciphertext_count * map.slots_per_ciphertext;
    ComplexMpcSharePair pair{
        {std::vector<std::uint64_t>(count),
         {{map.ciphertext_count, map.slots_per_ciphertext}}, logical.numeric,
         ComplexComponent::kReal},
        {std::vector<std::uint64_t>(count),
         {{map.ciphertext_count, map.slots_per_ciphertext}}, logical.numeric,
         ComplexComponent::kImaginary}};
    for (std::size_t physical = 0; physical < map.assignments.size(); ++physical) {
        const auto index = map.assignments[physical].real_logical_index;
        if (index != kUnusedLogicalIndex)
            pair.real.local_share[physical] = logical.local_share[index];
    }
    return pair;
}

Result<MpcShareTensor> CompactDenseReal(
    const ComplexMpcSharePair &physical, const PhysicalSlotMap &map,
    const TensorShape &shape, bool is_signed)
{
    auto valid = physical.Validate();
    if (!valid.ok()) return valid.status();
    std::vector<std::uint64_t> logical(map.logical_element_count);
    for (std::size_t physical_index = 0;
         physical_index < map.assignments.size(); ++physical_index) {
        const auto logical_index =
            map.assignments[physical_index].real_logical_index;
        if (logical_index != kUnusedLogicalIndex)
            logical[logical_index] = physical.real.local_share[physical_index];
    }
    auto numeric = physical.real.numeric;
    numeric.is_signed = is_signed;
    return MpcShareTensor{std::move(logical), shape, numeric,
                          ComplexComponent::kReal};
}

} // namespace

Result<void> Block3LevelPlanner::Validate(
    const HeParameterConfig &parameters, const SoftmaxConfig &softmax)
{
    const std::size_t depth = 1 + softmax.exp_squarings;
    if (parameters.terminal_prime_count != 1)
        return Status::FailedPrecondition(
            "Block 3 requires a single-prime conversion terminal level");
    if (parameters.coefficient_modulus_bits.size() != depth + 1)
        return Status::FailedPrecondition(
            "Block 3 modulus chain must contain depth+1 data primes");
    if (parameters.declared_depth != depth)
        return Status::FailedPrecondition(
            "Block 3 declared depth differs from exponent schedule");
    if (parameters.input_scale_bits != parameters.default_scale_bits ||
        parameters.output_scale_bits != parameters.default_scale_bits)
        return Status::FailedPrecondition(
            "Block 3 currently requires one configured CKKS scale");
    return {};
}

Result<PhysicalSlotMap> SoftmaxPackingPlanner::RealRows(
    const TransformerLayerConfig &layer,
    std::size_t slots_per_ciphertext,
    bool pad_rows_to_power_of_two)
{
    if (layer.tokens == 0 || layer.q_heads == 0 ||
        slots_per_ciphertext == 0)
        return Status::Invalid("Softmax row packing has an empty dimension");
    std::size_t stride = layer.tokens;
    if (pad_rows_to_power_of_two) {
        stride = 1;
        while (stride < layer.tokens) {
            if (stride > std::numeric_limits<std::size_t>::max() / 2)
                return Status::Invalid("Softmax row stride overflows size_t");
            stride *= 2;
        }
    }
    if (layer.tokens > std::numeric_limits<std::size_t>::max() /
            layer.q_heads)
        return Status::Invalid("Softmax logical size overflows size_t");
    const auto rows = layer.tokens * layer.q_heads;
    if (rows > std::numeric_limits<std::size_t>::max() / stride)
        return Status::Invalid("Softmax physical size overflows size_t");
    const auto physical_count = rows * stride;
    PhysicalSlotMap map;
    map.slots_per_ciphertext = slots_per_ciphertext;
    map.ciphertext_count =
        (physical_count + slots_per_ciphertext - 1) /
        slots_per_ciphertext;
    map.logical_element_count = rows * layer.tokens;
    map.assignments.resize(map.ciphertext_count * slots_per_ciphertext);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < layer.tokens; ++column) {
            const auto physical = row * stride + column;
            map.assignments[physical].real_logical_index =
                row * layer.tokens + column;
        }
    }
    auto valid = map.Validate();
    if (!valid.ok()) return valid.status();
    return map;
}

Result<SealCipherBatch> SoftmaxExponentKernel::Evaluate(
    const SealCipherBatch &shifted_scores, const HeContext &context,
    const SoftmaxConfig &config, SoftmaxExecutionReport *report,
    std::size_t he_workers)
{
    const auto started = Clock::now();
    try {
        auto planner = Block3LevelPlanner::Validate(context.parameters, config);
        if (!planner.ok()) return planner.status();
        if (shifted_scores.parameter_profile != context.profile ||
            shifted_scores.logical_layout != Layout::kSoftmaxRows)
            return Status::Invalid(
                "Block-3 exponent input has wrong profile or layout");
        auto map_valid = shifted_scores.slot_map.Validate();
        if (!map_valid.ok()) return map_valid.status();
        if (shifted_scores.ciphertexts.size() !=
            shifted_scores.slot_map.ciphertext_count)
            return Status::Invalid("Block-3 ciphertext/map count mismatch");
        for (const auto &entry : shifted_scores.slot_map.assignments)
            if (entry.imaginary_logical_index != kUnusedLogicalIndex)
                return Status::Invalid(
                    "Block-3 exponent requires real-only CKKS packing");

        SealCipherBatch output = shifted_scores;
        const double target = std::ldexp(
            1.0, context.parameters.default_scale_bits);
        auto parallel = detail::ParallelRanges(
            output.ciphertexts.size(), std::min<std::size_t>(he_workers, 4),
            1, [&](std::size_t begin, std::size_t end) -> Result<void> {
                seal::Evaluator evaluator(*context.context);
                seal::CKKSEncoder encoder(*context.context);
                for (std::size_t index = begin; index < end; ++index) {
                    auto &ciphertext = output.ciphertexts[index];
                    auto level = LevelOf(context, ciphertext.parms_id());
                    if (!level.ok()) return level.status();
                    if (level.value() != 0 ||
                        !CloseScale(ciphertext.scale(), std::ldexp(
                            1.0, context.parameters.input_scale_bits)))
                        return Status::FailedPrecondition(
                            "Block-3 exponent input must be fresh at input scale");

                    const double plain_scale = target * DroppedPrime(
                        context, ciphertext.parms_id()) / ciphertext.scale();
                    seal::Plaintext divisor;
                    encoder.encode(std::ldexp(1.0, -config.exp_divisor_bits),
                                   ciphertext.parms_id(), plain_scale, divisor);
                    evaluator.multiply_plain_inplace(ciphertext, divisor);
                    evaluator.rescale_to_next_inplace(ciphertext);
                    auto normalized = NormalizeScale(ciphertext, target);
                    if (!normalized.ok()) return normalized.status();

                    seal::Plaintext one;
                    encoder.encode(1.0, ciphertext.parms_id(), target, one);
                    evaluator.add_plain_inplace(ciphertext, one);
                    for (std::size_t i = 0; i < config.exp_squarings; ++i) {
                        evaluator.square_inplace(ciphertext);
                        evaluator.relinearize_inplace(
                            ciphertext, context.relinearization_keys);
                        evaluator.rescale_to_next_inplace(ciphertext);
                        normalized = NormalizeScale(ciphertext, target);
                        if (!normalized.ok()) return normalized.status();
                    }
                    level = LevelOf(context, ciphertext.parms_id());
                    if (!level.ok()) return level.status();
                    if (level.value() != 1 + config.exp_squarings ||
                        ciphertext.parms_id() !=
                            context.context->last_parms_id())
                        return Status::FailedPrecondition(
                            "Block-3 exponent did not finish at terminal level");
                }
                return {};
            }, detail::EstimatedHeWorkerBytes(
                context.parameters.polynomial_degree,
                context.parameters.coefficient_modulus_bits.size(), 12));
        if (!parallel.ok()) return parallel.status();
        if (report) {
            report->ciphertext_plain_multiplications +=
                output.ciphertexts.size();
            report->ciphertext_multiplications +=
                output.ciphertexts.size() * config.exp_squarings;
            report->relinearizations +=
                output.ciphertexts.size() * config.exp_squarings;
            report->rescales += output.ciphertexts.size() *
                (1 + config.exp_squarings);
            report->he_milliseconds += std::chrono::duration<double,
                std::milli>(Clock::now() - started).count();
        }
        return output;
    } catch (const std::exception &error) {
        return Status::Internal(std::string("Block-3 exponent: ") +
                                error.what());
    }
}

class MpcSoftmaxRuntime::Impl {
public:
    explicit Impl(std::shared_ptr<SharedMpcSession> session)
        : session_(std::move(session)), native_(session_->Acquire()),
          party_(native_->party())
    {}

    FixArray Secret(const std::vector<std::uint64_t> &shares,
                    const NumericFormat &numeric)
    {
        if (shares.empty() || shares.size() >
                static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::invalid_argument("SCI array size is unsupported");
        FixArray result(party_, static_cast<int>(shares.size()),
                        numeric.is_signed, numeric.ring_bits,
                        numeric.scale_bits);
        const auto mask = RingMask(numeric.ring_bits);
        for (std::size_t i = 0; i < shares.size(); ++i)
            result.data[i] = shares[i] & mask;
        return result;
    }

    std::vector<std::uint64_t> Shares(const FixArray &array)
    {
        return std::vector<std::uint64_t>(array.data,
                                         array.data + array.size);
    }

    std::uint64_t communication() const
    { return native_->communication_bytes(); }
    std::uint64_t rounds() const { return native_->rounds(); }
    void Cancel() noexcept { session_->Cancel(); }
    int party() const { return party_; }
    FixOp &fix() { return native_->fix(); }

    FixArray MultiplyUnsigned(const FixArray &left, const FixArray &right,
                              int output_bits)
    {
        if (left.signed_ || right.signed_ || left.size != right.size ||
            output_bits < left.ell || output_bits < right.ell ||
            output_bits >= 64)
            throw std::invalid_argument(
                "unsupported unsigned multiplication contract");
        auto product = fix().mul(left, right, output_bits);
        product.s = left.s + right.s;
        product.signed_ = false;
        return product;
    }

    FixArray ReciprocalUnitInterval(const FixArray &denominator,
                                    int output_bits, int output_scale)
    {
        if (denominator.signed_ || denominator.s != output_scale ||
            output_bits <= output_scale || output_bits + 1 >= 64 ||
            output_bits + output_scale >= 64)
            throw std::invalid_argument(
                "unsupported fixed-point reciprocal contract");

        NumericFormat input_numeric{
            output_bits, output_scale, false};
        std::vector<std::uint64_t> numerator_shares(denominator.size, 0);
        if (party_ == sci::ALICE)
            std::fill(numerator_shares.begin(), numerator_shares.end(),
                      std::uint64_t{1} << output_scale);
        auto numerator = Secret(numerator_shares, input_numeric);

        // Restoring binary division computes floor((1<<2s)/denominator).
        // Softmax guarantees denominator >= 1.0, hence the reciprocal lies
        // in [0,1] and needs one integer bit plus s fractional bits.
        const int work_bits = output_bits + 1;
        auto remainder = fix().extend(numerator, work_bits);
        auto divisor = fix().extend(denominator, work_bits);
        auto divisor_minus_one = fix().sub(divisor, std::uint64_t{1});

        // This SCI fork has an incorrect Boolean NOT implementation.  On an
        // integer ring, a >= b is exactly a > b-1 and avoids NOT entirely.
        auto take = fix().GT(remainder, divisor_minus_one);
        auto difference = fix().sub(remainder, divisor);
        remainder = fix().if_else(take, difference, remainder);
        auto quotient = fix().B2A(take, false, output_bits);
        quotient = fix().mul(
            quotient, std::uint64_t{1} << output_scale, output_bits);

        for (int bit = output_scale - 1; bit >= 0; --bit) {
            auto doubled = fix().mul(remainder, std::uint64_t{2}, work_bits);
            take = fix().GT(doubled, divisor_minus_one);
            difference = fix().sub(doubled, divisor);
            remainder = fix().if_else(take, difference, doubled);
            auto quotient_bit = fix().B2A(take, false, output_bits);
            auto contribution = fix().mul(
                quotient_bit, std::uint64_t{1} << bit, output_bits);
            quotient = fix().add(quotient, contribution);
        }
        quotient.s = output_scale;
        quotient.signed_ = false;
        return quotient;
    }

    bool RevealAnyZero(const FixArray &values)
    {
        std::vector<std::uint64_t> zero_shares(values.size, 0);
        NumericFormat format{values.ell, values.s, values.signed_};
        auto secret_zero = Secret(zero_shares, format);
        auto zero = fix().EQ(values, secret_zero);
        BoolArray any_zero(party_, 1);
        any_zero.data[0] = zero.data[0];
        for (int i = 1; i < zero.size; ++i) {
            BoolArray next(party_, 1);
            next.data[0] = zero.data[i];
            auto either = fix().bool_op->XOR(any_zero, next);
            auto both = fix().bool_op->AND(any_zero, next);
            any_zero = fix().bool_op->XOR(either, both);
        }
        auto server_result = fix().bool_op->output(
            sci::ALICE, any_zero);
        std::uint8_t any = 0;
        if (party_ == sci::ALICE) {
            any = server_result.data[0] & std::uint8_t{1};
            native_->fix().iopack->io->send_data(&any, sizeof(any));
        } else {
            native_->fix().iopack->io->recv_data(&any, sizeof(any));
        }
        return any != 0;
    }

private:
    std::shared_ptr<SharedMpcSession> session_;
    std::shared_ptr<SharedMpcSession::Impl> native_;
    int party_;
};

MpcSoftmaxRuntime::MpcSoftmaxRuntime(
    PartyRole role, std::string peer_address, int port)
    : session_(std::make_shared<SharedMpcSession>(
          role, std::move(peer_address), port))
{}

MpcSoftmaxRuntime::MpcSoftmaxRuntime(
    std::shared_ptr<SharedMpcSession> session)
    : session_(std::move(session))
{
    if (!session_) throw std::invalid_argument("MPC session is null");
}

MpcSoftmaxRuntime::~MpcSoftmaxRuntime() = default;
MpcSoftmaxRuntime::MpcSoftmaxRuntime(MpcSoftmaxRuntime &&) noexcept = default;
MpcSoftmaxRuntime &MpcSoftmaxRuntime::operator=(
    MpcSoftmaxRuntime &&) noexcept = default;

void MpcSoftmaxRuntime::ReleaseSession()
{
    std::atomic_store(&impl_, std::shared_ptr<Impl>{});
}

void MpcSoftmaxRuntime::Cancel() noexcept
{
    if (session_) session_->Cancel();
}

Result<MpcShareTensor> MpcSoftmaxRuntime::CenterAndMask(
    const MpcShareTensor &scores, const TransformerLayerConfig &layer,
    const PublicAttentionMask &mask, const SoftmaxConfig &config,
    SoftmaxExecutionReport *report)
{
    const auto started = Clock::now();
    try {
        if (!std::atomic_load(&impl_))
            std::atomic_store(&impl_, std::make_shared<Impl>(
                session_));
        auto score_valid = scores.Validate();
        if (!score_valid.ok()) return score_valid.status();
        if (scores.physical_shape !=
                TensorShape{{layer.tokens, layer.q_heads, layer.tokens}} ||
            !scores.numeric.is_signed)
            return Status::Invalid(
                "MPC Softmax scores require signed [tokens,heads,tokens]");
        auto validity = ValidMask(layer, mask);
        if (!validity.ok()) return validity.status();
        const auto before_comm = impl_->communication();
        const auto before_rounds = impl_->rounds();
        const auto ring_mask = RingMask(scores.numeric.ring_bits);
        const auto rows = layer.tokens * layer.q_heads;
        std::vector<std::uint64_t> maxima(rows);
        std::size_t comparisons = 0;
        // SCI FixOp::max treats each input FixArray as one row and batches
        // comparisons across rows. Group by public valid-column count so
        // padding never enters the maximum, yet rows with the same shape
        // share the same compare/mux rounds.
        std::map<std::size_t, std::vector<FixArray>> grouped_rows;
        std::map<std::size_t, std::vector<std::size_t>> grouped_indices;
        for (std::size_t row = 0; row < layer.tokens; ++row) {
            for (std::size_t head = 0; head < layer.q_heads; ++head) {
                std::vector<std::uint64_t> candidates;
                for (std::size_t column = 0; column < layer.tokens; ++column)
                    if (validity.value()[row * layer.tokens + column])
                        candidates.push_back(scores.local_share[
                            RowMajorIndex(row, head, column, layer)]);
                const auto width = candidates.size();
                grouped_rows[width].push_back(
                    impl_->Secret(candidates, scores.numeric));
                grouped_indices[width].push_back(
                    row * layer.q_heads + head);
                comparisons += candidates.size() - 1;
            }
        }
        for (auto &group : grouped_rows) {
            auto maximum = impl_->fix().max(group.second);
            const auto &indices = grouped_indices.at(group.first);
            for (std::size_t i = 0; i < indices.size(); ++i)
                maxima[indices[i]] = maximum.data[i];
        }

        const auto masked_value = EncodeFixed(config.masked_shift,
                                              scores.numeric);
        std::vector<std::uint64_t> shifted(scores.local_share.size());
        for (std::size_t row = 0; row < layer.tokens; ++row) {
            for (std::size_t head = 0; head < layer.q_heads; ++head) {
                const auto max_share = maxima[row * layer.q_heads + head];
                for (std::size_t column = 0; column < layer.tokens; ++column) {
                    const auto index = RowMajorIndex(row, head, column, layer);
                    if (validity.value()[row * layer.tokens + column])
                        shifted[index] =
                            (scores.local_share[index] - max_share) & ring_mask;
                    else
                        shifted[index] =
                            impl_->party() == sci::ALICE ? masked_value : 0;
                }
            }
        }
        if (report) {
            report->comparisons += comparisons;
            report->muxes += comparisons;
            report->mpc_communication_bytes +=
                impl_->communication() - before_comm;
            report->mpc_rounds += impl_->rounds() - before_rounds;
            report->mpc_milliseconds += std::chrono::duration<double,
                std::milli>(Clock::now() - started).count();
        }
        return MpcShareTensor{std::move(shifted), scores.physical_shape,
                              scores.numeric, ComplexComponent::kReal};
    } catch (const std::exception &error) {
        return Status::Internal(std::string("MPC Softmax centering: ") +
                                error.what());
    }
}

Result<MpcShareTensor> MpcSoftmaxRuntime::ClipAndNormalize(
    const MpcShareTensor &shifted_scores,
    const MpcShareTensor &exponentials,
    const TransformerLayerConfig &layer,
    const PublicAttentionMask &mask, const SoftmaxConfig &config,
    SoftmaxExecutionReport *report)
{
    const auto started = Clock::now();
    try {
        if (!std::atomic_load(&impl_))
            std::atomic_store(&impl_, std::make_shared<Impl>(
                session_));
        auto shifted_valid = shifted_scores.Validate();
        if (!shifted_valid.ok()) return shifted_valid.status();
        auto exp_valid = exponentials.Validate();
        if (!exp_valid.ok()) return exp_valid.status();
        const TensorShape expected{{layer.tokens, layer.q_heads,
                                    layer.tokens}};
        if (shifted_scores.physical_shape != expected ||
            exponentials.physical_shape != expected ||
            shifted_scores.numeric.ring_bits !=
                exponentials.numeric.ring_bits ||
            shifted_scores.numeric.scale_bits !=
                exponentials.numeric.scale_bits ||
            !shifted_scores.numeric.is_signed ||
            exponentials.numeric.is_signed)
            return Status::Invalid("MPC Softmax normalization contract mismatch");
        auto validity = ValidMask(layer, mask);
        if (!validity.ok()) return validity.status();

        const auto before_comm = impl_->communication();
        const auto before_rounds = impl_->rounds();
        auto shifted = impl_->Secret(shifted_scores.local_share,
                                     shifted_scores.numeric);
        auto exp = impl_->Secret(exponentials.local_share,
                                 exponentials.numeric);
        std::vector<std::uint64_t> threshold_shares(
            shifted_scores.local_share.size(), 0);
        if (impl_->party() == sci::ALICE)
            std::fill(threshold_shares.begin(), threshold_shares.end(),
                      EncodeFixed(config.clip_threshold -
                                      std::ldexp(1.0,
                                          -shifted_scores.numeric.scale_bits),
                                  shifted_scores.numeric));
        auto threshold = impl_->Secret(threshold_shares,
                                       shifted_scores.numeric);
        // This SCI fork's BoolOp::NOT toggles both XOR shares, so its GE
        // wrapper is not share-correct. On the discrete fixed-point grid,
        // x >= T is exactly x > (T - 1 ULP), which uses the proven LT/GT
        // primitive without a boolean negation.
        auto keep = impl_->fix().GT(shifted, threshold);
        auto clipped = impl_->fix().if_else(keep, exp, std::uint64_t{0});

        int token_bits = 0;
        for (std::size_t value = layer.tokens - 1; value != 0; value >>= 1U)
            ++token_bits;
        // One sign-free integer guard bit plus ceil(log2(tokens)) is enough
        // for a sum of probabilities represented at scale 2^s.  This is the
        // dynamic counterpart of BLB-SCI's exp_ell=19 choice for 32 tokens.
        const int accumulator_bits = std::min(
            exponentials.numeric.ring_bits,
            std::max(19, exponentials.numeric.scale_bits + token_bits + 1));
        auto clipped_short = impl_->fix().reduce(clipped, accumulator_bits);
        const auto ring_mask = RingMask(accumulator_bits);
        const auto rows = layer.tokens * layer.q_heads;
        std::vector<std::uint64_t> sums(rows, 0);
        for (std::size_t row = 0; row < layer.tokens; ++row) {
            for (std::size_t head = 0; head < layer.q_heads; ++head) {
                auto &sum = sums[row * layer.q_heads + head];
                for (std::size_t column = 0; column < layer.tokens; ++column)
                    sum = (sum + clipped_short.data[
                        RowMajorIndex(row, head, column, layer)]) & ring_mask;
            }
        }
        NumericFormat accumulator_numeric{
            accumulator_bits, exponentials.numeric.scale_bits, false};
        auto denominator = impl_->Secret(sums, accumulator_numeric);
        if (impl_->RevealAnyZero(denominator))
            return Status::FailedPrecondition(
                "Softmax denominator is zero after clipping");

        auto reciprocal = impl_->ReciprocalUnitInterval(
            denominator, accumulator_numeric.ring_bits,
            accumulator_numeric.scale_bits);
        std::vector<std::uint64_t> reciprocal_shares(
            shifted_scores.local_share.size());
        for (std::size_t row = 0; row < rows; ++row)
            std::fill_n(reciprocal_shares.begin() + row * layer.tokens,
                        layer.tokens, reciprocal.data[row]);
        auto reciprocal_expanded = impl_->Secret(
            reciprocal_shares, accumulator_numeric);
        auto product = impl_->MultiplyUnsigned(
            clipped_short, reciprocal_expanded,
            accumulator_numeric.ring_bits +
                accumulator_numeric.scale_bits);
        auto probabilities_short = impl_->fix().truncate_reduce(
            product, accumulator_numeric.scale_bits);
        auto probabilities = impl_->fix().extend(
            probabilities_short, exponentials.numeric.ring_bits);
        probabilities.signed_ = false;

        NumericFormat unsigned_numeric = exponentials.numeric;
        unsigned_numeric.is_signed = false;

        if (report) {
            const auto reciprocal_steps = rows *
                (static_cast<std::size_t>(accumulator_numeric.scale_bits) + 1);
            report->comparisons += shifted_scores.local_share.size() +
                                   reciprocal_steps;
            report->muxes += shifted_scores.local_share.size() +
                             reciprocal_steps;
            report->row_sums += rows;
            report->reciprocals += rows;
            report->secure_multiplications +=
                shifted_scores.local_share.size();
            report->mpc_communication_bytes +=
                impl_->communication() - before_comm;
            report->mpc_rounds += impl_->rounds() - before_rounds;
            report->mpc_milliseconds += std::chrono::duration<double,
                std::milli>(Clock::now() - started).count();
        }
        return MpcShareTensor{impl_->Shares(probabilities), expected,
                              unsigned_numeric, ComplexComponent::kReal};
    } catch (const std::exception &error) {
        return Status::Internal(std::string("MPC Softmax normalization: ") +
                                error.what());
    }
}

SealSciSoftmaxExecutor::SealSciSoftmaxExecutor(
    const HeraConfig &config, const TransformerLayerConfig &layer,
    HeContextRegistry &contexts, SealCipherTensorStore &he_tensors,
    CryptoTensorStore &mpc_tensors, CkksMpcConverter &converter,
    MpcSoftmaxRuntime &mpc, TensorIdAllocator *ids)
    : config_(config), layer_(layer), contexts_(contexts),
      he_tensors_(he_tensors), mpc_tensors_(mpc_tensors),
      converter_(converter), mpc_(mpc), ids_(ids)
{}

Result<SoftmaxOutputs> SealSciSoftmaxExecutor::Evaluate(
    const SoftmaxInputs &inputs, SoftmaxExecutionReport *report)
{
    try {
        if (report) *report = {};
        const TensorShape score_shape{{layer_.q_heads, layer_.tokens,
                                       layer_.tokens}};
        const auto &handle = inputs.scaled_attention_scores;
        if (!handle.valid() || handle.spec.domain != Domain::kHeCiphertext ||
            handle.spec.layout != Layout::kAttentionDiagonal ||
            handle.spec.shape != score_shape ||
            !handle.spec.he_parameter_profile)
            return Status::Invalid(
                "Softmax input is not a Block-2 attention score tensor");
        auto mask_valid = ValidMask(layer_, inputs.mask);
        if (!mask_valid.ok()) return mask_valid.status();

        const auto *execution = FindLayerExecution(config_, layer_.id);
        if (!execution)
            return Status::Invalid("Softmax layer has no HE placement");
        const auto region3 = execution->regions.find(HeRegionId::kRegion3);
        if (region3 == execution->regions.end())
            return Status::Invalid("Softmax Block-3 region is absent");
        const auto &block2_context = contexts_.Require(
            *handle.spec.he_parameter_profile, contexts_.role());
        const auto &block3_context = contexts_.Require(
            region3->second.parameter_profile, contexts_.role());
        auto block3_valid = Block3LevelPlanner::Validate(
            block3_context.parameters, config_.softmax);
        if (!block3_valid.ok()) return block3_valid.status();

        auto score_packing = MultiHeadPackingPlan::AttentionDiagonal(
            layer_, block2_context.slot_count());
        if (!score_packing.ok()) return score_packing.status();
        auto score_map = ToConversionMap(score_packing.value());
        if (!score_map.ok()) return score_map.status();
        std::optional<SealCipherBatch> score_batch;
        if (contexts_.role() == PartyRole::kServer) {
            auto tensor = he_tensors_.Get(handle.id);
            if (!tensor.ok()) return tensor.status();
            auto tensor_valid = tensor.value().Validate(block2_context);
            if (!tensor_valid.ok()) return tensor_valid.status();
            if (!SamePacking(tensor.value().physical_packing,
                             score_packing.value()))
                return Status::Invalid(
                    "Softmax ciphertext packing differs from Block-2 QK output");
            score_batch = SealCipherBatch{
                tensor.value().parameter_profile,
                tensor.value().ciphertexts,
                tensor.value().logical_spec.shape,
                tensor.value().logical_spec.layout,
                score_map.value()};
        }

        auto before = converter_.stats();
        auto score_physical = converter_.CkksToMpc(score_batch);
        if (!score_physical.ok()) return score_physical.status();
        auto after = converter_.stats();
        if (report) {
            ++report->ckks_to_mpc_calls;
            report->ckks_to_mpc_milliseconds +=
                after.milliseconds - before.milliseconds;
            report->conversion_communication_bytes +=
                after.communication_bytes - before.communication_bytes;
            report->conversion_rounds += after.rounds - before.rounds;
            report->a_mask_bytes +=
                after.a_mask_bytes - before.a_mask_bytes;
            report->z_mask_bytes +=
                after.z_mask_bytes - before.z_mask_bytes;
            report->internal_mod_switches += after.internal_mod_switches -
                                             before.internal_mod_switches;
        }
        auto scores = CompactScores(score_physical.value(),
                                    score_packing.value(), layer_);
        if (!scores.ok()) return scores.status();
        auto shifted = mpc_.CenterAndMask(
            scores.value(), layer_, inputs.mask, config_.softmax, report);
        if (!shifted.ok()) return shifted.status();
        // Keep the independent Softmax OT session alive across the HE
        // exponent segment. Ferret supports a stream of OT requests and the
        // converter uses different sockets, so rebuilding the session here
        // would only repeat its expensive setup.

        auto block3_map = SoftmaxPackingPlanner::RealRows(
            layer_, block3_context.slot_count(),
            config_.softmax.pad_rows_to_power_of_two);
        if (!block3_map.ok()) return block3_map.status();
        auto shifted_pair = ScatterDenseReal(
            shifted.value(), block3_map.value());
        before = converter_.stats();
        auto shifted_he = converter_.MpcToCkks(
            shifted_pair,
            {block3_context.profile, std::size_t{0},
             block3_context.parameters.input_scale_bits,
             shifted.value().physical_shape, Layout::kSoftmaxRows,
             block3_map.value()});
        if (!shifted_he.ok()) return shifted_he.status();
        after = converter_.stats();
        if (report) {
            ++report->mpc_to_ckks_calls;
            report->mpc_to_ckks_milliseconds +=
                after.milliseconds - before.milliseconds;
            report->conversion_communication_bytes +=
                after.communication_bytes - before.communication_bytes;
            report->conversion_rounds += after.rounds - before.rounds;
            report->a_mask_bytes +=
                after.a_mask_bytes - before.a_mask_bytes;
            report->z_mask_bytes +=
                after.z_mask_bytes - before.z_mask_bytes;
            report->internal_mod_switches += after.internal_mod_switches -
                                             before.internal_mod_switches;
        }

        std::optional<SealCipherBatch> exp_batch;
        if (contexts_.role() == PartyRole::kServer) {
            if (!shifted_he.value())
                return Status::Internal(
                    "server did not receive Block-3 ciphertexts");
            auto evaluated = SoftmaxExponentKernel::Evaluate(
                *shifted_he.value(), block3_context, config_.softmax, report,
                config_.threads);
            if (!evaluated.ok()) return evaluated.status();
            exp_batch = evaluated.take_value();
        }

        before = converter_.stats();
        auto exp_physical = converter_.CkksToMpc(exp_batch);
        if (!exp_physical.ok()) return exp_physical.status();
        after = converter_.stats();
        if (report) {
            ++report->ckks_to_mpc_calls;
            report->ckks_to_mpc_milliseconds +=
                after.milliseconds - before.milliseconds;
            report->conversion_communication_bytes +=
                after.communication_bytes - before.communication_bytes;
            report->conversion_rounds += after.rounds - before.rounds;
            report->a_mask_bytes +=
                after.a_mask_bytes - before.a_mask_bytes;
            report->z_mask_bytes +=
                after.z_mask_bytes - before.z_mask_bytes;
            report->internal_mod_switches += after.internal_mod_switches -
                                             before.internal_mod_switches;
        }
        auto exponentials = CompactDenseReal(
            exp_physical.value(), block3_map.value(),
            shifted.value().physical_shape,
            false);
        if (!exponentials.ok()) return exponentials.status();
        auto probabilities = mpc_.ClipAndNormalize(
            shifted.value(), exponentials.value(), layer_, inputs.mask,
            config_.softmax, report);
        if (!probabilities.ok()) return probabilities.status();
        // The session is retained for the next layer/inference. Its protocol
        // stream is independent from Attention/LN2 and conversion transports.

        TensorSpec output_spec;
        output_spec.shape = probabilities.value().physical_shape;
        output_spec.domain = Domain::kMpcArithmeticShare;
        output_spec.layout = Layout::kSoftmaxRows;
        output_spec.numeric = probabilities.value().numeric;
        const TensorId output_id = ids_ ? ids_->Next()
                                        : next_id_.fetch_add(1);
        auto stored = mpc_tensors_.PutMpcShare(
            output_id, probabilities.take_value());
        if (!stored.ok()) return stored.status();
        return SoftmaxOutputs{TensorHandle{output_id, output_spec}};
    } catch (const std::exception &error) {
        return Status::Internal(std::string("Softmax evaluation: ") +
                                error.what());
    }
}

} // namespace hera
