#include "hera/blb_reference.h"
#include "hera/softmax.h"
#include "hera/detail/he_level.h"
#include "hera/detail/he_packing.h"
#include "hera/detail/nonlinear_contracts.h"
#include "hera/detail/mpc_session_internal.h"
#include "hera/detail/parallel_ranges.h"

#include "OT/split-kkot.h"
#include "FloatingPoint/fixed-point.h"
#include "OT/ot_pack.h"
#include "utils/io_pack.h"
#include <seal/util/ntt.h>

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
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
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

using detail::LevelOf;

using detail::DroppedPrime;

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

struct PublicMaskPlan {
    std::size_t tokens{};
    PublicAttentionMask source;
    std::vector<bool> valid;
    std::vector<std::vector<std::size_t>> columns;
};

Result<std::shared_ptr<const PublicMaskPlan>> GetMaskPlan(
    const TransformerLayerConfig &layer, const PublicAttentionMask &mask)
{
    // One bounded, thread-local public plan is shared by validation, centering
    // and normalization. Compare contents, not the address of a mutable mask.
    thread_local std::shared_ptr<const PublicMaskPlan> cached;
    if (mask.kind == layer.attention_mask && cached &&
        cached->tokens == layer.tokens && cached->source.kind == mask.kind &&
        (mask.kind != AttentionMaskKind::kPublicPadding ||
         (cached->source.shape == mask.shape &&
          cached->source.additive_values == mask.additive_values)))
        return cached;
    auto validity = ValidMask(layer, mask);
    if (!validity.ok()) return validity.status();
    auto plan = std::make_shared<PublicMaskPlan>();
    plan->tokens = layer.tokens;
    plan->source = mask;
    plan->valid = validity.take_value();
    plan->columns.resize(layer.tokens);
    for (std::size_t row = 0; row < layer.tokens; ++row)
        for (std::size_t column = 0; column < layer.tokens; ++column)
            if (plan->valid[row * layer.tokens + column])
                plan->columns[row].push_back(column);
    cached = plan;
    return cached;
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

using detail::ToConversionMap;

bool SamePacking(const HePackingPlan &left, const HePackingPlan &right)
{
    return detail::PackingMatches(left, right);
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

ComplexMpcSharePair ScatterBoundaryShares(const MpcShareTensor &logical,
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
        const auto imaginary = map.assignments[physical].imaginary_logical_index;
        if (imaginary != kUnusedLogicalIndex) {
            const auto value = logical.local_share[imaginary];
            pair.imag.local_share[physical] = physical % map.slots_per_ciphertext & 1U
                ? (std::uint64_t{0} - value) & RingMask(logical.numeric.ring_bits)
                : value;
        }
    }
    return pair;
}

Result<MpcShareTensor> CompactBoundaryShares(
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
        const auto imaginary = map.assignments[physical_index].imaginary_logical_index;
        if (imaginary != kUnusedLogicalIndex) {
            const auto value = physical.imag.local_share[physical_index];
            logical[imaginary] = physical_index % map.slots_per_ciphertext & 1U
                ? (std::uint64_t{0} - value) & RingMask(physical.imag.numeric.ring_bits)
                : value;
        }
    }
    auto numeric = physical.real.numeric;
    numeric.is_signed = is_signed;
    return MpcShareTensor{std::move(logical), shape, numeric,
                          ComplexComponent::kReal};
}

struct PreparedExponent : PreparedBlock2Program {
    std::string signature;
    std::shared_ptr<const seal::Plaintext> divisor;
    std::shared_ptr<const seal::Plaintext> one;
};

struct PreparedSoftmaxPacking : PreparedBlock2Program {
    std::string signature;
    HePackingPlan scores;
    PhysicalSlotMap score_map;
    PhysicalSlotMap exponent_map;
    PhysicalSlotMap boundary_map;
};

struct PreparedBoundary : PreparedBlock2Program {
    std::string signature;
    seal::Plaintext fresh_phase;
    seal::Plaintext terminal_phase;
};

template <class Program>
Result<std::shared_ptr<const Program>> GetProgram(
    const std::string &tag, const std::string &signature,
    const std::function<Result<std::shared_ptr<const Program>>()>& compile)
{
    const auto scope = ModelPlaintextCacheScope::Current();
    if (scope.first) {
        const auto key = scope.second + "/softmax/" + tag;
        auto found = scope.first->GetPreparedBlock2Program(key);
        if (found.ok()) {
            auto program = std::dynamic_pointer_cast<const Program>(found.value());
            if (!program || program->signature != signature)
                return Status::FailedPrecondition("prepared Softmax contract mismatch");
            return program;
        }
        if (!scope.first->preparing_plaintexts()) return found.status();
        auto compiled = compile();
        if (!compiled.ok()) return compiled.status();
        auto stored = scope.first->PutPreparedBlock2Program(key, compiled.value());
        if (!stored.ok()) return stored.status();
        return compiled;
    }
    // Standalone kernels have no model store. Bound their immutable cache so
    // changing model geometries cannot retain unbounded packing/plaintext data.
    static std::mutex mutex;
    static std::map<std::string, std::shared_ptr<const Program>> cache;
    std::lock_guard<std::mutex> lock(mutex);
    auto found = cache.find(signature);
    if (found != cache.end()) return found->second;
    auto compiled = compile();
    if (!compiled.ok()) return compiled.status();
    if (cache.size() >= 8) cache.erase(cache.begin());
    cache.emplace(signature, compiled.value());
    return compiled;
}

Result<std::shared_ptr<const PreparedExponent>> GetExponentProgram(
    const HeContext &context, const SoftmaxConfig &config,
    bool doubled_input = false)
{
    auto valid = Block3LevelPlanner::Validate(context.parameters, config);
    if (!valid.ok()) return valid.status();
    std::ostringstream key;
    key << context.profile << '/' << context.parameters.input_scale_bits << '/'
        << context.parameters.default_scale_bits << '/' << config.exp_divisor_bits
        << '/' << config.exp_squarings << '/' << doubled_input;
    for (auto word : context.context->first_parms_id()) key << '/' << word;
    const auto signature = key.str();
    return GetProgram<PreparedExponent>("exponent", signature, [&]()
        -> Result<std::shared_ptr<const PreparedExponent>> {
        auto program = std::make_shared<PreparedExponent>();
        program->signature = signature;
        seal::CKKSEncoder encoder(*context.context);
        const auto input_id = context.context->first_parms_id();
        const auto next_id = context.context->first_context_data()
                                ->next_context_data()->parms_id();
        const double target = std::ldexp(1.0, context.parameters.default_scale_bits);
        const double plain_scale = target * DroppedPrime(context, input_id) /
            std::ldexp(1.0, context.parameters.input_scale_bits);
        auto divisor = CompileModelPlaintext(encoder,
            std::vector<std::complex<double>>(context.slot_count(),
                std::ldexp(1.0, -config.exp_divisor_bits - int(doubled_input))),
            input_id, plain_scale, doubled_input ? "softmax/divisor_doubled" : "softmax/divisor");
        if (!divisor.ok()) return divisor.status();
        auto one = CompileModelPlaintext(encoder,
            std::vector<std::complex<double>>(context.slot_count(), 1.0),
            next_id, target, "softmax/one");
        if (!one.ok()) return one.status();
        program->divisor = divisor.take_value();
        program->one = one.take_value();
        return std::shared_ptr<const PreparedExponent>(std::move(program));
    });
}

Result<std::shared_ptr<const PreparedBoundary>> GetBoundaryProgram(const HeContext &context)
{
    std::ostringstream key;
    key << context.profile;
    for (auto word : context.context->first_parms_id()) key << '/' << word;
    const auto signature = key.str();
    return GetProgram<PreparedBoundary>("boundary", signature, [&]()
        -> Result<std::shared_ptr<const PreparedBoundary>> {
        auto program = std::make_shared<PreparedBoundary>();
        program->signature = signature;
        const auto compile_phase = [&](seal::parms_id_type id) {
            const auto data = context.context->get_context_data(id);
            const auto degree = data->parms().poly_modulus_degree();
            const auto primes = data->parms().coeff_modulus().size();
            seal::Plaintext phase(degree * primes);
            std::fill_n(phase.data(), phase.coeff_count(), std::uint64_t{0});
            for (std::size_t rns = 0; rns < primes; ++rns) {
                phase[rns * degree + degree / 2] = 1;
                seal::util::ntt_negacyclic_harvey(
                    phase.data() + rns * degree, data->small_ntt_tables()[rns]);
            }
            phase.parms_id() = id;
            phase.scale() = 1.0;
            return phase;
        };
        // SEAL's generator 3 slot order evaluates X^(N/2) as i*(-1)^slot.
        program->fresh_phase = compile_phase(context.context->first_parms_id());
        program->terminal_phase = compile_phase(context.context->last_parms_id());
        return std::shared_ptr<const PreparedBoundary>(std::move(program));
    });
}

Result<std::shared_ptr<const PreparedSoftmaxPacking>> GetPackingProgram(
    const TransformerLayerConfig &layer, const HeContext &scores,
    const HeContext &exponent, const SoftmaxConfig &config)
{
    const auto signature = std::to_string(layer.tokens) + '/' +
        std::to_string(layer.q_heads) + '/' + std::to_string(layer.kv_heads) + '/' +
        std::to_string(layer.qk_head_width) + '/' + std::to_string(scores.slot_count()) +
        '/' + std::to_string(exponent.slot_count()) + '/' +
        std::to_string(config.pad_rows_to_power_of_two);
    return GetProgram<PreparedSoftmaxPacking>("packing", signature, [&]()
        -> Result<std::shared_ptr<const PreparedSoftmaxPacking>> {
        auto packing = MultiHeadPackingPlan::AttentionDiagonal(layer, scores.slot_count());
        if (!packing.ok()) return packing.status();
        auto score_map = ToConversionMap(packing.value());
        if (!score_map.ok()) return score_map.status();
        auto exponent_map = SoftmaxPackingPlanner::RealRows(
            layer, exponent.slot_count(), config.pad_rows_to_power_of_two);
        if (!exponent_map.ok()) return exponent_map.status();
        auto program = std::make_shared<PreparedSoftmaxPacking>();
        program->signature = signature;
        program->scores = packing.take_value();
        program->score_map = score_map.take_value();
        program->exponent_map = exponent_map.take_value();
        auto paired = SoftmaxPackingPlanner::PairedRealRows(program->exponent_map);
        if (!paired.ok()) return paired.status();
        program->boundary_map = paired.take_value();
        return std::shared_ptr<const PreparedSoftmaxPacking>(std::move(program));
    });
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

Result<PhysicalSlotMap> SoftmaxPackingPlanner::PairedRealRows(const PhysicalSlotMap &real)
{
    auto valid = real.Validate();
    if (!valid.ok()) return valid.status();
    PhysicalSlotMap paired;
    paired.ciphertext_count = (real.ciphertext_count + 1) / 2;
    paired.slots_per_ciphertext = real.slots_per_ciphertext;
    paired.logical_element_count = real.logical_element_count;
    paired.assignments.resize(paired.ciphertext_count * paired.slots_per_ciphertext);
    std::size_t tail_extent = 0;
    if (real.ciphertext_count & 1U)
        for (std::size_t slot = 0; slot < real.slots_per_ciphertext; ++slot)
            if (real.assignments[(real.ciphertext_count - 1) * real.slots_per_ciphertext + slot].real_logical_index != kUnusedLogicalIndex)
                tail_extent = slot + 1;
    for (std::size_t ct = 0; ct < real.ciphertext_count; ++ct)
        for (std::size_t slot = 0; slot < real.slots_per_ciphertext; ++slot) {
            const auto &entry = real.assignments[ct * real.slots_per_ciphertext + slot];
            if (entry.imaginary_logical_index != kUnusedLogicalIndex)
                return Status::Invalid("paired boundary requires a real source map");
            auto &target = paired.assignments[(ct / 2) * real.slots_per_ciphertext + slot];
            if ((ct & 1U) || ((real.ciphertext_count & 1U) && ct == real.ciphertext_count - 1 && slot >= (tail_extent + 1) / 2))
                target.imaginary_logical_index = entry.real_logical_index;
            else target.real_logical_index = entry.real_logical_index;
        }
    valid = paired.Validate();
    if (!valid.ok()) return valid.status();
    return paired;
}

Result<SealCipherBatch> SoftmaxExponentKernel::Evaluate(
    const SealCipherBatch &shifted_scores, const HeContext &context,
    const SoftmaxConfig &config, SoftmaxExecutionReport *report,
    std::size_t he_workers)
{
    return Evaluate(SealCipherBatch(shifted_scores), context, config, report, he_workers);
}

Result<void> SoftmaxExponentKernel::Prepare(
    const HeContext &context, const SoftmaxConfig &config)
{
    try {
        auto program = GetExponentProgram(context, config);
        if (!program.ok()) return program.status();
        return {};
    } catch (const std::exception &error) {
        return Status::Internal(std::string("Softmax exponent preparation: ") + error.what());
    }
}

Result<SealCipherBatch> SoftmaxExponentKernel::Evaluate(
    SealCipherBatch &&shifted_scores, const HeContext &context,
    const SoftmaxConfig &config, SoftmaxExecutionReport *report,
    std::size_t he_workers)
{
    return EvaluateReal(std::move(shifted_scores), context, config, false, report, he_workers);
}

Result<SealCipherBatch> SoftmaxExponentKernel::EvaluateReal(
    SealCipherBatch &&shifted_scores, const HeContext &context,
    const SoftmaxConfig &config, bool doubled_input,
    SoftmaxExecutionReport *report, std::size_t he_workers)
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

        auto program = GetExponentProgram(context, config, doubled_input);
        if (!program.ok()) return program.status();
        SealCipherBatch output = std::move(shifted_scores);
        const double target = std::ldexp(
            1.0, context.parameters.default_scale_bits);
        auto parallel = detail::ParallelRanges(
            output.ciphertexts.size(), he_workers,
            1, [&](std::size_t begin, std::size_t end) -> Result<void> {
                seal::Evaluator evaluator(*context.context);
                for (std::size_t index = begin; index < end; ++index) {
                    auto &ciphertext = output.ciphertexts[index];
                    auto level = LevelOf(context, ciphertext.parms_id());
                    if (!level.ok()) return level.status();
                    if (level.value() != 0 ||
                        !CloseScale(ciphertext.scale(), std::ldexp(
                            1.0, context.parameters.input_scale_bits)))
                        return Status::FailedPrecondition(
                            "Block-3 exponent input must be fresh at input scale");

                    evaluator.multiply_plain_inplace(ciphertext, *program.value()->divisor);
                    evaluator.rescale_to_next_inplace(ciphertext);
                    auto normalized = NormalizeScale(ciphertext, target);
                    if (!normalized.ok()) return normalized.status();

                    evaluator.add_plain_inplace(ciphertext, *program.value()->one);
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

Result<SealCipherBatch> SoftmaxExponentKernel::EvaluatePaired(
    SealCipherBatch &&shifted_scores, const PhysicalSlotMap &real_map,
    const HeContext &context, const SoftmaxConfig &config,
    SoftmaxExecutionReport *report, std::size_t he_workers)
{
    const auto started = Clock::now();
    try {
        auto expected = SoftmaxPackingPlanner::PairedRealRows(real_map);
        if (!expected.ok()) return expected.status();
        if (shifted_scores.parameter_profile != context.profile ||
            shifted_scores.logical_layout != Layout::kSoftmaxRows ||
            shifted_scores.slot_map.ciphertext_count != expected.value().ciphertext_count ||
            shifted_scores.slot_map.slots_per_ciphertext != expected.value().slots_per_ciphertext ||
            shifted_scores.slot_map.logical_element_count != expected.value().logical_element_count ||
            shifted_scores.slot_map.assignments != expected.value().assignments ||
            shifted_scores.ciphertexts.size() != expected.value().ciphertext_count)
            return Status::Invalid("paired exponent boundary has incompatible packing");
        if (!context.galois_keys.has_key(2 * context.parameters.polynomial_degree - 1))
            return Status::FailedPrecondition("paired exponent requires a conjugation key");
        auto phase = GetBoundaryProgram(context);
        if (!phase.ok()) return phase.status();
        SealCipherBatch real{shifted_scores.parameter_profile,
            std::vector<seal::Ciphertext>(real_map.ciphertext_count),
            shifted_scores.logical_shape, shifted_scores.logical_layout, real_map};
        auto parallel = detail::ParallelRanges(
            shifted_scores.ciphertexts.size(), he_workers, 1,
            [&](std::size_t begin, std::size_t end) -> Result<void> {
                seal::Evaluator evaluator(*context.context);
                for (std::size_t pair = begin; pair < end; ++pair) {
                    auto &source = shifted_scores.ciphertexts[pair];
                    if (source.parms_id() != context.context->first_parms_id() ||
                        !CloseScale(source.scale(), std::ldexp(1.0, context.parameters.input_scale_bits)))
                        return Status::FailedPrecondition("paired exponent must start fresh at input scale");
                    seal::Ciphertext conjugated;
                    evaluator.complex_conjugate(source, context.galois_keys, conjugated);
                    seal::Ciphertext imaginary;
                    evaluator.sub(conjugated, source, imaginary);
                    evaluator.multiply_plain_inplace(imaginary, phase.value()->fresh_phase);
                    evaluator.add(source, conjugated, real.ciphertexts[2 * pair]);
                    if (2 * pair + 1 < real.ciphertexts.size())
                        real.ciphertexts[2 * pair + 1] = std::move(imaginary);
                    else evaluator.add_inplace(real.ciphertexts[2 * pair], imaginary);
                }
                return {};
            }, detail::EstimatedHeWorkerBytes(context.parameters.polynomial_degree,
                context.parameters.coefficient_modulus_bits.size(), 12));
        if (!parallel.ok()) return parallel.status();
        if (report) {
            report->conjugations += shifted_scores.ciphertexts.size();
            report->phase_multiplications += real.ciphertexts.size() / 2;
            report->ciphertext_plain_multiplications += real.ciphertexts.size() / 2;
            report->he_milliseconds += std::chrono::duration<double, std::milli>(Clock::now() - started).count();
        }
        // Both recovered lanes equal 2*x. Fuse their 1/2 into the existing
        // divisor plaintext; this adds no rescale or multiplication depth.
        return EvaluateReal(std::move(real), context, config, true, report, he_workers);
    } catch (const std::exception &error) {
        return Status::Internal(std::string("paired exponent: ") + error.what());
    }
}

Result<SealCipherBatch> SoftmaxExponentKernel::PackPaired(
    SealCipherBatch &&real, const HeContext &context,
    SoftmaxExecutionReport *report, std::size_t he_workers)
{
    const auto started = Clock::now();
    try {
        auto map = SoftmaxPackingPlanner::PairedRealRows(real.slot_map);
        if (!map.ok()) return map.status();
        if (real.parameter_profile != context.profile ||
            real.logical_layout != Layout::kSoftmaxRows ||
            real.ciphertexts.size() != real.slot_map.ciphertext_count)
            return Status::Invalid("terminal pair input has incompatible metadata");
        auto phase = GetBoundaryProgram(context);
        if (!phase.ok()) return phase.status();
        SealCipherBatch output{real.parameter_profile,
            std::vector<seal::Ciphertext>(map.value().ciphertext_count),
            real.logical_shape, real.logical_layout, map.take_value()};
        auto parallel = detail::ParallelRanges(
            output.ciphertexts.size(), he_workers, 1,
            [&](std::size_t begin, std::size_t end) -> Result<void> {
                seal::Evaluator evaluator(*context.context);
                for (std::size_t pair = begin; pair < end; ++pair) {
                    auto &first = real.ciphertexts[2 * pair];
                    if (first.parms_id() != context.context->last_parms_id())
                        return Status::FailedPrecondition("packing requires terminal exponent ciphertexts");
                    if (2 * pair + 1 < real.ciphertexts.size()) {
                        auto &second = real.ciphertexts[2 * pair + 1];
                        if (second.parms_id() != first.parms_id() || !CloseScale(first.scale(), second.scale()))
                            return Status::FailedPrecondition("pair ciphertext levels/scales differ");
                        evaluator.multiply_plain_inplace(second, phase.value()->terminal_phase);
                        evaluator.add_inplace(first, second);
                    } else {
                        auto imaginary = first;
                        evaluator.multiply_plain_inplace(imaginary, phase.value()->terminal_phase);
                        evaluator.add_inplace(first, imaginary);
                    }
                    output.ciphertexts[pair] = std::move(first);
                }
                return {};
            }, detail::EstimatedHeWorkerBytes(context.parameters.polynomial_degree, 1, 8));
        if (!parallel.ok()) return parallel.status();
        if (report) {
            report->phase_multiplications += real.ciphertexts.size() / 2;
            report->ciphertext_plain_multiplications += real.ciphertexts.size() / 2;
            report->he_milliseconds += std::chrono::duration<double, std::milli>(Clock::now() - started).count();
        }
        return output;
    } catch (const std::exception &error) {
        return Status::Internal(std::string("terminal pair packing: ") + error.what());
    }
}

class MpcSoftmaxRuntime::Impl {
public:
    explicit Impl(std::shared_ptr<SharedMpcSession> session, std::size_t worker = 0)
        : session_(std::move(session)), native_(session_->Acquire(worker)),
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
    { return session_->communication_bytes(); }
    std::uint64_t rounds() const { return session_->rounds(); }
    void Cancel() noexcept { session_->Cancel(); }
    int party() const { return party_; }
    FixOp &fix() { return native_->fix(); }

    std::vector<std::uint64_t> MaximumRows(
        std::vector<std::uint64_t> candidates,
        std::vector<std::size_t> widths, const NumericFormat &numeric)
    {
        const auto mask = RingMask(numeric.ring_bits);
        for (auto &share : candidates) share &= mask;
        for (;;) {
            std::size_t pairs = 0;
            for (auto width : widths) pairs += width / 2;
            if (pairs == 0) return candidates;
            std::vector<std::uint64_t> left(pairs), right(pairs), maximum(pairs);
            std::size_t offset = 0, pair = 0;
            for (auto width : widths) {
                for (std::size_t i = 0; i + 1 < width; i += 2, ++pair) {
                    left[pair] = candidates[offset + i]; right[pair] = candidates[offset + i + 1];
                }
                offset += width;
            }
            constexpr std::size_t batch_limit = 262144;
            for (std::size_t begin = 0; begin < pairs; begin += batch_limit) {
                const auto end = std::min(pairs, begin + batch_limit);
                auto a = Secret({left.begin() + begin, left.begin() + end}, numeric);
                auto b = Secret({right.begin() + begin, right.begin() + end}, numeric);
                auto chosen = fix().if_else(fix().GT(a, b), a, b);
                std::copy_n(chosen.data, end - begin, maximum.begin() + begin);
            }
            std::vector<std::uint64_t> next;
            next.reserve(pairs + widths.size());
            offset = 0;
            pair = 0;
            for (auto &width : widths) {
                for (std::size_t i = 0; i < width / 2; ++i)
                    next.push_back(maximum[pair++]);
                if (width & 1U) next.push_back(candidates[offset + width - 1]);
                offset += width;
                width = (width + 1) / 2;
            }
            candidates = std::move(next);
        }
    }

    FixArray MultiplyUnsigned(const FixArray &left, const FixArray &right,
                              int output_bits)
    {
        if (left.signed_ || right.signed_ || left.size != right.size ||
            output_bits < left.ell || output_bits < right.ell ||
            output_bits >= 64)
            throw std::invalid_argument(
                "unsupported unsigned multiplication contract");
        // These operands are the clipped unit-interval exponential and its
        // row reciprocal, not arbitrary unsigned values. A guard bit makes
        // their reconstructed MSB public zero despite random local shares.
        const bool bounded = left.ell > left.s + 1 && right.ell > right.s + 1;
        std::vector<std::uint8_t> zero_msb(bounded ? left.size : 0, 0);
        auto product = fix().mul(left, right, output_bits,
                                 bounded ? zero_msb.data() : nullptr,
                                 bounded ? zero_msb.data() : nullptr);
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
        // Boolean quotient bits need not become arithmetic shares until all
        // restoring-division steps finish. Batch that independent conversion.
        BoolArray quotient_bits(party_, denominator.size * (output_scale + 1));
        std::copy_n(take.data, denominator.size,
                    quotient_bits.data + denominator.size * output_scale);

        for (int bit = output_scale - 1; bit >= 0; --bit) {
            auto doubled = fix().mul(remainder, std::uint64_t{2}, work_bits);
            take = fix().GT(doubled, divisor_minus_one);
            difference = fix().sub(doubled, divisor);
            remainder = fix().if_else(take, difference, doubled);
            std::copy_n(take.data, denominator.size,
                        quotient_bits.data + denominator.size * bit);
        }
        auto arithmetic_bits = fix().B2A(quotient_bits, false, output_bits);
        FixArray quotient(party_, denominator.size, false, output_bits, output_scale);
        const auto quotient_mask = RingMask(output_bits);
        for (int row = 0; row < denominator.size; ++row) {
            std::uint64_t value = 0;
            for (int bit = 0; bit <= output_scale; ++bit)
                value += arithmetic_bits.data[denominator.size * bit + row] << bit;
            quotient.data[row] = value & quotient_mask;
        }
        return quotient;
    }

    bool RevealAnyZero(const FixArray &values)
    {
        std::vector<std::uint64_t> zero_shares(values.size, 0);
        NumericFormat format{values.ell, values.s, values.signed_};
        auto secret_zero = Secret(zero_shares, format);
        auto zero = fix().EQ(values, secret_zero);
        while (zero.size > 1) {
            const int pairs = zero.size / 2;
            BoolArray left(party_, pairs), right(party_, pairs);
            for (int i = 0; i < pairs; ++i) {
                left.data[i] = zero.data[2 * i];
                right.data[i] = zero.data[2 * i + 1];
            }
            auto either = fix().bool_op->XOR(left, right);
            auto both = fix().bool_op->AND(left, right);
            auto combined = fix().bool_op->XOR(either, both);
            BoolArray next(party_, pairs + (zero.size & 1));
            std::copy_n(combined.data, pairs, next.data);
            if (zero.size & 1) next.data[pairs] = zero.data[zero.size - 1];
            zero = std::move(next);
        }
        auto server_result = fix().bool_op->output(
            sci::ALICE, zero);
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
        auto validity = GetMaskPlan(layer, mask);
        if (!validity.ok()) return validity.status();
        const auto before_comm = impl_->communication();
        const auto before_rounds = impl_->rounds();
        const auto ring_mask = RingMask(scores.numeric.ring_bits);
        const auto rows = layer.tokens * layer.q_heads;
        std::vector<std::uint64_t> maxima(rows);
        std::size_t comparisons = 0;
        for (const auto &columns : validity.value()->columns)
            comparisons += (columns.size() - 1) * layer.q_heads;
        // Partition complete rows. Ragged masks and odd candidates retain
        // exactly the same comparison tree within each row.
        session_->ForEachRange(rows, 1,
            [&](std::size_t worker, std::size_t begin, std::size_t end) {
                Impl local(session_, worker);
                std::vector<std::uint64_t> candidates;
                candidates.reserve((end - begin) * layer.tokens);
                std::vector<std::size_t> widths;
                widths.reserve(end - begin);
                for (std::size_t row = begin; row < end; ++row) {
                    const auto &columns = validity.value()->columns[row / layer.q_heads];
                    for (auto column : columns)
                        candidates.push_back(scores.local_share[row * layer.tokens + column]);
                    widths.push_back(columns.size());
                }
                const auto values = local.MaximumRows(std::move(candidates), std::move(widths), scores.numeric);
                std::copy(values.begin(), values.end(), maxima.begin() + begin);
            });

        const auto masked_value = EncodeFixed(config.masked_shift,
                                              scores.numeric);
        std::vector<std::uint64_t> shifted(scores.local_share.size());
        for (std::size_t row = 0; row < layer.tokens; ++row) {
            for (std::size_t head = 0; head < layer.q_heads; ++head) {
                const auto max_share = maxima[row * layer.q_heads + head];
                for (std::size_t column = 0; column < layer.tokens; ++column) {
                    const auto index = RowMajorIndex(row, head, column, layer);
                    if (validity.value()->valid[row * layer.tokens + column])
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
        auto validity = GetMaskPlan(layer, mask);
        if (!validity.ok()) return validity.status();

        auto accumulator = detail::SoftmaxAccumulatorBits(layer.tokens,
            exponentials.numeric.ring_bits, exponentials.numeric.scale_bits);
        if (!accumulator.ok()) return accumulator.status();
        const int accumulator_bits = accumulator.value();
        if (!std::atomic_load(&impl_))
            std::atomic_store(&impl_, std::make_shared<Impl>(
                session_));
        const auto before_comm = impl_->communication();
        const auto before_rounds = impl_->rounds();
        NumericFormat accumulator_numeric{accumulator_bits, exponentials.numeric.scale_bits, false};
        constexpr std::size_t batch_limit = 262144;
        const auto elements = shifted_scores.local_share.size();
        const auto rows = layer.tokens * layer.q_heads;
        std::vector<std::uint64_t> clipped(elements);
        session_->ForEachRange(rows, 1,
            [&](std::size_t worker, std::size_t row_begin, std::size_t row_end) {
                Impl local(session_, worker);
                for (std::size_t begin = row_begin * layer.tokens; begin < row_end * layer.tokens; begin += batch_limit) {
                    const auto end = std::min(row_end * layer.tokens, begin + batch_limit);
                    auto shifted = local.Secret({shifted_scores.local_share.begin() + begin,
                        shifted_scores.local_share.begin() + end}, shifted_scores.numeric);
                    auto exp = local.Secret({exponentials.local_share.begin() + begin,
                        exponentials.local_share.begin() + end}, exponentials.numeric);
                    std::vector<std::uint64_t> threshold_shares(end - begin);
                    if (local.party() == sci::ALICE)
                        std::fill(threshold_shares.begin(), threshold_shares.end(), EncodeFixed(
                            config.clip_threshold - std::ldexp(1., -shifted_scores.numeric.scale_bits),
                            shifted_scores.numeric));
                    auto threshold = local.Secret(threshold_shares, shifted_scores.numeric);
                    // GT(T - 1 ULP) preserves the inclusive fixed-point threshold.
                    auto keep = local.fix().GT(shifted, threshold);
                    auto exp_short = local.fix().reduce(exp, accumulator_bits);
                    auto selected = local.fix().if_else(keep, exp_short, std::uint64_t{0});
                    std::copy_n(selected.data, end - begin, clipped.begin() + begin);
                }
            });
        const auto ring_mask = RingMask(accumulator_bits);
        std::vector<std::uint64_t> sums(rows, 0);
        for (std::size_t row = 0; row < layer.tokens; ++row) {
            for (std::size_t head = 0; head < layer.q_heads; ++head) {
                auto &sum = sums[row * layer.q_heads + head];
                for (std::size_t column = 0; column < layer.tokens; ++column)
                    sum = (sum + clipped[
                        RowMajorIndex(row, head, column, layer)]) & ring_mask;
            }
        }
        // Preserve the original single aggregate validity bit. Revealing a
        // separate zero flag per shard would expose additional row information.
        bool any_zero = false;
        session_->ForEachRange(1, 1, [&](std::size_t, std::size_t, std::size_t) {
            auto denominator = impl_->Secret(sums, accumulator_numeric);
            any_zero = impl_->RevealAnyZero(denominator);
        });
        if (any_zero)
            return Status::FailedPrecondition(
                "Softmax denominator is zero after clipping");

        std::vector<std::uint64_t> reciprocals(rows);
        session_->ForEachRange(rows, 1,
            [&](std::size_t worker, std::size_t begin, std::size_t end) {
                Impl local(session_, worker);
                auto denominator = local.Secret({sums.begin() + begin, sums.begin() + end}, accumulator_numeric);
                auto reciprocal = local.ReciprocalUnitInterval(denominator,
                    accumulator_numeric.ring_bits, accumulator_numeric.scale_bits);
                std::copy_n(reciprocal.data, end - begin, reciprocals.begin() + begin);
            });
        std::vector<std::uint64_t> probability_shares(elements);
        // truncate_reduce returns an accumulator_bits-wide value; the
        // subsequent narrow-to-wide extension can only reconstruct x or
        // x-2^accumulator_bits. Thus this conservative signed bound follows
        // from the operation widths, even before using probability <= 1.
        const int probability_bound_bits = accumulator_bits;
        session_->ForEachRange(rows, 1,
            [&](std::size_t worker, std::size_t row_begin, std::size_t row_end) {
                Impl local(session_, worker);
                for (std::size_t begin = row_begin * layer.tokens; begin < row_end * layer.tokens; begin += batch_limit) {
                    const auto end = std::min(row_end * layer.tokens, begin + batch_limit);
                    std::vector<std::uint64_t> inverse(end - begin);
                    for (std::size_t i = begin; i < end; ++i) inverse[i - begin] = reciprocals[i / layer.tokens];
                    auto expanded = local.Secret(inverse, accumulator_numeric);
                    auto selected = local.Secret({clipped.begin() + begin, clipped.begin() + end}, accumulator_numeric);
                    auto product = local.MultiplyUnsigned(selected, expanded,
                        accumulator_bits + accumulator_numeric.scale_bits);
                    auto short_result = local.fix().truncate_reduce(product, accumulator_numeric.scale_bits);
                    const bool bounded = short_result.ell > short_result.s + 1;
                    std::vector<std::uint8_t> zero_msb(bounded ? short_result.size : 0, 0);
                    auto extended = local.fix().extend(short_result, exponentials.numeric.ring_bits,
                        bounded ? zero_msb.data() : nullptr);
                    std::copy_n(extended.data, end - begin, probability_shares.begin() + begin);
                }
            });

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
        return MpcShareTensor{std::move(probability_shares), expected,
                              unsigned_numeric, ComplexComponent::kReal,
                              probability_bound_bits < unsigned_numeric.ring_bits - 1
                                  ? std::optional<int>{probability_bound_bits}
                                  : std::nullopt};
    } catch (const std::exception &error) {
        return Status::Internal(std::string("MPC Softmax normalization: ") +
                                error.what());
    }
}

Result<void> SealSciSoftmaxExecutor::Prepare(
    const TransformerLayerConfig &layer, const HeContext &score_context,
    const HeContext &exponent_context, const SoftmaxConfig &config,
    bool prepare_exponent)
{
    try {
        auto program = GetPackingProgram(layer, score_context, exponent_context, config);
        if (!program.ok()) return program.status();
        if (!prepare_exponent) return {};
        const bool paired = true;
        auto exponent = GetExponentProgram(exponent_context, config, paired);
        if (!exponent.ok()) return exponent.status();
        if (paired) {
            auto boundary = GetBoundaryProgram(exponent_context);
            if (!boundary.ok()) return boundary.status();
        }
        return {};
    } catch (const std::exception &error) {
        return Status::Internal(std::string("Softmax preparation: ") + error.what());
    }
}

SealSciSoftmaxExecutor::SealSciSoftmaxExecutor(
    const HeraConfig &config, const TransformerLayerConfig &layer,
    HeContextRegistry &contexts, SealCipherTensorStore &he_tensors,
    CryptoTensorStore &mpc_tensors, BlbReferenceConverter &converter,
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
        auto mask_valid = GetMaskPlan(layer_, inputs.mask);
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

        auto packing_program = GetPackingProgram(
            layer_, block2_context, block3_context, config_.softmax);
        if (!packing_program.ok()) return packing_program.status();
        const auto &score_packing = packing_program.value()->scores;
        const auto &score_map = packing_program.value()->score_map;
        const auto &real_map = packing_program.value()->exponent_map;
        const bool paired_boundary = true;
        const auto &block3_map = paired_boundary
            ? packing_program.value()->boundary_map : real_map;
        std::optional<SealCipherBatch> score_batch;
        if (contexts_.role() == PartyRole::kServer) {
            auto tensor = he_tensors_.GetShared(handle.id);
            if (!tensor.ok()) return tensor.status();
            auto tensor_valid = tensor.value()->Validate(block2_context);
            if (!tensor_valid.ok()) return tensor_valid.status();
            if (!SamePacking(tensor.value()->physical_packing,
                             score_packing))
                return Status::Invalid(
                    "Softmax ciphertext packing differs from Block-2 QK output");
            score_batch = SealCipherBatch{
                tensor.value()->parameter_profile,
                tensor.value()->ciphertexts,
                tensor.value()->logical_spec.shape,
                tensor.value()->logical_spec.layout,
                score_map};
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
                                    score_packing, layer_);
        if (!scores.ok()) return scores.status();
        auto shifted = mpc_.CenterAndMask(
            scores.value(), layer_, inputs.mask, config_.softmax, report);
        if (!shifted.ok()) return shifted.status();
        // Keep the independent Softmax OT session alive across the HE
        // exponent segment. Ferret supports a stream of OT requests and the
        // converter uses different sockets, so rebuilding the session here
        // would only repeat its expensive setup.

        auto shifted_pair = ScatterBoundaryShares(
            shifted.value(), block3_map);
        before = converter_.stats();
        auto shifted_he = converter_.MpcToCkksPacked(
            shifted_pair,
            {block3_context.profile, std::size_t{0},
             block3_context.parameters.input_scale_bits,
             shifted.value().physical_shape, Layout::kSoftmaxRows,
             block3_map, !paired_boundary});
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
            auto evaluated = paired_boundary
                ? SoftmaxExponentKernel::EvaluatePaired(
                    std::move(*shifted_he.value()), real_map, block3_context,
                    config_.softmax, report, config_.threads)
                : SoftmaxExponentKernel::Evaluate(
                    std::move(*shifted_he.value()), block3_context, config_.softmax,
                    report, config_.threads);
            if (!evaluated.ok()) return evaluated.status();
            if (paired_boundary) {
                auto packed = SoftmaxExponentKernel::PackPaired(
                    evaluated.take_value(), block3_context, report, config_.threads);
                if (!packed.ok()) return packed.status();
                exp_batch = packed.take_value();
            } else exp_batch = evaluated.take_value();
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
        auto exponentials = CompactBoundaryShares(
            exp_physical.value(), block3_map,
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
