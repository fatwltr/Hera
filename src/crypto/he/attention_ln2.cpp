#include "hera/attention_ln2.h"
#include "hera/detail/he_level.h"
#include "hera/detail/level_schedule.h"
#include "hera/detail/he_packing.h"
#include "hera/detail/nonlinear_contracts.h"
#include "hera/detail/packing_utils.h"
#include "hera/detail/parallel_ranges.h"
#include "hera/detail/mpc_session_internal.h"

#include "OT/split-kkot.h"
#include "FloatingPoint/fp-math.h"
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
#include <complex>
#include <cstring>
#include <limits>
#include <list>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <tuple>

namespace hera {
namespace {

using Clock = std::chrono::steady_clock;
using Complex = std::complex<double>;
constexpr double kScaleTolerance = 1.0e-3;

std::size_t NextPowerOfTwo(std::size_t value)
{
    std::size_t result = 1;
    while (result < value) {
        if (result > std::numeric_limits<std::size_t>::max() / 2)
            throw std::overflow_error("next power of two overflows");
        result <<= 1U;
    }
    return result;
}

std::uint64_t RingMask(int bits)
{
    return bits == 64 ? std::numeric_limits<std::uint64_t>::max()
                      : (std::uint64_t{1} << bits) - 1;
}

using detail::LevelOf;

using detail::DroppedPrime;

bool CloseScale(double actual, double expected)
{
    const auto largest = std::max(std::abs(actual), std::abs(expected));
    return std::isfinite(largest) && largest > 0.0 &&
           std::abs(actual - expected) <= largest * kScaleTolerance;
}

Result<void> NormalizeScale(seal::Ciphertext &ciphertext, double target)
{
    if (!CloseScale(ciphertext.scale(), target)) {
        std::ostringstream message;
        message << "CKKS scale drift exceeds tolerance: actual_bits="
                << std::log2(ciphertext.scale()) << " target_bits="
                << std::log2(target);
        return Status::FailedPrecondition(message.str());
    }
    ciphertext.scale() = target;
    return {};
}

int CanonicalRotation(long long rotation, std::size_t slots)
{
    rotation %= static_cast<long long>(slots);
    if (rotation > static_cast<long long>(slots / 2))
        rotation -= static_cast<long long>(slots);
    if (rotation <= -static_cast<long long>(slots / 2))
        rotation += static_cast<long long>(slots);
    return static_cast<int>(rotation);
}

seal::Ciphertext Rotate(const seal::Ciphertext &input, int rotation,
                        const HeContext &context, seal::Evaluator &evaluator,
                        AttentionLn2ExecutionReport *report)
{
    seal::Ciphertext result = input;
    const int step = CanonicalRotation(rotation, context.slot_count());
    if (step == 0) return result;
    // RequiredRotationSteps pre-generates this exact offset. Keeping a
    // single evaluator call here guarantees one key-switch per rotation.
    evaluator.rotate_vector_inplace(result, step, context.galois_keys);
    if (report) ++report->rotations;
    return result;
}

struct HeadTile {
    std::size_t head_begin{};
    std::size_t head_count{};
    std::size_t lanes{};
    std::size_t probability_begin{};
    std::size_t probability_groups{};
    std::size_t value_begin{};
    std::size_t value_groups{};
};

Result<std::vector<HeadTile>> MakeTiles(
    const TransformerLayerConfig &layer, std::size_t slots)
{
    if (layer.tokens > slots)
        return Status::Invalid("one attention row does not fit in CKKS slots");
    const auto max_heads = slots / detail::TokenColumnStride(layer.tokens);
    if (max_heads == 0)
        return Status::Invalid("no attention head fits in CKKS slots");
    std::vector<HeadTile> result;
    std::size_t probability_begin = 0;
    std::size_t value_begin = 0;
    for (std::size_t begin = 0; begin < layer.q_heads;) {
        const auto heads = std::min(max_heads, layer.q_heads - begin);
        const auto block = detail::TokenColumnStride(layer.tokens) * heads;
        const auto lanes = detail::FloorPowerOfTwo(slots / block);
        const auto probability_groups =
            (layer.tokens + 2 * lanes - 1) / (2 * lanes);
        const auto value_groups =
            (layer.value_head_width + 2 * lanes - 1) / (2 * lanes);
        result.push_back({begin, heads, lanes, probability_begin,
                          probability_groups, value_begin, value_groups});
        probability_begin += probability_groups;
        value_begin += value_groups;
        begin += heads;
    }
    return result;
}

long long ReplicationRotation(std::size_t first, std::size_t populated,
                              std::size_t block)
{
    const auto shift = static_cast<long long>(populated * block);
    // Merge into the containing, aligned 2*populated-lane interval. Occupied
    // slots stay inside the power-of-two lane prefix even with a physical tail.
    return (first & populated) != 0 ? shift : -shift;
}

std::size_t ProbabilityIndex(std::size_t row, std::size_t head,
                             std::size_t column,
                             const TransformerLayerConfig &layer)
{
    return (row * layer.q_heads + head) * layer.tokens + column;
}

std::size_t AttentionOutputIndex(std::size_t token, std::size_t head,
                                 std::size_t dimension,
                                 const TransformerLayerConfig &layer)
{
    return (token * layer.q_heads + head) * layer.value_head_width +
           dimension;
}

using detail::ToConversionMap;

Result<ComplexMpcSharePair> ScatterReal(
    const MpcShareTensor &logical, const PhysicalSlotMap &map)
{
    auto valid = logical.Validate();
    if (!valid.ok()) return valid.status();
    auto count = logical.physical_shape.ElementCount();
    if (!count.ok()) return count.status();
    if (count.value() != map.logical_element_count)
        return Status::Invalid("logical MPC tensor and slot map differ");
    const auto physical = map.ciphertext_count * map.slots_per_ciphertext;
    MpcShareTensor real{{}, {{map.ciphertext_count, map.slots_per_ciphertext}},
                        logical.numeric, ComplexComponent::kReal,
                        logical.public_value_bound_bits};
    MpcShareTensor imag = real;
    imag.component = ComplexComponent::kImaginary;
    real.local_share.assign(physical, 0);
    imag.local_share.assign(physical, 0);
    for (std::size_t i = 0; i < physical; ++i) {
        const auto &entry = map.assignments[i];
        if (entry.real_logical_index != kUnusedLogicalIndex)
            real.local_share[i] =
                logical.local_share[entry.real_logical_index];
        if (entry.imaginary_logical_index != kUnusedLogicalIndex)
            imag.local_share[i] =
                logical.local_share[entry.imaginary_logical_index];
    }
    return ComplexMpcSharePair{std::move(real), std::move(imag)};
}

Result<MpcShareTensor> Compact(
    const ComplexMpcSharePair &physical, const PhysicalSlotMap &map,
    const TensorShape &shape, bool signed_values, bool paired_real = false)
{
    auto valid = physical.Validate();
    if (!valid.ok()) return valid.status();
    std::vector<std::uint64_t> logical(map.logical_element_count);
    for (std::size_t i = 0; i < map.assignments.size(); ++i) {
        const auto &entry = map.assignments[i];
        if (entry.real_logical_index != kUnusedLogicalIndex)
            logical[entry.real_logical_index] = physical.real.local_share[i];
        if (entry.imaginary_logical_index != kUnusedLogicalIndex)
            logical[entry.imaginary_logical_index] = paired_real
                ? detail::UndoPairedRealPhase(physical.imag.local_share[i],
                    i % map.slots_per_ciphertext, physical.imag.numeric.ring_bits)
                : physical.imag.local_share[i];
    }
    auto numeric = physical.real.numeric;
    numeric.is_signed = signed_values;
    return MpcShareTensor{std::move(logical), shape, numeric,
                          ComplexComponent::kReal};
}

Result<void> SamePacking(const HePackingPlan &left, const HePackingPlan &right)
{
    return detail::RequireSamePacking(left, right);
}

HePackingPlan AttentionOutputPacking(
    const TransformerLayerConfig &layer, std::size_t slots,
    const std::vector<HeadTile> &tiles)
{
    HePackingPlan result;
    result.layout = Layout::kAttentionOutput;
    result.logical_shape = {{layer.tokens, layer.q_heads,
                             layer.value_head_width}};
    result.slot_count = slots;
    result.logical_locations.resize(layer.tokens * layer.q_heads *
                                    layer.value_head_width);
    for (const auto &tile : tiles) {
        const auto block = detail::TokenColumnStride(layer.tokens) * tile.head_count;
        result.ciphertext_count = std::max(
            result.ciphertext_count, tile.value_begin + tile.value_groups);
        for (std::size_t local_head = 0; local_head < tile.head_count;
             ++local_head) {
            const auto head = tile.head_begin + local_head;
            for (std::size_t token = 0; token < layer.tokens; ++token) {
                for (std::size_t dimension = 0;
                     dimension < layer.value_head_width; ++dimension) {
                    const auto group = dimension / (2 * tile.lanes);
                    const auto in_pair = dimension % (2 * tile.lanes);
                    const auto lane = in_pair % tile.lanes;
                    const auto component = in_pair < tile.lanes
                        ? ComplexLane::kReal : ComplexLane::kImaginary;
                    const auto slot = lane * block +
                                      local_head * detail::TokenColumnStride(layer.tokens) + token;
                    result.logical_locations[AttentionOutputIndex(
                        token, head, dimension, layer)].push_back(
                            {tile.value_begin + group, slot, component, head});
                }
            }
        }
    }
    return result;
}

using AvPlain = std::shared_ptr<const seal::Plaintext>;
struct AvMaskPair { AvPlain main, wrap; };
struct AvTilePlaintexts {
    std::vector<std::vector<AvMaskPair>> seeds;
    std::vector<AvPlain> parity;
    std::vector<AvMaskPair> babies, recovery;
};
struct PreparedAv final : PreparedBlock2Program {
    std::string profile;
    seal::parms_id_type input_id{};
    double probability_scale{}, value_scale{};
    std::size_t tokens{}, heads{}, width{};
    std::vector<AvTilePlaintexts> tiles;
};

Result<std::shared_ptr<const PreparedAv>> GetAvPlaintexts(
    const TransformerLayerConfig &layer, const HeContext &context,
    seal::parms_id_type input_id, double probability_scale, double value_scale,
    AttentionLn2ExecutionReport *report)
{
    const auto cache = ModelPlaintextCacheScope::Current();
    std::string key = "shared/" + context.profile + "/av/" +
        std::to_string(layer.tokens) + "/" + std::to_string(layer.q_heads) + "/" +
        std::to_string(layer.value_head_width);
    for (auto word : input_id) key += "/" + std::to_string(word);
    if (cache.first) {
        auto found = cache.first->GetPreparedBlock2Program(key);
        if (found.ok()) {
            auto plan = std::dynamic_pointer_cast<const PreparedAv>(found.value());
            if (!plan || !CloseScale(plan->probability_scale, probability_scale) ||
                !CloseScale(plan->value_scale, value_scale))
                return Status::FailedPrecondition("prepared AV input scale mismatch");
            return plan;
        }
        if (!cache.first->preparing_plaintexts()) return found.status();
    }
    auto tiles = MakeTiles(layer, context.slot_count());
    if (!tiles.ok()) return tiles.status();
    auto first = context.context->get_context_data(input_id);
    if (!first || !first->next_context_data() || !first->next_context_data()->next_context_data() ||
        !first->next_context_data()->next_context_data()->next_context_data())
        return Status::FailedPrecondition("AV plaintext preparation requires three data levels");
    const auto product_id = first->next_context_data()->next_context_data()->parms_id();
    const auto prepared_a_scale = DroppedPrime(context, first->next_context_data()->parms_id());
    const double target = std::ldexp(1., context.parameters.default_scale_bits);
    auto plan = std::make_shared<PreparedAv>();
    plan->profile = context.profile; plan->input_id = input_id;
    plan->probability_scale = probability_scale; plan->value_scale = value_scale;
    plan->tokens = layer.tokens; plan->heads = layer.q_heads; plan->width = layer.value_head_width;
    seal::CKKSEncoder encoder(*context.context);
    ModelPlaintextCacheScope shared_scope(cache.first, "shared/" + context.profile);
    const auto encode = [&](const auto &values, seal::parms_id_type id, double scale) -> AvPlain {
        std::vector<Complex> complex(values.begin(), values.end());
        auto plain = CompileModelPlaintext(encoder, complex, id, scale, "attention_value.mask");
        if (!plain.ok()) throw std::invalid_argument(plain.status().message());
        if (report) ++report->attention_value_plaintext_encodings;
        return plain.take_value();
    };
    const auto stride = detail::TokenColumnStride(layer.tokens);
    for (const auto &tile : tiles.value()) {
        AvTilePlaintexts masks;
        const auto block = stride * tile.head_count;
        const auto babies = std::min(layer.tokens, 2 * tile.lanes);
        const auto giants = (layer.tokens + babies - 1) / babies;
        const bool paired = tile.lanes % 2 == 0;
        const double a_scale = prepared_a_scale * DroppedPrime(context, input_id) / probability_scale;
        const double seed_scale = paired ? std::sqrt(a_scale) : a_scale;
        if (paired) for (std::size_t local = 0; local < 2; ++local) {
            std::vector<double> mask(context.slot_count());
            for (std::size_t lane = local; lane < tile.lanes; lane += 2)
                for (std::size_t h = 0; h < tile.head_count; ++h)
                    std::fill_n(mask.begin() + lane * block + h * stride, layer.tokens, 1.);
            masks.parity.push_back(encode(mask, input_id, a_scale / seed_scale));
        }
        for (std::size_t giant = 0; giant < giants; ++giant) {
            const auto shift = giant * babies;
            const auto diagonals = (std::min(babies, layer.tokens - shift) + 1) / 2;
            std::vector<AvMaskPair> seeds;
            for (std::size_t first_lane = 0; first_lane < diagonals; first_lane += paired ? 2 : 1) {
                std::vector<double> main(context.slot_count()), wrap(context.slot_count());
                const auto selected = std::min<std::size_t>(paired ? 2 : 1, diagonals - first_lane);
                for (std::size_t lane = first_lane; lane < first_lane + selected; ++lane)
                    for (std::size_t row = 0; row < layer.tokens; ++row)
                        for (std::size_t h = 0; h < tile.head_count; ++h)
                            (row >= shift ? main : wrap)[lane * block + h * stride + row] = .5;
                seeds.push_back({encode(main, input_id, seed_scale),
                    shift ? encode(wrap, input_id, seed_scale) : AvPlain{}});
            }
            masks.seeds.push_back(std::move(seeds));
            std::vector<double> main(context.slot_count()), wrap(context.slot_count());
            for (std::size_t lane = 0; lane < tile.lanes; ++lane)
                for (std::size_t row = 0; row < layer.tokens; ++row)
                    for (std::size_t h = 0; h < tile.head_count; ++h)
                        (row + shift < layer.tokens ? main : wrap)[lane * block + h * stride + row] = 1.;
            const auto scale = DroppedPrime(context, product_id);
            masks.recovery.push_back({encode(main, product_id, scale),
                shift ? encode(wrap, product_id, scale) : AvPlain{}});
        }
        for (std::size_t baby = 0; baby < babies; ++baby) {
            std::vector<Complex> main(context.slot_count()), wrap(context.slot_count());
            const auto phase = baby % 2 ? Complex{0., -1.} : Complex{1., 0.};
            for (std::size_t lane = 0; lane < tile.lanes; ++lane)
                for (std::size_t row = 0; row < layer.tokens; ++row)
                    for (std::size_t h = 0; h < tile.head_count; ++h) {
                        const auto origin = (lane * block + h * stride + row + baby) % context.slot_count();
                        (row + baby < layer.tokens ? main : wrap)[origin] = phase;
                    }
            const double scale = target * DroppedPrime(context, input_id) / value_scale;
            masks.babies.push_back({encode(main, input_id, scale),
                baby ? encode(wrap, input_id, scale) : AvPlain{}});
        }
        plan->tiles.push_back(std::move(masks));
    }
    if (cache.first) {
        auto stored = cache.first->PutPreparedBlock2Program(key, plan);
        if (!stored.ok()) return stored.status();
    }
    return std::shared_ptr<const PreparedAv>(std::move(plan));
}

Result<SealCipherTensor> ComputeAttentionValue(
    const SealCipherTensor &probabilities, const SealCipherTensor &value,
    const TransformerLayerConfig &layer, const HeContext &context,
    AttentionLn2ExecutionReport *report, std::size_t he_workers)
{
    auto probability_valid = probabilities.Validate(context);
    if (!probability_valid.ok()) return probability_valid.status();
    auto value_valid = value.Validate(context);
    if (!value_valid.ok()) return value_valid.status();
    auto plan_result = AttentionLn2PackingPlanner::AttentionValue(
        layer, context.slot_count());
    if (!plan_result.ok()) return plan_result.status();
    auto same_probability = SamePacking(
        probabilities.physical_packing,
        plan_result.value().probability_packing);
    if (!same_probability.ok()) return same_probability.status();
    auto expected_value = MultiHeadPackingPlan::ProjectionOutput(
        layer, Layout::kMultiHeadV, context.slot_count());
    if (!expected_value.ok()) return expected_value.status();
    auto same_value = SamePacking(value.physical_packing,
                                  expected_value.value());
    if (!same_value.ok()) return same_value.status();

    const auto probability_level = LevelOf(
        context, probabilities.ciphertexts.front().parms_id());
    const auto value_level = LevelOf(
        context, value.ciphertexts.front().parms_id());
    if (!probability_level.ok()) return probability_level.status();
    if (!value_level.ok()) return value_level.status();
    if (probability_level.value() != value_level.value())
        return Status::FailedPrecondition(
            "A*V inputs must share the same CKKS level");
    auto remaining = context.context->get_context_data(
        probabilities.ciphertexts.front().parms_id());
    for (std::size_t step = 0; step < detail::kAttentionValueDepth; ++step) {
        if (!remaining || !remaining->next_context_data())
            return Status::FailedPrecondition(
                "A*V requires three remaining CKKS data levels");
        remaining = remaining->next_context_data();
    }
    for (const auto &ct : probabilities.ciphertexts)
        if (ct.parms_id() != value.ciphertexts.front().parms_id())
            return Status::FailedPrecondition(
                "Softmax and V do not have the same parms_id");
    const double target = std::ldexp(
        1.0, context.parameters.default_scale_bits);
    if (!CloseScale(probabilities.ciphertexts.front().scale(), target) ||
        !CloseScale(value.ciphertexts.front().scale(), target))
        return Status::FailedPrecondition("A*V input scale mismatch");
    for (const auto *tensor : {&probabilities, &value})
        for (const auto &ct : tensor->ciphertexts)
            if (ct.parms_id() != probabilities.ciphertexts.front().parms_id() ||
                std::abs(ct.scale() / target - 1.) > 1e-6)
                return Status::FailedPrecondition("AV preparation needs uniform input levels and scales");

    auto tiles_result = MakeTiles(layer, context.slot_count());
    if (!tiles_result.ok()) return tiles_result.status();
    const auto &tiles = tiles_result.value();
    auto program = GetAvPlaintexts(layer, context, probabilities.ciphertexts.front().parms_id(),
        probabilities.ciphertexts.front().scale(), value.ciphertexts.front().scale(), report);
    if (!program.ok()) return program.status();
    seal::Evaluator evaluator(*context.context);
    std::vector<seal::Ciphertext> output(
        plan_result.value().output_packing.ciphertext_count);

    for (std::size_t tile_index = 0; tile_index < tiles.size(); ++tile_index) {
        const auto &tile = tiles[tile_index];
        const auto &masks = program.value()->tiles[tile_index];
        const auto block = detail::TokenColumnStride(layer.tokens) * tile.head_count;
        // One complex A ciphertext is one giant group: each physical lane
        // carries two consecutive real diagonals. V retains its native packing.
        const auto baby_steps = std::min(layer.tokens, 2 * tile.lanes);
        const auto giant_steps =
            (layer.tokens + baby_steps - 1) / baby_steps;
        if (giant_steps != tile.probability_groups)
            return Status::Internal(
                "A*V BSGS groups disagree with the probability packing");

        const auto preprocessed_level = context.context->get_context_data(
            probabilities.ciphertexts[tile.probability_begin].parms_id())
                                ->next_context_data();
        if (!preprocessed_level || !preprocessed_level->next_context_data())
            return Status::FailedPrecondition("A*V chain is too short");
        // A is prepared at the prime that the ct-ct product will drop.
        // At the common seven-prime suffix, A(L4) * V(L4) rescales back to
        // the public target scale at L5 without rewriting scale metadata.
        const double prepared_a_scale = DroppedPrime(
            context, preprocessed_level->parms_id());

        // Step 1: apply the giant row displacement to the complex input
        // once, then replicate two complex diagonals together. Inner-rotation
        // masks are folded into seed selection so all A preprocessing still
        // consumes only one level.
        const auto probability_rotations_before = report ? report->rotations : 0;
        std::vector<seal::Ciphertext> prepared_a(layer.tokens);
        for (std::size_t giant = 0; giant < giant_steps; ++giant) {
            const auto diagonal_begin = giant * baby_steps;
            const auto valid_babies = std::min(
                baby_steps, layer.tokens - diagonal_begin);
            const auto complex_diagonals = (valid_babies + 1) / 2;
            const auto row_shift = diagonal_begin;
            const auto &source = probabilities.ciphertexts[
                tile.probability_begin + giant];
            auto main = Rotate(source, -static_cast<int>(
                row_shift), context, evaluator, report);
            std::optional<seal::Ciphertext> wrapped;
            if (row_shift != 0)
                wrapped = Rotate(source, static_cast<int>(
                    (layer.tokens - row_shift)),
                    context, evaluator, report);

            const bool paired = tile.lanes % 2 == 0;
            for (std::size_t first = 0; first < complex_diagonals;
                 first += paired ? 2 : 1) {
                const auto selected = std::min<std::size_t>(
                    paired ? 2 : 1, complex_diagonals - first);
                const auto &seed_masks = masks.seeds[giant][first / (paired ? 2 : 1)];
                seal::Ciphertext seed;
                evaluator.multiply_plain(main, *seed_masks.main, seed);
                if (report) ++report->ciphertext_plain_multiplications;
                if (wrapped) {
                    seal::Ciphertext term;
                    evaluator.multiply_plain(*wrapped, *seed_masks.wrap, term);
                    evaluator.add_inplace(seed, term);
                    if (report) ++report->ciphertext_plain_multiplications;
                }
                // Expand in place: select left/right from the seed position,
                // avoiding a separate alignment rotation and physical wrap.
                auto replicated = seed;
                std::size_t populated = paired ? 2 : 1;
                while (populated < tile.lanes) {
                    auto copy = Rotate(replicated,
                        static_cast<int>(ReplicationRotation(
                            first, populated, block)),
                        context, evaluator, report);
                    evaluator.add_inplace(replicated, copy);
                    populated *= 2;
                }
                for (std::size_t local = 0; local < selected; ++local) {
                    auto diagonal = replicated;
                    if (paired) {
                        evaluator.multiply_plain_inplace(diagonal, *masks.parity[local]);
                        auto complement = Rotate(diagonal,
                            local == 0 ? -static_cast<int>(block)
                                       : static_cast<int>(block),
                            context, evaluator, report);
                        evaluator.add_inplace(diagonal, complement);
                        if (report) ++report->ciphertext_plain_multiplications;
                    }
                    evaluator.rescale_to_next_inplace(diagonal);
                    auto normalized = NormalizeScale(diagonal, prepared_a_scale);
                    if (!normalized.ok()) return normalized.status();
                    seal::Ciphertext conjugated;
                    evaluator.complex_conjugate(
                        diagonal, context.galois_keys, conjugated);
                    const auto index = diagonal_begin + 2 * (first + local);
                    evaluator.add(diagonal, conjugated, prepared_a[index]);
                    if (index + 1 < diagonal_begin + valid_babies)
                        evaluator.sub(diagonal, conjugated, prepared_a[index + 1]);
                    // The odd diagonal is i*D. Step 2 folds -i into the
                    // corresponding V mask, making their product D*V without
                    // another plaintext multiplication or CKKS level.
                    if (report) {
                        ++report->rescales;
                        ++report->conjugations;
                    }
                }
            }
        }
        if (report)
            report->attention_probability_rotations +=
                report->rotations - probability_rotations_before;

        std::vector<AttentionLn2ExecutionReport> group_reports(
            tile.value_groups);
        auto group_status = detail::ParallelRanges(
            tile.value_groups, he_workers, 1,
            [&](std::size_t begin, std::size_t end) -> Result<void> {
            seal::Evaluator evaluator(*context.context);
            for (std::size_t value_group = begin;
                 value_group < end; ++value_group) {
            auto *report = &group_reports[value_group];
            const auto &v_source = value.ciphertexts[
                tile.value_begin + value_group];

            // Step 2: share one boundary rotation across all baby steps.
            // Masks are applied in source coordinates before a single baby
            // rotation; this also works with partially occupied CKKS vectors.
            std::optional<seal::Ciphertext> boundary;
            if (baby_steps > 1)
                boundary = Rotate(v_source, -static_cast<int>(layer.tokens),
                                  context, evaluator, report);
            std::vector<seal::Ciphertext> prepared_v(baby_steps);
            for (std::size_t baby = 0; baby < baby_steps; ++baby) {
                const auto shift = baby;
                seal::Ciphertext main;
                evaluator.multiply_plain(v_source, *masks.babies[baby].main, main);
                std::size_t multiplications = 1;
                if (baby != 0) {
                    seal::Ciphertext term;
                    evaluator.multiply_plain(*boundary, *masks.babies[baby].wrap, term);
                    evaluator.add_inplace(main, term);
                    ++multiplications;
                }
                main = Rotate(main, static_cast<int>(shift), context,
                              evaluator, report);
                evaluator.rescale_to_next_inplace(main);
                auto normalized = NormalizeScale(main, target);
                if (!normalized.ok()) return normalized.status();
                report->ciphertext_plain_multiplications += multiplications;
                ++report->rescales;
                prepared_v[baby] = std::move(main);
            }
            report->attention_value_rotations = report->rotations;

            // Step 3: ciphertext x ciphertext products. Products belonging to
            // one giant group are accumulated before one lazy relinearization
            // and rescale. Giant groups are then row-inner recovered and added.
            std::vector<seal::Ciphertext> partials;
            partials.reserve(giant_steps);
            for (std::size_t giant = 0; giant < giant_steps; ++giant) {
                const auto diagonal_begin = giant * baby_steps;
                const auto valid_babies = std::min(
                    baby_steps, layer.tokens - diagonal_begin);
                seal::Ciphertext partial;
                for (std::size_t baby = 0; baby < valid_babies; ++baby) {
                    seal::Ciphertext product;
                    evaluator.multiply(
                        prepared_a[diagonal_begin + baby],
                        prepared_v[baby], product);
                    if (baby == 0)
                        partial = std::move(product);
                    else
                        evaluator.add_inplace(partial, product);
                    if (report) ++report->ciphertext_multiplications;
                }
                evaluator.relinearize_inplace(
                    partial, context.relinearization_keys);
                evaluator.rescale_to_next_inplace(partial);
                auto normalized = NormalizeScale(partial, target);
                if (!normalized.ok()) return normalized.status();
                if (report) {
                    ++report->relinearizations;
                    ++report->rescales;
                }
                partials.push_back(std::move(partial));
            }

            const auto recovery_rotations_before = report->rotations;
            std::optional<seal::Ciphertext> accumulated;
            for (std::size_t giant = 0; giant < giant_steps; ++giant) {
                const auto row_shift = giant * baby_steps;
                const int shift = static_cast<int>(
                    row_shift);
                auto recovered = Rotate(
                    partials[giant], shift, context, evaluator, report);
                evaluator.multiply_plain_inplace(recovered, *masks.recovery[giant].main);
                std::size_t multiplications = 1;
                if (masks.recovery[giant].wrap) {
                    auto wrapped = Rotate(
                        partials[giant], shift - static_cast<int>(layer.tokens),
                        context, evaluator, report);
                    evaluator.multiply_plain_inplace(wrapped, *masks.recovery[giant].wrap);
                    evaluator.add_inplace(recovered, wrapped);
                    ++multiplications;
                }
                report->ciphertext_plain_multiplications += multiplications;
                if (!accumulated)
                    accumulated = std::move(recovered);
                else
                    evaluator.add_inplace(*accumulated, recovered);
            }
            if (!accumulated)
                return Status::Internal("A*V Step 3 produced no output");
            evaluator.rescale_to_next_inplace(*accumulated);
            auto normalized = NormalizeScale(*accumulated, target);
            if (!normalized.ok()) return normalized.status();
            ++report->rescales;
            output[tile.value_begin + value_group] =
                std::move(*accumulated);
            report->attention_recovery_rotations =
                report->rotations - recovery_rotations_before;
            }
            return {};
        }, detail::EstimatedHeWorkerBytes(
            context.parameters.polynomial_degree,
            context.parameters.coefficient_modulus_bits.size(),
            baby_steps + giant_steps + 8));
        if (!group_status.ok()) return group_status.status();
        if (report) {
            for (const auto &counts : group_reports) {
                report->rotations += counts.rotations;
                report->conjugations += counts.conjugations;
                report->ciphertext_plain_multiplications +=
                    counts.ciphertext_plain_multiplications;
                report->ciphertext_multiplications +=
                    counts.ciphertext_multiplications;
                report->relinearizations += counts.relinearizations;
                report->rescales += counts.rescales;
                report->modulus_switches += counts.modulus_switches;
                report->attention_value_rotations += counts.attention_value_rotations;
                report->attention_recovery_rotations += counts.attention_recovery_rotations;
            }
        }
    }

    SealCipherTensor result;
    result.parameter_profile = context.profile;
    result.ciphertexts = std::move(output);
    result.logical_spec.shape = {{layer.tokens, layer.q_heads,
                                  layer.value_head_width}};
    result.logical_spec.domain = Domain::kHeCiphertext;
    result.logical_spec.layout = Layout::kAttentionOutput;
    result.logical_spec.numeric = value.logical_spec.numeric;
    result.logical_spec.he_region = HeRegionId::kRegion2;
    result.logical_spec.he_parameter_profile = context.profile;
    result.logical_spec.planned_level = value_level.value() + detail::kAttentionValueDepth;
    result.logical_spec.planned_scale_bits =
        context.parameters.default_scale_bits;
    result.physical_packing = plan_result.take_value().output_packing;
    return result;
}

struct ProjectionKey {
    std::size_t source_ct{};
    std::size_t target_ct{};
    ComplexLane component{ComplexLane::kReal};
    int rotation{};
    bool operator<(const ProjectionKey &other) const noexcept
    {
        return std::tie(source_ct, target_ct, component, rotation) <
               std::tie(other.source_ct, other.target_ct,
                        other.component, other.rotation);
    }
};

Complex Extraction(ComplexLane component)
{
    return component == ComplexLane::kReal ? Complex{0.5, 0.0}
                                           : Complex{0.0, -0.5};
}

Complex Destination(ComplexLane component)
{
    return component == ComplexLane::kReal ? Complex{1.0, 0.0}
                                           : Complex{0.0, 1.0};
}

Result<std::shared_ptr<const PreparedProjectionPlan>> CompileRealOutputBsgs(
    const SealCipherTensor &input, const DenseMatrixParameter &weights,
    const std::optional<DenseVectorParameter> &bias,
    const TransformerLayerConfig &layer, const HeContext &context)
{
    auto split = AttentionOutputProjectionKernel::RealColumnBsgs(layer, context.slot_count());
    if (!split.ok()) return split.status();
    auto expected = AttentionLn2PackingPlanner::AttentionValue(layer, context.slot_count());
    if (!expected.ok()) return expected.status();
    auto same = SamePacking(input.physical_packing, expected.value().output_packing);
    if (!same.ok()) return Status::Invalid("Wo input must use canonical A*V packing");
    auto target = AttentionLn2PackingPlanner::RealStrictColumns(
        {{layer.tokens, layer.model_width}}, context.slot_count());
    if (!target.ok()) return target.status();
    const auto stride = split.value().token_stride;
    const auto columns = split.value().columns_per_ciphertext;
    struct SourceRows { std::optional<std::size_t> real, imaginary; };
    std::vector<std::vector<SourceRows>> rows(input.ciphertexts.size(),
                                             std::vector<SourceRows>(columns));
    // Invert the actual token-zero packing. A source slot's two components
    // can be different heads/dimensions in different packing geometries.
    for (std::size_t feature = 0; feature < AttentionWidth(layer); ++feature) {
        const auto &location = input.physical_packing.Locations(feature).front();
        auto &pair = rows.at(location.ciphertext).at(location.slot / stride);
        auto &row = location.component == ComplexLane::kReal ? pair.real : pair.imaginary;
        if (row) return Status::Invalid("Wo input has overlapping source components");
        row = feature;
    }
    struct DestinationColumn { std::size_t feature, slot; };
    std::vector<std::vector<DestinationColumn>> destinations(target.value().ciphertext_count);
    for (std::size_t feature = 0; feature < layer.model_width; ++feature) {
        const auto &location = target.value().Locations(feature).front();
        destinations[location.ciphertext].push_back({feature, location.slot});
    }
    const auto input_id = input.ciphertexts.front().parms_id();
    const auto next = context.context->get_context_data(input_id)->next_context_data();
    if (!next) return Status::FailedPrecondition("Wo has no next CKKS level");
    const double output_scale = std::ldexp(1.0, context.parameters.default_scale_bits);
    const double plain_scale = output_scale * DroppedPrime(context, input_id) /
                               input.ciphertexts.front().scale();
    auto prepared = std::make_shared<PreparedProjectionPlan>();
    prepared->profile = context.profile;
    prepared->input_parms_id = input_id;
    prepared->input_scale = input.ciphertexts.front().scale();
    prepared->input_packing = input.physical_packing;
    prepared->output_packing = target.take_value();
    prepared->raw_complex_input = true;
    prepared->realify_output = true;
    prepared->reuse_baby_rotations = true;
    prepared->bias_after_rescale = true;
    prepared->output_scale_bits = context.parameters.default_scale_bits;
    prepared->bias_plaintexts.resize(destinations.size());
    // Exact phase at scale 1: no CKKS encoding error or extra rescale.
    // Kept separate from model weights (and the virtual weight pool).
    const auto data = context.context->get_context_data(input_id);
    const auto degree = data->parms().poly_modulus_degree();
    const auto primes = data->parms().coeff_modulus().size();
    seal::Plaintext phase(degree * primes);
    std::fill_n(phase.data(), phase.coeff_count(), std::uint64_t{0});
    for (std::size_t rns = 0; rns < primes; ++rns) {
        phase[rns * degree + degree / 2] = 1;
        seal::util::ntt_negacyclic_harvey(
            phase.data() + rns * degree, data->small_ntt_tables()[rns]);
    }
    phase.parms_id() = input_id;
    phase.scale() = 1.0;
    prepared->fixed_phase = std::make_shared<const seal::Plaintext>(std::move(phase));
    seal::CKKSEncoder encoder(*context.context);
    for (std::size_t out = 0; out < destinations.size(); ++out) {
        bool initialized = false;
        for (std::size_t j = 0; j < split.value().giant_steps; ++j) {
            const auto giant = j * split.value().baby_steps * stride;
            PreparedProjectionGroup group;
            group.target_ct = out;
            group.giant_rotation = CanonicalRotation(static_cast<long long>(giant), context.slot_count());
            // One giant rotation after summing ALL source ciphertexts.
            for (std::size_t source = 0; source < rows.size(); ++source)
                for (std::size_t i = 0; i < split.value().baby_steps; ++i) {
                    const auto rotation = giant + i * stride;
                    std::vector<Complex> mask(context.slot_count());
                    std::vector<Complex> phase_mask(context.slot_count());
                    bool nonzero = false, phase_nonzero = false;
                    for (const auto &dest : destinations[out]) {
                        const auto source_column = ((dest.slot + rotation) % context.slot_count()) / stride;
                        const auto &pair = rows[source][source_column];
                        const double real = pair.real
                            ? weights.row_major[*pair.real * layer.model_width + dest.feature] : 0.0;
                        const double imaginary = pair.imaginary
                            ? weights.row_major[*pair.imaginary * layer.model_width + dest.feature] : 0.0;
                        nonzero |= real != 0.0;
                        phase_nonzero |= imaginary != 0.0;
                        // Offline Rot(diagonal, -giant): no HE rotation.
                        const auto adjusted_slot = (dest.slot + giant) % context.slot_count();
                        for (std::size_t token = 0; token < layer.tokens; ++token) {
                            const auto slot = adjusted_slot + token;
                            mask[slot] = Complex{0.5 * real, 0.0};
                            // X^(N/2) evaluates to i*(-1)^slot in SEAL's
                            // generator-3 slot order. This real sign mask
                            // turns it into -i for the second weight channel.
                            phase_mask[slot] = Complex{
                                (slot % 2 == 0 ? -0.5 : 0.5) * imaginary, 0.0};
                        }
                    }
                    const auto baby = CanonicalRotation(static_cast<long long>(i * stride),
                                                        context.slot_count());
                    if (nonzero) {
                        auto encoded = CompileProjectionPlaintext(encoder, mask, input_id, plain_scale,
                                                              "attention.wo.bsgs.diagonal", context);
                        if (!encoded.ok()) return encoded.status();
                        group.terms.push_back({source, ComplexLane::kReal, baby, encoded.take_value()});
                    }
                    if (phase_nonzero) {
                        auto encoded = CompileProjectionPlaintext(encoder, phase_mask, input_id, plain_scale,
                                                              "attention.wo.bsgs.diagonal", context);
                        if (!encoded.ok()) return encoded.status();
                        group.phase_terms.push_back({source, ComplexLane::kReal, baby, encoded.take_value()});
                    }
                }
            if (!group.terms.empty() || !group.phase_terms.empty()) {
                initialized = true;
                prepared->groups.push_back(std::move(group));
            }
        }
        if (!initialized) return Status::Internal("Wo generated an empty output group");
        if (bias) {
            std::vector<Complex> mask(context.slot_count());
            for (const auto &dest : destinations[out])
                for (std::size_t token = 0; token < layer.tokens; ++token)
                    mask[dest.slot + token] = bias->values[dest.feature];
            auto encoded = CompileModelPlaintext(encoder, mask, next->parms_id(), output_scale,
                                                  "attention.wo.bias");
            if (!encoded.ok()) return encoded.status();
            prepared->bias_plaintexts[out] = encoded.take_value();
        }
    }
    return std::shared_ptr<const PreparedProjectionPlan>(std::move(prepared));
}

Result<SealCipherTensor> ProjectRealOutput(
    const SealCipherTensor &input, const DenseMatrixParameter &weights,
    const std::optional<DenseVectorParameter> &bias,
    const TransformerLayerConfig &layer, const HeContext &context,
    AttentionLn2ExecutionReport *report, std::size_t he_workers)
{
    auto valid = input.Validate(context);
    if (!valid.ok()) return valid.status();
    if (weights.spec.rows != AttentionWidth(layer) || weights.spec.columns != layer.model_width ||
        weights.row_major.size() != AttentionWidth(layer) * layer.model_width)
        return Status::Invalid("Wo payload has the wrong rectangular shape");
    if (bias && bias->values.size() != layer.model_width)
        return Status::Invalid("bo payload has the wrong size");
    auto level = LevelOf(context, input.ciphertexts.front().parms_id());
    if (!level.ok()) return level.status();
    const auto cache = ModelPlaintextCacheScope::Current();
    const auto key = cache.second + "/attention/" + weights.spec.name + "/real.weights.bsgs";
    std::shared_ptr<const PreparedProjectionPlan> program;
    if (cache.first && !cache.first->preparing_plaintexts()) {
        auto prepared = cache.first->GetPreparedProjection(key);
        if (!prepared.ok()) return prepared.status();
        program = prepared.take_value();
        if (!program->raw_complex_input || !program->realify_output || !program->fixed_phase ||
            program->output_packing.logical_shape != TensorShape{{layer.tokens, layer.model_width}} ||
            program->output_packing.slot_count != context.slot_count())
            return Status::FailedPrecondition("prepared real Wo packing/mode mismatch");
    } else {
        auto compiled = CompileRealOutputBsgs(input, weights, bias, layer, context);
        if (!compiled.ok()) return compiled.status();
        program = compiled.take_value();
        if (cache.first) {
            auto stored = cache.first->PutPreparedProjection(key, program);
            if (!stored.ok()) return stored.status();
        }
    }
    std::vector<seal::Ciphertext> ciphertexts;
    if (cache.first && cache.first->preparing_plaintexts()) {
        const auto next = context.context->get_context_data(program->input_parms_id)->next_context_data();
        ciphertexts.resize(program->output_packing.ciphertext_count);
        for (auto &ct : ciphertexts) {
            ct.parms_id() = next->parms_id();
            ct.scale() = std::ldexp(1.0, program->output_scale_bits);
        }
    } else {
        PreparedProjectionCounts counts;
        auto evaluated = EvaluatePreparedProjection(*program, input, context, &counts, nullptr, he_workers);
        if (!evaluated.ok()) return evaluated.status();
        ciphertexts = evaluated.take_value();
        if (report) {
            report->rotations += counts.rotations;
            report->conjugations += counts.conjugations;
            report->ciphertext_plain_multiplications += counts.ciphertext_plain_multiplications;
            report->rescales += counts.rescales;
            report->wo_baby_rotations += counts.baby_rotations;
            report->wo_giant_rotations += counts.giant_rotations;
            report->wo_output_conjugations += counts.output_conjugations;
            report->wo_phase_multiplications += counts.phase_multiplications;
        }
    }
    SealCipherTensor result;
    result.parameter_profile = context.profile;
    result.ciphertexts = std::move(ciphertexts);
    result.physical_packing = program->output_packing;
    result.logical_spec = input.logical_spec;
    result.logical_spec.shape = {{layer.tokens, layer.model_width}};
    result.logical_spec.layout = Layout::kStrictColumn;
    result.logical_spec.he_region = HeRegionId::kRegion4;
    result.logical_spec.he_parameter_profile = context.profile;
    result.logical_spec.planned_level = level.value() + 1;
    result.logical_spec.planned_scale_bits = context.parameters.default_scale_bits;
    return result;
}

Result<SealCipherTensor> ProjectOutput(
    const SealCipherTensor &input, const DenseMatrixParameter &weights,
    const std::optional<DenseVectorParameter> &bias,
    const TransformerLayerConfig &layer, const HeContext &context,
    AttentionLn2ExecutionReport *report, std::size_t he_workers)
{
    auto valid = input.Validate(context);
    if (!valid.ok()) return valid.status();
    auto input_level = LevelOf(context, input.ciphertexts.front().parms_id());
    if (!input_level.ok()) return input_level.status();
    const auto input_width = layer.q_heads * layer.value_head_width;
    if (weights.spec.rows != input_width ||
        weights.spec.columns != layer.model_width ||
        weights.row_major.size() != input_width * layer.model_width)
        return Status::Invalid("Wo payload has the wrong rectangular shape");
    if (bias && bias->values.size() != layer.model_width)
        return Status::Invalid("bo payload has the wrong size");
    const auto cache = ModelPlaintextCacheScope::Current();
    const auto prepared_key = cache.second + "/attention/" +
                              weights.spec.name + "/complex";
    if (cache.first && !cache.first->preparing_plaintexts()) {
        auto prepared = cache.first->GetPreparedProjection(prepared_key);
        if (!prepared.ok()) return prepared.status();
        const auto &packing = prepared.value()->output_packing;
        if (packing.layout != Layout::kComplexStrictColumn ||
            packing.logical_shape !=
                TensorShape{{layer.tokens, layer.model_width}} ||
            packing.slot_count != context.slot_count())
            return Status::FailedPrecondition(
                "prepared Wo output packing mismatch");
        PreparedProjectionCounts prepared_counts;
        auto ciphertexts = EvaluatePreparedProjection(
            *prepared.value(), input, context, &prepared_counts);
        if (!ciphertexts.ok()) return ciphertexts.status();
        if (report) {
            report->rotations += prepared_counts.rotations;
            report->conjugations += prepared_counts.conjugations;
            report->ciphertext_plain_multiplications +=
                prepared_counts.ciphertext_plain_multiplications;
            report->rescales += prepared_counts.rescales;
        }
        SealCipherTensor result;
        result.parameter_profile = context.profile;
        result.ciphertexts = ciphertexts.take_value();
        result.logical_spec.shape = {{layer.tokens, layer.model_width}};
        result.logical_spec.domain = Domain::kHeCiphertext;
        result.logical_spec.layout = Layout::kComplexStrictColumn;
        result.logical_spec.numeric = input.logical_spec.numeric;
        result.logical_spec.he_region = HeRegionId::kRegion4;
        result.logical_spec.he_parameter_profile = context.profile;
        result.logical_spec.planned_level = input_level.value() + 1;
        result.logical_spec.planned_scale_bits =
            context.parameters.default_scale_bits;
        result.physical_packing = prepared.value()->output_packing;
        return result;
    }

    auto target_plan = AttentionLn2PackingPlanner::ComplexStrictColumns(
        {{layer.tokens, layer.model_width}}, context.slot_count());
    if (!target_plan.ok()) return target_plan.status();

    std::map<ProjectionKey, std::vector<Complex>> masks;
    for (std::size_t token = 0; token < layer.tokens; ++token) {
        for (std::size_t output_feature = 0;
             output_feature < layer.model_width; ++output_feature) {
            const auto output_logical = token * layer.model_width + output_feature;
            const auto &destination =
                target_plan.value().Locations(output_logical).front();
            for (std::size_t head = 0; head < layer.q_heads; ++head) {
                for (std::size_t dimension = 0;
                     dimension < layer.value_head_width; ++dimension) {
                    const auto input_column =
                        head * layer.value_head_width + dimension;
                    const auto input_logical = AttentionOutputIndex(
                        token, head, dimension, layer);
                    const auto &source =
                        input.physical_packing.Locations(input_logical).front();
                    const int rotation = CanonicalRotation(
                        static_cast<long long>(source.slot) -
                            static_cast<long long>(destination.slot),
                        context.slot_count());
                    ProjectionKey key{source.ciphertext,
                                      destination.ciphertext,
                                      source.component, rotation};
                    auto inserted = masks.try_emplace(
                        key, context.slot_count(), Complex{});
                    inserted.first->second[destination.slot] +=
                        weights.row_major[input_column * layer.model_width +
                                          output_feature] *
                        Extraction(source.component) *
                        Destination(destination.component);
                }
            }
        }
    }

    if (cache.first && cache.first->preparing_plaintexts()) {
        const auto input_id = input.ciphertexts.front().parms_id();
        const auto next = context.context->get_context_data(input_id)
                              ->next_context_data();
        if (!next)
            return Status::FailedPrecondition(
                "Wo preparation has no next CKKS level");
        const double target = std::ldexp(
            1.0, context.parameters.default_scale_bits);
        const double plain_scale = target * DroppedPrime(context, input_id) /
            input.ciphertexts.front().scale();
        seal::CKKSEncoder encoder(*context.context);
        auto prepared = std::make_shared<PreparedProjectionPlan>();
        prepared->profile = context.profile;
        prepared->input_parms_id = input_id;
        prepared->input_scale = input.ciphertexts.front().scale();
        prepared->input_packing = input.physical_packing;
        prepared->output_packing = target_plan.value();
        prepared->bias_plaintexts.resize(target_plan.value().ciphertext_count);
        prepared->bias_after_rescale = true;
        prepared->output_scale_bits = context.parameters.default_scale_bits;
        std::vector<bool> initialized(target_plan.value().ciphertext_count);
        for (const auto &entry : masks) {
            if (std::none_of(entry.second.begin(), entry.second.end(),
                             [](Complex value) { return std::abs(value) != 0.0; }))
                continue;
            auto encoded = CompileModelPlaintext(
                encoder, entry.second, input_id, plain_scale,
                "attention.wo.diagonal");
            if (!encoded.ok()) return encoded.status();
            const auto &key = entry.first;
            prepared->groups.push_back({
                key.target_ct, 0,
                {{key.source_ct, key.component, key.rotation,
                  encoded.take_value()}}});
            initialized[key.target_ct] = true;
        }
        for (std::size_t ct = 0; ct < initialized.size(); ++ct) {
            if (!initialized[ct])
                return Status::Internal("Wo generated an empty output group");
            if (!bias) continue;
            std::vector<Complex> slots(context.slot_count());
            for (std::size_t logical = 0;
                 logical < target_plan.value().logical_locations.size();
                 ++logical) {
                const auto feature = logical % layer.model_width;
                const auto &location =
                    target_plan.value().Locations(logical).front();
                if (location.ciphertext != ct) continue;
                if (location.component == ComplexLane::kReal)
                    slots[location.slot].real(bias->values[feature]);
                else
                    slots[location.slot].imag(bias->values[feature]);
            }
            auto encoded = CompileModelPlaintext(
                encoder, slots, next->parms_id(), target,
                "attention.wo.bias");
            if (!encoded.ok()) return encoded.status();
            prepared->bias_plaintexts[ct] = encoded.take_value();
        }
        auto stored = cache.first->PutPreparedProjection(
            prepared_key, std::move(prepared));
        if (!stored.ok()) return stored.status();
        SealCipherTensor result;
        result.parameter_profile = context.profile;
        result.ciphertexts.resize(target_plan.value().ciphertext_count);
        for (auto &ct : result.ciphertexts) {
            ct.parms_id() = next->parms_id();
            ct.scale() = target;
        }
        result.logical_spec.shape = {{layer.tokens, layer.model_width}};
        result.logical_spec.domain = Domain::kHeCiphertext;
        result.logical_spec.layout = Layout::kComplexStrictColumn;
        result.logical_spec.numeric = input.logical_spec.numeric;
        result.logical_spec.he_region = HeRegionId::kRegion4;
        result.logical_spec.he_parameter_profile = context.profile;
        result.logical_spec.planned_level = input_level.value() + 1;
        result.logical_spec.planned_scale_bits =
            context.parameters.default_scale_bits;
        result.physical_packing = target_plan.take_value();
        return result;
    }

    std::vector<seal::Ciphertext> output(
        target_plan.value().ciphertext_count);
    std::map<std::pair<std::size_t, ComplexLane>, seal::Ciphertext> extracted;
    const double target = std::ldexp(
        1.0, context.parameters.default_scale_bits);
    seal::Evaluator extraction_evaluator(*context.context);
    for (const auto &entry : masks) {
        if (std::none_of(entry.second.begin(), entry.second.end(),
                         [](Complex value) { return std::abs(value) != 0.0; }))
            continue;
        const auto base_key = std::make_pair(entry.first.source_ct,
                                             entry.first.component);
        auto base = extracted.find(base_key);
        if (base == extracted.end()) {
            const auto &source = input.ciphertexts[entry.first.source_ct];
            seal::Ciphertext conjugated;
            extraction_evaluator.complex_conjugate(
                source, context.galois_keys, conjugated);
            seal::Ciphertext component;
            if (entry.first.component == ComplexLane::kReal)
                extraction_evaluator.add(source, conjugated, component);
            else
                extraction_evaluator.sub(source, conjugated, component);
            base = extracted.emplace(base_key, std::move(component)).first;
            if (report) ++report->conjugations;
        }
    }
    using MaskIterator = decltype(masks)::const_iterator;
    std::vector<std::vector<MaskIterator>> entries(output.size());
    for (auto entry = masks.cbegin(); entry != masks.cend(); ++entry) {
        if (std::none_of(entry->second.begin(), entry->second.end(),
                         [](Complex value) { return std::abs(value) != 0.0; }))
            continue;
        entries[entry->first.target_ct].push_back(entry);
    }
    std::vector<AttentionLn2ExecutionReport> per_output(output.size());
    std::shared_ptr<PreparedProjectionPlan> prepared;
    std::vector<std::vector<PreparedProjectionGroup>> per_output_plan;
    if (cache.first) {
        prepared = std::make_shared<PreparedProjectionPlan>();
        prepared->profile = context.profile;
        prepared->input_parms_id = input.ciphertexts.front().parms_id();
        prepared->input_scale = input.ciphertexts.front().scale();
        prepared->input_packing = input.physical_packing;
        prepared->output_packing = target_plan.value();
        prepared->bias_plaintexts.resize(output.size());
        prepared->bias_after_rescale = true;
        prepared->output_scale_bits = context.parameters.default_scale_bits;
        per_output_plan.resize(output.size());
    }
    auto projected = detail::ParallelRanges(
        output.size(), he_workers, 1,
        [&](std::size_t begin, std::size_t end) -> Result<void> {
            std::optional<ModelPlaintextCacheScope> cache_scope;
            if (cache.first) cache_scope.emplace(cache.first, cache.second);
            seal::Evaluator evaluator(*context.context);
            seal::CKKSEncoder encoder(*context.context);
            for (std::size_t ct = begin; ct < end; ++ct) {
                auto *local_report = &per_output[ct];
                bool initialized = false;
                for (const auto &entry : entries[ct]) {
                    const auto base_key = std::make_pair(
                        entry->first.source_ct, entry->first.component);
                    auto rotated = Rotate(
                        extracted.at(base_key), entry->first.rotation,
                        context, evaluator, local_report);
                    const double plain_scale = target *
                        DroppedPrime(context, rotated.parms_id()) /
                        rotated.scale();
                    auto encoded = CompileModelPlaintext(
                        encoder, entry->second, rotated.parms_id(), plain_scale,
                        "attention.wo.diagonal");
                    if (!encoded.ok()) return encoded.status();
                    auto plain = encoded.take_value();
                    if (prepared)
                        per_output_plan[ct].push_back({
                            ct, 0, {{entry->first.source_ct,
                                     entry->first.component,
                                     entry->first.rotation, plain}}});
                    evaluator.multiply_plain_inplace(rotated, *plain);
                    ++local_report->ciphertext_plain_multiplications;
                    if (!initialized) {
                        output[ct] = std::move(rotated);
                        initialized = true;
                    } else {
                        evaluator.add_inplace(output[ct], rotated);
                    }
                }
                if (!initialized)
                    return Status::Internal("Wo generated an empty output group");
                evaluator.rescale_to_next_inplace(output[ct]);
                auto normalized = NormalizeScale(output[ct], target);
                if (!normalized.ok()) return normalized.status();
                ++local_report->rescales;
                if (bias) {
                    std::vector<Complex> slots(context.slot_count());
                    for (std::size_t logical = 0;
                         logical < target_plan.value().logical_locations.size();
                         ++logical) {
                        const auto feature = logical % layer.model_width;
                        const auto &location =
                            target_plan.value().Locations(logical).front();
                        if (location.ciphertext != ct) continue;
                        if (location.component == ComplexLane::kReal)
                            slots[location.slot].real(bias->values[feature]);
                        else
                            slots[location.slot].imag(bias->values[feature]);
                    }
                    auto encoded = CompileModelPlaintext(
                        encoder, slots, output[ct].parms_id(), target,
                        "attention.wo.bias");
                    if (!encoded.ok()) return encoded.status();
                    auto plain = encoded.take_value();
                    if (prepared) prepared->bias_plaintexts[ct] = plain;
                    evaluator.add_plain_inplace(output[ct], *plain);
                }
            }
            return {};
        }, detail::EstimatedHeWorkerBytes(
            context.parameters.polynomial_degree,
            context.parameters.coefficient_modulus_bits.size(), 12));
    if (!projected.ok()) return projected.status();
    if (report)
        for (const auto &local : per_output) {
            report->rotations += local.rotations;
            report->ciphertext_plain_multiplications +=
                local.ciphertext_plain_multiplications;
            report->rescales += local.rescales;
        }
    if (prepared) {
        for (auto &groups : per_output_plan)
            for (auto &group : groups)
                prepared->groups.push_back(std::move(group));
        auto stored = cache.first->PutPreparedProjection(
            prepared_key, std::move(prepared));
        if (!stored.ok()) return stored.status();
    }
    SealCipherTensor result;
    result.parameter_profile = context.profile;
    result.ciphertexts = std::move(output);
    result.logical_spec.shape = {{layer.tokens, layer.model_width}};
    result.logical_spec.domain = Domain::kHeCiphertext;
    result.logical_spec.layout = Layout::kComplexStrictColumn;
    result.logical_spec.numeric = input.logical_spec.numeric;
    result.logical_spec.he_region = HeRegionId::kRegion4;
    result.logical_spec.he_parameter_profile = context.profile;
    result.logical_spec.planned_level = input_level.value() + 1;
    result.logical_spec.planned_scale_bits =
        context.parameters.default_scale_bits;
    result.physical_packing = target_plan.take_value();
    return result;
}

Result<SealCipherTensor> AddResidual(
    SealCipherTensor projected, const SealCipherTensor &residual,
    const HeContext &context, AttentionLn2ExecutionReport *report)
{
    auto projected_valid = projected.Validate(context);
    if (!projected_valid.ok()) return projected_valid.status();
    auto residual_valid = residual.Validate(context);
    if (!residual_valid.ok()) return residual_valid.status();
    auto same = SamePacking(projected.physical_packing,
                            residual.physical_packing);
    if (!same.ok()) return Status::Invalid(
        "attention residual must match Wo output packing");
    if (projected.ciphertexts.size() != residual.ciphertexts.size())
        return Status::Invalid("attention residual ciphertext count mismatch");
    seal::Evaluator evaluator(*context.context);
    for (std::size_t i = 0; i < projected.ciphertexts.size(); ++i) {
        const auto &source = residual.ciphertexts[i];
        if (!CloseScale(source.scale(), projected.ciphertexts[i].scale()))
            return Status::FailedPrecondition(
                "attention residual scale does not match Wo output");
        auto residual_level = LevelOf(context, source.parms_id());
        auto projected_level = LevelOf(
            context, projected.ciphertexts[i].parms_id());
        if (!residual_level.ok()) return residual_level.status();
        if (!projected_level.ok()) return projected_level.status();
        if (residual_level.value() > projected_level.value())
            return Status::FailedPrecondition(
                "attention residual is below the Wo output level");
        if (source.parms_id() != projected.ciphertexts[i].parms_id()) {
            auto aligned = source;
            evaluator.mod_switch_to_inplace(
                aligned, projected.ciphertexts[i].parms_id());
            if (report) ++report->modulus_switches;
            evaluator.add_inplace(projected.ciphertexts[i], aligned);
        } else evaluator.add_inplace(projected.ciphertexts[i], source);
    }
    return projected;
}

// Bounded immutable plaintext plans per calling thread. Keeping contexts alive
// prevents pointer reuse from making a cached plan match another SEAL context.
struct LayerNormStatisticsPlaintexts {
    std::shared_ptr<seal::SEALContext> context;
    std::string profile;
    seal::parms_id_type input_id;
    std::size_t tokens{}, width{};
    double target{}, epsilon{}, centering_scale{}, variance_scale{};
    std::vector<double> input_scales;
    std::vector<std::shared_ptr<const seal::Plaintext>> valid, mean, variance_masks;
    std::vector<std::size_t> square_groups;
    seal::Plaintext lift, epsilon_plain;
    bool lazy_square_sum{true};
};

std::shared_ptr<const LayerNormStatisticsPlaintexts> PrepareStatisticsPlaintexts(
    const SealCipherTensor &input, const TransformerLayerConfig &layer,
    const HeContext &context, std::size_t stride, std::size_t features,
    std::size_t &encodings, int output_scale_bits)
{
    struct CacheEntry {
        std::shared_ptr<const LayerNormStatisticsPlaintexts> plan;
        std::size_t bytes;
    };
    static thread_local std::list<CacheEntry> cache;
    static thread_local std::size_t cached_bytes = 0;
    const double target = std::ldexp(1.0, output_scale_bits);
    for (auto entry = cache.begin(); entry != cache.end(); ++entry) {
        const auto &cached = entry->plan;
        if (cached->context != context.context || cached->profile != context.profile ||
            cached->tokens != layer.tokens || cached->width != layer.model_width ||
            cached->target != target || cached->epsilon != layer.attention_norm_epsilon ||
            cached->input_id != input.ciphertexts.front().parms_id() ||
            cached->input_scales.size() != input.ciphertexts.size()) continue;
        bool scales_match = true;
        for (std::size_t i = 0; i < input.ciphertexts.size(); ++i)
            scales_match &= cached->input_scales[i] == input.ciphertexts[i].scale();
        if (scales_match) {
            auto result = cached;
            cache.splice(cache.begin(), cache, entry);
            return result;
        }
    }
    auto plan = std::make_shared<LayerNormStatisticsPlaintexts>();
    plan->context = context.context;
    plan->profile = context.profile;
    plan->input_id = input.ciphertexts.front().parms_id();
    plan->tokens = layer.tokens;
    plan->width = layer.model_width;
    plan->target = target;
    plan->epsilon = layer.attention_norm_epsilon;
    plan->centering_scale = target * DroppedPrime(context, plan->input_id);
    const auto centered_data = context.context->get_context_data(
        plan->input_id)->next_context_data();
    const auto square_data = centered_data->next_context_data();
    plan->variance_scale = target * DroppedPrime(context, square_data->parms_id());
    seal::CKKSEncoder encoder(*context.context);
    std::size_t encoded_bytes = 0;
    const auto record_encoding = [&](const seal::Plaintext &plain) {
        ++encodings;
        encoded_bytes += plain.coeff_count() * sizeof(std::uint64_t);
    };
    const auto encode = [&](const auto &mask,
                            seal::parms_id_type id, double scale) {
        auto plain = std::make_shared<seal::Plaintext>();
        encoder.encode(mask, id, scale, *plain);
        record_encoding(*plain);
        return std::shared_ptr<const seal::Plaintext>(std::move(plain));
    };
    // Full blocks share masks. A tail block and distinct physical input scales
    // get separate encodings, including realify's sqrt(2) scale difference.
    using PlainPair = std::pair<std::shared_ptr<const seal::Plaintext>,
                                std::shared_ptr<const seal::Plaintext>>;
    std::map<std::pair<std::size_t, double>, PlainPair> centering;
    std::map<std::size_t, std::size_t> variances;
    const auto count = input.ciphertexts.size();
    plan->input_scales.reserve(count);
    plan->valid.reserve(count);
    plan->mean.reserve(count);
    plan->square_groups.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        const auto columns = std::min(features, layer.model_width - i * features);
        const auto scale = plan->centering_scale / input.ciphertexts[i].scale();
        plan->input_scales.push_back(input.ciphertexts[i].scale());
        const auto key = std::make_pair(columns, scale);
        auto found = centering.find(key);
        auto variance = variances.find(columns);
        if (found == centering.end() || variance == variances.end()) {
            std::vector<double> valid(context.slot_count(), 0.0);
            std::vector<double> average(context.slot_count(), 0.0);
            for (std::size_t column = 0; column < columns; ++column)
                for (std::size_t token = 0; token < layer.tokens; ++token) {
                    const auto slot = column * stride + token;
                    valid[slot] = 1.0;
                    average[slot] = 1.0 / static_cast<double>(layer.model_width);
                }
            if (found == centering.end())
                found = centering.emplace(key, PlainPair{
                    encode(valid, plan->input_id, scale),
                    encode(average, plan->input_id, scale)}).first;
            if (variance == variances.end()) {
                variance = variances.emplace(columns, plan->variance_masks.size()).first;
                std::vector<Complex> packed_average(average.size());
                for (std::size_t slot = 0; slot < average.size(); ++slot)
                    packed_average[slot] = slot % stride < (layer.tokens + 1) / 2
                        ? Complex{average[slot], 0.0} : Complex{0.0, average[slot]};
                plan->variance_masks.push_back(encode(
                    packed_average, square_data->parms_id(),
                    plan->variance_scale / target));
            }
        }
        plan->valid.push_back(found->second.first);
        plan->mean.push_back(found->second.second);
        plan->square_groups.push_back(variance->second);
    }
    const double lift_scale = std::sqrt(target * DroppedPrime(
        context, centered_data->parms_id())) / target;
    if (std::abs(std::round(lift_scale) / lift_scale - 1.0) > 1e-3)
        throw std::invalid_argument("LayerNorm square scale lift has insufficient precision; select another profile output scale");
    encoder.encode(1.0, centered_data->parms_id(), lift_scale, plan->lift);
    record_encoding(plan->lift);
    std::vector<std::size_t> group_sizes(plan->variance_masks.size());
    for (auto group : plan->square_groups) ++group_sizes[group];
    const auto maximum_group = *std::max_element(group_sizes.begin(), group_sizes.end());
    // Reserve public scale headroom for accumulation. Profiles without enough
    // room retain per-input relinearization/rescale and stream masked squares.
    const double sum_scale_bits = 2.0 * std::log2(target * lift_scale) +
        std::ceil(std::log2(static_cast<double>(maximum_group))) + 2.0;
    plan->lazy_square_sum = sum_scale_bits < centered_data->total_coeff_modulus_bit_count();
    std::vector<Complex> epsilon(context.slot_count());
    for (std::size_t token = 0; token < layer.tokens; ++token)
        epsilon[token] = token < (layer.tokens + 1) / 2
            ? Complex{layer.attention_norm_epsilon, 0.0} : Complex{0.0, layer.attention_norm_epsilon};
    encoder.encode(epsilon, square_data->parms_id(), plan->variance_scale,
                   plan->epsilon_plain);
    record_encoding(plan->epsilon_plain);
    // Bound persistent plaintext storage; an oversized plan remains call-local.
    constexpr std::size_t kCacheBytes = 64ULL << 20;
    const auto overhead = sizeof(LayerNormStatisticsPlaintexts) +
        plan->profile.size() + plan->variance_masks.capacity() * sizeof(std::shared_ptr<const seal::Plaintext>);
    encoded_bytes += overhead;
    if (encoded_bytes <= kCacheBytes &&
        count <= (kCacheBytes - encoded_bytes) /
            (sizeof(double) + sizeof(std::size_t) + 2 * sizeof(std::shared_ptr<const seal::Plaintext>))) {
        const auto bytes = encoded_bytes + count *
            (sizeof(double) + sizeof(std::size_t) + 2 * sizeof(std::shared_ptr<const seal::Plaintext>));
        while (!cache.empty() && (cache.size() >= 8 || cached_bytes > kCacheBytes - bytes)) {
            cached_bytes -= cache.back().bytes;
            cache.pop_back();
        }
        cache.push_front({plan, bytes});
        cached_bytes += bytes;
    }
    return plan;
}

struct StatisticsGeometry {
    std::size_t stride, features, input_level;
};

Result<StatisticsGeometry> ValidateStatisticsInput(
    const SealCipherTensor &input, const TransformerLayerConfig &layer,
    const HeContext &context)
{
    auto valid = input.Validate(context);
    if (!valid.ok()) return valid.status();
    if (input.ciphertexts.empty() || layer.tokens == 0 || layer.model_width == 0)
        return Status::Invalid("LayerNorm statistics has an empty input");
    const auto stride = NextPowerOfTwo(layer.tokens);
    const auto features = context.slot_count() / stride;
    if (features == 0 || (features & (features - 1)) != 0)
        return Status::FailedPrecondition(
            "strict-column reduction requires a power-of-two feature block");
    const auto &packing = input.physical_packing;
    if (packing.layout != Layout::kStrictColumn ||
        packing.logical_shape != TensorShape{{layer.tokens, layer.model_width}} ||
        packing.ciphertext_count != (layer.model_width + features - 1) / features)
        return Status::Invalid("LayerNorm statistics packing differs");
    // Validate the standard layout directly instead of allocating another T*D
    // collection of individual slot-location vectors on every invocation.
    for (std::size_t logical = 0; logical < packing.logical_locations.size();
         ++logical) {
        const auto &locations = packing.Locations(logical);
        const auto feature = logical % layer.model_width;
        const auto token = logical / layer.model_width;
        if (locations.size() != 1 ||
            locations.front().ciphertext != feature / features ||
            locations.front().slot != (feature % features) * stride + token ||
            locations.front().component != ComplexLane::kReal ||
            locations.front().packed_query_head)
            return Status::Invalid("LayerNorm statistics physical coordinates differ");
    }
    auto level = LevelOf(context, input.ciphertexts.front().parms_id());
    if (!level.ok()) return level.status();
    const auto input_level = level.value();
    const auto terminal_level = context.parameters.coefficient_modulus_bits.size() -
                                context.parameters.terminal_prime_count;
    if (input_level + 3 > terminal_level)
        return Status::FailedPrecondition(
            "LayerNorm statistics needs three rescale levels after input");
    for (const auto &ciphertext : input.ciphertexts) {
        if (ciphertext.parms_id() != input.ciphertexts.front().parms_id())
            return Status::FailedPrecondition(
                "LayerNorm statistics ciphertext levels differ");
        if (!std::isfinite(ciphertext.scale()) || ciphertext.scale() <= 0)
            return Status::Invalid("LayerNorm statistics input scale is invalid");
    }
    return StatisticsGeometry{stride, features, input_level};
}

Result<LayerNormStatisticsOutputs> ComputeLayerNormStatistics(
    const SealCipherTensor &input, const TransformerLayerConfig &layer,
    const HeContext &context, AttentionLn2ExecutionReport *report,
    std::size_t he_workers, std::optional<int> output_scale_bits)
{
    const int output_bits = output_scale_bits.value_or(context.parameters.default_scale_bits);
    if (output_bits != context.parameters.default_scale_bits && output_bits != context.parameters.output_scale_bits)
        return Status::Invalid("statistics output scale is absent from the profile");
    auto geometry = ValidateStatisticsInput(input, layer, context);
    if (!geometry.ok()) return geometry.status();
    const auto stride = geometry.value().stride;
    const auto features = geometry.value().features;
    const auto input_level = geometry.value().input_level;
    const auto count = input.ciphertexts.size();
    const auto worker_bytes = detail::EstimatedHeWorkerBytes(
        context.parameters.polynomial_degree,
        context.context->get_context_data(input.ciphertexts.front().parms_id())
            ->parms().coeff_modulus().size(), 8);
    const auto budget = detail::HeWorkerPool::Instance().MemoryBudget();
    auto workers = std::max<std::size_t>(1, std::min({he_workers, count,
        std::max<std::size_t>(1, budget / worker_bytes)}));
    if (detail::HeWorkerPool::OnWorkerThread()) workers = 1;
    std::size_t encodings = 0;
    const auto plains = PrepareStatisticsPlaintexts(
        input, layer, context, stride, features, encodings, output_bits);
    const double target = plains->target;
    if (report) {
        report->layer_norm_plaintext_encodings += encodings;
        report->layer_norm_workers = std::max(report->layer_norm_workers, workers);
    }
    seal::Evaluator evaluator(*context.context);
    auto reduce_features = [&](seal::Ciphertext sum) {
        seal::Ciphertext rotated;
        for (std::size_t step = 1; step < features; step <<= 1U) {
            evaluator.rotate_vector(sum, CanonicalRotation(
                static_cast<long long>(step * stride), context.slot_count()),
                context.galois_keys, rotated);
            evaluator.add_inplace(sum, rotated);
            if (report) ++report->rotations;
        }
        return sum;
    };
    std::vector<seal::Ciphertext> centered(count);
    std::vector<seal::Ciphertext> mean_parts(workers);
    const auto group_count = plains->variance_masks.size();
    std::vector<std::vector<seal::Ciphertext>> variance_parts(
        workers, std::vector<seal::Ciphertext>(group_count));
    std::vector<AttentionLn2ExecutionReport> local_reports(workers);
    auto prepared = detail::ParallelRanges(workers, workers, 1,
        [&](std::size_t begin, std::size_t end) -> Result<void> {
            seal::Evaluator local(*context.context);
            for (auto worker = begin; worker < end; ++worker) {
                bool initialized = false;
                for (auto i = count * worker / workers;
                     i < count * (worker + 1) / workers; ++i) {
                    local.multiply_plain(input.ciphertexts[i], *plains->valid[i],
                                         centered[i]);
                    seal::Ciphertext partial;
                    local.multiply_plain(input.ciphertexts[i], *plains->mean[i],
                                         partial);
                    auto normalized = NormalizeScale(centered[i], plains->centering_scale);
                    if (!normalized.ok()) return normalized.status();
                    normalized = NormalizeScale(partial, plains->centering_scale);
                    if (!normalized.ok()) return normalized.status();
                    if (!initialized) {
                        mean_parts[worker] = std::move(partial);
                        initialized = true;
                    } else local.add_inplace(mean_parts[worker], partial);
                    local_reports[worker].ciphertext_plain_multiplications += 2;
                }
            }
            return {};
        }, worker_bytes);
    if (!prepared.ok()) return prepared.status();
    auto mean = std::move(mean_parts.front());
    for (std::size_t worker = 1; worker < workers; ++worker)
        evaluator.add_inplace(mean, mean_parts[worker]);
    mean_parts.clear();
    // Rotation and addition are linear: R(sum_i masked_i) = sum_i R(masked_i).
    mean = reduce_features(std::move(mean));
    auto squared = detail::ParallelRanges(workers, workers, 1,
        [&](std::size_t begin, std::size_t end) -> Result<void> {
            seal::Evaluator local(*context.context);
            for (auto worker = begin; worker < end; ++worker) {
                for (auto i = count * worker / workers;
                     i < count * (worker + 1) / workers; ++i) {
                    local.sub_inplace(centered[i], mean);
                    local.rescale_to_next_inplace(centered[i]);
                    auto normalized = NormalizeScale(centered[i], target);
                    if (!normalized.ok()) return normalized.status();
                    // Square each input separately. Group only the subsequent
                    // linear work by identical masks, including a distinct tail.
                    auto square = centered[i];
                    local.multiply_plain_inplace(square, plains->lift);
                    local.square_inplace(square);
                    auto &counts = local_reports[worker];
                    if (!plains->lazy_square_sum) {
                        local.relinearize_inplace(square, context.relinearization_keys);
                        local.rescale_to_next_inplace(square);
                        normalized = NormalizeScale(square, target);
                        if (!normalized.ok()) return normalized.status();
                        local.multiply_plain_inplace(square, *plains->variance_masks[plains->square_groups[i]]);
                        normalized = NormalizeScale(square, plains->variance_scale);
                        if (!normalized.ok()) return normalized.status();
                        ++counts.relinearizations;
                        ++counts.rescales;
                        ++counts.ciphertext_plain_multiplications;
                    }
                    auto &partial = variance_parts[worker][plains->square_groups[i]];
                    if (partial.size() == 0) partial = std::move(square);
                    else local.add_inplace(partial, square);
                    ++counts.ciphertext_plain_multiplications;
                    ++counts.ciphertext_multiplications;
                    ++counts.rescales;
                }
            }
            return {};
        }, worker_bytes);
    if (!squared.ok()) return squared.status();
    seal::Ciphertext variance_sum;
    for (std::size_t group = 0; group < group_count; ++group) {
        seal::Ciphertext sum;
        for (auto &parts : variance_parts) {
            auto &partial = parts[group];
            if (partial.size() == 0) continue;
            if (sum.size() == 0) sum = std::move(partial);
            else evaluator.add_inplace(sum, partial);
            partial = seal::Ciphertext();
        }
        if (plains->lazy_square_sum) {
            evaluator.relinearize_inplace(sum, context.relinearization_keys);
            evaluator.rescale_to_next_inplace(sum);
            auto normalized = NormalizeScale(sum, target);
            if (!normalized.ok()) return normalized.status();
            evaluator.multiply_plain_inplace(sum, *plains->variance_masks[group]);
            normalized = NormalizeScale(sum, plains->variance_scale);
            if (!normalized.ok()) return normalized.status();
            if (report) {
                ++report->relinearizations;
                ++report->rescales;
                ++report->ciphertext_plain_multiplications;
            }
        }
        if (variance_sum.size() == 0) variance_sum = std::move(sum);
        else evaluator.add_inplace(variance_sum, sum);
    }
    variance_parts.clear();
    variance_sum = reduce_features(std::move(variance_sum));
    evaluator.add_plain_inplace(variance_sum, plains->epsilon_plain);
    evaluator.rescale_to_next_inplace(variance_sum);
    auto normalized = NormalizeScale(variance_sum, target);
    if (!normalized.ok()) return normalized.status();
    if (report) {
        ++report->rescales;
        for (const auto &counts : local_reports) {
            report->ciphertext_plain_multiplications += counts.ciphertext_plain_multiplications;
            report->ciphertext_multiplications += counts.ciphertext_multiplications;
            report->relinearizations += counts.relinearizations;
            report->rescales += counts.rescales;
        }
    }

    SealCipherTensor centered_tensor;
    centered_tensor.parameter_profile = context.profile;
    centered_tensor.ciphertexts = std::move(centered);
    centered_tensor.logical_spec.shape = {{layer.tokens, layer.model_width}};
    centered_tensor.logical_spec.domain = Domain::kHeCiphertext;
    centered_tensor.logical_spec.layout = Layout::kStrictColumn;
    centered_tensor.logical_spec.numeric = input.logical_spec.numeric;
    centered_tensor.logical_spec.he_region = input.logical_spec.he_region;
    centered_tensor.logical_spec.he_parameter_profile = context.profile;
    centered_tensor.logical_spec.planned_level = input_level + 1;
    centered_tensor.logical_spec.planned_scale_bits =
        output_bits;
    centered_tensor.physical_packing = input.physical_packing;

    HePackingPlan variance_plan;
    variance_plan.layout = Layout::kReduction;
    variance_plan.logical_shape = {{layer.tokens, 1}};
    variance_plan.slot_count = context.slot_count();
    variance_plan.ciphertext_count = 1;
    variance_plan.logical_locations.resize(layer.tokens);
    for (std::size_t token = 0; token < layer.tokens; ++token)
        variance_plan.logical_locations[token].push_back(
            {0, token, token < (layer.tokens + 1) / 2
                ? ComplexLane::kReal : ComplexLane::kImaginary, std::nullopt});
    SealCipherTensor variance_tensor;
    variance_tensor.parameter_profile = context.profile;
    variance_tensor.ciphertexts.push_back(std::move(variance_sum));
    variance_tensor.logical_spec.shape = {{layer.tokens, 1}};
    variance_tensor.logical_spec.domain = Domain::kHeCiphertext;
    variance_tensor.logical_spec.layout = Layout::kReduction;
    variance_tensor.logical_spec.numeric = input.logical_spec.numeric;
    variance_tensor.logical_spec.he_region = input.logical_spec.he_region;
    variance_tensor.logical_spec.he_parameter_profile = context.profile;
    variance_tensor.logical_spec.planned_level = input_level + 3;
    variance_tensor.logical_spec.planned_scale_bits =
        output_bits;
    variance_tensor.physical_packing = std::move(variance_plan);
    return LayerNormStatisticsOutputs{std::move(centered_tensor),
                                      std::move(variance_tensor)};
}

const PlainParameterHandle *FindPlain(const LayerWeightHandles &weights,
                                      const std::string &name)
{
    for (const auto &parameter : weights.plain_parameters)
        if (parameter.spec.name == name) return &parameter;
    return nullptr;
}

} // namespace

Result<AttentionValuePlan> AttentionLn2PackingPlanner::AttentionValue(
    const TransformerLayerConfig &layer, std::size_t slot_count)
{
    auto layer_valid = ValidateLayerConfig(layer);
    if (!layer_valid.ok()) return layer_valid.status();
    auto tiles_result = MakeTiles(layer, slot_count);
    if (!tiles_result.ok()) return tiles_result.status();
    HePackingPlan probabilities;
    probabilities.layout = Layout::kSoftmaxAvDiagonal;
    probabilities.logical_shape = {{layer.tokens, layer.q_heads,
                                     layer.tokens}};
    probabilities.slot_count = slot_count;
    probabilities.logical_locations.resize(
        layer.tokens * layer.q_heads * layer.tokens);
    std::set<int> rotations;
    for (const auto &tile : tiles_result.value()) {
        const auto block = detail::TokenColumnStride(layer.tokens) * tile.head_count;
        probabilities.ciphertext_count = std::max(
            probabilities.ciphertext_count,
            tile.probability_begin + tile.probability_groups);
        for (std::size_t local_head = 0; local_head < tile.head_count;
             ++local_head) {
            const auto head = tile.head_begin + local_head;
            for (std::size_t row = 0; row < layer.tokens; ++row) {
                for (std::size_t column = 0; column < layer.tokens;
                     ++column) {
                    const auto diagonal =
                        (column + layer.tokens - row) % layer.tokens;
                    const auto group = diagonal / (2 * tile.lanes);
                    const auto lane = (diagonal / 2) % tile.lanes;
                    const auto component = diagonal % 2 == 0
                        ? ComplexLane::kReal : ComplexLane::kImaginary;
                    const auto slot = lane * block +
                                      local_head * detail::TokenColumnStride(layer.tokens) + row;
                    probabilities.logical_locations[ProbabilityIndex(
                        row, head, column, layer)].push_back(
                            {tile.probability_begin + group, slot,
                             component, head});
                }
            }
        }
        const auto baby_steps = std::min(layer.tokens, 2 * tile.lanes);
        const auto giant_steps =
            (layer.tokens + baby_steps - 1) / baby_steps;
        const bool paired = tile.lanes % 2 == 0;
        for (std::size_t giant = 0; giant < giant_steps; ++giant) {
            const auto diagonal_begin = giant * baby_steps;
            if (diagonal_begin != 0) {
                rotations.insert(CanonicalRotation(
                    -static_cast<long long>(diagonal_begin),
                    slot_count));
                rotations.insert(CanonicalRotation(
                    static_cast<long long>((layer.tokens - diagonal_begin)), slot_count));
            }
            const auto valid_babies = std::min(
                baby_steps, layer.tokens - diagonal_begin);
            const auto complex_diagonals = (valid_babies + 1) / 2;
            for (std::size_t first = 0; first < complex_diagonals;
                 first += paired ? 2 : 1)
                for (std::size_t populated = paired ? 2 : 1;
                     populated < tile.lanes; populated *= 2)
                    rotations.insert(CanonicalRotation(
                        ReplicationRotation(first, populated, block),
                        slot_count));
        }
        if (paired) {
            rotations.insert(CanonicalRotation(
                -static_cast<long long>(block), slot_count));
            rotations.insert(CanonicalRotation(
                static_cast<long long>(block), slot_count));
        }

        // Shared V boundary and nonzero baby rotations.
        if (baby_steps > 1)
            rotations.insert(CanonicalRotation(
                -static_cast<long long>(layer.tokens), slot_count));
        for (std::size_t baby = 1; baby < baby_steps; ++baby)
            rotations.insert(CanonicalRotation(
                static_cast<long long>(baby), slot_count));
        for (std::size_t giant = 0; giant < giant_steps; ++giant) {
            const auto shift = giant * baby_steps;
            rotations.insert(CanonicalRotation(
                static_cast<long long>(shift), slot_count));
            if (shift != 0)
                rotations.insert(CanonicalRotation(
                    static_cast<long long>(shift) -
                        static_cast<long long>(layer.tokens), slot_count));
        }
    }
    auto valid = probabilities.Validate();
    if (!valid.ok()) return valid.status();
    auto map = ToConversionMap(probabilities);
    if (!map.ok()) return map.status();
    auto output = AttentionOutputPacking(layer, slot_count,
                                         tiles_result.value());
    valid = output.Validate();
    if (!valid.ok()) return valid.status();
    // SEAL step 0 requests the conjugation key used to realify prepared A.
    rotations.insert(0);
    return AttentionValuePlan{std::move(probabilities), std::move(output),
                              map.take_value(),
                              {rotations.begin(), rotations.end()}};
}

Result<HePackingPlan> AttentionLn2PackingPlanner::ComplexStrictColumns(
    const TensorShape &shape, std::size_t slot_count)
{
    if (shape.dimensions.size() != 2)
        return Status::Invalid("strict-column shape must be [tokens,width]");
    const auto tokens = shape.dimensions[0];
    const auto width = shape.dimensions[1];
    const auto stride = NextPowerOfTwo(tokens);
    if (stride > slot_count)
        return Status::Invalid("one strict-column feature exceeds CKKS slots");
    const auto columns = slot_count / stride;
    HePackingPlan result;
    result.layout = Layout::kComplexStrictColumn;
    result.logical_shape = shape;
    result.slot_count = slot_count;
    result.ciphertext_count = (width + 2 * columns - 1) / (2 * columns);
    result.logical_locations.resize(tokens * width);
    for (std::size_t token = 0; token < tokens; ++token) {
        for (std::size_t feature = 0; feature < width; ++feature) {
            const auto group = feature / (2 * columns);
            const auto local = feature % (2 * columns);
            result.logical_locations[token * width + feature].push_back(
                {group, (local % columns) * stride + token,
                 local < columns ? ComplexLane::kReal
                                 : ComplexLane::kImaginary,
                 std::nullopt});
        }
    }
    auto valid = result.Validate();
    if (!valid.ok()) return valid.status();
    return result;
}

Result<HePackingPlan> AttentionLn2PackingPlanner::RealStrictColumns(
    const TensorShape &shape, std::size_t slot_count)
{
    return MultiHeadPackingPlan::RealStrictColumns(shape, slot_count);
}

Result<void> AttentionLn2LevelPlanner::Validate(
    const HeParameterConfig &parameters)
{
    if (parameters.coefficient_modulus_bits.size() < 10 ||
        parameters.declared_depth < 9)
        return Status::FailedPrecondition(
            "attention/LN2 requires ten data primes and depth nine");
    if (parameters.terminal_prime_count != 1)
        return Status::FailedPrecondition(
            "attention/LN2 conversion requires one terminal prime");
    if (parameters.input_scale_bits != parameters.default_scale_bits ||
        parameters.output_scale_bits != parameters.default_scale_bits)
        return Status::FailedPrecondition(
            "attention/LN2 requires one public CKKS boundary scale");
    return {};
}

Result<void> AttentionLn2LevelPlanner::ValidateOutput(
    const HeParameterConfig &parameters)
{
    if (parameters.coefficient_modulus_bits.size() < 5 ||
        parameters.declared_depth < 4)
        return Status::FailedPrecondition(
            "Region 4 requires five data primes and depth four");
    if (parameters.terminal_prime_count != 1)
        return Status::FailedPrecondition(
            "Region 4 conversion requires one terminal prime");
    if (parameters.input_scale_bits != parameters.default_scale_bits ||
        parameters.output_scale_bits != parameters.default_scale_bits)
        return Status::FailedPrecondition(
            "Region 4 requires one public CKKS boundary scale");
    return {};
}

std::size_t AttentionLn2LevelPlanner::OutputInputLevel(
    const HeParameterConfig &parameters)
{
    return parameters.coefficient_modulus_bits.size() -
        parameters.terminal_prime_count - 4;
}

Result<void> MultiHeadAttentionValueKernel::Prepare(
    const TransformerLayerConfig &layer, const HeContext &context,
    std::size_t input_level, AttentionLn2ExecutionReport *report)
{
    try {
        auto data = context.context->first_context_data();
        for (std::size_t level = 0; data && level < input_level; ++level)
            data = data->next_context_data();
        if (!data) return Status::Invalid("AV preparation input level is absent");
        const double scale = std::ldexp(1., context.parameters.default_scale_bits);
        auto program = GetAvPlaintexts(layer, context, data->parms_id(), scale, scale, report);
        if (!program.ok()) return program.status();
        return {};
    } catch (const std::exception &error) {
        return Status::Internal(std::string("AV preparation failed: ") + error.what());
    }
}

Result<SealCipherTensor> MultiHeadAttentionValueKernel::Evaluate(
    const SealCipherTensor &probabilities, const SealCipherTensor &value,
    const TransformerLayerConfig &layer, const HeContext &context,
    AttentionLn2ExecutionReport *report, std::size_t he_workers)
{
    return ComputeAttentionValue(probabilities, value, layer, context, report,
                                 he_workers);
}

Result<SealCipherTensor> AttentionOutputProjectionKernel::Evaluate(
    const SealCipherTensor &attention, const DenseMatrixParameter &weights,
    const std::optional<DenseVectorParameter> &bias,
    const TransformerLayerConfig &layer, const HeContext &context,
    AttentionLn2ExecutionReport *report, std::size_t he_workers)
{
    return ProjectOutput(attention, weights, bias, layer, context, report,
                         he_workers);
}

Result<SealCipherTensor> AttentionOutputProjectionKernel::EvaluateReal(
    const SealCipherTensor &attention, const DenseMatrixParameter &weights,
    const std::optional<DenseVectorParameter> &bias,
    const TransformerLayerConfig &layer, const HeContext &context,
    AttentionLn2ExecutionReport *report, std::size_t he_workers)
{
    return ProjectRealOutput(attention, weights, bias, layer, context, report, he_workers);
}

Result<SealCipherTensor> AttentionResidualKernel::Add(
    SealCipherTensor projected, const SealCipherTensor &residual,
    const HeContext &context, AttentionLn2ExecutionReport *report)
{
    return AddResidual(std::move(projected), residual, context, report);
}

Result<SealCipherTensor> ComplexRealifyKernel::Evaluate(
    const SealCipherTensor &input, const TransformerLayerConfig &layer,
    const HeContext &context, AttentionLn2ExecutionReport *report)
{
    auto valid = input.Validate(context);
    if (!valid.ok()) return valid.status();
    auto complex_plan = AttentionLn2PackingPlanner::ComplexStrictColumns(
        {{layer.tokens, layer.model_width}}, context.slot_count());
    if (!complex_plan.ok()) return complex_plan.status();
    auto same = SamePacking(input.physical_packing, complex_plan.value());
    if (!same.ok()) return same.status();
    auto real_plan = AttentionLn2PackingPlanner::RealStrictColumns(
        {{layer.tokens, layer.model_width}}, context.slot_count());
    if (!real_plan.ok()) return real_plan.status();
    seal::Evaluator evaluator(*context.context);
    seal::CKKSEncoder encoder(*context.context);
    std::vector<seal::Ciphertext> output(real_plan.value().ciphertext_count);
    const auto stride = NextPowerOfTwo(layer.tokens);
    const auto columns = context.slot_count() / stride;
    for (std::size_t group = 0; group < input.ciphertexts.size(); ++group) {
        const auto &source = input.ciphertexts[group];
        seal::Ciphertext conjugated;
        evaluator.complex_conjugate(source, context.galois_keys, conjugated);
        seal::Ciphertext real;
        evaluator.add(source, conjugated, real);
        const double embedding_scale = std::sqrt(2.0);
        real.scale() *= 2.0;
        const auto real_index = 2 * group;
        if (real_index < output.size()) output[real_index] = std::move(real);
        const auto imaginary_begin = (2 * group + 1) * columns;
        if (imaginary_begin < layer.model_width) {
            seal::Ciphertext imaginary;
            evaluator.sub(source, conjugated, imaginary);
            std::vector<Complex> negative_i(
                context.slot_count(), Complex{0.0, -1.0});
            seal::Plaintext negative_i_plain;
            encoder.encode(negative_i, imaginary.parms_id(), embedding_scale,
                           negative_i_plain);
            evaluator.multiply_plain_inplace(imaginary, negative_i_plain);
            imaginary.scale() *= 2.0;
            output[real_index + 1] = std::move(imaginary);
            if (report) ++report->ciphertext_plain_multiplications;
        }
        if (report) ++report->conjugations;
    }
    SealCipherTensor result;
    result.parameter_profile = context.profile;
    result.ciphertexts = std::move(output);
    result.logical_spec = input.logical_spec;
    result.logical_spec.layout = Layout::kStrictColumn;
    // The exact physical scale includes the sqrt(2) canonical-embedding
    // alignment factor and is therefore intentionally tracked by each
    // ciphertext rather than rounded into the integer graph hint.
    result.logical_spec.planned_scale_bits.reset();
    result.physical_packing = real_plan.take_value();
    return result;
}

Result<LayerNormStatisticsOutputs> LayerNormStatisticsKernel::Evaluate(
    const SealCipherTensor &input, const TransformerLayerConfig &layer,
    const HeContext &context, AttentionLn2ExecutionReport *report,
    std::size_t he_workers, std::optional<int> output_scale_bits)
{
    try {
        return ComputeLayerNormStatistics(input, layer, context, report, he_workers, output_scale_bits);
    } catch (const std::exception &error) {
        return Status::FailedPrecondition(std::string("statistics evaluation failed: ") + error.what());
    }
}

Result<void> LayerNormStatisticsKernel::Prepare(
    const SealCipherTensor &input, const TransformerLayerConfig &layer,
    const HeContext &context, AttentionLn2ExecutionReport *report, std::optional<int> output_scale_bits)
{
    const int output_bits = output_scale_bits.value_or(context.parameters.default_scale_bits);
    if (output_bits != context.parameters.default_scale_bits && output_bits != context.parameters.output_scale_bits)
        return Status::Invalid("statistics output scale is absent from the profile");
    auto geometry = ValidateStatisticsInput(input, layer, context);
    if (!geometry.ok()) return geometry.status();
    try {
        std::size_t encodings = 0;
        PrepareStatisticsPlaintexts(input, layer, context,
            geometry.value().stride, geometry.value().features, encodings,
            output_scale_bits.value_or(context.parameters.default_scale_bits));
        if (report) report->layer_norm_plaintext_encodings += encodings;
        return {};
    } catch (const std::exception &error) {
        return Status::Internal(std::string("statistics preparation failed: ") + error.what());
    }
}

class MpcLayerNormRuntime::Impl {
public:
    explicit Impl(std::shared_ptr<SharedMpcSession> session, std::size_t worker = 0)
        : session_(std::move(session)), native_(session_->Acquire(worker)),
          party_(native_->party())
    {}

    Result<void> ValidateVariance(
        const MpcShareTensor &variance, int configured_bits,
        int output_scale)
    {
        auto valid = variance.Validate();
        if (!valid.ok()) return valid.status();
        if (variance.local_share.empty() || !variance.numeric.is_signed || output_scale < 1 ||
            output_scale > variance.numeric.scale_bits ||
            configured_bits <= output_scale + 1 || configured_bits > variance.numeric.ring_bits ||
            variance.local_share.size() > static_cast<std::size_t>(std::numeric_limits<int>::max() / 2))
            return Status::FailedPrecondition("LayerNorm rsqrt numeric contract is invalid");
        // Keep the configured integer range while variance uses more fractional
        // bits. SCI sqrt supports different input/output scales directly.
        auto planned = detail::PlanLayerNormRsqrt(variance.numeric.ring_bits,
            variance.numeric.scale_bits, configured_bits, output_scale);
        if (!planned.ok()) return planned.status();
        const auto &plan = planned.value();
        const int input_bits = plan.input_bits;
        auto &fix = native_->fix();
        const auto size = static_cast<int>(variance.local_share.size());
        // 0 < v < U iff (v-1) mod 2^ring_bits < U-1. SCI LT only
        // extracts the sign of the difference: zero-extend first so this
        // unsigned comparison cannot wrap into a negative difference.
        FixArray shifted(party_, size, false, variance.numeric.ring_bits, variance.numeric.scale_bits);
        const auto original_mask = RingMask(variance.numeric.ring_bits);
        const auto upper = std::uint64_t{1} << (input_bits - 1);
        for (int i = 0; i < size; ++i)
            shifted.data[i] = (variance.local_share[i] - (party_ == sci::ALICE ? 1 : 0)) & original_mask;
        auto left = fix.extend(shifted, variance.numeric.ring_bits + 1);
        FixArray right(party_, size, false, left.ell, variance.numeric.scale_bits);
        for (int i = 0; i < size; ++i) right.data[i] = party_ == sci::ALICE ? upper - 1 : 0;
        auto invalid = fix.LT(left, right);
        // Invert one XOR share; this SCI fork's NOT toggles both shares.
        if (party_ == sci::ALICE)
            for (int i = 0; i < size; ++i) invalid.data[i] ^= 1;
        while (invalid.size > 1) {
            const int pairs = invalid.size / 2;
            BoolArray a(party_, pairs), b(party_, pairs);
            for (int i = 0; i < pairs; ++i) {
                a.data[i] = invalid.data[2 * i]; b.data[i] = invalid.data[2 * i + 1];
            }
            auto either = fix.bool_op->XOR(a, b), both = fix.bool_op->AND(a, b);
            auto combined = fix.bool_op->XOR(either, both);
            BoolArray next(party_, pairs + (invalid.size & 1));
            std::copy_n(combined.data, pairs, next.data);
            if (invalid.size & 1) next.data[pairs] = invalid.data[invalid.size - 1];
            invalid = std::move(next);
        }
        auto revealed = fix.bool_op->output(sci::ALICE, invalid);
        std::uint8_t any = party_ == sci::ALICE ? revealed.data[0] & 1 : 0;
        if (party_ == sci::ALICE) fix.iopack->io->send_data(&any, sizeof(any));
        else fix.iopack->io->recv_data(&any, sizeof(any));
        if (any) return Status::FailedPrecondition("LayerNorm variance must be positive and fit the rsqrt working range");
        return {};
    }

    // The complete tensor has already passed one aggregate validity check.
    // Shards never reveal their individual validity bits.
    Result<MpcShareTensor> ReciprocalSqrt(
        const MpcShareTensor &variance, int configured_bits, int output_scale)
    {
        auto planned = detail::PlanLayerNormRsqrt(variance.numeric.ring_bits,
            variance.numeric.scale_bits, configured_bits, output_scale);
        if (!planned.ok()) return planned.status();
        const auto &plan = planned.value();
        const int output_bits = plan.output_bits;
        const auto size = static_cast<int>(variance.local_share.size());
        auto &fix = native_->fix();
        // SCI's inverse-sqrt exponent adjustment can overflow its half-width
        // workspace at the smallest quantum when the scale/width is odd.
        // Lift an odd scale exactly, and use an even native width. The public
        // input bound above remains the configured logical range.
        const int lift = plan.lift;
        const int native_bits = plan.native_bits;
        FixArray input(party_, size, true, native_bits, variance.numeric.scale_bits + lift);
        const int source_bits = native_bits - lift;
        const auto mask = RingMask(source_bits);
        if (source_bits <= variance.numeric.ring_bits) {
            for (int i = 0; i < size; ++i) input.data[i] = (variance.local_share[i] & mask) << lift;
        } else {
            FixArray original(party_, size, true, variance.numeric.ring_bits, variance.numeric.scale_bits);
            std::copy(variance.local_share.begin(), variance.local_share.end(), original.data);
            auto extended = fix.extend(original, source_bits);
            for (int i = 0; i < size; ++i) input.data[i] = extended.data[i] << lift;
        }
        FixArray output(party_, size, false, output_bits, output_scale);
        native_->math().math->sqrt(size, input.data, output.data,
            native_bits, output_bits, input.s, output_scale, true);
        auto widened = output_bits == variance.numeric.ring_bits
            ? output : native_->math().fix->extend(output, variance.numeric.ring_bits);
        return MpcShareTensor{
            std::vector<std::uint64_t>(widened.data, widened.data + widened.size),
            variance.physical_shape,
            {variance.numeric.ring_bits, output_scale, variance.numeric.is_signed},
            ComplexComponent::kReal,
            output_bits < variance.numeric.ring_bits - 1
                ? std::optional<int>{output_bits} : std::nullopt};
    }

    std::uint64_t communication() const
    { return native_->communication_bytes(); }
    std::uint64_t rounds() const { return native_->rounds(); }
    void Cancel() noexcept { session_->Cancel(); }

private:
    std::shared_ptr<SharedMpcSession> session_;
    std::shared_ptr<SharedMpcSession::Impl> native_;
    int party_{};
};

MpcLayerNormRuntime::MpcLayerNormRuntime(
    PartyRole role, std::string peer_address, int port)
    : session_(std::make_shared<SharedMpcSession>(
          role, std::move(peer_address), port))
{}

MpcLayerNormRuntime::MpcLayerNormRuntime(
    std::shared_ptr<SharedMpcSession> session)
    : session_(std::move(session))
{
    if (!session_) throw std::invalid_argument("MPC session is null");
}

MpcLayerNormRuntime::~MpcLayerNormRuntime() = default;
MpcLayerNormRuntime::MpcLayerNormRuntime(MpcLayerNormRuntime &&) noexcept = default;
MpcLayerNormRuntime &MpcLayerNormRuntime::operator=(
    MpcLayerNormRuntime &&) noexcept = default;

Result<MpcShareTensor> MpcLayerNormRuntime::ReciprocalSqrt(
    const MpcShareTensor &variance, int working_ring_bits,
    AttentionLn2ExecutionReport *report, std::optional<int> output_scale_bits)
{
    const auto started = Clock::now();
    try {
        auto valid = variance.Validate();
        if (!valid.ok()) return valid.status();
        if (!variance.numeric.is_signed || variance.local_share.empty() ||
            variance.local_share.size() > static_cast<std::size_t>(std::numeric_limits<int>::max() / 2))
            return Status::Invalid("LayerNorm rsqrt requires a nonempty signed tensor of supported size");
        auto planned = detail::PlanLayerNormRsqrt(variance.numeric.ring_bits,
            variance.numeric.scale_bits, working_ring_bits,
            output_scale_bits.value_or(variance.numeric.scale_bits));
        if (!planned.ok()) return planned.status();
        if (!std::atomic_load(&impl_))
            std::atomic_store(&impl_, std::make_shared<Impl>(
                session_));
        const auto before_bytes = session_->communication_bytes();
        const auto before_rounds = session_->rounds();
        const int output_scale = output_scale_bits.value_or(variance.numeric.scale_bits);
        Result<void> validated = Status::Internal("LayerNorm validation did not execute");
        // Reveal just the original global validity bit before starting any
        // shard. This preserves the failure and information-release contract.
        session_->ForEachRange(1, 1, [&](std::size_t, std::size_t, std::size_t) {
            validated = impl_->ValidateVariance(variance, working_ring_bits, output_scale);
        });
        if (!validated.ok()) return validated.status();
        std::vector<std::uint64_t> output(variance.local_share.size());
        session_->ForEachRange(output.size(), 1,
            [&](std::size_t worker, std::size_t begin, std::size_t end) {
                Impl local(session_, worker);
                MpcShareTensor shard{
                    {variance.local_share.begin() + begin, variance.local_share.begin() + end},
                    {{end - begin, 1}}, variance.numeric, variance.component};
                auto result = local.ReciprocalSqrt(shard, working_ring_bits, output_scale);
                if (!result.ok()) throw std::runtime_error(result.status().message());
                std::copy(result.value().local_share.begin(), result.value().local_share.end(),
                    output.begin() + begin);
            });
        if (report) {
            report->variance_scale_bits = variance.numeric.scale_bits;
            report->rsqrt_input_ring_bits = planned.value().native_bits;
            report->rsqrt_output_ring_bits = planned.value().output_bits;
            report->reciprocal_sqrt_workers = std::max(report->reciprocal_sqrt_workers,
                std::min(session_->worker_count(), output.size()));
            report->reciprocal_square_roots += variance.local_share.size();
            report->mpc_communication_bytes +=
                session_->communication_bytes() - before_bytes;
            report->mpc_rounds += session_->rounds() - before_rounds;
            report->mpc_milliseconds += std::chrono::duration<double,
                std::milli>(Clock::now() - started).count();
        }
        return MpcShareTensor{std::move(output), variance.physical_shape,
            {variance.numeric.ring_bits, output_scale, variance.numeric.is_signed},
            ComplexComponent::kReal,
            planned.value().output_bits < variance.numeric.ring_bits - 1
                ? std::optional<int>{planned.value().output_bits} : std::nullopt};
    } catch (const std::exception &error) {
        return Status::Internal(std::string("MPC reciprocal sqrt: ") +
                                error.what());
    }
}

void MpcLayerNormRuntime::ReleaseSession()
{
    std::atomic_store(&impl_, std::shared_ptr<Impl>{});
}

void MpcLayerNormRuntime::Cancel() noexcept
{
    if (session_) session_->Cancel();
}

SealSciAttentionLn2Executor::SealSciAttentionLn2Executor(
    const HeraConfig &config, const TransformerLayerConfig &layer,
    HeContextRegistry &contexts, SealCipherTensorStore &he_tensors,
    SealPlainParameterStore &parameters, CryptoTensorStore &mpc_tensors,
    CkksMpcConverter &converter, MpcLayerNormRuntime &mpc,
    TensorIdAllocator *ids)
    : config_(config), layer_(layer), contexts_(contexts),
      he_tensors_(he_tensors), parameters_(parameters),
      mpc_tensors_(mpc_tensors), converter_(converter), mpc_(mpc), ids_(ids)
{}

Result<std::vector<int>> SealSciAttentionLn2Executor::RequiredRotationSteps(
    const TransformerLayerConfig &layer,
    const HeParameterConfig &parameters)
{
    auto valid = Block2LevelPlanner::Validate(parameters);
    if (!valid.ok()) return valid.status();
    valid = AttentionLn2LevelPlanner::ValidateOutput(parameters);
    if (!valid.ok()) return valid.status();
    auto av = AttentionLn2PackingPlanner::AttentionValue(
        layer, parameters.polynomial_degree / 2);
    if (!av.ok()) return av.status();
    auto output = RequiredOutputRotationSteps(layer, parameters);
    if (!output.ok()) return output.status();
    std::set<int> steps(output.value().begin(), output.value().end());
    steps.insert(av.value().required_rotation_steps.begin(),
                 av.value().required_rotation_steps.end());
    return std::vector<int>(steps.begin(), steps.end());
}

Result<std::vector<int>> SealSciAttentionLn2Executor::RequiredOutputRotationSteps(
    const TransformerLayerConfig &layer,
    const HeParameterConfig &parameters)
{
    auto valid = AttentionLn2LevelPlanner::ValidateOutput(parameters);
    if (!valid.ok()) return valid.status();
    const auto slots = parameters.polynomial_degree / 2;
    auto projection = AttentionOutputProjectionKernel::RequiredRealRotationSteps(
        layer, slots);
    if (!projection.ok()) return projection.status();
    std::set<int> steps(projection.value().begin(), projection.value().end());
    // Statistics actually reduces over the entire strict-column capacity.
    const auto stride = NextPowerOfTwo(layer.tokens);
    const auto features_per_ciphertext = slots / stride;
    for (std::size_t step = 1; step < features_per_ciphertext; step <<= 1U)
        steps.insert(CanonicalRotation(
            static_cast<long long>(step * stride), slots));
    return std::vector<int>(steps.begin(), steps.end());
}

Result<AttentionOutputBsgsPlan> AttentionOutputProjectionKernel::RealColumnBsgs(
    const TransformerLayerConfig &layer, std::size_t slots)
{
    if (layer.tokens == 0 || layer.model_width == 0 || layer.q_heads == 0 ||
        layer.value_head_width == 0 || layer.tokens > slots)
        return Status::Invalid("real Wo BSGS has invalid dimensions");
    auto av = AttentionLn2PackingPlanner::AttentionValue(layer, slots);
    if (!av.ok()) return av.status();
    auto output = AttentionLn2PackingPlanner::RealStrictColumns(
        {{layer.tokens, layer.model_width}}, slots);
    if (!output.ok()) return output.status();
    AttentionOutputBsgsPlan result;
    result.token_stride = detail::TokenColumnStride(layer.tokens);
    if (slots % result.token_stride != 0)
        return Status::Invalid("real Wo BSGS requires complete token columns");
    result.columns_per_ciphertext = slots / result.token_stride;
    result.input_ciphertexts = av.value().output_packing.ciphertext_count;
    result.output_ciphertexts = output.value().ciphertext_count;
    result.rotation_upper_bound = std::numeric_limits<std::size_t>::max();
    double best_score = std::numeric_limits<double>::infinity();
    for (std::size_t baby = 1; baby <= result.columns_per_ciphertext; ++baby) {
        if (result.columns_per_ciphertext % baby) continue;
        const auto giant = result.columns_per_ciphertext / baby;
        const auto cost = result.input_ciphertexts * (baby - 1) +
                          result.output_ciphertexts * (giant - 1);
        const auto score = ProjectionBsgsCost(result.input_ciphertexts, result.output_ciphertexts,
            result.columns_per_ciphertext, baby, slots, true);
        if (score < best_score) {
            best_score = score;
            result.baby_steps = baby;
            result.giant_steps = giant;
            result.rotation_upper_bound = cost;
        }
    }
    if (result.baby_steps == 0) return Status::Invalid("Wo baby override must divide the slot columns");
    return result;
}

Result<std::vector<int>> AttentionOutputProjectionKernel::RequiredRealRotationSteps(
    const TransformerLayerConfig &layer, std::size_t slots)
{
    auto split = RealColumnBsgs(layer, slots);
    if (!split.ok()) return split.status();
    std::set<int> steps{0}; // Output conjugation.
    for (std::size_t i = 1; i < split.value().baby_steps; ++i)
        steps.insert(CanonicalRotation(static_cast<long long>(i * split.value().token_stride), slots));
    for (std::size_t j = 1; j < split.value().giant_steps; ++j)
        steps.insert(CanonicalRotation(static_cast<long long>(
            j * split.value().baby_steps * split.value().token_stride), slots));
    return std::vector<int>(steps.begin(), steps.end());
}

Result<AttentionLn2Outputs> SealSciAttentionLn2Executor::Evaluate(
    const AttentionLn2Inputs &inputs, const LayerWeightHandles &weights,
    AttentionLn2ExecutionReport *report)
{
    const auto started = Clock::now();
    try {
        if (report) *report = {};
        const auto *execution = FindLayerExecution(config_, layer_.id);
        if (!execution) return Status::Invalid("layer has no HE placement");
        const auto region2 = execution->regions.find(HeRegionId::kRegion2);
        const auto region4 = execution->regions.find(HeRegionId::kRegion4);
        if (region2 == execution->regions.end() ||
            region4 == execution->regions.end())
            return Status::Invalid("attention/LN2 regions are absent");
        if (region2->second.parameter_profile != region4->second.parameter_profile)
            return Status::Invalid("Region 4 must use the same parameter profile as Region 2");
        const auto &context = contexts_.Require(
            region2->second.parameter_profile, contexts_.role());
        const auto &output_context = contexts_.Require(
            region4->second.parameter_profile, contexts_.role());
        auto level_valid = Block2LevelPlanner::Validate(context.parameters);
        if (!level_valid.ok()) return level_valid.status();
        level_valid = AttentionLn2LevelPlanner::ValidateOutput(
            output_context.parameters);
        if (!level_valid.ok()) return level_valid.status();
        if (contexts_.role() == PartyRole::kServer &&
            (context.galois_keys.size() == 0 ||
             output_context.galois_keys.size() == 0))
            return Status::FailedPrecondition(
                "attention/LN2 Galois keys are not compiled");
        const auto wo_input_level =
            AttentionLn2LevelPlanner::OutputInputLevel(output_context.parameters);
        const auto av_input_level = detail::AttentionValueInputLevel(context.parameters);

        const TensorShape probability_shape{{layer_.tokens, layer_.q_heads,
                                             layer_.tokens}};
        if (!inputs.probability_shares.valid() ||
            inputs.probability_shares.spec.domain !=
                Domain::kMpcArithmeticShare ||
            inputs.probability_shares.spec.shape != probability_shape ||
            inputs.probability_shares.spec.layout != Layout::kSoftmaxRows)
            return Status::Invalid("attention/LN2 probability contract mismatch");
        auto probability = mpc_tensors_.GetMpcShare(
            inputs.probability_shares.id);
        if (!probability.ok()) return probability.status();
        if (probability.value().numeric.is_signed)
            return Status::Invalid("Softmax probabilities must be unsigned");

        auto av_plan = AttentionLn2PackingPlanner::AttentionValue(
            layer_, context.slot_count());
        if (!av_plan.ok()) return av_plan.status();
        // Allocate this public handle on both parties before entering the
        // server-only HE branch. Only the server stores the ciphertext
        // payload; both parties retain the same graph-level identity.
        const auto residual_id = ids_ ? ids_->Next()
                                      : next_id_.fetch_add(1);
        TensorSpec residual_spec;
        residual_spec.shape = {{layer_.tokens, layer_.model_width}};
        residual_spec.domain = Domain::kHeCiphertext;
        residual_spec.layout = Layout::kStrictColumn;
        residual_spec.numeric = config_.mpc_numeric;
        residual_spec.he_region = HeRegionId::kRegion4;
        residual_spec.he_parameter_profile = output_context.profile;
        residual_spec.planned_level = wo_input_level + 1;
        residual_spec.planned_scale_bits =
            output_context.parameters.default_scale_bits;
        auto probability_pair = ScatterReal(
            probability.value(), av_plan.value().probability_slot_map);
        if (!probability_pair.ok()) return probability_pair.status();

        const auto before_conversion = converter_.stats();
        auto probability_he = converter_.MpcToCkksPacked(
            probability_pair.value(),
            {context.profile, av_input_level,
             context.parameters.default_scale_bits,
             probability_shape, Layout::kSoftmaxAvDiagonal,
             av_plan.value().probability_slot_map, false,
             probability.value().public_value_bound_bits});
        if (!probability_he.ok()) return probability_he.status();
        auto after_conversion = converter_.stats();
        if (report) {
            ++report->mpc_to_ckks_calls;
            report->conversion_communication_bytes +=
                after_conversion.communication_bytes -
                before_conversion.communication_bytes;
            report->conversion_rounds += after_conversion.rounds -
                                         before_conversion.rounds;
            report->conversion_milliseconds += after_conversion.milliseconds -
                                               before_conversion.milliseconds;
        }

        std::optional<SealCipherTensor> attention_tensor;
        if (contexts_.role() == PartyRole::kServer) {
            if (!probability_he.value())
                return Status::Internal(
                    "server did not receive Softmax CKKS ciphertexts");
            auto value = he_tensors_.Get(inputs.value.id);
            if (!value.ok()) return value.status();
            auto av_id = detail::LevelId(context, av_input_level);
            if (!av_id.ok()) return av_id.status();
            seal::Evaluator evaluator(*context.context);
            for (auto &ct : value.value().ciphertexts) {
                auto level = LevelOf(context, ct.parms_id());
                if (!level.ok()) return level.status();
                if (level.value() > av_input_level)
                    return Status::FailedPrecondition("V is below the AV suffix input level");
                if (ct.parms_id() != av_id.value()) {
                    evaluator.mod_switch_to_inplace(ct, av_id.value());
                    if (report) ++report->modulus_switches;
                }
            }
            value.value().logical_spec.planned_level = av_input_level;
            SealCipherTensor a;
            a.parameter_profile = context.profile;
            a.ciphertexts = probability_he.value()->ciphertexts;
            a.logical_spec.shape = probability_shape;
            a.logical_spec.domain = Domain::kHeCiphertext;
            a.logical_spec.layout = Layout::kSoftmaxAvDiagonal;
            a.logical_spec.numeric = probability.value().numeric;
            a.logical_spec.he_region = HeRegionId::kRegion2;
            a.logical_spec.he_parameter_profile = context.profile;
            a.logical_spec.planned_level = av_input_level;
            a.logical_spec.planned_scale_bits =
                context.parameters.default_scale_bits;
            a.physical_packing = av_plan.value().probability_packing;
            auto attention = MultiHeadAttentionValueKernel::Evaluate(
                a, value.value(), layer_, context, report, config_.threads);
            if (!attention.ok()) return attention.status();
            attention_tensor = attention.take_value();
        }

        // A deep shared chain carries AV directly into the Wo/LN suffix.
        // Compact chains require a refresh because AV is already terminal.
        if (!detail::FusedAttentionOutput(context.parameters)) {
            auto source_map = ToConversionMap(av_plan.value().output_packing);
            if (!source_map.ok()) return source_map.status();
            std::optional<SealCipherBatch> batch;
            if (contexts_.role() == PartyRole::kServer)
                batch = SealCipherBatch{context.profile,
                    attention_tensor->ciphertexts,
                    av_plan.value().output_packing.logical_shape,
                    Layout::kAttentionOutput, source_map.value()};
            auto before = converter_.stats();
            auto physical = converter_.CkksToMpc(batch);
            if (!physical.ok()) return physical.status();
            auto logical = Compact(physical.value(), source_map.value(),
                av_plan.value().output_packing.logical_shape, true);
            if (!logical.ok()) return logical.status();
            auto destination = AttentionLn2PackingPlanner::AttentionValue(
                layer_, output_context.slot_count());
            if (!destination.ok()) return destination.status();
            auto destination_map = ToConversionMap(
                destination.value().output_packing);
            if (!destination_map.ok()) return destination_map.status();
            auto pair = ScatterReal(logical.value(), destination_map.value());
            if (!pair.ok()) return pair.status();
            auto refreshed = converter_.MpcToCkksPacked(pair.value(),
                {output_context.profile, wo_input_level,
                 output_context.parameters.input_scale_bits,
                 logical.value().physical_shape, Layout::kAttentionOutput,
                 destination_map.value()});
            if (!refreshed.ok()) return refreshed.status();
            const auto after = converter_.stats();
            if (report) {
                ++report->ckks_to_mpc_calls;
                ++report->mpc_to_ckks_calls;
                report->conversion_communication_bytes +=
                    after.communication_bytes - before.communication_bytes;
                report->conversion_rounds += after.rounds - before.rounds;
                report->conversion_milliseconds +=
                    after.milliseconds - before.milliseconds;
            }
            if (contexts_.role() == PartyRole::kServer) {
                if (!refreshed.value())
                    return Status::Internal("server did not receive Region-4 refresh");
                auto spec = attention_tensor->logical_spec;
                spec.he_region = HeRegionId::kRegion4;
                spec.he_parameter_profile = output_context.profile;
                spec.planned_level = wo_input_level;
                spec.planned_scale_bits = output_context.parameters.input_scale_bits;
                attention_tensor = SealCipherTensor{output_context.profile,
                    refreshed.value()->ciphertexts, spec,
                    destination.value().output_packing};
            }
        }

        std::optional<LayerNormStatisticsOutputs> statistics;
        if (contexts_.role() == PartyRole::kServer) {
            std::shared_ptr<const SealCipherTensor> residual;
            if (layer_.attention_residual) {
                auto loaded = he_tensors_.GetShared(inputs.attention_residual.id);
                if (!loaded.ok()) return loaded.status();
                residual = loaded.take_value();
            }

            if (weights.layer_id != layer_.id || !weights.wo.valid() ||
                weights.wo.spec.rows != AttentionWidth(layer_) ||
                weights.wo.spec.columns != layer_.model_width)
                return Status::Invalid("Wo handle has the wrong shape");
            auto wo = parameters_.GetMatrixShared(weights.wo.id);
            if (!wo.ok()) return wo.status();
            std::optional<DenseVectorParameter> bo;
            if (layer_.output_bias) {
                const auto *handle = FindPlain(weights, "bo");
                if (!handle || handle->spec.length != layer_.model_width)
                    return Status::Invalid("bo handle has the wrong shape");
                auto value_bo = parameters_.GetVector(handle->id);
                if (!value_bo.ok()) return value_bo.status();
                bo = value_bo.take_value();
            }
            auto projected = AttentionOutputProjectionKernel::EvaluateReal(
                *attention_tensor, *wo.value(), bo, layer_, output_context,
                report, config_.threads);
            if (!projected.ok()) return projected.status();
            if (layer_.attention_residual) {
                auto added = AttentionResidualKernel::Add(
                    projected.take_value(), *residual, output_context,
                    report);
                if (!added.ok()) return added.status();
                projected = added.take_value();
            }
            auto computed = LayerNormStatisticsKernel::Evaluate(
                projected.value(), layer_, output_context, report, config_.threads);
            if (!computed.ok()) return computed.status();
            statistics = computed.take_value();
            // Figure-10 mode gets the FFN skip from the complete LN2 tail,
            // not this pre-normalization hidden value.
            if (layer_.norm != NormPlacement::kNormalizedResidual && layer_.ffn_residual) {
                auto stored_residual = he_tensors_.Put(residual_id, projected.take_value());
                if (!stored_residual.ok()) return stored_residual.status();
            }
        }

        auto convert_tensor = [&](bool centered_tensor)
            -> Result<MpcShareTensor> {
            std::optional<SealCipherBatch> batch;
            PhysicalSlotMap map;
            TensorShape shape = centered_tensor
                ? TensorShape{{layer_.tokens, layer_.model_width}}
                : TensorShape{{layer_.tokens, 1}};
            Layout layout = centered_tensor ? Layout::kStrictColumn
                                            : Layout::kReduction;
            if (contexts_.role() == PartyRole::kServer) {
                auto &tensor = centered_tensor ? statistics->centered
                                               : statistics->variance;
                auto converted_map = ToConversionMap(centered_tensor
                    ? detail::PairedRealPacking(tensor.physical_packing)
                    : tensor.physical_packing);
                if (!converted_map.ok()) return converted_map.status();
                map = converted_map.take_value();
                const auto pair_started = Clock::now();
                auto ciphertexts = centered_tensor
                    ? detail::PairRealCiphertexts(tensor.ciphertexts, output_context, config_.threads)
                    : Result<std::vector<seal::Ciphertext>>{tensor.ciphertexts};
                if (!ciphertexts.ok()) return ciphertexts.status();
                if (report && centered_tensor) {
                    report->ciphertext_plain_multiplications += tensor.ciphertexts.size() / 2;
                    report->he_milliseconds += std::chrono::duration<double, std::milli>(
                        Clock::now() - pair_started).count();
                }
                batch = SealCipherBatch{output_context.profile, ciphertexts.take_value(),
                                        shape, layout, map};
            }
            auto before = converter_.stats();
            auto physical = converter_.CkksToMpc(batch, centered_tensor ? config_.mpc_numeric.scale_bits :
                std::max(config_.mpc_numeric.scale_bits, layer_.attention_norm_variance_scale_bits));
            if (!physical.ok()) return physical.status();
            auto after = converter_.stats();
            if (report) {
                ++report->ckks_to_mpc_calls;
                report->conversion_communication_bytes +=
                    after.communication_bytes - before.communication_bytes;
                report->conversion_rounds += after.rounds - before.rounds;
                report->conversion_milliseconds +=
                    after.milliseconds - before.milliseconds;
            }
            if (contexts_.role() == PartyRole::kClient) {
                // Rebuild the public map deterministically; no ciphertext or
                // secret value is involved.
                if (centered_tensor) {
                    auto plan = AttentionLn2PackingPlanner::RealStrictColumns(
                        shape, output_context.slot_count());
                    if (!plan.ok()) return plan.status();
                    auto converted_map = ToConversionMap(detail::PairedRealPacking(plan.value()));
                    if (!converted_map.ok()) return converted_map.status();
                    map = converted_map.take_value();
                } else {
                    HePackingPlan plan;
                    plan.layout = Layout::kReduction;
                    plan.logical_shape = shape;
                    plan.slot_count = output_context.slot_count();
                    plan.ciphertext_count = 1;
                    plan.logical_locations.resize(layer_.tokens);
                    for (std::size_t token = 0; token < layer_.tokens; ++token)
                        plan.logical_locations[token].push_back(
                            {0, token, token < (layer_.tokens + 1) / 2
                                ? ComplexLane::kReal : ComplexLane::kImaginary, std::nullopt});
                    auto converted_map = ToConversionMap(plan);
                    if (!converted_map.ok()) return converted_map.status();
                    map = converted_map.take_value();
                }
            }
            return Compact(physical.value(), map, shape, true, centered_tensor);
        };

        auto centered = convert_tensor(true);
        if (!centered.ok()) return centered.status();
        auto variance = convert_tensor(false);
        if (!variance.ok()) return variance.status();
        auto inverse = mpc_.ReciprocalSqrt(
            variance.value(), layer_.attention_norm_rsqrt_bits, report, config_.mpc_numeric.scale_bits);
        if (!inverse.ok()) return inverse.status();
        // Reuse the LayerNorm Ferret session across layers. Conversion runs
        // on independent transports and does not require resetting OT state.

        TensorSpec centered_spec;
        centered_spec.shape = centered.value().physical_shape;
        centered_spec.domain = Domain::kMpcArithmeticShare;
        centered_spec.layout = Layout::kStrictColumn;
        centered_spec.numeric = centered.value().numeric;
        TensorSpec inverse_spec;
        inverse_spec.shape = inverse.value().physical_shape;
        inverse_spec.domain = Domain::kMpcArithmeticShare;
        inverse_spec.layout = Layout::kReduction;
        inverse_spec.numeric = inverse.value().numeric;
        const auto centered_id = ids_ ? ids_->Next()
                                      : next_id_.fetch_add(1);
        const auto inverse_id = ids_ ? ids_->Next()
                                     : next_id_.fetch_add(1);
        auto stored = mpc_tensors_.PutMpcShare(
            centered_id, centered.take_value());
        if (!stored.ok()) return stored.status();
        stored = mpc_tensors_.PutMpcShare(
            inverse_id, inverse.take_value());
        if (!stored.ok()) return stored.status();
        if (report)
            report->he_milliseconds += std::chrono::duration<double,
                std::milli>(Clock::now() - started).count() -
                report->conversion_milliseconds - report->mpc_milliseconds;
        return AttentionLn2Outputs{
            TensorHandle{centered_id, centered_spec},
            TensorHandle{inverse_id, inverse_spec},
            layer_.norm == NormPlacement::kNormalizedResidual || !layer_.ffn_residual
                ? TensorHandle{} : TensorHandle{residual_id, residual_spec}};
    } catch (const std::exception &error) {
        return Status::Internal(std::string("attention/LN2 evaluation: ") +
                                error.what());
    }
}

} // namespace hera
