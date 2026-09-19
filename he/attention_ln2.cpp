#include "hera/attention_ln2.h"

#include "OT/split-kkot.h"
#include "FloatingPoint/fp-math.h"
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
#include <complex>
#include <cstring>
#include <limits>
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

Result<std::size_t> LevelOf(const HeContext &context,
                            seal::parms_id_type parms_id)
{
    std::size_t level = 0;
    for (auto data = context.context->first_context_data(); data;
         data = data->next_context_data(), ++level)
        if (data->parms_id() == parms_id) return level;
    return Status::Invalid("ciphertext is outside the shared attention context");
}

double DroppedPrime(const HeContext &context,
                    seal::parms_id_type parms_id)
{
    const auto data = context.context->get_context_data(parms_id);
    if (!data || !data->next_context_data())
        throw std::invalid_argument("operation needs another CKKS data level");
    return static_cast<double>(data->parms().coeff_modulus().back().value());
}

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
    int remaining = CanonicalRotation(rotation, context.slot_count());
    if (remaining == 0) return result;
    const int sign = remaining < 0 ? -1 : 1;
    auto magnitude = static_cast<std::uint64_t>(
        remaining < 0 ? -static_cast<long long>(remaining) : remaining);
    std::uint64_t bit = 1;
    while (magnitude != 0) {
        if (magnitude & 1U) {
            evaluator.rotate_vector_inplace(
                result, sign * static_cast<int>(bit), context.galois_keys);
            if (report) ++report->rotations;
        }
        magnitude >>= 1U;
        bit <<= 1U;
    }
    return result;
}

bool AnyNonzero(const std::vector<double> &mask)
{
    return std::any_of(mask.begin(), mask.end(),
                       [](double value) { return value != 0.0; });
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
    const auto max_heads = slots / layer.tokens;
    if (max_heads == 0)
        return Status::Invalid("no attention head fits in CKKS slots");
    std::vector<HeadTile> result;
    std::size_t probability_begin = 0;
    std::size_t value_begin = 0;
    for (std::size_t begin = 0; begin < layer.q_heads;) {
        const auto heads = std::min(max_heads, layer.q_heads - begin);
        const auto block = layer.tokens * heads;
        const auto lanes = slots / block;
        const auto probability_groups =
            (layer.tokens + lanes - 1) / lanes;
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

Result<PhysicalSlotMap> ToConversionMap(const HePackingPlan &packing)
{
    auto valid = packing.Validate();
    if (!valid.ok()) return valid.status();
    PhysicalSlotMap result;
    result.ciphertext_count = packing.ciphertext_count;
    result.slots_per_ciphertext = packing.slot_count;
    result.logical_element_count = packing.logical_locations.size();
    result.assignments.resize(result.ciphertext_count * result.slots_per_ciphertext);
    for (std::size_t logical = 0; logical < packing.logical_locations.size();
         ++logical) {
        const auto &locations = packing.Locations(logical);
        if (locations.size() != 1)
            return Status::Invalid(
                "conversion packing requires one physical lane per logical value");
        const auto &location = locations.front();
        auto &entry = result.assignments[
            location.ciphertext * packing.slot_count + location.slot];
        auto &target = location.component == ComplexLane::kReal
            ? entry.real_logical_index : entry.imaginary_logical_index;
        if (target != kUnusedLogicalIndex)
            return Status::Invalid("conversion packing lane is not unique");
        target = logical;
    }
    auto map_valid = result.Validate();
    if (!map_valid.ok()) return map_valid.status();
    return result;
}

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
                        logical.numeric, ComplexComponent::kReal};
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
    const TensorShape &shape, bool signed_values)
{
    auto valid = physical.Validate();
    if (!valid.ok()) return valid.status();
    std::vector<std::uint64_t> logical(map.logical_element_count);
    for (std::size_t i = 0; i < map.assignments.size(); ++i) {
        const auto &entry = map.assignments[i];
        if (entry.real_logical_index != kUnusedLogicalIndex)
            logical[entry.real_logical_index] = physical.real.local_share[i];
        if (entry.imaginary_logical_index != kUnusedLogicalIndex)
            logical[entry.imaginary_logical_index] = physical.imag.local_share[i];
    }
    auto numeric = physical.real.numeric;
    numeric.is_signed = signed_values;
    return MpcShareTensor{std::move(logical), shape, numeric,
                          ComplexComponent::kReal};
}

Result<void> SamePacking(const HePackingPlan &left,
                         const HePackingPlan &right)
{
    if (left.layout != right.layout ||
        left.logical_shape != right.logical_shape ||
        left.slot_count != right.slot_count ||
        left.ciphertext_count != right.ciphertext_count ||
        left.logical_locations.size() != right.logical_locations.size())
        return Status::Invalid("HE packing plans differ");
    for (std::size_t i = 0; i < left.logical_locations.size(); ++i) {
        const auto &a = left.logical_locations[i];
        const auto &b = right.logical_locations[i];
        if (a.size() != b.size()) return Status::Invalid("HE replicas differ");
        for (std::size_t j = 0; j < a.size(); ++j)
            if (a[j].ciphertext != b[j].ciphertext ||
                a[j].slot != b[j].slot ||
                a[j].component != b[j].component ||
                a[j].packed_query_head != b[j].packed_query_head)
                return Status::Invalid("HE physical coordinates differ");
    }
    return {};
}

seal::Plaintext EncodeMask(const std::vector<double> &mask,
                           seal::parms_id_type parms_id, double scale,
                           seal::CKKSEncoder &encoder)
{
    seal::Plaintext result;
    encoder.encode(mask, parms_id, scale, result);
    return result;
}

seal::Ciphertext MaskRescale(
    const seal::Ciphertext &input, const std::vector<double> &mask,
    double output_scale, const HeContext &context,
    seal::Evaluator &evaluator, seal::CKKSEncoder &encoder,
    AttentionLn2ExecutionReport *report)
{
    const double plain_scale = output_scale *
        DroppedPrime(context, input.parms_id()) / input.scale();
    auto plain = EncodeMask(mask, input.parms_id(), plain_scale, encoder);
    seal::Ciphertext result;
    evaluator.multiply_plain(input, plain, result);
    evaluator.rescale_to_next_inplace(result);
    auto normalized = NormalizeScale(result, output_scale);
    if (!normalized.ok()) throw std::invalid_argument(normalized.status().message());
    if (report) {
        ++report->ciphertext_plain_multiplications;
        ++report->rescales;
    }
    return result;
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
        const auto block = layer.tokens * tile.head_count;
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
                                      token * tile.head_count + local_head;
                    result.logical_locations[AttentionOutputIndex(
                        token, head, dimension, layer)].push_back(
                            {tile.value_begin + group, slot, component, head});
                }
            }
        }
    }
    return result;
}

Result<SealCipherTensor> ComputeAttentionValue(
    const SealCipherTensor &probabilities, const SealCipherTensor &value,
    const TransformerLayerConfig &layer, const HeContext &context,
    AttentionLn2ExecutionReport *report)
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
    if (probability_level.value() != 2 || value_level.value() != 2)
        return Status::FailedPrecondition(
            "A*V inputs must share the exact level-2 suffix");
    for (const auto &ct : probabilities.ciphertexts)
        if (ct.parms_id() != value.ciphertexts.front().parms_id())
            return Status::FailedPrecondition(
                "Softmax and V do not have the same parms_id");
    const double target = std::ldexp(
        1.0, context.parameters.default_scale_bits);
    if (!CloseScale(probabilities.ciphertexts.front().scale(), target) ||
        !CloseScale(value.ciphertexts.front().scale(), target))
        return Status::FailedPrecondition("A*V input scale mismatch");

    auto tiles_result = MakeTiles(layer, context.slot_count());
    if (!tiles_result.ok()) return tiles_result.status();
    const auto &tiles = tiles_result.value();
    seal::Evaluator evaluator(*context.context);
    seal::CKKSEncoder encoder(*context.context);
    std::vector<seal::Ciphertext> output(
        plan_result.value().output_packing.ciphertext_count);

    for (const auto &tile : tiles) {
        const auto block = layer.tokens * tile.head_count;
        // This is the dynamic counterpart of softmax_vh.cpp's complex-pair
        // protocol. One lane contains one complete [token, head] block. V uses
        // the real and imaginary components for two independent groups of
        // value columns, while A remains real-only.
        const auto baby_steps = std::min(layer.tokens, tile.lanes);
        const auto giant_steps =
            (layer.tokens + baby_steps - 1) / baby_steps;
        if (giant_steps != tile.probability_groups)
            return Status::Internal(
                "A*V BSGS groups disagree with the probability packing");

        const auto level3 = context.context->get_context_data(
            probabilities.ciphertexts[tile.probability_begin].parms_id())
                                ->next_context_data();
        if (!level3 || !level3->next_context_data())
            return Status::FailedPrecondition("A*V chain is too short");
        // A is deliberately prepared at the prime that the ct-ct product will
        // drop. Therefore A(level 3) * V(level 3) rescales exactly back to the
        // public target scale at level 4 without rewriting scale metadata.
        const double prepared_a_scale = DroppedPrime(
            context, level3->parms_id());

        // Step 1: preprocess A. For giant g and baby b, construct a ciphertext
        // replicated over all value lanes whose row p contains
        // A[p-gB, p+b]. The giant row displacement is recovered after the
        // baby products have been accumulated.
        std::vector<seal::Ciphertext> prepared_a(layer.tokens);
        for (std::size_t giant = 0; giant < giant_steps; ++giant) {
            const auto diagonal_begin = giant * baby_steps;
            const auto valid_babies = std::min(
                baby_steps, layer.tokens - diagonal_begin);
            const auto &source = probabilities.ciphertexts[
                tile.probability_begin + giant];

            if (tile.lanes > layer.tokens) {
                // Small test/model shapes can leave hundreds of physical lanes
                // unused. Isolate one diagonal and broadcast it by doubling;
                // this is equivalent to Step 1 but avoids O(lanes) rotations.
                for (std::size_t baby = 0; baby < valid_babies; ++baby) {
                    auto aligned = Rotate(
                        source, static_cast<int>(baby * block), context,
                        evaluator, report);
                    std::vector<double> first_lane(context.slot_count(), 0.0);
                    for (std::size_t row = 0; row < layer.tokens; ++row)
                        for (std::size_t head = 0;
                             head < tile.head_count; ++head)
                            first_lane[row * tile.head_count + head] = 1.0;
                    auto base = MaskRescale(
                        aligned, first_lane, prepared_a_scale, context,
                        evaluator, encoder, report);
                    auto replicated = base;
                    std::size_t populated = 1;
                    while (2 * populated <= tile.lanes) {
                        auto copy = Rotate(
                            replicated,
                            -static_cast<int>(populated * block), context,
                            evaluator, report);
                        evaluator.add_inplace(replicated, copy);
                        populated *= 2;
                    }
                    for (std::size_t lane = populated;
                         lane < tile.lanes; ++lane) {
                        auto copy = Rotate(
                            base, -static_cast<int>(lane * block), context,
                            evaluator, report);
                        evaluator.add_inplace(replicated, copy);
                    }
                    prepared_a[diagonal_begin + baby] =
                        std::move(replicated);
                }
                continue;
            }

            // BLB/softmax_vh Step 1: rotations are shared by all baby
            // diagonals in this giant group. Masks select the appropriate
            // rotated variant for each output lane.
            std::map<int, seal::Ciphertext> variants;
            const double plain_scale = prepared_a_scale *
                DroppedPrime(context, source.parms_id()) / source.scale();
            for (std::size_t baby = 0; baby < valid_babies; ++baby) {
                std::map<int, std::vector<double>> masks;
                for (std::size_t lane = 0; lane < tile.lanes; ++lane) {
                    for (std::size_t row = 0; row < layer.tokens; ++row) {
                        const auto source_row =
                            (row + layer.tokens - diagonal_begin) %
                            layer.tokens;
                        for (std::size_t head = 0;
                             head < tile.head_count; ++head) {
                            const auto destination = lane * block +
                                row * tile.head_count + head;
                            const auto origin = baby * block +
                                source_row * tile.head_count + head;
                            const int rotation = CanonicalRotation(
                                static_cast<long long>(origin) -
                                    static_cast<long long>(destination),
                                context.slot_count());
                            auto inserted = masks.emplace(
                                rotation, std::vector<double>(
                                    context.slot_count(), 0.0));
                            inserted.first->second[destination] = 1.0;
                        }
                    }
                }
                seal::Ciphertext accumulated;
                bool initialized = false;
                for (const auto &entry : masks) {
                    auto variant = variants.find(entry.first);
                    if (variant == variants.end())
                        variant = variants.emplace(
                            entry.first, Rotate(source, entry.first, context,
                                                evaluator, report)).first;
                    auto mask = EncodeMask(entry.second, source.parms_id(),
                                           plain_scale, encoder);
                    seal::Ciphertext term;
                    evaluator.multiply_plain(variant->second, mask, term);
                    if (!initialized) {
                        accumulated = std::move(term);
                        initialized = true;
                    } else {
                        evaluator.add_inplace(accumulated, term);
                    }
                    if (report)
                        ++report->ciphertext_plain_multiplications;
                }
                if (!initialized)
                    return Status::Internal(
                        "A*V Step 1 generated no diagonal terms");
                evaluator.rescale_to_next_inplace(accumulated);
                auto normalized = NormalizeScale(accumulated,
                                                  prepared_a_scale);
                if (!normalized.ok()) return normalized.status();
                if (report) ++report->rescales;
                prepared_a[diagonal_begin + baby] =
                    std::move(accumulated);
            }
        }

        for (std::size_t value_group = 0;
             value_group < tile.value_groups; ++value_group) {
            const auto &v_source = value.ciphertexts[
                tile.value_begin + value_group];

            // Step 2: preprocess V once per baby step. These row-inner
            // rotations are reused by every giant group. Since V is complex,
            // each operation simultaneously moves one real and one imaginary
            // group of value columns.
            std::vector<seal::Ciphertext> prepared_v(baby_steps);
            for (std::size_t baby = 0; baby < baby_steps; ++baby) {
                std::vector<double> nonwrap(context.slot_count(), 0.0);
                std::vector<double> wrap(context.slot_count(), 0.0);
                for (std::size_t lane = 0; lane < tile.lanes; ++lane) {
                    for (std::size_t row = 0; row < layer.tokens; ++row) {
                        auto &mask = row + baby < layer.tokens
                            ? nonwrap : wrap;
                        for (std::size_t head = 0;
                             head < tile.head_count; ++head)
                            mask[lane * block +
                                 row * tile.head_count + head] = 1.0;
                    }
                }
                const int shift = static_cast<int>(
                    baby * tile.head_count);
                auto main = Rotate(v_source, shift, context, evaluator,
                                   report);
                const double plain_scale = target *
                    DroppedPrime(context, v_source.parms_id()) /
                    v_source.scale();
                auto main_mask = EncodeMask(
                    nonwrap, v_source.parms_id(), plain_scale, encoder);
                evaluator.multiply_plain_inplace(main, main_mask);
                std::size_t multiplications = 1;
                if (AnyNonzero(wrap)) {
                    auto wrapped = Rotate(
                        v_source, shift - static_cast<int>(block), context,
                        evaluator, report);
                    auto wrap_mask = EncodeMask(
                        wrap, v_source.parms_id(), plain_scale, encoder);
                    evaluator.multiply_plain_inplace(wrapped, wrap_mask);
                    evaluator.add_inplace(main, wrapped);
                    ++multiplications;
                }
                evaluator.rescale_to_next_inplace(main);
                auto normalized = NormalizeScale(main, target);
                if (!normalized.ok()) return normalized.status();
                if (report) {
                    report->ciphertext_plain_multiplications +=
                        multiplications;
                    ++report->rescales;
                }
                prepared_v[baby] = std::move(main);
            }

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

            std::optional<seal::Ciphertext> accumulated;
            for (std::size_t giant = 0; giant < giant_steps; ++giant) {
                const auto row_shift = giant * baby_steps;
                std::vector<double> nonwrap(context.slot_count(), 0.0);
                std::vector<double> wrap(context.slot_count(), 0.0);
                for (std::size_t lane = 0; lane < tile.lanes; ++lane) {
                    for (std::size_t row = 0; row < layer.tokens; ++row) {
                        auto &mask = row + row_shift < layer.tokens
                            ? nonwrap : wrap;
                        for (std::size_t head = 0;
                             head < tile.head_count; ++head)
                            mask[lane * block +
                                 row * tile.head_count + head] = 1.0;
                    }
                }
                const int shift = static_cast<int>(
                    row_shift * tile.head_count);
                auto recovered = Rotate(
                    partials[giant], shift, context, evaluator, report);
                const double plain_scale = target *
                    DroppedPrime(context, partials[giant].parms_id()) /
                    partials[giant].scale();
                auto main_mask = EncodeMask(
                    nonwrap, partials[giant].parms_id(), plain_scale,
                    encoder);
                evaluator.multiply_plain_inplace(recovered, main_mask);
                std::size_t multiplications = 1;
                if (AnyNonzero(wrap)) {
                    auto wrapped = Rotate(
                        partials[giant], shift - static_cast<int>(block),
                        context, evaluator, report);
                    auto wrap_mask = EncodeMask(
                        wrap, partials[giant].parms_id(), plain_scale,
                        encoder);
                    evaluator.multiply_plain_inplace(wrapped, wrap_mask);
                    evaluator.add_inplace(recovered, wrapped);
                    ++multiplications;
                }
                evaluator.rescale_to_next_inplace(recovered);
                auto normalized = NormalizeScale(recovered, target);
                if (!normalized.ok()) return normalized.status();
                if (report) {
                    report->ciphertext_plain_multiplications +=
                        multiplications;
                    ++report->rescales;
                }
                if (!accumulated)
                    accumulated = std::move(recovered);
                else
                    evaluator.add_inplace(*accumulated, recovered);
            }
            if (!accumulated)
                return Status::Internal("A*V Step 3 produced no output");
            output[tile.value_begin + value_group] =
                std::move(*accumulated);
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
    result.logical_spec.planned_level = 5;
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

Result<SealCipherTensor> ProjectOutput(
    const SealCipherTensor &input, const DenseMatrixParameter &weights,
    const std::optional<DenseVectorParameter> &bias,
    const TransformerLayerConfig &layer, const HeContext &context,
    AttentionLn2ExecutionReport *report)
{
    auto valid = input.Validate(context);
    if (!valid.ok()) return valid.status();
    const auto input_width = layer.q_heads * layer.value_head_width;
    if (weights.spec.rows != input_width ||
        weights.spec.columns != layer.model_width ||
        weights.row_major.size() != input_width * layer.model_width)
        return Status::Invalid("Wo payload has the wrong rectangular shape");
    if (bias && bias->values.size() != layer.model_width)
        return Status::Invalid("bo payload has the wrong size");
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
                    auto inserted = masks.emplace(
                        key, std::vector<Complex>(context.slot_count()));
                    inserted.first->second[destination.slot] +=
                        weights.row_major[input_column * layer.model_width +
                                          output_feature] *
                        Extraction(source.component) *
                        Destination(destination.component);
                }
            }
        }
    }

    seal::Evaluator evaluator(*context.context);
    seal::CKKSEncoder encoder(*context.context);
    std::vector<seal::Ciphertext> output(
        target_plan.value().ciphertext_count);
    std::vector<bool> initialized(output.size(), false);
    std::map<std::pair<std::size_t, ComplexLane>, seal::Ciphertext> extracted;
    const double target = std::ldexp(
        1.0, context.parameters.default_scale_bits);
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
            evaluator.complex_conjugate(source, context.galois_keys,
                                        conjugated);
            seal::Ciphertext component;
            if (entry.first.component == ComplexLane::kReal)
                evaluator.add(source, conjugated, component);
            else
                evaluator.sub(source, conjugated, component);
            base = extracted.emplace(base_key, std::move(component)).first;
            if (report) ++report->conjugations;
        }
        auto rotated = Rotate(base->second, entry.first.rotation, context,
                              evaluator, report);
        const double plain_scale = target *
            DroppedPrime(context, rotated.parms_id()) / rotated.scale();
        auto encoded = EncodeModelPlaintext(
            encoder, entry.second, rotated.parms_id(), plain_scale,
            "attention.wo.diagonal");
        if (!encoded.ok()) return encoded.status();
        auto plain = encoded.take_value();
        evaluator.multiply_plain_inplace(rotated, plain);
        if (report) ++report->ciphertext_plain_multiplications;
        if (!initialized[entry.first.target_ct]) {
            output[entry.first.target_ct] = std::move(rotated);
            initialized[entry.first.target_ct] = true;
        } else {
            evaluator.add_inplace(output[entry.first.target_ct], rotated);
        }
    }
    for (std::size_t ct = 0; ct < output.size(); ++ct) {
        if (!initialized[ct])
            return Status::Internal("Wo generated an empty output group");
        evaluator.rescale_to_next_inplace(output[ct]);
        auto normalized = NormalizeScale(output[ct], target);
        if (!normalized.ok()) return normalized.status();
        if (report) ++report->rescales;
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
            auto encoded = EncodeModelPlaintext(
                encoder, slots, output[ct].parms_id(), target,
                "attention.wo.bias");
            if (!encoded.ok()) return encoded.status();
            auto plain = encoded.take_value();
            evaluator.add_plain_inplace(output[ct], plain);
        }
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
    result.logical_spec.planned_level = 6;
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
        "attention residual must use complex strict-column packing");
    if (projected.ciphertexts.size() != residual.ciphertexts.size())
        return Status::Invalid("attention residual ciphertext count mismatch");
    seal::Evaluator evaluator(*context.context);
    for (std::size_t i = 0; i < projected.ciphertexts.size(); ++i) {
        auto aligned = residual.ciphertexts[i];
        if (!CloseScale(aligned.scale(), projected.ciphertexts[i].scale()))
            return Status::FailedPrecondition(
                "attention residual scale does not match Wo output");
        auto residual_level = LevelOf(context, aligned.parms_id());
        auto projected_level = LevelOf(
            context, projected.ciphertexts[i].parms_id());
        if (!residual_level.ok()) return residual_level.status();
        if (!projected_level.ok()) return projected_level.status();
        if (residual_level.value() > projected_level.value())
            return Status::FailedPrecondition(
                "attention residual is below the Wo output level");
        if (aligned.parms_id() != projected.ciphertexts[i].parms_id()) {
            evaluator.mod_switch_to_inplace(
                aligned, projected.ciphertexts[i].parms_id());
            if (report) ++report->modulus_switches;
        }
        evaluator.add_inplace(projected.ciphertexts[i], aligned);
    }
    return projected;
}

Result<LayerNormStatisticsOutputs> ComputeLayerNormStatistics(
    const SealCipherTensor &input, const TransformerLayerConfig &layer,
    const HeContext &context, AttentionLn2ExecutionReport *report)
{
    auto valid = input.Validate(context);
    if (!valid.ok()) return valid.status();
    auto expected = AttentionLn2PackingPlanner::RealStrictColumns(
        {{layer.tokens, layer.model_width}}, context.slot_count());
    if (!expected.ok()) return expected.status();
    auto same = SamePacking(input.physical_packing, expected.value());
    if (!same.ok()) return same.status();
    auto level = LevelOf(context, input.ciphertexts.front().parms_id());
    if (!level.ok()) return level.status();
    const auto input_level = level.value();
    const auto terminal_level =
        context.parameters.coefficient_modulus_bits.size() -
        context.parameters.terminal_prime_count;
    if (input_level + 3 > terminal_level)
        return Status::FailedPrecondition(
            "LayerNorm statistics needs three rescale levels after input");
    for (const auto &ciphertext : input.ciphertexts) {
        auto current = LevelOf(context, ciphertext.parms_id());
        if (!current.ok()) return current.status();
        if (current.value() != input_level)
            return Status::FailedPrecondition(
                "LayerNorm statistics ciphertext levels differ");
    }
    const auto stride = NextPowerOfTwo(layer.tokens);
    const auto features_per_ciphertext = context.slot_count() / stride;
    if (features_per_ciphertext == 0 ||
        (features_per_ciphertext & (features_per_ciphertext - 1)) != 0)
        return Status::FailedPrecondition(
            "strict-column reduction requires a power-of-two feature block");

    seal::Evaluator evaluator(*context.context);
    seal::CKKSEncoder encoder(*context.context);
    const double target = std::ldexp(
        1.0, context.parameters.default_scale_bits);
    auto reduce_features = [&](const seal::Ciphertext &source) {
        auto sum = source;
        for (std::size_t step = 1; step < features_per_ciphertext;
             step <<= 1U) {
            auto rotated = Rotate(sum, static_cast<int>(step * stride),
                                  context, evaluator, report);
            evaluator.add_inplace(sum, rotated);
        }
        return sum;
    };
    std::vector<seal::Ciphertext> centered(input.ciphertexts.size());
    std::vector<seal::Ciphertext> scaled_inputs(input.ciphertexts.size());
    std::optional<seal::Ciphertext> mean;
    for (std::size_t i = 0; i < input.ciphertexts.size(); ++i) {
        const double centering_plain_scale = target *
            DroppedPrime(context, input.ciphertexts[i].parms_id()) /
            input.ciphertexts[i].scale();
        std::vector<double> valid(context.slot_count(), 0.0);
        std::vector<double> mean_mask(context.slot_count(), 0.0);
        for (std::size_t logical = 0;
             logical < input.physical_packing.logical_locations.size();
             ++logical) {
            const auto &location =
                input.physical_packing.Locations(logical).front();
            if (location.ciphertext != i) continue;
            valid[location.slot] = 1.0;
            mean_mask[location.slot] =
                1.0 / static_cast<double>(layer.model_width);
        }
        auto valid_plain = EncodeMask(valid, input.ciphertexts[i].parms_id(),
                                      centering_plain_scale, encoder);
        evaluator.multiply_plain(input.ciphertexts[i], valid_plain,
                                 scaled_inputs[i]);
        auto mean_plain = EncodeMask(
            mean_mask, input.ciphertexts[i].parms_id(),
            centering_plain_scale, encoder);
        seal::Ciphertext mean_part;
        evaluator.multiply_plain(input.ciphertexts[i], mean_plain,
                                 mean_part);
        mean_part = reduce_features(mean_part);
        if (!mean) mean = std::move(mean_part);
        else evaluator.add_inplace(*mean, mean_part);
        if (report) report->ciphertext_plain_multiplications += 2;
    }
    if (!mean) return Status::Internal("LayerNorm mean has no ciphertext");
    for (std::size_t i = 0; i < input.ciphertexts.size(); ++i) {
        centered[i] = std::move(scaled_inputs[i]);
        evaluator.sub_inplace(centered[i], *mean);
        evaluator.rescale_to_next_inplace(centered[i]);
        auto normalized = NormalizeScale(centered[i], target);
        if (!normalized.ok()) return normalized.status();
        if (report) ++report->rescales;
    }

    std::vector<seal::Ciphertext> squares(centered.size());
    for (std::size_t i = 0; i < centered.size(); ++i) {
        const double square_input_scale = std::sqrt(
            target * DroppedPrime(context, centered[i].parms_id()));
        const double lift_scale = square_input_scale / centered[i].scale();
        seal::Plaintext lift;
        encoder.encode(1.0, centered[i].parms_id(), lift_scale, lift);
        squares[i] = centered[i];
        evaluator.multiply_plain_inplace(squares[i], lift);
        evaluator.square_inplace(squares[i]);
        evaluator.relinearize_inplace(
            squares[i], context.relinearization_keys);
        evaluator.rescale_to_next_inplace(squares[i]);
        auto normalized = NormalizeScale(squares[i], target);
        if (!normalized.ok()) return normalized.status();
        if (report) {
            ++report->ciphertext_plain_multiplications;
            ++report->ciphertext_multiplications;
            ++report->relinearizations;
            ++report->rescales;
        }
    }
    const double variance_plain_scale = target *
        DroppedPrime(context, squares.front().parms_id()) /
        squares.front().scale();
    std::optional<seal::Ciphertext> variance_sum;
    for (std::size_t i = 0; i < squares.size(); ++i) {
        std::vector<double> variance_mask(context.slot_count(), 0.0);
        for (std::size_t logical = 0;
             logical < input.physical_packing.logical_locations.size();
             ++logical) {
            const auto &location =
                input.physical_packing.Locations(logical).front();
            if (location.ciphertext == i)
                variance_mask[location.slot] =
                    1.0 / static_cast<double>(layer.model_width);
        }
        auto variance_plain = EncodeMask(
            variance_mask, squares[i].parms_id(), variance_plain_scale,
            encoder);
        evaluator.multiply_plain_inplace(squares[i], variance_plain);
        auto partial = reduce_features(squares[i]);
        if (!variance_sum) variance_sum = std::move(partial);
        else evaluator.add_inplace(*variance_sum, partial);
        if (report) ++report->ciphertext_plain_multiplications;
    }
    if (!variance_sum)
        return Status::Internal("LayerNorm variance has no ciphertext");
    std::vector<double> epsilon(context.slot_count(), 0.0);
    for (std::size_t token = 0; token < layer.tokens; ++token)
        epsilon[token] = layer.attention_norm_epsilon;
    seal::Plaintext epsilon_plain;
    encoder.encode(epsilon, variance_sum->parms_id(),
                   variance_sum->scale(), epsilon_plain);
    evaluator.add_plain_inplace(*variance_sum, epsilon_plain);
    evaluator.rescale_to_next_inplace(*variance_sum);
    auto normalized = NormalizeScale(*variance_sum, target);
    if (!normalized.ok()) return normalized.status();
    if (report) {
        ++report->rescales;
    }

    auto centered_plan = expected.take_value();
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
        context.parameters.default_scale_bits;
    centered_tensor.physical_packing = std::move(centered_plan);

    HePackingPlan variance_plan;
    variance_plan.layout = Layout::kReduction;
    variance_plan.logical_shape = {{layer.tokens, 1}};
    variance_plan.slot_count = context.slot_count();
    variance_plan.ciphertext_count = 1;
    variance_plan.logical_locations.resize(layer.tokens);
    for (std::size_t token = 0; token < layer.tokens; ++token)
        variance_plan.logical_locations[token].push_back(
            {0, token, ComplexLane::kReal, std::nullopt});
    SealCipherTensor variance_tensor;
    variance_tensor.parameter_profile = context.profile;
    variance_tensor.ciphertexts = {std::move(*variance_sum)};
    variance_tensor.logical_spec.shape = {{layer.tokens, 1}};
    variance_tensor.logical_spec.domain = Domain::kHeCiphertext;
    variance_tensor.logical_spec.layout = Layout::kReduction;
    variance_tensor.logical_spec.numeric = input.logical_spec.numeric;
    variance_tensor.logical_spec.he_region = input.logical_spec.he_region;
    variance_tensor.logical_spec.he_parameter_profile = context.profile;
    variance_tensor.logical_spec.planned_level = input_level + 3;
    variance_tensor.logical_spec.planned_scale_bits =
        context.parameters.default_scale_bits;
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
        const auto block = layer.tokens * tile.head_count;
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
                    const auto group = diagonal / tile.lanes;
                    const auto lane = diagonal % tile.lanes;
                    const auto slot = lane * block +
                                      row * tile.head_count + local_head;
                    probabilities.logical_locations[ProbabilityIndex(
                        row, head, column, layer)].push_back(
                            {tile.probability_begin + group, slot,
                             ComplexLane::kReal, head});
                }
            }
        }
        for (std::size_t lane = 1; lane < tile.lanes; ++lane)
            rotations.insert(CanonicalRotation(
                static_cast<long long>(lane * block), slot_count));
        for (std::size_t diagonal = 1; diagonal < layer.tokens; ++diagonal) {
            rotations.insert(CanonicalRotation(
                static_cast<long long>(diagonal * tile.head_count), slot_count));
            rotations.insert(CanonicalRotation(
                static_cast<long long>(diagonal * tile.head_count) -
                    static_cast<long long>(block), slot_count));
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
    rotations.erase(0);
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
    if (shape.dimensions.size() != 2)
        return Status::Invalid("strict-column shape must be [tokens,width]");
    const auto tokens = shape.dimensions[0];
    const auto width = shape.dimensions[1];
    const auto stride = NextPowerOfTwo(tokens);
    if (stride > slot_count)
        return Status::Invalid("one strict-column feature exceeds CKKS slots");
    const auto columns = slot_count / stride;
    HePackingPlan result;
    result.layout = Layout::kStrictColumn;
    result.logical_shape = shape;
    result.slot_count = slot_count;
    result.ciphertext_count = (width + columns - 1) / columns;
    result.logical_locations.resize(tokens * width);
    for (std::size_t token = 0; token < tokens; ++token)
        for (std::size_t feature = 0; feature < width; ++feature)
            result.logical_locations[token * width + feature].push_back(
                {feature / columns, (feature % columns) * stride + token,
                 ComplexLane::kReal, std::nullopt});
    auto valid = result.Validate();
    if (!valid.ok()) return valid.status();
    return result;
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

Result<SealCipherTensor> MultiHeadAttentionValueKernel::Evaluate(
    const SealCipherTensor &probabilities, const SealCipherTensor &value,
    const TransformerLayerConfig &layer, const HeContext &context,
    AttentionLn2ExecutionReport *report)
{
    return ComputeAttentionValue(probabilities, value, layer, context, report);
}

Result<SealCipherTensor> AttentionOutputProjectionKernel::Evaluate(
    const SealCipherTensor &attention, const DenseMatrixParameter &weights,
    const std::optional<DenseVectorParameter> &bias,
    const TransformerLayerConfig &layer, const HeContext &context,
    AttentionLn2ExecutionReport *report)
{
    return ProjectOutput(attention, weights, bias, layer, context, report);
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
    const HeContext &context, AttentionLn2ExecutionReport *report)
{
    return ComputeLayerNormStatistics(input, layer, context, report);
}

class MpcLayerNormRuntime::Impl {
public:
    Impl(PartyRole role, const std::string &peer_address, int port)
        : party_(role == PartyRole::kServer ? sci::ALICE : sci::BOB),
          io_(std::make_unique<sci::IOPack>(
              party_, port,
              role == PartyRole::kServer ? std::string{} : peer_address)),
          ot_(std::make_unique<sci::OTPack>(io_.get(), party_))
    {
        math_ = std::make_unique<FPMath>(party_, io_.get(), ot_.get());
    }

    Result<MpcShareTensor> ReciprocalSqrt(
        const MpcShareTensor &variance, int working_ring_bits)
    {
        auto valid = variance.Validate();
        if (!valid.ok()) return valid.status();
        if (!variance.numeric.is_signed)
            return Status::Invalid("LayerNorm variance must use signed shares");
        if (variance.local_share.size() >
            static_cast<std::size_t>(std::numeric_limits<int>::max()))
            return Status::Invalid("LayerNorm variance tensor is too large");
        // The layer configuration states a public bound for the non-negative
        // variance through its working ring width. Reducing both additive
        // shares modulo that ring is exact when the bound holds and avoids an
        // unnecessarily wide MSNZB circuit inside BLB's rsqrt.
        if (working_ring_bits <= variance.numeric.scale_bits + 1 ||
            working_ring_bits > variance.numeric.ring_bits)
            return Status::FailedPrecondition(
                "LayerNorm rsqrt working ring must be greater than "
                "scale_bits+1 and no greater than the tensor ring width");
        FixArray input(party_, static_cast<int>(variance.local_share.size()),
                       true, working_ring_bits,
                       variance.numeric.scale_bits);
        const auto mask = RingMask(working_ring_bits);
        for (std::size_t i = 0; i < variance.local_share.size(); ++i)
            input.data[i] = variance.local_share[i] & mask;
        auto output = math_->sqrt(input, true);
        // rsqrt is positive. SCI's zero extension converts the working-ring
        // additive shares back to the public HERA tensor contract without
        // revealing the carry.
        output.signed_ = false;
        auto widened = working_ring_bits == variance.numeric.ring_bits
            ? output
            : math_->fix->extend(output, variance.numeric.ring_bits);
        return MpcShareTensor{
            std::vector<std::uint64_t>(widened.data,
                                       widened.data + widened.size),
            variance.physical_shape, variance.numeric,
            ComplexComponent::kReal};
    }

    std::uint64_t communication() const { return io_->get_comm(); }
    std::uint64_t rounds() const { return io_->get_rounds(); }
    void Cancel() noexcept { if (io_) io_->cancel(); }

private:
    int party_{};
    std::unique_ptr<sci::IOPack> io_;
    std::unique_ptr<sci::OTPack> ot_;
    std::unique_ptr<FPMath> math_;
};

MpcLayerNormRuntime::MpcLayerNormRuntime(
    PartyRole role, std::string peer_address, int port)
    : role_(role), peer_address_(std::move(peer_address)), port_(port)
{}

MpcLayerNormRuntime::~MpcLayerNormRuntime() = default;
MpcLayerNormRuntime::MpcLayerNormRuntime(MpcLayerNormRuntime &&) noexcept = default;
MpcLayerNormRuntime &MpcLayerNormRuntime::operator=(
    MpcLayerNormRuntime &&) noexcept = default;

Result<MpcShareTensor> MpcLayerNormRuntime::ReciprocalSqrt(
    const MpcShareTensor &variance, int working_ring_bits,
    AttentionLn2ExecutionReport *report)
{
    const auto started = Clock::now();
    try {
        if (!std::atomic_load(&impl_))
            std::atomic_store(&impl_, std::make_shared<Impl>(
                role_, peer_address_, port_));
        const auto before_bytes = impl_->communication();
        const auto before_rounds = impl_->rounds();
        auto result = impl_->ReciprocalSqrt(variance, working_ring_bits);
        if (!result.ok()) return result.status();
        if (report) {
            report->reciprocal_square_roots += variance.local_share.size();
            report->mpc_communication_bytes +=
                impl_->communication() - before_bytes;
            report->mpc_rounds += impl_->rounds() - before_rounds;
            report->mpc_milliseconds += std::chrono::duration<double,
                std::milli>(Clock::now() - started).count();
        }
        return result;
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
    auto active = std::atomic_load(&impl_);
    if (active) active->Cancel();
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
    auto valid = AttentionLn2LevelPlanner::Validate(parameters);
    if (!valid.ok()) return valid.status();
    std::set<int> steps{0};
    const auto slots = parameters.polynomial_degree / 2;
    // Composite rotations use a minimal signed binary key basis.
    for (std::size_t power = 1; power <= slots / 2; power <<= 1U) {
        steps.insert(static_cast<int>(power));
        if (power != slots / 2) steps.insert(-static_cast<int>(power));
        if (power > slots / 4) break;
    }
    auto av = AttentionLn2PackingPlanner::AttentionValue(layer, slots);
    if (!av.ok()) return av.status();
    steps.insert(av.value().required_rotation_steps.begin(),
                 av.value().required_rotation_steps.end());
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
        if (region2->second.parameter_profile !=
            region4->second.parameter_profile)
            return Status::FailedPrecondition(
                "Region 2 and Region 4 must share one HE profile");
        const auto &context = contexts_.Require(
            region2->second.parameter_profile, contexts_.role());
        auto level_valid = AttentionLn2LevelPlanner::Validate(
            context.parameters);
        if (!level_valid.ok()) return level_valid.status();
        if (context.galois_keys.size() == 0)
            return Status::FailedPrecondition(
                "attention/LN2 Galois keys are not compiled");

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
        residual_spec.layout = Layout::kComplexStrictColumn;
        residual_spec.numeric = config_.mpc_numeric;
        residual_spec.he_region = HeRegionId::kRegion4;
        residual_spec.he_parameter_profile = context.profile;
        residual_spec.planned_level = 6;
        residual_spec.planned_scale_bits =
            context.parameters.default_scale_bits;
        auto probability_pair = ScatterReal(
            probability.value(), av_plan.value().probability_slot_map);
        if (!probability_pair.ok()) return probability_pair.status();

        const auto before_conversion = converter_.stats();
        auto probability_he = converter_.MpcToCkks(
            probability_pair.value(),
            {context.profile, std::size_t{2},
             context.parameters.default_scale_bits,
             probability_shape, Layout::kSoftmaxAvDiagonal,
             av_plan.value().probability_slot_map});
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

        std::optional<LayerNormStatisticsOutputs> statistics;
        if (contexts_.role() == PartyRole::kServer) {
            if (!probability_he.value())
                return Status::Internal(
                    "server did not receive Softmax CKKS ciphertexts");
            auto value = he_tensors_.Get(inputs.value.id);
            if (!value.ok()) return value.status();
            auto residual = he_tensors_.Get(inputs.attention_residual.id);
            if (!residual.ok()) return residual.status();
            SealCipherTensor a;
            a.parameter_profile = context.profile;
            a.ciphertexts = probability_he.value()->ciphertexts;
            a.logical_spec.shape = probability_shape;
            a.logical_spec.domain = Domain::kHeCiphertext;
            a.logical_spec.layout = Layout::kSoftmaxAvDiagonal;
            a.logical_spec.numeric = probability.value().numeric;
            a.logical_spec.he_region = HeRegionId::kRegion2;
            a.logical_spec.he_parameter_profile = context.profile;
            a.logical_spec.planned_level = 2;
            a.logical_spec.planned_scale_bits =
                context.parameters.default_scale_bits;
            a.physical_packing = av_plan.value().probability_packing;
            auto attention = MultiHeadAttentionValueKernel::Evaluate(
                a, value.value(), layer_, context, report);
            if (!attention.ok()) return attention.status();

            if (weights.layer_id != layer_.id || !weights.wo.valid() ||
                weights.wo.spec.rows != AttentionWidth(layer_) ||
                weights.wo.spec.columns != layer_.model_width)
                return Status::Invalid("Wo handle has the wrong shape");
            auto wo = parameters_.GetMatrix(weights.wo.id);
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
            auto projected = AttentionOutputProjectionKernel::Evaluate(
                attention.value(), wo.value(), bo, layer_, context, report);
            if (!projected.ok()) return projected.status();
            if (layer_.attention_residual) {
                auto added = AttentionResidualKernel::Add(
                    projected.take_value(), residual.value(), context,
                    report);
                if (!added.ok()) return added.status();
                projected = added.take_value();
            }
            auto stored_residual = he_tensors_.Put(
                residual_id, projected.value());
            if (!stored_residual.ok()) return stored_residual.status();
            auto real = ComplexRealifyKernel::Evaluate(
                projected.value(), layer_, context, report);
            if (!real.ok()) return real.status();
            auto computed = LayerNormStatisticsKernel::Evaluate(
                real.value(), layer_, context, report);
            if (!computed.ok()) return computed.status();
            statistics = computed.take_value();
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
                auto converted_map = ToConversionMap(tensor.physical_packing);
                if (!converted_map.ok()) return converted_map.status();
                map = converted_map.take_value();
                batch = SealCipherBatch{context.profile, tensor.ciphertexts,
                                        shape, layout, map};
            }
            auto before = converter_.stats();
            auto physical = converter_.CkksToMpc(batch);
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
                        shape, context.slot_count());
                    if (!plan.ok()) return plan.status();
                    auto converted_map = ToConversionMap(plan.value());
                    if (!converted_map.ok()) return converted_map.status();
                    map = converted_map.take_value();
                } else {
                    HePackingPlan plan;
                    plan.layout = Layout::kReduction;
                    plan.logical_shape = shape;
                    plan.slot_count = context.slot_count();
                    plan.ciphertext_count = 1;
                    plan.logical_locations.resize(layer_.tokens);
                    for (std::size_t token = 0; token < layer_.tokens; ++token)
                        plan.logical_locations[token].push_back(
                            {0, token, ComplexLane::kReal, std::nullopt});
                    auto converted_map = ToConversionMap(plan);
                    if (!converted_map.ok()) return converted_map.status();
                    map = converted_map.take_value();
                }
            }
            return Compact(physical.value(), map, shape, true);
        };

        auto centered = convert_tensor(true);
        if (!centered.ok()) return centered.status();
        auto variance = convert_tensor(false);
        if (!variance.ok()) return variance.status();
        auto inverse = mpc_.ReciprocalSqrt(
            variance.value(), layer_.attention_norm_rsqrt_bits, report);
        if (!inverse.ok()) return inverse.status();
        mpc_.ReleaseSession();

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
            TensorHandle{residual_id, residual_spec}};
    } catch (const std::exception &error) {
        return Status::Internal(std::string("attention/LN2 evaluation: ") +
                                error.what());
    }
}

} // namespace hera
