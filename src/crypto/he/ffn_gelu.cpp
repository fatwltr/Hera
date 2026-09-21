#include "hera/ffn_gelu.h"
#include "hera/detail/mpc_session_internal.h"

#include <FloatingPoint/fixed-point.h>
#include <FloatingPoint/fp-math.h>
#include <OT/emp-ot.h>
#include <utils/emp-tool.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>

namespace hera {
namespace {

using Clock = std::chrono::steady_clock;
using Complex = std::complex<double>;

std::size_t NextPowerOfTwo(std::size_t value)
{
    std::size_t result = 1;
    while (result < value) {
        if (result > std::numeric_limits<std::size_t>::max() / 2)
            throw std::overflow_error("token stride overflows size_t");
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
                            seal::parms_id_type id)
{
    std::size_t level = 0;
    for (auto data = context.context->first_context_data(); data;
         data = data->next_context_data(), ++level)
        if (data->parms_id() == id) return level;
    return Status::Invalid("ciphertext parms_id is outside the HE profile");
}

Result<seal::parms_id_type> LevelId(const HeContext &context,
                                    std::size_t target)
{
    auto data = context.context->first_context_data();
    for (std::size_t level = 0; level < target && data; ++level)
        data = data->next_context_data();
    if (!data) return Status::FailedPrecondition("HE level is absent");
    return data->parms_id();
}

double DroppedPrime(const HeContext &context, seal::parms_id_type id)
{
    const auto data = context.context->get_context_data(id);
    if (!data || !data->next_context_data())
        throw std::invalid_argument("cannot rescale the terminal level");
    const auto &moduli = data->parms().coeff_modulus();
    return static_cast<double>(moduli.back().value());
}

bool CloseScale(double left, double right)
{
    if (!std::isfinite(left) || !std::isfinite(right) ||
        left <= 0.0 || right <= 0.0)
        return false;
    return std::abs(left - right) / std::max(left, right) < 1.0e-5;
}

Result<void> NormalizeScale(seal::Ciphertext &value, double target)
{
    if (!CloseScale(value.scale(), target))
        return Status::FailedPrecondition(
            "actual CKKS scale does not match the planned scale: actual=" +
            std::to_string(std::log2(value.scale())) + " bits target=" +
            std::to_string(std::log2(target)) + " bits");
    value.scale() = target;
    return {};
}

const PlainParameterHandle *FindPlain(const LayerWeightHandles &weights,
                                      const std::string &name)
{
    for (const auto &parameter : weights.plain_parameters)
        if (parameter.spec.name == name) return &parameter;
    return nullptr;
}

Result<void> SamePacking(const HePackingPlan &left,
                         const HePackingPlan &right)
{
    if (left.logical_shape != right.logical_shape ||
        left.slot_count != right.slot_count ||
        left.ciphertext_count != right.ciphertext_count ||
        left.logical_locations.size() != right.logical_locations.size())
        return Status::Invalid("ciphertext packing plans differ");
    for (std::size_t i = 0; i < left.logical_locations.size(); ++i) {
        const auto &a = left.logical_locations[i];
        const auto &b = right.logical_locations[i];
        if (a.size() != b.size())
            return Status::Invalid("ciphertext packing replicas differ");
        for (std::size_t j = 0; j < a.size(); ++j)
            if (a[j].ciphertext != b[j].ciphertext ||
                a[j].slot != b[j].slot ||
                a[j].component != b[j].component)
                return Status::Invalid("ciphertext coordinates differ");
    }
    return {};
}

Result<PhysicalSlotMap> ToConversionMap(const HePackingPlan &packing)
{
    PhysicalSlotMap map;
    map.ciphertext_count = packing.ciphertext_count;
    map.slots_per_ciphertext = packing.slot_count;
    map.logical_element_count = packing.logical_locations.size();
    map.assignments.resize(map.ciphertext_count * map.slots_per_ciphertext);
    for (std::size_t logical = 0;
         logical < packing.logical_locations.size(); ++logical) {
        const auto &locations = packing.logical_locations[logical];
        if (locations.size() != 1)
            return Status::Invalid(
                "conversion packing requires one lane per logical element");
        const auto &location = locations.front();
        auto &assignment = map.assignments[
            location.ciphertext * map.slots_per_ciphertext + location.slot];
        auto &index = location.component == ComplexLane::kReal
            ? assignment.real_logical_index
            : assignment.imaginary_logical_index;
        if (index != kUnusedLogicalIndex)
            return Status::Invalid("conversion packing lane collision");
        index = logical;
    }
    auto valid = map.Validate();
    if (!valid.ok()) return valid.status();
    return map;
}

Result<ComplexMpcSharePair> Scatter(
    const MpcShareTensor &logical, const HePackingPlan &packing)
{
    auto valid = logical.Validate();
    if (!valid.ok()) return valid.status();
    if (logical.local_share.size() != packing.logical_locations.size())
        return Status::Invalid("MPC tensor and packing size differ");
    const auto physical_count = packing.ciphertext_count * packing.slot_count;
    std::vector<std::uint64_t> real(physical_count, 0), imag(physical_count, 0);
    for (std::size_t i = 0; i < logical.local_share.size(); ++i) {
        const auto &location = packing.Locations(i).front();
        auto &plane = location.component == ComplexLane::kReal ? real : imag;
        plane[location.ciphertext * packing.slot_count + location.slot] =
            logical.local_share[i];
    }
    const TensorShape shape{{packing.ciphertext_count, packing.slot_count}};
    return ComplexMpcSharePair{
        {std::move(real), shape, logical.numeric, ComplexComponent::kReal},
        {std::move(imag), shape, logical.numeric,
         ComplexComponent::kImaginary}};
}

Result<ComplexMpcSharePair> ScatterInverse(
    const MpcShareTensor &inverse, const HePackingPlan &broadcast)
{
    auto valid = inverse.Validate();
    if (!valid.ok()) return valid.status();
    if (inverse.physical_shape.dimensions.size() != 2 ||
        inverse.physical_shape.dimensions[1] != 1 ||
        inverse.local_share.size() != broadcast.logical_locations.size())
        return Status::Invalid("inverse-stddev broadcast shape mismatch");
    const auto physical_count = broadcast.slot_count;
    std::vector<std::uint64_t> real(physical_count, 0), imag(physical_count, 0);
    for (std::size_t token = 0; token < inverse.local_share.size(); ++token)
        for (const auto &location : broadcast.Locations(token))
            real[location.slot] = inverse.local_share[token];
    const TensorShape shape{{1, broadcast.slot_count}};
    return ComplexMpcSharePair{
        {std::move(real), shape, inverse.numeric, ComplexComponent::kReal},
        {std::move(imag), shape, inverse.numeric,
         ComplexComponent::kImaginary}};
}

PhysicalSlotMap DensePhysicalRealMap(std::size_t slots)
{
    PhysicalSlotMap map;
    map.ciphertext_count = 1;
    map.slots_per_ciphertext = slots;
    map.logical_element_count = slots;
    map.assignments.resize(slots);
    for (std::size_t slot = 0; slot < slots; ++slot)
        map.assignments[slot].real_logical_index = slot;
    return map;
}

Result<MpcShareTensor> CompactSegment(
    const ComplexMpcSharePair &physical, const HePackingPlan &packing,
    std::size_t ciphertext_offset, TensorShape shape)
{
    std::vector<std::uint64_t> output(packing.logical_locations.size());
    for (std::size_t logical = 0; logical < output.size(); ++logical) {
        const auto &location = packing.Locations(logical).front();
        const auto physical_index =
            (ciphertext_offset + location.ciphertext) * packing.slot_count +
            location.slot;
        output[logical] = location.component == ComplexLane::kReal
            ? physical.real.local_share[physical_index]
            : physical.imag.local_share[physical_index];
    }
    return MpcShareTensor{std::move(output), std::move(shape),
                          physical.real.numeric, ComplexComponent::kReal};
}

int CanonicalRotation(long long value, std::size_t slots)
{
    long long result = value % static_cast<long long>(slots);
    if (result > static_cast<long long>(slots / 2))
        result -= static_cast<long long>(slots);
    if (result < -static_cast<long long>(slots / 2))
        result += static_cast<long long>(slots);
    return static_cast<int>(result);
}

} // namespace

Result<HePackingPlan> FfnGeluPackingPlanner::ComplexColumns(
    const TensorShape &shape, std::size_t slot_count, Layout layout)
{
    if (shape.dimensions.size() != 2 || shape.dimensions[0] == 0 ||
        shape.dimensions[1] == 0 || slot_count == 0)
        return Status::Invalid("complex column packing needs a 2D shape");
    const auto tokens = shape.dimensions[0];
    const auto width = shape.dimensions[1];
    const auto stride = NextPowerOfTwo(tokens);
    if (stride > slot_count)
        return Status::Invalid("one padded token column exceeds CKKS slots");
    const auto columns = slot_count / stride;
    if (columns == 0)
        return Status::Invalid("complex column packing has no feature lane");
    HePackingPlan plan;
    plan.layout = layout;
    plan.logical_shape = shape;
    plan.slot_count = slot_count;
    plan.ciphertext_count = (width + 2 * columns - 1) / (2 * columns);
    plan.logical_locations.resize(tokens * width);
    for (std::size_t token = 0; token < tokens; ++token) {
        for (std::size_t feature = 0; feature < width; ++feature) {
            const auto group = feature / (2 * columns);
            const auto local = feature % (2 * columns);
            plan.logical_locations[token * width + feature].push_back(
                {group, (local % columns) * stride + token,
                 local < columns ? ComplexLane::kReal
                                 : ComplexLane::kImaginary,
                 std::nullopt});
        }
    }
    auto valid = plan.Validate();
    if (!valid.ok()) return valid.status();
    return plan;
}

Result<HePackingPlan> FfnGeluPackingPlanner::RealColumns(
    const TensorShape &shape, std::size_t slot_count, Layout layout)
{
    if (shape.dimensions.size() != 2 || shape.dimensions[0] == 0 ||
        shape.dimensions[1] == 0 || slot_count == 0)
        return Status::Invalid("real column packing needs a 2D shape");
    const auto tokens = shape.dimensions[0];
    const auto width = shape.dimensions[1];
    const auto stride = NextPowerOfTwo(tokens);
    if (stride > slot_count)
        return Status::Invalid("one padded token column exceeds CKKS slots");
    const auto columns = slot_count / stride;
    HePackingPlan plan;
    plan.layout = layout;
    plan.logical_shape = shape;
    plan.slot_count = slot_count;
    plan.ciphertext_count = (width + columns - 1) / columns;
    plan.logical_locations.resize(tokens * width);
    for (std::size_t token = 0; token < tokens; ++token)
        for (std::size_t feature = 0; feature < width; ++feature)
            plan.logical_locations[token * width + feature].push_back(
                {feature / columns, (feature % columns) * stride + token,
                 ComplexLane::kReal, std::nullopt});
    auto valid = plan.Validate();
    if (!valid.ok()) return valid.status();
    return plan;
}

Result<FfnGeluPackingPlan> FfnGeluPackingPlanner::Build(
    const TransformerLayerConfig &layer, std::size_t slot_count)
{
    auto centered = ComplexColumns({{layer.tokens, layer.model_width}},
                                   slot_count, Layout::kComplexStrictColumn);
    if (!centered.ok()) return centered.status();
    auto projected = ComplexColumns({{layer.tokens, layer.ffn_width}},
                                    slot_count, Layout::kFfn);
    if (!projected.ok()) return projected.status();
    const auto stride = NextPowerOfTwo(layer.tokens);
    const auto columns = slot_count / stride;
    HePackingPlan inverse;
    inverse.layout = Layout::kReduction;
    inverse.logical_shape = {{layer.tokens, 1}};
    inverse.slot_count = slot_count;
    inverse.ciphertext_count = 1;
    inverse.logical_locations.resize(layer.tokens);
    for (std::size_t token = 0; token < layer.tokens; ++token)
        for (std::size_t column = 0; column < columns; ++column)
            inverse.logical_locations[token].push_back(
                {0, column * stride + token, ComplexLane::kReal,
                 std::nullopt});
    auto inverse_valid = inverse.Validate();
    if (!inverse_valid.ok()) return inverse_valid.status();

    auto real = RealColumns({{layer.tokens, layer.ffn_width}}, slot_count,
                            Layout::kStrictColumn);
    if (!real.ok()) return real.status();
    return FfnGeluPackingPlan{centered.take_value(), std::move(inverse),
                              projected.take_value(), real.take_value(),
                              stride, columns};
}

Result<void> FfnGeluLevelPlanner::Validate(
    const HeParameterConfig &parameters)
{
    if (parameters.coefficient_modulus_bits.size() < 8 ||
        parameters.declared_depth < 6)
        return Status::FailedPrecondition(
            "FFN/GeLU needs eight configured primes and depth six");
    if (parameters.terminal_prime_count != 1)
        return Status::FailedPrecondition(
            "FFN/GeLU conversion needs a single-prime terminal level");
    if (parameters.input_scale_bits != 45 ||
        parameters.default_scale_bits != 45 ||
        parameters.output_scale_bits != 46)
        return Status::FailedPrecondition(
            "FFN/GeLU currently requires input/default/output scales 45/45/46");
    const auto &bits = parameters.coefficient_modulus_bits;
    // The converter enters at terminal-depth (L1 for the default profile).
    // From there the useful suffix drops 60 bits for LN, four consecutive
    // 46-bit primes for Wu/realify and the two ct-ct GeLU stages, then a
    // 60-bit prime for coefficient products. The leading 60-bit prime is
    // intentionally outside the online suffix and only bounds CKKS special
    // prime construction.
    if (bits[bits.size() - 1] < 58 || bits[bits.size() - 2] < 58 ||
        std::abs(bits[bits.size() - 3] - 46) > 1 ||
        std::abs(bits[bits.size() - 4] - 46) > 1 ||
        std::abs(bits[bits.size() - 5] - 46) > 1 ||
        std::abs(bits[bits.size() - 6] - 46) > 1 ||
        bits[bits.size() - 7] < 58)
        return Status::FailedPrecondition(
            "FFN/GeLU suffix must drop 60,46,46,46,46,60 bits");
    return {};
}

Result<SealCipherTensor> FfnLayerNormTailKernel::Evaluate(
    const SealCipherTensor &centered,
    const SealCipherTensor &inverse_stddev,
    const DenseVectorParameter &gamma,
    const DenseVectorParameter &beta,
    const TransformerLayerConfig &layer,
    const HeContext &context,
    FfnGeluExecutionReport *report)
{
    auto centered_valid = centered.Validate(context);
    if (!centered_valid.ok()) return centered_valid.status();
    auto inverse_valid = inverse_stddev.Validate(context);
    if (!inverse_valid.ok()) return inverse_valid.status();
    if (centered.logical_spec.shape !=
            TensorShape{{layer.tokens, layer.model_width}} ||
        centered.logical_spec.layout != Layout::kComplexStrictColumn ||
        inverse_stddev.logical_spec.shape != TensorShape{{layer.tokens, 1}} ||
        inverse_stddev.ciphertexts.size() != 1 ||
        gamma.values.size() != layer.model_width ||
        beta.values.size() != layer.model_width)
        return Status::Invalid("FFN LayerNorm tail contract mismatch");
    auto plan = FfnGeluPackingPlanner::Build(layer, context.slot_count());
    if (!plan.ok()) return plan.status();
    auto same = SamePacking(centered.physical_packing,
                            plan.value().centered);
    if (!same.ok()) return same.status();
    const double input_scale = std::ldexp(
        1.0, context.parameters.input_scale_bits);
    const auto terminal_level =
        context.parameters.coefficient_modulus_bits.size() -
        context.parameters.terminal_prime_count;
    constexpr std::size_t kFfnGeluDepth = 6;
    if (terminal_level < kFfnGeluDepth)
        return Status::FailedPrecondition(
            "FFN/GeLU suffix needs six remaining CKKS levels");
    const auto input_level = terminal_level - kFfnGeluDepth;
    for (const auto &ct : centered.ciphertexts) {
        const auto level = LevelOf(context, ct.parms_id());
        if (!level.ok()) return level.status();
        if (!CloseScale(ct.scale(), input_scale) ||
            level.value() != input_level)
            return Status::FailedPrecondition(
                "FFN centered input must begin at its planned suffix level");
    }
    const auto &inverse_ct = inverse_stddev.ciphertexts.front();
    auto inverse_level = LevelOf(context, inverse_ct.parms_id());
    if (!inverse_level.ok()) return inverse_level.status();
    if (inverse_level.value() != input_level ||
        !CloseScale(inverse_ct.scale(), input_scale))
        return Status::FailedPrecondition(
            "FFN inverse-stddev must begin at the same suffix level");

    seal::Evaluator evaluator(*context.context);
    seal::CKKSEncoder encoder(*context.context);
    // A 45+45 ciphertext product followed by one 60-bit rescale would leave
    // only a 15-bit gamma plaintext if the output were forced immediately
    // back to 2^45. Keep a truthful 2^60 intermediate instead; Wu consumes it
    // with a 2^45 plaintext and returns to 2^45 at its own rescale. This keeps
    // the one-level lazy schedule while preserving model-parameter precision.
    const int affine_scale_bits =
        context.parameters.default_scale_bits + 15;
    const double target = std::ldexp(1.0, affine_scale_bits);
    SealCipherTensor result = centered;
    const auto columns = plan.value().columns_per_component;
    for (std::size_t group = 0; group < result.ciphertexts.size(); ++group) {
        auto &value = result.ciphertexts[group];
        evaluator.multiply_inplace(value, inverse_ct);
        evaluator.relinearize_inplace(value, context.relinearization_keys);
        if (report) {
            ++report->ciphertext_multiplications;
            ++report->relinearizations;
        }
        seal::Ciphertext conjugated;
        evaluator.complex_conjugate(value, context.galois_keys, conjugated);
        if (report) ++report->conjugations;
        std::vector<double> plus(context.slot_count(), 0.0);
        std::vector<double> minus(context.slot_count(), 0.0);
        std::vector<Complex> bias_slots(context.slot_count());
        for (std::size_t token = 0; token < layer.tokens; ++token) {
            for (std::size_t local = 0; local < columns; ++local) {
                const auto slot = local * plan.value().token_stride + token;
                const auto real_feature = group * 2 * columns + local;
                const auto imag_feature = real_feature + columns;
                const double gr = real_feature < layer.model_width
                    ? gamma.values[real_feature] : 0.0;
                const double gi = imag_feature < layer.model_width
                    ? gamma.values[imag_feature] : 0.0;
                plus[slot] = 0.5 * (gr + gi);
                minus[slot] = 0.5 * (gr - gi);
                if (real_feature < layer.model_width)
                    bias_slots[slot].real(beta.values[real_feature]);
                if (imag_feature < layer.model_width)
                    bias_slots[slot].imag(beta.values[imag_feature]);
            }
        }
        const double plain_scale = target * DroppedPrime(context,
            value.parms_id()) / value.scale();
        if (!std::isfinite(plain_scale) || plain_scale < 2.0)
            return Status::FailedPrecondition(
                "LayerNorm gamma plaintext scale is too small");
        auto encoded_plus = EncodeModelPlaintext(
            encoder, plus, value.parms_id(), plain_scale,
            "ffn.ln2.gamma.plus");
        if (!encoded_plus.ok()) return encoded_plus.status();
        auto plus_plain = encoded_plus.take_value();
        auto encoded_minus = EncodeModelPlaintext(
            encoder, minus, value.parms_id(), plain_scale,
            "ffn.ln2.gamma.minus");
        if (!encoded_minus.ok()) return encoded_minus.status();
        auto minus_plain = encoded_minus.take_value();
        evaluator.multiply_plain_inplace(value, plus_plain);
        evaluator.multiply_plain_inplace(conjugated, minus_plain);
        evaluator.add_inplace(value, conjugated);
        if (report) report->ciphertext_plain_multiplications += 2;
        evaluator.rescale_to_next_inplace(value);
        auto normalized = NormalizeScale(value, target);
        if (!normalized.ok()) return normalized.status();
        if (report) ++report->rescales;
        auto encoded_bias = EncodeModelPlaintext(
            encoder, bias_slots, value.parms_id(), target,
            "ffn.ln2.beta");
        if (!encoded_bias.ok()) return encoded_bias.status();
        auto bias_plain = encoded_bias.take_value();
        evaluator.add_plain_inplace(value, bias_plain);
    }
    result.logical_spec.he_region = HeRegionId::kRegion5;
    result.logical_spec.he_parameter_profile = context.profile;
    result.logical_spec.planned_level = input_level + 1;
    result.logical_spec.planned_scale_bits = affine_scale_bits;
    return result;
}

namespace {

struct ProjectionKey {
    std::size_t source_ct{};
    std::size_t target_ct{};
    ComplexLane source_component{ComplexLane::kReal};
    int rotation{};
    bool operator<(const ProjectionKey &other) const noexcept
    {
        return std::tie(source_ct, target_ct, source_component, rotation) <
               std::tie(other.source_ct, other.target_ct,
                        other.source_component, other.rotation);
    }
};

Complex ExtractionFactor(ComplexLane component)
{
    return component == ComplexLane::kReal ? Complex{0.5, 0.0}
                                           : Complex{0.0, -0.5};
}

Complex DestinationFactor(ComplexLane component)
{
    return component == ComplexLane::kReal ? Complex{1.0, 0.0}
                                           : Complex{0.0, 1.0};
}

} // namespace

Result<SealCipherTensor> ComplexColumnProjectionKernel::Evaluate(
    const SealCipherTensor &input,
    const DenseMatrixParameter &weights,
    const std::optional<DenseVectorParameter> &bias,
    const HePackingPlan &output_packing,
    HeRegionId output_region,
    int output_scale_bits,
    const HeContext &context,
    HeKernelOperationCounts *counts)
{
    auto valid = input.Validate(context);
    if (!valid.ok()) return valid.status();
    auto output_valid = output_packing.Validate();
    if (!output_valid.ok()) return output_valid.status();
    if (input.logical_spec.shape.dimensions.size() != 2 ||
        output_packing.logical_shape.dimensions.size() != 2 ||
        input.logical_spec.shape.dimensions[0] !=
            output_packing.logical_shape.dimensions[0] ||
        input.physical_packing.logical_shape != input.logical_spec.shape ||
        output_packing.slot_count != context.slot_count())
        return Status::Invalid("column projection packing contract mismatch");
    const auto tokens = input.logical_spec.shape.dimensions[0];
    const auto input_width = input.logical_spec.shape.dimensions[1];
    const auto output_width = output_packing.logical_shape.dimensions[1];
    if (weights.spec.rows != input_width ||
        weights.spec.columns != output_width ||
        weights.row_major.size() != input_width * output_width ||
        (bias && bias->values.size() != output_width))
        return Status::Invalid("rectangular projection weight contract mismatch");
    for (const auto &locations : input.physical_packing.logical_locations)
        if (locations.size() != 1)
            return Status::Invalid("projection input must have one physical lane");
    for (const auto &locations : output_packing.logical_locations)
        if (locations.size() != 1)
            return Status::Invalid("projection output must have one physical lane");
    std::map<ProjectionKey, std::vector<Complex>> masks;
    for (std::size_t token = 0; token < tokens; ++token) {
        for (std::size_t output = 0; output < output_width; ++output) {
            const auto &destination = output_packing.Locations(
                token * output_width + output).front();
            for (std::size_t feature = 0; feature < input_width;
                 ++feature) {
                const auto coefficient =
                    weights.row_major[feature * output_width + output];
                if (coefficient == 0.0) continue;
                const auto &source = input.physical_packing.Locations(
                    token * input_width + feature).front();
                const int rotation = CanonicalRotation(
                    static_cast<long long>(source.slot) -
                    static_cast<long long>(destination.slot),
                    context.slot_count());
                ProjectionKey key{source.ciphertext, destination.ciphertext,
                                  source.component, rotation};
                auto inserted = masks.emplace(
                    key, std::vector<Complex>(context.slot_count()));
                inserted.first->second[destination.slot] +=
                    coefficient *
                    ExtractionFactor(source.component) *
                    DestinationFactor(destination.component);
            }
        }
    }
    seal::Evaluator evaluator(*context.context);
    seal::CKKSEncoder encoder(*context.context);
    std::vector<seal::Ciphertext> output(
        output_packing.ciphertext_count);
    std::vector<bool> initialized(output.size(), false);
    std::map<std::pair<std::size_t, ComplexLane>, seal::Ciphertext> extracted;
    const double target = std::ldexp(1.0, output_scale_bits);
    for (const auto &entry : masks) {
        const auto base_key = std::make_pair(entry.first.source_ct,
                                             entry.first.source_component);
        auto base = extracted.find(base_key);
        if (base == extracted.end()) {
            const auto &source = input.ciphertexts[entry.first.source_ct];
            seal::Ciphertext conjugated;
            evaluator.complex_conjugate(source, context.galois_keys,
                                        conjugated);
            seal::Ciphertext component;
            if (entry.first.source_component == ComplexLane::kReal)
                evaluator.add(source, conjugated, component);
            else
                evaluator.sub(source, conjugated, component);
            base = extracted.emplace(base_key, std::move(component)).first;
            if (counts) ++counts->conjugations;
        }
        auto rotated = base->second;
        if (entry.first.rotation != 0) {
            evaluator.rotate_vector_inplace(rotated, entry.first.rotation,
                                            context.galois_keys);
            if (counts) ++counts->rotations;
        }
        const double plain_scale = target * DroppedPrime(
            context, rotated.parms_id()) / rotated.scale();
        auto encoded = EncodeModelPlaintext(
            encoder, entry.second, rotated.parms_id(), plain_scale,
            "ffn.projection.diagonal");
        if (!encoded.ok()) return encoded.status();
        auto plain = encoded.take_value();
        evaluator.multiply_plain_inplace(rotated, plain);
        if (counts) ++counts->ciphertext_plain_multiplications;
        if (!initialized[entry.first.target_ct]) {
            output[entry.first.target_ct] = std::move(rotated);
            initialized[entry.first.target_ct] = true;
        } else {
            evaluator.add_inplace(output[entry.first.target_ct], rotated);
        }
    }
    for (std::size_t ct = 0; ct < output.size(); ++ct) {
        if (!initialized[ct])
            return Status::Internal("projection produced an empty ciphertext group");
        if (bias) {
            std::vector<Complex> slots(context.slot_count());
            for (std::size_t logical = 0;
                 logical < output_packing.logical_locations.size();
                 ++logical) {
                const auto &location =
                    output_packing.Locations(logical).front();
                if (location.ciphertext != ct) continue;
                const double value = bias->values[logical % output_width];
                if (location.component == ComplexLane::kReal)
                    slots[location.slot].real(value);
                else
                    slots[location.slot].imag(value);
            }
            auto encoded_bias = EncodeModelPlaintext(
                encoder, slots, output[ct].parms_id(), output[ct].scale(),
                "ffn.projection.bias");
            if (!encoded_bias.ok()) return encoded_bias.status();
            auto plain = encoded_bias.take_value();
            evaluator.add_plain_inplace(output[ct], plain);
        }
        evaluator.rescale_to_next_inplace(output[ct]);
        auto normalized = NormalizeScale(output[ct], target);
        if (!normalized.ok()) return normalized.status();
        if (counts) ++counts->rescales;
    }
    SealCipherTensor result;
    result.parameter_profile = context.profile;
    result.ciphertexts = std::move(output);
    result.logical_spec.shape = output_packing.logical_shape;
    result.logical_spec.domain = Domain::kHeCiphertext;
    result.logical_spec.layout = output_packing.layout;
    result.logical_spec.numeric = input.logical_spec.numeric;
    result.logical_spec.he_region = output_region;
    result.logical_spec.he_parameter_profile = context.profile;
    auto input_level = LevelOf(context, input.ciphertexts.front().parms_id());
    if (!input_level.ok()) return input_level.status();
    result.logical_spec.planned_level = input_level.value() + 1;
    result.logical_spec.planned_scale_bits = output_scale_bits;
    result.physical_packing = output_packing;
    return result;
}

Result<std::vector<int>> ComplexColumnProjectionKernel::RequiredRotationSteps(
    const HePackingPlan &input_packing,
    const HePackingPlan &output_packing)
{
    auto input_valid = input_packing.Validate();
    if (!input_valid.ok()) return input_valid.status();
    auto output_valid = output_packing.Validate();
    if (!output_valid.ok()) return output_valid.status();
    if (input_packing.logical_shape.dimensions.size() != 2 ||
        output_packing.logical_shape.dimensions.size() != 2 ||
        input_packing.logical_shape.dimensions[0] !=
            output_packing.logical_shape.dimensions[0] ||
        input_packing.slot_count != output_packing.slot_count)
        return Status::Invalid("projection rotation plans are incompatible");
    const auto tokens = input_packing.logical_shape.dimensions[0];
    const auto in_width = input_packing.logical_shape.dimensions[1];
    const auto out_width = output_packing.logical_shape.dimensions[1];
    std::set<int> steps{0};
    for (std::size_t token = 0; token < tokens; ++token)
        for (std::size_t output = 0; output < out_width; ++output) {
            const auto &destination = output_packing.Locations(
                token * out_width + output).front();
            for (std::size_t input = 0; input < in_width; ++input) {
                const auto &source = input_packing.Locations(
                    token * in_width + input).front();
                steps.insert(CanonicalRotation(
                    static_cast<long long>(source.slot) -
                    static_cast<long long>(destination.slot),
                    input_packing.slot_count));
            }
        }
    return std::vector<int>(steps.begin(), steps.end());
}

Result<SealCipherTensor> FfnUpProjectionKernel::Evaluate(
    const SealCipherTensor &input,
    const DenseMatrixParameter &weights,
    const std::optional<DenseVectorParameter> &bias,
    const TransformerLayerConfig &layer,
    const HeContext &context,
    FfnGeluExecutionReport *report)
{
    if (input.logical_spec.shape !=
            TensorShape{{layer.tokens, layer.model_width}} ||
        input.logical_spec.layout != Layout::kComplexStrictColumn)
        return Status::Invalid("Wu projection contract mismatch");
    auto plans = FfnGeluPackingPlanner::Build(layer, context.slot_count());
    if (!plans.ok()) return plans.status();
    HeKernelOperationCounts counts;
    auto result = ComplexColumnProjectionKernel::Evaluate(
        input, weights, bias, plans.value().projected,
        HeRegionId::kRegion5, context.parameters.default_scale_bits,
        context, &counts);
    if (report) {
        report->rotations += counts.rotations;
        report->conjugations += counts.conjugations;
        report->ciphertext_plain_multiplications +=
            counts.ciphertext_plain_multiplications;
        report->rescales += counts.rescales;
    }
    return result;
}

Result<SealCipherTensor> ComplexColumnRealifyKernel::Evaluate(
    const SealCipherTensor &input,
    const HePackingPlan &real_output_packing,
    int output_scale_bits,
    const HeContext &context,
    HeKernelOperationCounts *counts)
{
    auto valid = input.Validate(context);
    if (!valid.ok()) return valid.status();
    auto output_valid = real_output_packing.Validate();
    if (!output_valid.ok()) return output_valid.status();
    if (input.logical_spec.shape != real_output_packing.logical_shape ||
        real_output_packing.slot_count != context.slot_count())
        return Status::Invalid("realify packing contract mismatch");
    const auto tokens = input.logical_spec.shape.dimensions[0];
    const auto width = input.logical_spec.shape.dimensions[1];
    const auto stride = NextPowerOfTwo(tokens);
    const auto columns = context.slot_count() / stride;
    auto expected_input = FfnGeluPackingPlanner::ComplexColumns(
        input.logical_spec.shape, context.slot_count(),
        input.logical_spec.layout);
    if (!expected_input.ok()) return expected_input.status();
    auto same = SamePacking(input.physical_packing, expected_input.value());
    if (!same.ok()) return same.status();
    auto expected_output = FfnGeluPackingPlanner::RealColumns(
        input.logical_spec.shape, context.slot_count(),
        real_output_packing.layout);
    if (!expected_output.ok()) return expected_output.status();
    same = SamePacking(real_output_packing, expected_output.value());
    if (!same.ok()) return same.status();
    seal::Evaluator evaluator(*context.context);
    seal::CKKSEncoder encoder(*context.context);
    std::vector<seal::Ciphertext> output(real_output_packing.ciphertext_count);
    const double wanted = std::ldexp(1.0, output_scale_bits);
    for (std::size_t group = 0; group < input.ciphertexts.size(); ++group) {
        const auto &source = input.ciphertexts[group];
        seal::Ciphertext conjugated;
        evaluator.complex_conjugate(source, context.galois_keys, conjugated);
        seal::Ciphertext real;
        evaluator.add(source, conjugated, real);
        const double plain_scale = wanted * DroppedPrime(
            context, real.parms_id()) / real.scale();
        std::vector<Complex> real_factor(
            context.slot_count(), Complex{0.5, 0.0});
        seal::Plaintext real_plain;
        encoder.encode(real_factor, real.parms_id(), plain_scale, real_plain);
        evaluator.multiply_plain_inplace(real, real_plain);
        evaluator.rescale_to_next_inplace(real);
        auto normalized = NormalizeScale(real, wanted);
        if (!normalized.ok()) return normalized.status();
        const auto real_index = 2 * group;
        if (real_index < output.size()) output[real_index] = std::move(real);
        const auto imaginary_begin =
            (2 * group + 1) * columns;
        if (imaginary_begin < width) {
            seal::Ciphertext imaginary;
            evaluator.sub(source, conjugated, imaginary);
            std::vector<Complex> factor(
                context.slot_count(),
                Complex{0.0, -0.5});
            seal::Plaintext factor_plain;
            encoder.encode(factor, imaginary.parms_id(), plain_scale,
                           factor_plain);
            evaluator.multiply_plain_inplace(imaginary, factor_plain);
            evaluator.rescale_to_next_inplace(imaginary);
            normalized = NormalizeScale(imaginary, wanted);
            if (!normalized.ok()) return normalized.status();
            output[real_index + 1] = std::move(imaginary);
            if (counts) {
                ++counts->ciphertext_plain_multiplications;
                ++counts->rescales;
            }
        }
        if (counts) {
            ++counts->conjugations;
            ++counts->ciphertext_plain_multiplications;
            ++counts->rescales;
        }
    }
    SealCipherTensor result;
    result.parameter_profile = context.profile;
    result.ciphertexts = std::move(output);
    result.logical_spec = input.logical_spec;
    result.logical_spec.layout = Layout::kStrictColumn;
    auto input_level = LevelOf(context, input.ciphertexts.front().parms_id());
    if (!input_level.ok()) return input_level.status();
    result.logical_spec.planned_level = input_level.value() + 1;
    result.logical_spec.planned_scale_bits = output_scale_bits;
    result.physical_packing = real_output_packing;
    return result;
}

Result<SealCipherTensor> FfnRealifyKernel::Evaluate(
    const SealCipherTensor &input,
    const TransformerLayerConfig &layer,
    const HeContext &context,
    FfnGeluExecutionReport *report)
{
    auto plans = FfnGeluPackingPlanner::Build(layer, context.slot_count());
    if (!plans.ok()) return plans.status();
    HeKernelOperationCounts counts;
    auto result = ComplexColumnRealifyKernel::Evaluate(
        input, plans.value().realified,
        context.parameters.output_scale_bits, context, &counts);
    if (report) {
        report->conjugations += counts.conjugations;
        report->ciphertext_plain_multiplications +=
            counts.ciphertext_plain_multiplications;
        report->rescales += counts.rescales;
    }
    return result;
}

Result<GeluPolynomialOutputs> GeluPolynomialKernel::Evaluate(
    const SealCipherTensor &input,
    const GeluConfig &config,
    const HeContext &context,
    FfnGeluExecutionReport *report)
{
    auto valid = input.Validate(context);
    if (!valid.ok()) return valid.status();
    const double target = std::ldexp(
        1.0, context.parameters.output_scale_bits);
    const auto terminal_level =
        context.parameters.coefficient_modulus_bits.size() -
        context.parameters.terminal_prime_count;
    constexpr std::size_t kPolynomialDepth = 3;
    if (terminal_level < kPolynomialDepth)
        return Status::FailedPrecondition(
            "GeLU polynomial needs three remaining CKKS levels");
    const auto polynomial_input_level = terminal_level - kPolynomialDepth;
    for (const auto &ct : input.ciphertexts) {
        auto level = LevelOf(context, ct.parms_id());
        if (!level.ok()) return level.status();
        if (level.value() != polynomial_input_level ||
            !CloseScale(ct.scale(), target))
            return Status::FailedPrecondition(
                "GeLU polynomial input is not at its suffix start");
    }
    seal::Evaluator evaluator(*context.context);
    seal::CKKSEncoder encoder(*context.context);
    SealCipherTensor negative = input;
    SealCipherTensor positive = input;
    const auto terminal_id = LevelId(context, terminal_level);
    if (!terminal_id.ok()) return terminal_id.status();

    for (std::size_t i = 0; i < input.ciphertexts.size(); ++i) {
        const auto &x = input.ciphertexts[i];
        seal::Ciphertext x2;
        evaluator.square(x, x2);
        evaluator.relinearize_inplace(x2, context.relinearization_keys);
        evaluator.rescale_to_next_inplace(x2);
        auto scale_status = NormalizeScale(x2, target);
        if (!scale_status.ok()) return scale_status.status();
        if (report) {
            ++report->ciphertext_multiplications;
            ++report->relinearizations;
            ++report->rescales;
        }

        seal::Ciphertext x_at3 = x;
        evaluator.mod_switch_to_inplace(x_at3, x2.parms_id());
        seal::Ciphertext x3;
        evaluator.multiply(x2, x_at3, x3);
        evaluator.relinearize_inplace(x3, context.relinearization_keys);
        evaluator.rescale_to_next_inplace(x3);
        scale_status = NormalizeScale(x3, target);
        if (!scale_status.ok()) return scale_status.status();
        seal::Ciphertext x4;
        evaluator.square(x2, x4);
        evaluator.relinearize_inplace(x4, context.relinearization_keys);
        evaluator.rescale_to_next_inplace(x4);
        scale_status = NormalizeScale(x4, target);
        if (!scale_status.ok()) return scale_status.status();
        if (report) {
            report->ciphertext_multiplications += 2;
            report->relinearizations += 2;
            report->rescales += 2;
            ++report->modulus_switches;
        }

        seal::Ciphertext x_at4 = x_at3;
        seal::Ciphertext x2_at4 = x2;
        evaluator.mod_switch_to_inplace(x_at4, x3.parms_id());
        evaluator.mod_switch_to_inplace(x2_at4, x3.parms_id());
        if (report) report->modulus_switches += 2;

        auto coefficient_term = [&](seal::Ciphertext value,
                                    double coefficient)
            -> Result<seal::Ciphertext> {
            const double plain_scale = target * DroppedPrime(
                context, value.parms_id()) / value.scale();
            seal::Plaintext plain;
            encoder.encode(coefficient, value.parms_id(), plain_scale, plain);
            evaluator.multiply_plain_inplace(value, plain);
            evaluator.rescale_to_next_inplace(value);
            auto normalized = NormalizeScale(value, target);
            if (!normalized.ok()) return normalized.status();
            if (report) {
                ++report->ciphertext_plain_multiplications;
                ++report->rescales;
            }
            return value;
        };

        auto t4 = coefficient_term(x4, config.a);
        auto t2 = coefficient_term(x2_at4, config.c);
        auto n3 = coefficient_term(x3, -config.b);
        auto p3 = coefficient_term(x3, config.b);
        auto n1 = coefficient_term(x_at4, 0.5 - config.d);
        auto p1 = coefficient_term(x_at4, 0.5 + config.d);
        if (!t4.ok()) return t4.status();
        if (!t2.ok()) return t2.status();
        if (!n3.ok()) return n3.status();
        if (!p3.ok()) return p3.status();
        if (!n1.ok()) return n1.status();
        if (!p1.ok()) return p1.status();

        seal::Ciphertext f0 = t4.value();
        evaluator.add_inplace(f0, t2.value());
        evaluator.add_inplace(f0, n3.value());
        evaluator.add_inplace(f0, n1.value());
        seal::Ciphertext f1 = t4.take_value();
        evaluator.add_inplace(f1, t2.take_value());
        evaluator.add_inplace(f1, p3.take_value());
        evaluator.add_inplace(f1, p1.take_value());
        seal::Plaintext constant;
        encoder.encode(config.e, f0.parms_id(), target, constant);
        evaluator.add_plain_inplace(f0, constant);
        evaluator.add_plain_inplace(f1, constant);
        if (f0.parms_id() != terminal_id.value()) {
            evaluator.mod_switch_to_inplace(f0, terminal_id.value());
            evaluator.mod_switch_to_inplace(f1, terminal_id.value());
            if (report) report->modulus_switches += 2;
        }
        negative.ciphertexts[i] = std::move(f0);
        positive.ciphertexts[i] = std::move(f1);
    }
    SealCipherTensor terminal_input = input;
    for (auto &ct : terminal_input.ciphertexts) {
        evaluator.mod_switch_to_inplace(ct, terminal_id.value());
        if (report) ++report->modulus_switches;
    }
    for (auto *tensor : {&terminal_input, &negative, &positive}) {
        tensor->logical_spec.planned_level = terminal_level;
        tensor->logical_spec.planned_scale_bits =
            context.parameters.output_scale_bits;
    }
    return GeluPolynomialOutputs{std::move(terminal_input),
                                 std::move(negative),
                                 std::move(positive)};
}

class MpcGeluRuntime::Impl {
public:
    explicit Impl(std::shared_ptr<SharedMpcSession> session)
        : session_(std::move(session)), native_(session_->Acquire()),
          party_(native_->party())
    {}

    FixArray Secret(const MpcShareTensor &tensor)
    {
        FixArray result(party_, static_cast<int>(tensor.local_share.size()),
                        tensor.numeric.is_signed, tensor.numeric.ring_bits,
                        tensor.numeric.scale_bits);
        const auto mask = RingMask(tensor.numeric.ring_bits);
        for (std::size_t i = 0; i < tensor.local_share.size(); ++i)
            result.data[i] = tensor.local_share[i] & mask;
        return result;
    }

    Result<MpcShareTensor> Evaluate(
        const MpcShareTensor &input,
        const MpcShareTensor &negative,
        const MpcShareTensor &positive,
        double threshold)
    {
        if (input.local_share.empty() || input.local_share.size() >
                static_cast<std::size_t>(std::numeric_limits<int>::max()))
            return Status::Invalid("GeLU MPC tensor size is unsupported");
        auto x = Secret(input);
        auto f0 = Secret(negative);
        auto f1 = Secret(positive);
        const auto encode = [&](double value) {
            const auto scaled = static_cast<std::int64_t>(
                std::llround(std::ldexp(value, input.numeric.scale_bits)));
            return static_cast<std::uint64_t>(scaled) &
                   RingMask(input.numeric.ring_bits);
        };
        auto &fix = native_->fix();
        auto b0 = fix.LT(x, encode(-threshold));
        auto b1 = fix.LT(x, std::uint64_t{0});
        auto b2 = fix.GT(x, encode(threshold));
        auto z0 = fix.bool_op->XOR(b0, b1);
        auto z1 = fix.bool_op->XOR(b1, b2);
        if (party_ == sci::ALICE)
            for (int i = 0; i < z1.size; ++i) z1.data[i] ^= 1U;
        FixArray zero(party_, x.size, true, x.ell, x.s);
        std::fill(zero.data, zero.data + zero.size, std::uint64_t{0});
        auto r0 = fix.if_else(z0, f0, zero);
        auto r1 = fix.if_else(z1, f1, zero);
        auto r2 = fix.if_else(b2, x, zero);
        auto result = fix.add(fix.add(r0, r1), r2);
        return MpcShareTensor{
            std::vector<std::uint64_t>(result.data,
                                       result.data + result.size),
            input.physical_shape, input.numeric, ComplexComponent::kReal};
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

MpcGeluRuntime::MpcGeluRuntime(
    PartyRole role, std::string peer_address, int port)
    : session_(std::make_shared<SharedMpcSession>(
          role, std::move(peer_address), port))
{}

MpcGeluRuntime::MpcGeluRuntime(
    std::shared_ptr<SharedMpcSession> session)
    : session_(std::move(session))
{
    if (!session_) throw std::invalid_argument("MPC session is null");
}

MpcGeluRuntime::~MpcGeluRuntime() = default;
MpcGeluRuntime::MpcGeluRuntime(MpcGeluRuntime &&) noexcept = default;
MpcGeluRuntime &MpcGeluRuntime::operator=(MpcGeluRuntime &&) noexcept = default;

Result<MpcShareTensor> MpcGeluRuntime::Evaluate(
    const MpcShareTensor &input,
    const MpcShareTensor &negative_branch,
    const MpcShareTensor &positive_branch,
    double threshold,
    FfnGeluExecutionReport *report)
{
    const auto started = Clock::now();
    try {
        for (const auto *tensor : {&input, &negative_branch,
                                   &positive_branch}) {
            auto valid = tensor->Validate();
            if (!valid.ok()) return valid.status();
            if (!tensor->numeric.is_signed)
                return Status::Invalid("GeLU requires signed MPC shares");
        }
        if (input.physical_shape != negative_branch.physical_shape ||
            input.physical_shape != positive_branch.physical_shape ||
            input.numeric.ring_bits != negative_branch.numeric.ring_bits ||
            input.numeric.scale_bits != negative_branch.numeric.scale_bits ||
            input.numeric.ring_bits != positive_branch.numeric.ring_bits ||
            input.numeric.scale_bits != positive_branch.numeric.scale_bits)
            return Status::Invalid("GeLU branch share contracts differ");
        if (!std::atomic_load(&impl_))
            std::atomic_store(&impl_, std::make_shared<Impl>(
                session_));
        const auto before_bytes = impl_->communication();
        const auto before_rounds = impl_->rounds();
        auto result = impl_->Evaluate(input, negative_branch,
                                      positive_branch, threshold);
        if (!result.ok()) return result.status();
        if (report) {
            report->comparisons += 3 * input.local_share.size();
            report->muxes += 3 * input.local_share.size();
            report->mpc_communication_bytes +=
                impl_->communication() - before_bytes;
            report->mpc_rounds += impl_->rounds() - before_rounds;
            report->mpc_milliseconds += std::chrono::duration<double,
                std::milli>(Clock::now() - started).count();
        }
        return result;
    } catch (const std::exception &error) {
        return Status::Internal(std::string("MPC GeLU failed: ") +
                                error.what());
    }
}

void MpcGeluRuntime::ReleaseSession()
{
    std::atomic_store(&impl_, std::shared_ptr<Impl>{});
}

void MpcGeluRuntime::Cancel() noexcept
{
    if (session_) session_->Cancel();
}

SealSciFfnGeluExecutor::SealSciFfnGeluExecutor(
    const HeraConfig &config,
    const TransformerLayerConfig &layer,
    HeContextRegistry &contexts,
    SealPlainParameterStore &parameters,
    CryptoTensorStore &tensors,
    CkksMpcConverter &converter,
    MpcGeluRuntime &mpc,
    TensorIdAllocator *ids)
    : config_(config), layer_(layer), contexts_(contexts),
      parameters_(parameters), tensors_(tensors), converter_(converter),
      mpc_(mpc), ids_(ids)
{}

Result<std::vector<int>> SealSciFfnGeluExecutor::RequiredRotationSteps(
    const TransformerLayerConfig &layer,
    const HeParameterConfig &parameters)
{
    auto level = FfnGeluLevelPlanner::Validate(parameters);
    if (!level.ok()) return level.status();
    auto layer_valid = ValidateLayerConfig(layer);
    if (!layer_valid.ok()) return layer_valid.status();
    const auto slots = parameters.polynomial_degree / 2;
    auto packing = FfnGeluPackingPlanner::Build(layer, slots);
    if (!packing.ok()) return packing.status();
    std::set<int> steps{0};
    // Wu performs direct source-slot to destination-slot rotations. SEAL
    // needs a key for every direct step; a power-of-two superset is not
    // sufficient because Evaluator::rotate_vector does not compose keys.
    for (std::size_t token = 0; token < layer.tokens; ++token) {
        for (std::size_t output = 0; output < layer.ffn_width; ++output) {
            const auto &destination = packing.value().projected.Locations(
                token * layer.ffn_width + output).front();
            for (std::size_t feature = 0; feature < layer.model_width;
                 ++feature) {
                const auto &source = packing.value().centered.Locations(
                    token * layer.model_width + feature).front();
                steps.insert(CanonicalRotation(
                    static_cast<long long>(source.slot) -
                    static_cast<long long>(destination.slot), slots));
            }
        }
    }
    return std::vector<int>(steps.begin(), steps.end());
}

namespace {

Result<PhysicalSlotMap> CombinedRealMap(const HePackingPlan &packing,
                                        std::size_t copies)
{
    PhysicalSlotMap map;
    map.ciphertext_count = packing.ciphertext_count * copies;
    map.slots_per_ciphertext = packing.slot_count;
    map.logical_element_count =
        packing.logical_locations.size() * copies;
    map.assignments.resize(map.ciphertext_count * map.slots_per_ciphertext);
    for (std::size_t copy = 0; copy < copies; ++copy) {
        for (std::size_t logical = 0;
             logical < packing.logical_locations.size(); ++logical) {
            const auto &location = packing.Locations(logical).front();
            if (location.component != ComplexLane::kReal)
                return Status::Invalid(
                    "combined GeLU conversion requires real-only packing");
            auto &assignment = map.assignments[
                (copy * packing.ciphertext_count + location.ciphertext) *
                    packing.slot_count + location.slot];
            assignment.real_logical_index =
                copy * packing.logical_locations.size() + logical;
        }
    }
    auto valid = map.Validate();
    if (!valid.ok()) return valid.status();
    return map;
}

} // namespace

Result<FfnGeluOutputs> SealSciFfnGeluExecutor::Evaluate(
    const FfnGeluInputs &inputs,
    const LayerWeightHandles &weights,
    FfnGeluExecutionReport *report)
{
    const auto started = Clock::now();
    try {
        if (report) *report = {};
        if (layer_.activation != ActivationKind::kGelu)
            return Status::KernelUnavailable(
                "concrete FFN stage currently implements only GeLU");
        const auto *execution = FindLayerExecution(config_, layer_.id);
        if (!execution)
            return Status::Invalid("FFN layer has no HE placement");
        const auto region = execution->regions.find(HeRegionId::kRegion5);
        if (region == execution->regions.end())
            return Status::Invalid("FFN region 5 is absent");
        const auto &context = contexts_.Require(
            region->second.parameter_profile, contexts_.role());
        auto level_valid = FfnGeluLevelPlanner::Validate(context.parameters);
        if (!level_valid.ok()) return level_valid.status();
        if (context.galois_keys.size() == 0)
            return Status::FailedPrecondition(
                "FFN/GeLU Galois keys have not been compiled");
        auto required = RequiredRotationSteps(layer_, context.parameters);
        if (!required.ok()) return required.status();
        const auto galois_tool =
            context.context->key_context_data()->galois_tool();
        for (const auto step : required.value())
            if (!context.galois_keys.has_key(
                    galois_tool->get_elt_from_step(step)))
                return Status::FailedPrecondition(
                    "FFN/GeLU is missing rotation/conjugation key for step " +
                    std::to_string(step));
        if (!context.relinearization_keys.has_key(2))
            return Status::FailedPrecondition(
                "FFN/GeLU is missing the degree-2 relinearization key");
        if (report) report->required_rotation_steps = required.value();

        const TensorShape centered_shape{{layer_.tokens,
                                          layer_.model_width}};
        const TensorShape inverse_shape{{layer_.tokens, 1}};
        if (!inputs.centered_shares.valid() ||
            !inputs.inverse_stddev_shares.valid() ||
            inputs.centered_shares.spec.domain !=
                Domain::kMpcArithmeticShare ||
            inputs.inverse_stddev_shares.spec.domain !=
                Domain::kMpcArithmeticShare ||
            inputs.centered_shares.spec.shape != centered_shape ||
            inputs.inverse_stddev_shares.spec.shape != inverse_shape)
            return Status::Invalid("FFN LayerNorm MPC input contract mismatch");
        auto centered = tensors_.GetMpcShare(inputs.centered_shares.id);
        if (!centered.ok()) return centered.status();
        auto inverse = tensors_.GetMpcShare(inputs.inverse_stddev_shares.id);
        if (!inverse.ok()) return inverse.status();
        if (!centered.value().numeric.is_signed ||
            !inverse.value().numeric.is_signed ||
            centered.value().numeric.ring_bits !=
                config_.mpc_numeric.ring_bits ||
            centered.value().numeric.scale_bits !=
                config_.mpc_numeric.scale_bits ||
            inverse.value().numeric.ring_bits !=
                config_.mpc_numeric.ring_bits ||
            inverse.value().numeric.scale_bits !=
                config_.mpc_numeric.scale_bits)
            return Status::Invalid("FFN inputs require signed configured MPC format");

        auto plans = FfnGeluPackingPlanner::Build(layer_,
                                                   context.slot_count());
        if (!plans.ok()) return plans.status();
        const auto terminal_level =
            context.parameters.coefficient_modulus_bits.size() -
            context.parameters.terminal_prime_count;
        constexpr std::size_t kFfnGeluDepth = 6;
        if (terminal_level < kFfnGeluDepth)
            return Status::FailedPrecondition(
                "FFN/GeLU profile has no six-level suffix");
        const auto ffn_start_level = terminal_level - kFfnGeluDepth;
        auto centered_pair = Scatter(centered.value(), plans.value().centered);
        if (!centered_pair.ok()) return centered_pair.status();
        auto inverse_pair = ScatterInverse(inverse.value(),
                                            plans.value().inverse_broadcast);
        if (!inverse_pair.ok()) return inverse_pair.status();
        auto centered_map = ToConversionMap(plans.value().centered);
        if (!centered_map.ok()) return centered_map.status();
        const auto inverse_map = DensePhysicalRealMap(context.slot_count());

        auto convert_to_he = [&](const ComplexMpcSharePair &shares,
                                 const TensorShape &shape, Layout layout,
                                 const PhysicalSlotMap &map)
            -> Result<std::optional<SealCipherBatch>> {
            const auto before = converter_.stats();
            auto output = converter_.MpcToCkks(
                shares,
                {context.profile, ffn_start_level,
                 context.parameters.input_scale_bits, shape, layout, map});
            if (!output.ok()) return output.status();
            const auto after = converter_.stats();
            if (report) {
                ++report->mpc_to_ckks_calls;
                report->conversion_communication_bytes +=
                    after.communication_bytes - before.communication_bytes;
                report->conversion_rounds += after.rounds - before.rounds;
                report->conversion_milliseconds +=
                    after.milliseconds - before.milliseconds;
            }
            return output;
        };
        auto centered_he = convert_to_he(
            centered_pair.value(), centered_shape,
            Layout::kComplexStrictColumn, centered_map.value());
        if (!centered_he.ok()) return centered_he.status();
        auto inverse_he = convert_to_he(
            inverse_pair.value(), {{1, context.slot_count()}},
            Layout::kReduction, inverse_map);
        if (!inverse_he.ok()) return inverse_he.status();

        std::optional<GeluPolynomialOutputs> polynomials;
        if (contexts_.role() == PartyRole::kServer) {
            if (!centered_he.value() || !inverse_he.value())
                return Status::Internal(
                    "server did not receive FFN MPC-to-CKKS outputs");
            if (weights.layer_id != layer_.id || !weights.wu.valid() ||
                weights.wu.spec.rows != layer_.model_width ||
                weights.wu.spec.columns != layer_.ffn_width)
                return Status::Invalid("Wu handle has the wrong shape");
            const auto *gamma_handle = FindPlain(weights, "ln2.gamma");
            const auto *beta_handle = FindPlain(weights, "ln2.beta");
            if (!gamma_handle || !beta_handle ||
                gamma_handle->spec.length != layer_.model_width ||
                beta_handle->spec.length != layer_.model_width)
                return Status::Invalid("LN2 gamma/beta handles are invalid");
            auto gamma = parameters_.GetVector(gamma_handle->id);
            if (!gamma.ok()) return gamma.status();
            auto beta = parameters_.GetVector(beta_handle->id);
            if (!beta.ok()) return beta.status();
            auto wu = parameters_.GetMatrix(weights.wu.id);
            if (!wu.ok()) return wu.status();
            std::optional<DenseVectorParameter> bu;
            if (layer_.ffn_bias) {
                const auto *handle = FindPlain(weights, "bu");
                if (!handle || handle->spec.length != layer_.ffn_width)
                    return Status::Invalid("bu handle has the wrong shape");
                auto value = parameters_.GetVector(handle->id);
                if (!value.ok()) return value.status();
                bu = value.take_value();
            }

            SealCipherTensor centered_tensor;
            centered_tensor.parameter_profile = context.profile;
            centered_tensor.ciphertexts = centered_he.value()->ciphertexts;
            centered_tensor.logical_spec = inputs.centered_shares.spec;
            centered_tensor.logical_spec.domain = Domain::kHeCiphertext;
            centered_tensor.logical_spec.layout = Layout::kComplexStrictColumn;
            centered_tensor.logical_spec.he_region = HeRegionId::kRegion5;
            centered_tensor.logical_spec.he_parameter_profile = context.profile;
            centered_tensor.logical_spec.planned_level = ffn_start_level;
            centered_tensor.logical_spec.planned_scale_bits =
                context.parameters.input_scale_bits;
            centered_tensor.physical_packing = plans.value().centered;

            SealCipherTensor inverse_tensor;
            inverse_tensor.parameter_profile = context.profile;
            inverse_tensor.ciphertexts = inverse_he.value()->ciphertexts;
            inverse_tensor.logical_spec = inputs.inverse_stddev_shares.spec;
            inverse_tensor.logical_spec.domain = Domain::kHeCiphertext;
            inverse_tensor.logical_spec.layout = Layout::kReduction;
            inverse_tensor.logical_spec.he_region = HeRegionId::kRegion5;
            inverse_tensor.logical_spec.he_parameter_profile = context.profile;
            inverse_tensor.logical_spec.planned_level = ffn_start_level;
            inverse_tensor.logical_spec.planned_scale_bits =
                context.parameters.input_scale_bits;
            inverse_tensor.physical_packing = plans.value().inverse_broadcast;

            auto normalized = FfnLayerNormTailKernel::Evaluate(
                centered_tensor, inverse_tensor, gamma.value(), beta.value(),
                layer_, context, report);
            if (!normalized.ok()) return normalized.status();
            auto projected = FfnUpProjectionKernel::Evaluate(
                normalized.value(), wu.value(), bu, layer_, context, report);
            if (!projected.ok()) return projected.status();
            auto real = FfnRealifyKernel::Evaluate(
                projected.value(), layer_, context, report);
            if (!real.ok()) return real.status();
            auto polynomial = GeluPolynomialKernel::Evaluate(
                real.value(), config_.gelu, context, report);
            if (!polynomial.ok()) return polynomial.status();
            polynomials = polynomial.take_value();
        }

        auto combined_map = CombinedRealMap(plans.value().realified, 3);
        if (!combined_map.ok()) return combined_map.status();
        std::optional<SealCipherBatch> batch;
        if (contexts_.role() == PartyRole::kServer) {
            std::vector<seal::Ciphertext> ciphertexts;
            for (const auto *tensor : {&polynomials->input,
                                       &polynomials->negative_branch,
                                       &polynomials->positive_branch})
                ciphertexts.insert(ciphertexts.end(),
                                   tensor->ciphertexts.begin(),
                                   tensor->ciphertexts.end());
            batch = SealCipherBatch{
                context.profile, std::move(ciphertexts),
                {{3, layer_.tokens, layer_.ffn_width}},
                Layout::kStrictColumn, combined_map.value()};
        }
        const auto before = converter_.stats();
        auto physical = converter_.CkksToMpc(batch);
        if (!physical.ok()) return physical.status();
        const auto after = converter_.stats();
        if (report) {
            ++report->ckks_to_mpc_calls;
            report->conversion_communication_bytes +=
                after.communication_bytes - before.communication_bytes;
            report->conversion_rounds += after.rounds - before.rounds;
            report->conversion_milliseconds +=
                after.milliseconds - before.milliseconds;
        }
        const TensorShape output_shape{{layer_.tokens, layer_.ffn_width}};
        const auto group_cts = plans.value().realified.ciphertext_count;
        auto x_share = CompactSegment(physical.value(),
                                      plans.value().realified, 0,
                                      output_shape);
        auto f0_share = CompactSegment(physical.value(),
                                       plans.value().realified, group_cts,
                                       output_shape);
        auto f1_share = CompactSegment(physical.value(),
                                       plans.value().realified, 2 * group_cts,
                                       output_shape);
        if (!x_share.ok()) return x_share.status();
        if (!f0_share.ok()) return f0_share.status();
        if (!f1_share.ok()) return f1_share.status();
        auto activation = mpc_.Evaluate(
            x_share.value(), f0_share.value(), f1_share.value(),
            config_.gelu.threshold, report);
        if (!activation.ok()) return activation.status();
        // Preserve the GeLU Ferret session for subsequent layers/runs.

        TensorSpec spec;
        spec.shape = output_shape;
        spec.domain = Domain::kMpcArithmeticShare;
        spec.layout = Layout::kFfn;
        spec.numeric = activation.value().numeric;
        const auto id = ids_ ? ids_->Next() : next_id_.fetch_add(1);
        auto stored = tensors_.PutMpcShare(id, activation.take_value());
        if (!stored.ok()) return stored.status();
        if (report)
            report->he_milliseconds += std::chrono::duration<double,
                std::milli>(Clock::now() - started).count() -
                report->conversion_milliseconds - report->mpc_milliseconds;
        return FfnGeluOutputs{TensorHandle{id, spec}};
    } catch (const std::exception &error) {
        return Status::Internal(std::string("FFN/GeLU stage failed: ") +
                                error.what());
    }
}

} // namespace hera
