#include "hera/blb_reference.h"
#include "hera/ffn_gelu.h"
#include "hera/detail/scoped_wall_timer.h"
#include "hera/detail/he_level.h"
#include "hera/detail/he_packing.h"
#include "hera/detail/mpc_session_internal.h"
#include "hera/detail/parallel_ranges.h"

#include <FloatingPoint/fixed-point.h>
#include <FloatingPoint/fp-math.h>
#include <OT/emp-ot.h>
#include <utils/emp-tool.h>

#include <seal/util/ntt.h>

#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <list>
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

using detail::LevelOf;

using detail::LevelId;

using detail::DroppedPrime;

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

Result<void> SamePacking(const HePackingPlan &left, const HePackingPlan &right)
{
    return detail::RequireSamePacking(left, right);
}

using detail::ToConversionMap;

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

Result<MpcShareTensor> CompactSegment(
    const ComplexMpcSharePair &physical, const HePackingPlan &packing,
    std::size_t ciphertext_offset, TensorShape shape,
    std::optional<ComplexLane> component_override = std::nullopt,
    bool compensate_phase = false)
{
    std::vector<std::uint64_t> output(packing.logical_locations.size());
    for (std::size_t logical = 0; logical < output.size(); ++logical) {
        const auto &location = packing.Locations(logical).front();
        const auto physical_index =
            (ciphertext_offset + location.ciphertext) * packing.slot_count +
            location.slot;
        const auto lane = component_override.value_or(location.component);
        auto value = lane == ComplexLane::kReal
            ? physical.real.local_share[physical_index]
            : physical.imag.local_share[physical_index];
        if (compensate_phase && lane == ComplexLane::kImaginary && (location.slot & 1))
            value = (std::uint64_t{0} - value) & RingMask(physical.real.numeric.ring_bits);
        output[logical] = value;
    }
    return MpcShareTensor{std::move(output), std::move(shape),
                          physical.real.numeric, ComplexComponent::kReal,
                          physical.real.public_value_bound_bits};
}

int CanonicalRotation(long long value, std::size_t slots)
{
    long long result = value % static_cast<long long>(slots);
    if (result > static_cast<long long>(slots / 2))
        result -= static_cast<long long>(slots);
    if (result <= -static_cast<long long>(slots / 2))
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
    auto projected = RealColumns(
        {{layer.tokens, layer.ffn_width}}, slot_count, Layout::kStrictColumn);
    if (!projected.ok()) return projected.status();
    const auto stride = NextPowerOfTwo(layer.tokens);
    const auto columns = slot_count / stride;
    auto inverse_plan = MultiHeadPackingPlan::TokenInverseBroadcast(layer.tokens, slot_count);
    if (!inverse_plan.ok()) return inverse_plan.status();
    auto inverse = inverse_plan.take_value();

    return FfnGeluPackingPlan{centered.take_value(), std::move(inverse),
                              projected.take_value(),
                              stride, columns};
}

Result<std::shared_ptr<const FfnGeluPackingPlan>> FfnGeluPackingPlanner::GetShared(
    const TransformerLayerConfig &layer, std::size_t slot_count)
{
    using Key = std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>;
    struct Entry { Key key; std::shared_ptr<const FfnGeluPackingPlan> plan; std::size_t bytes; };
    thread_local std::list<Entry> entries;
    thread_local std::size_t resident_bytes = 0;
    const Key key{layer.tokens, layer.model_width, layer.ffn_width, slot_count};
    for (auto it = entries.begin(); it != entries.end(); ++it)
        if (it->key == key) {
            auto plan = it->plan;
            entries.splice(entries.begin(), entries, it);
            return plan;
        }
    auto built = Build(layer, slot_count);
    if (!built.ok()) return built.status();
    auto plan = std::make_shared<const FfnGeluPackingPlan>(built.take_value());
    std::size_t bytes = sizeof(FfnGeluPackingPlan);
    for (const auto *packing : {&plan->centered, &plan->inverse_broadcast,
                                &plan->projected}) {
        bytes += packing->logical_locations.capacity() * sizeof(packing->logical_locations.front());
        for (const auto &locations : packing->logical_locations)
            bytes += locations.capacity() * sizeof(locations.front());
    }
    constexpr std::size_t kBudget = 128U * 1024U * 1024U;
    if (bytes <= kBudget) {
        while (!entries.empty() && (entries.size() >= 8 || resident_bytes + bytes > kBudget)) {
            resident_bytes -= entries.back().bytes;
            entries.pop_back();
        }
        entries.push_front({key, plan, bytes});
        resident_bytes += bytes;
    }
    return plan;
}

Result<void> FfnGeluLevelPlanner::Validate(const HeParameterConfig &parameters)
{
    const std::size_t depth = kFfnSuffixDepth;
    if (parameters.coefficient_modulus_bits.size() < depth + 1 || parameters.declared_depth < depth)
        return Status::FailedPrecondition("FFN/GeLU profile is shorter than the five-level real-Wu suffix");
    if (parameters.terminal_prime_count != 1)
        return Status::FailedPrecondition("FFN/GeLU conversion needs a single-prime terminal level");
    if (parameters.input_scale_bits != 45 || parameters.default_scale_bits != 45 || parameters.output_scale_bits != 46)
        return Status::FailedPrecondition("FFN/GeLU requires input/default/output scales 45/45/46");
    const auto &bits = parameters.coefficient_modulus_bits;
    if (bits[0] < 58 || bits[1] < 58 || bits[depth] < 58)
        return Status::FailedPrecondition("FFN/GeLU terminal/coefficient/LN primes must have at least 58 bits");
    for (std::size_t i = 2; i < depth; ++i)
        if (std::abs(bits[i] - 46) > 1)
            return Status::FailedPrecondition("FFN/GeLU polynomial/projection suffix primes must be 46 bits");
    return {};
}

namespace {

struct PreparedFfnAffine final : PreparedBlock2Program {
    std::string profile;
    seal::parms_id_type input_id{};
    double input_scale{}, output_scale{}, residual_scale{};
    struct Channel {
        std::shared_ptr<const seal::Plaintext> plus, minus, bias;
    };
    std::vector<Channel> channels;
    std::vector<double> gamma, beta, residual_gamma, residual_beta;
    std::vector<std::shared_ptr<const seal::Plaintext>> residual_gains, residual_biases;
};

} // namespace

Result<SealCipherTensor> FfnLayerNormTailKernel::Evaluate(
    const SealCipherTensor &centered,
    const SealCipherTensor &inverse_stddev,
    const DenseVectorParameter &gamma,
    const DenseVectorParameter &beta,
    const TransformerLayerConfig &layer,
    const HeContext &context,
    FfnGeluExecutionReport *report,
    std::size_t he_workers,
    const GammaFusedLayerNormParameters *original_affine,
    SealCipherTensor *normalized_residual)
{
    if ((original_affine == nullptr) != (normalized_residual == nullptr))
        return Status::Invalid("LN2 residual requires original affine and output together");
    if (original_affine && (original_affine->gamma.values.size() != layer.model_width ||
                            original_affine->beta.values.size() != layer.model_width))
        return Status::Invalid("LN2 original affine size mismatch");
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
    auto plan = FfnGeluPackingPlanner::GetShared(layer, context.slot_count());
    if (!plan.ok()) return plan.status();
    auto same = SamePacking(centered.physical_packing,
                            plan.value()->centered);
    if (!same.ok()) return same.status();
    same = SamePacking(inverse_stddev.physical_packing, plan.value()->inverse_broadcast);
    if (!same.ok()) return same.status();
    const double input_scale = std::ldexp(
        1.0, context.parameters.input_scale_bits);
    const auto terminal_level =
        context.parameters.coefficient_modulus_bits.size() -
        context.parameters.terminal_prime_count;
    const auto kFfnGeluDepth = kFfnSuffixDepth;
    if (terminal_level < kFfnGeluDepth)
        return Status::FailedPrecondition(
            "FFN/GeLU suffix needs five remaining CKKS levels");
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

    const auto cache = ModelPlaintextCacheScope::Current();
    const auto key = cache.second + "/ffn/ln2/" + layer.id;
    const auto input_id = centered.ciphertexts.front().parms_id();
    const auto next = context.context->get_context_data(input_id)->next_context_data();
    if (!next) return Status::FailedPrecondition("FFN LayerNorm has no next level");
    const int affine_bits = context.parameters.default_scale_bits + 15;
    const double target = std::ldexp(1.0, affine_bits);
    const int residual_bits = context.parameters.output_scale_bits;
    const double residual_target = std::ldexp(1.0, residual_bits);
    std::optional<HePackingPlan> residual_plan;
    if (normalized_residual) {
        auto built = FfnGeluPackingPlanner::RealColumns(
            {{layer.tokens, layer.model_width}}, context.slot_count(), Layout::kStrictColumn);
        if (!built.ok()) return built.status();
        residual_plan = built.take_value();
    }
    std::shared_ptr<const PreparedFfnAffine> program;
    if (cache.first && !cache.first->preparing_plaintexts()) {
        auto loaded = cache.first->GetPreparedBlock2Program(key);
        if (!loaded.ok()) return loaded.status();
        program = std::dynamic_pointer_cast<const PreparedFfnAffine>(loaded.value());
        if (!program || program->profile != context.profile || program->input_id != input_id ||
            !CloseScale(program->input_scale, input_scale) || program->output_scale != target ||
            program->channels.size() != centered.ciphertexts.size() ||
            program->gamma != gamma.values || program->beta != beta.values ||
            program->residual_scale != residual_target ||
            program->residual_gamma != (original_affine ? original_affine->gamma.values : std::vector<double>{}) ||
            program->residual_beta != (original_affine ? original_affine->beta.values : std::vector<double>{}))
            return Status::FailedPrecondition("prepared FFN LayerNorm contract mismatch");
    } else {
        auto compiled = std::make_shared<PreparedFfnAffine>();
        compiled->profile = context.profile;
        compiled->input_id = input_id;
        compiled->input_scale = input_scale;
        compiled->output_scale = target;
        compiled->residual_scale = residual_target;
        compiled->channels.resize(centered.ciphertexts.size());
        compiled->gamma = gamma.values;
        compiled->beta = beta.values;
        seal::CKKSEncoder encoder(*context.context);
        const double plain_scale = target * DroppedPrime(context, input_id) /
                                   (input_scale * input_scale);
        if (!std::isfinite(plain_scale) || plain_scale < 2.0)
            return Status::FailedPrecondition("LayerNorm gamma plaintext scale is too small");
        const auto columns = plan.value()->columns_per_component;
        std::vector<Complex> plus(context.slot_count()), minus(context.slot_count()),
            bias_slots(context.slot_count());
        for (std::size_t group = 0; group < compiled->channels.size(); ++group) {
            std::fill(plus.begin(), plus.end(), Complex{});
            std::fill(minus.begin(), minus.end(), Complex{});
            std::fill(bias_slots.begin(), bias_slots.end(), Complex{});
            bool needs_conjugate = false;
            for (std::size_t token = 0; token < layer.tokens; ++token)
                for (std::size_t local = 0; local < columns; ++local) {
                    const auto slot = local * plan.value()->token_stride + token;
                    const auto rf = group * 2 * columns + local, im = rf + columns;
                    if (rf >= layer.model_width) continue;
                    const double gr = gamma.values[rf];
                    // An unused imaginary channel can use the real gain: its
                    // padding is never consumed by Wu, so no split is needed.
                    const double gi = im < layer.model_width ? gamma.values[im] : gr;
                    plus[slot] = 0.5 * (gr + gi);
                    minus[slot] = 0.5 * (gr - gi);
                    needs_conjugate |= gr != gi;
                    bias_slots[slot].real(beta.values[rf]);
                    if (im < layer.model_width) bias_slots[slot].imag(beta.values[im]);
                }
            auto &channel = compiled->channels[group];
            auto encoded = CompileModelPlaintext(encoder, plus, input_id, plain_scale,
                                                  "ffn.ln2.gamma.plus");
            if (!encoded.ok()) return encoded.status();
            channel.plus = encoded.take_value();
            if (needs_conjugate) {
                encoded = CompileModelPlaintext(encoder, minus, input_id, plain_scale,
                                                "ffn.ln2.gamma.minus");
                if (!encoded.ok()) return encoded.status();
                channel.minus = encoded.take_value();
            }
            encoded = CompileModelPlaintext(encoder, bias_slots, next->parms_id(), target,
                                            "ffn.ln2.beta");
            if (!encoded.ok()) return encoded.status();
            channel.bias = encoded.take_value();
        }
        if (original_affine) {
            compiled->residual_gamma = original_affine->gamma.values;
            compiled->residual_beta = original_affine->beta.values;
            const double residual_scale = residual_target * DroppedPrime(context, input_id) /
                                          (input_scale * input_scale);
            for (std::size_t ct = 0; ct < residual_plan->ciphertext_count; ++ct) {
                std::fill(plus.begin(), plus.end(), Complex{});
                std::fill(bias_slots.begin(), bias_slots.end(), Complex{});
                bool nonzero = false;
                for (std::size_t local = 0; local < columns; ++local) {
                    const auto feature = ct * columns + local;
                    if (feature >= layer.model_width) break;
                    const auto gain = original_affine->gamma.values[feature];
                    nonzero |= gain != 0.0;
                    for (std::size_t token = 0; token < layer.tokens; ++token) {
                        const auto slot = local * plan.value()->token_stride + token;
                        // Extract a or b and apply original gamma before the
                        // same rescale as the fused Wu branch. No extra level.
                        plus[slot] = ct % 2 == 0 ? Complex{0.5 * gain, 0.0}
                                                : Complex{0.0, -0.5 * gain};
                        bias_slots[slot] = original_affine->beta.values[feature];
                    }
                }
                std::shared_ptr<const seal::Plaintext> gain;
                if (nonzero) {
                    auto encoded = CompileModelPlaintext(encoder, plus, input_id,
                        residual_scale, "ffn.ln2.residual.gamma");
                    if (!encoded.ok()) return encoded.status();
                    gain = encoded.take_value();
                }
                compiled->residual_gains.push_back(std::move(gain));
                auto encoded = CompileModelPlaintext(encoder, bias_slots, next->parms_id(),
                    residual_target, "ffn.ln2.residual.beta");
                if (!encoded.ok()) return encoded.status();
                compiled->residual_biases.push_back(encoded.take_value());
            }
        }
        program = compiled;
        if (cache.first) {
            auto stored = cache.first->PutPreparedBlock2Program(key, compiled);
            if (!stored.ok()) return stored.status();
        }
    }
    SealCipherTensor result;
    result.parameter_profile = context.profile;
    result.logical_spec = centered.logical_spec;
    result.physical_packing = centered.physical_packing;
    result.ciphertexts.resize(centered.ciphertexts.size());
    if (normalized_residual) {
        normalized_residual->parameter_profile = context.profile;
        normalized_residual->logical_spec = centered.logical_spec;
        normalized_residual->logical_spec.layout = Layout::kStrictColumn;
        normalized_residual->logical_spec.he_region = HeRegionId::kRegion5;
        normalized_residual->logical_spec.planned_level = input_level + 1;
        normalized_residual->logical_spec.planned_scale_bits = residual_bits;
        normalized_residual->physical_packing = *residual_plan;
        normalized_residual->ciphertexts.resize(residual_plan->ciphertext_count);
    }
    if (cache.first && cache.first->preparing_plaintexts()) {
        for (auto &ct : result.ciphertexts) { ct.parms_id() = next->parms_id(); ct.scale() = target; }
        if (normalized_residual) for (auto &ct : normalized_residual->ciphertexts) {
            ct.parms_id() = next->parms_id(); ct.scale() = residual_target;
        }
    } else {
        std::vector<FfnGeluExecutionReport> reports(result.ciphertexts.size());
        auto evaluated = detail::ParallelRanges(result.ciphertexts.size(),
            he_workers, 1,
            [&](std::size_t begin, std::size_t end) -> Result<void> {
                seal::Evaluator evaluator(*context.context);
                for (std::size_t group = begin; group < end; ++group) {
                    auto &value = result.ciphertexts[group];
                    auto &counts = reports[group];
                    evaluator.multiply(centered.ciphertexts[group], inverse_ct, value);
                    evaluator.relinearize_inplace(value, context.relinearization_keys);
                    ++counts.ciphertext_multiplications;
                    ++counts.relinearizations;
                    const auto &channel = program->channels[group];
                    seal::Ciphertext conjugated;
                    if (channel.minus || normalized_residual) {
                        evaluator.complex_conjugate(value, context.galois_keys, conjugated);
                        ++counts.conjugations;
                    }
                    if (normalized_residual) {
                        for (std::size_t component = 0; component < 2; ++component) {
                            const auto ct = 2 * group + component;
                            if (ct >= normalized_residual->ciphertexts.size()) break;
                            auto &output = normalized_residual->ciphertexts[ct];
                            if (program->residual_gains[ct]) {
                                if (component == 0) evaluator.add(value, conjugated, output);
                                else evaluator.sub(value, conjugated, output);
                                evaluator.multiply_plain_inplace(output, *program->residual_gains[ct]);
                                ++counts.ciphertext_plain_multiplications;
                            } else {
                                seal::Encryptor encryptor(*context.context, context.public_key);
                                encryptor.encrypt_zero(input_id, output);
                                output.scale() = residual_target * DroppedPrime(context, input_id);
                            }
                            evaluator.rescale_to_next_inplace(output);
                            auto normalized = NormalizeScale(output, residual_target);
                            if (!normalized.ok()) return normalized.status();
                            evaluator.add_plain_inplace(output, *program->residual_biases[ct]);
                            ++counts.rescales;
                        }
                    }
                    if (channel.minus) {
                        evaluator.multiply_plain_inplace(conjugated, *channel.minus);
                        evaluator.multiply_plain_inplace(value, *channel.plus);
                        evaluator.add_inplace(value, conjugated);
                        counts.ciphertext_plain_multiplications += 2;
                    } else {
                        evaluator.multiply_plain_inplace(value, *channel.plus);
                        ++counts.ciphertext_plain_multiplications;
                    }
                    evaluator.rescale_to_next_inplace(value);
                    auto normalized = NormalizeScale(value, target);
                    if (!normalized.ok()) return normalized.status();
                    ++counts.rescales;
                    evaluator.add_plain_inplace(value, *channel.bias);
                }
                return {};
            }, detail::EstimatedHeWorkerBytes(context.parameters.polynomial_degree,
                context.parameters.coefficient_modulus_bits.size(), 8));
        if (!evaluated.ok()) return evaluated.status();
        if (report) for (const auto &counts : reports) {
            report->ciphertext_multiplications += counts.ciphertext_multiplications;
            report->relinearizations += counts.relinearizations;
            report->ciphertext_plain_multiplications += counts.ciphertext_plain_multiplications;
            report->conjugations += counts.conjugations;
            report->rescales += counts.rescales;
        }
    }
    result.logical_spec.he_region = HeRegionId::kRegion5;
    result.logical_spec.he_parameter_profile = context.profile;
    result.logical_spec.planned_level = input_level + 1;
    result.logical_spec.planned_scale_bits = affine_bits;
    return result;
}

namespace {

struct ColumnBsgsSplit {
    std::size_t baby{}, giant{}, stride{}, columns{};
    bool raw_complex{};
};

Result<ColumnBsgsSplit> ColumnSplit(const HePackingPlan &input,
                                   const HePackingPlan &output)
{
    if (input.logical_shape.dimensions.size() != 2 ||
        output.logical_shape.dimensions.size() != 2 ||
        input.logical_shape.dimensions[0] != output.logical_shape.dimensions[0] ||
        input.slot_count != output.slot_count)
        return Status::Invalid("column BSGS packing shapes differ");
    const auto check = [](const HePackingPlan &plan) -> Result<bool> {
        bool real = true;
        for (const auto &locations : plan.logical_locations) {
            if (locations.size() != 1)
                return Status::Invalid("column BSGS requires one physical lane");
            real &= locations.front().component == ComplexLane::kReal;
        }
        auto expected = real
            ? FfnGeluPackingPlanner::RealColumns(plan.logical_shape, plan.slot_count, plan.layout)
            : FfnGeluPackingPlanner::ComplexColumns(plan.logical_shape, plan.slot_count, plan.layout);
        if (!expected.ok()) return expected.status();
        auto same = SamePacking(plan, expected.value());
        if (!same.ok()) return same.status();
        // A complex layout can have no logical imaginary features in its
        // tail; those unused lanes still must be excluded from the product.
        return real && plan.layout == Layout::kStrictColumn;
    };
    auto in_real = check(input), out_real = check(output);
    if (!in_real.ok()) return in_real.status();
    if (!out_real.ok()) return out_real.status();
    const auto stride = NextPowerOfTwo(input.logical_shape.dimensions[0]);
    const auto columns = input.slot_count / stride;
    const bool raw = !in_real.value() && out_real.value();
    std::vector<bool> used_bases(2 * input.ciphertext_count);
    for (std::size_t feature = 0; feature < input.logical_shape.dimensions[1]; ++feature) {
        const auto &loc = input.Locations(feature).front();
        used_bases[2 * loc.ciphertext + (raw || loc.component == ComplexLane::kReal ? 0 : 1)] = true;
    }
    const auto bases = static_cast<std::size_t>(std::count(used_bases.begin(), used_bases.end(), true));
    std::size_t best = 1;
    double cost = std::numeric_limits<double>::infinity();
    for (std::size_t baby = 1; baby <= columns; baby <<= 1U) {
        const auto candidate = ProjectionBsgsCost(bases, output.ciphertext_count, columns,
            baby, input.slot_count, raw, !in_real.value() && !raw);
        if (candidate < cost) { cost = candidate; best = baby; }
    }
    if (!std::isfinite(cost)) return Status::Invalid("column projection baby override must divide its columns");
    return ColumnBsgsSplit{best, columns / best, stride, columns, raw};
}

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
    HeKernelOperationCounts *counts,
    std::size_t he_workers)
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
    const auto cache = ModelPlaintextCacheScope::Current();
    const auto prepared_key = cache.second + "/ffn/" + weights.spec.name;
    if (cache.first && !cache.first->preparing_plaintexts()) {
        auto prepared = cache.first->GetPreparedProjection(prepared_key);
        if (!prepared.ok()) return prepared.status();
        if (!SamePreparedPacking(prepared.value()->output_packing,
                                 output_packing) ||
            prepared.value()->output_scale_bits != output_scale_bits)
            return Status::FailedPrecondition(
                "prepared column projection output contract mismatch");
        PreparedProjectionCounts prepared_counts;
        auto ciphertexts = EvaluatePreparedProjection(
            *prepared.value(), input, context, &prepared_counts, nullptr, he_workers);
        if (!ciphertexts.ok()) return ciphertexts.status();
        if (counts) {
            counts->rotations += prepared_counts.rotations;
            counts->conjugations += prepared_counts.conjugations;
            counts->ciphertext_plain_multiplications +=
                prepared_counts.ciphertext_plain_multiplications;
            counts->rescales += prepared_counts.rescales;
        }
        SealCipherTensor result;
        result.parameter_profile = context.profile;
        result.ciphertexts = ciphertexts.take_value();
        result.logical_spec.shape = output_packing.logical_shape;
        result.logical_spec.domain = Domain::kHeCiphertext;
        result.logical_spec.layout = output_packing.layout;
        result.logical_spec.numeric = input.logical_spec.numeric;
        result.logical_spec.he_region = output_region;
        result.logical_spec.he_parameter_profile = context.profile;
        auto input_level = LevelOf(
            context, input.ciphertexts.front().parms_id());
        if (!input_level.ok()) return input_level.status();
        result.logical_spec.planned_level = input_level.value() + 1;
        result.logical_spec.planned_scale_bits = output_scale_bits;
        result.physical_packing = prepared.value()->output_packing;
        return result;
    }
    auto split_result = ColumnSplit(input.physical_packing, output_packing);
    if (!split_result.ok()) return split_result.status();
    const auto split = split_result.value();
    const auto input_id = input.ciphertexts.front().parms_id();
    const auto data = context.context->get_context_data(input_id);
    const auto next = data->next_context_data();
    if (!next) return Status::FailedPrecondition("column projection has no next level");
    const double target = std::ldexp(1.0, output_scale_bits);
    const double plain_scale = target * DroppedPrime(context, input_id) /
                               input.ciphertexts.front().scale();
    auto prepared = std::make_shared<PreparedProjectionPlan>();
    prepared->profile = context.profile;
    prepared->input_parms_id = input_id;
    prepared->input_scale = input.ciphertexts.front().scale();
    prepared->input_packing = input.physical_packing;
    prepared->output_packing = output_packing;
    prepared->output_scale_bits = output_scale_bits;
    prepared->scale_tolerance = 1e-5;
    prepared->reuse_baby_rotations = true;
    prepared->raw_complex_input = split.raw_complex;
    prepared->realify_output = split.raw_complex;
    prepared->bias_after_rescale = true;
    prepared->bias_plaintexts.resize(output_packing.ciphertext_count);
    prepared->real_input = input.logical_spec.layout == Layout::kStrictColumn &&
        std::all_of(input.physical_packing.logical_locations.begin(),
        input.physical_packing.logical_locations.end(), [](const auto &locations) {
            return locations.front().component == ComplexLane::kReal;
        });
    if (split.raw_complex) {
        // Exact scale-1 phase: both Wd weight channels stay real-only.
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
    }
    const auto absent = std::numeric_limits<std::size_t>::max();
    using Columns = std::vector<std::size_t>;
    std::vector<std::array<Columns, 2>> sources(input.ciphertexts.size()),
        destinations(output_packing.ciphertext_count);
    for (auto *table : {&sources, &destinations})
        for (auto &ct : *table) for (auto &lane : ct) lane.assign(split.columns, absent);
    for (std::size_t feature = 0; feature < input_width; ++feature) {
        const auto &loc = input.physical_packing.Locations(feature).front();
        sources[loc.ciphertext][loc.component == ComplexLane::kReal ? 0 : 1]
            [loc.slot / split.stride] = feature;
    }
    for (std::size_t feature = 0; feature < output_width; ++feature) {
        const auto &loc = output_packing.Locations(feature).front();
        destinations[loc.ciphertext][loc.component == ComplexLane::kReal ? 0 : 1]
            [loc.slot / split.stride] = feature;
    }
    seal::CKKSEncoder encoder(*context.context);
    // Stream one diagonal mask at a time. Never materialize all slot vectors.
    std::vector<Complex> mask(context.slot_count());
    for (std::size_t out = 0; out < destinations.size(); ++out) {
        for (std::size_t giant = 0; giant < split.giant; ++giant) {
            const auto giant_columns = giant * split.baby;
            PreparedProjectionGroup group{out, CanonicalRotation(
                giant_columns * split.stride, context.slot_count()), {}};
            for (std::size_t source = 0; source < sources.size(); ++source)
                for (std::size_t component = 0; component < 2; ++component)
                    for (std::size_t baby = 0; baby < split.baby; ++baby) {
                        std::fill(mask.begin(), mask.end(), Complex{});
                        bool nonzero = false;
                        for (std::size_t column = 0; column < split.columns; ++column) {
                            const auto feature = sources[source][component]
                                [(column + giant_columns + baby) % split.columns];
                            if (feature == absent) continue;
                            for (std::size_t lane = 0; lane < 2; ++lane) {
                                const auto destination = destinations[out][lane][column];
                                if (destination == absent) continue;
                                const auto weight = weights.row_major[feature * output_width + destination];
                                if (weight == 0.0) continue;
                                for (std::size_t token = 0; token < tokens; ++token) {
                                    const auto slot = ((column + giant_columns) % split.columns) *
                                                      split.stride + token;
                                    Complex factor;
                                    if (split.raw_complex)
                                        factor = component == 0 ? Complex{0.5, 0.0}
                                            : Complex{slot % 2 == 0 ? -0.5 : 0.5, 0.0};
                                    else factor = (prepared->real_input ? Complex{1.0, 0.0}
                                        : ExtractionFactor(component == 0 ? ComplexLane::kReal : ComplexLane::kImaginary)) *
                                        DestinationFactor(lane == 0 ? ComplexLane::kReal : ComplexLane::kImaginary);
                                    mask[slot] += weight * factor;
                                }
                                nonzero = true;
                            }
                        }
                        if (!nonzero) continue;
                        auto encoded = CompileModelPlaintext(encoder, mask, input_id,
                            plain_scale, "ffn.projection.diagonal");
                        if (!encoded.ok()) return encoded.status();
                        PreparedProjectionTerm term{source,
                            split.raw_complex || component == 0 ? ComplexLane::kReal : ComplexLane::kImaginary,
                            CanonicalRotation(baby * split.stride, context.slot_count()), encoded.take_value()};
                        if (split.raw_complex && component == 1)
                            group.phase_terms.push_back(std::move(term));
                        else group.terms.push_back(std::move(term));
                    }
            if (!group.terms.empty() || !group.phase_terms.empty())
                prepared->groups.push_back(std::move(group));
        }
        if (!bias) continue;
        std::fill(mask.begin(), mask.end(), Complex{});
        for (std::size_t lane = 0; lane < 2; ++lane)
            for (std::size_t column = 0; column < split.columns; ++column) {
                const auto feature = destinations[out][lane][column];
                if (feature == absent) continue;
                for (std::size_t token = 0; token < tokens; ++token)
                    mask[column * split.stride + token] += bias->values[feature] *
                        DestinationFactor(lane == 0 ? ComplexLane::kReal : ComplexLane::kImaginary);
            }
        auto encoded = CompileModelPlaintext(encoder, mask, next->parms_id(),
                                              target, "ffn.projection.bias");
        if (!encoded.ok()) return encoded.status();
        prepared->bias_plaintexts[out] = encoded.take_value();
    }
    const bool preparing = cache.first && cache.first->preparing_plaintexts();
    std::vector<seal::Ciphertext> output;
    if (preparing) {
        output.resize(output_packing.ciphertext_count);
        for (auto &ct : output) { ct.parms_id() = next->parms_id(); ct.scale() = target; }
    } else {
        PreparedProjectionCounts evaluated_counts;
        auto evaluated = EvaluatePreparedProjection(*prepared, input, context,
            &evaluated_counts, nullptr, he_workers);
        if (!evaluated.ok()) return evaluated.status();
        output = evaluated.take_value();
        if (counts) {
            counts->rotations += evaluated_counts.rotations;
            counts->conjugations += evaluated_counts.conjugations;
            counts->ciphertext_plain_multiplications += evaluated_counts.ciphertext_plain_multiplications;
            counts->rescales += evaluated_counts.rescales;
        }
    }
    if (cache.first) {
        auto stored = cache.first->PutPreparedProjection(prepared_key, prepared);
        if (!stored.ok()) return stored.status();
    }
    SealCipherTensor result;
    result.parameter_profile = context.profile;
    result.ciphertexts = std::move(output);
    result.logical_spec = input.logical_spec;
    result.logical_spec.shape = output_packing.logical_shape;
    result.logical_spec.layout = output_packing.layout;
    result.logical_spec.he_region = output_region;
    result.logical_spec.he_parameter_profile = context.profile;
    auto level = LevelOf(context, input_id);
    if (!level.ok()) return level.status();
    result.logical_spec.planned_level = level.value() + 1;
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
    auto split = ColumnSplit(input_packing, output_packing);
    if (!split.ok()) return split.status();
    std::set<int> steps{0}; // zero requests the conjugation automorphism.
    // Tail matrices need only the offsets which can contain a coefficient.
    // Generating all F offsets for tiny widths wastes large Galois keys.
    const auto in_width = input_packing.logical_shape.dimensions[1];
    const auto out_width = output_packing.logical_shape.dimensions[1];
    std::vector<bool> sources(split.value().columns), destinations(split.value().columns);
    for (std::size_t i = 0; i < in_width; ++i)
        sources[input_packing.Locations(i).front().slot / split.value().stride] = true;
    for (std::size_t i = 0; i < out_width; ++i)
        destinations[output_packing.Locations(i).front().slot / split.value().stride] = true;
    for (std::size_t source = 0; source < sources.size(); ++source) {
        if (!sources[source]) continue;
        for (std::size_t destination = 0; destination < destinations.size(); ++destination) {
            if (!destinations[destination]) continue;
            const auto diagonal = (source + split.value().columns - destination) % split.value().columns;
            const auto baby = diagonal % split.value().baby;
            const auto giant = diagonal - baby;
            if (baby) steps.insert(CanonicalRotation(baby * split.value().stride, input_packing.slot_count));
            if (giant) steps.insert(CanonicalRotation(giant * split.value().stride, input_packing.slot_count));
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
    FfnGeluExecutionReport *report,
    std::size_t he_workers)
{
    if (input.logical_spec.shape !=
            TensorShape{{layer.tokens, layer.model_width}} ||
        input.logical_spec.layout != Layout::kComplexStrictColumn)
        return Status::Invalid("Wu projection contract mismatch");
    auto plans = FfnGeluPackingPlanner::GetShared(layer, context.slot_count());
    if (!plans.ok()) return plans.status();
    HeKernelOperationCounts counts;
    auto result = ComplexColumnProjectionKernel::Evaluate(
        input, weights, bias, plans.value()->projected,
        HeRegionId::kRegion5, context.parameters.output_scale_bits,
        context, &counts, he_workers);
    if (report) {
        report->rotations += counts.rotations;
        report->conjugations += counts.conjugations;
        report->ciphertext_plain_multiplications +=
            counts.ciphertext_plain_multiplications;
        report->rescales += counts.rescales;
    }
    return result;
}

namespace {
struct PreparedRealify final : PreparedBlock2Program {
    std::string profile;
    seal::parms_id_type input_id{};
    double input_scale{}, output_scale{}, residual_scale{};
    std::shared_ptr<const seal::Plaintext> real, imaginary;
};
} // namespace

Result<SealCipherTensor> ComplexColumnRealifyKernel::Evaluate(
    const SealCipherTensor &input,
    const HePackingPlan &real_output_packing,
    int output_scale_bits,
    const HeContext &context,
    HeKernelOperationCounts *counts,
    std::size_t he_workers)
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
    const auto input_id = input.ciphertexts.front().parms_id();
    const auto next = context.context->get_context_data(input_id)->next_context_data();
    if (!next) return Status::FailedPrecondition("realify has no next CKKS level");
    const double wanted = std::ldexp(1.0, output_scale_bits);
    const double source_scale = input.ciphertexts.front().scale();
    for (const auto &ct : input.ciphertexts)
        if (ct.parms_id() != input_id || !CloseScale(ct.scale(), source_scale))
            return Status::FailedPrecondition("realify input level/scale is inconsistent");
    const auto cache = ModelPlaintextCacheScope::Current();
    const auto key = cache.second + "/ffn/realify/" + std::to_string(output_scale_bits);
    std::shared_ptr<const PreparedRealify> program;
    if (cache.first && !cache.first->preparing_plaintexts()) {
        auto loaded = cache.first->GetPreparedBlock2Program(key);
        if (!loaded.ok()) return loaded.status();
        program = std::dynamic_pointer_cast<const PreparedRealify>(loaded.value());
        if (!program || program->profile != context.profile || program->input_id != input_id ||
            !CloseScale(program->input_scale, source_scale) || program->output_scale != wanted)
            return Status::FailedPrecondition("prepared realify contract mismatch");
    } else {
        auto compiled = std::make_shared<PreparedRealify>();
        compiled->profile = context.profile;
        compiled->input_id = input_id;
        compiled->input_scale = source_scale;
        compiled->output_scale = wanted;
        seal::CKKSEncoder encoder(*context.context);
        const double plain_scale = wanted * DroppedPrime(context, input_id) / source_scale;
        auto encoded = CompileModelPlaintext(encoder,
            std::vector<Complex>(context.slot_count(), Complex{0.5, 0.0}),
            input_id, plain_scale, "ffn.realify.constant");
        if (!encoded.ok()) return encoded.status();
        compiled->real = encoded.take_value();
        encoded = CompileModelPlaintext(encoder,
            std::vector<Complex>(context.slot_count(), Complex{0.0, -0.5}),
            input_id, plain_scale, "ffn.realify.constant");
        if (!encoded.ok()) return encoded.status();
        compiled->imaginary = encoded.take_value();
        program = compiled;
        if (cache.first) {
            auto stored = cache.first->PutPreparedBlock2Program(key, compiled);
            if (!stored.ok()) return stored.status();
        }
    }
    std::vector<seal::Ciphertext> output(real_output_packing.ciphertext_count);
    if (cache.first && cache.first->preparing_plaintexts()) {
        for (auto &ct : output) { ct.parms_id() = next->parms_id(); ct.scale() = wanted; }
    } else {
        std::vector<HeKernelOperationCounts> reports(input.ciphertexts.size());
        auto evaluated = detail::ParallelRanges(input.ciphertexts.size(),
            he_workers, 1,
            [&](std::size_t begin, std::size_t end) -> Result<void> {
                seal::Evaluator evaluator(*context.context);
                for (std::size_t group = begin; group < end; ++group) {
                    const auto &source = input.ciphertexts[group];
                    seal::Ciphertext conjugated;
                    evaluator.complex_conjugate(source, context.galois_keys, conjugated);
                    ++reports[group].conjugations;
                    auto &real = output[2 * group];
                    evaluator.add(source, conjugated, real);
                    evaluator.multiply_plain_inplace(real, *program->real);
                    evaluator.rescale_to_next_inplace(real);
                    auto normalized = NormalizeScale(real, wanted);
                    if (!normalized.ok()) return normalized.status();
                    ++reports[group].ciphertext_plain_multiplications;
                    ++reports[group].rescales;
                    if ((2 * group + 1) * columns >= width) continue;
                    auto &imaginary = output[2 * group + 1];
                    evaluator.sub(source, conjugated, imaginary);
                    evaluator.multiply_plain_inplace(imaginary, *program->imaginary);
                    evaluator.rescale_to_next_inplace(imaginary);
                    normalized = NormalizeScale(imaginary, wanted);
                    if (!normalized.ok()) return normalized.status();
                    ++reports[group].ciphertext_plain_multiplications;
                    ++reports[group].rescales;
                }
                return {};
            }, detail::EstimatedHeWorkerBytes(context.parameters.polynomial_degree,
                context.parameters.coefficient_modulus_bits.size(), 8));
        if (!evaluated.ok()) return evaluated.status();
        if (counts) for (const auto &local : reports) {
            counts->conjugations += local.conjugations;
            counts->ciphertext_plain_multiplications += local.ciphertext_plain_multiplications;
            counts->rescales += local.rescales;
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

namespace {

struct PreparedGeluPolynomial final : PreparedBlock2Program {
    std::string profile;
    seal::parms_id_type input_id{};
    double scale{};
    std::array<double, 5> coefficients{};
    // x4, x3, x2, x and final constant, respectively.
    std::array<std::shared_ptr<const seal::Plaintext>, 5> plaintexts;
    std::shared_ptr<const seal::Plaintext> conversion_phase;
};

HePackingPlan GeluBranchPacking(const HePackingPlan &real)
{
    auto result = real;
    const auto count = real.logical_locations.size();
    result.logical_shape.dimensions.insert(result.logical_shape.dimensions.begin(), 2);
    result.layout = Layout::kFfn;
    result.logical_locations.resize(2 * count);
    for (std::size_t logical = 0; logical < count; ++logical) {
        result.logical_locations[count + logical] = real.logical_locations[logical];
        result.logical_locations[count + logical].front().component = ComplexLane::kImaginary;
    }
    return result;
}

} // namespace

Result<GeluPolynomialOutputs> GeluPolynomialKernel::Evaluate(
    const SealCipherTensor &input,
    const GeluConfig &config,
    const HeContext &context,
    FfnGeluExecutionReport *report,
    std::size_t he_workers)
{
    auto valid = input.Validate(context);
    if (!valid.ok()) return valid.status();
    if (input.logical_spec.shape.dimensions.size() != 2 ||
        std::any_of(input.physical_packing.logical_locations.begin(),
                    input.physical_packing.logical_locations.end(), [](const auto &locations) {
            return locations.size() != 1 || locations.front().component != ComplexLane::kReal;
        })) return Status::Invalid("GeLU polynomial requires real-only input lanes");
    const double target = std::ldexp(1.0, context.parameters.output_scale_bits);
    const auto terminal_level = context.parameters.coefficient_modulus_bits.size() -
                                context.parameters.terminal_prime_count;
    constexpr std::size_t kPolynomialDepth = 3;
    if (terminal_level < kPolynomialDepth)
        return Status::FailedPrecondition("GeLU polynomial needs three remaining CKKS levels");
    const auto polynomial_input_level = terminal_level - kPolynomialDepth;
    for (const auto &ct : input.ciphertexts) {
        auto level = LevelOf(context, ct.parms_id());
        if (!level.ok()) return level.status();
        if (level.value() != polynomial_input_level || !CloseScale(ct.scale(), target))
            return Status::FailedPrecondition("GeLU polynomial input is not at its suffix start");
    }
    const auto terminal_id = LevelId(context, terminal_level);
    const auto coefficients_id = LevelId(context, terminal_level - 1);
    if (!terminal_id.ok()) return terminal_id.status();
    if (!coefficients_id.ok()) return coefficients_id.status();
    const auto cache = ModelPlaintextCacheScope::Current();
    const auto key = cache.second + "/ffn/gelu";
    const std::array<double, 5> coefficients{config.a, config.b, config.c, config.d, config.e};
    for (const auto coefficient : coefficients)
        if (!std::isfinite(coefficient)) return Status::Invalid("GeLU coefficients must be finite");
    std::shared_ptr<const PreparedGeluPolynomial> program;
    if (cache.first && !cache.first->preparing_plaintexts()) {
        auto loaded = cache.first->GetPreparedBlock2Program(key);
        if (!loaded.ok()) return loaded.status();
        program = std::dynamic_pointer_cast<const PreparedGeluPolynomial>(loaded.value());
        if (!program || program->profile != context.profile || program->scale != target ||
            program->input_id != input.ciphertexts.front().parms_id() || program->coefficients != coefficients)
            return Status::FailedPrecondition("prepared GeLU polynomial contract mismatch");
    } else {
        auto compiled = std::make_shared<PreparedGeluPolynomial>();
        compiled->profile = context.profile;
        compiled->input_id = input.ciphertexts.front().parms_id();
        compiled->scale = target;
        compiled->coefficients = coefficients;
        const std::array<Complex, 5> packed{
            Complex{config.a, config.a}, Complex{-config.b, config.b},
            Complex{config.c, config.c}, Complex{0.5 - config.d, 0.5 + config.d},
            Complex{config.e, config.e}};
        seal::CKKSEncoder encoder(*context.context);
        for (std::size_t i = 0; i < packed.size(); ++i) {
            const auto id = i == 4 ? terminal_id.value() : coefficients_id.value();
            const double scale = i == 4 ? target : DroppedPrime(context, coefficients_id.value());
            auto encoded = CompileModelPlaintext(encoder,
                std::vector<Complex>(context.slot_count(), packed[i]), id, scale,
                "ffn.gelu.constant");
            if (!encoded.ok()) return encoded.status();
            compiled->plaintexts[i] = encoded.take_value();
        }
        auto phase = detail::MakeRealPairPhase(context, terminal_id.value());
        if (!phase.ok()) return phase.status();
        compiled->conversion_phase = std::make_shared<seal::Plaintext>(phase.take_value());
        program = compiled;
        if (cache.first) {
            auto stored = cache.first->PutPreparedBlock2Program(key, compiled);
            if (!stored.ok()) return stored.status();
        }
    }
    SealCipherTensor branches, terminal_input;
    branches.parameter_profile = terminal_input.parameter_profile = context.profile;
    terminal_input.logical_spec = branches.logical_spec = input.logical_spec;
    terminal_input.physical_packing = input.physical_packing;
    branches.physical_packing = GeluBranchPacking(input.physical_packing);
    branches.logical_spec.shape = branches.physical_packing.logical_shape;
    branches.logical_spec.layout = branches.physical_packing.layout;
    branches.ciphertexts.resize(input.ciphertexts.size());
    terminal_input.ciphertexts.resize(input.ciphertexts.size());
    if (cache.first && cache.first->preparing_plaintexts()) {
        for (auto *tensor : {&branches, &terminal_input})
            for (auto &ct : tensor->ciphertexts) { ct.parms_id() = terminal_id.value(); ct.scale() = target; }
    } else {
        std::vector<FfnGeluExecutionReport> reports(input.ciphertexts.size());
        auto evaluated = detail::ParallelRanges(input.ciphertexts.size(),
            he_workers, 1,
            [&](std::size_t begin, std::size_t end) -> Result<void> {
                seal::Evaluator evaluator(*context.context);
                for (std::size_t i = begin; i < end; ++i) {
                    auto &counts = reports[i];
                    const auto &x = input.ciphertexts[i];
                    seal::Ciphertext x2, x3, x4;
                    const auto finish_product = [&](seal::Ciphertext &value) -> Result<void> {
                        evaluator.relinearize_inplace(value, context.relinearization_keys);
                        evaluator.rescale_to_next_inplace(value);
                        ++counts.ciphertext_multiplications;
                        ++counts.relinearizations;
                        ++counts.rescales;
                        return NormalizeScale(value, target);
                    };
                    evaluator.square(x, x2);
                    auto normalized = finish_product(x2);
                    if (!normalized.ok()) return normalized.status();
                    auto x_aligned = x;
                    evaluator.mod_switch_to_inplace(x_aligned, x2.parms_id());
                    evaluator.multiply(x2, x_aligned, x3);
                    normalized = finish_product(x3);
                    if (!normalized.ok()) return normalized.status();
                    evaluator.square(x2, x4);
                    normalized = finish_product(x4);
                    if (!normalized.ok()) return normalized.status();
                    evaluator.mod_switch_to_inplace(x_aligned, x3.parms_id());
                    evaluator.mod_switch_to_inplace(x2, x3.parms_id());
                    counts.modulus_switches += 3;
                    // Four products share one high scale; sum before rescale.
                    const std::array<const seal::Ciphertext *, 4> powers{&x4, &x3, &x2, &x_aligned};
                    auto &packed = branches.ciphertexts[i];
                    for (std::size_t power = 0; power < powers.size(); ++power) {
                        seal::Ciphertext term;
                        evaluator.multiply_plain(*powers[power], *program->plaintexts[power], term);
                        ++counts.ciphertext_plain_multiplications;
                        if (power == 0) packed = std::move(term);
                        else evaluator.add_inplace(packed, term);
                    }
                    evaluator.rescale_to_next_inplace(packed);
                    normalized = NormalizeScale(packed, target);
                    if (!normalized.ok()) return normalized.status();
                    ++counts.rescales;
                    evaluator.add_plain_inplace(packed, *program->plaintexts[4]);
                    evaluator.mod_switch_to(x, terminal_id.value(), terminal_input.ciphertexts[i]);
                    ++counts.modulus_switches;
                }
                return {};
            }, detail::EstimatedHeWorkerBytes(context.parameters.polynomial_degree,
                context.parameters.coefficient_modulus_bits.size(), 12));
        if (!evaluated.ok()) return evaluated.status();
        if (report) for (const auto &counts : reports) {
            report->ciphertext_plain_multiplications += counts.ciphertext_plain_multiplications;
            report->ciphertext_multiplications += counts.ciphertext_multiplications;
            report->relinearizations += counts.relinearizations;
            report->rescales += counts.rescales;
            report->modulus_switches += counts.modulus_switches;
        }
    }
    for (auto *tensor : {&terminal_input, &branches}) {
        tensor->logical_spec.planned_level = terminal_level;
        tensor->logical_spec.planned_scale_bits = context.parameters.output_scale_bits;
    }
    return GeluPolynomialOutputs{std::move(terminal_input), std::move(branches), program->conversion_phase};
}

class MpcGeluRuntime::Impl {
public:
    explicit Impl(std::shared_ptr<SharedMpcSession> session, std::size_t worker = 0)
        : session_(std::move(session)), native_(session_->Acquire(worker)),
          party_(native_->party())
    {}

    FixArray Secret(const MpcShareTensor &tensor, std::size_t begin, std::size_t end)
    {
        FixArray result(party_, static_cast<int>(end - begin),
                        tensor.numeric.is_signed, tensor.numeric.ring_bits,
                        tensor.numeric.scale_bits);
        const auto mask = RingMask(tensor.numeric.ring_bits);
        for (std::size_t i = begin; i < end; ++i)
            result.data[i - begin] = tensor.local_share[i] & mask;
        return result;
    }

    std::vector<std::uint64_t> EvaluateRange(
        const MpcShareTensor &input,
        const MpcShareTensor &negative,
        const MpcShareTensor &positive,
        double threshold, std::size_t begin, std::size_t end)
    {
        auto x = Secret(input, begin, end);
        auto f0 = Secret(negative, begin, end);
        auto f1 = Secret(positive, begin, end);
        const auto encode = [&](double value) {
            const auto scaled = static_cast<std::int64_t>(
                std::llround(std::ldexp(value, input.numeric.scale_bits)));
            return static_cast<std::uint64_t>(scaled) &
                   RingMask(input.numeric.ring_bits);
        };
        auto &fix = native_->fix();
        const int size = x.size;
        std::vector<std::uint64_t> output(size);
        const auto mask = RingMask(x.ell);
        const auto upper = party_ == sci::ALICE ? encode(threshold) : 0;
        // SCI/Ferret scratch grows with the number of simultaneous OTs.
        // Bound the batch instead of tripling its peak for large FFNs.
        constexpr int kMaxBatchValues = 1 << 18;
        for (int offset = 0; offset < size;) {
            const int batch_size = std::min(kMaxBatchValues, size - offset);
            FixArray values(party_, batch_size, true, x.ell, x.s);
            std::copy_n(x.data + offset, batch_size, values.data);
            auto sign = fix.MSB(values);
            BoolArray select(party_, 2 * batch_size);
            FixArray left(party_, 2 * batch_size, true, x.ell, x.s),
                right(party_, 2 * batch_size, true, x.ell, x.s);
            for (int i = 0; i < batch_size; ++i) {
                select.data[i] = select.data[batch_size + i] = sign.data[i];
                left.data[i] = 0;
                right.data[i] = values.data[i];
                left.data[batch_size + i] = f0.data[offset + i];
                right.data[batch_size + i] = f1.data[offset + i];
            }
            // One batched mux computes ReLU(x) and the sign-selected branch.
            // abs(x)=2*ReLU(x)-x needs neither truncation nor a fourth mux.
            auto selected = fix.if_else(select, left, right);
            FixArray absolute(party_, batch_size, true, x.ell, x.s),
                bound(party_, batch_size, true, x.ell, x.s),
                tail(party_, batch_size, true, x.ell, x.s),
                polynomial(party_, batch_size, true, x.ell, x.s);
            for (int i = 0; i < batch_size; ++i) {
                tail.data[i] = selected.data[i];
                polynomial.data[i] = selected.data[batch_size + i];
                absolute.data[i] = (2 * selected.data[i] - values.data[i]) & mask;
                bound.data[i] = upper;
            }
            // GT extracts MSB(T-|x|). For 0<=|x|<=2^(ell-1) and
            // 0<=T<2^(ell-1), this difference fits the signed ring, even
            // for its most-negative x. Equality stays inside both branches.
            auto outside = fix.GT(absolute, bound);
            auto chosen = fix.if_else(outside, tail, polynomial);
            std::copy_n(chosen.data, batch_size, output.begin() + offset);
            offset += batch_size;
        }
        return output;
    }

    Result<MpcShareTensor> Evaluate(
        const MpcShareTensor &input, const MpcShareTensor &negative,
        const MpcShareTensor &positive, double threshold)
    {
        if (input.local_share.empty() || input.local_share.size() >
                static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            !std::isfinite(threshold) || threshold <= 0 ||
            std::round(std::ldexp(threshold, input.numeric.scale_bits)) >=
                std::ldexp(1., input.numeric.ring_bits - 1))
            return Status::Invalid("GeLU MPC size or threshold is unsupported");
        std::vector<std::uint64_t> output(input.local_share.size());
        session_->ForEachRange(output.size(), 1,
            [&](std::size_t worker, std::size_t begin, std::size_t end) {
                Impl local(session_, worker);
                auto values = local.EvaluateRange(input, negative, positive, threshold, begin, end);
                std::copy(values.begin(), values.end(), output.begin() + begin);
            });
        std::optional<int> output_bound;
        if (input.public_value_bound_bits && negative.public_value_bound_bits && positive.public_value_bound_bits)
            output_bound = std::max({*input.public_value_bound_bits,
                *negative.public_value_bound_bits, *positive.public_value_bound_bits});
        return MpcShareTensor{std::move(output), input.physical_shape,
                             input.numeric, ComplexComponent::kReal, output_bound};
    }

    std::uint64_t communication() const
    { return session_->communication_bytes(); }
    std::uint64_t rounds() const { return session_->rounds(); }
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
            report->comparisons += 2 * input.local_share.size();
            report->muxes += 3 * input.local_share.size();
            report->mpc_workers = std::max(report->mpc_workers,
                std::min(session_->worker_count(), input.local_share.size()));
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
    BlbReferenceConverter &converter,
    MpcGeluRuntime &mpc,
    TensorIdAllocator *ids,
    SealCipherTensorStore *he_tensors)
    : config_(config), layer_(layer), contexts_(contexts),
      parameters_(parameters), tensors_(tensors), converter_(converter),
      mpc_(mpc), ids_(ids), he_tensors_(he_tensors)
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
    auto packing = FfnGeluPackingPlanner::GetShared(layer, slots);
    if (!packing.ok()) return packing.status();
    auto result = ComplexColumnProjectionKernel::RequiredRotationSteps(
        packing.value()->centered, packing.value()->projected);
    if (!result.ok()) return result.status();
    result.value().push_back(0);
    result.value().push_back(static_cast<int>(slots / 2));
    return result;
}

namespace {

Result<PhysicalSlotMap> CombinedGeluMap(const HePackingPlan &packing)
{
    auto input = ToConversionMap(detail::PairedRealPacking(packing));
    if (!input.ok()) return input.status();
    auto branches = ToConversionMap(GeluBranchPacking(packing));
    if (!branches.ok()) return branches.status();
    const auto count = packing.logical_locations.size();
    for (auto assignment : branches.value().assignments) {
        if (assignment.real_logical_index != kUnusedLogicalIndex) assignment.real_logical_index += count;
        if (assignment.imaginary_logical_index != kUnusedLogicalIndex) assignment.imaginary_logical_index += count;
        input.value().assignments.push_back(assignment);
    }
    input.value().ciphertext_count += branches.value().ciphertext_count;
    input.value().logical_element_count += branches.value().logical_element_count;
    return input;
}

} // namespace

Result<FfnGeluOutputs> SealSciFfnGeluExecutor::Evaluate(
    const FfnGeluInputs &inputs,
    const LayerWeightHandles &weights,
    FfnGeluExecutionReport *report)
{
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
        auto required = RequiredRotationSteps(layer_, context.parameters);
        if (!required.ok()) return required.status();
        if (contexts_.role() == PartyRole::kServer) {
            if (context.galois_keys.size() == 0)
                return Status::FailedPrecondition(
                    "FFN/GeLU Galois keys have not been compiled");
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
        }
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

        auto plans = FfnGeluPackingPlanner::GetShared(layer_,
                                                   context.slot_count());
        if (!plans.ok()) return plans.status();
        const auto terminal_level =
            context.parameters.coefficient_modulus_bits.size() -
            context.parameters.terminal_prime_count;
        const auto kFfnGeluDepth = kFfnSuffixDepth;
        if (terminal_level < kFfnGeluDepth)
            return Status::FailedPrecondition(
                "FFN/GeLU profile has no five-level suffix");
        const auto ffn_start_level = terminal_level - kFfnGeluDepth;
        auto centered_pair = Scatter(centered.value(), plans.value()->centered);
        if (!centered_pair.ok()) return centered_pair.status();
        auto inverse_pair = ScatterInverse(inverse.value(),
                                            plans.value()->inverse_broadcast);
        if (!inverse_pair.ok()) return inverse_pair.status();
        auto centered_map = ToConversionMap(plans.value()->centered);
        if (!centered_map.ok()) return centered_map.status();
        // Same profile/level/scale: concatenate centered and inverse into one
        // conversion frame, then move ciphertexts into their original contracts.
        auto packed_shares = centered_pair.take_value();
        auto inverse_shares = inverse_pair.take_value();
        for (auto pair : {std::make_pair(&packed_shares.real, &inverse_shares.real),
                          std::make_pair(&packed_shares.imag, &inverse_shares.imag)}) {
            pair.first->local_share.insert(pair.first->local_share.end(),
                pair.second->local_share.begin(), pair.second->local_share.end());
            pair.first->physical_shape = {{plans.value()->centered.ciphertext_count + 1,
                                          context.slot_count()}};
        }
        auto input_map = centered_map.take_value();
        const auto centered_cts = input_map.ciphertext_count;
        const auto centered_logical = input_map.logical_element_count;
        ++input_map.ciphertext_count;
        input_map.logical_element_count += context.slot_count();
        input_map.assignments.resize(input_map.ciphertext_count * context.slot_count());
        for (std::size_t slot = 0; slot < context.slot_count(); ++slot)
            input_map.assignments[centered_cts * context.slot_count() + slot]
                .real_logical_index = centered_logical + slot;
        auto map_valid = input_map.Validate();
        if (!map_valid.ok()) return map_valid.status();
        const auto before_input = converter_.stats();
        std::vector<std::optional<int>> input_bounds(centered_cts + 1);
        input_bounds.back() = inverse.value().public_value_bound_bits;
        MpcToCkksTarget input_target{context.profile, ffn_start_level, context.parameters.input_scale_bits,
            {{input_map.logical_element_count}}, Layout::kComplexStrictColumn, input_map,
            false, std::nullopt, std::move(input_bounds)};
        input_target.public_slot_periods.resize(centered_cts + 1);
        input_target.public_slot_periods.back()[0] = NextPowerOfTwo(layer_.tokens);
        auto converted = converter_.MpcToCkksPacked(packed_shares, input_target);
        if (!converted.ok()) return converted.status();
        const auto after_input = converter_.stats();
        if (report) {
            ++report->mpc_to_ckks_calls;
            report->conversion_communication_bytes += after_input.communication_bytes - before_input.communication_bytes;
            report->conversion_rounds += after_input.rounds - before_input.rounds;
            report->conversion_milliseconds += after_input.milliseconds - before_input.milliseconds;
        }

        const bool retain_normalized = layer_.norm == NormPlacement::kNormalizedResidual && layer_.ffn_residual;
        const bool shared_skip = layer_.norm == NormPlacement::kNormalizedResidual &&
            execution->regions.at(HeRegionId::kRegion1).parameter_profile == context.profile;
        if (shared_skip && retain_normalized && !he_tensors_)
            return Status::FailedPrecondition("shared LN2 residual requires an HE tensor store");
        std::optional<TensorHandle> he_residual_handle;
        std::optional<SealCipherTensor> normalized_residual;
        std::optional<GeluPolynomialOutputs> polynomials;
        if (contexts_.role() == PartyRole::kServer) {
            detail::ScopedWallTimer he_timer(report ? &report->he_milliseconds : nullptr);
            if (!converted.value() || converted.value()->ciphertexts.size() != centered_cts + 1)
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
            auto gamma = parameters_.GetVectorShared(gamma_handle->id);
            if (!gamma.ok()) return gamma.status();
            auto beta = parameters_.GetVectorShared(beta_handle->id);
            if (!beta.ok()) return beta.status();
            std::optional<GammaFusedLayerNormParameters> original;
            if (retain_normalized) {
                original = GammaFusedLayerNormParameters{*gamma.value(), *beta.value()};
                normalized_residual.emplace();
            }
            auto affine = parameters_.GetGammaFusedLayerNorm(weights.wu.id);
            if (affine.ok()) {
                gamma = std::shared_ptr<const DenseVectorParameter>(affine.value(), &affine.value()->gamma);
                beta = std::shared_ptr<const DenseVectorParameter>(affine.value(), &affine.value()->beta);
            }
            auto wu = parameters_.GetMatrixShared(weights.wu.id);
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
            auto &converted_cts = converted.value()->ciphertexts;
            centered_tensor.ciphertexts.reserve(centered_cts);
            for (std::size_t ct = 0; ct < centered_cts; ++ct)
                centered_tensor.ciphertexts.push_back(std::move(converted_cts[ct]));
            centered_tensor.logical_spec = inputs.centered_shares.spec;
            centered_tensor.logical_spec.domain = Domain::kHeCiphertext;
            centered_tensor.logical_spec.layout = Layout::kComplexStrictColumn;
            centered_tensor.logical_spec.he_region = HeRegionId::kRegion5;
            centered_tensor.logical_spec.he_parameter_profile = context.profile;
            centered_tensor.logical_spec.planned_level = ffn_start_level;
            centered_tensor.logical_spec.planned_scale_bits =
                context.parameters.input_scale_bits;
            centered_tensor.physical_packing = plans.value()->centered;

            SealCipherTensor inverse_tensor;
            inverse_tensor.parameter_profile = context.profile;
            inverse_tensor.ciphertexts.push_back(std::move(converted_cts.back()));
            inverse_tensor.logical_spec = inputs.inverse_stddev_shares.spec;
            inverse_tensor.logical_spec.domain = Domain::kHeCiphertext;
            inverse_tensor.logical_spec.layout = Layout::kReduction;
            inverse_tensor.logical_spec.he_region = HeRegionId::kRegion5;
            inverse_tensor.logical_spec.he_parameter_profile = context.profile;
            inverse_tensor.logical_spec.planned_level = ffn_start_level;
            inverse_tensor.logical_spec.planned_scale_bits =
                context.parameters.input_scale_bits;
            inverse_tensor.physical_packing = plans.value()->inverse_broadcast;

            auto normalized = FfnLayerNormTailKernel::Evaluate(
                centered_tensor, inverse_tensor, *gamma.value(), *beta.value(),
                layer_, context, report, config_.threads,
                original ? &*original : nullptr,
                normalized_residual ? &*normalized_residual : nullptr);
            if (!normalized.ok()) return normalized.status();
            auto projected = FfnUpProjectionKernel::Evaluate(
                normalized.value(), *wu.value(), bu, layer_, context, report,
                config_.threads);
            if (!projected.ok()) return projected.status();
            auto polynomial = GeluPolynomialKernel::Evaluate(
                projected.value(), config_.gelu, context, report,
                config_.threads);
            if (!polynomial.ok()) return polynomial.status();
            polynomials = polynomial.take_value();
        }

        if (shared_skip && retain_normalized) {
            const auto residual_id = ids_ ? ids_->Next() : next_id_.fetch_add(1);
            auto residual_spec = inputs.centered_shares.spec;
            residual_spec.domain = Domain::kHeCiphertext;
            residual_spec.layout = Layout::kStrictColumn;
            residual_spec.he_region = HeRegionId::kRegion5;
            residual_spec.he_parameter_profile = context.profile;
            residual_spec.planned_level = ffn_start_level + 1;
            residual_spec.planned_scale_bits = context.parameters.output_scale_bits;
            if (contexts_.role() == PartyRole::kServer) {
                normalized_residual->logical_spec = residual_spec;
                auto stored = he_tensors_->Put(residual_id, std::move(*normalized_residual));
                if (!stored.ok()) return stored.status();
                normalized_residual.reset();
            }
            he_residual_handle = TensorHandle{residual_id, residual_spec};
        }

        auto combined_map = CombinedGeluMap(plans.value()->projected);
        if (!combined_map.ok()) return combined_map.status();
        std::optional<HePackingPlan> residual_plan;
        if (retain_normalized && !shared_skip) {
            auto built = FfnGeluPackingPlanner::RealColumns(
                {{layer_.tokens, layer_.model_width}}, context.slot_count(), Layout::kStrictColumn);
            if (!built.ok()) return built.status();
            residual_plan = detail::PairedRealPacking(built.value());
            auto map = ToConversionMap(*residual_plan);
            if (!map.ok()) return map.status();
            const auto logical_offset = combined_map.value().logical_element_count;
            for (auto assignment : map.value().assignments) {
                if (assignment.real_logical_index != kUnusedLogicalIndex)
                    assignment.real_logical_index += logical_offset;
                if (assignment.imaginary_logical_index != kUnusedLogicalIndex)
                    assignment.imaginary_logical_index += logical_offset;
                combined_map.value().assignments.push_back(assignment);
            }
            combined_map.value().logical_element_count += map.value().logical_element_count;
            combined_map.value().ciphertext_count += map.value().ciphertext_count;
        }
        std::optional<SealCipherBatch> batch;
        if (contexts_.role() == PartyRole::kServer) {
            detail::ScopedWallTimer he_timer(report ? &report->he_milliseconds : nullptr);
            std::vector<seal::Ciphertext> ciphertexts;
            const auto paired_count = (polynomials->input.ciphertexts.size() + 1) / 2;
            ciphertexts.resize(paired_count);
            auto paired = detail::ParallelRanges(paired_count, config_.threads, 1,
                [&](std::size_t begin, std::size_t end) -> Result<void> {
                    seal::Evaluator evaluator(*context.context);
                    for (std::size_t pair = begin; pair < end; ++pair) {
                        auto value = std::move(polynomials->input.ciphertexts[2 * pair]);
                        if (2 * pair + 1 < polynomials->input.ciphertexts.size()) {
                            auto imaginary = std::move(polynomials->input.ciphertexts[2 * pair + 1]);
                            evaluator.multiply_plain_inplace(imaginary, *polynomials->conversion_phase);
                            evaluator.add_inplace(value, imaginary);
                        } else {
                            auto imaginary = value;
                            evaluator.multiply_plain_inplace(imaginary, *polynomials->conversion_phase);
                            evaluator.add_inplace(value, imaginary);
                        }
                        ciphertexts[pair] = std::move(value);
                    }
                    return {};
                }, detail::EstimatedHeWorkerBytes(context.parameters.polynomial_degree, 1, 4));
            if (!paired.ok()) return paired.status();
            if (report) report->ciphertext_plain_multiplications += (polynomials->input.ciphertexts.size() + 1) / 2;
            ciphertexts.insert(ciphertexts.end(),
                std::make_move_iterator(polynomials->branches.ciphertexts.begin()),
                std::make_move_iterator(polynomials->branches.ciphertexts.end()));
            if (normalized_residual) {
                auto paired_residual = detail::PairRealCiphertexts(
                    std::move(normalized_residual->ciphertexts), context, config_.threads);
                if (!paired_residual.ok()) return paired_residual.status();
                auto extra = paired_residual.take_value();
                ciphertexts.insert(ciphertexts.end(),
                    std::make_move_iterator(extra.begin()), std::make_move_iterator(extra.end()));
            }
            batch = SealCipherBatch{
                context.profile, std::move(ciphertexts),
                {{combined_map.value().logical_element_count}},
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
        const auto paired_input = detail::PairedRealPacking(plans.value()->projected);
        const auto group_cts = paired_input.ciphertext_count;
        auto x_share = CompactSegment(physical.value(), paired_input, 0,
                                      output_shape, std::nullopt, true);
        auto f0_share = CompactSegment(physical.value(),
                                       plans.value()->projected, group_cts,
                                       output_shape);
        auto f1_share = CompactSegment(physical.value(),
                                       plans.value()->projected, group_cts,
                                       output_shape, ComplexLane::kImaginary);
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
        std::optional<TensorHandle> residual_handle;
        if (residual_plan) {
            const TensorShape shape{{layer_.tokens, layer_.model_width}};
            auto residual_share = CompactSegment(physical.value(), *residual_plan, group_cts + plans.value()->projected.ciphertext_count, shape, std::nullopt, true);
            if (!residual_share.ok()) return residual_share.status();
            auto residual_spec = spec;
            residual_spec.shape = shape;
            residual_spec.layout = Layout::kStrictColumn;
            const auto residual_id = ids_ ? ids_->Next() : next_id_.fetch_add(1);
            auto stored = tensors_.PutMpcShare(residual_id, residual_share.take_value());
            if (!stored.ok()) return stored.status();
            residual_handle = TensorHandle{residual_id, residual_spec};
        }
        return FfnGeluOutputs{TensorHandle{id, spec}, residual_handle, he_residual_handle};
    } catch (const std::exception &error) {
        return Status::Internal(std::string("FFN/GeLU stage failed: ") +
                                error.what());
    }
}

} // namespace hera
