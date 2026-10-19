#include "hera/block2_he.h"
#include "hera/detail/he_level.h"
#include "hera/detail/packing_utils.h"
#include "hera/detail/parallel_ranges.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <tuple>

namespace hera {
namespace {

using Complex = std::complex<double>;

constexpr double kScaleTolerance = 1.0e-3;

std::vector<std::shared_ptr<const seal::SEALContext::ContextData>> Levels(
    const HeContext &context)
{
    std::vector<std::shared_ptr<const seal::SEALContext::ContextData>> result;
    for (auto data = context.context->first_context_data(); data;
         data = data->next_context_data())
        result.push_back(data);
    return result;
}

using detail::LevelOf;

bool CloseScale(double actual, double expected)
{
    const auto largest = std::max(std::abs(actual), std::abs(expected));
    return std::isfinite(largest) && largest != 0.0 &&
           std::abs(actual - expected) <= largest * kScaleTolerance;
}

void NormalizeScale(seal::Ciphertext &ciphertext, double target)
{
    if (!CloseScale(ciphertext.scale(), target)) {
        std::ostringstream message;
        message << "CKKS scale drift exceeds tolerance: actual_bits="
                << std::log2(ciphertext.scale()) << " target_bits="
                << std::log2(target);
        throw std::invalid_argument(message.str());
    }
    ciphertext.scale() = target;
}

using detail::DroppedPrime;

int CanonicalRotation(long long value, std::size_t slots)
{
    value %= static_cast<long long>(slots);
    if (value > static_cast<long long>(slots / 2))
        value -= static_cast<long long>(slots);
    if (value <= -static_cast<long long>(slots / 2))
        value += static_cast<long long>(slots);
    return static_cast<int>(value);
}

seal::Ciphertext RotateDirect(const seal::Ciphertext &input, int rotation,
                              const HeContext &context,
                              seal::Evaluator &evaluator,
                              Block2ExecutionReport *report)
{
    seal::Ciphertext result = input;
    const int step = CanonicalRotation(rotation, context.slot_count());
    if (step == 0) return result;
    // Every planned offset has its own Galois key. Do not decompose this
    // logical rotation into signed powers of two: one call below therefore
    // performs exactly one automorphism and one key-switch.
    evaluator.rotate_vector_inplace(result, step, context.galois_keys);
    if (report) ++report->rotations;
    return result;
}

Result<void> ValidateMatrixHandle(const WeightHandle &handle,
                                  const MatrixSpec &expected)
{
    if (!handle.valid() || handle.spec.name != expected.name ||
        handle.spec.rows != expected.rows ||
        handle.spec.columns != expected.columns)
        return Status::Invalid(expected.name + " handle has the wrong shape");
    return {};
}

TensorSpec MakeHeSpec(const TensorShape &shape, Layout layout,
                      const std::string &profile, std::size_t level,
                      int scale_bits)
{
    TensorSpec result;
    result.shape = shape;
    result.domain = Domain::kHeCiphertext;
    result.layout = layout;
    result.numeric = {43, 13};
    result.he_region = HeRegionId::kRegion2;
    result.he_parameter_profile = profile;
    result.planned_level = level;
    result.planned_scale_bits = scale_bits;
    return result;
}

Result<void> SamePhysicalPacking(const HePackingPlan &left,
                                 const HePackingPlan &right)
{
    if (left.layout != right.layout ||
        left.logical_shape != right.logical_shape ||
        left.slot_count != right.slot_count ||
        left.ciphertext_count != right.ciphertext_count ||
        left.logical_locations.size() != right.logical_locations.size())
        return Status::Invalid("LayerNorm inputs use different packing plans");
    for (std::size_t i = 0; i < left.logical_locations.size(); ++i) {
        const auto &a = left.logical_locations[i];
        const auto &b = right.logical_locations[i];
        if (a.size() != b.size())
            return Status::Invalid("LayerNorm inputs use different replicas");
        for (std::size_t j = 0; j < a.size(); ++j)
            if (a[j].ciphertext != b[j].ciphertext ||
                a[j].slot != b[j].slot ||
                a[j].component != b[j].component)
                return Status::Invalid(
                    "LayerNorm inputs use different physical coordinates");
    }
    return {};
}

Result<void> ValidateRealColumnInput(const HePackingPlan &packing,
                                     const TransformerLayerConfig &layer,
                                     std::size_t slots)
{
    if (packing.layout != Layout::kStrictColumn ||
        packing.logical_shape != TensorShape{{layer.tokens, layer.model_width}} ||
        packing.slot_count != slots || layer.tokens == 0 || layer.tokens > slots)
        return Status::Invalid("Block 2 requires real strict-column [tokens,model_width]");
    std::size_t stride = 1;
    while (stride < layer.tokens) {
        if (stride > slots / 2)
            return Status::Invalid("padded token column exceeds CKKS slots");
        stride *= 2;
    }
    const auto columns = slots / stride;
    if (packing.ciphertext_count != layer.model_width / columns +
            (layer.model_width % columns != 0))
        return Status::Invalid("Block 2 strict-column ciphertext count mismatch");
    for (std::size_t logical = 0; logical < packing.logical_locations.size(); ++logical) {
        const auto token = logical / layer.model_width;
        const auto feature = logical % layer.model_width;
        const auto &locations = packing.Locations(logical);
        if (locations.size() != 1 ||
            locations.front().component != ComplexLane::kReal ||
            locations.front().ciphertext != feature / columns ||
            locations.front().slot != (feature % columns) * stride + token ||
            locations.front().packed_query_head)
            return Status::Invalid("Block 2 input must preserve real feature columns");
    }
    return {};
}

using PlainRef = std::shared_ptr<const seal::Plaintext>;

PlainRef CompileFixedPlain(seal::CKKSEncoder &encoder,
                          const std::vector<double> &values,
                          seal::parms_id_type id, double scale,
                          std::string_view tag)
{
    std::vector<Complex> complex_values(values.begin(), values.end());
    auto encoded = CompileModelPlaintext(encoder, complex_values, id, scale, tag);
    if (!encoded.ok()) throw std::invalid_argument(encoded.status().message());
    return encoded.take_value();
}

struct PreparedLnTail final : PreparedBlock2Program {
    std::string profile;
    seal::parms_id_type input_id{}, output_id{};
    double input_scale{}, target_scale{}, dropped{};
    HePackingPlan packing, inverse_packing;
    std::vector<double> gamma, beta, centered_scales, inverse_scales;
    std::vector<PlainRef> gains, biases, residual_gains, residual_biases;
    std::vector<double> residual_gamma, residual_beta;
};

std::shared_ptr<const PreparedLnTail> CompileLnTail(
    const SealCipherTensor &centered, const SealCipherTensor &inverse,
    const GammaFusedLayerNormParameters &affine,
    const TransformerLayerConfig &layer, const HeContext &context,
    const GammaFusedLayerNormParameters *original_affine)
{
    auto program = std::make_shared<PreparedLnTail>();
    program->profile = context.profile;
    program->input_id = centered.ciphertexts.front().parms_id();
    program->output_id = context.context->get_context_data(program->input_id)
        ->next_context_data()->parms_id();
    program->input_scale = centered.ciphertexts.front().scale();
    program->target_scale = std::ldexp(1.0, context.parameters.default_scale_bits);
    program->dropped = DroppedPrime(context, program->input_id);
    program->packing = centered.physical_packing;
    program->inverse_packing = inverse.physical_packing;
    program->gamma = affine.gamma.values;
    program->beta = affine.beta.values;
    if (original_affine) {
        program->residual_gamma = original_affine->gamma.values;
        program->residual_beta = original_affine->beta.values;
    }
    const auto stride = detail::TokenColumnStride(layer.tokens);
    const auto columns = context.slot_count() / stride;
    seal::CKKSEncoder encoder(*context.context);
    for (std::size_t ct = 0; ct < centered.ciphertexts.size(); ++ct) {
        const double gain_scale = program->target_scale * program->dropped /
            (centered.ciphertexts[ct].scale() * inverse.ciphertexts[inverse.ciphertexts.size() == 1 ? 0 : ct].scale());
        if (!std::isfinite(gain_scale) || gain_scale < 1.0)
            throw std::invalid_argument("LayerNorm gain scale cannot realize the configured target");
        program->centered_scales.push_back(centered.ciphertexts[ct].scale());
        program->inverse_scales.push_back(inverse.ciphertexts[inverse.ciphertexts.size() == 1 ? 0 : ct].scale());
        std::vector<double> gains(context.slot_count()), beta(context.slot_count());
        bool has_gain = false;
        const auto end = std::min(layer.model_width, (ct + 1) * columns);
        for (std::size_t feature = ct * columns; feature < end; ++feature) {
            const auto begin = (feature % columns) * stride;
            std::fill_n(gains.begin() + begin, layer.tokens, affine.gamma.values[feature]);
            std::fill_n(beta.begin() + begin, layer.tokens, affine.beta.values[feature]);
            has_gain |= affine.gamma.values[feature] != 0.0;
        }
        program->gains.push_back(has_gain ? CompileFixedPlain(encoder, gains,
            program->input_id, gain_scale, "block2.ln1.real.gain") : PlainRef{});
        program->biases.push_back(CompileFixedPlain(encoder, beta,
            program->output_id, program->target_scale, "block2.ln1.real.beta"));
        if (original_affine) {
            std::fill(gains.begin(), gains.end(), 0.0);
            std::fill(beta.begin(), beta.end(), 0.0);
            has_gain = false;
            for (std::size_t feature = ct * columns; feature < end; ++feature) {
                const auto begin = (feature % columns) * stride;
                const double gain = original_affine->gamma.values[feature];
                std::fill_n(gains.begin() + begin, layer.tokens, gain);
                std::fill_n(beta.begin() + begin, layer.tokens, original_affine->beta.values[feature]);
                has_gain |= gain != 0.0;
            }
            program->residual_gains.push_back(has_gain ? CompileFixedPlain(encoder, gains,
                program->input_id, gain_scale, "block2.ln1.residual.gamma") : PlainRef{});
            program->residual_biases.push_back(CompileFixedPlain(encoder, beta,
                program->output_id, program->target_scale, "block2.ln1.residual.beta"));
        }
    }
    return program;
}

Result<SealCipherTensor> LayerNormTail(
    const SealCipherTensor &centered,
    const SealCipherTensor &inverse,
    const GammaFusedLayerNormParameters &affine,
    const TransformerLayerConfig &layer,
    const HeContext &context,
    Block2ExecutionReport *report,
    const GammaFusedLayerNormParameters *original_affine,
    SealCipherTensor *normalized_residual, std::size_t he_workers)
{
    if ((original_affine == nullptr) != (normalized_residual == nullptr))
        return Status::Invalid("LN1 residual requires original affine and output together");
    if (original_affine && (original_affine->gamma.values.size() != layer.model_width ||
                            original_affine->beta.values.size() != layer.model_width))
        return Status::Invalid("LN1 original affine size mismatch");
    auto centered_valid = centered.Validate(context);
    if (!centered_valid.ok()) return centered_valid.status();
    auto inverse_valid = inverse.Validate(context);
    if (!inverse_valid.ok()) return inverse_valid.status();
    const bool broadcast = inverse.logical_spec.shape == TensorShape{{layer.tokens, 1}};
    auto inverse_plan = broadcast
        ? MultiHeadPackingPlan::TokenInverseBroadcast(layer.tokens, context.slot_count())
        : Result<HePackingPlan>{centered.physical_packing};
    if (!inverse_plan.ok()) return inverse_plan.status();
    auto packing_valid = SamePhysicalPacking(inverse_plan.value(), inverse.physical_packing);
    if (!packing_valid.ok()) return packing_valid.status();
    if (centered.logical_spec.shape !=
            TensorShape{{layer.tokens, layer.model_width}} ||
        (!broadcast && inverse.logical_spec.shape != centered.logical_spec.shape) ||
        centered.logical_spec.layout != Layout::kStrictColumn)
        return Status::Invalid(
            "Block 2 LayerNorm inputs must be real strict-column [tokens,model_width]");
    auto columns_valid = ValidateRealColumnInput(
        centered.physical_packing, layer, context.slot_count());
    if (!columns_valid.ok()) return columns_valid.status();
    if (affine.gamma.values.size() != layer.model_width ||
        affine.beta.values.size() != layer.model_width)
        return Status::Invalid("LayerNorm affine parameter size mismatch");
    const auto input_id = centered.ciphertexts.front().parms_id();
    const auto input_level = LevelOf(context, input_id);
    if (!input_level.ok()) return input_level.status();
    if (input_level.value() != 0)
        return Status::FailedPrecondition(
            "Block 2 LayerNorm inputs must be at fresh level 0");
    const double wanted_input = std::ldexp(
        1.0, context.parameters.input_scale_bits);
    for (std::size_t ct = 0; ct < centered.ciphertexts.size(); ++ct)
        if (centered.ciphertexts[ct].parms_id() != input_id ||
            inverse.ciphertexts[broadcast ? 0 : ct].parms_id() != input_id ||
            !CloseScale(centered.ciphertexts[ct].scale(), wanted_input) ||
            !CloseScale(inverse.ciphertexts[inverse.ciphertexts.size() == 1 ? 0 : ct].scale(), wanted_input))
            return Status::FailedPrecondition(
                "LayerNorm inputs have incompatible level or scale");
    const auto next = context.context->get_context_data(input_id)->next_context_data();
    if (!next)
        return Status::FailedPrecondition("LayerNorm tail has no next level");
    const double target = std::ldexp(
        1.0, context.parameters.default_scale_bits);
    const double dropped = DroppedPrime(context, input_id);
    const auto cache = ModelPlaintextCacheScope::Current();
    const bool preparing = cache.first && cache.first->preparing_plaintexts();
    const auto program_key = cache.second + "/block2/ln1";
    std::shared_ptr<const PreparedLnTail> program;
    if (cache.first && !preparing) {
        auto found = cache.first->GetPreparedBlock2Program(program_key);
        if (!found.ok()) return found.status();
        program = std::dynamic_pointer_cast<const PreparedLnTail>(found.value());
        if (!program || program->profile != context.profile ||
            program->input_id != input_id || !CloseScale(program->input_scale, wanted_input) ||
            program->gamma != affine.gamma.values || program->beta != affine.beta.values ||
            program->residual_gamma != (original_affine ? original_affine->gamma.values : std::vector<double>{}) ||
            program->residual_beta != (original_affine ? original_affine->beta.values : std::vector<double>{}) ||
            !SamePreparedPacking(program->packing, centered.physical_packing) ||
            !SamePreparedPacking(program->inverse_packing, inverse.physical_packing))
            return Status::FailedPrecondition("prepared LayerNorm tail contract mismatch");
        for (std::size_t ct = 0; ct < centered.ciphertexts.size(); ++ct)
            if (std::abs(centered.ciphertexts[ct].scale() / program->centered_scales[ct] - 1.0) > 1e-6 ||
                std::abs(inverse.ciphertexts[inverse.ciphertexts.size() == 1 ? 0 : ct].scale() / program->inverse_scales[ct] - 1.0) > 1e-6)
                return Status::FailedPrecondition("prepared LayerNorm tail scale mismatch");
    } else {
        program = CompileLnTail(centered, inverse, affine, layer, context, original_affine);
        if (preparing) {
            auto stored = cache.first->PutPreparedBlock2Program(program_key, program);
            if (!stored.ok()) return stored.status();
        }
    }
    SealCipherTensor result;
    result.parameter_profile = centered.parameter_profile;
    result.logical_spec = centered.logical_spec;
    result.physical_packing = centered.physical_packing;
    result.ciphertexts.resize(centered.ciphertexts.size());
    if (normalized_residual) *normalized_residual = result;
    std::vector<Block2ExecutionReport> counts(result.ciphertexts.size());
    auto evaluated = detail::ParallelRanges(result.ciphertexts.size(),
        he_workers, 1,
        [&](std::size_t begin, std::size_t end) -> Result<void> {
        seal::Evaluator evaluator(*context.context);
        for (std::size_t ct = begin; ct < end; ++ct) {
        auto *report = &counts[ct];
        if (preparing) {
            result.ciphertexts[ct].parms_id() = program->output_id;
            result.ciphertexts[ct].scale() = target;
            if (normalized_residual) normalized_residual->ciphertexts[ct] = result.ciphertexts[ct];
            continue;
        }
        seal::Ciphertext product;
        if (program->gains[ct] || (normalized_residual && program->residual_gains[ct])) {
            evaluator.multiply(centered.ciphertexts[ct], inverse.ciphertexts[broadcast ? 0 : ct], product);
            evaluator.relinearize_inplace(product, context.relinearization_keys);
            if (report) { ++report->ciphertext_multiplications; ++report->relinearizations; }
        }
        const auto finish = [&](const PlainRef &gain, const PlainRef &bias,
                                seal::Ciphertext &output) {
            if (gain) {
                evaluator.multiply_plain(product, *gain, output);
                if (report) ++report->ciphertext_plain_multiplications;
            } else {
                seal::Encryptor encryptor(*context.context, context.public_key);
                encryptor.encrypt_zero(input_id, output);
                output.scale() = target * dropped;
            }
            evaluator.rescale_to_next_inplace(output);
            NormalizeScale(output, target);
            evaluator.add_plain_inplace(output, *bias);
            if (report) ++report->rescales;
        };
        finish(program->gains[ct], program->biases[ct], result.ciphertexts[ct]);
        if (normalized_residual)
            finish(program->residual_gains[ct], program->residual_biases[ct],
                   normalized_residual->ciphertexts[ct]);
        }
        return {};
    }, detail::EstimatedHeWorkerBytes(context.parameters.polynomial_degree,
        context.parameters.coefficient_modulus_bits.size(), 6));
    if (!evaluated.ok()) return evaluated.status();
    if (report) for (const auto &count : counts) {
        report->ciphertext_multiplications += count.ciphertext_multiplications;
        report->ciphertext_plain_multiplications += count.ciphertext_plain_multiplications;
        report->relinearizations += count.relinearizations;
        report->rescales += count.rescales;
    }
    result.logical_spec.planned_level = 1;
    result.logical_spec.planned_scale_bits =
        context.parameters.default_scale_bits;
    if (normalized_residual) normalized_residual->logical_spec = result.logical_spec;
    return result;
}

Result<std::shared_ptr<const PreparedProjectionPlan>> CompileProjection(
    const SealCipherTensor &input, const DenseMatrixParameter &weights,
    const std::optional<DenseVectorParameter> &bias, Layout layout,
    const TransformerLayerConfig &layer, const HeContext &context)
{
    auto packing = MultiHeadPackingPlan::ProjectionOutput(layer, layout, context.slot_count());
    if (!packing.ok()) return packing.status();
    auto decision = MultiHeadPackingPlan::ProjectionDecision(layer, layout, context.slot_count());
    if (!decision.ok()) return decision.status();
    auto split = ComplexDiagonalProjectionPlanner::ColumnBsgs(layer, context.slot_count());
    if (!split.ok()) return split.status();
    auto prepared = std::make_shared<PreparedProjectionPlan>();
    prepared->profile = context.profile;
    prepared->input_parms_id = input.ciphertexts.front().parms_id();
    prepared->input_scale = input.ciphertexts.front().scale();
    prepared->input_packing = input.physical_packing;
    prepared->output_packing = packing.take_value();
    prepared->packing_decision = decision.take_value();
    prepared->real_input = true;
    prepared->reuse_baby_rotations = true;
    prepared->conjugate_output = layout == Layout::kMultiHeadK;
    prepared->output_scale_bits = context.parameters.default_scale_bits;
    prepared->bias_plaintexts.resize(prepared->output_packing.ciphertext_count);
    const auto heads = layout == Layout::kMultiHeadQ ? layer.q_heads : layer.kv_heads;
    const auto width = layout == Layout::kMultiHeadV ? layer.value_head_width : layer.qk_head_width;
    const auto output_columns = heads * width;
    const auto slots = context.slot_count();
    const auto stride = split.value().token_stride;
    const auto columns = split.value().columns_per_ciphertext;
    const double sign = prepared->conjugate_output ? -1.0 : 1.0;
    const double factor = layout == Layout::kMultiHeadV ? 1.0 : 1.0 / std::sqrt(
        2.0 * (layout == Layout::kMultiHeadQ ? static_cast<double>(width) : 1.0));
    const double target = std::ldexp(1.0, prepared->output_scale_bits);
    const double plain_scale = target * DroppedPrime(context, prepared->input_parms_id) /
        prepared->input_scale;
    seal::CKKSEncoder encoder(*context.context);
    // Index only token-zero destinations. Each weight is repeated down its
    // token column; plaintext giant adjustment is applied while filling slots.
    struct ColumnDestination { std::size_t column, slot; Complex phase; };
    std::vector<std::vector<ColumnDestination>> destinations(prepared->output_packing.ciphertext_count);
    for (std::size_t column = 0; column < output_columns; ++column)
        for (const auto &dest : prepared->output_packing.Locations(column))
            destinations[dest.ciphertext].push_back({column, dest.slot,
                dest.component == ComplexLane::kReal ? Complex{factor, 0.0} : Complex{0.0, sign * factor}});
    // Only one slot vector is live. Encode and release each diagonal before
    // generating the next; never retain the full dense mask collection.
    for (std::size_t ct = 0; ct < destinations.size(); ++ct) {
        for (std::size_t j = 0; j < split.value().giant_steps; ++j) {
            PreparedProjectionGroup group;
            group.target_ct = ct;
            group.giant_rotation = static_cast<int>(j * split.value().baby_steps * stride);
            for (std::size_t source_ct = 0; source_ct < input.ciphertexts.size(); ++source_ct)
                for (std::size_t i = 0; i < split.value().baby_steps; ++i) {
                    const auto baby = i * stride;
                    const auto rotation = static_cast<std::size_t>(group.giant_rotation) + baby;
                    std::vector<Complex> mask(slots);
                    bool nonzero = false;
                    for (const auto &dest : destinations[ct]) {
                        const auto source_slot = (dest.slot + rotation) % slots;
                        const auto feature = source_ct * columns + source_slot / stride;
                        if (feature >= layer.model_width) continue;
                        const auto coefficient = weights.row_major[feature * output_columns + dest.column] * dest.phase;
                        nonzero |= coefficient != Complex{};
                        const auto adjusted_slot = (dest.slot + static_cast<std::size_t>(group.giant_rotation)) % slots;
                        for (std::size_t token = 0; token < layer.tokens; ++token)
                            mask[adjusted_slot + token] += coefficient;
                    }
                    if (!nonzero) continue;
                    auto encoded = CompileModelPlaintext(encoder, mask, prepared->input_parms_id,
                        plain_scale, "block2.projection.diagonal");
                    if (!encoded.ok()) return encoded.status();
                    group.terms.push_back({source_ct, ComplexLane::kReal,
                        static_cast<int>(baby), encoded.take_value()});
                }
            if (!group.terms.empty()) prepared->groups.push_back(std::move(group));
        }
        if (bias) {
            std::vector<Complex> mask(slots);
            for (const auto &dest : destinations[ct])
                for (std::size_t token = 0; token < layer.tokens; ++token)
                    mask[dest.slot + token] += bias->values[dest.column] * dest.phase;
            auto encoded = CompileModelPlaintext(encoder, mask, prepared->input_parms_id,
                prepared->input_scale * plain_scale, "block2.projection.bias");
            if (!encoded.ok()) return encoded.status();
            prepared->bias_plaintexts[ct] = encoded.take_value();
        }
    }
    return std::shared_ptr<const PreparedProjectionPlan>(std::move(prepared));
}

Result<SealCipherTensor> Project(
    const SealCipherTensor &input, const DenseMatrixParameter &weights,
    const std::optional<DenseVectorParameter> &bias, Layout output_layout,
    const TransformerLayerConfig &layer, const HeContext &context,
    PreparedProjectionRotationCache &rotation_cache,
    Block2ExecutionReport *report, std::size_t he_workers)
{
    if (input.logical_spec.shape != TensorShape{{layer.tokens, layer.model_width}} ||
        input.logical_spec.layout != Layout::kStrictColumn)
        return Status::Invalid("projection expects real strict-column [tokens,model_width]");
    const auto heads = output_layout == Layout::kMultiHeadQ ? layer.q_heads : layer.kv_heads;
    const auto width = output_layout == Layout::kMultiHeadV ? layer.value_head_width : layer.qk_head_width;
    if (weights.spec.rows != layer.model_width || weights.spec.columns != heads * width ||
        weights.row_major.size() != layer.model_width * heads * width ||
        (bias && bias->values.size() != heads * width))
        return Status::Invalid("projection weight/bias payload has the wrong shape");
    if (input.ciphertexts.empty()) return Status::Invalid("projection input has no ciphertexts");
    auto valid = input.Validate(context);
    if (!valid.ok()) return valid.status();
    auto columns_valid = ValidateRealColumnInput(input.physical_packing, layer, context.slot_count());
    if (!columns_valid.ok()) return columns_valid.status();
    auto decision = MultiHeadPackingPlan::ProjectionDecision(layer, output_layout, context.slot_count());
    if (!decision.ok()) return decision.status();
    const auto cache = ModelPlaintextCacheScope::Current();
    const bool preparing = cache.first && cache.first->preparing_plaintexts();
    const auto key = cache.second + "/block2/" + weights.spec.name;
    std::shared_ptr<const PreparedProjectionPlan> program;
    if (cache.first && !preparing) {
        auto found = cache.first->GetPreparedProjection(key);
        if (!found.ok()) return found.status();
        program = found.take_value();
        const auto &compiled = program->packing_decision;
        if (program->output_packing.layout != output_layout ||
            program->output_packing.logical_shape != TensorShape{{layer.tokens, heads, width}} ||
            !program->real_input || program->conjugate_output != (output_layout == Layout::kMultiHeadK) ||
            !compiled || compiled->target != decision.value().target ||
            compiled->heads_per_tile != decision.value().heads_per_tile ||
            compiled->head_groups != decision.value().head_groups)
            return Status::FailedPrecondition("prepared Block-2 W*E packing decision mismatch");
    } else {
        auto compiled = CompileProjection(input, weights, bias, output_layout, layer, context);
        if (!compiled.ok()) return compiled.status();
        program = compiled.take_value();
        if (preparing) {
            auto stored = cache.first->PutPreparedProjection(key, program);
            if (!stored.ok()) return stored.status();
        }
    }
    auto level = LevelOf(context, input.ciphertexts.front().parms_id());
    if (!level.ok()) return level.status();
    SealCipherTensor result;
    if (preparing) {
        const auto next = context.context->get_context_data(program->input_parms_id)->next_context_data();
        result.ciphertexts.resize(program->output_packing.ciphertext_count);
        for (auto &ct : result.ciphertexts) {
            ct.parms_id() = next->parms_id();
            ct.scale() = std::ldexp(1.0, program->output_scale_bits);
        }
    } else {
        PreparedProjectionCounts counts;
        auto output = EvaluatePreparedProjection(*program, input, context, &counts, &rotation_cache, he_workers);
        if (!output.ok()) return output.status();
        result.ciphertexts = output.take_value();
        if (report) {
            report->rotations += counts.rotations;
            report->conjugations += counts.conjugations;
            report->ciphertext_plain_multiplications += counts.ciphertext_plain_multiplications;
            report->rescales += counts.rescales;
        }
    }
    result.parameter_profile = context.profile;
    result.logical_spec = MakeHeSpec(program->output_packing.logical_shape,
        output_layout, context.profile, level.value() + 1, context.parameters.default_scale_bits);
    result.logical_spec.numeric = input.logical_spec.numeric;
    result.physical_packing = program->output_packing;
    result.conjugated_complex = program->conjugate_output;
    return result;
}

struct HeadTile {
    std::size_t head_begin{};
    std::size_t head_count{};
    std::size_t lanes{};
    std::size_t input_groups{};
    std::size_t input_begin{};
    std::size_t output_groups{};
    std::size_t output_begin{};
};

std::vector<HeadTile> QkTiles(const TransformerLayerConfig &layer,
                              std::size_t slots)
{
    std::vector<HeadTile> result;
    std::size_t input_begin = 0;
    std::size_t output_begin = 0;
    for (std::size_t begin = 0; begin < layer.q_heads;) {
        const auto heads = detail::QkHeadsPerTile(
            layer.tokens, layer.q_heads - begin, slots);
        const auto lanes = slots / (detail::TokenColumnStride(layer.tokens) * heads);
        const auto input_groups =
            (layer.qk_head_width + 2 * lanes - 1) / (2 * lanes);
        const auto output_groups =
            (layer.tokens + 2 * lanes - 1) / (2 * lanes);
        result.push_back({begin, heads, lanes, input_groups, input_begin,
                          output_groups, output_begin});
        input_begin += input_groups;
        output_begin += output_groups;
        begin += heads;
    }
    return result;
}

Result<void> CollectProjectionRotationSteps(
    const TransformerLayerConfig &layer, const HePackingPlan &input_packing,
    Layout output_layout, std::size_t slots, std::set<int> &steps)
{
    auto input_valid = input_packing.Validate();
    if (!input_valid.ok()) return input_valid.status();
    auto columns_valid = ValidateRealColumnInput(input_packing, layer, slots);
    if (!columns_valid.ok()) return columns_valid.status();
    auto output = MultiHeadPackingPlan::ProjectionOutput(
        layer, output_layout, slots);
    if (!output.ok()) return output.status();
    const auto logical_heads = output_layout == Layout::kMultiHeadQ
        ? layer.q_heads : layer.kv_heads;
    const auto width = output_layout == Layout::kMultiHeadV
        ? layer.value_head_width : layer.qk_head_width;
    std::size_t stride = 1;
    while (stride < layer.tokens) stride *= 2;
    const auto columns = slots / stride;
    const bool full_column_cycle = layer.model_width >= columns && slots % stride == 0;
    std::vector<unsigned char> present(slots, 0);
    std::vector<unsigned char> residue_classes(full_column_cycle ? stride : 0, 0);
    for (std::size_t token = 0; token < layer.tokens; ++token)
        for (std::size_t head = 0; head < logical_heads; ++head)
            for (std::size_t dimension = 0; dimension < width; ++dimension) {
                const auto logical = (token * logical_heads + head) * width + dimension;
                for (const auto &destination : output.value().Locations(logical)) {
                    const auto first = (token + slots - destination.slot) % slots;
                    if (full_column_cycle) {
                        residue_classes[first % stride] = 1;
                    } else {
                        for (std::size_t feature = 0;
                             feature < std::min(layer.model_width, columns); ++feature)
                            present[(first + feature * stride) % slots] = 1;
                    }
                }
            }
    if (full_column_cycle)
        for (std::size_t residue = 0; residue < slots; ++residue)
            present[residue] = residue_classes[residue % stride];

    // All Q/K/V plans use one column split, so their baby cache is reusable.
    auto split = ComplexDiagonalProjectionPlanner::ColumnBsgs(layer, slots);
    if (!split.ok()) return split.status();
    const auto base = split.value().token_stride * split.value().baby_steps;
    for (std::size_t residue = 0; residue < slots; ++residue) {
        if (!present[residue]) continue;
        const auto giant = static_cast<int>((residue / base) * base);
        const auto baby = static_cast<int>(residue) - giant;
        const auto baby_step = CanonicalRotation(baby, slots);
        const auto giant_step = CanonicalRotation(giant, slots);
        if (baby_step != 0) steps.insert(baby_step);
        if (giant_step != 0) steps.insert(giant_step);
    }
    return {};
}

struct QkRoute {
    int rotation{};
    std::size_t output_group{};
    std::vector<std::size_t> lanes;
    PlainRef mask;
};

std::vector<QkRoute> QkRoutes(std::size_t diagonal, const HeadTile &tile,
                             const TransformerLayerConfig &layer,
                             std::size_t slots)
{
    const auto block = detail::TokenColumnStride(layer.tokens) * tile.head_count;
    std::map<std::pair<int, std::size_t>, std::vector<std::size_t>> routes;
    for (std::size_t lane = 0;
         lane < std::min(tile.lanes, layer.qk_head_width); ++lane) {
        const auto destination = (diagonal + lane) % layer.tokens;
        const auto output_lane = destination % tile.lanes;
        const int rotation = CanonicalRotation(
            (static_cast<long long>(lane) - static_cast<long long>(output_lane)) *
                static_cast<long long>(block), slots);
        routes[{rotation, destination / tile.lanes}].push_back(output_lane);
    }
    std::vector<QkRoute> result;
    for (auto &entry : routes)
        result.push_back({entry.first.first, entry.first.second,
                          std::move(entry.second), {}});
    return result;
}

MultiHeadQkKernel::TileBsgsPlan QkPlan(const TransformerLayerConfig &layer,
                                 std::size_t slots, const HeadTile &tile)
{
    MultiHeadQkKernel::TileBsgsPlan result;
    result.head_begin = tile.head_begin;
    result.heads_per_tile = tile.head_count;
    result.lanes = tile.lanes;
    result.input_groups = tile.input_groups;
    const auto rows = std::min({tile.lanes, layer.qk_head_width, layer.tokens});
    std::size_t preprocessing_rotations = std::numeric_limits<std::size_t>::max();
    for (std::size_t baby = 1; baby <= rows; ++baby) {
        const auto giant = (rows + baby - 1) / baby;
        const auto rotations = baby - 1 + 2 * (giant - 1) + (baby > 1 ? 1 : 0);
        if (rotations < preprocessing_rotations) {
            preprocessing_rotations = rotations;
            result.preprocessing_baby_steps = baby;
            result.preprocessing_giant_steps = giant;
        }
    }
    preprocessing_rotations *= tile.input_groups;
    std::vector<std::vector<QkRoute>> routes(layer.tokens);
    std::size_t lane_rotations = 0;
    for (std::size_t diagonal = 0; diagonal < layer.tokens; ++diagonal) {
        routes[diagonal] = QkRoutes(diagonal, tile, layer, slots);
        std::set<int> offsets;
        for (const auto &route : routes[diagonal])
            if (route.rotation != 0) offsets.insert(route.rotation);
        lane_rotations += offsets.size();
    }
    std::size_t best = std::numeric_limits<std::size_t>::max();
    for (std::size_t baby = 1; baby <= layer.tokens; ++baby) {
        if (layer.tokens % baby != 0) continue;
        const auto giant = layer.tokens / baby;
        std::size_t conjugations = 0, restores = 0;
        for (std::size_t j = 0; j < giant; ++j) {
            std::set<std::size_t> groups, pairs;
            for (std::size_t i = 0; i < baby; ++i)
                for (const auto &route : routes[j * baby + i]) {
                    groups.insert(route.output_group);
                    pairs.insert(route.output_group / 2);
                }
            conjugations += groups.size();
            if (j != 0) restores += 2 * pairs.size();
        }
        const auto rotations = preprocessing_rotations + lane_rotations + restores +
            2 * tile.input_groups * (baby + giant - 2);
        const auto cost = rotations + conjugations;
        if (cost < best) {
            best = cost;
            result.baby_steps = baby;
            result.giant_steps = giant;
            result.rotations = rotations;
            result.conjugations = conjugations;
        }
    }
    result.relinearizations = layer.tokens;
    result.ciphertext_multiplications = result.relinearizations * tile.input_groups;
    return result;
}

void CollectQkRotationSteps(const TransformerLayerConfig &layer,
                            std::size_t slots, std::set<int> &steps)
{
    const auto add = [&](long long shift) {
        const auto rotation = CanonicalRotation(shift, slots);
        if (rotation != 0) steps.insert(rotation);
    };
    for (const auto &tile : QkTiles(layer, slots)) {
        const auto plan = QkPlan(layer, slots, tile);
        for (std::size_t i = 1; i < plan.preprocessing_baby_steps; ++i) add(i);
        for (std::size_t j = 0; j < plan.preprocessing_giant_steps; ++j) {
            const auto shift = j * plan.preprocessing_baby_steps;
            add(shift);
            if (shift != 0 || plan.preprocessing_baby_steps > 1)
                add(static_cast<long long>(shift) - static_cast<long long>(layer.tokens));
        }
        for (std::size_t i = 1; i < plan.baby_steps; ++i) {
            add(i);
            add(static_cast<long long>(i) - layer.tokens);
        }
        for (std::size_t j = 1; j < plan.giant_steps; ++j) {
            const auto shift = static_cast<long long>(j * plan.baby_steps);
            add(-shift);
            add(static_cast<long long>(layer.tokens) - shift);
            add(shift);
            add(shift - static_cast<long long>(layer.tokens));
        }
        for (std::size_t diagonal = 0; diagonal < layer.tokens; ++diagonal)
            for (const auto &route : QkRoutes(diagonal, tile, layer, slots))
                add(route.rotation);
    }
}

std::vector<double> InnerRotationMask(const HeadTile &tile,
                                      const TransformerLayerConfig &layer,
                                      std::size_t slots,
                                      std::size_t diagonal,
                                      bool wrap)
{
    std::vector<double> mask(slots, 0.0);
    const auto block = detail::TokenColumnStride(layer.tokens) * tile.head_count;
    for (std::size_t lane = 0; lane < tile.lanes; ++lane) {
        for (std::size_t row = 0; row < layer.tokens; ++row) {
            const bool row_wrap = diagonal != 0 &&
                                  row >= layer.tokens - diagonal;
            if (row_wrap != wrap) continue;
            for (std::size_t head = 0; head < tile.head_count; ++head)
                mask[lane * block + head * detail::TokenColumnStride(layer.tokens) + row] = 1.0;
        }
    }
    return mask;
}

void AddCipher(seal::Ciphertext &sum, bool &initialized,
               seal::Ciphertext term, seal::Evaluator &evaluator)
{
    if (!initialized) {
        sum = std::move(term);
        initialized = true;
    } else {
        evaluator.add_inplace(sum, term);
    }
}

using InnerMasks = std::array<PlainRef, 2>;
struct QkPreprocessingTerm { std::size_t baby; PlainRef mask; };
struct QkPreprocessingGroup { int rotation; std::vector<QkPreprocessingTerm> terms; };
struct PreparedQkTile {
    HeadTile tile;
    MultiHeadQkKernel::TileBsgsPlan plan;
    std::vector<std::vector<QkRoute>> routes;
    std::vector<QkPreprocessingGroup> preprocessing;
    std::vector<InnerMasks> k_masks, q_masks;
    // [giant][real/imaginary phase][normal/wrap branch]
    std::vector<std::array<InnerMasks, 2>> restore_masks;
};
struct PreparedQk final : PreparedBlock2Program {
    std::string profile;
    seal::parms_id_type input_id{};
    double q_scale{}, k_scale{}, k_preprocessed_scale{};
    std::size_t tokens{}, heads{}, kv_heads{}, width{};
    HePackingPlan q_packing, k_packing, output_packing;
    std::vector<PreparedQkTile> tiles;
};

std::shared_ptr<const PreparedQk> CompileQk(
    const TransformerLayerConfig &layer, const HeContext &context,
    double q_scale, double k_scale)
{
    const auto cache = ModelPlaintextCacheScope::Current();
    // These masks depend on geometry and the HE profile, never model weights.
    // Identical layers share their encoded masks instead of duplicating them.
    ModelPlaintextCacheScope mask_scope(cache.first, "shared/" + context.profile);
    auto program = std::make_shared<PreparedQk>();
    program->profile = context.profile;
    program->tokens = layer.tokens;
    program->heads = layer.q_heads;
    program->kv_heads = layer.kv_heads;
    program->width = layer.qk_head_width;
    program->q_scale = q_scale;
    program->k_scale = k_scale;
    const auto levels = Levels(context);
    const auto terminal = context.parameters.coefficient_modulus_bits.size() -
        context.parameters.terminal_prime_count;
    if (terminal < MultiHeadQkKernel::kDepth)
        throw std::invalid_argument("QK suffix requires four remaining CKKS levels");
    const auto start = terminal - MultiHeadQkKernel::kDepth;
    program->input_id = levels.at(start)->parms_id();
    const double target = std::ldexp(1.0, context.parameters.default_scale_bits);
    program->k_preprocessed_scale = target * DroppedPrime(context, levels.at(start + 1)->parms_id()) / q_scale;
    const double total_mask_scale = program->k_preprocessed_scale * DroppedPrime(context, program->input_id) / k_scale;
    const double first_scale = std::sqrt(total_mask_scale);
    const double second_scale = total_mask_scale / first_scale;
    auto q_pack = MultiHeadPackingPlan::ProjectionOutput(layer, Layout::kMultiHeadQ, context.slot_count());
    auto k_pack = MultiHeadPackingPlan::ProjectionOutput(layer, Layout::kMultiHeadK, context.slot_count());
    auto out_pack = MultiHeadPackingPlan::AttentionDiagonal(layer, context.slot_count());
    if (!q_pack.ok() || !k_pack.ok() || !out_pack.ok())
        throw std::invalid_argument("QK compilation could not construct its packing");
    program->q_packing = q_pack.take_value();
    program->k_packing = k_pack.take_value();
    program->output_packing = out_pack.take_value();
    seal::CKKSEncoder encoder(*context.context);
    const auto slots = context.slot_count();
    const auto stride = detail::TokenColumnStride(layer.tokens);
    for (const auto &tile : QkTiles(layer, slots)) {
        PreparedQkTile compiled;
        compiled.tile = tile;
        compiled.plan = QkPlan(layer, slots, tile);
        const auto &plan = compiled.plan;
        const auto block = stride * tile.head_count;
        const auto rows = std::min({tile.lanes, layer.qk_head_width, layer.tokens});
        for (std::size_t j = 0; j < plan.preprocessing_giant_steps; ++j)
            for (const bool wrap : {false, true}) {
                QkPreprocessingGroup group;
                group.rotation = static_cast<int>(j * plan.preprocessing_baby_steps) -
                    (wrap ? static_cast<int>(layer.tokens) : 0);
                for (std::size_t i = 0; i < plan.preprocessing_baby_steps; ++i) {
                    const auto offset = j * plan.preprocessing_baby_steps + i;
                    if (offset >= rows || (wrap && offset == 0)) continue;
                    std::vector<double> mask(slots);
                    for (std::size_t lane = 0; lane < std::min(tile.lanes, layer.qk_head_width); ++lane) {
                        if (lane % layer.tokens != offset) continue;
                        for (std::size_t head = 0; head < tile.head_count; ++head)
                            for (std::size_t row = 0; row < layer.tokens; ++row) {
                                if ((row >= layer.tokens - offset) != wrap) continue;
                                const auto destination = lane * block + head * stride + row;
                                const auto adjusted = (static_cast<long long>(destination) + group.rotation +
                                    static_cast<long long>(slots)) % static_cast<long long>(slots);
                                mask[static_cast<std::size_t>(adjusted)] = 1.0;
                            }
                    }
                    group.terms.push_back({i, CompileFixedPlain(encoder, mask,
                        program->input_id, first_scale, "block2.qk.mask")});
                }
                if (!group.terms.empty()) compiled.preprocessing.push_back(std::move(group));
            }
        auto inner_masks = [&](std::size_t shift, double scale) {
            InnerMasks masks;
            for (const bool wrap : {false, true}) {
                if (wrap && shift == 0) continue;
                masks[wrap ? 1 : 0] = CompileFixedPlain(encoder,
                    InnerRotationMask(tile, layer, slots, shift, wrap),
                    program->input_id, scale, "block2.qk.mask");
            }
            return masks;
        };
        for (std::size_t i = 0; i < plan.baby_steps; ++i)
            compiled.k_masks.push_back(inner_masks(i, second_scale));
        compiled.q_masks.resize(plan.giant_steps);
        for (std::size_t j = 1; j < plan.giant_steps; ++j)
            compiled.q_masks[j] = inner_masks(layer.tokens - j * plan.baby_steps,
                target * DroppedPrime(context, program->input_id) / q_scale);
        for (std::size_t delta = 0; delta < layer.tokens; ++delta) {
            auto routes = QkRoutes(delta, tile, layer, slots);
            for (auto &route : routes) {
                std::vector<double> mask(slots);
                for (const auto lane : route.lanes)
                    for (std::size_t head = 0; head < tile.head_count; ++head)
                        for (std::size_t row = 0; row < layer.tokens; ++row)
                            mask[lane * block + head * stride + row] = 1.0;
                const auto id = levels.at(start + 2)->parms_id();
                route.mask = CompileFixedPlain(encoder, mask, id, DroppedPrime(context, id), "block2.qk.mask");
            }
            compiled.routes.push_back(std::move(routes));
        }
        compiled.restore_masks.resize(plan.giant_steps);
        for (std::size_t j = 0; j < plan.giant_steps; ++j) {
            const auto giant = j * plan.baby_steps;
            for (std::size_t parity = 0; parity < 2; ++parity)
                for (const bool wrap : {false, true}) {
                    if (wrap && giant == 0) continue;
                    const Complex phase = parity == 0 ? Complex{1.0, 0.0} : Complex{0.0, 1.0};
                    std::vector<Complex> mask(slots);
                    for (std::size_t lane = 0; lane < tile.lanes; ++lane)
                        for (std::size_t head = 0; head < tile.head_count; ++head)
                            for (std::size_t row = 0; row < layer.tokens; ++row) {
                                if ((row >= layer.tokens - giant) != wrap) continue;
                                const auto dest = lane * block + head * stride + row;
                                const auto rotation = static_cast<long long>(giant) -
                                    (wrap ? static_cast<long long>(layer.tokens) : 0);
                                const auto source = (static_cast<long long>(dest) + rotation +
                                    static_cast<long long>(slots)) % static_cast<long long>(slots);
                                mask[static_cast<std::size_t>(source)] = phase;
                            }
                    const auto id = levels.at(start + 3)->parms_id();
                    auto encoded = CompileModelPlaintext(encoder, mask, id, DroppedPrime(context, id), "block2.qk.mask");
                    if (!encoded.ok()) throw std::invalid_argument(encoded.status().message());
                    compiled.restore_masks[j][parity][wrap ? 1 : 0] = encoded.take_value();
                }
        }
        program->tiles.push_back(std::move(compiled));
    }
    return program;
}

seal::Ciphertext InnerRotateQk(
    const seal::Ciphertext &input, long long rotation,
    const TransformerLayerConfig &layer, const HeContext &context,
    seal::Evaluator &evaluator, const InnerMasks &masks,
    double output_scale, Block2ExecutionReport *report)
{
    rotation %= static_cast<long long>(layer.tokens);
    if (rotation < 0) rotation += layer.tokens;
    const auto shift = static_cast<std::size_t>(rotation);
    seal::Ciphertext result;
    bool initialized = false;
    for (const bool wrap : {false, true}) {
        if (wrap && shift == 0) continue;
        auto rotated = RotateDirect(input,
            static_cast<int>(shift) - (wrap ? static_cast<int>(layer.tokens) : 0), context, evaluator, report);
        evaluator.multiply_plain_inplace(rotated, *masks[wrap ? 1 : 0]);
        if (report) ++report->ciphertext_plain_multiplications;
        AddCipher(result, initialized, std::move(rotated), evaluator);
    }
    evaluator.rescale_to_next_inplace(result);
    NormalizeScale(result, output_scale);
    if (report) ++report->rescales;
    return result;
}

seal::Ciphertext PreprocessQkK(
    const seal::Ciphertext &input, const PreparedQkTile &compiled,
    const HeContext &context, seal::Evaluator &evaluator,
    Block2ExecutionReport *report)
{
    std::vector<seal::Ciphertext> babies(compiled.plan.preprocessing_baby_steps);
    for (std::size_t i = 1; i < compiled.plan.preprocessing_baby_steps; ++i)
        babies[i] = RotateDirect(input, static_cast<int>(i), context, evaluator, report);
    seal::Ciphertext result;
    bool initialized = false;
    for (const auto &group : compiled.preprocessing) {
        seal::Ciphertext sum;
        bool sum_initialized = false;
        for (const auto &term : group.terms) {
            seal::Ciphertext product;
            evaluator.multiply_plain(term.baby == 0 ? input : babies[term.baby], *term.mask, product);
            if (report) ++report->ciphertext_plain_multiplications;
            AddCipher(sum, sum_initialized, std::move(product), evaluator);
        }
        if (group.rotation != 0) {
            evaluator.rotate_vector_inplace(sum, group.rotation, context.galois_keys);
            if (report) ++report->rotations;
        }
        AddCipher(result, initialized, std::move(sum), evaluator);
    }
    return result;
}

Result<SealCipherTensor> ComputeQk(
    const SealCipherTensor &q, const SealCipherTensor &k,
    const TransformerLayerConfig &layer, const HeContext &context,
    Block2ExecutionReport *report, std::size_t he_workers)
{
    auto q_valid = q.Validate(context);
    if (!q_valid.ok()) return q_valid.status();
    auto k_valid = k.Validate(context);
    if (!k_valid.ok()) return k_valid.status();
    if (q.logical_spec.layout != Layout::kMultiHeadQ ||
        k.logical_spec.layout != Layout::kMultiHeadK)
        return Status::Invalid("QK requires native multihead Q/K packing");
    const auto cache = ModelPlaintextCacheScope::Current();
    std::shared_ptr<const PreparedQk> program;
    if (cache.first) {
        auto found = cache.first->GetPreparedBlock2Program(cache.second + "/block2/qk");
        if (!found.ok()) return found.status();
        program = std::dynamic_pointer_cast<const PreparedQk>(found.value());
        if (!program || program->profile != context.profile || program->tokens != layer.tokens ||
            program->heads != layer.q_heads || program->kv_heads != layer.kv_heads ||
            program->width != layer.qk_head_width)
            return Status::FailedPrecondition("prepared QK shape/profile mismatch");
    } else {
        program = CompileQk(layer, context, q.ciphertexts.front().scale(), k.ciphertexts.front().scale());
    }
    if (!SamePreparedPacking(q.physical_packing, program->q_packing) ||
        !SamePreparedPacking(k.physical_packing, program->k_packing))
        return Status::Invalid("QK input does not match its head-group packing decision");
    if (program->input_id != q.ciphertexts.front().parms_id() ||
        program->input_id != k.ciphertexts.front().parms_id() ||
        !CloseScale(program->q_scale, q.ciphertexts.front().scale()) ||
        !CloseScale(program->k_scale, k.ciphertexts.front().scale()))
        return Status::FailedPrecondition("prepared QK input level/scale mismatch");
    const auto q_level = LevelOf(context, q.ciphertexts.front().parms_id());
    const auto k_level = LevelOf(context, k.ciphertexts.front().parms_id());
    if (!q_level.ok()) return q_level.status();
    if (!k_level.ok()) return k_level.status();
    const auto levels = Levels(context);
    const auto terminal_level =
        context.parameters.coefficient_modulus_bits.size() -
        context.parameters.terminal_prime_count;
    constexpr auto kQkDepth = MultiHeadQkKernel::kDepth;
    if (terminal_level < kQkDepth)
        return Status::FailedPrecondition(
            "QK suffix requires four remaining CKKS levels");
    const auto qk_start_level = terminal_level - kQkDepth;
    if (q_level.value() != qk_start_level ||
        k_level.value() != qk_start_level)
        return Status::FailedPrecondition(
            "QK inputs must start exactly four levels before terminal");
    if (qk_start_level + 1 >= levels.size())
        return Status::FailedPrecondition("QK suffix exceeds modulus chain");
    const double scale44 = std::ldexp(
        1.0, context.parameters.default_scale_bits);
    // K is rescaled once before the ct-ct product. Choose its resulting scale
    // from the *next* prime (the one dropped by that product), not merely from
    // the bit width of the current prime. This keeps arbitrary configured RNS
    // primes numerically aligned without rewriting scale metadata.
    const double k_preprocessed_scale = program->k_preprocessed_scale;
    seal::Evaluator evaluator(*context.context);
    std::vector<seal::Ciphertext> unpaired;
    std::size_t expected_input = 0;
    for (const auto &compiled : program->tiles) expected_input += compiled.tile.input_groups;
    if (q.ciphertexts.size() != expected_input ||
        k.ciphertexts.size() != expected_input)
        return Status::Invalid("QK ciphertext count does not match packing plan");

    if (q.conjugated_complex || !k.conjugated_complex)
        return Status::Invalid("QK requires standard Q and conjugated K from Wk preprocessing");
    for (const auto *tensor : {&q, &k})
        for (const auto &ct : tensor->ciphertexts)
            if (ct.parms_id() != tensor->ciphertexts.front().parms_id() ||
                !CloseScale(ct.scale(), scale44) || ct.size() != 2)
                return Status::Invalid("QK requires size-two inputs with consistent levels and default scales");

    for (const auto &compiled : program->tiles) {
        const auto &tile = compiled.tile;
        const auto &plan = compiled.plan;
        const auto &routes = compiled.routes;
        const auto active_primes = context.context->get_context_data(
            q.ciphertexts.front().parms_id())->parms().coeff_modulus().size();
        const auto worker_bytes = [&](std::size_t live) {
            return detail::EstimatedHeWorkerBytes(
                context.parameters.polynomial_degree, active_primes, live);
        };
        // Prepare each K base once, then rotate independent (baby, input)
        // pairs. Pre-sized outputs and task-local counts avoid shared mutation.
        std::vector<seal::Ciphertext> k_preprocessed(tile.input_groups);
        std::vector<Block2ExecutionReport> preparation_reports(tile.input_groups);
        std::size_t preparation_workers = 0;
        auto prepared = detail::ParallelRanges(tile.input_groups, he_workers, 1,
            [&](std::size_t begin, std::size_t end) -> Result<void> {
                seal::Evaluator worker(*context.context);
                for (auto group = begin; group < end; ++group)
                    k_preprocessed[group] = PreprocessQkK(
                        k.ciphertexts[tile.input_begin + group], compiled,
                        context, worker, &preparation_reports[group]);
                return {};
            }, worker_bytes(plan.preprocessing_baby_steps + 4), &preparation_workers);
        if (!prepared.ok()) return prepared.status();
        if (report) report->qk_preparation_workers = std::max(
            report->qk_preparation_workers, preparation_workers);
        std::vector<std::vector<seal::Ciphertext>> k_babies(plan.baby_steps);
        for (auto &baby : k_babies) baby.resize(tile.input_groups);
        std::vector<Block2ExecutionReport> baby_reports(plan.baby_steps * tile.input_groups);
        prepared = detail::ParallelRanges(baby_reports.size(), he_workers, 1,
            [&](std::size_t begin, std::size_t end) -> Result<void> {
                seal::Evaluator worker(*context.context);
                for (auto job = begin; job < end; ++job) {
                    const auto baby = job / tile.input_groups;
                    const auto group = job % tile.input_groups;
                    k_babies[baby][group] = InnerRotateQk(
                        k_preprocessed[group], static_cast<long long>(baby),
                        layer, context, worker, compiled.k_masks[baby],
                        k_preprocessed_scale, &baby_reports[job]);
                }
                return {};
            }, worker_bytes(4), &preparation_workers);
        if (!prepared.ok()) return prepared.status();
        k_preprocessed.clear();
        if (report) {
            report->qk_preparation_workers = std::max(
                report->qk_preparation_workers, preparation_workers);
            for (const auto *reports : {&preparation_reports, &baby_reports})
                for (const auto &counts : *reports) {
                    report->rotations += counts.rotations;
                    report->ciphertext_plain_multiplications += counts.ciphertext_plain_multiplications;
                    report->rescales += counts.rescales;
                }
        }

        std::vector<std::vector<seal::Ciphertext>> giant_outputs(plan.giant_steps);
        std::vector<Block2ExecutionReport> giant_reports(plan.giant_steps);
        std::size_t product_workers = 0;
        auto ready = detail::ParallelRanges(
            plan.giant_steps, he_workers, 1,
            [&](std::size_t begin, std::size_t end) -> Result<void> {
                seal::Evaluator worker(*context.context);
                for (std::size_t j = begin; j < end; ++j) {
                    auto *counts = &giant_reports[j];
                    const auto giant = j * plan.baby_steps;
                    std::vector<seal::Ciphertext> q_rotated;
                    for (std::size_t g = 0; g < tile.input_groups; ++g) {
                        const auto &source = q.ciphertexts[tile.input_begin + g];
                        if (giant == 0) {
                            auto copy = source;
                            worker.mod_switch_to_inplace(copy, k_babies[0][g].parms_id());
                            q_rotated.push_back(std::move(copy));
                        } else {
                            q_rotated.push_back(InnerRotateQk(source,
                                -static_cast<long long>(giant), layer, context, worker,
                                compiled.q_masks[j], scale44, counts));
                        }
                    }
                    std::map<std::size_t, seal::Ciphertext> diagonal_groups;
                    for (std::size_t i = 0; i < plan.baby_steps; ++i) {
                        seal::Ciphertext product;
                        bool product_initialized = false;
                        for (std::size_t g = 0; g < tile.input_groups; ++g) {
                            seal::Ciphertext term;
                            worker.multiply(q_rotated[g], k_babies[i][g], term);
                            ++counts->ciphertext_multiplications;
                            AddCipher(product, product_initialized, std::move(term), worker);
                        }
                        worker.relinearize_inplace(product, context.relinearization_keys);
                        worker.rescale_to_next_inplace(product);
                        NormalizeScale(product, scale44);
                        ++counts->relinearizations;
                        ++counts->rescales;
                        std::map<int, seal::Ciphertext> aligned;
                        for (const auto &route : routes[giant + i]) {
                            const seal::Ciphertext *source = &product;
                            if (route.rotation != 0) {
                                auto found = aligned.find(route.rotation);
                                if (found == aligned.end())
                                    found = aligned.emplace(route.rotation,
                                        RotateDirect(product, route.rotation, context, worker, counts)).first;
                                source = &found->second;
                            }
                            seal::Ciphertext term;
                            worker.multiply_plain(*source, *route.mask, term);
                            ++counts->ciphertext_plain_multiplications;
                            const auto inserted = diagonal_groups.try_emplace(route.output_group);
                            if (inserted.second) inserted.first->second = std::move(term);
                            else worker.add_inplace(inserted.first->second, term);
                        }
                    }
                    for (auto &entry : diagonal_groups) {
                        worker.rescale_to_next_inplace(entry.second);
                        NormalizeScale(entry.second, scale44);
                        ++counts->rescales;
                        seal::Ciphertext conjugate;
                        worker.complex_conjugate(entry.second, context.galois_keys, conjugate);
                        worker.add_inplace(entry.second, conjugate);
                        ++counts->conjugations;
                    }

                    // Fuse real/imaginary output phases with the row-restore
                    // masks. Add groups BEFORE each giant automorphism.
                    std::map<std::pair<std::size_t, bool>, seal::Ciphertext> branches;
                    for (const auto &entry : diagonal_groups) {
                        for (const bool wrap : {false, true}) {
                            if (wrap && giant == 0) continue;
                            seal::Ciphertext term;
                            worker.multiply_plain(entry.second,
                                *compiled.restore_masks[j][entry.first % 2][wrap ? 1 : 0], term);
                            ++counts->ciphertext_plain_multiplications;
                            const auto inserted = branches.try_emplace(std::make_pair(entry.first / 2, wrap));
                            if (inserted.second) inserted.first->second = std::move(term);
                            else worker.add_inplace(inserted.first->second, term);
                        }
                    }
                    std::vector<seal::Ciphertext> outputs(tile.output_groups);
                    std::vector<bool> initialized(tile.output_groups, false);
                    for (auto &entry : branches) {
                        const auto rotation = static_cast<int>(giant) -
                            (entry.first.second ? static_cast<int>(layer.tokens) : 0);
                        if (rotation != 0) {
                            worker.rotate_vector_inplace(entry.second, rotation, context.galois_keys);
                            ++counts->rotations;
                        }
                        auto term = std::move(entry.second);
                        const auto ct = entry.first.first;
                        if (!initialized[ct]) {
                            outputs[ct] = std::move(term);
                            initialized[ct] = true;
                        } else worker.add_inplace(outputs[ct], term);
                    }
                    // Keep the common restore-mask scale until all giant
                    // contributions to each output have been accumulated.
                    giant_outputs[j] = std::move(outputs);
                }
                return {};
            }, worker_bytes(
                // Use the active input level, retaining headroom for size-three
                // products and key-switch scratch; K babies are shared.
                16 + tile.input_groups + 5 * tile.output_groups), &product_workers);
        if (!ready.ok()) return ready.status();
        if (report) report->qk_product_workers = std::max(
            report->qk_product_workers, product_workers);
        std::vector<seal::Ciphertext> output(tile.output_groups);
        std::vector<bool> initialized(tile.output_groups, false);
        for (std::size_t j = 0; j < plan.giant_steps; ++j) {
            for (std::size_t ct = 0; ct < output.size(); ++ct) {
                if (giant_outputs[j][ct].size() == 0) continue;
                if (!initialized[ct]) {
                    output[ct] = std::move(giant_outputs[j][ct]);
                    initialized[ct] = true;
                } else evaluator.add_inplace(output[ct], giant_outputs[j][ct]);
            }
            if (report) {
                const auto &counts = giant_reports[j];
                report->rotations += counts.rotations;
                report->conjugations += counts.conjugations;
                report->ciphertext_plain_multiplications += counts.ciphertext_plain_multiplications;
                report->ciphertext_multiplications += counts.ciphertext_multiplications;
                report->relinearizations += counts.relinearizations;
                report->rescales += counts.rescales;
            }
        }
        for (std::size_t ct = 0; ct < output.size(); ++ct) {
            if (!initialized[ct]) return Status::Internal("QK output group is empty");
            evaluator.rescale_to_next_inplace(output[ct]);
            NormalizeScale(output[ct], scale44);
            if (report) ++report->rescales;
            unpaired.push_back(std::move(output[ct]));
        }
    }

    const auto &output_plan = program->output_packing;
    if (unpaired.size() != output_plan.ciphertext_count)
        return Status::Internal("QK output count differs from compiled packing");
    SealCipherTensor result;
    result.parameter_profile = context.profile;
    result.ciphertexts = std::move(unpaired);
    result.logical_spec = MakeHeSpec(
        {{layer.q_heads, layer.tokens, layer.tokens}},
        Layout::kAttentionDiagonal, context.profile, terminal_level,
        context.parameters.output_scale_bits);
    result.logical_spec.numeric = q.logical_spec.numeric;
    result.physical_packing = output_plan;
    return result;
}

} // namespace

Result<void> Block2LevelPlanner::Validate(
    const HeParameterConfig &parameters)
{
    if (parameters.coefficient_modulus_bits.size() < 7 ||
        parameters.declared_depth < 6)
        return Status::FailedPrecondition(
            "Block 2 requires at least seven data primes and depth six");
    if (parameters.terminal_prime_count != 1)
        return Status::FailedPrecondition(
            "Block 2 terminal conversion boundary requires one prime");
    if (parameters.input_scale_bits != parameters.default_scale_bits ||
        parameters.output_scale_bits != parameters.default_scale_bits)
        return Status::FailedPrecondition(
            "Block 2 currently requires one configured CKKS scale");
    return {};
}

Result<SealCipherTensor> LayerNormTailKernel::Evaluate(
    const SealCipherTensor &centered,
    const SealCipherTensor &inverse_stddev,
    const GammaFusedLayerNormParameters &affine,
    const TransformerLayerConfig &layer,
    const HeContext &context,
    Block2ExecutionReport *report,
    const GammaFusedLayerNormParameters *original_affine,
    SealCipherTensor *normalized_residual, std::size_t he_workers)
{
    try {
        return LayerNormTail(centered, inverse_stddev, affine, layer, context, report,
                             original_affine, normalized_residual, he_workers);
    } catch (const std::exception &error) {
        return Status::Invalid(std::string("LayerNorm tail failed: ") + error.what());
    }
}

Result<std::vector<int>> ComplexDiagonalProjectionKernel::RequiredRotationSteps(
    const TransformerLayerConfig &layer,
    const HePackingPlan &input_packing, Layout output_layout)
{
    std::set<int> steps;
    auto collected = CollectProjectionRotationSteps(
        layer, input_packing, output_layout, input_packing.slot_count, steps);
    if (!collected.ok()) return collected.status();
    return std::vector<int>(steps.begin(), steps.end());
}

class ComplexDiagonalProjectionKernel::Impl {
public:
    Impl(const TransformerLayerConfig &configured_layer,
         const HeContext &configured_context,
         Block2ExecutionReport *configured_report, std::size_t configured_workers)
        : layer(configured_layer), context(configured_context),
          report(configured_report), he_workers(configured_workers)
    {}

    const TransformerLayerConfig &layer;
    const HeContext &context;
    Block2ExecutionReport *report;
    std::size_t he_workers;
    PreparedProjectionRotationCache prepared_rotations;
    // A cached rotation is valid only for the exact immutable source batch.
    std::vector<std::tuple<const std::uint64_t *, seal::parms_id_type, double, std::uint64_t>>
        source_identity;
};

ComplexDiagonalProjectionKernel::ComplexDiagonalProjectionKernel(
    const TransformerLayerConfig &layer, const HeContext &context,
    Block2ExecutionReport *report, std::size_t he_workers)
    : impl_(std::make_unique<Impl>(layer, context, report, he_workers))
{}

ComplexDiagonalProjectionKernel::~ComplexDiagonalProjectionKernel() = default;
ComplexDiagonalProjectionKernel::ComplexDiagonalProjectionKernel(
    ComplexDiagonalProjectionKernel &&) noexcept = default;
ComplexDiagonalProjectionKernel &
ComplexDiagonalProjectionKernel::operator=(
    ComplexDiagonalProjectionKernel &&) noexcept = default;

Result<SealCipherTensor> ComplexDiagonalProjectionKernel::Evaluate(
    const SealCipherTensor &input,
    const DenseMatrixParameter &weights,
    const std::optional<DenseVectorParameter> &bias,
    Layout output_layout)
{
    decltype(impl_->source_identity) identity;
    identity.reserve(input.ciphertexts.size());
    for (const auto &ct : input.ciphertexts) {
        if (ct.size() != 0 && !seal::is_buffer_valid(ct))
            return Status::Invalid("projection source ciphertext buffer is invalid");
        std::uint64_t digest = 1469598103934665603ULL;
        const auto words = ct.size() * ct.poly_modulus_degree() * ct.coeff_modulus_size();
        for (std::size_t i = 0; i < words; ++i) digest = (digest ^ ct.data()[i]) * 1099511628211ULL;
        identity.emplace_back(ct.data(), ct.parms_id(), ct.scale(), digest);
    }
    if (identity != impl_->source_identity) {
        impl_->prepared_rotations = {};
        impl_->source_identity = std::move(identity);
    }
    return Project(input, weights, bias, output_layout, impl_->layer,
                   impl_->context, impl_->prepared_rotations, impl_->report, impl_->he_workers);
}

Result<void> MultiHeadQkKernel::Prepare(
    const TransformerLayerConfig &layer, const HeContext &context)
{
    auto valid = ValidateLayerConfig(layer);
    if (!valid.ok()) return valid.status();
    if (detail::TokenColumnStride(layer.tokens) > context.slot_count())
        return Status::Invalid("one QK token column does not fit in CKKS slots");
    const auto cache = ModelPlaintextCacheScope::Current();
    if (!cache.first || !cache.first->preparing_plaintexts())
        return Status::FailedPrecondition("QK compilation requires Prepare()");
    try {
        const auto shared_key = "shared/" + context.profile + "/block2/qk/" +
            std::to_string(layer.tokens) + "/" + std::to_string(layer.q_heads) + "/" +
            std::to_string(layer.kv_heads) + "/" + std::to_string(layer.qk_head_width);
        auto found = cache.first->GetPreparedBlock2Program(shared_key);
        std::shared_ptr<const PreparedBlock2Program> program;
        if (found.ok()) program = found.take_value();
        else {
            const double scale = std::ldexp(1.0, context.parameters.default_scale_bits);
            program = CompileQk(layer, context, scale, scale);
            auto stored = cache.first->PutPreparedBlock2Program(shared_key, program);
            if (!stored.ok()) return stored.status();
        }
        return cache.first->PutPreparedBlock2Program(cache.second + "/block2/qk", program);
    } catch (const std::exception &error) {
        return Status::Invalid(std::string("QK compilation failed: ") + error.what());
    }
}

Result<SealCipherTensor> MultiHeadQkKernel::Evaluate(
    const SealCipherTensor &q, const SealCipherTensor &k,
    const TransformerLayerConfig &layer, const HeContext &context,
    Block2ExecutionReport *report, std::size_t he_workers)
{
    try {
        return ComputeQk(q, k, layer, context, report, he_workers);
    } catch (const std::exception &error) {
        return Status::Invalid(std::string("QK failed: ") + error.what());
    }
}

Result<MultiHeadQkKernel::BsgsPlan> MultiHeadQkKernel::Plan(
    const TransformerLayerConfig &layer, std::size_t slot_count)
{
    auto valid = ValidateLayerConfig(layer);
    if (!valid.ok()) return valid.status();
    if (detail::TokenColumnStride(layer.tokens) > slot_count)
        return Status::Invalid("one QK token column does not fit in CKKS slots");
    const auto tiles = QkTiles(layer, slot_count);
    BsgsPlan result;
    for (const auto &tile : tiles) {
        auto plan = QkPlan(layer, slot_count, tile);
        result.rotations += plan.rotations;
        result.conjugations += plan.conjugations;
        result.relinearizations += plan.relinearizations;
        result.ciphertext_multiplications += plan.ciphertext_multiplications;
        result.head_tiles.push_back(std::move(plan));
    }
    return result;
}

Result<std::vector<int>> MultiHeadQkKernel::RequiredRotationSteps(
    const TransformerLayerConfig &layer, std::size_t slot_count)
{
    auto valid = ValidateLayerConfig(layer);
    if (!valid.ok()) return valid.status();
    if (detail::TokenColumnStride(layer.tokens) > slot_count)
        return Status::Invalid("one QK token column does not fit in CKKS slots");
    std::set<int> steps{0};
    CollectQkRotationSteps(layer, slot_count, steps);
    return std::vector<int>(steps.begin(), steps.end());
}

SealBlock2Executor::SealBlock2Executor(
    const HeraConfig &config, const TransformerLayerConfig &layer,
    HeContextRegistry &contexts, SealCipherTensorStore &tensors,
    SealPlainParameterStore &parameters,
    const std::vector<int> *required_rotations)
    : config_(config), layer_(layer), contexts_(contexts), tensors_(tensors),
      parameters_(parameters), required_rotations_(required_rotations)
{}

Result<std::vector<int>> SealBlock2Executor::RequiredRotationSteps(
    const TransformerLayerConfig &layer,
    const HeParameterConfig &parameters)
{
    const auto slots = parameters.polynomial_degree / 2;
    auto input = MultiHeadPackingPlan::RealStrictColumns(
        {{layer.tokens, layer.model_width}}, slots);
    if (!input.ok()) return input.status();
    return RequiredRotationSteps(layer, parameters, input.value());
}

Result<std::vector<int>> SealBlock2Executor::RequiredRotationSteps(
    const TransformerLayerConfig &layer,
    const HeParameterConfig &parameters,
    const HePackingPlan &input_packing)
{
    auto valid = Block2LevelPlanner::Validate(parameters);
    if (!valid.ok()) return valid.status();
    auto layer_valid = ValidateLayerConfig(layer);
    if (!layer_valid.ok()) return layer_valid.status();
    const auto slots = parameters.polynomial_degree / 2;
    if (layer.tokens > slots)
        return Status::Invalid("Block 2 token dimension exceeds CKKS slots");
    // Step zero requests SEAL's conjugation key. Every non-zero element is
    // an exact rotate_vector offset used by the kernels below.
    std::set<int> steps{0, static_cast<int>(slots / 2)};
    for (const auto layout : {Layout::kMultiHeadQ, Layout::kMultiHeadK,
                              Layout::kMultiHeadV}) {
        auto collected = CollectProjectionRotationSteps(
            layer, input_packing, layout, slots, steps);
        if (!collected.ok()) return collected.status();
    }
    CollectQkRotationSteps(layer, slots, steps);
    return std::vector<int>(steps.begin(), steps.end());
}

Result<Block2HeOutputs> SealBlock2Executor::Evaluate(
    const Block2HeInputs &inputs, const LayerWeightHandles &weights,
    Block2ExecutionReport *report, SealCipherTensor *normalized_residual)
{
    try {
        if (weights.layer_id != layer_.id)
            return Status::Invalid("Block 2 weight layer ID mismatch");
        const auto *execution = FindLayerExecution(config_, layer_.id);
        if (!execution)
            return Status::Invalid("Block 2 layer has no HE placement");
        const auto region = execution->regions.find(HeRegionId::kRegion2);
        if (region == execution->regions.end())
            return Status::Invalid("Block 2 region is not configured");
        const auto &context = contexts_.Require(
            region->second.parameter_profile, contexts_.role());
        auto level_valid = Block2LevelPlanner::Validate(context.parameters);
        if (!level_valid.ok()) return level_valid.status();
        if (report) {
            if (required_rotations_) {
                report->required_rotation_steps = *required_rotations_;
            } else {
                auto required = RequiredRotationSteps(layer_,
                                                      context.parameters);
                if (!required.ok()) return required.status();
                report->required_rotation_steps = required.take_value();
            }
        }
        if (context.galois_keys.size() == 0)
            return Status::FailedPrecondition(
                "Block 2 Galois keys have not been compiled");
        if (report) {
            const auto rotations = std::move(report->required_rotation_steps);
            *report = {};
            report->required_rotation_steps = rotations;
        }

        auto schema = DeriveWeightSchema(layer_);
        if (!schema.ok()) return schema.status();
        for (const auto &entry : std::vector<std::pair<const WeightHandle *,
                                                       const MatrixSpec *>>{
                 {&weights.wq, &schema.value().wq},
                 {&weights.wk, &schema.value().wk},
                 {&weights.wv, &schema.value().wv}}) {
            auto valid = ValidateMatrixHandle(*entry.first, *entry.second);
            if (!valid.ok()) return valid.status();
        }
        auto affine = parameters_.GetGammaFusedLayerNorm(weights.wq.id);
        if (!affine.ok()) return affine.status();
        for (auto id : {weights.wk.id, weights.wv.id}) {
            auto same = parameters_.GetGammaFusedLayerNorm(id);
            if (!same.ok()) return same.status();
            if (same.value() != affine.value())
                return Status::Invalid("Q/K/V use different LN1 gamma fusion");
        }
        const auto load_bias = [&](std::string_view name)
            -> Result<std::optional<DenseVectorParameter>> {
            if (!layer_.qkv_bias) return std::optional<DenseVectorParameter>{};
            for (const auto &handle : weights.plain_parameters)
                if (handle.spec.name == name) {
                    auto value = parameters_.GetVector(handle.id);
                    if (!value.ok()) return value.status();
                    return std::optional<DenseVectorParameter>{value.take_value()};
                }
            return Status::Invalid(std::string(name) + " bias handle is absent");
        };
        auto bq = load_bias("bq");
        if (!bq.ok()) return bq.status();
        auto bk = load_bias("bk");
        if (!bk.ok()) return bk.status();
        auto bv = load_bias("bv");
        if (!bv.ok()) return bv.status();

        auto project_qkv = [&]() -> Result<std::array<SealCipherTensor, 3>> {
            auto centered = tensors_.GetShared(inputs.centered.id);
            if (!centered.ok()) return centered.status();
            auto inverse = tensors_.GetShared(inputs.inverse_stddev.id);
            if (!inverse.ok()) return inverse.status();
            std::optional<GammaFusedLayerNormParameters> original;
            if (normalized_residual) {
                const auto load = [&](std::string_view name) -> Result<DenseVectorParameter> {
                    for (const auto &handle : weights.plain_parameters)
                        if (handle.spec.name == name) return parameters_.GetVector(handle.id);
                    return Status::Invalid("original LN1 affine handle absent");
                };
                auto gamma = load("ln1.gamma"), beta = load("ln1.beta");
                if (!gamma.ok()) return gamma.status();
                if (!beta.ok()) return beta.status();
                original = GammaFusedLayerNormParameters{gamma.take_value(), beta.take_value()};
            }
            auto normalized = LayerNormTailKernel::Evaluate(
                *centered.value(), *inverse.value(), *affine.value(), layer_, context, report,
                original ? &*original : nullptr, normalized_residual, config_.threads);
            if (!normalized.ok()) return normalized.status();

            auto q_weights = parameters_.GetMatrixShared(weights.wq.id);
            if (!q_weights.ok()) return q_weights.status();
            auto k_weights = parameters_.GetMatrixShared(weights.wk.id);
            if (!k_weights.ok()) return k_weights.status();
            auto v_weights = parameters_.GetMatrixShared(weights.wv.id);
            if (!v_weights.ok()) return v_weights.status();
            const auto terminal_level =
                context.parameters.coefficient_modulus_bits.size() -
                context.parameters.terminal_prime_count;
            constexpr auto kQkDepth = MultiHeadQkKernel::kDepth;
            const auto qk_start_level = terminal_level - kQkDepth;
            if (qk_start_level == 0)
                return Status::FailedPrecondition(
                    "QK suffix leaves no level for its projection");
            const auto qk_projection_input_level = qk_start_level - 1;
            auto normalized_level = LevelOf(
                context, normalized.value().ciphertexts.front().parms_id());
            if (!normalized_level.ok()) return normalized_level.status();
            if (normalized_level.value() > qk_projection_input_level)
                return Status::FailedPrecondition(
                    "normalized input is already below the QK suffix start");
            std::optional<SealCipherTensor> switched_input;
            if (normalized_level.value() < qk_projection_input_level) {
                const auto levels = Levels(context);
                if (qk_projection_input_level >= levels.size())
                    return Status::FailedPrecondition(
                        "QK projection suffix exceeds modulus chain");
                switched_input = normalized.value();
                const auto target = levels[qk_projection_input_level]->parms_id();
                seal::Evaluator evaluator(*context.context);
                for (auto &ct : switched_input->ciphertexts)
                    evaluator.mod_switch_to_inplace(ct, target);
                switched_input->logical_spec.planned_level = qk_projection_input_level;
            }
            const auto &qk_input = switched_input ? *switched_input : normalized.value();

            // With the seven-prime Region-2 profile, Q/K/V consume the exact same
            // L1 batch. Share baby rotations; giant groups remain weight-specific.
            // Evaluate projections sequentially because this cache is mutable.
            ComplexDiagonalProjectionKernel projection(layer_, context, report, config_.threads);
            auto q = projection.Evaluate(
                qk_input, *q_weights.value(), bq.value(), Layout::kMultiHeadQ);
            if (!q.ok()) return q.status();
            auto k = projection.Evaluate(
                qk_input, *k_weights.value(), bk.value(), Layout::kMultiHeadK);
            if (!k.ok()) return k.status();
            auto v = projection.Evaluate(
                normalized.value(), *v_weights.value(), bv.value(), Layout::kMultiHeadV);
            if (!v.ok()) return v.status();
            return std::array<SealCipherTensor, 3>{q.take_value(), k.take_value(), v.take_value()};
        };
        auto projected = project_qkv();
        if (!projected.ok()) return projected.status();
        // Temporary LN input views, X and its shared baby cache leave scope before QK.
        auto scores = MultiHeadQkKernel::Evaluate(
            projected.value()[0], projected.value()[1], layer_, context, report,
            config_.threads);
        if (!scores.ok()) return scores.status();

        auto score_handle = tensors_.Allocate(scores.take_value());
        if (!score_handle.ok()) return score_handle.status();
        auto value_handle = tensors_.Allocate(std::move(projected.value()[2]));
        if (!value_handle.ok()) return value_handle.status();
        return Block2HeOutputs{score_handle.take_value(),
                               value_handle.take_value()};
    } catch (const std::exception &error) {
        return Status::Internal(std::string("Block 2 HE evaluation: ") +
                                error.what());
    }
}

} // namespace hera
