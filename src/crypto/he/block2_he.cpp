#include "hera/block2_he.h"

#include <algorithm>
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

Result<std::size_t> LevelOf(const HeContext &context,
                            seal::parms_id_type id)
{
    std::size_t level = 0;
    for (auto data = context.context->first_context_data(); data;
         data = data->next_context_data(), ++level)
        if (data->parms_id() == id) return level;
    return Status::Invalid("ciphertext parms_id is outside the HE profile");
}

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

double DroppedPrime(const HeContext &context, seal::parms_id_type id)
{
    const auto data = context.context->get_context_data(id);
    if (!data || !data->next_context_data())
        throw std::invalid_argument("operation needs another CKKS data level");
    const auto &moduli = data->parms().coeff_modulus();
    if (moduli.empty())
        throw std::logic_error("CKKS context data has no coefficient modulus");
    return static_cast<double>(moduli.back().value());
}

int CanonicalRotation(long long value, std::size_t slots)
{
    value %= static_cast<long long>(slots);
    if (value > static_cast<long long>(slots / 2))
        value -= static_cast<long long>(slots);
    if (value <= -static_cast<long long>(slots / 2))
        value += static_cast<long long>(slots);
    return static_cast<int>(value);
}

seal::Ciphertext RotateComposite(const seal::Ciphertext &input, int rotation,
                                 const HeContext &context,
                                 seal::Evaluator &evaluator,
                                 Block2ExecutionReport *report)
{
    seal::Ciphertext result = input;
    int remaining = CanonicalRotation(rotation, context.slot_count());
    if (remaining == 0) return result;
    const int sign = remaining < 0 ? -1 : 1;
    std::uint64_t magnitude = static_cast<std::uint64_t>(
        remaining < 0 ? -static_cast<long long>(remaining) : remaining);
    std::uint64_t bit = 1;
    while (magnitude != 0) {
        if (magnitude & 1U) {
            evaluator.rotate_vector_inplace(result,
                                            sign * static_cast<int>(bit),
                                            context.galois_keys);
            if (report) ++report->rotations;
        }
        magnitude >>= 1U;
        bit <<= 1U;
    }
    return result;
}

Complex DestinationFactor(ComplexLane component)
{
    return component == ComplexLane::kReal ? Complex{1.0, 0.0}
                                           : Complex{0.0, 1.0};
}

Complex ExtractionFactor(ComplexLane component)
{
    // z+conj(z)=2 Re(z), z-conj(z)=2i Im(z).
    return component == ComplexLane::kReal ? Complex{0.5, 0.0}
                                           : Complex{0.0, -0.5};
}

template <typename T>
bool AnyNonzero(const std::vector<T> &values)
{
    return std::any_of(values.begin(), values.end(), [](const T &value) {
        return std::abs(value) > 0.0;
    });
}

const PlainParameterHandle *FindPlain(const LayerWeightHandles &weights,
                                      const std::string &name)
{
    for (const auto &plain : weights.plain_parameters)
        if (plain.spec.name == name) return &plain;
    return nullptr;
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

Result<void> ValidateVectorHandle(const PlainParameterHandle *handle,
                                  const std::string &name,
                                  std::size_t length)
{
    if (!handle || !handle->valid() || handle->spec.name != name ||
        handle->spec.length != length)
        return Status::Invalid(name + " handle has the wrong shape");
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

Result<SealCipherTensor> LayerNormTail(
    const SealCipherTensor &centered,
    const SealCipherTensor &inverse,
    const DenseVectorParameter &gamma,
    const DenseVectorParameter &beta,
    const TransformerLayerConfig &layer,
    const HeContext &context,
    Block2ExecutionReport *report)
{
    auto centered_valid = centered.Validate(context);
    if (!centered_valid.ok()) return centered_valid.status();
    auto inverse_valid = inverse.Validate(context);
    if (!inverse_valid.ok()) return inverse_valid.status();
    auto packing_valid = SamePhysicalPacking(centered.physical_packing,
                                              inverse.physical_packing);
    if (!packing_valid.ok()) return packing_valid.status();
    if (centered.logical_spec.shape !=
            TensorShape{{layer.tokens, layer.model_width}} ||
        inverse.logical_spec.shape != centered.logical_spec.shape ||
        centered.logical_spec.layout != Layout::kSpatial)
        return Status::Invalid(
            "Block 2 LayerNorm inputs must be dense [tokens,model_width]");
    if (gamma.values.size() != layer.model_width ||
        beta.values.size() != layer.model_width)
        return Status::Invalid("LayerNorm affine parameter size mismatch");
    const auto centered_level = LevelOf(context,
                                        centered.ciphertexts.front().parms_id());
    if (!centered_level.ok()) return centered_level.status();
    const auto inverse_level = LevelOf(context,
                                       inverse.ciphertexts.front().parms_id());
    if (!inverse_level.ok()) return inverse_level.status();
    if (centered_level.value() != 0 || inverse_level.value() != 0)
        return Status::FailedPrecondition(
            "Block 2 LayerNorm inputs must be at fresh level 0");
    const double wanted_input = std::ldexp(
        1.0, context.parameters.input_scale_bits);
    for (std::size_t i = 0; i < centered.ciphertexts.size(); ++i) {
        if (centered.ciphertexts[i].parms_id() !=
                inverse.ciphertexts[i].parms_id() ||
            !CloseScale(centered.ciphertexts[i].scale(), wanted_input) ||
            !CloseScale(inverse.ciphertexts[i].scale(), wanted_input))
            return Status::FailedPrecondition(
                "LayerNorm inputs have incompatible level or scale");
    }

    seal::Evaluator evaluator(*context.context);
    seal::CKKSEncoder encoder(*context.context);
    SealCipherTensor result = centered;
    const double target = std::ldexp(
        1.0, context.parameters.default_scale_bits);
    for (std::size_t ct = 0; ct < result.ciphertexts.size(); ++ct) {
        std::vector<double> gamma_real(context.slot_count(), 0.0);
        std::vector<double> gamma_imag(context.slot_count(), 0.0);
        bool has_real = false;
        bool has_imag = false;
        for (std::size_t logical = 0;
             logical < result.physical_packing.logical_locations.size();
             ++logical) {
            const auto feature = logical % layer.model_width;
            for (const auto &location :
                 result.physical_packing.logical_locations[logical]) {
                if (location.ciphertext != ct) continue;
                if (location.component == ComplexLane::kReal) {
                    gamma_real[location.slot] = gamma.values[feature];
                    has_real = true;
                } else {
                    gamma_imag[location.slot] = gamma.values[feature];
                    has_imag = true;
                }
            }
        }
        if (!has_real && !has_imag)
            return Status::Internal("LayerNorm produced an empty ciphertext");

        seal::Ciphertext centered_conjugated;
        seal::Ciphertext inverse_conjugated;
        evaluator.complex_conjugate(centered.ciphertexts[ct],
                                    context.galois_keys,
                                    centered_conjugated);
        evaluator.complex_conjugate(inverse.ciphertexts[ct],
                                    context.galois_keys,
                                    inverse_conjugated);
        if (report) report->conjugations += 2;

        const double dropped = DroppedPrime(
            context, centered.ciphertexts[ct].parms_id());
        const double gamma_scale =
            target * dropped /
            (centered.ciphertexts[ct].scale() *
             inverse.ciphertexts[ct].scale());
        if (!std::isfinite(gamma_scale) || gamma_scale < 1.0)
            return Status::FailedPrecondition(
                "LayerNorm gamma scale cannot realize the configured target");
        seal::Ciphertext affine;
        bool initialized = false;
        if (has_real) {
            seal::Ciphertext centered_real;
            seal::Ciphertext inverse_real;
            evaluator.add(centered.ciphertexts[ct], centered_conjugated,
                          centered_real);
            evaluator.add(inverse.ciphertexts[ct], inverse_conjugated,
                          inverse_real);
            seal::Ciphertext real;
            evaluator.multiply(centered_real, inverse_real, real);
            std::vector<Complex> mask(context.slot_count());
            for (std::size_t slot = 0; slot < mask.size(); ++slot)
                mask[slot] = {0.25 * gamma_real[slot], 0.0};
            auto encoded = EncodeModelPlaintext(
                encoder, mask, real.parms_id(), gamma_scale,
                "block2.ln1.gamma.real");
            if (!encoded.ok()) return encoded.status();
            auto plain = encoded.take_value();
            evaluator.multiply_plain_inplace(real, plain);
            affine = std::move(real);
            initialized = true;
            if (report) {
                ++report->ciphertext_multiplications;
                ++report->ciphertext_plain_multiplications;
            }
        }
        if (has_imag) {
            seal::Ciphertext centered_imag;
            seal::Ciphertext inverse_imag;
            evaluator.sub(centered.ciphertexts[ct], centered_conjugated,
                          centered_imag);
            evaluator.sub(inverse.ciphertexts[ct], inverse_conjugated,
                          inverse_imag);
            seal::Ciphertext imag;
            evaluator.multiply(centered_imag, inverse_imag, imag);
            std::vector<Complex> mask(context.slot_count());
            for (std::size_t slot = 0; slot < mask.size(); ++slot)
                // (z-z*)*(w-w*) = -4 Im(z)Im(w). Multiplication
                // by -i/4 places the independent result in the imag lane.
                mask[slot] = {0.0, -0.25 * gamma_imag[slot]};
            auto encoded = EncodeModelPlaintext(
                encoder, mask, imag.parms_id(), gamma_scale,
                "block2.ln1.gamma.imag");
            if (!encoded.ok()) return encoded.status();
            auto plain = encoded.take_value();
            evaluator.multiply_plain_inplace(imag, plain);
            if (initialized)
                evaluator.add_inplace(affine, imag);
            else
                affine = std::move(imag);
            if (report) {
                ++report->ciphertext_multiplications;
                ++report->ciphertext_plain_multiplications;
            }
        }
        evaluator.relinearize_inplace(affine, context.relinearization_keys);
        evaluator.rescale_to_next_inplace(affine);
        NormalizeScale(affine, target);
        if (report) {
            ++report->relinearizations;
            ++report->rescales;
        }

        std::vector<Complex> beta_slots(context.slot_count());
        for (std::size_t logical = 0;
             logical < result.physical_packing.logical_locations.size();
             ++logical) {
            const auto feature = logical % layer.model_width;
            for (const auto &location :
                 result.physical_packing.logical_locations[logical]) {
                if (location.ciphertext != ct) continue;
                if (location.component == ComplexLane::kReal)
                    beta_slots[location.slot].real(beta.values[feature]);
                else
                    beta_slots[location.slot].imag(beta.values[feature]);
            }
        }
        auto encoded_beta = EncodeModelPlaintext(
            encoder, beta_slots, affine.parms_id(), target,
            "block2.ln1.beta");
        if (!encoded_beta.ok()) return encoded_beta.status();
        auto beta_plain = encoded_beta.take_value();
        evaluator.add_plain_inplace(affine, beta_plain);
        result.ciphertexts[ct] = std::move(affine);
    }
    result.logical_spec.planned_level = 1;
    result.logical_spec.planned_scale_bits =
        context.parameters.default_scale_bits;
    return result;
}

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

struct RotationKey {
    std::size_t source_ct{};
    ComplexLane component{ComplexLane::kReal};
    int rotation{};

    bool operator<(const RotationKey &other) const noexcept
    {
        return std::tie(source_ct, component, rotation) <
               std::tie(other.source_ct, other.component, other.rotation);
    }
};

using RotationCache = std::map<RotationKey, seal::Ciphertext>;

struct ProjectionGiantKey {
    std::size_t source_ct{};
    std::size_t target_ct{};
    ComplexLane source_component{ComplexLane::kReal};
    int giant_rotation{};

    bool operator<(const ProjectionGiantKey &other) const noexcept
    {
        return std::tie(source_ct, target_ct, source_component,
                        giant_rotation) <
               std::tie(other.source_ct, other.target_ct,
                        other.source_component, other.giant_rotation);
    }
};

Result<SealCipherTensor> Project(
    const SealCipherTensor &input,
    const DenseMatrixParameter &weights,
    const std::optional<DenseVectorParameter> &bias,
    Layout output_layout,
    const TransformerLayerConfig &layer,
    const HeContext &context,
    RotationCache &rotation_cache,
    Block2ExecutionReport *report)
{
    if (input.logical_spec.shape !=
            TensorShape{{layer.tokens, layer.model_width}} ||
        input.logical_spec.layout != Layout::kSpatial)
        return Status::Invalid("projection expects spatial [tokens,model_width]");
    const std::size_t logical_heads = output_layout == Layout::kMultiHeadQ
        ? layer.q_heads : layer.kv_heads;
    const std::size_t width = output_layout == Layout::kMultiHeadV
        ? layer.value_head_width : layer.qk_head_width;
    const std::size_t output_columns = logical_heads * width;
    if (weights.spec.rows != layer.model_width ||
        weights.spec.columns != output_columns ||
        weights.row_major.size() != layer.model_width * output_columns)
        return Status::Invalid(weights.spec.name +
                               " payload has the wrong rectangular shape");
    if (bias && bias->values.size() != output_columns)
        return Status::Invalid(weights.spec.name + " bias has the wrong size");
    auto projection_plan = ComplexDiagonalProjectionPlanner::Build(
        layer, output_layout, context.slot_count());
    if (!projection_plan.ok()) return projection_plan.status();
    const auto &plan = projection_plan.value().output_packing;
    if (input.ciphertexts.empty())
        return Status::Invalid("projection input has no ciphertexts");
    auto input_level = LevelOf(context,
                               input.ciphertexts.front().parms_id());
    if (!input_level.ok()) return input_level.status();
    for (const auto &ct : input.ciphertexts) {
        auto level = LevelOf(context, ct.parms_id());
        if (!level.ok()) return level.status();
        if (level.value() != input_level.value())
            return Status::FailedPrecondition(
                "projection input ciphertexts are at different levels");
    }

    std::map<ProjectionKey, std::vector<Complex>> masks;
    const double q_scale = output_layout == Layout::kMultiHeadQ
        ? 1.0 / std::sqrt(static_cast<double>(layer.qk_head_width)) : 1.0;
    for (std::size_t token = 0; token < layer.tokens; ++token) {
        for (std::size_t head = 0; head < logical_heads; ++head) {
            for (std::size_t dimension = 0; dimension < width; ++dimension) {
                const auto output_logical =
                    (token * logical_heads + head) * width + dimension;
                const auto output_column = head * width + dimension;
                for (const auto &destination :
                     plan.Locations(output_logical)) {
                    for (std::size_t feature = 0;
                         feature < layer.model_width; ++feature) {
                        const auto input_logical =
                            token * layer.model_width + feature;
                        const auto &sources =
                            input.physical_packing.Locations(input_logical);
                        if (sources.size() != 1)
                            return Status::Invalid(
                                "spatial projection input must not be replicated");
                        const auto &source = sources.front();
                        const int rotation = CanonicalRotation(
                            static_cast<long long>(source.slot) -
                                static_cast<long long>(destination.slot),
                            context.slot_count());
                        ProjectionKey key{source.ciphertext,
                                          destination.ciphertext,
                                          source.component, rotation};
                        auto inserted = masks.emplace(
                            key, std::vector<Complex>(context.slot_count()));
                        const auto coefficient =
                            weights.row_major[feature * output_columns +
                                              output_column] * q_scale;
                        inserted.first->second[destination.slot] +=
                            coefficient * ExtractionFactor(source.component) *
                            DestinationFactor(destination.component);
                    }
                }
            }
        }
    }

    // Baby-step/giant-step decomposition. For the native projection-input
    // packing all shifts are multiples of tokens*heads, so gcd selects that
    // stride and only the actually used baby rotations are materialized.
    std::size_t base = context.slot_count();
    for (const auto &entry : masks) {
        int rotation = CanonicalRotation(entry.first.rotation,
                                         context.slot_count());
        if (rotation < 0) rotation += static_cast<int>(context.slot_count());
        if (rotation != 0)
            base = std::gcd(base, static_cast<std::size_t>(rotation));
    }
    if (base == context.slot_count() || base == 0)
        base = std::max<std::size_t>(
            1, static_cast<std::size_t>(std::sqrt(context.slot_count())));

    using BabyMasks = std::map<int, std::vector<Complex>>;
    std::map<ProjectionGiantKey, BabyMasks> bsgs;
    for (const auto &entry : masks) {
        if (!AnyNonzero(entry.second)) continue;
        int residue = CanonicalRotation(entry.first.rotation,
                                        context.slot_count());
        if (residue < 0) residue += static_cast<int>(context.slot_count());
        const int giant = static_cast<int>((residue / base) * base);
        const int baby = residue - giant;
        ProjectionGiantKey key{entry.first.source_ct,
                               entry.first.target_ct,
                               entry.first.source_component, giant};
        auto inserted = bsgs[key].emplace(
            baby, std::vector<Complex>(context.slot_count()));
        auto &adjusted = inserted.first->second;
        for (std::size_t slot = 0; slot < context.slot_count(); ++slot) {
            if (std::abs(entry.second[slot]) == 0.0) continue;
            const auto shifted =
                (slot + static_cast<std::size_t>(giant)) %
                context.slot_count();
            adjusted[shifted] += entry.second[slot];
        }
    }

    seal::Evaluator evaluator(*context.context);
    seal::CKKSEncoder encoder(*context.context);
    std::vector<seal::Ciphertext> output(plan.ciphertext_count);
    std::vector<bool> initialized(output.size(), false);
    const double target = std::ldexp(
        1.0, context.parameters.default_scale_bits);
    for (const auto &entry : bsgs) {
        const auto &key = entry.first;
        const RotationKey base_key{key.source_ct, key.source_component, 0};
        auto source_base = rotation_cache.find(base_key);
        if (source_base == rotation_cache.end()) {
            const auto &source = input.ciphertexts.at(key.source_ct);
            seal::Ciphertext conjugated;
            evaluator.complex_conjugate(source, context.galois_keys,
                                        conjugated);
            if (report) ++report->conjugations;
            seal::Ciphertext extracted;
            if (key.source_component == ComplexLane::kReal)
                evaluator.add(source, conjugated, extracted);
            else
                evaluator.sub(source, conjugated, extracted);
            source_base = rotation_cache.emplace(base_key,
                                                  std::move(extracted)).first;
        }
        seal::Ciphertext giant_sum;
        bool giant_initialized = false;
        for (const auto &baby_entry : entry.second) {
            const RotationKey baby_key{key.source_ct, key.source_component,
                                       baby_entry.first};
            auto baby = rotation_cache.find(baby_key);
            if (baby == rotation_cache.end()) {
                auto rotated = RotateComposite(
                    source_base->second, baby_entry.first, context,
                    evaluator, report);
                baby = rotation_cache.emplace(baby_key,
                                               std::move(rotated)).first;
            }
            const double dropped = DroppedPrime(
                context, baby->second.parms_id());
            const double plain_scale =
                target * dropped / baby->second.scale();
            auto encoded = EncodeModelPlaintext(
                encoder, baby_entry.second, baby->second.parms_id(),
                plain_scale, "block2.projection.diagonal");
            if (!encoded.ok()) return encoded.status();
            auto plain = encoded.take_value();
            seal::Ciphertext term;
            evaluator.multiply_plain(baby->second, plain, term);
            if (report) ++report->ciphertext_plain_multiplications;
            if (!giant_initialized) {
                giant_sum = std::move(term);
                giant_initialized = true;
            } else {
                evaluator.add_inplace(giant_sum, term);
            }
        }
        if (key.giant_rotation != 0)
            giant_sum = RotateComposite(giant_sum, key.giant_rotation,
                                        context, evaluator, report);
        if (!initialized[key.target_ct]) {
            output[key.target_ct] = std::move(giant_sum);
            initialized[key.target_ct] = true;
        } else {
            evaluator.add_inplace(output[key.target_ct], giant_sum);
        }
    }

    for (std::size_t ct = 0; ct < output.size(); ++ct) {
        if (!initialized[ct])
            return Status::Internal("projection generated an empty output group");
        if (bias) {
            std::vector<Complex> bias_slots(context.slot_count());
            for (std::size_t logical = 0;
                 logical < plan.logical_locations.size(); ++logical) {
                const auto head = (logical / width) % logical_heads;
                const auto dimension = logical % width;
                const double value =
                    bias->values[head * width + dimension] * q_scale;
                for (const auto &location : plan.Locations(logical)) {
                    if (location.ciphertext != ct) continue;
                    if (location.component == ComplexLane::kReal)
                        bias_slots[location.slot].real(value);
                    else
                        bias_slots[location.slot].imag(value);
                }
            }
            auto encoded_bias = EncodeModelPlaintext(
                encoder, bias_slots, output[ct].parms_id(),
                output[ct].scale(), "block2.projection.bias");
            if (!encoded_bias.ok()) return encoded_bias.status();
            auto bias_plain = encoded_bias.take_value();
            evaluator.add_plain_inplace(output[ct], bias_plain);
        }
        evaluator.rescale_to_next_inplace(output[ct]);
        NormalizeScale(output[ct], target);
        if (report) ++report->rescales;
    }

    SealCipherTensor result;
    result.parameter_profile = context.profile;
    result.ciphertexts = std::move(output);
    result.logical_spec = MakeHeSpec(
        plan.logical_shape, output_layout, context.profile,
        input_level.value() + 1,
        context.parameters.default_scale_bits);
    result.logical_spec.numeric = input.logical_spec.numeric;
    result.physical_packing = projection_plan.take_value().output_packing;
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
    const auto max_heads = slots / layer.tokens;
    for (std::size_t begin = 0; begin < layer.q_heads;) {
        const auto heads = std::min(max_heads, layer.q_heads - begin);
        const auto lanes = slots / (layer.tokens * heads);
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

std::vector<double> InnerRotationMask(const HeadTile &tile,
                                      const TransformerLayerConfig &layer,
                                      std::size_t slots,
                                      std::size_t diagonal,
                                      bool wrap)
{
    std::vector<double> mask(slots, 0.0);
    const auto block = layer.tokens * tile.head_count;
    for (std::size_t lane = 0; lane < tile.lanes; ++lane) {
        for (std::size_t row = 0; row < layer.tokens; ++row) {
            const bool row_wrap = diagonal != 0 &&
                                  row >= layer.tokens - diagonal;
            if (row_wrap != wrap) continue;
            for (std::size_t head = 0; head < tile.head_count; ++head)
                mask[lane * block + row * tile.head_count + head] = 1.0;
        }
    }
    return mask;
}

seal::Ciphertext MultiplyMaskAndRescale(
    const seal::Ciphertext &input, const std::vector<double> &mask,
    double output_scale, const HeContext &context,
    seal::Evaluator &evaluator, seal::CKKSEncoder &encoder,
    Block2ExecutionReport *report)
{
    const double plain_scale =
        output_scale * DroppedPrime(context, input.parms_id()) / input.scale();
    seal::Plaintext plain;
    encoder.encode(mask, input.parms_id(), plain_scale, plain);
    seal::Ciphertext result;
    evaluator.multiply_plain(input, plain, result);
    evaluator.rescale_to_next_inplace(result);
    NormalizeScale(result, output_scale);
    if (report) {
        ++report->ciphertext_plain_multiplications;
        ++report->rescales;
    }
    return result;
}

seal::Ciphertext InnerRotateK(
    const seal::Ciphertext &input, std::size_t diagonal,
    const HeadTile &tile, const TransformerLayerConfig &layer,
    const HeContext &context, seal::Evaluator &evaluator,
    seal::CKKSEncoder &encoder, double output_scale,
    Block2ExecutionReport *report)
{
    const auto nonwrap_mask = InnerRotationMask(
        tile, layer, context.slot_count(), diagonal, false);
    const auto wrap_mask = InnerRotationMask(
        tile, layer, context.slot_count(), diagonal, true);
    const int shift = static_cast<int>(diagonal * tile.head_count);
    const int block = static_cast<int>(layer.tokens * tile.head_count);
    auto nonwrap = RotateComposite(input, shift, context, evaluator, report);
    const double plain_scale = output_scale *
        DroppedPrime(context, input.parms_id()) / input.scale();
    seal::Plaintext nonwrap_plain;
    encoder.encode(nonwrap_mask, input.parms_id(), plain_scale,
                   nonwrap_plain);
    evaluator.multiply_plain_inplace(nonwrap, nonwrap_plain);
    std::size_t multiplications = 1;
    if (AnyNonzero(wrap_mask)) {
        auto wrap = RotateComposite(input, shift - block, context, evaluator,
                                    report);
        seal::Plaintext wrap_plain;
        encoder.encode(wrap_mask, input.parms_id(), plain_scale, wrap_plain);
        evaluator.multiply_plain_inplace(wrap, wrap_plain);
        evaluator.add_inplace(nonwrap, wrap);
        ++multiplications;
    }
    evaluator.rescale_to_next_inplace(nonwrap);
    NormalizeScale(nonwrap, output_scale);
    if (report) {
        report->ciphertext_plain_multiplications += multiplications;
        ++report->rescales;
    }
    return nonwrap;
}

Result<SealCipherTensor> ComputeQk(
    const SealCipherTensor &q, const SealCipherTensor &k,
    const TransformerLayerConfig &layer, const HeContext &context,
    Block2ExecutionReport *report)
{
    auto q_valid = q.Validate(context);
    if (!q_valid.ok()) return q_valid.status();
    auto k_valid = k.Validate(context);
    if (!k_valid.ok()) return k_valid.status();
    if (q.logical_spec.layout != Layout::kMultiHeadQ ||
        k.logical_spec.layout != Layout::kMultiHeadK)
        return Status::Invalid("QK requires native multihead Q/K packing");
    const auto q_level = LevelOf(context, q.ciphertexts.front().parms_id());
    const auto k_level = LevelOf(context, k.ciphertexts.front().parms_id());
    if (!q_level.ok()) return q_level.status();
    if (!k_level.ok()) return k_level.status();
    const auto levels = Levels(context);
    const auto terminal_level =
        context.parameters.coefficient_modulus_bits.size() -
        context.parameters.terminal_prime_count;
    constexpr std::size_t kQkDepth = 5;
    if (terminal_level < kQkDepth)
        return Status::FailedPrecondition(
            "QK suffix requires five remaining CKKS levels");
    const auto qk_start_level = terminal_level - kQkDepth;
    if (q_level.value() != qk_start_level ||
        k_level.value() != qk_start_level)
        return Status::FailedPrecondition(
            "QK inputs must start exactly five levels before terminal");
    if (qk_start_level + 1 >= levels.size())
        return Status::FailedPrecondition("QK suffix exceeds modulus chain");
    const double scale44 = std::ldexp(
        1.0, context.parameters.default_scale_bits);
    // K is rescaled once before the ct-ct product. Choose its resulting scale
    // from the *next* prime (the one dropped by that product), not merely from
    // the bit width of the current prime. This keeps arbitrary configured RNS
    // primes numerically aligned without rewriting scale metadata.
    const double k_preprocessed_scale =
        scale44 * DroppedPrime(
            context, levels.at(qk_start_level + 1)->parms_id()) /
        q.ciphertexts.front().scale();
    seal::Evaluator evaluator(*context.context);
    seal::CKKSEncoder encoder(*context.context);
    const auto tiles = QkTiles(layer, context.slot_count());
    std::vector<seal::Ciphertext> unpaired;
    std::size_t expected_input = 0;
    for (const auto &tile : tiles) expected_input += tile.input_groups;
    if (q.ciphertexts.size() != expected_input ||
        k.ciphertexts.size() != expected_input)
        return Status::Invalid("QK ciphertext count does not match packing plan");

    // BLB-style complex reduction starts by conjugating every K group once;
    // the result is reused for all attention diagonals.
    std::vector<seal::Ciphertext> conjugated_k(k.ciphertexts.size());
    for (std::size_t i = 0; i < k.ciphertexts.size(); ++i) {
        evaluator.complex_conjugate(k.ciphertexts[i], context.galois_keys,
                                    conjugated_k[i]);
        if (report) ++report->conjugations;
    }

    for (const auto &tile : tiles) {
        const auto block = layer.tokens * tile.head_count;
        std::vector<seal::Ciphertext> tile_unpaired(
            (layer.tokens + tile.lanes - 1) / tile.lanes);
        std::vector<bool> initialized(tile_unpaired.size(), false);
        for (std::size_t diagonal = 0; diagonal < layer.tokens; ++diagonal) {
            seal::Ciphertext diagonal_sum;
            bool diagonal_initialized = false;
            for (std::size_t group = 0; group < tile.input_groups; ++group) {
                const auto index = tile.input_begin + group;
                auto rotated_k = InnerRotateK(
                    conjugated_k[index], diagonal, tile, layer, context, evaluator,
                    encoder, k_preprocessed_scale, report);
                auto q_at_level = q.ciphertexts[index];
                evaluator.mod_switch_to_inplace(q_at_level,
                                                rotated_k.parms_id());
                seal::Ciphertext product;
                evaluator.multiply(q_at_level, rotated_k, product);
                evaluator.relinearize_inplace(product,
                                               context.relinearization_keys);
                evaluator.rescale_to_next_inplace(product);
                NormalizeScale(product, scale44);
                if (report) {
                    ++report->ciphertext_multiplications;
                    ++report->relinearizations;
                    ++report->rescales;
                }
                if (!diagonal_initialized) {
                    diagonal_sum = std::move(product);
                    diagonal_initialized = true;
                } else {
                    evaluator.add_inplace(diagonal_sum, product);
                }
            }

            // Sum all reduction lanes into lane zero. Each complex product
            // already contributes two dimensions through Re(Q*conj(K)).
            seal::Ciphertext lanes_sum = diagonal_sum;
            const auto active_lanes = std::min(
                tile.lanes, layer.qk_head_width);
            for (std::size_t lane = 1; lane < active_lanes; ++lane) {
                auto rotated = RotateComposite(
                    diagonal_sum, static_cast<int>(lane * block), context,
                    evaluator, report);
                evaluator.add_inplace(lanes_sum, rotated);
            }
            std::vector<double> lane_zero(context.slot_count(), 0.0);
            for (std::size_t row = 0; row < layer.tokens; ++row)
                for (std::size_t head = 0; head < tile.head_count; ++head)
                    lane_zero[row * tile.head_count + head] = 1.0;
            auto aligned = MultiplyMaskAndRescale(
                lanes_sum, lane_zero, scale44, context, evaluator, encoder,
                report);

            const auto output_group = diagonal / tile.lanes;
            const auto output_lane = diagonal % tile.lanes;
            if (output_lane != 0)
                aligned = RotateComposite(
                    aligned, -static_cast<int>(output_lane * block), context,
                    evaluator, report);
            if (!initialized[output_group]) {
                tile_unpaired[output_group] = std::move(aligned);
                initialized[output_group] = true;
            } else {
                evaluator.add_inplace(tile_unpaired[output_group], aligned);
            }
        }

        // Recovery mask is nontrivial for tail groups and also fixes the
        // level contract for every group before real/imag pairing.
        for (std::size_t group = 0; group < tile_unpaired.size(); ++group) {
            std::vector<double> recovery(context.slot_count(), 0.0);
            const auto first_diagonal = group * tile.lanes;
            for (std::size_t lane = 0; lane < tile.lanes; ++lane) {
                if (first_diagonal + lane >= layer.tokens) continue;
                for (std::size_t row = 0; row < layer.tokens; ++row)
                    for (std::size_t head = 0; head < tile.head_count; ++head)
                        recovery[lane * block + row * tile.head_count + head] =
                            1.0;
            }
            tile_unpaired[group] = MultiplyMaskAndRescale(
                tile_unpaired[group], recovery, scale44, context, evaluator,
                encoder, report);
        }

        for (std::size_t pair = 0; pair < (tile_unpaired.size() + 1) / 2;
             ++pair) {
            auto even = tile_unpaired[2 * pair];
            seal::Ciphertext even_conjugated;
            evaluator.complex_conjugate(even, context.galois_keys,
                                        even_conjugated);
            evaluator.add_inplace(even, even_conjugated);
            if (report) ++report->conjugations;
            std::vector<Complex> half(context.slot_count(), {0.5, 0.0});
            const double half_scale = scale44 *
                DroppedPrime(context, even.parms_id()) / even.scale();
            seal::Plaintext half_plain;
            encoder.encode(half, even.parms_id(), half_scale, half_plain);
            evaluator.multiply_plain_inplace(even, half_plain);
            if (report) ++report->ciphertext_plain_multiplications;

            if (2 * pair + 1 < tile_unpaired.size()) {
                auto odd = tile_unpaired[2 * pair + 1];
                seal::Ciphertext odd_conjugated;
                evaluator.complex_conjugate(odd, context.galois_keys,
                                            odd_conjugated);
                evaluator.add_inplace(odd, odd_conjugated);
                if (report) ++report->conjugations;
                std::vector<Complex> half_i(context.slot_count(),
                                            {0.0, 0.5});
                seal::Plaintext half_i_plain;
                encoder.encode(half_i, odd.parms_id(), half_scale,
                               half_i_plain);
                evaluator.multiply_plain_inplace(odd, half_i_plain);
                evaluator.add_inplace(even, odd);
                if (report) ++report->ciphertext_plain_multiplications;
            }
            evaluator.rescale_to_next_inplace(even);
            NormalizeScale(even, scale44);
            if (report) ++report->rescales;
            unpaired.push_back(std::move(even));
        }
    }

    auto output_plan = MultiHeadPackingPlan::AttentionDiagonal(
        layer, context.slot_count());
    if (!output_plan.ok()) return output_plan.status();
    if (unpaired.size() != output_plan.value().ciphertext_count)
        return Status::Internal("QK output count differs from compiled packing");
    SealCipherTensor result;
    result.parameter_profile = context.profile;
    result.ciphertexts = std::move(unpaired);
    result.logical_spec = MakeHeSpec(
        {{layer.q_heads, layer.tokens, layer.tokens}},
        Layout::kAttentionDiagonal, context.profile, terminal_level,
        context.parameters.output_scale_bits);
    result.logical_spec.numeric = q.logical_spec.numeric;
    result.physical_packing = output_plan.take_value();
    return result;
}

} // namespace

Result<void> Block2LevelPlanner::Validate(
    const HeParameterConfig &parameters)
{
    if (parameters.coefficient_modulus_bits.size() < 8 ||
        parameters.declared_depth < 7)
        return Status::FailedPrecondition(
            "Block 2 requires at least eight data primes and depth seven");
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
    const DenseVectorParameter &gamma,
    const DenseVectorParameter &beta,
    const TransformerLayerConfig &layer,
    const HeContext &context,
    Block2ExecutionReport *report)
{
    return LayerNormTail(centered, inverse_stddev, gamma, beta, layer,
                         context, report);
}

class ComplexDiagonalProjectionKernel::Impl {
public:
    Impl(const TransformerLayerConfig &configured_layer,
         const HeContext &configured_context,
         Block2ExecutionReport *configured_report)
        : layer(configured_layer), context(configured_context),
          report(configured_report)
    {}

    const TransformerLayerConfig &layer;
    const HeContext &context;
    Block2ExecutionReport *report;
    RotationCache rotations;
};

ComplexDiagonalProjectionKernel::ComplexDiagonalProjectionKernel(
    const TransformerLayerConfig &layer, const HeContext &context,
    Block2ExecutionReport *report)
    : impl_(std::make_unique<Impl>(layer, context, report))
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
    return Project(input, weights, bias, output_layout, impl_->layer,
                   impl_->context, impl_->rotations, impl_->report);
}

Result<SealCipherTensor> MultiHeadQkKernel::Evaluate(
    const SealCipherTensor &q, const SealCipherTensor &k,
    const TransformerLayerConfig &layer, const HeContext &context,
    Block2ExecutionReport *report)
{
    return ComputeQk(q, k, layer, context, report);
}

SealBlock2Executor::SealBlock2Executor(
    const HeraConfig &config, const TransformerLayerConfig &layer,
    HeContextRegistry &contexts, SealCipherTensorStore &tensors,
    SealPlainParameterStore &parameters)
    : config_(config), layer_(layer), contexts_(contexts), tensors_(tensors),
      parameters_(parameters)
{}

Result<std::vector<int>> SealBlock2Executor::RequiredRotationSteps(
    const TransformerLayerConfig &layer,
    const HeParameterConfig &parameters)
{
    auto valid = Block2LevelPlanner::Validate(parameters);
    if (!valid.ok()) return valid.status();
    auto layer_valid = ValidateLayerConfig(layer);
    if (!layer_valid.ok()) return layer_valid.status();
    const auto slots = parameters.polynomial_degree / 2;
    if (layer.tokens > slots)
        return Status::Invalid("Block 2 token dimension exceeds CKKS slots");
    std::set<int> steps{0};
    for (std::size_t power = 1; power <= slots / 2; power <<= 1U) {
        steps.insert(static_cast<int>(power));
        if (power != slots / 2)
            steps.insert(-static_cast<int>(power));
        if (power > slots / 2 / 2) break;
    }
    return std::vector<int>(steps.begin(), steps.end());
}

Result<Block2HeOutputs> SealBlock2Executor::Evaluate(
    const Block2HeInputs &inputs, const LayerWeightHandles &weights,
    Block2ExecutionReport *report)
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
        auto required = RequiredRotationSteps(layer_, context.parameters);
        if (!required.ok()) return required.status();
        if (context.galois_keys.size() == 0)
            return Status::FailedPrecondition(
                "Block 2 Galois keys have not been compiled");
        if (report) {
            *report = {};
            report->required_rotation_steps = required.value();
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
        const auto *gamma_handle = FindPlain(weights, "ln1.gamma");
        const auto *beta_handle = FindPlain(weights, "ln1.beta");
        auto gamma_valid = ValidateVectorHandle(
            gamma_handle, "ln1.gamma", layer_.model_width);
        if (!gamma_valid.ok()) return gamma_valid.status();
        auto beta_valid = ValidateVectorHandle(
            beta_handle, "ln1.beta", layer_.model_width);
        if (!beta_valid.ok()) return beta_valid.status();

        auto centered = tensors_.Get(inputs.centered.id);
        if (!centered.ok()) return centered.status();
        auto inverse = tensors_.Get(inputs.inverse_stddev.id);
        if (!inverse.ok()) return inverse.status();
        auto gamma = parameters_.GetVector(gamma_handle->id);
        if (!gamma.ok()) return gamma.status();
        auto beta = parameters_.GetVector(beta_handle->id);
        if (!beta.ok()) return beta.status();
        auto normalized = LayerNormTailKernel::Evaluate(
            centered.value(), inverse.value(), gamma.value(), beta.value(),
            layer_, context, report);
        if (!normalized.ok()) return normalized.status();

        auto q_weights = parameters_.GetMatrix(weights.wq.id);
        if (!q_weights.ok()) return q_weights.status();
        auto k_weights = parameters_.GetMatrix(weights.wk.id);
        if (!k_weights.ok()) return k_weights.status();
        auto v_weights = parameters_.GetMatrix(weights.wv.id);
        if (!v_weights.ok()) return v_weights.status();
        auto load_bias = [&](const std::string &name)
            -> Result<std::optional<DenseVectorParameter>> {
            const auto *handle = FindPlain(weights, name);
            if (!layer_.qkv_bias) return std::optional<DenseVectorParameter>{};
            if (!handle)
                return Status::Invalid(name + " handle is absent");
            auto value = parameters_.GetVector(handle->id);
            if (!value.ok()) return value.status();
            return std::optional<DenseVectorParameter>{value.take_value()};
        };
        auto bq = load_bias("bq");
        if (!bq.ok()) return bq.status();
        auto bk = load_bias("bk");
        if (!bk.ok()) return bk.status();
        auto bv = load_bias("bv");
        if (!bv.ok()) return bv.status();

        const auto terminal_level =
            context.parameters.coefficient_modulus_bits.size() -
            context.parameters.terminal_prime_count;
        constexpr std::size_t kQkDepth = 5;
        const auto qk_start_level = terminal_level - kQkDepth;
        if (qk_start_level == 0)
            return Status::FailedPrecondition(
                "QK suffix leaves no level for its projection");
        const auto qk_projection_input_level = qk_start_level - 1;
        auto qk_normalized = normalized.value();
        auto normalized_level = LevelOf(
            context, qk_normalized.ciphertexts.front().parms_id());
        if (!normalized_level.ok()) return normalized_level.status();
        if (normalized_level.value() > qk_projection_input_level)
            return Status::FailedPrecondition(
                "normalized input is already below the QK suffix start");
        if (normalized_level.value() < qk_projection_input_level) {
            const auto levels = Levels(context);
            if (qk_projection_input_level >= levels.size())
                return Status::FailedPrecondition(
                    "QK projection suffix exceeds modulus chain");
            const auto target = levels[qk_projection_input_level]->parms_id();
            seal::Evaluator evaluator(*context.context);
            for (auto &ct : qk_normalized.ciphertexts)
                evaluator.mod_switch_to_inplace(ct, target);
            qk_normalized.logical_spec.planned_level =
                qk_projection_input_level;
        }

        // Q and K share rotations at the suffix level. V deliberately keeps
        // the earlier level-2 representation required by Attention x V.
        ComplexDiagonalProjectionKernel qk_projection(layer_, context, report);
        auto q = qk_projection.Evaluate(
            qk_normalized, q_weights.value(), bq.value(),
            Layout::kMultiHeadQ);
        if (!q.ok()) return q.status();
        auto k = qk_projection.Evaluate(
            qk_normalized, k_weights.value(), bk.value(),
            Layout::kMultiHeadK);
        if (!k.ok()) return k.status();
        ComplexDiagonalProjectionKernel value_projection(
            layer_, context, report);
        auto v = value_projection.Evaluate(
            normalized.value(), v_weights.value(), bv.value(),
            Layout::kMultiHeadV);
        if (!v.ok()) return v.status();
        auto scores = MultiHeadQkKernel::Evaluate(
            q.value(), k.value(), layer_, context, report);
        if (!scores.ok()) return scores.status();

        auto score_handle = tensors_.Allocate(scores.take_value());
        if (!score_handle.ok()) return score_handle.status();
        auto value_handle = tensors_.Allocate(v.take_value());
        if (!value_handle.ok()) return value_handle.status();
        return Block2HeOutputs{score_handle.take_value(),
                               value_handle.take_value()};
    } catch (const std::exception &error) {
        return Status::Internal(std::string("Block 2 HE evaluation: ") +
                                error.what());
    }
}

} // namespace hera
