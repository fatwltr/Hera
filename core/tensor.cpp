#include "hera/tensor.h"

#include <limits>

namespace hera {

Result<std::size_t> TensorShape::ElementCount() const
{
    if (dimensions.empty())
        return Status::Invalid("tensor shape is empty");
    std::size_t result = 1;
    for (const auto dimension : dimensions) {
        if (dimension == 0)
            return Status::Invalid("tensor shape contains a zero dimension");
        if (result > std::numeric_limits<std::size_t>::max() / dimension)
            return Status::Invalid("tensor element count overflows size_t");
        result *= dimension;
    }
    return result;
}

const WeightHandle *TransformerWeightHandles::Find(
    const std::string &layer_id, const std::string &weight_name) const
{
    for (const auto &layer : layers) {
        if (layer.layer_id != layer_id) continue;
        if (weight_name == "Wq") return &layer.wq;
        if (weight_name == "Wk") return &layer.wk;
        if (weight_name == "Wv") return &layer.wv;
        if (weight_name == "Wo") return &layer.wo;
        if (weight_name == "Wu") return &layer.wu;
        if (weight_name == "Wd") return &layer.wd;
        return nullptr;
    }
    return nullptr;
}

const PlainParameterHandle *TransformerWeightHandles::FindPlain(
    const std::string &layer_id, const std::string &parameter_name) const
{
    for (const auto &layer : layers) {
        if (layer.layer_id != layer_id) continue;
        for (const auto &parameter : layer.plain_parameters)
            if (parameter.spec.name == parameter_name) return &parameter;
        return nullptr;
    }
    return nullptr;
}

Result<void> ValidateTensorSpec(const TensorSpec &spec,
                                const std::string &name)
{
    const auto count = spec.shape.ElementCount();
    if (!count.ok()) return Status::Invalid(name + ": " + count.status().message());
    if (spec.numeric.ring_bits < 1 || spec.numeric.ring_bits > 63 ||
        spec.numeric.scale_bits < 0 ||
        spec.numeric.scale_bits >= spec.numeric.ring_bits)
        return Status::Invalid(name + ": invalid numeric format");
    if (spec.domain == Domain::kMpcBooleanShare &&
        (spec.numeric.ring_bits != 1 || spec.numeric.scale_bits != 0))
        return Status::Invalid(name + ": boolean shares require 1/0 format");
    if (spec.domain == Domain::kHeCiphertext &&
        (!spec.he_region || !spec.he_parameter_profile))
        return Status::Invalid(name + ": HE tensor has no region/profile");
    if (spec.domain != Domain::kHeCiphertext &&
        (spec.he_region || spec.he_parameter_profile))
        return Status::Invalid(name +
                               ": non-HE tensor has an HE region/profile");
    return {};
}

bool SameTensorContract(const TensorSpec &left,
                        const TensorSpec &right) noexcept
{
    return left.shape == right.shape && left.domain == right.domain &&
           left.layout == right.layout &&
           left.numeric.ring_bits == right.numeric.ring_bits &&
           left.numeric.scale_bits == right.numeric.scale_bits &&
           left.numeric.is_signed == right.numeric.is_signed &&
           left.he_region == right.he_region &&
           left.he_parameter_profile == right.he_parameter_profile &&
           left.planned_level == right.planned_level &&
           left.planned_scale_bits == right.planned_scale_bits;
}

bool SatisfiesTensorContract(const TensorSpec &actual,
                             const TensorSpec &expected) noexcept
{
    if (actual.shape != expected.shape || actual.domain != expected.domain ||
        actual.layout != expected.layout ||
        actual.numeric.ring_bits != expected.numeric.ring_bits ||
        actual.numeric.scale_bits != expected.numeric.scale_bits ||
        actual.numeric.is_signed != expected.numeric.is_signed ||
        actual.he_region != expected.he_region ||
        actual.he_parameter_profile != expected.he_parameter_profile)
        return false;
    if (expected.planned_level &&
        actual.planned_level != expected.planned_level)
        return false;
    if (expected.planned_scale_bits &&
        actual.planned_scale_bits != expected.planned_scale_bits)
        return false;
    return true;
}

std::string ToString(Domain value)
{
    switch (value) {
    case Domain::kHeCiphertext: return "he_ciphertext";
    case Domain::kMpcArithmeticShare: return "mpc_arithmetic";
    case Domain::kMpcBooleanShare: return "mpc_boolean";
    case Domain::kPlaintext: return "plaintext";
    }
    return "unknown";
}

std::string ToString(Layout value)
{
    switch (value) {
    case Layout::kSpatial: return "spatial";
    case Layout::kReduction: return "reduction";
    case Layout::kMultiHeadQ: return "multihead_q";
    case Layout::kMultiHeadK: return "multihead_k";
    case Layout::kMultiHeadV: return "multihead_v";
    case Layout::kAttentionDiagonal: return "attention_diagonal";
    case Layout::kSoftmaxRows: return "softmax_rows";
    case Layout::kSoftmaxAvDiagonal: return "softmax_av_diagonal";
    case Layout::kAttentionOutput: return "attention_output";
    case Layout::kComplexStrictColumn: return "complex_strict_column";
    case Layout::kStrictColumn: return "strict_column";
    case Layout::kFfn: return "ffn";
    }
    return "unknown";
}

std::string ToString(HeRegionId value)
{
    switch (value) {
    case HeRegionId::kRegion1: return "region1";
    case HeRegionId::kRegion2: return "region2";
    case HeRegionId::kRegion3: return "region3";
    case HeRegionId::kRegion4: return "region4";
    case HeRegionId::kRegion5: return "region5";
    }
    return "unknown";
}

Result<HeRegionId> ParseHeRegionId(const std::string &value)
{
    if (value == "region1") return HeRegionId::kRegion1;
    if (value == "region2") return HeRegionId::kRegion2;
    if (value == "region3") return HeRegionId::kRegion3;
    if (value == "region4") return HeRegionId::kRegion4;
    if (value == "region5") return HeRegionId::kRegion5;
    return Status::Invalid("unknown HE region: " + value);
}

Result<Layout> ParseLayout(const std::string &value)
{
    if (value == "spatial") return Layout::kSpatial;
    if (value == "reduction") return Layout::kReduction;
    if (value == "multihead_q") return Layout::kMultiHeadQ;
    if (value == "multihead_k") return Layout::kMultiHeadK;
    if (value == "multihead_v") return Layout::kMultiHeadV;
    if (value == "attention_diagonal") return Layout::kAttentionDiagonal;
    if (value == "softmax_rows") return Layout::kSoftmaxRows;
    if (value == "softmax_av_diagonal")
        return Layout::kSoftmaxAvDiagonal;
    if (value == "attention_output") return Layout::kAttentionOutput;
    if (value == "complex_strict_column")
        return Layout::kComplexStrictColumn;
    if (value == "strict_column") return Layout::kStrictColumn;
    if (value == "ffn") return Layout::kFfn;
    return Status::Invalid("unknown layout: " + value);
}

} // namespace hera
