#pragma once

#include "hera/status.h"

#include <algorithm>
#include <cstddef>

namespace hera::detail {

struct LayerNormRsqrtPlan {
    int input_bits{};
    int output_bits{};
    int lift{};
    int native_bits{};
};

// BLB's CKKS LayerNorm calls FPMath::sqrt(..., true), whose SCI schedule
// uses an (m+1)-bit LUT backed by eight KKOT types, and uint64_t products.
// Check that schedule explicitly: upstream asserts disappear in Release.
inline Result<LayerNormRsqrtPlan> PlanLayerNormRsqrt(
    int ring_bits, int variance_scale, int configured_bits, int output_scale)
{
    if (ring_bits < 2 || ring_bits > 63 || variance_scale < 1 ||
        variance_scale >= ring_bits - 1 || output_scale < 1 ||
        output_scale > variance_scale || configured_bits <= output_scale + 1 ||
        configured_bits > ring_bits)
        return Status::Invalid("LayerNorm rsqrt numeric contract is invalid");

    const int half_scale = (output_scale + 1) / 2;
    const int m = output_scale <= 14 ? half_scale : (half_scale + 1) / 2;
    if (m + 1 > 8)
        return Status::Invalid(
            "LayerNorm rsqrt output scale exceeds SCI's 8-bit LUT capacity (maximum scale 28)");

    LayerNormRsqrtPlan plan;
    plan.input_bits = std::min(ring_bits, configured_bits + variance_scale - output_scale);
    plan.output_bits = std::max(configured_bits, output_scale + (variance_scale + 1) / 2 + 2);
    plan.lift = variance_scale & 1;
    plan.native_bits = (plan.input_bits + plan.lift + 1) & ~1;
    if (plan.input_bits <= variance_scale + 1 || plan.native_bits > 64 ||
        plan.native_bits - 2 < output_scale || plan.native_bits - m - 2 < 0)
        return Status::Invalid("LayerNorm rsqrt working range is too narrow");
    if (plan.output_bits > ring_bits)
        return Status::Invalid("LayerNorm rsqrt result does not fit the sharing ring");
    // SQRT_LOOKUP_SCALE=2. Include the second Newton iteration and the
    // final exponent adjustment, not only the public input/output widths.
    const int intermediate_bits = std::max({2 * m + 4 + output_scale + 2,
        output_scale + m + 4, output_scale + 1 + plan.native_bits / 2,
        output_scale > 14 ? 2 * output_scale + 3 : 0});
    if (intermediate_bits > 64)
        return Status::Invalid("LayerNorm rsqrt SCI intermediate exceeds 64 bits");
    return plan;
}

inline Result<int> SoftmaxAccumulatorBits(
    std::size_t tokens, int ring_bits, int scale_bits)
{
    if (tokens == 0 || ring_bits < 2 || ring_bits > 63 ||
        scale_bits < 1 || scale_bits >= ring_bits)
        return Status::Invalid("Softmax normalization numeric contract is invalid");
    int token_bits = 0;
    for (auto value = tokens - 1; value; value >>= 1U) ++token_bits;
    const int required = scale_bits + token_bits + 1;
    const int bits = std::min(ring_bits, std::max(19, required));
    if (bits < required || bits <= scale_bits ||
        bits + 1 >= 64 || bits + scale_bits >= 64)
        return Status::Invalid("Softmax normalization accumulator/reciprocal workspace exceeds its ring or 64-bit backend");
    return bits;
}

} // namespace hera::detail
