#include "blb_encoding.h"

#include <algorithm>

namespace hera::conversion_internal::blb {

Result<M2cEncodingPlan> PlanM2cEncoding(
    std::size_t degree, int ring_bits, int scale_bits,
    int he_scale_bits, int statistical_bits, int fft_bits,
    bool any_packed_real)
{
    if (degree < 2 || (degree & (degree - 1)) || ring_bits < 2 || ring_bits > 63 ||
        scale_bits < 0 || scale_bits >= ring_bits || statistical_bits < 1 || statistical_bits > 63 ||
        fft_bits < 1 || fft_bits > 50 || he_scale_bits < scale_bits || he_scale_bits > 126)
        return Status::Invalid("invalid BLB M2C precision contract");
    const int shift = he_scale_bits - scale_bits;
    const int base_bits = ring_bits + statistical_bits;
    if ((any_packed_real && shift == 0) || base_bits + fft_bits + 2 > 127 || base_bits + shift > 127)
        return Status::FailedPrecondition("BLB M2C precision exceeds the 127-bit workspace");
    M2cEncodingPlan plan;
    plan.degree = degree;
    plan.slot_ring_bits = ring_bits;
    plan.slot_scale_bits = scale_bits;
    plan.slot_work_bits = base_bits;
    plan.root_scale_bits = fft_bits;
    plan.guard_bits = std::min(20, 127 - base_bits - fft_bits - 2);
    plan.coefficient_precision_bits = std::min(
        {8, shift - static_cast<int>(any_packed_real), plan.guard_bits, 63 - ring_bits});
    plan.coefficient_ring_bits = ring_bits + plan.coefficient_precision_bits;
    plan.coefficient_scale_bits = scale_bits + plan.coefficient_precision_bits;
    plan.coefficient_work_bits = plan.coefficient_ring_bits + statistical_bits;
    plan.he_scale_bits = he_scale_bits;
    plan.shift = shift;
    plan.rns_work_bits = base_bits + shift;
    return plan;
}

Result<EncodedCoefficientShares> EncodePreparedSlots(
    const PreparedSlotShares &input, const M2cEncodingPlan &plan,
    const FixedCkksTransform &transform, bool negative_share_orientation)
{
    if (input.ring_bits != plan.slot_work_bits || input.scale_bits != plan.slot_scale_bits ||
        input.values.real.size() != plan.degree / 2 || input.values.imag.size() != plan.degree / 2 ||
        transform.degree() != plan.degree || transform.root_scale_bits() != plan.root_scale_bits ||
        transform.working_bits() != plan.slot_work_bits + plan.guard_bits ||
        plan.coefficient_ring_bits < 2 || plan.coefficient_ring_bits > 63 ||
        plan.coefficient_precision_bits < 0 || plan.coefficient_precision_bits > plan.guard_bits)
        return Status::Invalid("BLB local Encode requires prepared wide slot shares");
    auto guarded = input.values;
    for (auto *lane : {&guarded.real, &guarded.imag})
        for (auto &value : *lane)
            value = WideShiftLeft(value, plan.guard_bits, transform.working_bits());
    auto encoded = transform.Encode(guarded);
    if (!encoded.ok()) return encoded.status();
    EncodedCoefficientShares result;
    result.ring_bits = plan.coefficient_ring_bits;
    result.scale_bits = plan.coefficient_scale_bits;
    result.values.resize(plan.degree);
    const auto mask = (std::uint64_t{1} << result.ring_bits) - 1;
    for (std::size_t i = 0; i < plan.degree; ++i)
        result.values[i] = WideLocalTruncate(encoded.value()[i],
            plan.guard_bits - plan.coefficient_precision_bits, transform.working_bits(),
            negative_share_orientation).low & mask;
    return result;
}
} // namespace hera::conversion_internal::blb
