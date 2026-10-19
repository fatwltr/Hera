#pragma once

#include "fixed_ckks_transform.h"

namespace hera::conversion_internal::blb {

// Explicit bit/scale ledger for BLB Algorithm 2's fixed-point realization.
// The encoder input is ALREADY a sharing in slot_work_bits. Adapting a narrow
// MPC sharing to this contract is a separate, counted secure extension.
struct M2cEncodingPlan {
    std::size_t degree{};
    int slot_ring_bits{}, slot_scale_bits{};
    int slot_work_bits{}, guard_bits{}, root_scale_bits{};
    int coefficient_ring_bits{}, coefficient_scale_bits{};
    int coefficient_work_bits{}, coefficient_precision_bits{};
    int he_scale_bits{}, shift{}, rns_work_bits{};
};
Result<M2cEncodingPlan> PlanM2cEncoding(
    std::size_t degree, int slot_ring_bits, int slot_scale_bits,
    int he_scale_bits, int statistical_bits, int fft_bits,
    bool any_packed_real);

struct PreparedSlotShares {
    SlotSharePair values;
    int ring_bits{};
    int scale_bits{};
};
struct EncodedCoefficientShares {
    std::vector<std::uint64_t> values;
    int ring_bits{};
    int scale_bits{};
};

// Algorithm 2, step 1: LOCAL encoding, with no network or OT. The transform
// and this function together account for root products, 1/N and truncation.
// Result coefficients have an explicitly smaller ring, so the later R2F
// stage must not reuse the pre-FFT fixed wrap correction without refreshing.
Result<EncodedCoefficientShares> EncodePreparedSlots(
    const PreparedSlotShares &slots, const M2cEncodingPlan &plan,
    const FixedCkksTransform &transform, bool negative_share_orientation);

} // namespace hera::conversion_internal::blb
