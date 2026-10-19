#pragma once

#include "hera/status.h"
#include "hera/wide_ring.h"

#include <cstddef>
#include <optional>
#include <vector>

namespace hera::conversion_internal {

struct WideComplex {
    WideRingShare real;
    WideRingShare imag;
};

struct SlotSharePair {
    std::vector<WideRingShare> real;
    std::vector<WideRingShare> imag;
};

// Fixed-point implementation of CKKS' canonical embedding. It deliberately
// operates on local additive shares; no plaintext reconstruction occurs.
class FixedCkksTransform {
public:
    static Result<FixedCkksTransform> Create(std::size_t polynomial_degree,
                                             int root_scale_bits,
                                             int working_bits,
                                             bool negative_share_truncation = false);

    // Forward canonical embedding: N real coefficient shares -> N/2 complex
    // slot shares in Microsoft SEAL's matrix-representation order.
    Result<SlotSharePair> Decode(
        const std::vector<WideRingShare> &coefficient_shares) const;

    // Inverse canonical embedding: N/2 complex slot shares -> N real
    // coefficient shares. The caller is responsible for the CKKS scale.
    Result<std::vector<WideRingShare>> Encode(
        const SlotSharePair &slot_shares) const;

    // Conservative public bound for the reconstructed coefficient integer
    // after guard truncation. Includes fixed-root and local rounding error;
    // subject to the transform's existing statistical truncation contract.
    std::optional<int> EncodedCoefficientBoundBits(
        int slot_bound_bits, int guard_bits, int retained_precision_bits) const;
    // Public coefficient contract -> reconstructed decoded fixed-point range.
    // Uses the same root/rounding and statistical assumptions as Decode.
    std::optional<int> DecodedSlotBoundBits(
        int coefficient_bound_bits, int guard_bits, int scale_shift) const;

    std::size_t degree() const noexcept { return degree_; }
    std::size_t slot_count() const noexcept { return degree_ / 2; }
    int root_scale_bits() const noexcept { return root_scale_bits_; }
    int working_bits() const noexcept { return working_bits_; }

private:
    void TransformToReverse(std::vector<WideComplex> &values) const;
    void TransformFromReverse(std::vector<WideComplex> &values) const;

    std::size_t degree_{};
    int root_scale_bits_{};
    int working_bits_{};
    bool negative_share_truncation_{};
    std::vector<std::size_t> matrix_index_map_;
    std::vector<std::pair<std::int64_t, std::int64_t>> root_powers_;
    std::vector<std::pair<std::int64_t, std::int64_t>> inverse_root_powers_;
};

} // namespace hera::conversion_internal
