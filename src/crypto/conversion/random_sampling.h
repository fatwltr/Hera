#pragma once

#include "hera/conversion.h"

namespace hera::conversion_internal {

class FixedCkksTransform;

// Exact uniform sampling. Prime sampling rejects the incomplete high interval
// of a 64-bit word instead of applying a biased modulo reduction.
std::vector<std::uint64_t> SampleUniformPrime(IRandomStream &stream,
                                               std::size_t size,
                                               std::uint64_t modulus);
std::vector<std::uint64_t> SampleUniformRing(IRandomStream &stream,
                                              std::size_t size, int bits);

// Locally encode a known complex Z_(2^ell) mask into coefficient residues
// modulo q. scale_shift converts its MPC fixed-point scale to the HE scale.
// Coefficients may exceed q; signed reduction is done after wide encoding.
Result<std::vector<std::uint64_t>> EncodeRingMask(
    const std::vector<std::uint64_t> &real,
    const std::vector<std::uint64_t> &imag, int ring_bits, int scale_shift,
    int guard_bits, const FixedCkksTransform &transform, std::uint64_t modulus);

} // namespace hera::conversion_internal
