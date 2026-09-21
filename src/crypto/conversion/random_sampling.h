#pragma once

#include "hera/conversion.h"

namespace hera::conversion_internal {

// Exact uniform sampling. Prime sampling rejects the incomplete high interval
// of a 64-bit word instead of applying a biased modulo reduction.
std::vector<std::uint64_t> SampleUniformPrime(IRandomStream &stream,
                                               std::size_t size,
                                               std::uint64_t modulus);
std::vector<std::uint64_t> SampleUniformRing(IRandomStream &stream,
                                              std::size_t size, int bits);

} // namespace hera::conversion_internal
