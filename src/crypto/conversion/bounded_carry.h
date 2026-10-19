#pragma once

#include "hera/wide_ring.h"

#include <cstdint>
#include <vector>

namespace sci { class NetIO; }
namespace cheetah { template<class IO> class SilentOT; }

namespace hera::conversion_internal {

// One independently keyed forward pool; bounded conversion needs no reverse OT.
using ConversionOt = cheetah::SilentOT<sci::NetIO>;


// Each party owns one PRIVATE input bit, rather than an XOR share of a bit.
// Produces additive shares of weight * (alice_bit OR bob_bit) in Z_(2^bits).
// Uses fresh OT material for every element, with no intermediate Boolean
// carry sharing or separate B2A. Supports narrow and wide output rings.
std::vector<WideRingShare> LocalBitOrToArithmetic(
    int party, ConversionOt &ot, const std::vector<std::uint8_t> &local_bits,
    int bits, std::uint64_t weight = 1);

// PRECONDITION: the centered secret reconstructed modulo the odd modulus q
// satisfies |x| < floor(q/4). This is a PUBLIC producer contract; local
// uniformly random shares are never range-tested to choose the protocol.
std::vector<WideRingShare> BoundedCenteredPrimeToRing(
    int party, ConversionOt &ot, const std::vector<std::uint64_t> &shares,
    std::uint64_t modulus, int output_bits);

} // namespace hera::conversion_internal
