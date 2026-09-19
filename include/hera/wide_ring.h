#pragma once

#include "hera/status.h"

#include <cstdint>
#include <string>

namespace hera {

// A wire-stable two-limb representation for rings wider than uint64_t. HERA
// deliberately caps working rings at 127 bits so signed intermediate products
// remain representable by the compiler's native 128-bit integer extension.
struct WideRingShare {
    std::uint64_t low{};
    std::uint64_t high{};
};

Result<void> ValidateWideRingBits(int bits);
WideRingShare WideFromLow(std::uint64_t value, int bits);
WideRingShare WideAdd(WideRingShare left, WideRingShare right, int bits);
WideRingShare WideSubtract(WideRingShare left, WideRingShare right, int bits);
WideRingShare WideNegate(WideRingShare value, int bits);
WideRingShare WideShiftLeft(WideRingShare value, int shift, int bits);
WideRingShare WideSignedShiftRight(WideRingShare value, int shift, int bits);

// Two-party local probabilistic truncation convention used by BLB/EzPC.
// The server applies arithmetic shift directly. The client applies
// -ARS(-share), so the reconstructed result differs from exact truncation by
// at most one except with the statistical-extension failure probability.
WideRingShare WideLocalTruncate(WideRingShare value, int shift, int bits,
                                bool negative_share_orientation);
WideRingShare WideMultiplySignedPublic(WideRingShare value,
                                       std::int64_t multiplier,
                                       int bits);
// Interprets both local ring shares as signed two's-complement integers,
// computes (left*left_multiplier + right*right_multiplier) >> shift in a
// signed 128-bit accumulator, and reduces the result back to the ring.
WideRingShare WideMultiplyAccumulateShift(
    WideRingShare left, std::int64_t left_multiplier,
    WideRingShare right, std::int64_t right_multiplier,
    int shift, int bits);
std::uint64_t WideModuloPrime(WideRingShare value, int bits,
                              std::uint64_t prime);
std::string WideToHex(WideRingShare value);

} // namespace hera
