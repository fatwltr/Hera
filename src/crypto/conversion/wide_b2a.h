#pragma once

#include "hera/wide_ring.h"

#include <cstdint>
#include <vector>

namespace sci { class OTPack; }

namespace hera::conversion_internal {

// XOR-shared bits -> additive shares in Z_(2^bits), using one full-width
// additive correlation per random OT. No independent carry between limbs.
std::vector<WideRingShare> BooleanToWideArithmetic(
    int party, sci::OTPack &ot, const std::vector<std::uint8_t> &shares,
    int bits);

} // namespace hera::conversion_internal
