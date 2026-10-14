#pragma once

#include "hera/config.h"

namespace hera::detail {

// These schedules are used after profile/depth validation.
inline constexpr std::size_t kAttentionValueDepth = 3;

inline bool FusedAttentionOutput(const HeParameterConfig &profile)
{
    return profile.coefficient_modulus_bits.size() >= 10 && profile.declared_depth >= 9;
}

inline std::size_t AttentionValueInputLevel(const HeParameterConfig &profile)
{
    return profile.coefficient_modulus_bits.size() -
        profile.terminal_prime_count - kAttentionValueDepth -
        (FusedAttentionOutput(profile) ? 4 : 0);
}

// Wd needs one level; its output must retain the three-level LN1 suffix.
// The complete LN2 skip may start above this level and be mod-switched for add.
inline std::size_t SharedFfnOutputLevel(const HeParameterConfig &profile)
{
    const auto terminal = profile.coefficient_modulus_bits.size() -
        profile.terminal_prime_count;
    return terminal - 3;
}

} // namespace hera::detail
