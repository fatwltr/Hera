#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace hera::detail {

// SCI opens sockets at p, p+50 and p+100. Pack workers into disjoint
// 150-port blocks; 50 is a socket offset, not a worker-count limit.
inline std::size_t MpcWorkerPortOffset(std::size_t worker)
{
    return (worker / 50) * 150 + worker % 50;
}

inline std::size_t MpcWorkerPortSpan(std::size_t workers)
{
    if (workers == 0 || workers > 65535)
        throw std::invalid_argument("MPC workers do not fit the TCP port space");
    const auto span = MpcWorkerPortOffset(workers - 1) + 101;
    if (span > 65535)
        throw std::invalid_argument("MPC workers do not fit the TCP port space");
    return span;
}

// Preserve the converter's p+3*i layout. Its three socket sets have
// distinct residues modulo three, so they cannot overlap.
inline std::size_t ConversionWorkerPortSpan(std::size_t workers)
{
    if (workers == 0 || workers > (65535 - 101) / 3 + 1)
        throw std::invalid_argument("conversion workers do not fit the TCP port space");
    return 3 * (workers - 1) + 101;
}

inline void ValidateWorkerPortRange(int base, std::size_t span)
{
    if (base < 1024 || base > 65535 || span > 65536 - static_cast<std::size_t>(base))
        throw std::invalid_argument("worker sockets exceed the available TCP port range");
}

inline int RuntimeMpcPort(int base, std::size_t conversion_workers,
                         std::size_t mpc_workers)
{
    const auto conversion_span = ConversionWorkerPortSpan(conversion_workers);
    ValidateWorkerPortRange(base, 100 + conversion_span);
    const auto mpc_offset = std::max(std::size_t{300}, 100 + conversion_span);
    const auto mpc_span = MpcWorkerPortSpan(mpc_workers);
    ValidateWorkerPortRange(base, mpc_offset + mpc_span);
    return base + static_cast<int>(mpc_offset);
}

} // namespace hera::detail
