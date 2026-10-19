#pragma once

#include <chrono>

namespace hera::detail {

// Time an enclosing computation interval on its caller thread. Parallel work
// must finish before the scope exits; nested intervals must use other totals.
class ScopedWallTimer {
public:
    explicit ScopedWallTimer(double *milliseconds) noexcept
        : milliseconds_(milliseconds), started_(Clock::now()) {}
    ScopedWallTimer(const ScopedWallTimer &) = delete;
    ScopedWallTimer &operator=(const ScopedWallTimer &) = delete;
    ~ScopedWallTimer()
    {
        if (milliseconds_)
            *milliseconds_ += std::chrono::duration<double, std::milli>(
                Clock::now() - started_).count();
    }

private:
    using Clock = std::chrono::steady_clock;
    double *milliseconds_;
    Clock::time_point started_;
};

} // namespace hera::detail
