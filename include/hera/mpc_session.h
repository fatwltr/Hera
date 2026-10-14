#pragma once

#include "hera/conversion.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace hera {

// Persistent independent SCI channels shared by the nonlinear MPC kernels.
// Stages remain ordered; independent ranges within a stage use distinct OT
// streams. Both parties must use the same worker count and public partition.
class SharedMpcSession {
public:
    class Impl;

    // IOPack uses p, p+50, p+100. Consecutive worker ports are disjoint
    // for at most 50 workers; reserve the complete p..p+149 range.
    SharedMpcSession(PartyRole role, std::string peer_address, int port,
                     std::size_t workers = 1);
    ~SharedMpcSession();

    SharedMpcSession(const SharedMpcSession &) = delete;
    SharedMpcSession &operator=(const SharedMpcSession &) = delete;

    // Establishes the sockets and VOLE/Silent-OT state during setup.  Acquire
    // also initializes lazily for standalone kernel tests.
    Result<void> Prepare();
    std::shared_ptr<Impl> Acquire(std::size_t worker = 0);
    std::size_t worker_count() const noexcept { return worker_count_; }

    // Runs on persistent protocol threads, including the single-worker case.
    // Joins every range before returning and cancels all channels on failure.
    void ForEachRange(std::size_t units, std::size_t minimum_units_per_worker,
        const std::function<void(std::size_t, std::size_t, std::size_t)> &operation);

    void Cancel() noexcept;
    std::uint64_t communication_bytes() const noexcept;
    std::uint64_t rounds() const noexcept;

private:
    PartyRole role_;
    std::string peer_address_;
    int port_{};
    std::size_t worker_count_{};
    mutable std::mutex mutex_;
    std::mutex operation_mutex_;
    std::vector<std::shared_ptr<Impl>> workers_;
};

} // namespace hera
