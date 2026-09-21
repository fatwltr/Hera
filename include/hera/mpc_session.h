#pragma once

#include "hera/conversion.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

namespace hera {

// A single ordered SCI channel shared by the nonlinear MPC kernels.  SCI OT
// streams are stateful, so callers must use this session sequentially and in
// the same order on both parties.  HERA's stage orchestrator provides that
// ordering.  Conversion keeps its own worker sessions because those workers
// execute concurrently.
class SharedMpcSession {
public:
    class Impl;

    SharedMpcSession(PartyRole role, std::string peer_address, int port);
    ~SharedMpcSession();

    SharedMpcSession(const SharedMpcSession &) = delete;
    SharedMpcSession &operator=(const SharedMpcSession &) = delete;

    // Establishes the sockets and VOLE/Silent-OT state during setup.  Acquire
    // also initializes lazily for standalone kernel tests.
    Result<void> Prepare();
    std::shared_ptr<Impl> Acquire();

    void Cancel() noexcept;
    std::uint64_t communication_bytes() const noexcept;
    std::uint64_t rounds() const noexcept;

private:
    PartyRole role_;
    std::string peer_address_;
    int port_{};
    mutable std::mutex mutex_;
    std::shared_ptr<Impl> impl_;
};

} // namespace hera
