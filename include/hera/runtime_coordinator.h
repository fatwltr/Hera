#pragma once

#include "hera/conversion.h"

#include <chrono>
#include <functional>
#include <memory>
#include <string>

namespace hera {

// A control plane independent of the ciphertext and OT transports. It never
// carries cryptographic payloads; its only purpose is liveness, stage-order
// agreement and prompt cancellation after a peer failure.
class RuntimeCoordinator {
public:
    static Result<std::unique_ptr<RuntimeCoordinator>> Connect(
        PartyRole role, const std::string &peer_address, int port,
        std::chrono::milliseconds heartbeat = std::chrono::seconds(2),
        std::chrono::milliseconds timeout = std::chrono::seconds(30),
        // Zero preserves the heartbeat timeout as the stage deadline.
        std::chrono::milliseconds stage_timeout = std::chrono::milliseconds(0));

    ~RuntimeCoordinator();
    RuntimeCoordinator(const RuntimeCoordinator &) = delete;
    RuntimeCoordinator &operator=(const RuntimeCoordinator &) = delete;

    void SetCancellationCallback(std::function<void()> callback);
    Result<void> Start();
    Result<void> StageComplete(std::size_t layer, std::size_t stage);
    void Abort(const Status &status, std::size_t layer,
               std::size_t stage) noexcept;
    Result<void> Finish();
    void Cancel() noexcept;
    Status peer_status() const;

private:
    class Impl;
    explicit RuntimeCoordinator(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace hera
