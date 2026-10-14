#pragma once

#include "hera/mpc_session.h"
#include "hera/detail/protocol_worker.h"

#include "FloatingPoint/fixed-point.h"
#include "FloatingPoint/fp-math.h"
#include "OT/ot_pack.h"
#include "utils/io_pack.h"

#include <memory>
#include <functional>
#include <future>

namespace hera {

// Internal native view.  It lives in a separate header so the framework-only
// target never needs SCI headers.
class SharedMpcSession::Impl {
public:
    Impl(PartyRole role, const std::string &peer_address, int port);
    ~Impl();

    int party() const noexcept { return party_; }
    FixOp &fix() noexcept { return *math_->fix; }
    FPMath &math() noexcept { return *math_; }
    std::uint64_t communication_bytes() const noexcept;
    std::uint64_t rounds() const noexcept;
    void Cancel() noexcept;
    void NegotiateWorkers(std::size_t workers);
    std::future<void> Submit(std::function<void()> operation);

private:
    int party_{};
    std::unique_ptr<sci::IOPack> io_;
    std::unique_ptr<sci::OTPack> ot_;
    std::unique_ptr<FPMath> math_;
    // Join protocol threads before destroying their native objects.
    std::unique_ptr<detail::ProtocolWorkerThread> executor_;
};

} // namespace hera
