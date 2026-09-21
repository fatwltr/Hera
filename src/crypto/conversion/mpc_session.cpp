#include "hera/detail/mpc_session_internal.h"

#include <exception>
#include <utility>

namespace hera {

SharedMpcSession::Impl::Impl(
    PartyRole role, const std::string &peer_address, int port)
    : party_(role == PartyRole::kServer ? sci::ALICE : sci::BOB),
      io_(std::make_unique<sci::IOPack>(
          party_, port,
          role == PartyRole::kServer ? std::string{} : peer_address)),
      ot_(std::make_unique<sci::OTPack>(io_.get(), party_)),
      math_(std::make_unique<FPMath>(party_, io_.get(), ot_.get()))
{}

std::uint64_t SharedMpcSession::Impl::communication_bytes() const noexcept
{
    return io_ ? io_->get_comm() : 0;
}

std::uint64_t SharedMpcSession::Impl::rounds() const noexcept
{
    return io_ ? io_->get_rounds() : 0;
}

void SharedMpcSession::Impl::Cancel() noexcept
{
    if (io_) io_->cancel();
}

SharedMpcSession::SharedMpcSession(
    PartyRole role, std::string peer_address, int port)
    : role_(role), peer_address_(std::move(peer_address)), port_(port)
{}

SharedMpcSession::~SharedMpcSession() = default;

std::shared_ptr<SharedMpcSession::Impl> SharedMpcSession::Acquire()
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!impl_)
        impl_ = std::make_shared<Impl>(role_, peer_address_, port_);
    return impl_;
}

Result<void> SharedMpcSession::Prepare()
{
    try {
        (void)Acquire();
        return {};
    } catch (const std::exception &error) {
        return Status::Internal(
            std::string("failed to prepare shared MPC session: ") +
            error.what());
    }
}

void SharedMpcSession::Cancel() noexcept
{
    std::shared_ptr<Impl> active;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        active = impl_;
    }
    if (active) active->Cancel();
}

std::uint64_t SharedMpcSession::communication_bytes() const noexcept
{
    std::lock_guard<std::mutex> lock(mutex_);
    return impl_ ? impl_->communication_bytes() : 0;
}

std::uint64_t SharedMpcSession::rounds() const noexcept
{
    std::lock_guard<std::mutex> lock(mutex_);
    return impl_ ? impl_->rounds() : 0;
}

} // namespace hera
