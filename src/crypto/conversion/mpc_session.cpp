#include "hera/detail/mpc_session_internal.h"
#include "hera/detail/worker_ports.h"

#include <exception>
#include <algorithm>
#include <condition_variable>
#include <stdexcept>
#include <thread>
#include <utility>

namespace hera {

SharedMpcSession::Impl::Impl(
    PartyRole role, const std::string &peer_address, int port)
    : party_(role == PartyRole::kServer ? sci::ALICE : sci::BOB),
      io_(std::make_unique<sci::IOPack>(
          party_, port,
          role == PartyRole::kServer ? std::string{} : peer_address)),
      ot_(std::make_unique<sci::OTPack>(io_.get(), party_)),
      math_(std::make_unique<FPMath>(party_, io_.get(), ot_.get())),
      executor_(std::make_unique<detail::ProtocolWorkerThread>())
{}

SharedMpcSession::Impl::~Impl() = default;

std::future<void> SharedMpcSession::Impl::Submit(std::function<void()> operation)
{ return executor_->Submit(std::move(operation)); }

void SharedMpcSession::Impl::NegotiateWorkers(std::size_t workers)
{
    const std::uint64_t contract[] = {0x484552414d504301ULL, workers};
    std::uint64_t peer[2]{};
    if (party_ == sci::ALICE) {
        io_->io->send_data(contract, sizeof(contract));
        io_->io->recv_data(peer, sizeof(peer));
    } else {
        io_->io->recv_data(peer, sizeof(peer));
        io_->io->send_data(contract, sizeof(contract));
    }
    if (peer[0] != contract[0] || peer[1] != contract[1])
        throw std::invalid_argument("MPC worker contracts differ between parties");
}

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
    PartyRole role, std::string peer_address, int port, std::size_t workers)
    : role_(role), peer_address_(std::move(peer_address)), port_(port), worker_count_(workers)
{
    detail::ValidateWorkerPortRange(port, detail::MpcWorkerPortSpan(workers));
}

SharedMpcSession::~SharedMpcSession() = default;

std::shared_ptr<SharedMpcSession::Impl> SharedMpcSession::Acquire(std::size_t worker)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (worker >= worker_count_) throw std::out_of_range("MPC worker index");
    if (workers_.empty()) {
        auto first = std::make_shared<Impl>(role_, peer_address_, port_);
        first->NegotiateWorkers(worker_count_);
        workers_.push_back(std::move(first));
        for (std::size_t i = 1; i < worker_count_; ++i)
            workers_.push_back(std::make_shared<Impl>(role_, peer_address_,
                port_ + static_cast<int>(detail::MpcWorkerPortOffset(i))));
    }
    if (workers_.size() != worker_count_)
        throw std::runtime_error("MPC pool initialization did not complete");
    return workers_[worker];
}

void SharedMpcSession::ForEachRange(std::size_t units, std::size_t minimum_units_per_worker,
    const std::function<void(std::size_t, std::size_t, std::size_t)> &operation)
{
    if (!units) return;
    if (!minimum_units_per_worker) throw std::invalid_argument("MPC range grain is zero");
    std::lock_guard<std::mutex> ordered(operation_mutex_);
    (void)Acquire();
    const auto active = std::min(worker_count_, (units - 1) / minimum_units_per_worker + 1);
    std::vector<std::future<void>> futures;
    futures.reserve(active);
    std::exception_ptr failure;
    try {
        for (std::size_t i = 0; i < active; ++i) {
            const auto begin = units / active * i + std::min(i, units % active);
            const auto end = begin + units / active + (i < units % active ? 1 : 0);
            futures.push_back(Acquire(i)->Submit([&, i, begin, end] {
                try { operation(i, begin, end); }
                catch (...) { Cancel(); throw; }
            }));
        }
    } catch (...) { failure = std::current_exception(); Cancel(); }
    for (auto &future : futures) {
        try { future.get(); }
        catch (...) { if (!failure) failure = std::current_exception(); }
    }
    if (failure) std::rethrow_exception(failure);
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
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto &worker : workers_) worker->Cancel();
}

std::uint64_t SharedMpcSession::communication_bytes() const noexcept
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::uint64_t total = 0;
    for (const auto &worker : workers_) total += worker->communication_bytes();
    return total;
}

std::uint64_t SharedMpcSession::rounds() const noexcept
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::uint64_t total = 0;
    for (const auto &worker : workers_) total += worker->rounds();
    return total;
}

} // namespace hera
