#pragma once

#include "hera/status.h"

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <future>
#include <limits>
#include <mutex>
#include <memory>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>
#include <unistd.h>

namespace hera::detail {

// One process-local pool is reused by all independent HE kernels. A nested
// range call on a pool worker runs serially to avoid waiting on itself.
class HeWorkerPool {
public:
    static HeWorkerPool &Instance()
    {
        struct ProcessPool {
            pid_t pid{getpid()};
            std::unique_ptr<HeWorkerPool> pool{new HeWorkerPool};
        };
        static ProcessPool state;
        if (state.pid != getpid()) {
            // Threads do not survive fork. Destroying their copied std::thread
            // objects is also invalid, so discard the inherited holder.
            const auto inherited_budget = state.pool->memory_budget_;
            state.pool.release();
            state.pool.reset(new HeWorkerPool);
            state.pool->memory_budget_ = inherited_budget;
            state.pid = getpid();
        }
        return *state.pool;
    }

    HeWorkerPool(const HeWorkerPool &) = delete;
    HeWorkerPool &operator=(const HeWorkerPool &) = delete;

    void SetMemoryBudget(std::size_t bytes)
    {
        if (bytes == 0) throw std::invalid_argument("HE memory budget is zero");
        std::lock_guard<std::mutex> lock(mutex_);
        memory_budget_ = bytes;
    }

    std::size_t MemoryBudget() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return memory_budget_;
    }

    std::size_t WorkerThreadCount() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return workers_.size();
    }

    static bool OnWorkerThread() noexcept { return on_worker_; }

    template <typename Callable>
    auto Submit(std::size_t wanted_threads, Callable &&callable)
        -> std::future<std::invoke_result_t<std::decay_t<Callable>>>
    {
        using Return = std::invoke_result_t<std::decay_t<Callable>>;
        auto task = std::make_shared<std::packaged_task<Return()>>(
            std::forward<Callable>(callable));
        auto result = task->get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto target = std::max<std::size_t>(1, wanted_threads);
            while (workers_.size() < target)
                workers_.emplace_back([this] { WorkerLoop(); });
            queue_.emplace_back([task] { (*task)(); });
        }
        wake_.notify_one();
        return result;
    }

private:
    HeWorkerPool() = default;
public:
    ~HeWorkerPool()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        wake_.notify_all();
        for (auto &worker : workers_) worker.join();
    }

private:
    void WorkerLoop()
    {
        on_worker_ = true;
        for (;;) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                wake_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
                if (stopping_ && queue_.empty()) return;
                task = std::move(queue_.front());
                queue_.pop_front();
            }
            task();
        }
    }

    inline static thread_local bool on_worker_ = false;
    mutable std::mutex mutex_;
    std::condition_variable wake_;
    std::deque<std::function<void()>> queue_;
    std::vector<std::thread> workers_;
    std::size_t memory_budget_{512ULL << 20};
    bool stopping_{};
};

template <typename Callable>
auto SubmitLocalTask(std::size_t max_inflight, Callable &&operation)
    -> std::future<std::invoke_result_t<std::decay_t<Callable>>>
{
    if (max_inflight <= 1 || HeWorkerPool::OnWorkerThread())
        return std::async(std::launch::deferred, std::forward<Callable>(operation));
    return HeWorkerPool::Instance().Submit(max_inflight, std::forward<Callable>(operation));
}

inline std::size_t EstimatedHeWorkerBytes(std::size_t degree,
                                          std::size_t data_primes,
                                          std::size_t live_ciphertexts)
{
    const auto limit = std::numeric_limits<std::size_t>::max();
    if (degree == 0 || data_primes == 0 || live_ciphertexts == 0) return 0;
    const auto primes = data_primes + 1;
    if (degree > limit / primes / 16 ||
        live_ciphertexts > limit / (degree * primes * 16))
        return limit;
    return degree * primes * 16 * live_ciphertexts;
}

template <typename Operation>
Result<void> ParallelRanges(std::size_t count, std::size_t requested_workers,
                            std::size_t minimum_grain, Operation operation,
                            std::size_t estimated_bytes_per_worker = 0,
                            std::size_t *actual_workers = nullptr)
{
    if (actual_workers) *actual_workers = 0;
    if (count == 0) return {};
    const auto grain = std::max<std::size_t>(1, minimum_grain);
    auto workers = std::max<std::size_t>(1, std::min(
        requested_workers, 1 + (count - 1) / grain));
    if (estimated_bytes_per_worker != 0)
        workers = std::min(workers, std::max<std::size_t>(
            1, HeWorkerPool::Instance().MemoryBudget() /
                   estimated_bytes_per_worker));
    if (HeWorkerPool::OnWorkerThread()) workers = 1;
    if (actual_workers) *actual_workers = workers;
    std::vector<Status> statuses(workers);
    std::vector<std::future<void>> futures;
    futures.reserve(workers - 1);
    const auto run = [&](std::size_t worker) {
        try {
            auto result = operation(count * worker / workers,
                                    count * (worker + 1) / workers);
            if (!result.ok()) statuses[worker] = result.status();
        } catch (const std::exception &error) {
            statuses[worker] = Status::Internal(error.what());
        } catch (...) {
            statuses[worker] = Status::Internal("parallel HE worker failed");
        }
    };
    try {
        for (std::size_t worker = 1; worker < workers; ++worker)
            futures.push_back(HeWorkerPool::Instance().Submit(
                workers - 1, [&, worker] { run(worker); }));
    } catch (const std::exception &error) {
        for (auto &future : futures) future.wait();
        return Status::Internal(std::string("starting HE worker: ") +
                                error.what());
    }
    run(0);
    for (auto &future : futures) future.get();
    for (const auto &status : statuses)
        if (!status.ok()) return status;
    return {};
}

} // namespace hera::detail
