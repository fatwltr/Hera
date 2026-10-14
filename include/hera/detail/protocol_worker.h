#pragma once

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace hera::detail {

// Keep SCI's OpenMP team off the caller thread. In particular, a caller may
// fork another party after destroying a converter; inheriting a main-thread
// OpenMP team in that child would leave it waiting for vanished worker threads.
class ProtocolWorkerThread {
public:
    ProtocolWorkerThread() : thread_([this] { Run(); }) {}
    ~ProtocolWorkerThread()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        wake_.notify_one();
        thread_.join();
    }

    std::future<void> Submit(std::function<void()> operation)
    {
        auto task = std::make_shared<std::packaged_task<void()>>(std::move(operation));
        auto result = task->get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (pending_ || stopping_)
                throw std::logic_error("protocol worker is not idle");
            pending_ = [task] { (*task)(); };
        }
        wake_.notify_one();
        return result;
    }

private:
    void Run()
    {
        for (;;) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                wake_.wait(lock, [&] { return stopping_ || pending_; });
                if (!pending_) return;
                task = std::move(pending_);
                pending_ = {};
            }
            task();
        }
    }

    std::mutex mutex_;
    std::condition_variable wake_;
    std::function<void()> pending_;
    bool stopping_{};
    std::thread thread_;
};

} // namespace hera::detail
