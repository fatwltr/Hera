#include "hera/runtime_coordinator.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

namespace hera {
namespace {

constexpr std::uint32_t kMagic = 0x48455241U;
constexpr std::uint16_t kVersion = 1;
constexpr std::size_t kMaxMessage = 4096;
enum class Kind : std::uint16_t { kHeartbeat = 1, kStage = 2,
                                  kAbort = 3, kFinish = 4 };

struct Header {
    std::uint32_t magic;
    std::uint16_t version;
    std::uint16_t kind;
    std::uint64_t sequence;
    std::uint32_t layer;
    std::uint32_t stage;
    std::uint32_t status;
    std::uint32_t bytes;
};

std::uint64_t HostToNetwork64(std::uint64_t value)
{
    const auto high = htonl(static_cast<std::uint32_t>(value >> 32U));
    const auto low = htonl(static_cast<std::uint32_t>(value));
    return (static_cast<std::uint64_t>(low) << 32U) | high;
}

std::uint64_t NetworkToHost64(std::uint64_t value)
{
    return HostToNetwork64(value);
}

bool WriteAll(int fd, const void *data, std::size_t bytes)
{
    auto *cursor = static_cast<const std::uint8_t *>(data);
    while (bytes != 0) {
        const auto written = ::send(fd, cursor, bytes, MSG_NOSIGNAL);
        if (written <= 0) return false;
        cursor += written;
        bytes -= static_cast<std::size_t>(written);
    }
    return true;
}

bool ReadAll(int fd, void *data, std::size_t bytes)
{
    auto *cursor = static_cast<std::uint8_t *>(data);
    while (bytes != 0) {
        const auto read = ::recv(fd, cursor, bytes, 0);
        if (read <= 0) return false;
        cursor += read;
        bytes -= static_cast<std::size_t>(read);
    }
    return true;
}

Result<int> OpenSocket(PartyRole role, const std::string &peer, int port,
                       std::chrono::milliseconds timeout)
{
    if (port < 1 || port > 65535)
        return Status::Invalid("coordinator port is outside [1,65535]");
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    if (role == PartyRole::kServer) {
        const int listener = ::socket(AF_INET, SOCK_STREAM, 0);
        if (listener < 0) return Status::Internal("coordinator socket failed");
        int reuse = 1;
        setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_ANY);
        address.sin_port = htons(static_cast<std::uint16_t>(port));
        if (::bind(listener, reinterpret_cast<sockaddr *>(&address),
                   sizeof(address)) != 0 || ::listen(listener, 1) != 0) {
            ::close(listener);
            return Status::Internal("coordinator bind/listen failed");
        }
        const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now()).count();
        pollfd descriptor{listener, POLLIN, 0};
        if (::poll(&descriptor, 1, static_cast<int>(remaining)) <= 0) {
            ::close(listener);
            return Status::DeadlineExceeded("coordinator accept timed out");
        }
        const int connected = ::accept(listener, nullptr, nullptr);
        ::close(listener);
        if (connected < 0) return Status::Internal("coordinator accept failed");
        return connected;
    }
    if (peer.empty()) return Status::Invalid("coordinator peer address is empty");
    while (std::chrono::steady_clock::now() < deadline) {
        const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) return Status::Internal("coordinator socket failed");
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_port = htons(static_cast<std::uint16_t>(port));
        if (::inet_pton(AF_INET, peer.c_str(), &address.sin_addr) == 1 &&
            ::connect(fd, reinterpret_cast<sockaddr *>(&address),
                      sizeof(address)) == 0)
            return fd;
        ::close(fd);
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
    return Status::DeadlineExceeded("coordinator connect timed out");
}

} // namespace

class RuntimeCoordinator::Impl {
public:
    Impl(int socket, std::chrono::milliseconds heartbeat,
         std::chrono::milliseconds timeout)
        : socket_(socket), heartbeat_(heartbeat), timeout_(timeout),
          last_receive_(std::chrono::steady_clock::now())
    {}

    ~Impl() { Cancel(); Join(); }

    Result<void> Start()
    {
        if (running_.exchange(true))
            return Status::FailedPrecondition("coordinator already started");
        monitor_ = std::thread([this] { Monitor(); });
        return {};
    }

    void SetCancellationCallback(std::function<void()> callback)
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        cancellation_ = std::move(callback);
    }

    Result<void> StageComplete(std::size_t layer, std::size_t stage)
    {
        const auto sequence = next_sequence_.fetch_add(1);
        if (!Send(Kind::kStage, sequence, layer, stage, Status::Ok(), {}))
            return Fail(Status::Cancelled("coordinator stage send failed"));
        std::unique_lock<std::mutex> lock(state_mutex_);
        if (!condition_.wait_for(lock, timeout_, [&] {
                return !peer_status_.ok() ||
                       (peer_sequence_ == sequence && peer_layer_ == layer &&
                        peer_stage_ == stage);
            })) {
            lock.unlock();
            return Fail(Status::DeadlineExceeded(
                "peer did not complete the matching stage within 30 seconds"));
        }
        return peer_status_.ok() ? Result<void>{} : Result<void>{peer_status_};
    }

    void Abort(const Status &status, std::size_t layer,
               std::size_t stage) noexcept
    {
        Send(Kind::kAbort, next_sequence_.load(), layer, stage, status,
             status.message().substr(0, kMaxMessage));
        FailNoThrow(status);
    }

    Result<void> Finish()
    {
        finishing_ = true;
        Send(Kind::kFinish, next_sequence_.load(), 0, 0, Status::Ok(), {});
        {
            std::unique_lock<std::mutex> lock(state_mutex_);
            condition_.wait_for(lock, timeout_, [&] {
                return peer_finished_ || !peer_status_.ok();
            });
            if (!peer_finished_ && peer_status_.ok())
                peer_status_ = Status::DeadlineExceeded(
                    "peer did not finish within 30 seconds");
        }
        running_ = false;
        ::shutdown(socket_, SHUT_RDWR);
        Join();
        return peer_status_.ok() ? Result<void>{} : Result<void>{peer_status_};
    }

    void Cancel() noexcept
    {
        running_ = false;
        if (socket_ >= 0) ::shutdown(socket_, SHUT_RDWR);
        condition_.notify_all();
    }

    Status peer_status() const
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        return peer_status_;
    }

private:
    bool Send(Kind kind, std::uint64_t sequence, std::size_t layer,
              std::size_t stage, const Status &status,
              const std::string &message) noexcept
    {
        if (message.size() > kMaxMessage) return false;
        Header header{kMagic, kVersion, static_cast<std::uint16_t>(kind),
                      sequence, static_cast<std::uint32_t>(layer),
                      static_cast<std::uint32_t>(stage),
                      static_cast<std::uint32_t>(status.code()),
                      static_cast<std::uint32_t>(message.size())};
        header.magic = htonl(header.magic);
        header.version = htons(header.version);
        header.kind = htons(header.kind);
        header.sequence = HostToNetwork64(header.sequence);
        header.layer = htonl(header.layer);
        header.stage = htonl(header.stage);
        header.status = htonl(header.status);
        header.bytes = htonl(header.bytes);
        std::lock_guard<std::mutex> lock(send_mutex_);
        return WriteAll(socket_, &header, sizeof(header)) &&
               (message.empty() || WriteAll(socket_, message.data(),
                                             message.size()));
    }

    Result<void> Fail(Status status)
    {
        FailNoThrow(status);
        return status;
    }

    void FailNoThrow(Status status) noexcept
    {
        std::function<void()> callback;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (peer_status_.ok()) peer_status_ = std::move(status);
            callback = cancellation_;
        }
        condition_.notify_all();
        if (callback) callback();
    }

    void Monitor()
    {
        auto next_heartbeat = std::chrono::steady_clock::now();
        while (running_) {
            const auto now = std::chrono::steady_clock::now();
            if (now >= next_heartbeat) {
                if (!Send(Kind::kHeartbeat, next_sequence_.load(), 0, 0,
                          Status::Ok(), {})) {
                    bool finished = false;
                    {
                        std::lock_guard<std::mutex> lock(state_mutex_);
                        finished = peer_finished_;
                    }
                    if (!finished && !finishing_)
                        FailNoThrow(Status::Cancelled(
                            "coordinator peer closed"));
                    break;
                }
                next_heartbeat = now + heartbeat_;
            }
            pollfd descriptor{socket_, POLLIN | POLLHUP | POLLERR, 0};
            const int ready = ::poll(&descriptor, 1, 100);
            if (ready < 0) {
                FailNoThrow(Status::Cancelled("coordinator socket closed"));
                break;
            }
            if (ready > 0 && (descriptor.revents & POLLIN)) {
                Header header{};
                if (!ReadAll(socket_, &header, sizeof(header))) {
                    if (finishing_) {
                        std::lock_guard<std::mutex> lock(state_mutex_);
                        peer_finished_ = true;
                        condition_.notify_all();
                    } else
                        FailNoThrow(Status::Cancelled(
                            "coordinator reached EOF"));
                    break;
                }
                header.magic = ntohl(header.magic);
                header.version = ntohs(header.version);
                header.kind = ntohs(header.kind);
                header.sequence = NetworkToHost64(header.sequence);
                header.layer = ntohl(header.layer);
                header.stage = ntohl(header.stage);
                header.status = ntohl(header.status);
                header.bytes = ntohl(header.bytes);
                if (header.magic != kMagic || header.version != kVersion ||
                    header.bytes > kMaxMessage) {
                    FailNoThrow(Status::PeerAborted(
                        "invalid runtime control frame"));
                    break;
                }
                std::string message(header.bytes, '\0');
                if (header.bytes && !ReadAll(socket_, message.data(),
                                             header.bytes)) {
                    if (!finishing_)
                        FailNoThrow(Status::Cancelled(
                            "truncated control frame"));
                    break;
                }
                last_receive_ = std::chrono::steady_clock::now();
                const auto kind = static_cast<Kind>(header.kind);
                if (kind == Kind::kStage) {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    peer_sequence_ = header.sequence;
                    peer_layer_ = header.layer;
                    peer_stage_ = header.stage;
                    condition_.notify_all();
                } else if (kind == Kind::kAbort) {
                    FailNoThrow(Status::PeerAborted(message.empty()
                        ? "peer aborted runtime stage" : message));
                    break;
                } else if (kind == Kind::kFinish) {
                    {
                        std::lock_guard<std::mutex> lock(state_mutex_);
                        peer_finished_ = true;
                    }
                    condition_.notify_all();
                    // Keep the monitor alive until the local side also calls
                    // Finish; this prevents FIN/POLLHUP from racing the peer's
                    // final control frame.
                }
            }
            if (ready > 0 &&
                (descriptor.revents & (POLLHUP | POLLERR | POLLNVAL))) {
                bool finished = false;
                {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    finished = peer_finished_;
                }
                if (!finished && finishing_) {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    peer_finished_ = true;
                    condition_.notify_all();
                } else if (!finished)
                    FailNoThrow(Status::Cancelled(
                        "coordinator socket closed"));
                break;
            }
            bool finished = false;
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                finished = peer_finished_;
            }
            if (!finished && std::chrono::steady_clock::now() -
                    last_receive_ > timeout_) {
                FailNoThrow(Status::DeadlineExceeded(
                    "runtime peer heartbeat timed out after 30 seconds"));
                break;
            }
        }
        running_ = false;
        condition_.notify_all();
    }

    void Join()
    {
        if (monitor_.joinable() &&
            monitor_.get_id() != std::this_thread::get_id()) monitor_.join();
        if (socket_ >= 0) { ::close(socket_); socket_ = -1; }
    }

    int socket_{-1};
    std::chrono::milliseconds heartbeat_;
    std::chrono::milliseconds timeout_;
    std::atomic<bool> running_{false};
    std::atomic<bool> finishing_{false};
    std::atomic<std::uint64_t> next_sequence_{0};
    std::thread monitor_;
    mutable std::mutex state_mutex_;
    std::mutex send_mutex_;
    std::condition_variable condition_;
    Status peer_status_;
    std::uint64_t peer_sequence_{static_cast<std::uint64_t>(-1)};
    std::size_t peer_layer_{};
    std::size_t peer_stage_{};
    bool peer_finished_{};
    std::chrono::steady_clock::time_point last_receive_;
    std::function<void()> cancellation_;
};

RuntimeCoordinator::RuntimeCoordinator(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
RuntimeCoordinator::~RuntimeCoordinator() = default;

Result<std::unique_ptr<RuntimeCoordinator>> RuntimeCoordinator::Connect(
    PartyRole role, const std::string &peer_address, int port,
    std::chrono::milliseconds heartbeat, std::chrono::milliseconds timeout)
{
    if (heartbeat.count() <= 0 || timeout <= heartbeat)
        return Status::Invalid("invalid coordinator heartbeat/timeout");
    auto socket = OpenSocket(role, peer_address, port, timeout);
    if (!socket.ok()) return socket.status();
    return std::unique_ptr<RuntimeCoordinator>(new RuntimeCoordinator(
        std::make_unique<Impl>(socket.value(), heartbeat, timeout)));
}

void RuntimeCoordinator::SetCancellationCallback(std::function<void()> callback)
{ impl_->SetCancellationCallback(std::move(callback)); }
Result<void> RuntimeCoordinator::Start() { return impl_->Start(); }
Result<void> RuntimeCoordinator::StageComplete(std::size_t layer,
                                               std::size_t stage)
{ return impl_->StageComplete(layer, stage); }
void RuntimeCoordinator::Abort(const Status &status, std::size_t layer,
                               std::size_t stage) noexcept
{ impl_->Abort(status, layer, stage); }
Result<void> RuntimeCoordinator::Finish() { return impl_->Finish(); }
void RuntimeCoordinator::Cancel() noexcept { impl_->Cancel(); }
Status RuntimeCoordinator::peer_status() const { return impl_->peer_status(); }

} // namespace hera
