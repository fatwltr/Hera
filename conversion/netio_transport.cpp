#include "hera/conversion.h"

#include "utils/net_io_channel.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace hera {
namespace {

class NetIoTransport final : public IProtocolTransport {
public:
    NetIoTransport(PartyRole role, const std::string &peer_address, int port)
        : io_(role == PartyRole::kServer ? nullptr : peer_address.c_str(), port)
    {
        if (port < 1 || port > 65535)
            throw std::invalid_argument("protocol port is outside [1,65535]");
        if (role == PartyRole::kClient && peer_address.empty())
            throw std::invalid_argument("client peer address is empty");
    }

    void Send(const void *data, std::size_t bytes) override
    {
        const auto *cursor = static_cast<const std::uint8_t *>(data);
        while (bytes != 0) {
            const std::size_t chunk = std::min<std::size_t>(
                bytes, static_cast<std::size_t>(std::numeric_limits<int>::max()));
            io_.send_data(cursor, static_cast<int>(chunk));
            cursor += chunk;
            bytes -= chunk;
        }
        ++logical_rounds_;
    }

    void Receive(void *data, std::size_t bytes) override
    {
        auto *cursor = static_cast<std::uint8_t *>(data);
        while (bytes != 0) {
            const std::size_t chunk = std::min<std::size_t>(
                bytes, static_cast<std::size_t>(std::numeric_limits<int>::max()));
            io_.recv_data(cursor, static_cast<int>(chunk));
            cursor += chunk;
            bytes -= chunk;
        }
        ++logical_rounds_;
    }

    void Cancel() noexcept override { io_.cancel(); }

    std::uint64_t bytes() const noexcept override { return io_.counter; }
    std::uint64_t rounds() const noexcept override
    {
        return std::max<std::uint64_t>(logical_rounds_, io_.num_rounds);
    }

private:
    sci::NetIO io_;
    std::uint64_t logical_rounds_{};
};

} // namespace

std::shared_ptr<IProtocolTransport> MakeNetIoTransport(
    PartyRole role, const std::string &peer_address, int port)
{
    return std::make_shared<NetIoTransport>(role, peer_address, port);
}

} // namespace hera
