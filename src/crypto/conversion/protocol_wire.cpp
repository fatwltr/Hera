#include "hera/conversion.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace hera {
namespace {

constexpr std::uint64_t kBundleMagic = 0x484552414b455953ULL;
constexpr std::uint64_t kMaximumBlobBytes = std::uint64_t{1} << 34;
constexpr std::size_t kBundleChunkBytes = 256 * 1024;

void SendBytes(IProtocolTransport &transport,
               const std::vector<std::uint8_t> &bytes)
{
    const std::uint64_t size = bytes.size();
    transport.Send(&size, sizeof(size));
    for (std::size_t offset = 0; offset < bytes.size();) {
        const auto part = std::min(kBundleChunkBytes, bytes.size() - offset);
        transport.Send(bytes.data() + offset, part);
        offset += part;
    }
}

Result<std::vector<std::uint8_t>> ReceiveBytes(IProtocolTransport &transport)
{
    std::uint64_t size = 0;
    transport.Receive(&size, sizeof(size));
    if (size > kMaximumBlobBytes)
        return Status::Invalid("public-key bundle blob is too large");
    std::vector<std::uint8_t> result(static_cast<std::size_t>(size));
    for (std::size_t offset = 0; offset < result.size();) {
        const auto part = std::min(kBundleChunkBytes, result.size() - offset);
        transport.Receive(result.data() + offset, part);
        offset += part;
    }
    return result;
}

} // namespace

Result<void> SendPublicContextBundle(
    IProtocolTransport &transport,
    const SerializedPublicContextBundle &bundle)
{
    try {
        const std::uint64_t magic = kBundleMagic;
        const std::uint32_t version = bundle.protocol_version;
        const std::uint32_t count =
            static_cast<std::uint32_t>(bundle.profiles.size());
        transport.Send(&magic, sizeof(magic));
        transport.Send(&version, sizeof(version));
        transport.Send(&count, sizeof(count));
        for (const auto &profile : bundle.profiles) {
            if (profile.profile.empty() || profile.profile.size() > 4096)
                return Status::Invalid("public-key profile name is invalid");
            const std::uint32_t name_bytes =
                static_cast<std::uint32_t>(profile.profile.size());
            transport.Send(&name_bytes, sizeof(name_bytes));
            transport.Send(profile.profile.data(), profile.profile.size());
            SendBytes(transport, profile.public_key);
            SendBytes(transport, profile.relinearization_keys);
            SendBytes(transport, profile.galois_keys);
        }
        return {};
    } catch (const std::exception &error) {
        return Status::Internal(std::string("sending public HE bundle: ") +
                                error.what());
    }
}

Result<SerializedPublicContextBundle> ReceivePublicContextBundle(
    IProtocolTransport &transport)
{
    try {
        std::uint64_t magic = 0;
        std::uint32_t version = 0, count = 0;
        transport.Receive(&magic, sizeof(magic));
        transport.Receive(&version, sizeof(version));
        transport.Receive(&count, sizeof(count));
        if (magic != kBundleMagic || version != 2 || count > 1024)
            return Status::FailedPrecondition(
                "invalid public HE bundle frame");
        SerializedPublicContextBundle result;
        result.protocol_version = version;
        for (std::uint32_t i = 0; i < count; ++i) {
            std::uint32_t name_bytes = 0;
            transport.Receive(&name_bytes, sizeof(name_bytes));
            if (name_bytes == 0 || name_bytes > 4096)
                return Status::Invalid("public-key profile name is invalid");
            SerializedProfilePublicMaterial profile;
            profile.profile.resize(name_bytes);
            transport.Receive(profile.profile.data(), profile.profile.size());
            auto public_key = ReceiveBytes(transport);
            auto relin = ReceiveBytes(transport);
            auto galois = ReceiveBytes(transport);
            if (!public_key.ok()) return public_key.status();
            if (!relin.ok()) return relin.status();
            if (!galois.ok()) return galois.status();
            profile.public_key = public_key.take_value();
            profile.relinearization_keys = relin.take_value();
            profile.galois_keys = galois.take_value();
            result.profiles.push_back(std::move(profile));
        }
        return result;
    } catch (const std::exception &error) {
        return Status::Internal(std::string("receiving public HE bundle: ") +
                                error.what());
    }
}

} // namespace hera
