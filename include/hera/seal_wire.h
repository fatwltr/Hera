#pragma once

#include "hera/conversion.h"

#include <seal/seal.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <istream>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <streambuf>
#include <vector>

namespace hera {

// Each SEAL object is a sequence of bounded chunks followed by a zero-length
// chunk. Neither endpoint materializes a second full serialized ciphertext.
constexpr std::size_t kSealWireChunkBytes = 256 * 1024;
constexpr std::uint64_t kMaximumSealWireBytes = std::uint64_t{1} << 34;

class SealWireOutputBuffer final : public std::streambuf {
public:
    explicit SealWireOutputBuffer(IProtocolTransport &transport)
        : transport_(transport), buffer_(kSealWireChunkBytes)
    {
        setp(buffer_.data(), buffer_.data() + buffer_.size());
    }

    void Finish()
    {
        Flush();
        const std::uint32_t terminator = 0;
        transport_.Send(&terminator, sizeof(terminator));
    }

protected:
    int_type overflow(int_type value) override
    {
        Flush();
        if (!traits_type::eq_int_type(value, traits_type::eof())) {
            *pptr() = traits_type::to_char_type(value);
            pbump(1);
        }
        return traits_type::not_eof(value);
    }

    std::streamsize xsputn(const char *data, std::streamsize size) override
    {
        const auto requested = size;
        while (size > 0) {
            if (pptr() == epptr()) Flush();
            const auto part = std::min<std::streamsize>(size, epptr() - pptr());
            std::memcpy(pptr(), data, static_cast<std::size_t>(part));
            pbump(static_cast<int>(part));
            data += part;
            size -= part;
        }
        return requested;
    }

    int sync() override
    {
        Flush();
        return 0;
    }

private:
    void Flush()
    {
        const auto size = static_cast<std::uint32_t>(pptr() - pbase());
        if (size == 0) return;
        if (sent_ > kMaximumSealWireBytes - size)
            throw std::runtime_error("serialized SEAL object is too large");
        transport_.Send(&size, sizeof(size));
        transport_.Send(pbase(), size);
        sent_ += size;
        setp(buffer_.data(), buffer_.data() + buffer_.size());
    }

    IProtocolTransport &transport_;
    std::vector<char> buffer_;
    std::uint64_t sent_{};
};

class SealWireInputBuffer final : public std::streambuf {
public:
    explicit SealWireInputBuffer(IProtocolTransport &transport)
        : transport_(transport), buffer_(kSealWireChunkBytes)
    {
        setg(buffer_.data(), buffer_.data(), buffer_.data());
    }

    void Finish()
    {
        if (gptr() != egptr())
            throw std::runtime_error("trailing bytes in SEAL object frame");
        std::uint32_t size = 0;
        transport_.Receive(&size, sizeof(size));
        if (size != 0)
            throw std::runtime_error("SEAL object frame was not terminated");
    }

protected:
    int_type underflow() override
    {
        if (gptr() != egptr())
            return traits_type::to_int_type(*gptr());
        std::uint32_t size = 0;
        transport_.Receive(&size, sizeof(size));
        if (size == 0) return traits_type::eof();
        if (size > buffer_.size() || received_ > kMaximumSealWireBytes - size)
            throw std::runtime_error("invalid SEAL wire chunk size");
        transport_.Receive(buffer_.data(), size);
        received_ += size;
        setg(buffer_.data(), buffer_.data(), buffer_.data() + size);
        return traits_type::to_int_type(*gptr());
    }

private:
    IProtocolTransport &transport_;
    std::vector<char> buffer_;
    std::uint64_t received_{};
};

template <typename SealObject>
void SendSealObject(IProtocolTransport &transport, const SealObject &object,
                    seal::compr_mode_type compression =
                        seal::compr_mode_type::none)
{
    if (!seal::Serialization::IsSupportedComprMode(compression))
        throw std::invalid_argument("requested SEAL compression is unavailable");
    SealWireOutputBuffer buffer(transport);
    std::ostream stream(&buffer);
    stream.exceptions(std::ios::badbit | std::ios::failbit);
    object.save(stream, compression);
    buffer.Finish();
}

template <typename SealObject>
SealObject ReceiveSealObject(IProtocolTransport &transport,
                             const seal::SEALContext &context)
{
    SealWireInputBuffer buffer(transport);
    std::istream stream(&buffer);
    stream.exceptions(std::ios::badbit);
    SealObject result;
    result.load(context, stream);
    buffer.Finish();
    return result;
}

} // namespace hera
