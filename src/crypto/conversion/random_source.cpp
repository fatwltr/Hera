#include "hera/conversion.h"

#include <openssl/hmac.h>
#include <openssl/sha.h>

#include <sys/random.h>

#include <array>
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <mutex>
#include <set>
#include <stdexcept>

namespace hera {
namespace {

using Key = std::array<std::uint8_t, 32>;

Key Hash(const void *data, std::size_t bytes)
{
    Key output{};
    SHA256(static_cast<const unsigned char *>(data), bytes, output.data());
    return output;
}

Key Derive(const Key &master, const std::string &label)
{
    Key output{};
    unsigned int bytes = 0;
    if (!HMAC(EVP_sha256(), master.data(), static_cast<int>(master.size()),
              reinterpret_cast<const unsigned char *>(label.data()),
              label.size(), output.data(), &bytes) || bytes != output.size())
        throw std::runtime_error("HMAC-SHA256 domain derivation failed");
    return output;
}

class HmacStream final : public IRandomStream {
public:
    explicit HmacStream(Key key) : key_(key) {}

    void Fill(void *data, std::size_t bytes) override
    {
        auto *destination = static_cast<std::uint8_t *>(data);
        while (bytes != 0) {
            if (position_ == block_.size()) {
                std::array<std::uint8_t, 8> encoded{};
                for (std::size_t i = 0; i < encoded.size(); ++i)
                    encoded[encoded.size() - 1 - i] =
                        static_cast<std::uint8_t>(counter_ >> (8 * i));
                unsigned int output_bytes = 0;
                if (!HMAC(EVP_sha256(), key_.data(), static_cast<int>(key_.size()),
                          encoded.data(), encoded.size(), block_.data(),
                          &output_bytes) || output_bytes != block_.size())
                    throw std::runtime_error("HMAC-SHA256 random stream failed");
                ++counter_;
                position_ = 0;
            }
            const auto take = std::min(bytes, block_.size() - position_);
            std::memcpy(destination, block_.data() + position_, take);
            position_ += take;
            destination += take;
            bytes -= take;
        }
    }

private:
    Key key_{};
    Key block_{};
    std::size_t position_{block_.size()};
    std::uint64_t counter_{};
};

class DomainSeparatedRandomSource final : public IRandomSource {
public:
    explicit DomainSeparatedRandomSource(Key master) : master_(master) {}

    Result<std::unique_ptr<IRandomStream>> OpenStream(
        const std::string &domain_label) override
    {
        if (domain_label.empty())
            return Status::Invalid("random domain label is empty");
        std::lock_guard<std::mutex> lock(mutex_);
        if (!labels_.insert(domain_label).second)
            return Status::FailedPrecondition(
                "random domain label was reused: " + domain_label);
        try {
            return std::unique_ptr<IRandomStream>(
                new HmacStream(Derive(master_, domain_label)));
        } catch (const std::exception &error) {
            return Status::Internal(error.what());
        }
    }

private:
    Key master_{};
    std::mutex mutex_;
    std::set<std::string> labels_;
};

Key OsSeed()
{
    Key key{};
    std::size_t offset = 0;
    while (offset < key.size()) {
        const auto result = getrandom(key.data() + offset,
                                      key.size() - offset, 0);
        if (result < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(std::string("getrandom failed: ") +
                                     std::strerror(errno));
        }
        offset += static_cast<std::size_t>(result);
    }
    return key;
}

} // namespace

std::shared_ptr<IRandomSource> MakeProductionRandomSource()
{
    return std::make_shared<DomainSeparatedRandomSource>(OsSeed());
}

std::shared_ptr<IRandomSource> MakeDeterministicRandomSourceForTesting(
    const std::vector<std::uint8_t> &seed)
{
    if (seed.empty())
        throw std::invalid_argument("deterministic random seed is empty");
    return std::make_shared<DomainSeparatedRandomSource>(
        Hash(seed.data(), seed.size()));
}

} // namespace hera
