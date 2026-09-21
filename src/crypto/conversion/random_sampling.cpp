#include "random_sampling.h"

#include <limits>
#include <stdexcept>

namespace hera::conversion_internal {
namespace {

std::uint64_t LowMask(int bits)
{
    return bits == 64 ? std::numeric_limits<std::uint64_t>::max()
                      : (std::uint64_t{1} << bits) - 1;
}

} // namespace

std::vector<std::uint64_t> SampleUniformPrime(IRandomStream &stream,
                                               std::size_t size,
                                               std::uint64_t modulus)
{
    if (modulus < 2) throw std::invalid_argument("invalid sampling modulus");
    const unsigned __int128 universe =
        static_cast<unsigned __int128>(1) << 64;
    const unsigned __int128 limit = (universe / modulus) * modulus;
    std::vector<std::uint64_t> result(size);
    std::size_t written = 0;
    while (written != size) {
        std::uint64_t word{};
        stream.Fill(&word, sizeof(word));
        if (static_cast<unsigned __int128>(word) >= limit) continue;
        result[written++] = word % modulus;
    }
    return result;
}

std::vector<std::uint64_t> SampleUniformRing(IRandomStream &stream,
                                              std::size_t size, int bits)
{
    if (bits < 1 || bits > 64)
        throw std::invalid_argument("invalid uniform ring width");
    std::vector<std::uint64_t> result(size);
    if (!result.empty())
        stream.Fill(result.data(), result.size() * sizeof(result.front()));
    const std::uint64_t mask = LowMask(bits);
    for (auto &value : result) value &= mask;
    return result;
}

} // namespace hera::conversion_internal
