#include "bounded_carry.h"

#include "bounded_ot.h"
#include "utils/net_io_channel.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>

namespace hera::conversion_internal {
namespace {

WideRingShare FromBlock(emp::block block, int bits)
{
    const WideRingShare value{
        static_cast<std::uint64_t>(_mm_extract_epi64(block, 0)),
        static_cast<std::uint64_t>(_mm_extract_epi64(block, 1))};
    return WideAdd(value, {}, bits);
}

struct BlockBuffer {
    explicit BlockBuffer(std::size_t count) : data(new emp::block[count]) {}
    ~BlockBuffer() { delete[] data; }
    BlockBuffer(const BlockBuffer &) = delete;
    BlockBuffer &operator=(const BlockBuffer &) = delete;
    emp::block *data;
};

} // namespace

std::vector<WideRingShare> LocalBitOrToArithmetic(
    int party, ConversionOt &ot, const std::vector<std::uint8_t> &local_bits,
    int bits, std::uint64_t weight)
{
    if (bits < 1 || bits > 127 ||
        (party != sci::ALICE && party != sci::BOB) ||
        local_bits.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("invalid arithmetic OR contract");
    const auto count = local_bits.size();
    std::vector<WideRingShare> result(count);
    if (!count) return result;
    std::unique_ptr<bool[]> choices(new bool[count]);
    for (std::size_t i = 0; i < count; ++i) choices[i] = local_bits[i] & 1U;
    if (bits <= 64) {
        const auto mask = bits == 64 ? UINT64_MAX : (std::uint64_t{1} << bits) - 1;
        std::vector<std::uint64_t> output(count);
        if (party == sci::ALICE) {
            std::vector<std::uint64_t> delta(count);
            for (std::size_t i = 0; i < count; ++i)
                delta[i] = choices[i] ? 0 : weight & mask;
            ot.send_cot(output.data(), delta.data(), count, bits);
            ot.ferret->io->flush();
            for (std::size_t i = 0; i < count; ++i)
                result[i] = {(weight * choices[i] - output[i]) & mask, 0};
        } else {
            ot.recv_cot(output.data(), choices.get(), count, bits);
            for (std::size_t i = 0; i < count; ++i) result[i] = {output[i] & mask, 0};
        }
        return result;
    }
    const auto bytes_per_value = static_cast<std::size_t>((bits + 7) / 8);
    if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()) / bytes_per_value)
        throw std::invalid_argument("arithmetic OR wire payload exceeds NetIO length");
    std::vector<std::uint8_t> correction(count * bytes_per_value);
    BlockBuffer pad0(count);
    if (party == sci::ALICE) {
        BlockBuffer pad1(count);
        ot.send_ot_rm_cc(pad0.data, pad1.data, count);
        for (std::size_t i = 0; i < count; ++i) {
            const auto x0 = FromBlock(pad0.data[i], bits);
            const auto x1 = FromBlock(pad1.data[i], bits);
            const auto delta = WideFromLow(choices[i] ? 0 : weight, bits);
            const auto y = WideAdd(WideAdd(x0, delta, bits), x1, bits);
            std::memcpy(correction.data() + i * bytes_per_value, &y.low, sizeof(y.low));
            std::memcpy(correction.data() + i * bytes_per_value + sizeof(y.low),
                        &y.high, bytes_per_value - sizeof(y.low));
            result[i] = WideSubtract(WideFromLow(choices[i] ? weight : 0, bits), x0, bits);
        }
        ot.ferret->io->send_data(correction.data(), correction.size());
        ot.ferret->io->flush();
    } else {
        ot.recv_ot_rm_cc(pad0.data, choices.get(), count);
        ot.ferret->io->recv_data(correction.data(), correction.size());
        for (std::size_t i = 0; i < count; ++i) {
            auto value = FromBlock(pad0.data[i], bits);
            if (choices[i]) {
                WideRingShare y{};
                std::memcpy(&y.low, correction.data() + i * bytes_per_value, sizeof(y.low));
                std::memcpy(&y.high, correction.data() + i * bytes_per_value + sizeof(y.low),
                            bytes_per_value - sizeof(y.low));
                value = WideSubtract(y, value, bits);
            }
            result[i] = value;
        }
    }
    return result;
}

std::vector<WideRingShare> BoundedCenteredPrimeToRing(
    int party, ConversionOt &ot, const std::vector<std::uint64_t> &shares,
    std::uint64_t modulus, int output_bits)
{
    if (modulus < 5 || !(modulus & 1U) || modulus > (UINT64_MAX >> 1))
        throw std::invalid_argument("invalid bounded prime-to-ring modulus");
    const auto center = modulus / 4;
    std::vector<std::uint64_t> adjusted(shares.size());
    std::vector<std::uint8_t> high(shares.size());
    for (std::size_t i = 0; i < shares.size(); ++i) {
        auto value = shares[i] % modulus;
        if (party == sci::ALICE)
            value = value >= modulus - center ? value - (modulus - center) : value + center;
        adjusted[i] = value;
        high[i] = value >= modulus / 2 + 1;
    }
    auto correction = LocalBitOrToArithmetic(party, ot, high, output_bits, modulus);
    for (std::size_t i = 0; i < shares.size(); ++i) {
        correction[i] = WideSubtract(WideFromLow(adjusted[i], output_bits), correction[i], output_bits);
        if (party == sci::ALICE)
            correction[i] = WideSubtract(correction[i], WideFromLow(center, output_bits), output_bits);
    }
    return correction;
}

} // namespace hera::conversion_internal
