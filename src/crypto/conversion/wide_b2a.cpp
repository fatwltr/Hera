#include "wide_b2a.h"

#include "OT/ot_pack.h"

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

std::vector<WideRingShare> BooleanToWideArithmetic(
    int party, sci::OTPack &ot, const std::vector<std::uint8_t> &shares,
    int bits)
{
    if (bits < 65 || bits > 127 ||
        (party != sci::ALICE && party != sci::BOB) ||
        shares.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("invalid wide B2A contract");
    const auto count = shares.size();
    std::vector<WideRingShare> result(count);
    if (count == 0) return result;
    const auto bytes_per_value = static_cast<std::size_t>((bits + 7) / 8);
    if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()) / bytes_per_value)
        throw std::invalid_argument("wide B2A wire payload exceeds NetIO length");
    std::vector<std::uint8_t> correction(count * bytes_per_value);
    // The SCI facade provides correlation-robust hashed random messages
    // through send/recv_ot_rm_cc. Keep the complete 128-bit pad so reduction
    // to Z_(2^bits) is uniform even above 64 bits.
    BlockBuffer pad0(count);
    if (party == sci::ALICE) {
        BlockBuffer pad1(count);
        ot.iknp_straight->send_ot_rm_cc(pad0.data, pad1.data, count);
        for (std::size_t i = 0; i < count; ++i) {
            const auto x0 = FromBlock(pad0.data[i], bits);
            const auto x1 = FromBlock(pad1.data[i], bits);
            const auto bit = WideFromLow(shares[i] & 1U, bits);
            // Receiver obtains x0 for choice=0 and x0-2*sender_bit
            // for choice=1. Adding its Boolean share reconstructs XOR.
            const auto delta = WideNegate(WideShiftLeft(bit, 1, bits), bits);
            const auto y = WideAdd(WideAdd(x0, delta, bits), x1, bits);
            std::memcpy(correction.data() + i * bytes_per_value,
                        &y.low, sizeof(y.low));
            std::memcpy(correction.data() + i * bytes_per_value + sizeof(y.low),
                        &y.high, bytes_per_value - sizeof(y.low));
            result[i] = WideSubtract(bit, x0, bits);
        }
        ot.iknp_straight->ferret->io->send_data(
            correction.data(), correction.size());
        ot.iknp_straight->ferret->io->flush();
    } else {
        std::unique_ptr<bool[]> choices(new bool[count]);
        for (std::size_t i = 0; i < count; ++i) choices[i] = shares[i] & 1U;
        ot.iknp_straight->recv_ot_rm_cc(pad0.data, choices.get(), count);
        ot.iknp_straight->ferret->io->recv_data(
            correction.data(), correction.size());
        for (std::size_t i = 0; i < count; ++i) {
            auto received = FromBlock(pad0.data[i], bits);
            if (choices[i]) {
                WideRingShare y{};
                std::memcpy(&y.low, correction.data() + i * bytes_per_value, sizeof(y.low));
                std::memcpy(&y.high, correction.data() + i * bytes_per_value + sizeof(y.low),
                            bytes_per_value - sizeof(y.low));
                received = WideSubtract(y, received, bits);
            }
            result[i] = WideAdd(received,
                WideFromLow(choices[i] ? 1 : 0, bits), bits);
        }
    }
    return result;
}

} // namespace hera::conversion_internal
