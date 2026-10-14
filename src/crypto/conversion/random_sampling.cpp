#include "random_sampling.h"
#include "fixed_ckks_transform.h"

#include <seal/util/uintcore.h>

#include <algorithm>
#include <array>
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
    std::array<std::uint64_t, 256> candidates;
    std::size_t written = 0;
    while (written != size) {
        const auto count = std::min(candidates.size(), size - written);
        stream.Fill(candidates.data(), count * sizeof(candidates.front()));
        for (std::size_t i = 0; i < count; ++i) {
            const auto word = candidates[i];
            if (static_cast<unsigned __int128>(word) >= limit) continue;
            result[written++] = word % modulus;
        }
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

Result<std::vector<std::uint64_t>> EncodeRingMask(
    const std::vector<std::uint64_t> &real,
    const std::vector<std::uint64_t> &imag, int ring_bits, int scale_shift,
    int guard_bits, const FixedCkksTransform &transform, std::uint64_t modulus)
{
    if (ring_bits < 2 || ring_bits > 63 || scale_shift < 0 || guard_bits < 0 ||
        modulus < 2 || real.size() != transform.slot_count() ||
        imag.size() != transform.slot_count())
        return Status::Invalid("invalid Z-mask encoding parameters");
    const int bits = transform.working_bits();
    // The inverse transform normalizes only at its last stage. Reserve room
    // for the unnormalized butterflies as well as complex component growth.
    const std::int64_t required_bits = static_cast<std::int64_t>(ring_bits) +
        scale_shift + guard_bits +
        seal::util::get_power_of_two(transform.degree()) + 2;
    if (required_bits > bits)
        return Status::FailedPrecondition("Z-mask encoding workspace is too small");

    const auto mask = LowMask(ring_bits);
    const auto sign = std::uint64_t{1} << (ring_bits - 1);
    const auto ring_modulus = WideShiftLeft(WideFromLow(1, bits), ring_bits, bits);
    const auto encode_slot = [&](std::uint64_t value) {
        value &= mask;
        auto wide = WideFromLow(value, bits);
        if (value & sign) wide = WideSubtract(wide, ring_modulus, bits);
        return WideShiftLeft(wide, scale_shift + guard_bits, bits);
    };
    SlotSharePair slots;
    slots.real.resize(real.size());
    slots.imag.resize(imag.size());
    for (std::size_t i = 0; i < real.size(); ++i) {
        slots.real[i] = encode_slot(real[i]);
        slots.imag[i] = encode_slot(imag[i]);
    }
    auto encoded = transform.Encode(slots);
    if (!encoded.ok()) return encoded.status();
    std::vector<std::uint64_t> result(transform.degree());
    for (std::size_t i = 0; i < result.size(); ++i) {
        const auto coefficient = WideSignedShiftRight(encoded.value()[i],
                                                       guard_bits, bits);
        const bool negative = bits <= 64
            ? ((coefficient.low >> (bits - 1)) & 1U) != 0
            : ((coefficient.high >> (bits - 65)) & 1U) != 0;
        const auto magnitude = negative ? WideNegate(coefficient, bits)
                                        : coefficient;
        const auto residue = WideModuloPrime(magnitude, bits, modulus);
        result[i] = negative && residue != 0 ? modulus - residue : residue;
    }
    return result;
}

} // namespace hera::conversion_internal
