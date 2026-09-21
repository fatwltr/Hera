#include "hera/wide_ring.h"

#include <iomanip>
#include <sstream>

namespace hera {
namespace {

using U128 = unsigned __int128;
using I128 = __int128;

U128 Raw(WideRingShare value)
{
    return (static_cast<U128>(value.high) << 64) | value.low;
}

WideRingShare Split(U128 value, int bits)
{
    const U128 mask = (U128{1} << bits) - 1;
    value &= mask;
    return {static_cast<std::uint64_t>(value),
            static_cast<std::uint64_t>(value >> 64)};
}

I128 SignedValue(WideRingShare value, int bits)
{
    const U128 raw = Raw(value) & ((U128{1} << bits) - 1);
    const U128 sign = U128{1} << (bits - 1);
    if ((raw & sign) == 0) return static_cast<I128>(raw);
    return -static_cast<I128>((U128{1} << bits) - raw);
}

} // namespace

Result<void> ValidateWideRingBits(int bits)
{
    if (bits < 1 || bits > 127)
        return Status::Invalid("wide ring bits must be in [1,127]");
    return {};
}

WideRingShare WideFromLow(std::uint64_t value, int bits)
{
    return Split(value, bits);
}

WideRingShare WideAdd(WideRingShare left, WideRingShare right, int bits)
{
    return Split(Raw(left) + Raw(right), bits);
}

WideRingShare WideSubtract(WideRingShare left, WideRingShare right, int bits)
{
    return Split(Raw(left) - Raw(right), bits);
}

WideRingShare WideNegate(WideRingShare value, int bits)
{
    return Split(U128{0} - Raw(value), bits);
}

WideRingShare WideShiftLeft(WideRingShare value, int shift, int bits)
{
    if (shift <= 0) return Split(Raw(value), bits);
    if (shift >= 128) return {};
    return Split(Raw(value) << shift, bits);
}

WideRingShare WideSignedShiftRight(WideRingShare value, int shift, int bits)
{
    if (shift <= 0) return Split(Raw(value), bits);
    I128 signed_value = SignedValue(value, bits);
    if (shift >= bits)
        signed_value = signed_value < 0 ? -1 : 0;
    else
        signed_value >>= shift;
    return Split(static_cast<U128>(signed_value), bits);
}

WideRingShare WideLocalTruncate(WideRingShare value, int shift, int bits,
                                bool negative_share_orientation)
{
    if (!negative_share_orientation)
        return WideSignedShiftRight(value, shift, bits);
    return WideNegate(
        WideSignedShiftRight(WideNegate(value, bits), shift, bits), bits);
}

WideRingShare WideMultiplyAccumulateShift(
    WideRingShare left, std::int64_t left_multiplier,
    WideRingShare right, std::int64_t right_multiplier,
    int shift, int bits)
{
    I128 result = SignedValue(left, bits) * static_cast<I128>(left_multiplier) +
                  SignedValue(right, bits) * static_cast<I128>(right_multiplier);
    if (shift > 0) result >>= shift;
    return Split(static_cast<U128>(result), bits);
}

WideRingShare WideMultiplySignedPublic(WideRingShare value,
                                       std::int64_t multiplier, int bits)
{
    U128 magnitude = multiplier < 0
                         ? static_cast<U128>(-(static_cast<I128>(multiplier)))
                         : static_cast<U128>(multiplier);
    U128 product = Raw(value) * magnitude;
    if (multiplier < 0) product = U128{0} - product;
    return Split(product, bits);
}

std::uint64_t WideModuloPrime(WideRingShare value, int bits,
                              std::uint64_t prime)
{
    if (prime < 2) return 0;
    const U128 mask = (U128{1} << bits) - 1;
    return static_cast<std::uint64_t>((Raw(value) & mask) % prime);
}

std::string WideToHex(WideRingShare value)
{
    std::ostringstream stream;
    stream << "0x" << std::hex << value.high << std::setfill('0')
           << std::setw(16) << value.low;
    return stream.str();
}

} // namespace hera
