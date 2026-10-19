#include "blb_ring_protocol.h"
#include "bounded_carry.h"

#include "bounded_ot.h"
#include "utils/net_io_channel.h"

#include <seal/util/common.h>
#include <seal/util/ntt.h>
#include <seal/util/uintarithsmallmod.h>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace hera::conversion_internal::blb {
namespace {
std::uint64_t Mask(int bits)
{
    return bits == 64 ? UINT64_MAX : (std::uint64_t{1} << bits) - 1;
}
void CheckCount(std::size_t n)
{
    if (n > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("BLB ring batch exceeds SCI integer count");
}
} // namespace

RingProtocol::RingProtocol(int party, ConversionOt &ot)
    : party_(party), ot_(ot)
{
    if ((party != sci::ALICE && party != sci::BOB) ||
        !ot.ferret || !ot.ferret->io)
        throw std::invalid_argument("invalid BLB ring protocol party/session");
}

std::vector<WideRingShare> RingProtocol::FieldToRing(
    const std::vector<std::uint64_t> &local, std::uint64_t q,
    int output_bits)
{
    CheckCount(local.size());
    if (q < 5 || !(q & 1U) || q > (UINT64_MAX >> 1))
        throw std::invalid_argument("invalid BLB source field modulus");
    const int field_bits = seal::util::get_significant_bit_count(q);
    if (output_bits < field_bits || output_bits > 127)
        throw std::invalid_argument("invalid BLB field-to-ring width");
    if (local.empty()) return {};
    return BoundedCenteredPrimeToRing(party_, ot_, local, q, output_bits);
}

std::vector<WideRingShare> RingProtocol::ExtendSigned(
    const std::vector<std::uint64_t> &local, int input_bits,
    int output_bits)
{
    CheckCount(local.size());
    if (input_bits < 2 || input_bits > 63 || output_bits <= input_bits ||
        output_bits > 127 || output_bits - input_bits > 63)
        throw std::invalid_argument("invalid BLB signed-extension widths");
    if (local.empty()) return {};
    const auto n = local.size();
    const auto mask = Mask(input_bits);
    const auto center = std::uint64_t{1} << (input_bits - 2);
    const int extension_bits = output_bits - input_bits;
    std::vector<std::uint64_t> adjusted(n), arithmetic(n);
    for (std::size_t i = 0; i < n; ++i)
        adjusted[i] = (local[i] + (party_ == sci::ALICE ? center : 0)) & mask;
    std::vector<std::uint8_t> high(n);
    for (std::size_t i = 0; i < n; ++i) high[i] = adjusted[i] >> (input_bits - 1);
    auto carry = LocalBitOrToArithmetic(party_, ot_, high, extension_bits);
    for (std::size_t i = 0; i < n; ++i) arithmetic[i] = carry[i].low;
    std::vector<WideRingShare> result(n);
    for (std::size_t i = 0; i < n; ++i) {
        auto correction = WideShiftLeft(WideFromLow(arithmetic[i], extension_bits), input_bits, output_bits);
        result[i] = WideSubtract(WideFromLow(adjusted[i], output_bits), correction, output_bits);
        if (party_ == sci::ALICE)
            result[i] = WideSubtract(result[i], WideFromLow(center, output_bits), output_bits);
    }
    return result;
}

std::vector<std::uint64_t> RnsWrapCorrection(
    int bits, const seal::SEALContext::ContextData &target)
{
    if (bits < 1 || bits > 127) throw std::invalid_argument("invalid BLB RNS ring width");
    const auto &moduli = target.parms().coeff_modulus();
    std::vector<std::uint64_t> result(moduli.size());
    const std::uint64_t power[2] = {bits < 64 ? std::uint64_t{1} << bits : 0,
        bits >= 64 ? std::uint64_t{1} << (bits - 64) : 0};
    for (std::size_t j = 0; j < moduli.size(); ++j)
        result[j] = seal::util::barrett_reduce_128(power, moduli[j]);
    return result;
}

seal::Plaintext ProjectExtendedRingToRns(
    const std::vector<WideRingShare> &coefficients, int bits,
    const seal::SEALContext::ContextData &target,
    const std::vector<std::uint64_t> &wrap_correction, bool subtract_wrap)
{
    const auto &parms = target.parms();
    const auto &moduli = parms.coeff_modulus();
    const auto degree = parms.poly_modulus_degree();
    if (bits < 1 || bits > 127 || coefficients.size() != degree ||
        moduli.empty() || wrap_correction.size() != moduli.size())
        throw std::invalid_argument("invalid BLB extended-ring RNS contract");
    seal::Plaintext plain(degree * moduli.size());
    for (std::size_t j = 0; j < moduli.size(); ++j) {
        for (std::size_t i = 0; i < degree; ++i) {
            const std::uint64_t limbs[2] = {
                bits < 64 ? coefficients[i].low & Mask(bits) : coefficients[i].low,
                bits <= 64 ? 0 : coefficients[i].high & Mask(bits - 64)};
            auto value = seal::util::barrett_reduce_128(limbs, moduli[j]);
            if (subtract_wrap) value = seal::util::sub_uint_mod(value, wrap_correction[j], moduli[j]);
            plain[j * degree + i] = value;
        }
        seal::util::ntt_negacyclic_harvey(plain.data() + j * degree, target.small_ntt_tables()[j]);
    }
    return plain;
}
} // namespace hera::conversion_internal::blb
