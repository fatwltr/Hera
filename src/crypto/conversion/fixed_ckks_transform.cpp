#include "fixed_ckks_transform.h"

#include <seal/memorymanager.h>
#include <seal/util/croots.h>
#include <seal/util/uintcore.h>

#include <cmath>
#include <complex>
#include <limits>

namespace hera::conversion_internal {
namespace {

WideComplex Add(WideComplex left, WideComplex right, int bits)
{
    return {WideAdd(left.real, right.real, bits),
            WideAdd(left.imag, right.imag, bits)};
}

WideComplex Subtract(WideComplex left, WideComplex right, int bits)
{
    return {WideSubtract(left.real, right.real, bits),
            WideSubtract(left.imag, right.imag, bits)};
}

WideComplex MultiplyRoot(WideComplex value,
                         std::pair<std::int64_t, std::int64_t> root,
                         int shift, int bits, bool negative_orientation)
{
    WideComplex product{
        WideMultiplyAccumulateShift(value.real, root.first, value.imag,
                                    -root.second, 0, bits),
        WideMultiplyAccumulateShift(value.real, root.second, value.imag,
                                    root.first, 0, bits)};
    return {WideLocalTruncate(product.real, shift, bits,
                              negative_orientation),
            WideLocalTruncate(product.imag, shift, bits,
                              negative_orientation)};
}

WideComplex ShiftRight(WideComplex value, int shift, int bits,
                       bool negative_orientation)
{
    return {WideLocalTruncate(value.real, shift, bits, negative_orientation),
            WideLocalTruncate(value.imag, shift, bits, negative_orientation)};
}

} // namespace

Result<FixedCkksTransform> FixedCkksTransform::Create(
    std::size_t polynomial_degree, int root_scale_bits, int working_bits,
    bool negative_share_truncation)
{
    if (polynomial_degree < 2 ||
        (polynomial_degree & (polynomial_degree - 1)) != 0)
        return Status::Invalid("CKKS transform degree must be a power of two");
    if (root_scale_bits < 1 || root_scale_bits > 50)
        return Status::Invalid("CKKS transform root scale must be in [1,50]");
    const int log_degree = seal::util::get_power_of_two(polynomial_degree);
    if (log_degree > root_scale_bits)
        return Status::Invalid(
            "CKKS transform degree exceeds the fixed-point root precision");
    auto width = ValidateWideRingBits(working_bits);
    if (!width.ok()) return width.status();
    if (working_bits + root_scale_bits + 2 > 127)
        return Status::Invalid(
            "CKKS transform multiply exceeds the signed 127-bit workspace");

    FixedCkksTransform result;
    result.degree_ = polynomial_degree;
    result.root_scale_bits_ = root_scale_bits;
    result.working_bits_ = working_bits;
    result.negative_share_truncation_ = negative_share_truncation;

    const std::size_t slots = polynomial_degree / 2;
    const std::uint64_t cyclotomic_order = 2 * polynomial_degree;
    result.matrix_index_map_.resize(polynomial_degree);
    std::uint64_t position = 1;
    for (std::size_t i = 0; i < slots; ++i) {
        const auto index1 = (position - 1) >> 1;
        const auto index2 = (cyclotomic_order - position - 1) >> 1;
        result.matrix_index_map_[i] = static_cast<std::size_t>(
            seal::util::reverse_bits(index1, log_degree));
        result.matrix_index_map_[slots + i] = static_cast<std::size_t>(
            seal::util::reverse_bits(index2, log_degree));
        position = (position * 3) & (cyclotomic_order - 1);
    }

    seal::util::ComplexRoots roots(cyclotomic_order,
                                   seal::MemoryManager::GetPool());
    result.root_powers_.resize(polynomial_degree);
    result.inverse_root_powers_.resize(polynomial_degree);
    const double root_scale = std::ldexp(1.0, root_scale_bits);
    const auto quantize = [root_scale](std::complex<double> root) {
        return std::make_pair(
            static_cast<std::int64_t>(std::llround(root.real() * root_scale)),
            static_cast<std::int64_t>(std::llround(root.imag() * root_scale)));
    };
    for (std::size_t i = 1; i < polynomial_degree; ++i) {
        result.root_powers_[i] = quantize(roots.get_root(
            static_cast<std::size_t>(seal::util::reverse_bits(i, log_degree))));
        result.inverse_root_powers_[i] = quantize(std::conj(roots.get_root(
            static_cast<std::size_t>(
                seal::util::reverse_bits(i - 1, log_degree) + 1))));
    }
    return result;
}

void FixedCkksTransform::TransformToReverse(
    std::vector<WideComplex> &values) const
{
    std::size_t gap = degree_ >> 1;
    std::size_t root_offset = 0;
    for (std::size_t groups = 1; groups < degree_;
         groups <<= 1, gap >>= 1) {
        for (std::size_t group = 0; group < groups; ++group) {
            const auto root = root_powers_[root_offset + group + 1];
            const std::size_t offset = group * gap * 2;
            for (std::size_t j = 0; j < gap; ++j) {
                const WideComplex left = values[offset + j];
                const WideComplex product = MultiplyRoot(
                    values[offset + gap + j], root, root_scale_bits_,
                    working_bits_, negative_share_truncation_);
                values[offset + j] = Add(left, product, working_bits_);
                values[offset + gap + j] =
                    Subtract(left, product, working_bits_);
            }
        }
        root_offset += groups;
    }
}

void FixedCkksTransform::TransformFromReverse(
    std::vector<WideComplex> &values) const
{
    // Match BLB's inverse-DWT schedule: let the first log2(N)-1 butterfly
    // levels grow in the wide ring and fold the public 1/N normalization into
    // the final level. Normalizing by 1/2 at every level is algebraically
    // equivalent over the reals, but it performs log2(N) probabilistic local
    // truncations and lets their one-bit rounding errors accumulate. BLB only
    // truncates root products at intermediate levels and normalizes once.
    const int log_degree = seal::util::get_power_of_two(degree_);
    std::size_t gap = 1;
    std::size_t root_offset = 0;
    for (std::size_t groups = degree_ >> 1; groups >= 1;
         groups >>= 1, gap <<= 1) {
        for (std::size_t group = 0; group < groups; ++group) {
            const auto root = inverse_root_powers_[root_offset + group + 1];
            const std::size_t offset = group * gap * 2;
            for (std::size_t j = 0; j < gap; ++j) {
                const WideComplex left = values[offset + j];
                const WideComplex right = values[offset + gap + j];
                const WideComplex sum = Add(left, right, working_bits_);
                const WideComplex difference =
                    Subtract(left, right, working_bits_);
                if (groups == 1) {
                    values[offset + j] = ShiftRight(
                        sum, log_degree, working_bits_,
                        negative_share_truncation_);
                    const auto scaled_root = std::make_pair(
                        root.first >> log_degree,
                        root.second >> log_degree);
                    values[offset + gap + j] = MultiplyRoot(
                        difference, scaled_root, root_scale_bits_,
                        working_bits_, negative_share_truncation_);
                } else {
                    values[offset + j] = sum;
                    values[offset + gap + j] = MultiplyRoot(
                        difference, root, root_scale_bits_, working_bits_,
                        negative_share_truncation_);
                }
            }
        }
        root_offset += groups;
        if (groups == 1) break;
    }
}

Result<SlotSharePair> FixedCkksTransform::Decode(
    const std::vector<WideRingShare> &coefficient_shares) const
{
    if (coefficient_shares.size() != degree_)
        return Status::Invalid("Decode requires exactly N coefficient shares");
    std::vector<WideComplex> values(degree_);
    for (std::size_t i = 0; i < degree_; ++i)
        values[i].real = coefficient_shares[i];
    TransformToReverse(values);

    SlotSharePair result;
    result.real.resize(slot_count());
    result.imag.resize(slot_count());
    for (std::size_t i = 0; i < slot_count(); ++i) {
        const auto value = values[matrix_index_map_[i]];
        result.real[i] = value.real;
        result.imag[i] = value.imag;
    }
    return result;
}

Result<std::vector<WideRingShare>> FixedCkksTransform::Encode(
    const SlotSharePair &slot_shares) const
{
    if (slot_shares.real.size() != slot_count() ||
        slot_shares.imag.size() != slot_count())
        return Status::Invalid(
            "Encode requires N/2 real and N/2 imaginary slot shares");
    std::vector<WideComplex> values(degree_);
    for (std::size_t i = 0; i < slot_count(); ++i) {
        values[matrix_index_map_[i]] =
            {slot_shares.real[i], slot_shares.imag[i]};
        values[matrix_index_map_[slot_count() + i]] =
            {slot_shares.real[i],
             WideNegate(slot_shares.imag[i], working_bits_)};
    }
    TransformFromReverse(values);
    std::vector<WideRingShare> result(degree_);
    for (std::size_t i = 0; i < degree_; ++i) result[i] = values[i].real;
    return result;
}

std::optional<int> FixedCkksTransform::EncodedCoefficientBoundBits(
    int slot_bound_bits, int guard_bits, int retained_precision_bits) const
{
    if (working_bits_ < 3 || slot_bound_bits < 0 || guard_bits < 0 ||
        retained_precision_bits < 0 || retained_precision_bits > guard_bits ||
        slot_bound_bits + guard_bits >= 100)
        return std::nullopt;
    using U128 = unsigned __int128;
    // Track a complex Euclidean norm. Quantizing each unit root to S=2^f
    // changes its norm by at most 1/S. Componentwise local truncation adds
    // a reconstructed complex error of norm < 2, under the existing
    // statistical truncation contract. This avoids the loose 3^log(N) bound.
    auto magnitude = (3 * (U128{1} << (slot_bound_bits + guard_bits)) + 1) / 2;
    const auto limit = U128{1} << (working_bits_ - 2);
    const int levels = seal::util::get_power_of_two(degree_);
    const auto root_scale = U128{1} << root_scale_bits_;
    const auto ceil_div = [](U128 value, U128 divisor) {
        return (value + divisor - 1) / divisor;
    };
    for (int level = 1; level < levels; ++level) {
        // Also reserve room for the untruncated root product.
        if (magnitude >= limit / (2 * (root_scale + 1))) return std::nullopt;
        magnitude = 2 * magnitude + ceil_div(2 * magnitude, root_scale) + 2;
    }
    // The final root is arithmetically shifted by log2(N), so its norm
    // is <= S/N + 2. Include this coarser quantization and final rounding.
    if (magnitude >= limit / (2 * (root_scale / degree_ + 2)))
        return std::nullopt;
    magnitude = ceil_div(2 * magnitude, degree_) +
        ceil_div(4 * magnitude, root_scale) + 2;
    const auto divisor = U128{1} << (guard_bits - retained_precision_bits);
    magnitude = ceil_div(magnitude, divisor) + 1;
    int bound_bits = 0;
    while (magnitude != 0) { ++bound_bits; magnitude >>= 1; }
    return bound_bits;
}

std::optional<int> FixedCkksTransform::DecodedSlotBoundBits(
    int coefficient_bound_bits, int guard_bits, int scale_shift) const
{
    if (coefficient_bound_bits < 0 || guard_bits < 0 || scale_shift < 0 ||
        coefficient_bound_bits + guard_bits >= 120 || scale_shift + guard_bits >= 127)
        return std::nullopt;
    using U128 = unsigned __int128;
    auto magnitude = U128{1} << (coefficient_bound_bits + guard_bits);
    const auto scale = U128{1} << root_scale_bits_;
    const auto limit = U128{1} << (working_bits_ - 2);
    const int levels = seal::util::get_power_of_two(degree_);
    for (int level = 0; level < levels; ++level) {
        if (magnitude >= limit / (scale + 1)) return std::nullopt;
        magnitude = 2 * magnitude + (magnitude + scale - 1) / scale + 2;
    }
    const auto divisor = U128{1} << (scale_shift + guard_bits);
    magnitude = (magnitude + divisor - 1) / divisor + 1;
    int bits = 0;
    while (magnitude) { ++bits; magnitude >>= 1; }
    return bits;
}

} // namespace hera::conversion_internal
