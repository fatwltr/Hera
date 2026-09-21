#include "hera/conversion.h"
#include "hera/wide_ring.h"

#include "fixed_ckks_transform.h"
#include "random_sampling.h"

#include <cstdint>
#include <iostream>
#include <set>
#include <stdexcept>

namespace {

using hera::WideRingShare;
using U128 = unsigned __int128;

U128 Raw(WideRingShare value)
{
    return (static_cast<U128>(value.high) << 64) | value.low;
}

__int128 Signed(WideRingShare value, int bits)
{
    const U128 modulus = U128{1} << bits;
    const U128 raw = Raw(value) & (modulus - 1);
    return (raw & (U128{1} << (bits - 1)))
        ? -static_cast<__int128>(modulus - raw)
        : static_cast<__int128>(raw);
}

void Require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}

} // namespace

int main()
{
    try {
        constexpr int bits = 83;
        Require(!hera::ValidateWideRingBits(128).ok(),
                "128-bit wide ring was accepted");
        const WideRingShare a{0xfffffffffffffff0ULL, 0x12345};
        const WideRingShare b{0x31, 0x2};
        const auto sum = hera::WideAdd(a, b, bits);
        Require(Raw(sum) == ((Raw(a) + Raw(b)) & ((U128{1} << bits) - 1)),
                "wide add failed");
        const auto restored = hera::WideSubtract(sum, b, bits);
        Require(Raw(restored) == (Raw(a) & ((U128{1} << bits) - 1)),
                "wide subtract failed");
        const auto minus_three = hera::WideFromLow((std::uint64_t{1} << 43) - 3,
                                                    43);
        const auto shifted = hera::WideSignedShiftRight(minus_three, 1, 43);
        Require(Signed(shifted, 43) == -2, "signed wide shift failed");

        const auto map = hera::PhysicalSlotMap::DenseComplex(29, 8);
        Require(map.Validate().ok(), "dense complex map is invalid");
        Require(map.ciphertext_count == 2, "dense complex count is wrong");
        std::set<std::size_t> indices;
        for (const auto &entry : map.assignments) {
            if (entry.real_logical_index != hera::kUnusedLogicalIndex)
                indices.insert(entry.real_logical_index);
            if (entry.imaginary_logical_index != hera::kUnusedLogicalIndex)
                indices.insert(entry.imaginary_logical_index);
        }
        Require(indices.size() == 29, "dense complex map lost coordinates");

        auto transform_result =
            hera::conversion_internal::FixedCkksTransform::Create(16, 20, 93);
        Require(transform_result.ok(), "fixed CKKS transform creation failed");
        auto transform = transform_result.take_value();
        hera::conversion_internal::SlotSharePair slots;
        slots.real.resize(8);
        slots.imag.resize(8);
        for (std::size_t i = 0; i < 8; ++i) {
            slots.real[i] = hera::WideFromLow((1000 + 7 * i) * 1024, 93);
            slots.imag[i] = hera::WideNegate(
                hera::WideFromLow((20 + i) * 1024, 93), 93);
        }

        // Verify that BLB's asymmetric local truncation remains linear over
        // independently random additive shares (the case used by MpcToCkks).
        constexpr int share_bits = 43;
        constexpr int extended_bits = 83;
        constexpr int guard = 20;
        auto server_transform =
            hera::conversion_internal::FixedCkksTransform::Create(
                16, 20, extended_bits + guard, false).take_value();
        auto client_transform =
            hera::conversion_internal::FixedCkksTransform::Create(
                16, 20, extended_bits + guard, true).take_value();
        hera::conversion_internal::SlotSharePair server_slots, client_slots;
        server_slots.real.resize(8);
        server_slots.imag.resize(8);
        client_slots.real.resize(8);
        client_slots.imag.resize(8);
        for (std::size_t i = 0; i < 8; ++i) {
            const auto clear_real = hera::WideFromLow(1000 + i, extended_bits);
            const auto clear_imag = hera::WideNegate(
                hera::WideFromLow(30 + i, extended_bits), extended_bits);
            const WideRingShare random_real{
                0x91726354abcdef01ULL,
                static_cast<std::uint64_t>((0x12345 + i) & 0x7ffff)};
            const WideRingShare random_imag{
                0x123456789abcdef0ULL,
                static_cast<std::uint64_t>((0x54321 + i) & 0x7ffff)};
            server_slots.real[i] = hera::WideShiftLeft(random_real, guard,
                                                       extended_bits + guard);
            server_slots.imag[i] = hera::WideShiftLeft(random_imag, guard,
                                                       extended_bits + guard);
            client_slots.real[i] = hera::WideShiftLeft(
                hera::WideSubtract(clear_real, random_real, extended_bits),
                guard, extended_bits + guard);
            client_slots.imag[i] = hera::WideShiftLeft(
                hera::WideSubtract(clear_imag, random_imag, extended_bits),
                guard, extended_bits + guard);
        }
        auto server_coeff = server_transform.Encode(server_slots).take_value();
        auto client_coeff = client_transform.Encode(client_slots).take_value();
        std::vector<WideRingShare> clear_coeff(16);
        for (std::size_t i = 0; i < clear_coeff.size(); ++i) {
            const auto s = hera::WideLocalTruncate(
                server_coeff[i], guard, extended_bits + guard, false);
            const auto c = hera::WideLocalTruncate(
                client_coeff[i], guard, extended_bits + guard, true);
            const std::uint64_t ring = (s.low + c.low) &
                ((std::uint64_t{1} << share_bits) - 1);
            clear_coeff[i] = (ring & (std::uint64_t{1} << (share_bits - 1)))
                ? hera::WideNegate(
                      hera::WideFromLow((std::uint64_t{1} << share_bits) - ring,
                                        extended_bits),
                      extended_bits)
                : hera::WideFromLow(ring, extended_bits);
            clear_coeff[i] = hera::WideShiftLeft(
                clear_coeff[i], guard, extended_bits + guard);
        }
        auto reconstructed = server_transform.Decode(clear_coeff).take_value();
        for (std::size_t i = 0; i < 8; ++i) {
            const auto rr = hera::WideSignedShiftRight(
                reconstructed.real[i], guard, extended_bits + guard);
            const auto ri = hera::WideSignedShiftRight(
                reconstructed.imag[i], guard, extended_bits + guard);
            Require(Signed(rr, extended_bits + guard) >=
                        static_cast<__int128>(997 + i) &&
                    Signed(rr, extended_bits + guard) <=
                        static_cast<__int128>(1003 + i),
                    "shared complex Encode real error is too large");
            Require(Signed(ri, extended_bits + guard) >=
                        -static_cast<__int128>(33 + i) &&
                    Signed(ri, extended_bits + guard) <=
                        -static_cast<__int128>(27 + i),
                    "shared complex Encode imag error is too large");
        }
        auto encoded = transform.Encode(slots);
        Require(encoded.ok(), "complex fixed Encode failed");
        auto decoded = transform.Decode(encoded.value());
        Require(decoded.ok(), "complex fixed Decode failed");
        for (std::size_t i = 0; i < 8; ++i) {
            const auto actual_real = hera::WideSignedShiftRight(
                decoded.value().real[i], 10, 93);
            const auto actual_imag = hera::WideSignedShiftRight(
                decoded.value().imag[i], 10, 93);
            const auto real_error = Signed(actual_real, 93) -
                                    Signed(slots.real[i], 93) / 1024;
            const auto imag_error = Signed(actual_imag, 93) -
                                    Signed(slots.imag[i], 93) / 1024;
            Require(real_error >= -3 && real_error <= 3,
                    "complex real transform error is too large");
            Require(imag_error >= -3 && imag_error <= 3,
                    "complex imaginary transform error is too large");
        }

        auto random = hera::MakeDeterministicRandomSourceForTesting(
            {1, 2, 3, 4});
        auto a_stream = random->OpenStream("hera/c2m/A/test/0");
        auto zr_stream = random->OpenStream("hera/c2m/Z-real/test/0");
        auto zi_stream = random->OpenStream("hera/c2m/Z-imag/test/0");
        Require(a_stream.ok() && zr_stream.ok() && zi_stream.ok(),
                "domain-separated streams were not created");
        std::uint64_t av{}, rv{}, iv{};
        a_stream.value()->Fill(&av, sizeof(av));
        zr_stream.value()->Fill(&rv, sizeof(rv));
        zi_stream.value()->Fill(&iv, sizeof(iv));
        Require(av != rv && av != iv && rv != iv,
                "random domains produced identical first blocks");
        Require(!random->OpenStream("hera/c2m/A/test/0").ok(),
                "random domain reuse was not rejected");

        auto distribution_stream = random->OpenStream("sampling/regression");
        Require(distribution_stream.ok(), "sampling stream creation failed");
        constexpr std::size_t sample_count = 170000;
        auto samples = hera::conversion_internal::SampleUniformPrime(
            *distribution_stream.value(), sample_count, 17);
        std::vector<std::size_t> buckets(17);
        for (const auto value : samples) {
            Require(value < buckets.size(), "prime sample is out of range");
            ++buckets[value];
        }
        const std::size_t expected = sample_count / buckets.size();
        for (const auto count : buckets)
            Require(count > expected * 9 / 10 && count < expected * 11 / 10,
                    "prime rejection sampler uniformity regression failed");

        std::cout << "HERA_CONVERSION_UNIT_STATUS=PASS\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "HERA_CONVERSION_UNIT_STATUS=FAIL: " << error.what()
                  << '\n';
        return 1;
    }
}
