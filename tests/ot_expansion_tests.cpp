#include "OT/ot_pack.h"
#undef AES_KEY

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace {

std::uint64_t Next(std::uint64_t &state)
{
    state ^= state << 13; state ^= state >> 7; state ^= state << 17;
    return state;
}

#ifdef __x86_64__
template <int Keys>
void CheckAesExpansion(std::uint64_t &seed)
{
    constexpr int count = (1 << Keys) - 1;
    emp::block actual[count], expected[count];
    emp::AES_KEY keys[Keys];
    for (int trial = 0; trial < 128; ++trial) {
        for (int key = 0; key < Keys; ++key)
            emp::AES_set_encrypt_key(emp::makeBlock(Next(seed), Next(seed)), &keys[key]);
        for (int i = 0; i < count; ++i)
            actual[i] = trial == 0 ? emp::zero_block : trial == 1
                ? emp::makeBlock(UINT64_MAX, UINT64_MAX) : emp::makeBlock(Next(seed), Next(seed));
        std::memcpy(expected, actual, sizeof(actual));
        // Independent AES-ECB implementation and explicit key groups,
        // rather than the optimized index mapping or round expansion.
        int offset = 0;
        for (int key = 0; key < Keys; ++key) {
            emp::AES_ecb_encrypt_blks(expected + offset, 1U << key, &keys[key]);
            offset += 1 << key;
        }
        cheetah::ParaEncExp<Keys>(actual, keys);
        if (std::memcmp(actual, expected, sizeof(actual)))
            throw std::runtime_error("OT AES expansion differs from independent AES-ECB reference");
    }
}
#endif

void CheckPrefixPads(std::uint64_t &seed)
{
    for (int bits = 0; bits <= 8; ++bits) {
        const int choices = 1 << bits;
        emp::block hashes[510], actual[256], expected[256];
        for (int trial = 0; trial < 128; ++trial) {
            for (int i = 0; i < 2 * choices - 2; ++i)
                hashes[i] = emp::makeBlock(Next(seed), Next(seed));
            for (int choice = 0; choice < choices; ++choice) {
                expected[choice] = emp::zero_block;
                // Original independent per-leaf mathematical sum.
                for (int bit = 0; bit < bits; ++bit) {
                    const auto prefix = choice & ((1 << bit) - 1);
                    const auto half = choice & (1 << bit) ? choices - 1 : 0;
                    expected[choice] ^= hashes[(1 << bit) - 1 + prefix + half];
                }
                actual[choice] = emp::makeBlock(UINT64_MAX, UINT64_MAX);
            }
            cheetah::ExpandOtPrefixPads(actual, hashes, bits);
            if (std::memcmp(actual, expected, choices * sizeof(emp::block)))
                throw std::runtime_error("OT prefix pads differ from independent leaf sums");
        }
    }
}

} // namespace

int main()
{
    try {
        std::uint64_t seed = 0x9876abcde1234ULL;
#ifdef __x86_64__
        CheckAesExpansion<1>(seed); CheckAesExpansion<2>(seed);
        CheckAesExpansion<3>(seed); CheckAesExpansion<4>(seed);
        CheckAesExpansion<5>(seed); CheckAesExpansion<6>(seed);
        CheckAesExpansion<7>(seed); CheckAesExpansion<8>(seed);
#endif
        CheckPrefixPads(seed);
        std::cout << "HERA_OT_EXPANSION_STATUS=PASS\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "HERA_OT_EXPANSION_STATUS=FAIL " << error.what() << '\n';
        return 1;
    }
}
