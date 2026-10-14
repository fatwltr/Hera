#pragma once

#include <cstddef>

namespace hera::detail {

// Token columns share the input's power-of-two stride, including padding.
inline std::size_t TokenColumnStride(std::size_t tokens) noexcept
{
    std::size_t stride = 1;
    while (stride < tokens) stride *= 2;
    return stride;
}

// Largest group fitting the remaining heads and dividing the slot columns.
// Apply again to each remainder: 12 heads and 128 columns become 8 + 4.
// Q/K producers and the QK consumer must use this same decision.
inline std::size_t QkHeadsPerTile(std::size_t tokens, std::size_t remaining_heads,
                                  std::size_t slots) noexcept
{
    const auto columns = slots / TokenColumnStride(tokens);
    auto heads = remaining_heads < columns ? remaining_heads : columns;
    while (heads != 0 && columns % heads != 0) --heads;
    return heads;
}

// Largest power of two that fits a physical lane capacity; zero stays zero.
inline std::size_t FloorPowerOfTwo(std::size_t capacity) noexcept
{
    if (capacity == 0) return 0;
    std::size_t lanes = 1;
    while (lanes <= capacity / 2) lanes *= 2;
    return lanes;
}

} // namespace hera::detail
