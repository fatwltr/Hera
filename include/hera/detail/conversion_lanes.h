#pragma once

#include "hera/conversion.h"
#include <algorithm>

namespace hera::detail {

struct SlotExtensionPlan {
    bool identity{true};
    std::vector<std::size_t> representatives;
    std::vector<std::size_t> inverse;
    std::size_t padding{};
};

// Public planning only. No decisions depend on local/secret values.
inline SlotExtensionPlan PlanSlotExtensions(const MpcToCkksTarget &target,
    std::size_t base, std::size_t count, bool canonical_padding)
{
    const auto slots = target.slot_map.slots_per_ciphertext;
    const std::size_t planes = target.public_zero_imaginary ? 1 : 2;
    const bool sparse = canonical_padding || !target.public_slot_periods.empty();
    SlotExtensionPlan result;
    for (std::size_t plane = 0; plane < planes && result.identity; ++plane)
        for (std::size_t part = 0; part < count && result.identity; ++part) {
            const auto ct = base + part;
            const auto period = target.public_slot_periods.empty() ? 0 : target.public_slot_periods[ct][plane];
            if (period && period < slots) { result.identity = false; break; }
            if (sparse)
                for (std::size_t slot = 0; slot < slots; ++slot) {
                    const auto &entry = target.slot_map.assignments[ct * slots + slot];
                    if ((plane ? entry.imaginary_logical_index : entry.real_logical_index) == kUnusedLogicalIndex) {
                        result.identity = false; break;
                    }
                }
        }
    if (result.identity) return result;
    result.inverse.assign(planes * count * slots, kUnusedLogicalIndex);
    result.representatives.reserve(result.inverse.size());
    for (std::size_t plane = 0; plane < planes; ++plane)
        for (std::size_t part = 0; part < count; ++part) {
            const auto ct = base + part;
            const auto period = target.public_slot_periods.empty() ? 0 : target.public_slot_periods[ct][plane];
            std::vector<std::size_t> first(period && period < slots ? period : 0, kUnusedLogicalIndex);
            for (std::size_t slot = 0; slot < slots; ++slot) {
                const auto index = plane * count * slots + part * slots + slot;
                const auto &entry = target.slot_map.assignments[ct * slots + slot];
                const auto logical = plane ? entry.imaginary_logical_index : entry.real_logical_index;
                if (sparse && logical == kUnusedLogicalIndex) { ++result.padding; continue; }
                if (!first.empty() && first[slot % period] != kUnusedLogicalIndex)
                    result.inverse[index] = first[slot % period];
                else {
                    result.inverse[index] = result.representatives.size();
                    if (!first.empty()) first[slot % period] = result.inverse[index];
                    result.representatives.push_back(index);
                }
            }
        }
    return result;
}

} // namespace hera::detail
