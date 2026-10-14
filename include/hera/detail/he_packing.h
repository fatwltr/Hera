#pragma once

#include "hera/block2_he.h"
#include "hera/conversion.h"
#include <algorithm>

namespace hera::detail {

struct RealConversionGroup {
    std::size_t first{};
    std::size_t second{kUnusedLogicalIndex};
    // A lone real ciphertext uses its lower/upper slot ranges as two lanes.
    std::size_t split{};
    bool real{};
};

struct RealConversionPacking {
    PhysicalSlotMap map;
    std::vector<RealConversionGroup> groups;
    std::vector<int> rotations;
};

// Pure public planning, shared by conversion and minimal rotation-key setup.
Result<RealConversionPacking> PlanRealConversion(const PhysicalSlotMap &map);

inline HePackingPlan PairedRealPacking(const HePackingPlan &real)
{
    auto paired = real;
    paired.ciphertext_count = (real.ciphertext_count + 1) / 2;
    std::size_t tail_extent = 0;
    if (real.ciphertext_count & 1U)
        for (const auto &locations : real.logical_locations)
            if (locations.front().ciphertext == real.ciphertext_count - 1)
                tail_extent = std::max(tail_extent, locations.front().slot + 1);
    for (auto &locations : paired.logical_locations) {
        auto &location = locations.front();
        const bool tail_imaginary = (real.ciphertext_count & 1U) &&
            location.ciphertext == real.ciphertext_count - 1 &&
            location.slot >= (tail_extent + 1) / 2;
        location.component = (location.ciphertext & 1) || tail_imaginary
            ? ComplexLane::kImaginary : ComplexLane::kReal;
        location.ciphertext /= 2;
    }
    return paired;
}

// Exact scale-1 X^(N/2) pairing at the current level. Its imaginary phase
// alternates with slot parity; callers undo it on extracted MPC imag shares.
Result<seal::Plaintext> MakeRealPairPhase(
    const HeContext &context, const seal::parms_id_type &id);
Result<std::vector<seal::Ciphertext>> PairRealCiphertexts(
    std::vector<seal::Ciphertext> input, const HeContext &context,
    std::size_t workers);

inline std::uint64_t UndoPairedRealPhase(std::uint64_t share,
                                        std::size_t slot, int ring_bits)
{
    return (slot & 1) ? (std::uint64_t{0} - share) &
        ((std::uint64_t{1} << ring_bits) - 1) : share;
}

inline Result<PhysicalSlotMap> ToConversionMap(const HePackingPlan &packing)
{
    auto valid = packing.Validate();
    if (!valid.ok()) return valid.status();
    PhysicalSlotMap result;
    result.ciphertext_count = packing.ciphertext_count;
    result.slots_per_ciphertext = packing.slot_count;
    result.logical_element_count = packing.logical_locations.size();
    result.assignments.resize(result.ciphertext_count * result.slots_per_ciphertext);
    for (std::size_t logical = 0; logical < packing.logical_locations.size();
         ++logical) {
        const auto &locations = packing.Locations(logical);
        if (locations.size() != 1)
            return Status::Invalid(
                "conversion packing requires one physical lane per logical value");
        const auto &location = locations.front();
        auto &entry = result.assignments[
            location.ciphertext * packing.slot_count + location.slot];
        auto &target = location.component == ComplexLane::kReal
            ? entry.real_logical_index : entry.imaginary_logical_index;
        if (target != kUnusedLogicalIndex)
            return Status::Invalid("conversion packing lane is not unique");
        target = logical;
    }
    auto map_valid = result.Validate();
    if (!map_valid.ok()) return map_valid.status();
    return result;
}

inline Result<void> RequireSamePacking(const HePackingPlan &left,
                                       const HePackingPlan &right)
{
    if (left.layout != right.layout ||
        left.logical_shape != right.logical_shape ||
        left.slot_count != right.slot_count ||
        left.ciphertext_count != right.ciphertext_count ||
        left.logical_locations.size() != right.logical_locations.size())
        return Status::Invalid("HE packing plans differ");
    for (std::size_t i = 0; i < left.logical_locations.size(); ++i) {
        const auto &a = left.logical_locations[i];
        const auto &b = right.logical_locations[i];
        if (a.size() != b.size()) return Status::Invalid("HE replicas differ");
        for (std::size_t j = 0; j < a.size(); ++j)
            if (a[j].ciphertext != b[j].ciphertext ||
                a[j].slot != b[j].slot ||
                a[j].component != b[j].component ||
                a[j].packed_query_head != b[j].packed_query_head)
                return Status::Invalid("HE physical coordinates differ");
    }
    return {};
}

inline bool PackingMatches(const HePackingPlan &left, const HePackingPlan &right)
{
    return RequireSamePacking(left, right).ok();
}

} // namespace hera::detail
