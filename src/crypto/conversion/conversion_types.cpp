#include "hera/conversion.h"

#include <limits>
#include <set>

namespace hera {

PhysicalSlotMap PhysicalSlotMap::DenseComplex(std::size_t logical_elements,
                                              std::size_t slots)
{
    PhysicalSlotMap result;
    result.slots_per_ciphertext = slots;
    result.logical_element_count = logical_elements;
    result.ciphertext_count =
        slots == 0 ? 0 : (logical_elements + 2 * slots - 1) / (2 * slots);
    result.assignments.resize(result.ciphertext_count * slots);
    std::size_t logical = 0;
    for (auto &entry : result.assignments) {
        if (logical < logical_elements) entry.real_logical_index = logical++;
    }
    for (auto &entry : result.assignments) {
        if (logical < logical_elements)
            entry.imaginary_logical_index = logical++;
    }
    return result;
}

PhysicalSlotMap PhysicalSlotMap::DenseReal(std::size_t logical_elements,
                                           std::size_t slots)
{
    PhysicalSlotMap result;
    result.slots_per_ciphertext = slots;
    result.logical_element_count = logical_elements;
    result.ciphertext_count =
        slots == 0 ? 0 : (logical_elements + slots - 1) / slots;
    result.assignments.resize(result.ciphertext_count * slots);
    for (std::size_t i = 0; i < logical_elements; ++i)
        result.assignments[i].real_logical_index = i;
    return result;
}

Result<void> PhysicalSlotMap::Validate() const
{
    if (ciphertext_count == 0 || slots_per_ciphertext == 0)
        return Status::Invalid("physical slot map has an empty dimension");
    if (ciphertext_count >
        std::numeric_limits<std::size_t>::max() / slots_per_ciphertext)
        return Status::Invalid("physical slot map size overflows size_t");
    if (assignments.size() != ciphertext_count * slots_per_ciphertext)
        return Status::Invalid("physical slot map assignment count mismatch");
    std::vector<bool> seen(logical_element_count, false);
    for (const auto &entry : assignments) {
        const std::size_t indices[] = {entry.real_logical_index,
                                       entry.imaginary_logical_index};
        for (const auto index : indices) {
            if (index == kUnusedLogicalIndex) continue;
            if (index >= logical_element_count)
                return Status::Invalid(
                    "physical slot map logical index is out of range");
            if (seen[index])
                return Status::Invalid(
                    "physical slot map contains a duplicate logical index");
            seen[index] = true;
        }
    }
    for (const bool value : seen)
        if (!value)
            return Status::Invalid(
                "physical slot map does not cover every logical element");
    return {};
}

Result<void> MpcShareTensor::Validate() const
{
    const auto count = physical_shape.ElementCount();
    if (!count.ok()) return count.status();
    if (count.value() != local_share.size())
        return Status::Invalid("MPC physical shape/share count mismatch");
    if (numeric.ring_bits < 2 || numeric.ring_bits > 63 ||
        numeric.scale_bits < 0 || numeric.scale_bits >= numeric.ring_bits)
        return Status::Invalid("MPC share has an invalid numeric format");
    if (public_value_bound_bits && (*public_value_bound_bits < 0 ||
        *public_value_bound_bits >= numeric.ring_bits))
        return Status::Invalid("MPC public value bound does not fit its signed ring");
    const std::uint64_t mask =
        (std::uint64_t{1} << numeric.ring_bits) - 1;
    for (const auto value : local_share)
        if ((value & ~mask) != 0)
            return Status::Invalid("MPC share is not reduced to its ring");
    return {};
}

Result<void> ComplexMpcSharePair::Validate() const
{
    const auto real_valid = real.Validate();
    if (!real_valid.ok()) return real_valid.status();
    const auto imag_valid = imag.Validate();
    if (!imag_valid.ok()) return imag_valid.status();
    if (real.component != ComplexComponent::kReal ||
        imag.component != ComplexComponent::kImaginary)
        return Status::Invalid("complex MPC components are mislabeled");
    if (real.physical_shape != imag.physical_shape ||
        real.numeric.ring_bits != imag.numeric.ring_bits ||
        real.numeric.scale_bits != imag.numeric.scale_bits ||
        real.numeric.is_signed != imag.numeric.is_signed ||
        real.public_value_bound_bits != imag.public_value_bound_bits)
        return Status::Invalid("complex MPC component contracts differ");
    return {};
}

} // namespace hera
