#include "hera/detail/he_packing.h"
#include "hera/detail/parallel_ranges.h"

#include <seal/util/ntt.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace hera::detail {

Result<RealConversionPacking> PlanRealConversion(const PhysicalSlotMap &source)
{
    auto valid = source.Validate();
    if (!valid.ok()) return valid.status();
    RealConversionPacking plan;
    plan.map.slots_per_ciphertext = source.slots_per_ciphertext;
    plan.map.logical_element_count = source.logical_element_count;
    const auto slots = source.slots_per_ciphertext;
    std::vector<std::size_t> real;
    for (std::size_t ct = 0; ct < source.ciphertext_count; ++ct) {
        bool complex = false;
        for (std::size_t slot = 0; slot < slots; ++slot)
            complex |= source.assignments[ct * slots + slot].imaginary_logical_index != kUnusedLogicalIndex;
        if (complex) plan.groups.push_back({ct, kUnusedLogicalIndex, 0, false});
        else real.push_back(ct);
    }
    for (std::size_t i = 0; i < real.size(); i += 2) {
        RealConversionGroup group{real[i], kUnusedLogicalIndex, 0, true};
        if (i + 1 < real.size()) group.second = real[i + 1];
        else {
            std::size_t extent = 0;
            for (std::size_t slot = 0; slot < slots; ++slot)
                if (source.assignments[group.first * slots + slot].real_logical_index != kUnusedLogicalIndex)
                    extent = slot + 1;
            if (extent > 1) {
                group.split = 1;
                while (2 * group.split < extent) group.split *= 2;
                const int rotation = group.split == slots / 2
                    ? static_cast<int>(group.split) : -static_cast<int>(group.split);
                plan.rotations.push_back(rotation);
            }
        }
        plan.groups.push_back(group);
    }
    plan.map.ciphertext_count = plan.groups.size();
    plan.map.assignments.resize(plan.groups.size() * slots);
    for (std::size_t g = 0; g < plan.groups.size(); ++g) {
        const auto &group = plan.groups[g];
        for (std::size_t slot = 0; slot < slots; ++slot) {
            const auto &entry = source.assignments[group.first * slots + slot];
            if (!group.real) plan.map.assignments[g * slots + slot] = entry;
            else if (group.split && slot >= group.split)
                plan.map.assignments[g * slots + slot - group.split].imaginary_logical_index = entry.real_logical_index;
            else plan.map.assignments[g * slots + slot].real_logical_index = entry.real_logical_index;
            if (group.second != kUnusedLogicalIndex)
                plan.map.assignments[g * slots + slot].imaginary_logical_index =
                    source.assignments[group.second * slots + slot].real_logical_index;
        }
    }
    valid = plan.map.Validate();
    if (!valid.ok()) return valid.status();
    return plan;
}

Result<seal::Plaintext> MakeRealPairPhase(
    const HeContext &context, const seal::parms_id_type &id)
{
    const auto data = context.context->get_context_data(id);
    if (!data) return Status::Invalid("real boundary phase has an unknown parms_id");
    const auto degree = data->parms().poly_modulus_degree();
    const auto primes = data->parms().coeff_modulus().size();
    seal::Plaintext phase(degree * primes);
    std::fill_n(phase.data(), phase.coeff_count(), std::uint64_t{0});
    for (std::size_t prime = 0; prime < primes; ++prime) {
        phase.data()[prime * degree + degree / 2] = 1;
        seal::util::ntt_negacyclic_harvey(phase.data() + prime * degree,
                                        data->small_ntt_tables()[prime]);
    }
    phase.parms_id() = id;
    phase.scale() = 1.0;
    return phase;
}

Result<std::vector<seal::Ciphertext>> PairRealCiphertexts(
    std::vector<seal::Ciphertext> input, const HeContext &context,
    std::size_t workers)
{
    if (input.empty()) return Status::Invalid("cannot pair an empty real boundary");
    const auto id = input.front().parms_id();
    const auto data = context.context->get_context_data(id);
    if (!data) return Status::Invalid("real boundary has an unknown parms_id");
    for (const auto &ciphertext : input)
        if (ciphertext.parms_id() != id || !ciphertext.is_ntt_form() ||
            !std::isfinite(ciphertext.scale()) || ciphertext.scale() <= 0 ||
            std::abs(ciphertext.scale() / input.front().scale() - 1) > 1e-6)
            return Status::Invalid("paired real ciphertext levels/scales differ");
    const auto degree = data->parms().poly_modulus_degree();
    const auto primes = data->parms().coeff_modulus().size();
    auto phase_result = MakeRealPairPhase(context, id);
    if (!phase_result.ok()) return phase_result.status();
    const auto phase = phase_result.take_value();
    std::vector<seal::Ciphertext> paired((input.size() + 1) / 2);
    const auto status = ParallelRanges(paired.size(), workers, 1,
        [&](std::size_t begin, std::size_t end) -> Result<void> {
            seal::Evaluator evaluator(*context.context);
            for (std::size_t i = begin; i < end; ++i) {
                auto value = std::move(input[2*i]);
                if (2*i+1 < input.size()) {
                    auto imaginary = std::move(input[2*i+1]);
                    evaluator.multiply_plain_inplace(imaginary, phase);
                    evaluator.add_inplace(value, imaginary);
                } else {
                    auto imaginary = value;
                    evaluator.multiply_plain_inplace(imaginary, phase);
                    evaluator.add_inplace(value, imaginary);
                }
                paired[i] = std::move(value);
            }
            return {};
        }, EstimatedHeWorkerBytes(degree, primes, 4));
    if (!status.ok()) return status.status();
    return paired;
}

} // namespace hera::detail
