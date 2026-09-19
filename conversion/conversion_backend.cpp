#include "hera/conversion_backend.h"

#include <limits>

namespace hera {

Result<void> CryptoTensorStore::PutCiphertext(
    TensorId id, std::optional<SealCipherBatch> payload)
{
    if (id == 0) return Status::Invalid("ciphertext payload ID is zero");
    if (payload) {
        auto map = payload->slot_map.Validate();
        if (!map.ok()) return map.status();
        if (payload->ciphertexts.size() != payload->slot_map.ciphertext_count)
            return Status::Invalid("ciphertext payload count/map mismatch");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (ciphertexts_.count(id) || mpc_shares_.count(id))
        return Status::FailedPrecondition("payload ID is already registered");
    ciphertexts_.emplace(id, std::move(payload));
    return {};
}

Result<void> CryptoTensorStore::PutMpcShare(TensorId id,
                                            MpcShareTensor payload)
{
    if (id == 0) return Status::Invalid("MPC payload ID is zero");
    auto valid = payload.Validate();
    if (!valid.ok()) return valid.status();
    std::lock_guard<std::mutex> lock(mutex_);
    if (ciphertexts_.count(id) || mpc_shares_.count(id))
        return Status::FailedPrecondition("payload ID is already registered");
    mpc_shares_.emplace(id, std::move(payload));
    return {};
}

Result<std::optional<SealCipherBatch>> CryptoTensorStore::GetCiphertext(
    TensorId id) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = ciphertexts_.find(id);
    if (found == ciphertexts_.end())
        return Status::NotFound("ciphertext payload is not registered");
    return found->second;
}

Result<MpcShareTensor> CryptoTensorStore::GetMpcShare(TensorId id) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = mpc_shares_.find(id);
    if (found == mpc_shares_.end())
        return Status::NotFound("MPC payload is not registered");
    return found->second;
}

Result<ConversionPackingRegistry::Key>
ConversionPackingRegistry::MakeKey(const TensorSpec &spec)
{
    if (!spec.he_parameter_profile)
        return Status::Invalid("HE tensor has no parameter profile");
    return Key{*spec.he_parameter_profile, spec.layout,
               spec.shape.dimensions};
}

Result<void> ConversionPackingRegistry::Register(const TensorSpec &spec,
                                                  PhysicalSlotMap map)
{
    auto key = MakeKey(spec);
    if (!key.ok()) return key.status();
    auto valid = map.Validate();
    if (!valid.ok()) return valid.status();
    auto count = spec.shape.ElementCount();
    if (!count.ok()) return count.status();
    if (map.logical_element_count != count.value())
        return Status::Invalid("registered slot map has wrong logical size");
    const auto inserted = maps_.emplace(key.take_value(), std::move(map));
    if (!inserted.second)
        return Status::FailedPrecondition(
            "a slot map is already registered for this tensor contract");
    return {};
}

Result<PhysicalSlotMap> ConversionPackingRegistry::Resolve(
    const TensorSpec &spec, std::size_t slots) const
{
    auto key = MakeKey(spec);
    if (!key.ok()) return key.status();
    const auto found = maps_.find(key.value());
    if (found != maps_.end()) {
        if (found->second.slots_per_ciphertext != slots)
            return Status::FailedPrecondition(
                "registered slot map uses the wrong CKKS slot count");
        return found->second;
    }
    auto count = spec.shape.ElementCount();
    if (!count.ok()) return count.status();
    return PhysicalSlotMap::DenseComplex(count.value(), slots);
}

SealSciConversionBackend::SealSciConversionBackend(
    PartyRole role, HeContextRegistry &contexts, CkksMpcConverter &converter,
    CryptoTensorStore &store, ConversionPackingRegistry &packing)
    : role_(role), contexts_(contexts), converter_(converter), store_(store),
      packing_(packing)
{}

Result<TensorHandle> SealSciConversionBackend::HeToMpc(
    const TensorHandle &ciphertext, const ConversionAttributes &attributes,
    const TensorSpec &output_spec)
{
    if (ciphertext.spec.domain != Domain::kHeCiphertext ||
        attributes.source_domain != Domain::kHeCiphertext ||
        attributes.target_domain != Domain::kMpcArithmeticShare ||
        output_spec.domain != Domain::kMpcArithmeticShare)
        return Status::Invalid("HeToMpc graph contract has invalid domains");
    if (!ciphertext.spec.he_parameter_profile)
        return Status::Invalid("HeToMpc input has no HE profile");
    const auto &profile = contexts_.Require(
        *ciphertext.spec.he_parameter_profile, role_);
    auto map_result = packing_.Resolve(ciphertext.spec, profile.slot_count());
    if (!map_result.ok()) return map_result.status();
    const auto map = map_result.take_value();

    std::optional<SealCipherBatch> server_input;
    if (role_ == PartyRole::kServer) {
        auto payload = store_.GetCiphertext(ciphertext.id);
        if (!payload.ok()) return payload.status();
        if (!payload.value())
            return Status::FailedPrecondition(
                "server HE handle has no ciphertext payload");
        server_input = payload.take_value();
        if (server_input->slot_map.assignments != map.assignments)
            return Status::FailedPrecondition(
                "ciphertext payload and compiled slot map differ");
    }
    auto converted = converter_.CkksToMpc(server_input);
    if (!converted.ok()) return converted.status();
    auto pair = converted.take_value();
    auto logical_count = output_spec.shape.ElementCount();
    if (!logical_count.ok()) return logical_count.status();
    if (logical_count.value() != map.logical_element_count)
        return Status::FailedPrecondition(
            "HeToMpc output shape and slot map differ");
    std::vector<std::uint64_t> compact(logical_count.value());
    for (std::size_t physical = 0; physical < map.assignments.size(); ++physical) {
        const auto assignment = map.assignments[physical];
        if (assignment.real_logical_index != kUnusedLogicalIndex)
            compact[assignment.real_logical_index] =
                pair.real.local_share[physical];
        if (assignment.imaginary_logical_index != kUnusedLogicalIndex)
            compact[assignment.imaginary_logical_index] =
                pair.imag.local_share[physical];
    }
    const TensorId id = next_id_.fetch_add(1);
    MpcShareTensor logical{std::move(compact), output_spec.shape,
                           output_spec.numeric, ComplexComponent::kReal};
    auto stored = store_.PutMpcShare(id, std::move(logical));
    if (!stored.ok()) return stored.status();
    return TensorHandle{id, output_spec};
}

Result<TensorHandle> SealSciConversionBackend::MpcToHe(
    const TensorHandle &share, const ConversionAttributes &attributes,
    const TensorSpec &output_spec)
{
    if (share.spec.domain != Domain::kMpcArithmeticShare ||
        attributes.source_domain != Domain::kMpcArithmeticShare ||
        attributes.target_domain != Domain::kHeCiphertext ||
        output_spec.domain != Domain::kHeCiphertext)
        return Status::Invalid("MpcToHe graph contract has invalid domains");
    if (!output_spec.he_parameter_profile)
        return Status::Invalid("MpcToHe output has no HE profile");
    auto local = store_.GetMpcShare(share.id);
    if (!local.ok()) return local.status();
    const auto &profile = contexts_.Require(*output_spec.he_parameter_profile,
                                            role_);
    auto map_result = packing_.Resolve(output_spec, profile.slot_count());
    if (!map_result.ok()) return map_result.status();
    auto map = map_result.take_value();

    const std::size_t physical_count =
        map.ciphertext_count * map.slots_per_ciphertext;
    const NumericFormat numeric = local.value().numeric;
    ComplexMpcSharePair pair{
        {std::vector<std::uint64_t>(physical_count),
         TensorShape{{map.ciphertext_count, map.slots_per_ciphertext}}, numeric,
         ComplexComponent::kReal},
        {std::vector<std::uint64_t>(physical_count),
         TensorShape{{map.ciphertext_count, map.slots_per_ciphertext}}, numeric,
         ComplexComponent::kImaginary}};
    for (std::size_t physical = 0; physical < map.assignments.size(); ++physical) {
        const auto assignment = map.assignments[physical];
        if (assignment.real_logical_index != kUnusedLogicalIndex)
            pair.real.local_share[physical] =
                local.value().local_share[assignment.real_logical_index];
        if (assignment.imaginary_logical_index != kUnusedLogicalIndex)
            pair.imag.local_share[physical] =
                local.value().local_share[assignment.imaginary_logical_index];
    }
    const int scale_bits = output_spec.planned_scale_bits.value_or(
        profile.parameters.input_scale_bits);
    MpcToCkksTarget target{*output_spec.he_parameter_profile,
                           output_spec.planned_level, scale_bits,
                           output_spec.shape, output_spec.layout, map};
    auto converted = converter_.MpcToCkks(pair, target);
    if (!converted.ok()) return converted.status();
    const TensorId id = next_id_.fetch_add(1);
    auto stored = store_.PutCiphertext(id, converted.take_value());
    if (!stored.ok()) return stored.status();
    return TensorHandle{id, output_spec};
}

} // namespace hera
