#include "hera/block5_bridge.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>

namespace hera {
namespace {

Result<std::size_t> LevelOf(const HeContext &context,
                            seal::parms_id_type id)
{
    std::size_t level = 0;
    for (auto data = context.context->first_context_data(); data;
         data = data->next_context_data(), ++level)
        if (data->parms_id() == id) return level;
    return Status::Invalid("ciphertext is outside its declared HE profile");
}

bool CloseScale(double left, double right)
{
    return std::isfinite(left) && std::isfinite(right) && left > 0.0 &&
           right > 0.0 &&
           std::abs(left - right) / std::max(left, right) < 1.0e-3;
}

bool SamePacking(const HePackingPlan &left, const HePackingPlan &right)
{
    if (left.layout != right.layout ||
        left.logical_shape != right.logical_shape ||
        left.slot_count != right.slot_count ||
        left.ciphertext_count != right.ciphertext_count ||
        left.logical_locations.size() != right.logical_locations.size())
        return false;
    for (std::size_t i = 0; i < left.logical_locations.size(); ++i) {
        if (left.logical_locations[i].size() !=
            right.logical_locations[i].size()) return false;
        for (std::size_t j = 0; j < left.logical_locations[i].size(); ++j) {
            const auto &a = left.logical_locations[i][j];
            const auto &b = right.logical_locations[i][j];
            if (a.ciphertext != b.ciphertext || a.slot != b.slot ||
                a.component != b.component) return false;
        }
    }
    return true;
}

const PlainParameterHandle *FindPlain(const LayerWeightHandles &weights,
                                      const std::string &name)
{
    for (const auto &parameter : weights.plain_parameters)
        if (parameter.spec.name == name) return &parameter;
    return nullptr;
}

Result<PhysicalSlotMap> ToConversionMap(const HePackingPlan &packing)
{
    auto valid = packing.Validate();
    if (!valid.ok()) return valid.status();
    PhysicalSlotMap map;
    map.ciphertext_count = packing.ciphertext_count;
    map.slots_per_ciphertext = packing.slot_count;
    map.logical_element_count = packing.logical_locations.size();
    map.assignments.resize(map.ciphertext_count * map.slots_per_ciphertext);
    for (std::size_t logical = 0;
         logical < packing.logical_locations.size(); ++logical) {
        const auto &locations = packing.Locations(logical);
        if (locations.size() != 1)
            return Status::Invalid(
                "conversion requires exactly one lane per logical value");
        const auto &location = locations.front();
        auto &assignment = map.assignments[
            location.ciphertext * map.slots_per_ciphertext + location.slot];
        auto &index = location.component == ComplexLane::kReal
            ? assignment.real_logical_index
            : assignment.imaginary_logical_index;
        if (index != kUnusedLogicalIndex)
            return Status::Invalid("conversion packing lane collision");
        index = logical;
    }
    auto map_valid = map.Validate();
    if (!map_valid.ok()) return map_valid.status();
    return map;
}

Result<ComplexMpcSharePair> Scatter(const MpcShareTensor &logical,
                                    const HePackingPlan &packing)
{
    auto valid = logical.Validate();
    if (!valid.ok()) return valid.status();
    if (logical.local_share.size() != packing.logical_locations.size())
        return Status::Invalid("share and HE packing element counts differ");
    const auto count = packing.ciphertext_count * packing.slot_count;
    std::vector<std::uint64_t> real(count), imag(count);
    for (std::size_t logical_index = 0;
         logical_index < logical.local_share.size(); ++logical_index) {
        const auto &location = packing.Locations(logical_index).front();
        auto &plane = location.component == ComplexLane::kReal ? real : imag;
        plane[location.ciphertext * packing.slot_count + location.slot] =
            logical.local_share[logical_index];
    }
    const TensorShape physical{{packing.ciphertext_count,
                                packing.slot_count}};
    return ComplexMpcSharePair{
        {std::move(real), physical, logical.numeric, ComplexComponent::kReal},
        {std::move(imag), physical, logical.numeric,
         ComplexComponent::kImaginary}};
}

Result<MpcShareTensor> Compact(const ComplexMpcSharePair &physical,
                               const HePackingPlan &packing,
                               const TensorShape &shape,
                               bool is_signed)
{
    auto valid = physical.Validate();
    if (!valid.ok()) return valid.status();
    std::vector<std::uint64_t> values(packing.logical_locations.size());
    for (std::size_t logical = 0; logical < values.size(); ++logical) {
        const auto &location = packing.Locations(logical).front();
        const auto index = location.ciphertext * packing.slot_count +
                           location.slot;
        values[logical] = location.component == ComplexLane::kReal
            ? physical.real.local_share[index]
            : physical.imag.local_share[index];
    }
    auto numeric = physical.real.numeric;
    numeric.is_signed = is_signed;
    return MpcShareTensor{std::move(values), shape, numeric,
                          ComplexComponent::kReal};
}

HePackingPlan VariancePlan(std::size_t tokens, std::size_t slots)
{
    HePackingPlan plan;
    plan.layout = Layout::kReduction;
    plan.logical_shape = {{tokens, 1}};
    plan.slot_count = slots;
    plan.ciphertext_count = 1;
    plan.logical_locations.resize(tokens);
    for (std::size_t token = 0; token < tokens; ++token)
        plan.logical_locations[token].push_back(
            {0, token, ComplexLane::kReal, std::nullopt});
    return plan;
}

TensorSpec HeSpec(const TensorShape &shape, Layout layout,
                  HeRegionId region, const HeContext &context,
                  std::size_t level, int scale_bits,
                  const NumericFormat &numeric)
{
    TensorSpec spec;
    spec.shape = shape;
    spec.domain = Domain::kHeCiphertext;
    spec.layout = layout;
    spec.numeric = numeric;
    spec.he_region = region;
    spec.he_parameter_profile = context.profile;
    spec.planned_level = level;
    spec.planned_scale_bits = scale_bits;
    return spec;
}

void AccumulateConversion(const ConversionStats &before,
                          const ConversionStats &after,
                          Block5BridgeExecutionReport *report,
                          bool mpc_to_ckks)
{
    if (!report) return;
    if (mpc_to_ckks) ++report->mpc_to_ckks_calls;
    else ++report->ckks_to_mpc_calls;
    report->conversion_communication_bytes +=
        after.communication_bytes - before.communication_bytes;
    report->conversion_rounds += after.rounds - before.rounds;
    report->conversion_milliseconds += after.milliseconds - before.milliseconds;
}

} // namespace

SealSciBlock5BridgeExecutor::SealSciBlock5BridgeExecutor(
    const HeraConfig &config, std::size_t current_layer_index,
    HeContextRegistry &contexts, SealCipherTensorStore &he_tensors,
    SealPlainParameterStore &parameters, CryptoTensorStore &mpc_tensors,
    CkksMpcConverter &converter, MpcLayerNormRuntime &mpc_layer_norm,
    TensorIdAllocator *ids)
    : config_(config), current_layer_index_(current_layer_index),
      contexts_(contexts), he_tensors_(he_tensors), parameters_(parameters),
      mpc_tensors_(mpc_tensors), converter_(converter),
      mpc_layer_norm_(mpc_layer_norm), ids_(ids)
{}

Result<std::map<std::string, std::vector<int>>>
SealSciBlock5BridgeExecutor::RequiredRotationSteps(
    const HeraConfig &config, std::size_t current_layer_index)
{
    if (current_layer_index >= config.layers.size())
        return Status::Invalid("Block-5 layer index is out of range");
    std::map<std::string, std::set<int>> collected;
    const auto add_projection = [&](const TransformerLayerConfig &layer,
                                    const std::string &profile,
                                    std::size_t slots) -> Result<void> {
        auto input = FfnGeluPackingPlanner::ComplexColumns(
            {{layer.tokens, layer.ffn_width}}, slots, Layout::kFfn);
        if (!input.ok()) return input.status();
        auto output = FfnGeluPackingPlanner::ComplexColumns(
            {{layer.tokens, layer.model_width}}, slots,
            Layout::kComplexStrictColumn);
        if (!output.ok()) return output.status();
        auto steps = ComplexColumnProjectionKernel::RequiredRotationSteps(
            input.value(), output.value());
        if (!steps.ok()) return steps.status();
        collected[profile].insert(steps.value().begin(), steps.value().end());
        collected[profile].insert(0); // conjugation key
        return {};
    };
    const auto &layer = config.layers[current_layer_index];
    const auto *execution = FindLayerExecution(config, layer.id);
    if (!execution) return Status::Invalid("Block-5 layer placement is absent");
    const auto shared = execution->regions.find(HeRegionId::kRegion4);
    if (shared == execution->regions.end())
        return Status::Invalid("Block-5 shared residual region is absent");
    const auto profile_it = config.he_parameter_profiles.find(
        shared->second.parameter_profile);
    if (profile_it == config.he_parameter_profiles.end())
        return Status::Invalid("Block-5 HE profile is absent");
    auto added = add_projection(layer, shared->second.parameter_profile,
        profile_it->second.polynomial_degree / 2);
    if (!added.ok()) return added.status();

    if (current_layer_index + 1 < config.layers.size()) {
        const auto &next = config.layers[current_layer_index + 1];
        if (next.norm != NormPlacement::kPreNorm)
            return Status::KernelUnavailable(
                "automatic cross-layer bridge supports Pre-LN only");
        const auto *next_execution = FindLayerExecution(config, next.id);
        if (!next_execution)
            return Status::Invalid("next-layer placement is absent");
        const auto region1 = next_execution->regions.find(HeRegionId::kRegion1);
        const auto region2 = next_execution->regions.find(HeRegionId::kRegion2);
        if (region1 == next_execution->regions.end() ||
            region2 == next_execution->regions.end())
            return Status::Invalid("next-layer Region 1/2 is absent");
        const auto r1 = config.he_parameter_profiles.find(
            region1->second.parameter_profile);
        const auto r2 = config.he_parameter_profiles.find(
            region2->second.parameter_profile);
        if (r1 == config.he_parameter_profiles.end() ||
            r2 == config.he_parameter_profiles.end())
            return Status::Invalid("next-layer HE profile is absent");
        const auto stride = [&] {
            std::size_t value = 1;
            while (value < next.tokens) value <<= 1U;
            return value;
        }();
        const auto columns = r1->second.polynomial_degree / 2 / stride;
        for (std::size_t step = 1; step < columns; step <<= 1U)
            collected[region1->second.parameter_profile].insert(
                static_cast<int>(step * stride));
        auto block2 = SealBlock2Executor::RequiredRotationSteps(
            next, r2->second);
        if (!block2.ok()) return block2.status();
        collected[region2->second.parameter_profile].insert(
            block2.value().begin(), block2.value().end());
    }
    std::map<std::string, std::vector<int>> result;
    for (auto &[profile, steps] : collected)
        result.emplace(profile,
                       std::vector<int>(steps.begin(), steps.end()));
    return result;
}

Result<Block5BridgeOutputs> SealSciBlock5BridgeExecutor::Evaluate(
    const Block5BridgeInputs &inputs,
    const LayerWeightHandles &current_weights,
    const std::optional<LayerWeightHandles> &next_weights,
    Block5BridgeExecutionReport *report)
{
    try {
        if (report) *report = {};
        if (current_layer_index_ >= config_.layers.size())
            return Status::Invalid("Block-5 layer index is out of range");
        const auto &layer = config_.layers[current_layer_index_];
        if (report) {
            auto rotations = RequiredRotationSteps(config_,
                                                   current_layer_index_);
            if (!rotations.ok()) return rotations.status();
            for (const auto &[profile, steps] : rotations.value())
                report->required_rotation_steps.insert(
                    report->required_rotation_steps.end(),
                    steps.begin(), steps.end());
            std::sort(report->required_rotation_steps.begin(),
                      report->required_rotation_steps.end());
            report->required_rotation_steps.erase(
                std::unique(report->required_rotation_steps.begin(),
                            report->required_rotation_steps.end()),
                report->required_rotation_steps.end());
        }
        if (layer.norm != NormPlacement::kPreNorm)
            return Status::KernelUnavailable(
                "Block-5 bridge supports Pre-LN only");
        if (current_weights.layer_id != layer.id)
            return Status::Invalid("Wd weight layer ID mismatch");
        if (!inputs.activation_shares.valid() ||
            inputs.activation_shares.spec.domain !=
                Domain::kMpcArithmeticShare ||
            inputs.activation_shares.spec.shape !=
                TensorShape{{layer.tokens, layer.ffn_width}})
            return Status::Invalid("GeLU activation share contract mismatch");
        if (layer.ffn_residual != inputs.ffn_residual.has_value())
            return Status::Invalid(
                "FFN residual presence does not match layer configuration");

        const auto *execution = FindLayerExecution(config_, layer.id);
        if (!execution) return Status::Invalid("layer HE placement is absent");
        const auto shared_region = execution->regions.find(HeRegionId::kRegion4);
        if (shared_region == execution->regions.end())
            return Status::Invalid("shared residual HE region is absent");
        const auto &shared = contexts_.Require(
            shared_region->second.parameter_profile, contexts_.role());
        if (shared.galois_keys.size() == 0)
            return Status::FailedPrecondition("Block-5 Galois keys are absent");

        auto activation = mpc_tensors_.GetMpcShare(
            inputs.activation_shares.id);
        if (!activation.ok()) return activation.status();
        auto activation_plan = FfnGeluPackingPlanner::ComplexColumns(
            {{layer.tokens, layer.ffn_width}}, shared.slot_count(),
            Layout::kFfn);
        if (!activation_plan.ok()) return activation_plan.status();
        auto output_plan = FfnGeluPackingPlanner::ComplexColumns(
            {{layer.tokens, layer.model_width}}, shared.slot_count(),
            Layout::kComplexStrictColumn);
        if (!output_plan.ok()) return output_plan.status();
        auto real_plan = FfnGeluPackingPlanner::RealColumns(
            {{layer.tokens, layer.model_width}}, shared.slot_count(),
            Layout::kStrictColumn);
        if (!real_plan.ok()) return real_plan.status();

        std::size_t residual_level = 0;
        std::optional<SealCipherTensor> residual;
        if (inputs.ffn_residual) {
            if (inputs.ffn_residual->spec.he_parameter_profile !=
                    shared.profile ||
                inputs.ffn_residual->spec.layout !=
                    Layout::kComplexStrictColumn ||
                inputs.ffn_residual->spec.shape !=
                    TensorShape{{layer.tokens, layer.model_width}} ||
                !inputs.ffn_residual->spec.planned_level)
                return Status::Invalid("retained FFN residual contract mismatch");
            residual_level = *inputs.ffn_residual->spec.planned_level;
            if (residual_level == 0)
                return Status::FailedPrecondition(
                    "FFN residual has no predecessor level for Wd");
            if (contexts_.role() == PartyRole::kServer) {
                auto loaded = he_tensors_.Get(inputs.ffn_residual->id);
                if (!loaded.ok()) return loaded.status();
                residual = loaded.take_value();
                auto same = residual->Validate(shared);
                if (!same.ok()) return same.status();
                if (!SamePacking(residual->physical_packing,
                                 output_plan.value()))
                    return Status::Invalid(
                        "retained residual physical packing differs from Wd");
                auto actual = LevelOf(shared,
                    residual->ciphertexts.front().parms_id());
                if (!actual.ok()) return actual.status();
                if (actual.value() != residual_level)
                    return Status::FailedPrecondition(
                        "residual metadata does not match its parms_id");
            }
        } else {
            // Without a residual, use the conventional level-6 Wd output.
            residual_level = 6;
        }

        const auto terminal_level =
            shared.parameters.coefficient_modulus_bits.size() -
            shared.parameters.terminal_prime_count;
        const bool has_next_layer =
            current_layer_index_ + 1 < config_.layers.size();
        // A cross-layer hidden state is immediately converted by the next
        // Pre-LN entry. Place Wd one level before terminal so accurate
        // realification lands exactly on the conversion boundary. The final
        // layer keeps the earlier output level because it is returned as HE.
        const auto wd_output_level = has_next_layer
            ? terminal_level - 1 : residual_level;
        if (wd_output_level == 0 || wd_output_level >= terminal_level)
            return Status::FailedPrecondition(
                "shared profile has no Wd/realify suffix");
        if (residual_level > wd_output_level)
            return Status::FailedPrecondition(
                "FFN residual is already below the planned Wd output");

        auto activation_pair = Scatter(activation.value(),
                                       activation_plan.value());
        if (!activation_pair.ok()) return activation_pair.status();
        auto activation_map = ToConversionMap(activation_plan.value());
        if (!activation_map.ok()) return activation_map.status();
        auto before = converter_.stats();
        auto activation_he = converter_.MpcToCkks(
            activation_pair.value(),
            {shared.profile, wd_output_level - 1,
             shared.parameters.default_scale_bits,
             {{layer.tokens, layer.ffn_width}}, Layout::kFfn,
             activation_map.take_value()});
        if (!activation_he.ok()) return activation_he.status();
        AccumulateConversion(before, converter_.stats(), report, true);

        const auto allocate_id = [&]() {
            return ids_ ? ids_->Next() : next_id_.fetch_add(1);
        };
        const auto hidden_id = allocate_id();
        const auto hidden_level = wd_output_level + 1;
        const auto hidden_spec = HeSpec(
            {{layer.tokens, layer.model_width}}, Layout::kStrictColumn,
            HeRegionId::kRegion4, shared, hidden_level,
            shared.parameters.default_scale_bits, config_.mpc_numeric);
        if (contexts_.role() == PartyRole::kServer) {
            if (!activation_he.value())
                return Status::Internal("server did not receive GeLU ciphertext");
            SealCipherTensor encoded;
            encoded.parameter_profile = shared.profile;
            encoded.ciphertexts = activation_he.value()->ciphertexts;
            encoded.logical_spec = HeSpec(
                {{layer.tokens, layer.ffn_width}}, Layout::kFfn,
                HeRegionId::kRegion4, shared, wd_output_level - 1,
                shared.parameters.default_scale_bits, config_.mpc_numeric);
            encoded.physical_packing = activation_plan.value();

            if (!current_weights.wd.valid() ||
                current_weights.wd.spec.rows != layer.ffn_width ||
                current_weights.wd.spec.columns != layer.model_width)
                return Status::Invalid("Wd handle has the wrong shape");
            auto wd = parameters_.GetMatrix(current_weights.wd.id);
            if (!wd.ok()) return wd.status();
            std::optional<DenseVectorParameter> bd;
            if (layer.ffn_bias) {
                const auto *handle = FindPlain(current_weights, "bd");
                if (!handle || handle->spec.length != layer.model_width)
                    return Status::Invalid("bd handle has the wrong shape");
                auto loaded = parameters_.GetVector(handle->id);
                if (!loaded.ok()) return loaded.status();
                bd = loaded.take_value();
            }
            auto down = ComplexColumnProjectionKernel::Evaluate(
                encoded, wd.value(), bd, output_plan.value(),
                HeRegionId::kRegion4,
                shared.parameters.default_scale_bits, shared,
                report ? &report->he : nullptr);
            if (!down.ok()) return down.status();
            if (layer.ffn_residual) {
                auto packing = down.value().physical_packing.Validate();
                if (!packing.ok()) return packing.status();
                if (down.value().ciphertexts.size() !=
                    residual->ciphertexts.size())
                    return Status::Invalid("Wd/residual ciphertext counts differ");
                seal::Evaluator evaluator(*shared.context);
                for (std::size_t i = 0; i < down.value().ciphertexts.size(); ++i) {
                    auto aligned_residual = residual->ciphertexts[i];
                    if (aligned_residual.parms_id() !=
                        down.value().ciphertexts[i].parms_id())
                        evaluator.mod_switch_to_inplace(
                            aligned_residual,
                            down.value().ciphertexts[i].parms_id());
                    if (!CloseScale(down.value().ciphertexts[i].scale(),
                                    aligned_residual.scale()))
                        return Status::FailedPrecondition(
                            "Wd and residual require compatible scales");
                    evaluator.add_inplace(down.value().ciphertexts[i],
                                          aligned_residual);
                }
                if (report) ++report->direct_residual_additions;
            }
            auto hidden = ComplexColumnRealifyKernel::Evaluate(
                down.value(), real_plan.value(),
                shared.parameters.default_scale_bits, shared,
                report ? &report->he : nullptr);
            if (!hidden.ok()) return hidden.status();
            auto stored = he_tensors_.Put(hidden_id, hidden.take_value());
            if (!stored.ok()) return stored.status();
        }

        Block5BridgeOutputs outputs{TensorHandle{hidden_id, hidden_spec},
                                    std::nullopt};
        if (current_layer_index_ + 1 == config_.layers.size()) {
            if (next_weights)
                return Status::Invalid("last layer received unexpected next weights");
            return outputs;
        }
        // The graph-wide orchestrator closes one layer at a time and lets the
        // reusable Pre-LN entry stage refresh the next residual. Keep the old
        // cross-layer path only when next-layer weights are explicitly passed.
        if (!next_weights)
            return outputs;
        const auto &next = config_.layers[current_layer_index_ + 1];
        if (next.norm != NormPlacement::kPreNorm)
            return Status::KernelUnavailable(
                "automatic cross-layer bridge supports Pre-LN only");
        if (next.tokens != layer.tokens || next.model_width != layer.model_width)
            return Status::Invalid("adjacent hidden-state shapes differ");
        const auto *next_execution = FindLayerExecution(config_, next.id);
        if (!next_execution)
            return Status::Invalid("next-layer placement is absent");
        const auto region1_it = next_execution->regions.find(HeRegionId::kRegion1);
        const auto region2_it = next_execution->regions.find(HeRegionId::kRegion2);
        if (region1_it == next_execution->regions.end() ||
            region2_it == next_execution->regions.end())
            return Status::Invalid("next-layer Region 1/2 is absent");
        const auto &region1 = contexts_.Require(
            region1_it->second.parameter_profile, contexts_.role());
        const auto &region2 = contexts_.Require(
            region2_it->second.parameter_profile, contexts_.role());

        std::optional<SealCipherBatch> hidden_batch;
        if (contexts_.role() == PartyRole::kServer) {
            auto hidden = he_tensors_.Get(hidden_id);
            if (!hidden.ok()) return hidden.status();
            auto map = ToConversionMap(hidden.value().physical_packing);
            if (!map.ok()) return map.status();
            hidden_batch = SealCipherBatch{
                shared.profile, hidden.value().ciphertexts,
                hidden_spec.shape, hidden_spec.layout, map.take_value()};
        }
        before = converter_.stats();
        auto hidden_physical = converter_.CkksToMpc(hidden_batch);
        if (!hidden_physical.ok()) return hidden_physical.status();
        AccumulateConversion(before, converter_.stats(), report, false);
        auto hidden_shares = Compact(hidden_physical.value(), real_plan.value(),
                                     hidden_spec.shape, true);
        if (!hidden_shares.ok()) return hidden_shares.status();

        auto region1_plan = FfnGeluPackingPlanner::RealColumns(
            hidden_spec.shape, region1.slot_count(), Layout::kStrictColumn);
        if (!region1_plan.ok()) return region1_plan.status();
        auto region1_pair = Scatter(hidden_shares.value(), region1_plan.value());
        if (!region1_pair.ok()) return region1_pair.status();
        auto region1_map = ToConversionMap(region1_plan.value());
        if (!region1_map.ok()) return region1_map.status();
        before = converter_.stats();
        auto region1_he = converter_.MpcToCkks(
            region1_pair.value(),
            {region1.profile, std::size_t{0},
             region1.parameters.input_scale_bits, hidden_spec.shape,
             Layout::kStrictColumn, region1_map.take_value()});
        if (!region1_he.ok()) return region1_he.status();
        AccumulateConversion(before, converter_.stats(), report, true);

        std::optional<LayerNormStatisticsOutputs> statistics;
        if (contexts_.role() == PartyRole::kServer) {
            if (!region1_he.value())
                return Status::Internal(
                    "server did not receive next Region-1 ciphertext");
            SealCipherTensor input;
            input.parameter_profile = region1.profile;
            input.ciphertexts = region1_he.value()->ciphertexts;
            input.logical_spec = HeSpec(
                hidden_spec.shape, Layout::kStrictColumn,
                HeRegionId::kRegion1, region1, 0,
                region1.parameters.input_scale_bits, config_.mpc_numeric);
            input.physical_packing = region1_plan.value();
            auto computed = LayerNormStatisticsKernel::Evaluate(
                input, next, region1,
                report ? &report->layer_norm : nullptr);
            if (!computed.ok()) return computed.status();
            statistics = computed.take_value();
        }

        auto convert_statistics = [&](bool centered)
            -> Result<MpcShareTensor> {
            const auto shape = centered
                ? TensorShape{{next.tokens, next.model_width}}
                : TensorShape{{next.tokens, 1}};
            const auto plan = centered ? region1_plan.value()
                                       : VariancePlan(next.tokens,
                                                      region1.slot_count());
            std::optional<SealCipherBatch> batch;
            if (contexts_.role() == PartyRole::kServer) {
                const auto &tensor = centered ? statistics->centered
                                              : statistics->variance;
                auto map = ToConversionMap(tensor.physical_packing);
                if (!map.ok()) return map.status();
                batch = SealCipherBatch{region1.profile, tensor.ciphertexts,
                                        shape, plan.layout,
                                        map.take_value()};
            }
            auto before_local = converter_.stats();
            auto physical = converter_.CkksToMpc(batch);
            if (!physical.ok()) return physical.status();
            AccumulateConversion(before_local, converter_.stats(), report,
                                 false);
            return Compact(physical.value(), plan, shape, true);
        };
        auto centered = convert_statistics(true);
        if (!centered.ok()) return centered.status();
        auto variance = convert_statistics(false);
        if (!variance.ok()) return variance.status();
        auto inverse = mpc_layer_norm_.ReciprocalSqrt(
            variance.value(), next.attention_norm_rsqrt_bits,
            report ? &report->layer_norm : nullptr);
        if (!inverse.ok()) return inverse.status();
        mpc_layer_norm_.ReleaseSession();

        const auto centered_id = allocate_id();
        const auto inverse_id = allocate_id();
        TensorSpec centered_spec;
        centered_spec.shape = centered.value().physical_shape;
        centered_spec.domain = Domain::kMpcArithmeticShare;
        centered_spec.layout = Layout::kStrictColumn;
        centered_spec.numeric = centered.value().numeric;
        TensorSpec inverse_spec;
        inverse_spec.shape = inverse.value().physical_shape;
        inverse_spec.domain = Domain::kMpcArithmeticShare;
        inverse_spec.layout = Layout::kReduction;
        inverse_spec.numeric = inverse.value().numeric;
        auto stored = mpc_tensors_.PutMpcShare(centered_id, centered.value());
        if (!stored.ok()) return stored.status();
        stored = mpc_tensors_.PutMpcShare(inverse_id, inverse.value());
        if (!stored.ok()) return stored.status();

        auto dense = MultiHeadPackingPlan::DenseSpatial(
            hidden_spec.shape, region2.slot_count());
        if (!dense.ok()) return dense.status();
        auto centered_dense = Scatter(centered.value(), dense.value());
        if (!centered_dense.ok()) return centered_dense.status();
        MpcShareTensor inverse_broadcast;
        inverse_broadcast.physical_shape = hidden_spec.shape;
        inverse_broadcast.numeric = inverse.value().numeric;
        inverse_broadcast.component = ComplexComponent::kReal;
        inverse_broadcast.local_share.resize(next.tokens * next.model_width);
        for (std::size_t token = 0; token < next.tokens; ++token)
            for (std::size_t feature = 0; feature < next.model_width; ++feature)
                inverse_broadcast.local_share[token * next.model_width + feature] =
                    inverse.value().local_share[token];
        auto inverse_dense = Scatter(inverse_broadcast, dense.value());
        if (!inverse_dense.ok()) return inverse_dense.status();
        auto dense_map = ToConversionMap(dense.value());
        if (!dense_map.ok()) return dense_map.status();
        before = converter_.stats();
        auto centered_he = converter_.MpcToCkks(
            centered_dense.value(),
            {region2.profile, std::size_t{0},
             region2.parameters.input_scale_bits, hidden_spec.shape,
             Layout::kSpatial, dense_map.value()});
        if (!centered_he.ok()) return centered_he.status();
        AccumulateConversion(before, converter_.stats(), report, true);
        before = converter_.stats();
        auto inverse_he = converter_.MpcToCkks(
            inverse_dense.value(),
            {region2.profile, std::size_t{0},
             region2.parameters.input_scale_bits, hidden_spec.shape,
             Layout::kSpatial, dense_map.value()});
        if (!inverse_he.ok()) return inverse_he.status();
        AccumulateConversion(before, converter_.stats(), report, true);

        const auto block_centered_id = allocate_id();
        const auto block_inverse_id = allocate_id();
        const auto score_id = allocate_id();
        const auto value_id = allocate_id();
        const auto block_input_spec = HeSpec(
            hidden_spec.shape, Layout::kSpatial, HeRegionId::kRegion2,
            region2, 0, region2.parameters.input_scale_bits,
            config_.mpc_numeric);
        if (contexts_.role() == PartyRole::kServer) {
            if (!centered_he.value() || !inverse_he.value())
                return Status::Internal(
                    "server did not receive next Block-2 inputs");
            SealCipherTensor centered_tensor{
                region2.profile, centered_he.value()->ciphertexts,
                block_input_spec, dense.value()};
            SealCipherTensor inverse_tensor{
                region2.profile, inverse_he.value()->ciphertexts,
                block_input_spec, dense.value()};
            stored = he_tensors_.Put(block_centered_id,
                                     std::move(centered_tensor));
            if (!stored.ok()) return stored.status();
            stored = he_tensors_.Put(block_inverse_id,
                                     std::move(inverse_tensor));
            if (!stored.ok()) return stored.status();
            SealBlock2Executor block2(config_, next, contexts_, he_tensors_,
                                      parameters_);
            auto result = block2.Evaluate(
                {{block_centered_id, block_input_spec},
                 {block_inverse_id, block_input_spec}},
                *next_weights, report ? &report->next_block2 : nullptr);
            if (!result.ok()) return result.status();
            auto scores = he_tensors_.Get(result.value().scaled_attention_scores.id);
            if (!scores.ok()) return scores.status();
            auto values = he_tensors_.Get(result.value().value.id);
            if (!values.ok()) return values.status();
            stored = he_tensors_.Put(score_id, scores.value());
            if (!stored.ok()) return stored.status();
            stored = he_tensors_.Put(value_id, values.value());
            if (!stored.ok()) return stored.status();
        }
        auto score_spec = HeSpec(
            {{next.q_heads, next.tokens, next.tokens}},
            Layout::kAttentionDiagonal, HeRegionId::kRegion2, region2,
            region2.parameters.coefficient_modulus_bits.size() -
                region2.parameters.terminal_prime_count,
            region2.parameters.output_scale_bits, config_.mpc_numeric);
        auto value_spec = HeSpec(
            {{next.tokens, next.kv_heads, next.value_head_width}},
            Layout::kMultiHeadV, HeRegionId::kRegion2, region2, 2,
            region2.parameters.default_scale_bits, config_.mpc_numeric);
        outputs.next_layer = NextLayerBridgeOutputs{
            {centered_id, centered_spec}, {inverse_id, inverse_spec},
            {{score_id, score_spec}, {value_id, value_spec}}};
        return outputs;
    } catch (const std::exception &error) {
        return Status::Internal(std::string("Block-5 bridge evaluation: ") +
                                error.what());
    }
}

} // namespace hera
