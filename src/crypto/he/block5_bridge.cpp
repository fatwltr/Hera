#include "hera/blb_reference.h"
#include "hera/block5_bridge.h"
#include "hera/detail/he_level.h"
#include "hera/detail/level_schedule.h"
#include "hera/detail/he_packing.h"
#include "hera/pre_ln_entry.h"
#include "hera/detail/scoped_wall_timer.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>

namespace hera {
namespace {

using detail::LevelOf;

bool CloseScale(double left, double right)
{
    return std::isfinite(left) && std::isfinite(right) && left > 0.0 &&
           right > 0.0 &&
           std::abs(left - right) / std::max(left, right) < 1.0e-3;
}

bool SamePacking(const HePackingPlan &left, const HePackingPlan &right)
{
    return detail::PackingMatches(left, right);
}

const PlainParameterHandle *FindPlain(const LayerWeightHandles &weights,
                                      const std::string &name)
{
    for (const auto &parameter : weights.plain_parameters)
        if (parameter.spec.name == name) return &parameter;
    return nullptr;
}

using detail::ToConversionMap;

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
    BlbReferenceConverter &converter, MpcLayerNormRuntime &mpc_layer_norm,
    TensorIdAllocator *ids)
    : config_(config), current_layer_index_(current_layer_index),
      contexts_(contexts), he_tensors_(he_tensors), parameters_(parameters),
      mpc_tensors_(mpc_tensors), converter_(converter),
      mpc_layer_norm_(mpc_layer_norm), ids_(ids)
{}

Result<std::map<std::string, std::vector<int>>>
SealSciBlock5BridgeExecutor::RequiredRotationSteps(
    const HeraConfig &config, std::size_t current_layer_index,
    bool include_next_layer)
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
        auto output = FfnGeluPackingPlanner::RealColumns(
            {{layer.tokens, layer.model_width}}, slots, Layout::kStrictColumn);
        if (!output.ok()) return output.status();
        auto steps = ComplexColumnProjectionKernel::RequiredRotationSteps(
            input.value(), output.value());
        if (!steps.ok()) return steps.status();
        collected[profile].insert(steps.value().begin(), steps.value().end());
        collected[profile].insert(0); // conjugation key
        for (const auto *boundary : {&input.value(), &output.value()}) {
            auto map = ToConversionMap(*boundary);
            if (!map.ok()) return map.status();
            auto packing = detail::PlanRealConversion(map.value());
            if (!packing.ok()) return packing.status();
            collected[profile].insert(packing.value().rotations.begin(), packing.value().rotations.end());
        }
        return {};
    };
    const auto &layer = config.layers[current_layer_index];
    const auto *execution = FindLayerExecution(config, layer.id);
    if (!execution) return Status::Invalid("Block-5 layer placement is absent");
    const bool shared_skip = layer.norm == NormPlacement::kNormalizedResidual &&
        execution->regions.at(HeRegionId::kRegion1).parameter_profile ==
            execution->regions.at(HeRegionId::kRegion5).parameter_profile;
    const auto shared = execution->regions.find(shared_skip ? HeRegionId::kRegion5 : HeRegionId::kRegion4);
    if (shared == execution->regions.end())
        return Status::Invalid("Block-5 shared residual region is absent");
    const auto profile_it = config.he_parameter_profiles.find(
        shared->second.parameter_profile);
    if (profile_it == config.he_parameter_profiles.end())
        return Status::Invalid("Block-5 HE profile is absent");
    if (!shared_skip && execution->regions.at(HeRegionId::kRegion2).parameter_profile !=
        shared->second.parameter_profile)
        return Status::Invalid("Region 4 must use the same parameter profile as Region 2");
    auto added = add_projection(layer, shared->second.parameter_profile,
        profile_it->second.polynomial_degree / 2);
    if (!added.ok()) return added.status();

    if (include_next_layer && current_layer_index + 1 < config.layers.size()) {
        const auto &next = config.layers[current_layer_index + 1];
        if (next.norm == NormPlacement::kPostNorm)
            return Status::KernelUnavailable(
                "automatic cross-layer bridge supports pre or normalized_residual, not post");
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
        collected[region1->second.parameter_profile].insert(0);
        auto real_plan = AttentionLn2PackingPlanner::RealStrictColumns(
            {{next.tokens, next.model_width}}, r1->second.polynomial_degree / 2);
        if (!real_plan.ok()) return real_plan.status();
        auto map = ToConversionMap(real_plan.value());
        if (!map.ok()) return map.status();
        auto packing = detail::PlanRealConversion(map.value());
        if (!packing.ok()) return packing.status();
        collected[region1->second.parameter_profile].insert(
            packing.value().rotations.begin(), packing.value().rotations.end());
        collected[region2->second.parameter_profile].insert(0);
        collected[region2->second.parameter_profile].insert(static_cast<int>(r2->second.polynomial_degree / 4));
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
        if (layer.norm == NormPlacement::kPostNorm)
            return Status::KernelUnavailable(
                "Block-5 bridge supports pre or normalized_residual, not post");
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
        const bool shared_skip = layer.norm == NormPlacement::kNormalizedResidual &&
            execution->regions.at(HeRegionId::kRegion1).parameter_profile ==
                execution->regions.at(HeRegionId::kRegion5).parameter_profile;
        const auto wd_region = shared_skip ? HeRegionId::kRegion5 : HeRegionId::kRegion4;
        const auto shared_region = execution->regions.find(wd_region);
        if (shared_region == execution->regions.end())
            return Status::Invalid("shared residual HE region is absent");
        const auto &shared = contexts_.Require(
            shared_region->second.parameter_profile, contexts_.role());
        const int wd_scale_bits = shared_skip ? shared.parameters.output_scale_bits : shared.parameters.default_scale_bits;
        if (contexts_.role() == PartyRole::kServer && shared.galois_keys.size() == 0)
            return Status::FailedPrecondition("Block-5 Galois keys are absent");

        auto activation = mpc_tensors_.GetMpcShare(
            inputs.activation_shares.id);
        if (!activation.ok()) return activation.status();
        auto activation_plan = FfnGeluPackingPlanner::ComplexColumns(
            {{layer.tokens, layer.ffn_width}}, shared.slot_count(),
            Layout::kFfn);
        if (!activation_plan.ok()) return activation_plan.status();
        if (!shared_skip && execution->regions.at(HeRegionId::kRegion2).parameter_profile != shared.profile)
            return Status::Invalid("Region 4 must use the same parameter profile as Region 2");
        auto output_plan = FfnGeluPackingPlanner::RealColumns(
            {{layer.tokens, layer.model_width}}, shared.slot_count(), Layout::kStrictColumn);
        if (!output_plan.ok()) return output_plan.status();

        const bool normalized_skip = layer.norm == NormPlacement::kNormalizedResidual;
        std::size_t residual_level = 0;
        std::optional<SealCipherTensor> residual;
        std::optional<MpcShareTensor> residual_shares;
        if (inputs.ffn_residual && normalized_skip && !shared_skip) {
            if (inputs.ffn_residual->spec.domain != Domain::kMpcArithmeticShare ||
                inputs.ffn_residual->spec.layout != Layout::kStrictColumn ||
                inputs.ffn_residual->spec.shape != TensorShape{{layer.tokens, layer.model_width}})
                return Status::Invalid("normalized FFN residual must contain complete LN2 shares");
            auto loaded = mpc_tensors_.GetMpcShare(inputs.ffn_residual->id);
            if (!loaded.ok()) return loaded.status();
            residual_shares = loaded.take_value();
            if (residual_shares->numeric.ring_bits != activation.value().numeric.ring_bits ||
                residual_shares->numeric.scale_bits != activation.value().numeric.scale_bits ||
                !residual_shares->numeric.is_signed)
                return Status::Invalid("normalized FFN residual numeric format mismatch");
            residual_level = AttentionLn2LevelPlanner::OutputInputLevel(shared.parameters) + 1;
        } else if (inputs.ffn_residual) {
            if (inputs.ffn_residual->spec.domain != Domain::kHeCiphertext ||
                inputs.ffn_residual->spec.planned_scale_bits != std::optional<int>{wd_scale_bits} ||
                inputs.ffn_residual->spec.he_parameter_profile !=
                    shared.profile ||
                inputs.ffn_residual->spec.layout != output_plan.value().layout ||
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
            // Match the LN2 tail level even when its skip is disabled.
            residual_level = shared_skip
                ? shared.parameters.coefficient_modulus_bits.size() - shared.parameters.terminal_prime_count - (kFfnSuffixDepth - 1)
                : AttentionLn2LevelPlanner::OutputInputLevel(shared.parameters) + 1;
        }

        const auto terminal_level =
            shared.parameters.coefficient_modulus_bits.size() -
            shared.parameters.terminal_prime_count;
        // Shared R1/R5: encrypt the activation immediately above the LN1
        // statistics suffix, rather than carrying unused high-level primes.
        // Standard Pre-LN retains the Wo boundary for its raw hidden skip.
        const auto wd_output_level = shared_skip
            ? std::max(residual_level, detail::SharedFfnOutputLevel(shared.parameters))
            : residual_level;
        if (wd_output_level == 0 || wd_output_level > terminal_level)
            return Status::FailedPrecondition("shared profile has no Wd suffix");

        auto activation_pair = Scatter(activation.value(),
                                       activation_plan.value());
        if (!activation_pair.ok()) return activation_pair.status();
        auto activation_map = ToConversionMap(activation_plan.value());
        if (!activation_map.ok()) return activation_map.status();
        TensorShape conversion_shape{{layer.tokens, layer.ffn_width}};
        std::vector<std::optional<int>> conversion_bounds(
            activation_map.value().ciphertext_count, activation.value().public_value_bound_bits);
        if (residual_shares) {
            auto pair = Scatter(*residual_shares, output_plan.value());
            if (!pair.ok()) return pair.status();
            auto map = ToConversionMap(output_plan.value());
            if (!map.ok()) return map.status();
            conversion_bounds.insert(conversion_bounds.end(), map.value().ciphertext_count,
                                     residual_shares->public_value_bound_bits);
            const auto logical_offset = activation_map.value().logical_element_count;
            for (auto assignment : map.value().assignments) {
                if (assignment.real_logical_index != kUnusedLogicalIndex)
                    assignment.real_logical_index += logical_offset;
                activation_map.value().assignments.push_back(assignment);
            }
            activation_map.value().logical_element_count += map.value().logical_element_count;
            activation_map.value().ciphertext_count += map.value().ciphertext_count;
            for (auto planes : {std::make_pair(&activation_pair.value().real, &pair.value().real),
                                std::make_pair(&activation_pair.value().imag, &pair.value().imag)}) {
                planes.first->local_share.insert(planes.first->local_share.end(),
                    planes.second->local_share.begin(), planes.second->local_share.end());
                planes.first->physical_shape = {{activation_map.value().ciphertext_count, shared.slot_count()}};
            }
            conversion_shape = {{activation_map.value().logical_element_count}};
        }
        auto before = converter_.stats();
        auto activation_he = converter_.MpcToCkksPacked(
            activation_pair.value(),
            {shared.profile, wd_output_level - 1,
             wd_scale_bits,
             conversion_shape, Layout::kFfn, activation_map.take_value(),
             false, std::nullopt, std::move(conversion_bounds)});
        if (!activation_he.ok()) return activation_he.status();
        AccumulateConversion(before, converter_.stats(), report, true);

        const auto allocate_id = [&]() {
            return (ids_ ? ids_ : &owned_ids_)->Next();
        };
        const auto hidden_id = allocate_id();
        const auto hidden_level = wd_output_level;
        const auto hidden_spec = HeSpec(
            {{layer.tokens, layer.model_width}}, Layout::kStrictColumn,
            wd_region, shared, hidden_level,
            wd_scale_bits, config_.mpc_numeric);
        if (contexts_.role() == PartyRole::kServer) {
            detail::ScopedWallTimer he_timer(report ? &report->he_milliseconds : nullptr);
            if (!activation_he.value())
                return Status::Internal("server did not receive GeLU ciphertext");
            SealCipherTensor encoded;
            encoded.parameter_profile = shared.profile;
            auto &converted = activation_he.value()->ciphertexts;
            const auto activation_cts = activation_plan.value().ciphertext_count;
            if (converted.size() != activation_cts +
                (residual_shares ? output_plan.value().ciphertext_count : 0))
                return Status::Internal("batched activation/residual ciphertext count mismatch");
            for (std::size_t ct = 0; ct < activation_cts; ++ct)
                encoded.ciphertexts.push_back(std::move(converted[ct]));
            if (residual_shares) {
                residual.emplace();
                residual->parameter_profile = shared.profile;
                residual->physical_packing = output_plan.value();
                residual->logical_spec = hidden_spec;
                residual->logical_spec.planned_level = wd_output_level - 1;
                for (std::size_t ct = activation_cts; ct < converted.size(); ++ct)
                    residual->ciphertexts.push_back(std::move(converted[ct]));
            }
            encoded.logical_spec = HeSpec(
                {{layer.tokens, layer.ffn_width}}, Layout::kFfn,
                wd_region, shared, wd_output_level - 1,
                wd_scale_bits, config_.mpc_numeric);
            encoded.physical_packing = activation_plan.value();

            if (!current_weights.wd.valid() ||
                current_weights.wd.spec.rows != layer.ffn_width ||
                current_weights.wd.spec.columns != layer.model_width)
                return Status::Invalid("Wd handle has the wrong shape");
            auto wd = parameters_.GetMatrixShared(current_weights.wd.id);
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
                encoded, *wd.value(), bd, output_plan.value(),
                wd_region,
                wd_scale_bits, shared,
                report ? &report->he : nullptr, config_.threads);
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
            auto stored = he_tensors_.Put(hidden_id, down.take_value());
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
        // reusable Pre-LN entry stage prepare the next residual. Explicit
        // bridging delegates to that same stage, including its frozen cache.
        if (!next_weights)
            return outputs;
        SealSciPreLnEntryExecutor entry(config_, current_layer_index_ + 1,
            contexts_, he_tensors_, mpc_tensors_, parameters_, converter_, mpc_layer_norm_,
            ids_ ? ids_ : &owned_ids_);
        PreLnEntryExecutionReport next_report;
        PreLnEntryShareOutputs retained;
        auto next = entry.Evaluate(outputs.hidden_states, *next_weights,
            report ? &next_report : nullptr, &retained);
        if (!next.ok()) return next.status();
        if (report) {
            report->ckks_to_mpc_calls += next_report.ckks_to_mpc_calls;
            report->mpc_to_ckks_calls += next_report.mpc_to_ckks_calls;
            report->conversion_communication_bytes += next_report.conversion_communication_bytes;
            report->conversion_rounds += next_report.conversion_rounds;
            report->conversion_milliseconds += next_report.conversion_milliseconds;
            report->he_milliseconds += next_report.he_milliseconds;
            report->layer_norm = std::move(next_report.statistics);
            report->next_block2 = std::move(next_report.block2);
        }
        outputs.next_layer = NextLayerBridgeOutputs{
            retained.centered_shares, retained.inverse_stddev_shares,
            next.value().block2, next.value().attention_residual};
        return outputs;
    } catch (const std::exception &error) {
        return Status::Internal(std::string("Block-5 bridge evaluation: ") +
                                error.what());
    }
}

} // namespace hera
