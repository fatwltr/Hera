#include "hera/blb_reference.h"
#include "hera/pre_ln_entry.h"
#include "hera/detail/he_level.h"
#include "hera/detail/he_packing.h"
#include "hera/detail/packing_utils.h"
#include "hera/detail/scoped_wall_timer.h"

#include <chrono>
#include <algorithm>

namespace hera {
namespace {
using Clock = std::chrono::steady_clock;

using detail::LevelId;

using detail::ToConversionMap;

Result<ComplexMpcSharePair> Scatter(const MpcShareTensor &logical,
                                    const HePackingPlan &packing)
{
    auto valid = logical.Validate();
    if (!valid.ok()) return valid.status();
    if (logical.local_share.size() != packing.logical_locations.size())
        return Status::Invalid("share and packing logical sizes differ");
    const auto count = packing.ciphertext_count * packing.slot_count;
    std::vector<std::uint64_t> real(count), imag(count);
    for (std::size_t i = 0; i < logical.local_share.size(); ++i) {
        for (const auto &location : packing.Locations(i)) {
            auto &plane = location.component == ComplexLane::kReal
                ? real : imag;
            plane[location.ciphertext * packing.slot_count + location.slot] =
                logical.local_share[i];
        }
    }
    const TensorShape physical{{packing.ciphertext_count, packing.slot_count}};
    return ComplexMpcSharePair{
        {std::move(real), physical, logical.numeric, ComplexComponent::kReal},
        {std::move(imag), physical, logical.numeric,
         ComplexComponent::kImaginary}};
}

Result<MpcShareTensor> Compact(const ComplexMpcSharePair &physical,
                               const HePackingPlan &packing,
                               const TensorShape &shape, bool is_signed,
                               bool paired_real = false)
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
        if (paired_real && location.component == ComplexLane::kImaginary)
            values[logical] = detail::UndoPairedRealPhase(
                values[logical], location.slot, physical.real.numeric.ring_bits);
    }
    auto numeric = physical.real.numeric;
    numeric.is_signed = is_signed;
    return MpcShareTensor{std::move(values), shape, numeric,
                          ComplexComponent::kReal};
}

HePackingPlan VariancePacking(std::size_t tokens, std::size_t slots)
{
    HePackingPlan result;
    result.layout = Layout::kReduction;
    result.logical_shape = {{tokens, 1}};
    result.slot_count = slots;
    result.ciphertext_count = 1;
    result.logical_locations.resize(tokens);
    for (std::size_t token = 0; token < tokens; ++token)
        result.logical_locations[token].push_back(
            {0, token, token < (tokens + 1) / 2 ? ComplexLane::kReal : ComplexLane::kImaginary, std::nullopt});
    return result;
}

TensorSpec HeSpec(const TensorShape &shape, Layout layout, HeRegionId region,
                  const HeContext &context, std::size_t level,
                  int scale_bits, const NumericFormat &numeric)
{
    TensorSpec result;
    result.shape = shape;
    result.domain = Domain::kHeCiphertext;
    result.layout = layout;
    result.numeric = numeric;
    result.he_region = region;
    result.he_parameter_profile = context.profile;
    result.planned_level = level;
    result.planned_scale_bits = scale_bits;
    return result;
}

void AddConversion(const ConversionStats &before, const ConversionStats &after,
                   PreLnEntryExecutionReport *report, bool to_he)
{
    if (!report) return;
    if (to_he) ++report->mpc_to_ckks_calls;
    else ++report->ckks_to_mpc_calls;
    report->conversion_communication_bytes +=
        after.communication_bytes - before.communication_bytes;
    report->conversion_rounds += after.rounds - before.rounds;
    report->conversion_milliseconds += after.milliseconds - before.milliseconds;
}

} // namespace

SealSciPreLnEntryExecutor::SealSciPreLnEntryExecutor(
    const HeraConfig &config, std::size_t layer_index,
    HeContextRegistry &contexts, RuntimeTensorArena &arena,
    SealPlainParameterStore &parameters, BlbReferenceConverter &converter,
    MpcLayerNormRuntime &mpc_layer_norm, const CompiledLayerExecutionPlan *plan)
    : SealSciPreLnEntryExecutor(config, layer_index, contexts, arena.he, arena.mpc,
          parameters, converter, mpc_layer_norm, &arena.ids, plan)
{}

SealSciPreLnEntryExecutor::SealSciPreLnEntryExecutor(
    const HeraConfig &config, std::size_t layer_index,
    HeContextRegistry &contexts, SealCipherTensorStore &he_tensors,
    CryptoTensorStore &mpc_tensors, SealPlainParameterStore &parameters,
    BlbReferenceConverter &converter, MpcLayerNormRuntime &mpc_layer_norm,
    TensorIdAllocator *ids, const CompiledLayerExecutionPlan *plan)
    : config_(config), layer_index_(layer_index), contexts_(contexts),
      he_tensors_(he_tensors), mpc_tensors_(mpc_tensors), ids_(ids),
      parameters_(parameters), converter_(converter), mpc_layer_norm_(mpc_layer_norm), plan_(plan)
{}

Result<PreLnEntryOutputs> SealSciPreLnEntryExecutor::Evaluate(
    const TensorHandle &hidden, const LayerWeightHandles &weights,
    PreLnEntryExecutionReport *report, PreLnEntryShareOutputs *retained_shares)
{
    try {
        if (report) *report = {};
        if (retained_shares) *retained_shares = {};
        if (layer_index_ >= config_.layers.size())
            return Status::Invalid("Pre-LN layer index is out of range");
        const auto &layer = config_.layers[layer_index_];
        if (layer.norm == NormPlacement::kPostNorm)
            return Status::KernelUnavailable("Pre-LN entry supports pre or normalized_residual, not post");
        std::optional<ModelPlaintextCacheScope> layer_cache;
        if (parameters_.plaintext_cache_frozen() || parameters_.preparing_plaintexts())
            layer_cache.emplace(&parameters_, layer.id);
        if (weights.layer_id != layer.id)
            return Status::Invalid("Pre-LN weight layer ID mismatch");
        const TensorShape hidden_shape{{layer.tokens, layer.model_width}};
        if (!hidden.valid() || hidden.spec.domain != Domain::kHeCiphertext ||
            hidden.spec.shape != hidden_shape ||
            hidden.spec.layout != Layout::kStrictColumn ||
            !hidden.spec.he_parameter_profile)
            return Status::Invalid(
                "Pre-LN input must be a real strict-column HE tensor");

        const auto *execution = FindLayerExecution(config_, layer.id);
        if (!execution) return Status::Invalid("Pre-LN placement is absent");
        const auto r1_it = execution->regions.find(HeRegionId::kRegion1);
        const auto r2_it = execution->regions.find(HeRegionId::kRegion2);
        const auto r4_it = execution->regions.find(HeRegionId::kRegion4);
        if (r1_it == execution->regions.end() ||
            r2_it == execution->regions.end() ||
            r4_it == execution->regions.end())
            return Status::Invalid("Pre-LN Region 1/2/4 placement is absent");
        const auto &source = contexts_.Require(
            *hidden.spec.he_parameter_profile, contexts_.role());
        const auto &region1 = contexts_.Require(
            r1_it->second.parameter_profile, contexts_.role());
        const auto &shared = contexts_.Require(
            r2_it->second.parameter_profile, contexts_.role());
        const auto &output_context = contexts_.Require(
            r4_it->second.parameter_profile, contexts_.role());
        if (shared.profile != output_context.profile)
            return Status::Invalid("Region 4 must use the same parameter profile as Region 2");
        auto output_valid = AttentionLn2LevelPlanner::ValidateOutput(output_context.parameters);
        if (!output_valid.ok()) return output_valid.status();
        const auto region1_terminal =
            region1.parameters.coefficient_modulus_bits.size() -
            region1.parameters.terminal_prime_count;
        constexpr std::size_t kRegion1StatisticsDepth = 3;
        if (region1_terminal < kRegion1StatisticsDepth)
            return Status::FailedPrecondition(
                "Region-1 profile has no three-level statistics suffix");
        const auto region1_start =
            region1_terminal - kRegion1StatisticsDepth;
        const auto statistics_input_bits = layer.norm == NormPlacement::kNormalizedResidual &&
            region1.profile == execution->regions.at(HeRegionId::kRegion5).parameter_profile
                ? region1.parameters.output_scale_bits : region1.parameters.input_scale_bits;
        if (plan_ && (plan_->layer_id != layer.id ||
                      plan_->statistics_profile != region1.profile ||
                      plan_->attention_profile != shared.profile ||
                      plan_->attention_output_profile != output_context.profile ||
                      plan_->statistics_input_level != region1_start))
            return Status::FailedPrecondition(
                "Pre-LN compiled plan does not match its HE contexts");

        auto source_plan = AttentionLn2PackingPlanner::RealStrictColumns(
            hidden_shape, source.slot_count());
        if (!source_plan.ok()) return source_plan.status();
        auto source_map = ToConversionMap(detail::PairedRealPacking(source_plan.value()));
        if (!source_map.ok()) return source_map.status();
        std::optional<SealCipherBatch> source_batch;
        std::optional<SealCipherTensor> source_tensor;
        if (contexts_.role() == PartyRole::kServer) {
            auto loaded = he_tensors_.Get(hidden.id);
            if (!loaded.ok()) return loaded.status();
            source_tensor = loaded.take_value();
            auto valid = source_tensor->Validate(source);
            if (!valid.ok()) return valid.status();
        }
        const bool direct_region1 = source.profile == region1.profile &&
            hidden.spec.planned_level &&
            *hidden.spec.planned_level <= region1_start &&
            hidden.spec.planned_scale_bits ==
                std::optional<int>{statistics_input_bits};
        auto before = converter_.stats();
        std::optional<MpcShareTensor> hidden_shares;
        if (!direct_region1) {
            if (contexts_.role() == PartyRole::kServer) {
                detail::ScopedWallTimer he_timer(report ? &report->he_milliseconds : nullptr);
                auto paired = detail::PairRealCiphertexts(source_tensor->ciphertexts, source, config_.threads);
                if (!paired.ok()) return paired.status();
                source_batch = SealCipherBatch{
                    source.profile, paired.take_value(), hidden_shape,
                    Layout::kStrictColumn, source_map.value()};
            }
            auto source_physical = converter_.CkksToMpc(source_batch);
            if (!source_physical.ok()) return source_physical.status();
            AddConversion(before, converter_.stats(), report, false);
            auto compact = Compact(source_physical.value(), detail::PairedRealPacking(source_plan.value()), hidden_shape, true, true);
            if (!compact.ok()) return compact.status();
            hidden_shares = compact.take_value();
        }

        const auto shared_terminal =
            shared.parameters.coefficient_modulus_bits.size() -
            shared.parameters.terminal_prime_count;
        if (plan_ && plan_->attention_scores_level != shared_terminal)
            return Status::FailedPrecondition(
                "Pre-LN compiled attention terminal level mismatch");
        const auto attention_residual_level =
            AttentionLn2LevelPlanner::OutputInputLevel(output_context.parameters) + 1;
        if (plan_ && plan_->attention_residual_level != attention_residual_level)
            return Status::FailedPrecondition("compiled residual level mismatch");
        constexpr auto residual_layout = Layout::kStrictColumn;
        if (plan_ && plan_->attention_residual_layout != residual_layout)
            return Status::FailedPrecondition("compiled residual layout mismatch");
        const auto residual_id = (ids_ ? ids_ : &owned_ids_)->Next();
        auto residual_spec = HeSpec(
            hidden_shape, residual_layout, HeRegionId::kRegion4,
            output_context, attention_residual_level,
            output_context.parameters.input_scale_bits, config_.mpc_numeric);
        const bool normalized_skip = layer.norm == NormPlacement::kNormalizedResidual;
        if (normalized_skip) {
            residual_spec.planned_level = 1;
        } else if (layer.attention_residual) {
            // Preserve the original hidden value, not the LN1-normalized QKV
            // input. Only its copy used by conversion may drop to terminal.
            if (source.profile != output_context.profile ||
                !hidden.spec.planned_level ||
                *hidden.spec.planned_level > attention_residual_level ||
                hidden.spec.planned_scale_bits !=
                    std::optional<int>{output_context.parameters.default_scale_bits})
                return Status::FailedPrecondition(
                    "shared Region-2/4 residual needs original real hidden "
                    "in the shared profile at or above the Wo output level");
            residual_spec.planned_level = hidden.spec.planned_level;
            if (contexts_.role() == PartyRole::kServer) {
                auto retained = *source_tensor;
                retained.logical_spec = residual_spec;
                auto stored = he_tensors_.Put(residual_id, std::move(retained));
                if (!stored.ok()) return stored.status();
            }
        }

        // Region-1 statistics consume exactly three levels. Start at the
        // terminal suffix so variance naturally reaches the conversion
        // boundary instead of performing a final conversion-only drop.
        auto region1_plan = AttentionLn2PackingPlanner::RealStrictColumns(
            hidden_shape, region1.slot_count());
        if (!region1_plan.ok()) return region1_plan.status();
        std::optional<SealCipherTensor> region1_input;
        if (contexts_.role() == PartyRole::kServer && direct_region1) {
            detail::ScopedWallTimer he_timer(report ? &report->he_milliseconds : nullptr);
            region1_input = std::move(source_tensor);
            auto target = LevelId(region1, region1_start);
            if (!target.ok()) return target.status();
            seal::Evaluator evaluator(*region1.context);
            for (auto &ct : region1_input->ciphertexts)
                if (ct.parms_id() != target.value())
                    evaluator.mod_switch_to_inplace(ct, target.value());
            region1_input->logical_spec.planned_level = region1_start;
            region1_input->logical_spec.he_region = HeRegionId::kRegion1;
        } else if (!direct_region1) {
            auto pair = Scatter(hidden_shares.value(), region1_plan.value());
            if (!pair.ok()) return pair.status();
            auto map = ToConversionMap(region1_plan.value());
            if (!map.ok()) return map.status();
            before = converter_.stats();
            auto refreshed = converter_.MpcToCkksPacked(
                pair.value(), {region1.profile, region1_start,
                    statistics_input_bits, hidden_shape,
                    Layout::kStrictColumn, map.value()});
            if (!refreshed.ok()) return refreshed.status();
            AddConversion(before, converter_.stats(), report, true);
            if (contexts_.role() == PartyRole::kServer) {
                if (!refreshed.value())
                    return Status::Internal(
                        "server did not receive Region-1 refresh");
                region1_input = SealCipherTensor{
                    region1.profile, refreshed.value()->ciphertexts,
                    HeSpec(hidden_shape, Layout::kStrictColumn,
                           HeRegionId::kRegion1, region1, region1_start,
                           statistics_input_bits,
                           config_.mpc_numeric), region1_plan.value()};
            }
        }

        std::optional<LayerNormStatisticsOutputs> statistics;
        if (contexts_.role() == PartyRole::kServer) {
            detail::ScopedWallTimer he_timer(report ? &report->he_milliseconds : nullptr);
            auto computed = LayerNormStatisticsKernel::Evaluate(
                *region1_input, layer, region1,
                report ? &report->statistics : nullptr, config_.threads, statistics_input_bits);
            if (!computed.ok()) return computed.status();
            statistics = computed.take_value();
        }

        auto convert_statistic = [&](bool centered)
            -> Result<MpcShareTensor> {
            const TensorShape shape = centered
                ? hidden_shape : TensorShape{{layer.tokens, 1}};
            const auto packing = centered ? detail::PairedRealPacking(region1_plan.value())
                : VariancePacking(layer.tokens, region1.slot_count());
            auto map = ToConversionMap(packing);
            if (!map.ok()) return map.status();
            std::optional<SealCipherBatch> batch;
            if (contexts_.role() == PartyRole::kServer) {
                detail::ScopedWallTimer he_timer(report ? &report->he_milliseconds : nullptr);
                const auto &tensor = centered ? statistics->centered
                                              : statistics->variance;
                auto ciphertexts = centered
                    ? detail::PairRealCiphertexts(tensor.ciphertexts, region1, config_.threads)
                    : Result<std::vector<seal::Ciphertext>>{tensor.ciphertexts};
                if (!ciphertexts.ok()) return ciphertexts.status();
                batch = SealCipherBatch{region1.profile, ciphertexts.take_value(),
                                        shape, packing.layout, map.value()};
            }
            auto local_before = converter_.stats();
            auto physical = converter_.CkksToMpc(batch, centered ? config_.mpc_numeric.scale_bits :
                std::max(config_.mpc_numeric.scale_bits, layer.attention_norm_variance_scale_bits));
            if (!physical.ok()) return physical.status();
            AddConversion(local_before, converter_.stats(), report, false);
            return Compact(physical.value(), packing, shape, true, centered);
        };
        auto centered = convert_statistic(true);
        if (!centered.ok()) return centered.status();
        auto variance = convert_statistic(false);
        if (!variance.ok()) return variance.status();
        const auto mpc_started = Clock::now();
        auto inverse = mpc_layer_norm_.ReciprocalSqrt(
            variance.value(), layer.attention_norm_rsqrt_bits,
            report ? &report->statistics : nullptr, config_.mpc_numeric.scale_bits);
        if (!inverse.ok()) return inverse.status();
        // The LayerNorm runtime is shared by the orchestrator; retaining its
        // Ferret state avoids another base/setup phase at LN2 and next layers.
        if (report)
            report->mpc_milliseconds += std::chrono::duration<double,
                std::milli>(Clock::now() - mpc_started).count();

        auto dense = MultiHeadPackingPlan::RealStrictColumns(
            hidden_shape, shared.slot_count());
        if (!dense.ok()) return dense.status();
        auto centered_pair = Scatter(centered.value(), dense.value());
        if (!centered_pair.ok()) return centered_pair.status();
        auto inverse_plan = MultiHeadPackingPlan::TokenInverseBroadcast(layer.tokens, shared.slot_count());
        if (!inverse_plan.ok()) return inverse_plan.status();
        auto inverse_pair = Scatter(inverse.value(), inverse_plan.value());
        if (!inverse_pair.ok()) return inverse_pair.status();
        const auto centered_cts = dense.value().ciphertext_count;
        auto packed = centered_pair.take_value();
        for (auto member : {&ComplexMpcSharePair::real, &ComplexMpcSharePair::imag}) {
            auto &plane = packed.*member;
            const auto &extra = inverse_pair.value().*member;
            plane.local_share.insert(plane.local_share.end(), extra.local_share.begin(), extra.local_share.end());
            plane.physical_shape = {{centered_cts + 1, shared.slot_count()}};
        }
        auto map = ToConversionMap(dense.value());
        if (!map.ok()) return map.status();
        auto packed_map = map.take_value();
        const auto inverse_offset = packed_map.logical_element_count;
        packed_map.ciphertext_count = centered_cts + 1;
        packed_map.logical_element_count += shared.slot_count();
        packed_map.assignments.resize((centered_cts + 1) * shared.slot_count());
        for (std::size_t slot = 0; slot < shared.slot_count(); ++slot)
            packed_map.assignments[centered_cts * shared.slot_count() + slot].real_logical_index = inverse_offset + slot;
        auto local_before = converter_.stats();
        std::vector<std::optional<int>> input_bounds(centered_cts + 1);
        input_bounds.back() = inverse.value().public_value_bound_bits;
        MpcToCkksTarget input_target{shared.profile, std::size_t{0}, shared.parameters.input_scale_bits,
            {{packed_map.logical_element_count}}, Layout::kStrictColumn, packed_map, true,
            std::nullopt, std::move(input_bounds)};
        input_target.public_slot_periods.resize(centered_cts + 1);
        input_target.public_slot_periods.back()[0] = detail::TokenColumnStride(layer.tokens);
        auto refreshed = converter_.MpcToCkksPacked(packed, input_target);
        if (!refreshed.ok()) return refreshed.status();
        AddConversion(local_before, converter_.stats(), report, true);

        const auto centered_id = (ids_ ? ids_ : &owned_ids_)->Next();
        const auto inverse_id = (ids_ ? ids_ : &owned_ids_)->Next();
        const auto block_spec = HeSpec(
            hidden_shape, Layout::kStrictColumn, HeRegionId::kRegion2, shared, 0,
            shared.parameters.input_scale_bits, config_.mpc_numeric);
        auto inverse_spec = block_spec;
        inverse_spec.shape = {{layer.tokens, 1}};
        inverse_spec.layout = Layout::kReduction;
        if (contexts_.role() == PartyRole::kServer) {
            if (!refreshed.value() || refreshed.value()->ciphertexts.size() != centered_cts + 1)
                return Status::Internal("server did not receive batched Block-2 inputs");
            auto &ciphertexts = refreshed.value()->ciphertexts;
            std::vector<seal::Ciphertext> inverse_cts;
            inverse_cts.push_back(std::move(ciphertexts.back()));
            ciphertexts.pop_back();
            auto stored = he_tensors_.Put(centered_id,
                {shared.profile, std::move(ciphertexts), block_spec, dense.value()});
            if (!stored.ok()) return stored.status();
            stored = he_tensors_.Put(inverse_id,
                {shared.profile, std::move(inverse_cts), inverse_spec, inverse_plan.value()});
            if (!stored.ok()) return stored.status();
        }

        const auto score_id = (ids_ ? ids_ : &owned_ids_)->Next();
        const auto value_id = (ids_ ? ids_ : &owned_ids_)->Next();
        TensorSpec score_spec = HeSpec(
            {{layer.q_heads, layer.tokens, layer.tokens}},
            Layout::kAttentionDiagonal, HeRegionId::kRegion2, shared,
            shared_terminal,
            shared.parameters.output_scale_bits, config_.mpc_numeric);
        TensorSpec value_spec = HeSpec(
            {{layer.tokens, layer.kv_heads, layer.value_head_width}},
            Layout::kMultiHeadV, HeRegionId::kRegion2, shared,
            plan_ ? plan_->retained_value_level : std::size_t{2},
            shared.parameters.default_scale_bits, config_.mpc_numeric);
        if (contexts_.role() == PartyRole::kServer) {
            detail::ScopedWallTimer he_timer(report ? &report->he_milliseconds : nullptr);
            SealBlock2Executor block2(config_, layer, contexts_, he_tensors_,
                                      parameters_,
                                      plan_ ? &plan_->block2_rotation_steps
                                            : nullptr);
            SealCipherTensor normalized_residual;
            auto evaluated = block2.Evaluate(
                {{centered_id, block_spec}, {inverse_id, inverse_spec}},
                weights, report ? &report->block2 : nullptr,
                normalized_skip && layer.attention_residual ? &normalized_residual : nullptr);
            if (!evaluated.ok()) return evaluated.status();
            if (normalized_skip && layer.attention_residual) {
                normalized_residual.logical_spec = residual_spec;
                auto stored = he_tensors_.Put(residual_id, std::move(normalized_residual));
                if (!stored.ok()) return stored.status();
            }
            auto stored = he_tensors_.Rename(evaluated.value().scaled_attention_scores.id, score_id);
            if (!stored.ok()) return stored.status();
            stored = he_tensors_.Rename(evaluated.value().value.id, value_id);
            if (!stored.ok()) return stored.status();
            he_tensors_.Erase(centered_id);
            he_tensors_.Erase(inverse_id);
        }
        PreLnEntryOutputs outputs;
        outputs.block2 = {{score_id, score_spec}, {value_id, value_spec}};
        if (layer.attention_residual) outputs.attention_residual = {residual_id, residual_spec};
        if (retained_shares) {
            const auto retain = [&](MpcShareTensor shares, Layout layout) -> Result<TensorHandle> {
                const auto id = (ids_ ? ids_ : &owned_ids_)->Next();
                TensorSpec spec;
                spec.shape = shares.physical_shape;
                spec.domain = Domain::kMpcArithmeticShare;
                spec.layout = layout;
                spec.numeric = shares.numeric;
                auto stored = mpc_tensors_.PutMpcShare(id, std::move(shares));
                if (!stored.ok()) return stored.status();
                return TensorHandle{id, spec};
            };
            auto centered_handle = retain(centered.take_value(), Layout::kStrictColumn);
            if (!centered_handle.ok()) return centered_handle.status();
            auto inverse_handle = retain(inverse.take_value(), Layout::kReduction);
            if (!inverse_handle.ok()) return inverse_handle.status();
            retained_shares->centered_shares = centered_handle.take_value();
            retained_shares->inverse_stddev_shares = inverse_handle.take_value();
        }
        return outputs;
    } catch (const std::exception &error) {
        return Status::Internal(std::string("Pre-LN entry failed: ") +
                                error.what());
    }
}

} // namespace hera
