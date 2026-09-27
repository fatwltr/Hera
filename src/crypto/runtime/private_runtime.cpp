#include "hera/private_runtime.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <set>

namespace hera {
namespace {

using Clock = std::chrono::steady_clock;

Result<seal::parms_id_type> LevelId(const HeContext &context,
                                    std::size_t target_level)
{
    auto data = context.context->first_context_data();
    for (std::size_t level = 0; data && level < target_level; ++level)
        data = data->next_context_data();
    if (!data)
        return Status::FailedPrecondition(
            "requested level exceeds the CKKS modulus chain");
    return data->parms_id();
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
    for (std::size_t logical = 0; logical < packing.logical_locations.size();
         ++logical) {
        const auto &locations = packing.Locations(logical);
        if (locations.size() != 1)
            return Status::Invalid(
                "conversion boundary requires one physical lane per value");
        const auto &location = locations.front();
        auto &entry = map.assignments[
            location.ciphertext * packing.slot_count + location.slot];
        auto &index = location.component == ComplexLane::kReal
            ? entry.real_logical_index : entry.imaginary_logical_index;
        if (index != kUnusedLogicalIndex)
            return Status::Invalid("conversion slot map is not injective");
        index = logical;
    }
    valid = map.Validate();
    if (!valid.ok()) return valid.status();
    return map;
}

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
                               const TensorShape &shape, bool is_signed)
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
            {0, token, ComplexLane::kReal, std::nullopt});
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

const LayerWeightHandles *FindLayerWeights(
    const TransformerWeightHandles &weights, const std::string &id)
{
    for (const auto &layer : weights.layers)
        if (layer.layer_id == id) return &layer;
    return nullptr;
}

void AppendStage(ExecutionReport *report, std::size_t layer,
                 std::string name, Clock::time_point started,
                 const Status &status,
                 ExecutionReport::StageMetrics metrics = {})
{
    if (std::getenv("HERA_RUNTIME_TRACE"))
        std::cerr << "HERA_RUNTIME_STAGE layer=" << layer
                  << " name=" << name
                  << " status=" << (status.ok() ? "OK" : status.message())
                  << '\n';
    if (!report) return;
    report->stages.push_back({layer, std::move(name),
        std::chrono::duration<double, std::milli>(Clock::now() - started).count(),
        status, std::move(metrics)});
}

ExecutionReport::StageMetrics Metrics(const PreLnEntryExecutionReport &value,
                                      std::size_t tokens)
{
    return {value.conversion_communication_bytes, value.conversion_rounds,
            value.ckks_to_mpc_calls + value.mpc_to_ckks_calls,
            value.block2.rotations,
            value.block2.ciphertext_plain_multiplications,
            value.block2.ciphertext_multiplications,
            value.block2.relinearizations, value.block2.rescales, tokens};
}

ExecutionReport::StageMetrics Metrics(const SoftmaxExecutionReport &value)
{
    return {value.conversion_communication_bytes +
                value.mpc_communication_bytes,
            value.conversion_rounds + value.mpc_rounds,
            value.ckks_to_mpc_calls + value.mpc_to_ckks_calls,
            value.rotations, value.ciphertext_plain_multiplications,
            value.ciphertext_multiplications, value.relinearizations,
            value.rescales, value.comparisons + value.muxes + value.row_sums +
                value.reciprocals + value.secure_multiplications};
}

ExecutionReport::StageMetrics Metrics(
    const AttentionLn2ExecutionReport &value)
{
    return {value.conversion_communication_bytes +
                value.mpc_communication_bytes,
            value.conversion_rounds + value.mpc_rounds,
            value.ckks_to_mpc_calls + value.mpc_to_ckks_calls,
            value.rotations, value.ciphertext_plain_multiplications,
            value.ciphertext_multiplications, value.relinearizations,
            value.rescales, value.reciprocal_square_roots};
}

ExecutionReport::StageMetrics Metrics(const FfnGeluExecutionReport &value)
{
    return {value.conversion_communication_bytes +
                value.mpc_communication_bytes,
            value.conversion_rounds + value.mpc_rounds,
            value.ckks_to_mpc_calls + value.mpc_to_ckks_calls,
            value.rotations, value.ciphertext_plain_multiplications,
            value.ciphertext_multiplications, value.relinearizations,
            value.rescales, value.comparisons + value.muxes};
}

ExecutionReport::StageMetrics Metrics(
    const Block5BridgeExecutionReport &value)
{
    return {value.conversion_communication_bytes, value.conversion_rounds,
            value.ckks_to_mpc_calls + value.mpc_to_ckks_calls,
            value.he.rotations, value.he.ciphertext_plain_multiplications,
            value.he.ciphertext_multiplications, value.he.relinearizations,
            value.he.rescales, 0};
}

Result<void> ValidateRuntimeConfig(const HeraConfig &config)
{
    for (const auto &layer : config.layers) {
        if (layer.norm != NormPlacement::kPreNorm)
            return Status::KernelUnavailable(
                "real HERA runtime currently supports Pre-LN only");
        if (layer.activation != ActivationKind::kGelu)
            return Status::KernelUnavailable(
                "real HERA runtime currently supports GeLU only");
    }
    return {};
}

Result<seal::parms_id_type> ParmsAtLevel(const HeContext &context,
                                         std::size_t level)
{
    auto data = context.context->first_context_data();
    for (std::size_t current = 0; data && current < level; ++current)
        data = data->next_context_data();
    if (!data) return Status::FailedPrecondition(
        "plaintext preparation level is outside the modulus chain");
    return data->parms_id();
}

// A Prepare-only contract carrier: no encryption, ciphertext coefficient
// allocation, or evaluator work is needed to compile model plaintexts.
Result<SealCipherTensor> PreparationTensor(
    const HeContext &context, const HePackingPlan &packing,
    TensorSpec spec, std::size_t level, int scale_bits)
{
    auto valid = packing.Validate();
    if (!valid.ok()) return valid.status();
    auto parms = ParmsAtLevel(context, level);
    if (!parms.ok()) return parms.status();
    std::vector<seal::Ciphertext> ciphertexts(packing.ciphertext_count);
    for (auto &ciphertext : ciphertexts) {
        ciphertext.parms_id() = parms.value();
        ciphertext.scale() = std::ldexp(1.0, scale_bits);
    }
    spec.domain = Domain::kHeCiphertext;
    spec.layout = packing.layout;
    spec.he_parameter_profile = context.profile;
    spec.planned_level = level;
    spec.planned_scale_bits = scale_bits;
    return SealCipherTensor{context.profile, std::move(ciphertexts),
                            spec, packing};
}

std::uint64_t WeightManifest(const TransformerWeightHandles &weights)
{
    std::uint64_t hash = 1469598103934665603ULL;
    const auto mix = [&](std::uint64_t value) {
        hash ^= value;
        hash *= 1099511628211ULL;
    };
    for (const auto &layer : weights.layers) {
        for (const char ch : layer.layer_id) mix(static_cast<unsigned char>(ch));
        for (const auto *matrix : {&layer.wq, &layer.wk, &layer.wv,
                                   &layer.wo, &layer.wu, &layer.wd}) {
            mix(matrix->id); mix(matrix->spec.rows); mix(matrix->spec.columns);
            mix(matrix->spec.has_bias);
        }
        for (const auto &plain : layer.plain_parameters) {
            mix(plain.id); mix(plain.spec.length);
            for (const char ch : plain.spec.name)
                mix(static_cast<unsigned char>(ch));
        }
    }
    return hash;
}

class PlaintextPreparationGuard {
public:
    explicit PlaintextPreparationGuard(SealPlainParameterStore &store)
        : store_(store)
    {
        store_.BeginPlaintextPreparation();
    }

    ~PlaintextPreparationGuard()
    {
        if (!committed_) store_.AbortPlaintextPreparation();
    }

    void Commit()
    {
        store_.FreezePlaintextCache();
        committed_ = true;
    }

private:
    SealPlainParameterStore &store_;
    bool committed_{};
};

} // namespace

SealSciPreLnEntryExecutor::SealSciPreLnEntryExecutor(
    const HeraConfig &config, std::size_t layer_index,
    HeContextRegistry &contexts, RuntimeTensorArena &arena,
    SealPlainParameterStore &parameters, CkksMpcConverter &converter,
    MpcLayerNormRuntime &mpc_layer_norm,
    const CompiledLayerExecutionPlan *plan)
    : config_(config), layer_index_(layer_index), contexts_(contexts),
      arena_(arena), parameters_(parameters), converter_(converter),
      mpc_layer_norm_(mpc_layer_norm), plan_(plan)
{}

Result<PreLnEntryOutputs> SealSciPreLnEntryExecutor::Evaluate(
    const TensorHandle &hidden, const LayerWeightHandles &weights,
    PreLnEntryExecutionReport *report)
{
    const auto started = Clock::now();
    try {
        if (report) *report = {};
        if (layer_index_ >= config_.layers.size())
            return Status::Invalid("Pre-LN layer index is out of range");
        const auto &layer = config_.layers[layer_index_];
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
        if (r2_it->second.parameter_profile !=
            r4_it->second.parameter_profile)
            return Status::FailedPrecondition(
                "Region 2 and Region 4 must share one HE profile");
        const auto &source = contexts_.Require(
            *hidden.spec.he_parameter_profile, contexts_.role());
        const auto &region1 = contexts_.Require(
            r1_it->second.parameter_profile, contexts_.role());
        const auto &shared = contexts_.Require(
            r2_it->second.parameter_profile, contexts_.role());
        const auto region1_terminal =
            region1.parameters.coefficient_modulus_bits.size() -
            region1.parameters.terminal_prime_count;
        constexpr std::size_t kRegion1StatisticsDepth = 3;
        if (region1_terminal < kRegion1StatisticsDepth)
            return Status::FailedPrecondition(
                "Region-1 profile has no three-level statistics suffix");
        const auto region1_start =
            region1_terminal - kRegion1StatisticsDepth;
        if (plan_ && (plan_->layer_id != layer.id ||
                      plan_->statistics_profile != region1.profile ||
                      plan_->attention_profile != shared.profile ||
                      plan_->statistics_input_level != region1_start))
            return Status::FailedPrecondition(
                "Pre-LN compiled plan does not match its HE contexts");

        auto source_plan = AttentionLn2PackingPlanner::RealStrictColumns(
            hidden_shape, source.slot_count());
        if (!source_plan.ok()) return source_plan.status();
        auto source_map = ToConversionMap(source_plan.value());
        if (!source_map.ok()) return source_map.status();
        std::optional<SealCipherBatch> source_batch;
        std::optional<SealCipherTensor> source_tensor;
        if (contexts_.role() == PartyRole::kServer) {
            auto loaded = arena_.he.Get(hidden.id);
            if (!loaded.ok()) return loaded.status();
            source_tensor = loaded.take_value();
            auto valid = source_tensor->Validate(source);
            if (!valid.ok()) return valid.status();
            source_batch = SealCipherBatch{
                source.profile, source_tensor->ciphertexts, hidden_shape,
                Layout::kStrictColumn, source_map.value()};
        }
        auto before = converter_.stats();
        auto source_physical = converter_.CkksToMpc(source_batch);
        if (!source_physical.ok()) return source_physical.status();
        AddConversion(before, converter_.stats(), report, false);
        auto hidden_shares = Compact(source_physical.value(),
                                     source_plan.value(), hidden_shape, true);
        if (!hidden_shares.ok()) return hidden_shares.status();

        // The residual has no HE work before Wo, so create it directly at
        // Wo's output level. This avoids carrying a long, unused RNS prefix
        // and removes the online L0->L6 modulus switch at residual addition.
        const auto shared_terminal =
            shared.parameters.coefficient_modulus_bits.size() -
            shared.parameters.terminal_prime_count;
        if (plan_ && plan_->attention_scores_level != shared_terminal)
            return Status::FailedPrecondition(
                "Pre-LN compiled attention terminal level mismatch");
        constexpr std::size_t kLayerNormStatisticsDepth = 3;
        if (shared_terminal < kLayerNormStatisticsDepth)
            return Status::FailedPrecondition(
                "shared profile has no LayerNorm statistics suffix");
        const auto attention_residual_level =
            shared_terminal - kLayerNormStatisticsDepth;
        auto residual_plan = AttentionLn2PackingPlanner::ComplexStrictColumns(
            hidden_shape, shared.slot_count());
        if (!residual_plan.ok()) return residual_plan.status();
        auto residual_pair = Scatter(hidden_shares.value(),
                                     residual_plan.value());
        if (!residual_pair.ok()) return residual_pair.status();
        auto residual_map = ToConversionMap(residual_plan.value());
        if (!residual_map.ok()) return residual_map.status();
        before = converter_.stats();
        auto residual_he = converter_.MpcToCkks(
            residual_pair.value(),
            {shared.profile, attention_residual_level,
             shared.parameters.input_scale_bits, hidden_shape,
             Layout::kComplexStrictColumn, residual_map.value()});
        if (!residual_he.ok()) return residual_he.status();
        AddConversion(before, converter_.stats(), report, true);
        const auto residual_id = arena_.ids.Next();
        const auto residual_spec = HeSpec(
            hidden_shape, Layout::kComplexStrictColumn, HeRegionId::kRegion4,
            shared, attention_residual_level,
            shared.parameters.input_scale_bits,
            config_.mpc_numeric);
        if (contexts_.role() == PartyRole::kServer) {
            if (!residual_he.value())
                return Status::Internal("server did not receive HE residual");
            SealCipherTensor tensor{shared.profile,
                residual_he.value()->ciphertexts, residual_spec,
                residual_plan.value()};
            auto stored = arena_.he.Put(residual_id, std::move(tensor));
            if (!stored.ok()) return stored.status();
        }

        // Region-1 statistics consume exactly three levels. Start at the
        // terminal suffix so variance naturally reaches the conversion
        // boundary instead of performing a final conversion-only drop.
        auto region1_plan = AttentionLn2PackingPlanner::RealStrictColumns(
            hidden_shape, region1.slot_count());
        if (!region1_plan.ok()) return region1_plan.status();
        std::optional<SealCipherTensor> region1_input;
        const bool direct_region1 = source.profile == region1.profile &&
            hidden.spec.planned_level &&
            *hidden.spec.planned_level <= region1_start &&
            hidden.spec.planned_scale_bits ==
                std::optional<int>{region1.parameters.input_scale_bits};
        if (contexts_.role() == PartyRole::kServer && direct_region1) {
            region1_input = source_tensor;
            auto target = LevelId(region1, region1_start);
            if (!target.ok()) return target.status();
            seal::Evaluator evaluator(*region1.context);
            for (auto &ct : region1_input->ciphertexts)
                if (ct.parms_id() != target.value())
                    evaluator.mod_switch_to_inplace(ct, target.value());
            region1_input->logical_spec.planned_level = region1_start;
        } else if (!direct_region1) {
            auto pair = Scatter(hidden_shares.value(), region1_plan.value());
            if (!pair.ok()) return pair.status();
            auto map = ToConversionMap(region1_plan.value());
            if (!map.ok()) return map.status();
            before = converter_.stats();
            auto refreshed = converter_.MpcToCkks(
                pair.value(), {region1.profile, region1_start,
                    region1.parameters.input_scale_bits, hidden_shape,
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
                           region1.parameters.input_scale_bits,
                           config_.mpc_numeric), region1_plan.value()};
            }
        }

        std::optional<LayerNormStatisticsOutputs> statistics;
        if (contexts_.role() == PartyRole::kServer) {
            auto computed = LayerNormStatisticsKernel::Evaluate(
                *region1_input, layer, region1, nullptr);
            if (!computed.ok()) return computed.status();
            statistics = computed.take_value();
        }

        auto convert_statistic = [&](bool centered)
            -> Result<MpcShareTensor> {
            const TensorShape shape = centered
                ? hidden_shape : TensorShape{{layer.tokens, 1}};
            const auto packing = centered ? region1_plan.value()
                : VariancePacking(layer.tokens, region1.slot_count());
            auto map = ToConversionMap(packing);
            if (!map.ok()) return map.status();
            std::optional<SealCipherBatch> batch;
            if (contexts_.role() == PartyRole::kServer) {
                const auto &tensor = centered ? statistics->centered
                                              : statistics->variance;
                batch = SealCipherBatch{region1.profile, tensor.ciphertexts,
                                        shape, packing.layout, map.value()};
            }
            auto local_before = converter_.stats();
            auto physical = converter_.CkksToMpc(batch);
            if (!physical.ok()) return physical.status();
            AddConversion(local_before, converter_.stats(), report, false);
            return Compact(physical.value(), packing, shape, true);
        };
        auto centered = convert_statistic(true);
        if (!centered.ok()) return centered.status();
        auto variance = convert_statistic(false);
        if (!variance.ok()) return variance.status();
        const auto mpc_started = Clock::now();
        auto inverse = mpc_layer_norm_.ReciprocalSqrt(
            variance.value(), layer.attention_norm_rsqrt_bits, nullptr);
        if (!inverse.ok()) return inverse.status();
        // The LayerNorm runtime is shared by the orchestrator; retaining its
        // Ferret state avoids another base/setup phase at LN2 and next layers.
        if (report)
            report->mpc_milliseconds += std::chrono::duration<double,
                std::milli>(Clock::now() - mpc_started).count();

        auto dense = MultiHeadPackingPlan::DenseSpatial(
            hidden_shape, shared.slot_count());
        if (!dense.ok()) return dense.status();
        auto centered_pair = Scatter(centered.value(), dense.value());
        if (!centered_pair.ok()) return centered_pair.status();
        MpcShareTensor broadcast;
        broadcast.physical_shape = hidden_shape;
        broadcast.numeric = inverse.value().numeric;
        broadcast.component = ComplexComponent::kReal;
        broadcast.local_share.resize(layer.tokens * layer.model_width);
        for (std::size_t token = 0; token < layer.tokens; ++token)
            for (std::size_t feature = 0; feature < layer.model_width; ++feature)
                broadcast.local_share[token * layer.model_width + feature] =
                    inverse.value().local_share[token];
        auto inverse_pair = Scatter(broadcast, dense.value());
        if (!inverse_pair.ok()) return inverse_pair.status();
        auto dense_map = ToConversionMap(dense.value());
        if (!dense_map.ok()) return dense_map.status();
        auto to_block2 = [&](const ComplexMpcSharePair &pair)
            -> Result<std::optional<SealCipherBatch>> {
            auto local_before = converter_.stats();
            auto result = converter_.MpcToCkks(
                pair, {shared.profile, std::size_t{0},
                    shared.parameters.input_scale_bits, hidden_shape,
                    Layout::kSpatial, dense_map.value()});
            if (!result.ok()) return result.status();
            AddConversion(local_before, converter_.stats(), report, true);
            return result;
        };
        auto centered_he = to_block2(centered_pair.value());
        if (!centered_he.ok()) return centered_he.status();
        auto inverse_he = to_block2(inverse_pair.value());
        if (!inverse_he.ok()) return inverse_he.status();
        const auto centered_id = arena_.ids.Next();
        const auto inverse_id = arena_.ids.Next();
        const auto block_spec = HeSpec(
            hidden_shape, Layout::kSpatial, HeRegionId::kRegion2, shared, 0,
            shared.parameters.input_scale_bits, config_.mpc_numeric);
        if (contexts_.role() == PartyRole::kServer) {
            if (!centered_he.value() || !inverse_he.value())
                return Status::Internal("server did not receive Block-2 inputs");
            auto stored = arena_.he.Put(centered_id,
                {shared.profile, centered_he.value()->ciphertexts,
                 block_spec, dense.value()});
            if (!stored.ok()) return stored.status();
            stored = arena_.he.Put(inverse_id,
                {shared.profile, inverse_he.value()->ciphertexts,
                 block_spec, dense.value()});
            if (!stored.ok()) return stored.status();
        }

        const auto score_id = arena_.ids.Next();
        const auto value_id = arena_.ids.Next();
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
            SealBlock2Executor block2(config_, layer, contexts_, arena_.he,
                                      parameters_,
                                      plan_ ? &plan_->block2_rotation_steps
                                            : nullptr);
            auto evaluated = block2.Evaluate(
                {{centered_id, block_spec}, {inverse_id, block_spec}},
                weights, report ? &report->block2 : nullptr);
            if (!evaluated.ok()) return evaluated.status();
            auto scores = arena_.he.Get(
                evaluated.value().scaled_attention_scores.id);
            if (!scores.ok()) return scores.status();
            auto values = arena_.he.Get(evaluated.value().value.id);
            if (!values.ok()) return values.status();
            auto stored = arena_.he.Put(score_id, scores.take_value());
            if (!stored.ok()) return stored.status();
            stored = arena_.he.Put(value_id, values.take_value());
            if (!stored.ok()) return stored.status();
            arena_.he.Erase(evaluated.value().scaled_attention_scores.id);
            arena_.he.Erase(evaluated.value().value.id);
            arena_.he.Erase(centered_id);
            arena_.he.Erase(inverse_id);
        }
        if (report)
            report->he_milliseconds = std::chrono::duration<double,
                std::milli>(Clock::now() - started).count() -
                report->conversion_milliseconds - report->mpc_milliseconds;
        return PreLnEntryOutputs{
            {{score_id, score_spec}, {value_id, value_spec}},
            {residual_id, residual_spec}};
    } catch (const std::exception &error) {
        return Status::Internal(std::string("Pre-LN entry failed: ") +
                                error.what());
    }
}

SealSciTransformerRuntime::SealSciTransformerRuntime(
    const HeraConfig &config, HeContextRegistry &contexts,
    RuntimeTensorArena &arena, SealPlainParameterStore &parameters,
    CkksMpcConverter &converter, MpcSoftmaxRuntime &softmax_mpc,
    MpcLayerNormRuntime &layer_norm_mpc, MpcGeluRuntime &gelu_mpc,
    RuntimeCoordinator *coordinator,
    std::shared_ptr<const CompiledTransformerExecutionPlan> plan)
    : config_(config), contexts_(contexts), arena_(arena),
      parameters_(parameters), converter_(converter),
      softmax_mpc_(softmax_mpc), layer_norm_mpc_(layer_norm_mpc),
      gelu_mpc_(gelu_mpc), coordinator_(coordinator),
      plan_(std::move(plan))
{}

Result<void> SealSciTransformerRuntime::Prepare(
    const TransformerWeightHandles &weights, PreparationReport *report)
{
    const auto started = Clock::now();
    auto supported = ValidateRuntimeConfig(config_);
    if (!supported.ok()) return supported.status();
    if (!plan_) {
        auto compiled = CompiledTransformerExecutionPlan::Compile(config_);
        if (!compiled.ok()) return compiled.status();
        plan_ = std::make_shared<CompiledTransformerExecutionPlan>(
            compiled.take_value());
    }
    if (plan_->layers.size() != config_.layers.size())
        return Status::Invalid("compiled runtime layer count mismatch");
    for (std::size_t i = 0; i < plan_->layers.size(); ++i)
        if (plan_->layers[i].layer_id != config_.layers[i].id)
            return Status::Invalid("compiled runtime layer ID mismatch");
    if (weights.layers.size() != config_.layers.size())
        return Status::Invalid("preparation weight layer count mismatch");
    const auto manifest = WeightManifest(weights);
    if (prepared_ && manifest == prepared_manifest_) {
        if (report) {
            *report = {};
            report->cache_hits = parameters_.encoded_plaintext_count();
            report->virtual_weight_requests =
                parameters_.virtual_weight_requests();
            report->virtual_weight_cache_hits =
                parameters_.virtual_weight_cache_hits();
            report->milliseconds = std::chrono::duration<double, std::milli>(
                Clock::now() - started).count();
        }
        return {};
    }
    // Build into a staging cache. An early return leaves the previous
    // prepared model intact and never enables online model encoding.
    PlaintextPreparationGuard preparation(parameters_);
    if (contexts_.role() == PartyRole::kServer) {
        for (std::size_t i = 0; i < config_.layers.size(); ++i) {
            const auto &layer = config_.layers[i];
            ModelPlaintextCacheScope layer_cache(&parameters_, layer.id);
            const auto *handles = FindLayerWeights(weights, layer.id);
            if (!handles)
                return Status::Invalid(layer.id + ": preparation weights absent");
            const auto *execution = FindLayerExecution(config_, layer.id);
            if (!execution)
                return Status::Invalid(layer.id + ": HE placement absent");
            const auto plain = [&](std::string_view name)
                -> Result<DenseVectorParameter> {
                for (const auto &handle : handles->plain_parameters)
                    if (handle.spec.name == name)
                        return parameters_.GetVector(handle.id);
                return Status::Invalid(std::string(name) + " handle absent");
            };
            const auto matrix = [&](const WeightHandle &handle) {
                return parameters_.GetMatrixShared(handle.id);
            };
            const auto region_context = [&](HeRegionId id)
                -> const HeContext & {
                return contexts_.Require(
                    execution->regions.at(id).parameter_profile,
                    PartyRole::kServer);
            };

            // Block 2: LN1 affine and Q/K/V transformed diagonals.
            const auto &block2_context = region_context(HeRegionId::kRegion2);
            auto dense = MultiHeadPackingPlan::DenseSpatial(
                {{layer.tokens, layer.model_width}},
                block2_context.slot_count());
            if (!dense.ok()) return dense.status();
            TensorSpec dense_spec;
            dense_spec.shape = {{layer.tokens, layer.model_width}};
            dense_spec.numeric = config_.mpc_numeric;
            dense_spec.he_region = HeRegionId::kRegion2;
            auto centered = PreparationTensor(
                block2_context, dense.value(), dense_spec, 0,
                block2_context.parameters.input_scale_bits);
            if (!centered.ok()) return centered.status();
            auto inverse = PreparationTensor(
                block2_context, dense.value(), dense_spec, 0,
                block2_context.parameters.input_scale_bits);
            if (!inverse.ok()) return inverse.status();
            auto gamma1 = plain("ln1.gamma");
            auto beta1 = plain("ln1.beta");
            if (!gamma1.ok()) return gamma1.status();
            if (!beta1.ok()) return beta1.status();
            auto normalized = LayerNormTailKernel::Evaluate(
                centered.value(), inverse.value(), gamma1.value(),
                beta1.value(), layer, block2_context);
            if (!normalized.ok()) return normalized.status();
            const auto block2_terminal =
                block2_context.parameters.coefficient_modulus_bits.size() -
                block2_context.parameters.terminal_prime_count;
            constexpr std::size_t kQkDepth = 5;
            if (block2_terminal <= kQkDepth)
                return Status::FailedPrecondition(
                    "Block-2 profile has no projection level before QK suffix");
            const auto qk_projection_input = block2_terminal - kQkDepth - 1;
            auto qk_normalized = normalized.value();
            auto qk_target = LevelId(block2_context, qk_projection_input);
            if (!qk_target.ok()) return qk_target.status();
            for (auto &ct : qk_normalized.ciphertexts)
                if (ct.parms_id() != qk_target.value())
                    ct.parms_id() = qk_target.value();
            qk_normalized.logical_spec.planned_level = qk_projection_input;
            ComplexDiagonalProjectionKernel qk_projection(
                layer, block2_context);
            ComplexDiagonalProjectionKernel value_projection(
                layer, block2_context);
            for (const auto &entry : std::vector<std::tuple<
                     const WeightHandle *, std::string_view, Layout>>{
                     {&handles->wq, "bq", Layout::kMultiHeadQ},
                     {&handles->wk, "bk", Layout::kMultiHeadK},
                     {&handles->wv, "bv", Layout::kMultiHeadV}}) {
                auto weights_value = matrix(*std::get<0>(entry));
                if (!weights_value.ok()) return weights_value.status();
                std::optional<DenseVectorParameter> bias;
                if (layer.qkv_bias) {
                    auto loaded = plain(std::get<1>(entry));
                    if (!loaded.ok()) return loaded.status();
                    bias = loaded.take_value();
                }
                const auto layout = std::get<2>(entry);
                auto compiled = layout == Layout::kMultiHeadV
                    ? value_projection.Evaluate(
                        normalized.value(), *weights_value.value(), bias,
                        layout)
                    : qk_projection.Evaluate(
                        qk_normalized, *weights_value.value(), bias, layout);
                if (!compiled.ok()) return compiled.status();
            }

            // Attention output Wo/bo starts from the A*V recovery packing at
            // its planned level 5.
            const auto &shared = region_context(HeRegionId::kRegion4);
            auto av = AttentionLn2PackingPlanner::AttentionValue(
                layer, shared.slot_count());
            if (!av.ok()) return av.status();
            TensorSpec attention_spec;
            attention_spec.shape = av.value().output_packing.logical_shape;
            attention_spec.numeric = config_.mpc_numeric;
            attention_spec.he_region = HeRegionId::kRegion4;
            auto attention = PreparationTensor(
                shared, av.value().output_packing, attention_spec, 5,
                shared.parameters.default_scale_bits);
            if (!attention.ok()) return attention.status();
            auto wo = matrix(handles->wo);
            if (!wo.ok()) return wo.status();
            std::optional<DenseVectorParameter> bo;
            if (layer.output_bias) {
                auto loaded = plain("bo");
                if (!loaded.ok()) return loaded.status();
                bo = loaded.take_value();
            }
            auto wo_compiled = AttentionOutputProjectionKernel::Evaluate(
                attention.value(), *wo.value(), bo, layer, shared);
            if (!wo_compiled.ok()) return wo_compiled.status();

            // Region 5 LN2 affine and Wu/bu.
            const auto &ffn_context = region_context(HeRegionId::kRegion5);
            auto ffn_plan = FfnGeluPackingPlanner::Build(
                layer, ffn_context.slot_count());
            if (!ffn_plan.ok()) return ffn_plan.status();
            TensorSpec ffn_centered_spec;
            ffn_centered_spec.shape = {{layer.tokens, layer.model_width}};
            ffn_centered_spec.numeric = config_.mpc_numeric;
            ffn_centered_spec.he_region = HeRegionId::kRegion5;
            const auto ffn_terminal =
                ffn_context.parameters.coefficient_modulus_bits.size() -
                ffn_context.parameters.terminal_prime_count;
            constexpr std::size_t kFfnGeluDepth = 6;
            if (ffn_terminal < kFfnGeluDepth)
                return Status::FailedPrecondition(
                    "FFN preparation profile has no six-level suffix");
            const auto ffn_start = ffn_terminal - kFfnGeluDepth;
            auto ffn_centered = PreparationTensor(
                ffn_context, ffn_plan.value().centered,
                ffn_centered_spec, ffn_start,
                ffn_context.parameters.input_scale_bits);
            if (!ffn_centered.ok()) return ffn_centered.status();
            TensorSpec inverse_spec;
            inverse_spec.shape = {{layer.tokens, 1}};
            inverse_spec.numeric = config_.mpc_numeric;
            inverse_spec.he_region = HeRegionId::kRegion5;
            auto ffn_inverse = PreparationTensor(
                ffn_context, ffn_plan.value().inverse_broadcast,
                inverse_spec, ffn_start,
                ffn_context.parameters.input_scale_bits);
            if (!ffn_inverse.ok()) return ffn_inverse.status();
            auto gamma2 = plain("ln2.gamma");
            auto beta2 = plain("ln2.beta");
            if (!gamma2.ok()) return gamma2.status();
            if (!beta2.ok()) return beta2.status();
            auto ffn_normalized = FfnLayerNormTailKernel::Evaluate(
                ffn_centered.value(), ffn_inverse.value(), gamma2.value(),
                beta2.value(), layer, ffn_context);
            if (!ffn_normalized.ok()) return ffn_normalized.status();
            auto wu = matrix(handles->wu);
            if (!wu.ok()) return wu.status();
            std::optional<DenseVectorParameter> bu;
            if (layer.ffn_bias) {
                auto loaded = plain("bu");
                if (!loaded.ok()) return loaded.status();
                bu = loaded.take_value();
            }
            auto wu_compiled = FfnUpProjectionKernel::Evaluate(
                ffn_normalized.value(), *wu.value(), bu, layer, ffn_context);
            if (!wu_compiled.ok()) return wu_compiled.status();

            // Wd/bd use the terminal suffix for cross-layer conversion; the
            // final layer retains the conventional L5->L6->L7 HE output.
            auto activation_plan = FfnGeluPackingPlanner::ComplexColumns(
                {{layer.tokens, layer.ffn_width}}, shared.slot_count(),
                Layout::kFfn);
            auto output_plan = FfnGeluPackingPlanner::ComplexColumns(
                {{layer.tokens, layer.model_width}}, shared.slot_count(),
                Layout::kComplexStrictColumn);
            if (!activation_plan.ok()) return activation_plan.status();
            if (!output_plan.ok()) return output_plan.status();
            TensorSpec activation_spec;
            activation_spec.shape = {{layer.tokens, layer.ffn_width}};
            activation_spec.numeric = config_.mpc_numeric;
            activation_spec.he_region = HeRegionId::kRegion4;
            const auto shared_terminal =
                shared.parameters.coefficient_modulus_bits.size() -
                shared.parameters.terminal_prime_count;
            const auto wd_input_level = i + 1 < config_.layers.size()
                ? shared_terminal - 2 : std::size_t{5};
            auto activation = PreparationTensor(
                shared, activation_plan.value(), activation_spec,
                wd_input_level,
                shared.parameters.default_scale_bits);
            if (!activation.ok()) return activation.status();
            auto wd = matrix(handles->wd);
            if (!wd.ok()) return wd.status();
            std::optional<DenseVectorParameter> bd;
            if (layer.ffn_bias) {
                auto loaded = plain("bd");
                if (!loaded.ok()) return loaded.status();
                bd = loaded.take_value();
            }
            auto wd_compiled = ComplexColumnProjectionKernel::Evaluate(
                activation.value(), *wd.value(), bd, output_plan.value(),
                HeRegionId::kRegion4,
                shared.parameters.default_scale_bits, shared);
            if (!wd_compiled.ok()) return wd_compiled.status();
        }
    }
    preparation.Commit();
    prepared_manifest_ = manifest;
    prepared_ = true;
    if (report) {
        *report = {};
        report->encoded_plaintext_count =
            parameters_.encoded_plaintext_count();
        report->encoded_bytes = parameters_.encoded_plaintext_bytes();
        report->cache_hits = parameters_.plaintext_cache_hits();
        report->virtual_weight_requests =
            parameters_.virtual_weight_requests();
        report->virtual_weight_cache_hits =
            parameters_.virtual_weight_cache_hits();
        report->milliseconds = std::chrono::duration<double, std::milli>(
            Clock::now() - started).count();
    }
    return {};
}

Result<TensorHandle> SealSciTransformerRuntime::Evaluate(
    const TensorHandle &input, const TransformerWeightHandles &weights,
    const PublicAttentionMask &mask, ExecutionReport *report)
{
    auto supported = ValidateRuntimeConfig(config_);
    if (!supported.ok()) {
        if (coordinator_) coordinator_->Abort(supported.status(), 0, 0);
        return supported.status();
    }
    if (!prepared_) {
        auto status = Status::FailedPrecondition(
            "Prepare() must complete before online inference");
        if (coordinator_) coordinator_->Abort(status, 0, 0);
        return status;
    }
    if (prepared_manifest_ != WeightManifest(weights)) {
        auto status = Status::FailedPrecondition(
            "online weight manifest differs from prepared model");
        if (coordinator_) coordinator_->Abort(status, 0, 0);
        return status;
    }
    if (report) {
        report->nodes.clear();
        report->stages.clear();
        report->online_milliseconds = 0.0;
        report->communication_bytes = 0;
        report->rounds = 0;
    }
    const auto online_started = Clock::now();
    TensorHandle hidden = input;
    for (std::size_t index = 0; index < config_.layers.size(); ++index) {
        const auto &layer = config_.layers[index];
        ModelPlaintextCacheScope layer_cache(&parameters_, layer.id);
        const auto *layer_weights = FindLayerWeights(weights, layer.id);
        if (!layer_weights) {
            auto status = Status::Invalid(
                layer.id + ": runtime weights are absent");
            if (coordinator_) coordinator_->Abort(status, index, 0);
            return status;
        }

        auto started = Clock::now();
        PreLnEntryExecutionReport entry_report;
        SealSciPreLnEntryExecutor entry(
            config_, index, contexts_, arena_, parameters_, converter_,
            layer_norm_mpc_, &plan_->layers[index]);
        auto block2 = entry.Evaluate(hidden, *layer_weights, &entry_report);
        if (!block2.ok()) {
            AppendStage(report, index, "pre_ln_block2", started,
                        block2.status());
            if (coordinator_) coordinator_->Abort(block2.status(), index, 0);
            return block2.status();
        }
        // Once Pre-LN has refreshed a previous-layer result, that ciphertext
        // is dead. Preserve only the caller-owned network input for repeats.
        if (hidden.id != input.id) arena_.he.Erase(hidden.id);
        AppendStage(report, index, "pre_ln_block2", started, Status::Ok(),
                    Metrics(entry_report, layer.tokens));
        if (coordinator_) {
            auto synced = coordinator_->StageComplete(index, 0);
            if (!synced.ok()) return synced.status();
        }

        started = Clock::now();
        SoftmaxExecutionReport softmax_report;
        SealSciSoftmaxExecutor softmax(
            config_, layer, contexts_, arena_.he, arena_.mpc, converter_,
            softmax_mpc_, &arena_.ids);
        auto probabilities = softmax.Evaluate(
            {block2.value().block2.scaled_attention_scores, mask},
            &softmax_report);
        if (!probabilities.ok()) {
            AppendStage(report, index, "softmax", started,
                        probabilities.status());
            if (coordinator_) coordinator_->Abort(probabilities.status(), index, 1);
            return probabilities.status();
        }
        arena_.he.Erase(block2.value().block2.scaled_attention_scores.id);
        AppendStage(report, index, "softmax", started, Status::Ok(),
                    Metrics(softmax_report));
        if (coordinator_) {
            auto synced = coordinator_->StageComplete(index, 1);
            if (!synced.ok()) return synced.status();
        }

        started = Clock::now();
        AttentionLn2ExecutionReport attention_report;
        SealSciAttentionLn2Executor attention(
            config_, layer, contexts_, arena_.he, parameters_, arena_.mpc,
            converter_, layer_norm_mpc_, &arena_.ids);
        auto ln2 = attention.Evaluate(
            {probabilities.value().probability_shares,
             block2.value().block2.value,
             block2.value().attention_residual},
            *layer_weights, &attention_report);
        if (!ln2.ok()) {
            AppendStage(report, index, "attention_ln2", started,
                        ln2.status());
            if (coordinator_) coordinator_->Abort(ln2.status(), index, 2);
            return ln2.status();
        }
        arena_.mpc.Erase(probabilities.value().probability_shares.id);
        arena_.he.Erase(block2.value().block2.value.id);
        arena_.he.Erase(block2.value().attention_residual.id);
        AppendStage(report, index, "attention_ln2", started, Status::Ok(),
                    Metrics(attention_report));
        if (coordinator_) {
            auto synced = coordinator_->StageComplete(index, 2);
            if (!synced.ok()) return synced.status();
        }

        started = Clock::now();
        FfnGeluExecutionReport ffn_report;
        SealSciFfnGeluExecutor ffn(
            config_, layer, contexts_, parameters_, arena_.mpc, converter_,
            gelu_mpc_, &arena_.ids);
        auto activation = ffn.Evaluate(
            {ln2.value().centered_shares,
             ln2.value().inverse_stddev_shares},
            *layer_weights, &ffn_report);
        if (!activation.ok()) {
            AppendStage(report, index, "ffn_gelu", started,
                        activation.status());
            if (coordinator_) coordinator_->Abort(activation.status(), index, 3);
            return activation.status();
        }
        arena_.mpc.Erase(ln2.value().centered_shares.id);
        arena_.mpc.Erase(ln2.value().inverse_stddev_shares.id);
        AppendStage(report, index, "ffn_gelu", started, Status::Ok(),
                    Metrics(ffn_report));
        if (coordinator_) {
            auto synced = coordinator_->StageComplete(index, 3);
            if (!synced.ok()) return synced.status();
        }

        started = Clock::now();
        Block5BridgeExecutionReport bridge_report;
        SealSciBlock5BridgeExecutor bridge(
            config_, index, contexts_, arena_.he, parameters_, arena_.mpc,
            converter_, layer_norm_mpc_, &arena_.ids);
        auto closed = bridge.Evaluate(
            {activation.value().activation_shares,
             layer.ffn_residual
                ? std::optional<TensorHandle>{ln2.value().ffn_residual}
                : std::nullopt},
            *layer_weights, std::nullopt, &bridge_report);
        if (!closed.ok()) {
            AppendStage(report, index, "block5_bridge", started,
                        closed.status());
            if (coordinator_) coordinator_->Abort(closed.status(), index, 4);
            return closed.status();
        }
        arena_.mpc.Erase(activation.value().activation_shares.id);
        arena_.he.Erase(ln2.value().ffn_residual.id);
        hidden = closed.value().hidden_states;
        AppendStage(report, index, "block5_bridge", started, Status::Ok(),
                    Metrics(bridge_report));
        if (coordinator_) {
            auto synced = coordinator_->StageComplete(index, 4);
            if (!synced.ok()) return synced.status();
        }

        if (report) {
            report->communication_bytes +=
                entry_report.conversion_communication_bytes +
                softmax_report.conversion_communication_bytes +
                softmax_report.mpc_communication_bytes +
                attention_report.conversion_communication_bytes +
                attention_report.mpc_communication_bytes +
                ffn_report.conversion_communication_bytes +
                ffn_report.mpc_communication_bytes +
                bridge_report.conversion_communication_bytes;
            report->rounds += entry_report.conversion_rounds +
                softmax_report.conversion_rounds + softmax_report.mpc_rounds +
                attention_report.conversion_rounds + attention_report.mpc_rounds +
                ffn_report.conversion_rounds + ffn_report.mpc_rounds +
                bridge_report.conversion_rounds;
        }
    }
    if (report)
        report->online_milliseconds = std::chrono::duration<double,
            std::milli>(Clock::now() - online_started).count();
    return hidden;
}

Result<std::map<std::string, std::vector<int>>>
SealSciTransformerRuntime::RequiredRotationSteps(const HeraConfig &config)
{
    auto plan = CompiledTransformerExecutionPlan::Compile(config);
    if (!plan.ok()) return plan.status();
    return plan.take_value().rotation_steps_by_profile;
}

Result<TransformerWeightHandles> RegisterTransformerWeights(
    const HeraConfig &config,
    const std::optional<TransformerWeightData> &server_data,
    SealPlainParameterStore &parameters)
{
    if (server_data && server_data->layers.size() != config.layers.size())
        return Status::Invalid("weight data layer count mismatch");
    TransformerWeightHandles result;
    WeightId next = 1;
    for (std::size_t i = 0; i < config.layers.size(); ++i) {
        const auto &layer = config.layers[i];
        auto schema = DeriveWeightSchema(layer);
        if (!schema.ok()) return schema.status();
        LayerWeightHandles handles{
            layer.id,
            {next++, schema.value().wq}, {next++, schema.value().wk},
            {next++, schema.value().wv}, {next++, schema.value().wo},
            {next++, schema.value().wu}, {next++, schema.value().wd}, {}};
        for (const auto &vector : schema.value().vectors)
            handles.plain_parameters.push_back({next++, vector});
        if (server_data) {
            const auto &data = server_data->layers[i];
            const std::pair<const WeightHandle *, const std::vector<double> *>
                matrices[] = {{&handles.wq, &data.wq},
                              {&handles.wk, &data.wk},
                              {&handles.wv, &data.wv},
                              {&handles.wo, &data.wo},
                              {&handles.wu, &data.wu},
                              {&handles.wd, &data.wd}};
            for (const auto &matrix : matrices) {
                auto stored = parameters.PutMatrix(
                    *matrix.first, *matrix.second);
                if (!stored.ok()) return stored.status();
            }
            for (const auto &handle : handles.plain_parameters) {
                const auto found = data.vectors.find(handle.spec.name);
                if (found == data.vectors.end())
                    return Status::Invalid(layer.id + "." +
                                           handle.spec.name + " is absent");
                auto stored = parameters.PutVector(handle, found->second);
                if (!stored.ok()) return stored.status();
            }
        }
        result.layers.push_back(std::move(handles));
    }
    return result;
}

TransformerWeightData MakeDeterministicTransformerWeights(
    const HeraConfig &config, std::uint64_t seed)
{
    auto next_value = [&seed](double magnitude) {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        const auto centered = static_cast<double>(seed % 2001) / 1000.0 - 1.0;
        return centered * magnitude;
    };
    TransformerWeightData result;
    for (const auto &layer : config.layers) {
        auto schema = DeriveWeightSchema(layer).value();
        LayerWeightData data;
        const auto fill_matrix = [&](const MatrixSpec &spec,
                                     std::vector<double> &values) {
            values.resize(spec.rows * spec.columns);
            const auto magnitude = 0.08 /
                std::sqrt(static_cast<double>(std::max<std::size_t>(1,
                                                                    spec.rows)));
            for (auto &value : values) value = next_value(magnitude);
        };
        fill_matrix(schema.wq, data.wq);
        fill_matrix(schema.wk, data.wk);
        fill_matrix(schema.wv, data.wv);
        fill_matrix(schema.wo, data.wo);
        fill_matrix(schema.wu, data.wu);
        fill_matrix(schema.wd, data.wd);
        for (const auto &vector : schema.vectors) {
            auto &values = data.vectors[vector.name];
            values.resize(vector.length);
            if (vector.name.find("gamma") != std::string::npos)
                std::fill(values.begin(), values.end(), 1.0);
            else if (vector.name.find("beta") != std::string::npos)
                std::fill(values.begin(), values.end(), 0.0);
            else
                for (auto &value : values) value = next_value(0.005);
        }
        result.layers.push_back(std::move(data));
    }
    return result;
}

} // namespace hera
