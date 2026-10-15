#include "hera/private_runtime.h"
#include "hera/detail/he_level.h"
#include "hera/detail/level_schedule.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <set>

namespace hera {
namespace {

using Clock = std::chrono::steady_clock;

using detail::LevelId;

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
    report->communication_bytes += metrics.communication_bytes;
    report->rounds += metrics.rounds;
    report->stages.push_back({layer, std::move(name),
        std::chrono::duration<double, std::milli>(Clock::now() - started).count(),
        status, std::move(metrics)});
}

ExecutionReport::StageMetrics Metrics(const PreLnEntryExecutionReport &value,
                                      std::size_t tokens)
{
    return {value.conversion_communication_bytes + value.statistics.mpc_communication_bytes,
            value.conversion_rounds + value.statistics.mpc_rounds,
            value.ckks_to_mpc_calls + value.mpc_to_ckks_calls,
            value.block2.rotations + value.statistics.rotations,
            value.block2.ciphertext_plain_multiplications + value.statistics.ciphertext_plain_multiplications,
            value.block2.ciphertext_multiplications + value.statistics.ciphertext_multiplications,
            value.block2.relinearizations + value.statistics.relinearizations,
            value.block2.rescales + value.statistics.rescales, tokens,
            value.statistics.layer_norm_workers, value.statistics.reciprocal_sqrt_workers};
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
            value.rescales, value.reciprocal_square_roots,
            value.layer_norm_workers, value.reciprocal_sqrt_workers};
}

ExecutionReport::StageMetrics Metrics(const FfnGeluExecutionReport &value)
{
    return {value.conversion_communication_bytes +
                value.mpc_communication_bytes,
            value.conversion_rounds + value.mpc_rounds,
            value.ckks_to_mpc_calls + value.mpc_to_ckks_calls,
            value.rotations, value.ciphertext_plain_multiplications,
            value.ciphertext_multiplications, value.relinearizations,
            value.rescales, value.comparisons + value.muxes, 0, 0, value.mpc_workers};
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
        if (layer.norm == NormPlacement::kPostNorm)
            return Status::KernelUnavailable(
                "real HERA runtime currently supports pre or normalized_residual, not post");
        if (layer.activation != ActivationKind::kGelu)
            return Status::KernelUnavailable(
                "real HERA runtime currently supports GeLU only");
    }
    return {};
}

// A Prepare-only contract carrier: no encryption, ciphertext coefficient
// allocation, or evaluator work is needed to compile model plaintexts.
Result<SealCipherTensor> PreparationTensor(
    const HeContext &context, const HePackingPlan &packing,
    TensorSpec spec, std::size_t level, int scale_bits)
{
    auto valid = packing.Validate();
    if (!valid.ok()) return valid.status();
    auto parms = LevelId(context, level);
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

            // Block 2: dynamic normalization and pre-fused Q/K/V diagonals.
            const auto &block2_context = region_context(HeRegionId::kRegion2);
            const auto &statistics_context = region_context(HeRegionId::kRegion1);
            const auto statistics_terminal = statistics_context.parameters.coefficient_modulus_bits.size() -
                statistics_context.parameters.terminal_prime_count;
            if (statistics_terminal < 3)
                return Status::FailedPrecondition("LN1 preparation needs three statistics levels");
            auto statistics_plan = AttentionLn2PackingPlanner::RealStrictColumns(
                {{layer.tokens, layer.model_width}}, statistics_context.slot_count());
            if (!statistics_plan.ok()) return statistics_plan.status();
            TensorSpec statistics_spec;
            statistics_spec.shape = {{layer.tokens, layer.model_width}};
            statistics_spec.numeric = config_.mpc_numeric;
            statistics_spec.he_region = HeRegionId::kRegion1;
            auto statistics_input = PreparationTensor(statistics_context, statistics_plan.value(),
                statistics_spec, statistics_terminal - 3, plan_->layers[i].statistics_input_scale_bits);
            if (!statistics_input.ok()) return statistics_input.status();
            auto statistics_prepared = LayerNormStatisticsKernel::Prepare(
                statistics_input.value(), layer, statistics_context, nullptr, plan_->layers[i].statistics_input_scale_bits);
            if (!statistics_prepared.ok()) return statistics_prepared.status();
            auto dense = MultiHeadPackingPlan::RealStrictColumns(
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
            auto broadcast = MultiHeadPackingPlan::TokenInverseBroadcast(layer.tokens, block2_context.slot_count());
            if (!broadcast.ok()) return broadcast.status();
            auto ln1_inverse_spec = dense_spec;
            ln1_inverse_spec.shape = {{layer.tokens, 1}};
            auto inverse = PreparationTensor(
                block2_context, broadcast.value(), ln1_inverse_spec, 0,
                block2_context.parameters.input_scale_bits);
            if (!inverse.ok()) return inverse.status();
            auto affine = parameters_.GetGammaFusedLayerNorm(handles->wq.id);
            if (!affine.ok()) return affine.status();
            std::optional<GammaFusedLayerNormParameters> original_ln1;
            SealCipherTensor ln1_residual;
            if (layer.norm == NormPlacement::kNormalizedResidual && layer.attention_residual) {
                auto gamma = plain("ln1.gamma"), beta = plain("ln1.beta");
                if (!gamma.ok()) return gamma.status();
                if (!beta.ok()) return beta.status();
                original_ln1 = GammaFusedLayerNormParameters{gamma.take_value(), beta.take_value()};
            }
            auto normalized = LayerNormTailKernel::Evaluate(
                centered.value(), inverse.value(), *affine.value(), layer, block2_context,
                nullptr, original_ln1 ? &*original_ln1 : nullptr,
                original_ln1 ? &ln1_residual : nullptr);
            if (!normalized.ok()) return normalized.status();
            const auto block2_terminal =
                block2_context.parameters.coefficient_modulus_bits.size() -
                block2_context.parameters.terminal_prime_count;
            constexpr auto kQkDepth = MultiHeadQkKernel::kDepth;
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
            ComplexDiagonalProjectionKernel projection(layer, block2_context);
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
                const auto &source = layout == Layout::kMultiHeadV
                    ? normalized.value() : qk_normalized;
                auto compiled = projection.Evaluate(
                    source, *weights_value.value(), bias, layout);
                if (!compiled.ok()) return compiled.status();
            }

            auto qk_compiled = MultiHeadQkKernel::Prepare(layer, block2_context);
            if (!qk_compiled.ok()) return qk_compiled.status();

            auto av_compiled = MultiHeadAttentionValueKernel::Prepare(
                layer, block2_context, plan_->layers[i].attention_value_input_level);
            if (!av_compiled.ok()) return av_compiled.status();

            auto softmax_compiled = SealSciSoftmaxExecutor::Prepare(
                layer, block2_context, region_context(HeRegionId::kRegion3), config_.softmax);
            if (!softmax_compiled.ok()) return softmax_compiled.status();

            // Compile Wo at the refreshed Region-4 level when profiles differ.
            const auto &shared = region_context(HeRegionId::kRegion4);
            const auto wo_input_level =
                plan_->layers[i].attention_output_input_level;
            auto av = AttentionLn2PackingPlanner::AttentionValue(
                layer, shared.slot_count());
            if (!av.ok()) return av.status();
            TensorSpec attention_spec;
            attention_spec.shape = av.value().output_packing.logical_shape;
            attention_spec.numeric = config_.mpc_numeric;
            attention_spec.he_region = HeRegionId::kRegion4;
            auto attention = PreparationTensor(
                shared, av.value().output_packing, attention_spec, wo_input_level,
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
            auto wo_compiled = AttentionOutputProjectionKernel::EvaluateReal(
                attention.value(), *wo.value(), bo, layer, shared);
            if (!wo_compiled.ok()) return wo_compiled.status();
            statistics_prepared = LayerNormStatisticsKernel::Prepare(
                wo_compiled.value(), layer, shared);
            if (!statistics_prepared.ok()) return statistics_prepared.status();

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
            const auto kFfnGeluDepth = kFfnSuffixDepth;
            if (ffn_terminal < kFfnGeluDepth)
                return Status::FailedPrecondition(
                    "FFN preparation profile has no five-level suffix");
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
            auto ln2_affine = parameters_.GetGammaFusedLayerNorm(handles->wu.id);
            if (!ln2_affine.ok()) return ln2_affine.status();
            std::optional<GammaFusedLayerNormParameters> original_ln2;
            SealCipherTensor ln2_residual;
            if (layer.norm == NormPlacement::kNormalizedResidual && layer.ffn_residual) {
                auto gamma = plain("ln2.gamma"), beta = plain("ln2.beta");
                if (!gamma.ok()) return gamma.status();
                if (!beta.ok()) return beta.status();
                original_ln2 = GammaFusedLayerNormParameters{gamma.take_value(), beta.take_value()};
            }
            auto ffn_normalized = FfnLayerNormTailKernel::Evaluate(
                ffn_centered.value(), ffn_inverse.value(), ln2_affine.value()->gamma,
                ln2_affine.value()->beta, layer, ffn_context,
                nullptr, config_.threads, original_ln2 ? &*original_ln2 : nullptr,
                original_ln2 ? &ln2_residual : nullptr);
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
            auto gelu_compiled = GeluPolynomialKernel::Evaluate(
                wu_compiled.value(), config_.gelu, ffn_context);
            if (!gelu_compiled.ok()) return gelu_compiled.status();

            // A shared R1/R5 profile retains complete LN2 directly for Wd.
            const bool shared_skip = layer.norm == NormPlacement::kNormalizedResidual &&
                statistics_context.profile == ffn_context.profile;
            const auto &wd_context = shared_skip ? ffn_context : shared;
            const int wd_scale_bits = shared_skip ? wd_context.parameters.output_scale_bits : wd_context.parameters.default_scale_bits;
            const auto wd_region = shared_skip ? HeRegionId::kRegion5 : HeRegionId::kRegion4;
            auto activation_plan = FfnGeluPackingPlanner::ComplexColumns(
                {{layer.tokens, layer.ffn_width}}, wd_context.slot_count(),
                Layout::kFfn);
            auto output_plan = FfnGeluPackingPlanner::RealColumns(
                {{layer.tokens, layer.model_width}}, wd_context.slot_count(),
                Layout::kStrictColumn);
            if (!activation_plan.ok()) return activation_plan.status();
            if (!output_plan.ok()) return output_plan.status();
            TensorSpec activation_spec;
            activation_spec.shape = {{layer.tokens, layer.ffn_width}};
            activation_spec.numeric = config_.mpc_numeric;
            activation_spec.he_region = wd_region;
            const auto wd_input_level = plan_->layers[i].ffn_output_level - 1;
            auto activation = PreparationTensor(
                wd_context, activation_plan.value(), activation_spec,
                wd_input_level,
                wd_scale_bits);
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
                wd_region,
                wd_scale_bits, wd_context);
            if (!wd_compiled.ok()) return wd_compiled.status();
        }
    }
    if (contexts_.role() == PartyRole::kClient) {
        // The client uses the same public conversion maps, but never evaluates
        // the HE exponent and therefore needs no encoded exponent plaintexts.
        for (const auto &layer : config_.layers) {
            ModelPlaintextCacheScope layer_cache(&parameters_, layer.id);
            const auto *execution = FindLayerExecution(config_, layer.id);
            if (!execution) return Status::Invalid(layer.id + ": HE placement absent");
            const auto &scores = contexts_.Require(
                execution->regions.at(HeRegionId::kRegion2).parameter_profile,
                PartyRole::kClient);
            const auto &exponent = contexts_.Require(
                execution->regions.at(HeRegionId::kRegion3).parameter_profile,
                PartyRole::kClient);
            auto compiled = SealSciSoftmaxExecutor::Prepare(
                layer, scores, exponent, config_.softmax, false);
            if (!compiled.ok()) return compiled.status();
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
            gelu_mpc_, &arena_.ids, &arena_.he);
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
                ? (layer.norm == NormPlacement::kNormalizedResidual
                    ? (activation.value().normalized_residual_he
                        ? activation.value().normalized_residual_he
                        : activation.value().normalized_residual_shares)
                    : std::optional<TensorHandle>{ln2.value().ffn_residual})
                : std::nullopt},
            *layer_weights, std::nullopt, &bridge_report);
        if (!closed.ok()) {
            AppendStage(report, index, "block5_bridge", started,
                        closed.status());
            if (coordinator_) coordinator_->Abort(closed.status(), index, 4);
            return closed.status();
        }
        arena_.mpc.Erase(activation.value().activation_shares.id);
        if (activation.value().normalized_residual_shares)
            arena_.mpc.Erase(activation.value().normalized_residual_shares->id);
        if (activation.value().normalized_residual_he)
            arena_.he.Erase(activation.value().normalized_residual_he->id);
        arena_.he.Erase(ln2.value().ffn_residual.id);
        hidden = closed.value().hidden_states;
        AppendStage(report, index, "block5_bridge", started, Status::Ok(),
                    Metrics(bridge_report));
        if (coordinator_) {
            auto synced = coordinator_->StageComplete(index, 4);
            if (!synced.ok()) return synced.status();
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
            auto fused = parameters.FuseLayerNormGammaIntoQkv(handles);
            if (!fused.ok()) return fused.status();
            fused = parameters.FuseLayerNormGammaIntoWu(handles);
            if (!fused.ok()) return fused.status();
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
