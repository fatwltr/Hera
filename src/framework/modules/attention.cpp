#include "hera/modules.h"
#include "hera/detail/level_schedule.h"

#include "module_utils.h"

#include <cmath>

namespace hera {

Result<AttentionProjectionOutputs>
AttentionGraphModule::BuildProjectionsAndScores(
    GraphBuilder &graph, TensorId input,
    const TransformerLayerConfig &layer, HeRegionId region,
    const std::string &prefix)
{
    try {
        using namespace module_internal;
        const auto source = EnsureRegion(graph, input, region, Layout::kStrictColumn,
                                         prefix + ".input");
        const auto &base = graph.Spec(source);
        const TensorShape q_shape{{layer.tokens, layer.q_heads,
                                   layer.qk_head_width}};
        const TensorShape k_shape{{layer.tokens, layer.kv_heads,
                                   layer.qk_head_width}};
        const TensorShape v_shape{{layer.tokens, layer.kv_heads,
                                   layer.value_head_width}};
        auto q_spec = HeSpec(base, q_shape, Layout::kMultiHeadQ, region);
        auto k_spec = HeSpec(base, k_shape, Layout::kMultiHeadK, region);
        auto v_spec = HeSpec(base, v_shape, Layout::kMultiHeadV, region);
        const auto &parameters = graph.ActiveParameters(region);
        const auto terminal_level =
            parameters.coefficient_modulus_bits.size() -
            parameters.terminal_prime_count;
        constexpr std::size_t kQkDepth = 4;
        if (terminal_level < kQkDepth)
            return Status::FailedPrecondition(
                prefix + ": QK suffix is shorter than four levels");
        const auto qk_projection_level = terminal_level - kQkDepth;
        const auto value_projection_level =
            base.planned_level.value_or(1) + 1;
        const auto output_scale =
            graph.ActiveParameters(region).default_scale_bits;
        q_spec.planned_level = qk_projection_level;
        k_spec.planned_level = qk_projection_level;
        v_spec.planned_level = value_projection_level;
        q_spec.planned_scale_bits = output_scale;
        k_spec.planned_scale_bits = output_scale;
        v_spec.planned_scale_bits = output_scale;
        const auto q = graph.AddNode(
            prefix + ".Wq", OpKind::kMatMulCipherPlain, {source}, q_spec,
            MatMulAttributes{layer.id, "Wq", q_shape, Layout::kMultiHeadQ,
                             region, false,
                             HeKernelKind::kComplexDiagonalProjection});
        const auto k = graph.AddNode(
            prefix + ".Wk", OpKind::kMatMulCipherPlain, {source}, k_spec,
            MatMulAttributes{layer.id, "Wk", k_shape, Layout::kMultiHeadK,
                             region, false,
                             HeKernelKind::kComplexDiagonalProjection});
        const auto v = graph.AddNode(
            prefix + ".Wv", OpKind::kMatMulCipherPlain, {source}, v_spec,
            MatMulAttributes{layer.id, "Wv", v_shape, Layout::kMultiHeadV,
                             region, layer.q_heads != layer.kv_heads,
                             HeKernelKind::kComplexDiagonalProjection});
        const TensorShape scores_shape{{layer.q_heads, layer.tokens,
                                        layer.tokens}};
        auto scores_spec = HeSpec(base, scores_shape,
                                  Layout::kAttentionDiagonal, region);
        scores_spec.planned_level = terminal_level;
        scores_spec.planned_scale_bits =
            graph.ActiveParameters(region).output_scale_bits;
        const auto scores = graph.AddNode(
            prefix + ".qk", OpKind::kMatMulCipherCipher, {q, k}, scores_spec,
            MatMulAttributes{layer.id, "", scores_shape,
                             Layout::kAttentionDiagonal, region,
                             layer.q_heads != layer.kv_heads,
                             HeKernelKind::kMultiHeadQk});
        // 1/sqrt(head_width) is folded into Wq and bq by the HE plaintext
        // planner, avoiding an online multiplication level.
        return AttentionProjectionOutputs{q, k, v, scores};
    } catch (const std::exception &error) {
        return Status::Internal(prefix + ": " + error.what());
    }
}

Result<AttentionOutputGraphTensors>
AttentionGraphModule::BuildAttentionValueAndOutput(
    GraphBuilder &graph, TensorId probabilities, TensorId value,
    TensorId residual, const TransformerLayerConfig &layer,
    HeRegionId attention_region, HeRegionId output_region,
    const std::string &prefix)
{
    try {
        using namespace module_internal;
        const auto &probability_spec = graph.Spec(probabilities);
        const auto &attention_parameters = graph.ActiveParameters(attention_region);
        if (attention_parameters.coefficient_modulus_bits.size() <
            attention_parameters.terminal_prime_count + detail::kAttentionValueDepth)
            return Status::FailedPrecondition(prefix + ": AV needs three remaining levels");
        const auto av_input_level = detail::AttentionValueInputLevel(
            attention_parameters);
        const auto a = probability_spec.domain == Domain::kMpcArithmeticShare
            ? MpcToHe(graph, probabilities, attention_region,
                      Layout::kSoftmaxAvDiagonal,
                      prefix + ".probabilities.mpc_to_he", av_input_level)
            : EnsureRegion(graph, probabilities, attention_region,
                           Layout::kSoftmaxAvDiagonal,
                           prefix + ".probabilities");
        const auto v = EnsureRegion(graph, value, attention_region,
                                    Layout::kMultiHeadV,
                                    prefix + ".value");
        const TensorShape av_shape{{layer.tokens, layer.q_heads,
                                    layer.value_head_width}};
        auto av_spec = HeSpec(graph.Spec(a), av_shape,
                              Layout::kAttentionOutput, attention_region);
        av_spec.planned_level = av_input_level + detail::kAttentionValueDepth;
        av_spec.planned_scale_bits =
            graph.ActiveParameters(attention_region).default_scale_bits;
        const auto av = graph.AddNode(
            prefix + ".attention_value", OpKind::kMatMulCipherCipher,
            {a, v}, av_spec,
            MatMulAttributes{layer.id, "", av_shape,
                             Layout::kAttentionOutput, attention_region,
                             layer.q_heads != layer.kv_heads,
                             HeKernelKind::kMultiHeadAttentionValue});
        if (graph.ActiveProfile(attention_region) != graph.ActiveProfile(output_region))
            return Status::Invalid("Region 4 must use the same parameter profile as Region 2");
        const auto &output_parameters = graph.ActiveParameters(output_region);
        const auto output_terminal = output_parameters.coefficient_modulus_bits.size() -
            output_parameters.terminal_prime_count;
        if (output_terminal < 4)
            return Status::FailedPrecondition(
                prefix + ": Region 4 needs four Wo/LN2 levels");
        const auto wo_input_level = output_terminal - 4;
        auto av_output_region = av;
        if (!detail::FusedAttentionOutput(attention_parameters)) {
            const auto av_share = HeToMpc(graph, av, Layout::kAttentionOutput,
                prefix + ".attention_value_output_region.bridge.he_to_mpc");
            av_output_region = MpcToHe(
                graph, av_share, output_region, Layout::kAttentionOutput,
                prefix + ".attention_value_output_region.bridge.mpc_to_he", wo_input_level);
        }
        const TensorShape output_shape{{layer.tokens, layer.model_width}};
        constexpr auto output_layout = Layout::kStrictColumn;
        auto output_spec = HeSpec(graph.Spec(av_output_region), output_shape,
                                  output_layout, output_region);
        output_spec.planned_level =
            graph.Spec(av_output_region).planned_level.value_or(0) + 1;
        output_spec.planned_scale_bits =
            graph.ActiveParameters(output_region).default_scale_bits;
        auto projected = graph.AddNode(
            prefix + ".Wo", OpKind::kMatMulCipherPlain,
            {av_output_region}, output_spec,
            MatMulAttributes{layer.id, "Wo", output_shape,
                             output_layout, output_region,
                             false,
                             HeKernelKind::kAttentionOutputProjection});
        if (layer.attention_residual) {
            const auto residual_ready = EnsureRegion(
                graph, residual, output_region,
                output_layout, prefix + ".residual",
                output_spec.planned_level);
            projected = graph.AddNode(prefix + ".residual_add",
                                      OpKind::kElementwiseAdd,
                                      {projected, residual_ready}, output_spec);
        }
        return AttentionOutputGraphTensors{projected, projected};
    } catch (const std::exception &error) {
        return Status::Internal(prefix + ": " + error.what());
    }
}

} // namespace hera
