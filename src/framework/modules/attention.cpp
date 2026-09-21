#include "hera/modules.h"

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
        const auto source = EnsureRegion(graph, input, region, Layout::kSpatial,
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
        constexpr std::size_t kQkDepth = 5;
        if (terminal_level < kQkDepth)
            return Status::FailedPrecondition(
                prefix + ": QK suffix is shorter than five levels");
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
        const auto a = probability_spec.domain == Domain::kMpcArithmeticShare
            ? MpcToHe(graph, probabilities, attention_region,
                      Layout::kSoftmaxAvDiagonal,
                      prefix + ".probabilities.mpc_to_he", 2)
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
        av_spec.planned_level = 5;
        av_spec.planned_scale_bits =
            graph.ActiveParameters(attention_region).default_scale_bits;
        const auto av = graph.AddNode(
            prefix + ".attention_value", OpKind::kMatMulCipherCipher,
            {a, v}, av_spec,
            MatMulAttributes{layer.id, "", av_shape,
                             Layout::kAttentionOutput, attention_region,
                             layer.q_heads != layer.kv_heads,
                             HeKernelKind::kMultiHeadAttentionValue});
        const auto av_output_region = EnsureRegion(
            graph, av, output_region, Layout::kAttentionOutput,
            prefix + ".attention_value_output_region");
        const TensorShape output_shape{{layer.tokens, layer.model_width}};
        auto output_spec = HeSpec(graph.Spec(av_output_region), output_shape,
                                  Layout::kComplexStrictColumn, output_region);
        output_spec.planned_level = 6;
        output_spec.planned_scale_bits =
            graph.ActiveParameters(output_region).default_scale_bits;
        auto projected = graph.AddNode(
            prefix + ".Wo", OpKind::kMatMulCipherPlain,
            {av_output_region}, output_spec,
            MatMulAttributes{layer.id, "Wo", output_shape,
                             Layout::kComplexStrictColumn, output_region,
                             false,
                             HeKernelKind::kAttentionOutputProjection});
        if (layer.attention_residual) {
            const auto residual_ready = EnsureRegion(
                graph, residual, output_region,
                Layout::kComplexStrictColumn, prefix + ".residual");
            projected = graph.AddNode(prefix + ".residual_add",
                                      OpKind::kElementwiseAdd,
                                      {projected, residual_ready}, output_spec);
        }
        auto real_spec = output_spec;
        real_spec.layout = Layout::kStrictColumn;
        real_spec.planned_scale_bits =
            graph.ActiveParameters(output_region).default_scale_bits + 1;
        const auto real = graph.AddNode(prefix + ".realify_complex",
                                        OpKind::kHeRealifyComplex,
                                        {projected}, real_spec);
        return AttentionOutputGraphTensors{real, projected};
    } catch (const std::exception &error) {
        return Status::Internal(prefix + ": " + error.what());
    }
}

} // namespace hera
