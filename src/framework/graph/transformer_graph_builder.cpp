#include "hera/graph.h"

#include "hera/modules.h"

namespace hera {
namespace {

Result<AttentionOutputGraphTensors> BuildAttention(
    GraphBuilder &graph, TensorId input, TensorId residual,
    const TransformerLayerConfig &layer,
    const SoftmaxConfig &softmax_config,
    const PlacementConfig &placement, const std::string &prefix)
{
    auto projections = AttentionGraphModule::BuildProjectionsAndScores(
        graph, input, layer, placement.attention_main,
        prefix + ".attention");
    if (!projections.ok()) return projections.status();
    auto softmax = SoftmaxGraphModule::Build(
        graph, projections.value().scaled_scores, layer,
        softmax_config,
        placement.softmax_polynomial,
        prefix + ".softmax");
    if (!softmax.ok()) return softmax.status();
    return AttentionGraphModule::BuildAttentionValueAndOutput(
        graph, softmax.value(), projections.value().v, residual, layer,
        placement.attention_main, placement.attention_output_ln2,
        prefix + ".attention_output");
}

Result<TensorId> BuildPreNormLayer(GraphBuilder &graph, TensorId input,
                                   const TransformerLayerConfig &layer,
                                   const SoftmaxConfig &softmax_config,
                                   const GeluConfig &gelu_config,
                                   const PlacementConfig &placement,
                                   bool has_next_layer)
{
    const bool normalized_skip = layer.norm == NormPlacement::kNormalizedResidual;
    const bool shared_ffn = normalized_skip &&
        graph.ActiveProfile(placement.ln_statistics) == graph.ActiveProfile(placement.ffn_activation);
    auto ln1 = LayerNormGraphModule::Build(
        graph, input,
        {layer.id + ".ln1", layer.model_width, placement.ln_statistics,
         placement.attention_main, layer.attention_norm_epsilon,
         shared_ffn ? std::optional<int>{graph.ActiveParameters(placement.ln_statistics).output_scale_bits} : std::nullopt});
    if (!ln1.ok()) return ln1.status();
    auto attention = BuildAttention(graph, ln1.value(), normalized_skip ? ln1.value() : input,
                                    layer, softmax_config, placement, layer.id);
    if (!attention.ok()) return attention.status();
    TensorId ln2_residual{};
    auto ln2 = LayerNormGraphModule::Build(
        graph, attention.value().hidden_states,
        {layer.id + ".ln2", layer.model_width,
         placement.attention_output_ln2, placement.ffn_activation,
         layer.attention_norm_epsilon},
        normalized_skip && layer.ffn_residual ? &ln2_residual : nullptr);
    if (!ln2.ok()) return ln2.status();
    return FeedForwardGraphModule::Build(
        graph, ln2.value(), normalized_skip ? ln2_residual : attention.value().ffn_residual, layer,
        gelu_config, placement.ffn_activation,
        shared_ffn ? placement.ffn_activation : placement.attention_output_ln2, has_next_layer,
        layer.id + ".ffn");
}

Result<TensorId> BuildPostNormLayer(GraphBuilder &graph, TensorId input,
                                    const TransformerLayerConfig &layer,
                                    const SoftmaxConfig &softmax_config,
                                    const GeluConfig &gelu_config,
                                    const PlacementConfig &placement,
                                    bool has_next_layer)
{
    auto attention = BuildAttention(graph, input, input, layer,
                                    softmax_config, placement, layer.id);
    if (!attention.ok()) return attention.status();
    auto ln1 = LayerNormGraphModule::Build(
        graph, attention.value().hidden_states,
        {layer.id + ".post_ln1", layer.model_width,
         placement.attention_output_ln2, placement.ffn_activation,
         layer.attention_norm_epsilon});
    if (!ln1.ok()) return ln1.status();
    auto ffn = FeedForwardGraphModule::Build(
        graph, ln1.value(), ln1.value(), layer, gelu_config,
        placement.ffn_activation, placement.attention_output_ln2,
        has_next_layer, layer.id + ".ffn");
    if (!ffn.ok()) return ffn.status();
    return LayerNormGraphModule::Build(
        graph, ffn.value(),
        {layer.id + ".post_ln2", layer.model_width,
         placement.attention_output_ln2, placement.ffn_activation,
         layer.attention_norm_epsilon});
}

} // namespace

Result<TransformerGraph> BuildTransformerGraph(const HeraConfig &config)
{
    const auto valid = ValidateHeraConfig(config);
    if (!valid.ok()) return valid.status();
    try {
        GraphBuilder graph(config);
        const auto &first = config.layers.front();
        graph.SetActiveLayer(first.id);
        const auto entry_region = config.placement.attention_main;
        TensorSpec input_spec;
        input_spec.shape = {{first.tokens, first.model_width}};
        input_spec.domain = Domain::kHeCiphertext;
        // The real runtime consumes a real-only strict-column ciphertext so
        // Region-1 LayerNorm reductions have an unambiguous physical input
        // contract. Other layouts require an explicit producer/repack step.
        input_spec.layout = first.norm == NormPlacement::kPostNorm
            ? Layout::kSpatial : Layout::kStrictColumn;
        input_spec.numeric = config.mpc_numeric;
        input_spec.he_region = entry_region;
        graph.SetActiveLayer(first.id);
        auto current = graph.AddInput("encrypted_hidden_states", input_spec);

        for (std::size_t layer_index = 0;
             layer_index < config.layers.size(); ++layer_index) {
            const auto &layer = config.layers[layer_index];
            const bool has_next_layer = layer_index + 1 < config.layers.size();
            graph.SetActiveLayer(layer.id);
            Result<TensorId> next = layer.norm != NormPlacement::kPostNorm
                ? BuildPreNormLayer(graph, current, layer, config.softmax,
                                    config.gelu, config.placement,
                                    has_next_layer)
                : BuildPostNormLayer(graph, current, layer, config.softmax,
                                     config.gelu, config.placement,
                                     has_next_layer);
            if (!next.ok()) return next.status();
            current = next.value();
        }
        return graph.Finish(current);
    } catch (const std::exception &error) {
        return Status::Internal(std::string("cannot build Transformer graph: ") +
                                error.what());
    }
}

} // namespace hera
