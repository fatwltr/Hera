#pragma once

#include "hera/graph.h"

namespace hera {

struct LayerNormBuildConfig {
    std::string prefix;
    std::size_t width{};
    HeRegionId statistics_region{HeRegionId::kRegion1};
    HeRegionId finish_region{HeRegionId::kRegion2};
    double epsilon{1e-5};
    std::optional<int> statistics_scale_bits{};
};

struct AttentionProjectionOutputs {
    TensorId q{};
    TensorId k{};
    TensorId v{};
    TensorId scaled_scores{};
};

struct AttentionOutputGraphTensors {
    TensorId hidden_states{}; // realified input to LN2
    TensorId ffn_residual{};  // complex Wo + attention residual
};

class LayerNormGraphModule {
public:
    // Builds mean/variance in HE, converts only the nonlinear rsqrt portion to
    // MPC, refreshes live centered values, and returns normalized HE data.
    static Result<TensorId> Build(GraphBuilder &graph, TensorId input,
                                  const LayerNormBuildConfig &config,
                                  TensorId *normalized_residual = nullptr);
};

class AttentionGraphModule {
public:
    // Adds arbitrary-dimension Wq/Wk/Wv and Q*K^T nodes. No packing or HE
    // matrix multiplication is performed by this graph-building method.
    static Result<AttentionProjectionOutputs> BuildProjectionsAndScores(
        GraphBuilder &graph, TensorId input,
        const TransformerLayerConfig &layer, HeRegionId region,
        const std::string &prefix);

    // Adds A*V, migrates the result and residual into the output region, then
    // adds Wo and the attention residual.
    static Result<AttentionOutputGraphTensors> BuildAttentionValueAndOutput(
        GraphBuilder &graph, TensorId probabilities, TensorId value,
        TensorId residual, const TransformerLayerConfig &layer,
        HeRegionId attention_region, HeRegionId output_region,
        const std::string &prefix);
};

class SoftmaxGraphModule {
public:
    // Builds the staged Figure-10 Softmax graph. It returns 43/13 arithmetic
    // probability shares: max, clipping, sum, reciprocal and normalization
    // are MPC; only the repeated-squaring exponent approximation is HE.
    static Result<TensorId> Build(GraphBuilder &graph, TensorId scores,
                                  const TransformerLayerConfig &layer,
                                  const SoftmaxConfig &config,
                                  HeRegionId polynomial_region,
                                  const std::string &prefix);
};

class ActivationGraphModule {
public:
    // Builds compare/mux MPC guards around a configurable HE polynomial. The
    // concrete approximation and protocols are backend responsibilities.
    static Result<TensorId> Build(GraphBuilder &graph, TensorId input,
                                  ActivationKind activation,
                                  const GeluConfig &gelu,
                                  HeRegionId output_region,
                                  const std::string &prefix);
};

class FeedForwardGraphModule {
public:
    // Adds Wu, the selected activation subgraph, Wd, and optional residual.
    static Result<TensorId> Build(GraphBuilder &graph, TensorId input,
                                  TensorId residual,
                                  const TransformerLayerConfig &layer,
                                  const GeluConfig &gelu,
                                  HeRegionId activation_region,
                                  HeRegionId output_region,
                                  bool has_next_layer,
                                  const std::string &prefix);
};

} // namespace hera
