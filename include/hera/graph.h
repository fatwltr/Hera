#pragma once

#include "hera/config.h"

#include <iosfwd>
#include <map>
#include <variant>

namespace hera {

using NodeId = std::uint64_t;

enum class OpKind {
    kMatMulCipherPlain,
    kMatMulCipherCipher,
    kElementwiseAdd,
    kElementwiseMultiply,
    kElementwiseAddPlain,
    kElementwiseMultiplyPlain,
    kScalarAdd,
    kScalarMultiply,
    kReduceSum,
    kBroadcast,
    kPolynomial,
    kApplyPublicMask,
    kCompare,
    kBooleanXor,
    kBooleanNot,
    kMux,
    kMpcReduceMax,
    kMpcPrepareMaskedScores,
    kMpcReduceSum,
    kMpcMultiplyFixed,
    kMpcAdd,
    kReciprocal,
    kReciprocalSqrt,
    kHeToMpc,
    kMpcToHe,
    kLocalShareRepack,
    kHeRealifyComplex,
};

enum class HeKernelKind {
    kGeneric,
    kLayerNormTail,
    kLayerNormResidual, // Original affine + real channel extraction, one tail rescale.
    kComplexDiagonalProjection,
    kMultiHeadQk,
    kMultiHeadAttentionValue,
    kAttentionOutputProjection,
    kLayerNormStatistics,
    kFfnUpProjection,
    kFfnDownProjection,
    kGeluPolynomial,
};

struct EmptyAttributes {};

struct MatMulAttributes {
    std::string layer_id;
    std::string weight_name;
    TensorShape output_shape;
    Layout output_layout{Layout::kSpatial};
    std::optional<HeRegionId> region;
    bool grouped_heads{};
    HeKernelKind kernel{HeKernelKind::kGeneric};
};

struct ElementwisePlainAttributes {
    std::string layer_id;
    std::string parameter_name;
    HeKernelKind kernel{HeKernelKind::kGeneric};
};

struct ReductionAttributes {
    std::size_t dimension{};
    bool keep_dimension{true};
};

struct BroadcastAttributes {
    std::size_t dimension{};
    TensorShape output_shape;
};

struct ScalarAttributes {
    double value{};
};

struct PolynomialAttributes {
    std::string approximation;
    std::vector<double> coefficients;
};

struct PublicMaskAttributes {
    AttentionMaskKind kind{AttentionMaskKind::kNone};
};

struct MpcNonlinearAttributes {
    std::string purpose;
    double public_value{};
    std::size_t dimension{};
    int output_ring_bits{};
    int output_scale_bits{};
};

struct ConversionAttributes {
    Domain source_domain{Domain::kHeCiphertext};
    Domain target_domain{Domain::kMpcArithmeticShare};
    std::optional<HeRegionId> target_region;
    Layout target_layout{Layout::kSpatial};
    NumericFormat target_numeric;
    // A conversion may deliberately enter a suffix of an existing HE chain.
    // Absence retains the fresh-level-zero behavior.
    std::optional<std::size_t> target_level;
};

struct RepackAttributes {
    Layout source_layout{Layout::kSpatial};
    Layout target_layout{Layout::kSpatial};
};

using OpAttributes = std::variant<
    EmptyAttributes, MatMulAttributes, ReductionAttributes,
    BroadcastAttributes, ScalarAttributes, PolynomialAttributes,
    PublicMaskAttributes, MpcNonlinearAttributes, ConversionAttributes,
    RepackAttributes, ElementwisePlainAttributes>;

struct GraphNode {
    NodeId id{};
    std::string name;
    OpKind kind{OpKind::kElementwiseAdd};
    std::vector<TensorId> inputs;
    TensorId output{};
    OpAttributes attributes;
};

class TransformerGraph {
public:
    const std::vector<GraphNode> &nodes() const noexcept { return nodes_; }
    const std::map<TensorId, TensorSpec> &tensors() const noexcept
    {
        return tensors_;
    }
    TensorId input() const noexcept { return input_; }
    TensorId output() const noexcept { return output_; }

    const TensorSpec *FindTensor(TensorId id) const;
    std::string Dump() const;

private:
    friend class GraphBuilder;
    std::vector<GraphNode> nodes_;
    std::map<TensorId, TensorSpec> tensors_;
    TensorId input_{};
    TensorId output_{};
};

class GraphBuilder {
public:
    explicit GraphBuilder(const HeraConfig &config) : config_(config) {}

    void SetActiveLayer(const std::string &layer_id);
    const std::string &ActiveLayer() const noexcept { return active_layer_; }
    const TransformerLayerConfig &ActiveLayerConfig() const;
    const std::string &ActiveProfile(HeRegionId region) const;
    const HeParameterConfig &ActiveParameters(HeRegionId region) const;

    TensorId AddInput(std::string name, TensorSpec spec);
    TensorId AddNode(std::string name, OpKind kind,
                     std::vector<TensorId> inputs, TensorSpec output_spec,
                     OpAttributes attributes = EmptyAttributes{});
    Result<TransformerGraph> Finish(TensorId output);

    const TensorSpec &Spec(TensorId id) const;

private:
    void CompleteHeContract(TensorSpec &spec, bool fresh_chain) const;

    const HeraConfig &config_;
    std::string active_layer_;
    TransformerGraph graph_;
    TensorId next_tensor_id_{1};
    NodeId next_node_id_{1};
};

Result<void> ValidateGraph(const TransformerGraph &graph);
Result<TransformerGraph> BuildTransformerGraph(const HeraConfig &config);

std::string ToString(OpKind value);

} // namespace hera
