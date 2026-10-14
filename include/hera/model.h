#pragma once

#include "hera/status.h"

#include <cstddef>
#include <string>
#include <vector>

namespace hera {

// NormalizedResidual follows Figure 10: each LN output feeds both the
// projection and its residual. It is distinct from standard Pre-LN.
enum class NormPlacement { kPreNorm, kPostNorm, kNormalizedResidual };
enum class ActivationKind { kGelu, kRelu, kSilu };
enum class AttentionMaskKind { kNone, kPublicPadding, kPublicCausal };

struct TransformerLayerConfig {
    std::string id;
    std::size_t tokens{};
    std::size_t model_width{};
    std::size_t q_heads{};
    std::size_t kv_heads{};
    std::size_t qk_head_width{};
    std::size_t value_head_width{};
    std::size_t ffn_width{};
    NormPlacement norm{NormPlacement::kPreNorm};
    ActivationKind activation{ActivationKind::kGelu};
    AttentionMaskKind attention_mask{AttentionMaskKind::kPublicPadding};
    bool qkv_bias{true};
    bool output_bias{true};
    bool ffn_bias{true};
    bool attention_residual{true};
    bool ffn_residual{true};
    // Epsilon used by the normalization immediately following the attention
    // residual. It is public and is encoded as a CKKS plaintext.
    double attention_norm_epsilon{1.0e-5};
    // Internal ring width used by the MPC reciprocal-square-root circuit.
    // The public input/output contract remains numeric.mpc_ring_bits. A
    // narrower working ring is valid only when every non-negative variance
    // fits its signed fixed-point range.
    int attention_norm_rsqrt_bits{20};
    // Local variance precision; centered and inverse shares keep the global
    // MPC format. Preserve epsilon through HE->MPC before computing rsqrt.
    int attention_norm_variance_scale_bits{24};
};

// LN2 tail (1), real-output Wu (1), GeLU polynomial (3).
inline constexpr std::size_t kFfnSuffixDepth = 5;

struct MatrixSpec {
    std::string name;
    std::size_t rows{};
    std::size_t columns{};
    bool has_bias{};
};

// Plain vector parameter consumed by an HE elementwise operation. These are
// separate from MatrixSpec so LayerNorm affine vectors and projection biases
// cannot be mistaken for dense matrices.
struct VectorSpec {
    std::string name;
    std::size_t length{};
};

struct LayerWeightSchema {
    MatrixSpec wq;
    MatrixSpec wk;
    MatrixSpec wv;
    MatrixSpec wo;
    MatrixSpec wu;
    MatrixSpec wd;
    std::vector<VectorSpec> vectors;
};

std::size_t QueryWidth(const TransformerLayerConfig &layer);
std::size_t KeyWidth(const TransformerLayerConfig &layer);
std::size_t ValueWidth(const TransformerLayerConfig &layer);
std::size_t AttentionWidth(const TransformerLayerConfig &layer);

Result<void> ValidateLayerConfig(const TransformerLayerConfig &layer);
Result<LayerWeightSchema> DeriveWeightSchema(
    const TransformerLayerConfig &layer);
Result<void> ValidateLayerSequence(
    const std::vector<TransformerLayerConfig> &layers);

std::string ToString(NormPlacement value);
std::string ToString(ActivationKind value);
std::string ToString(AttentionMaskKind value);

} // namespace hera
