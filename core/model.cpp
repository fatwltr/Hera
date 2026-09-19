#include "hera/model.h"

#include <limits>
#include <cmath>

namespace hera {
namespace {

Result<std::size_t> CheckedMultiply(std::size_t left, std::size_t right,
                                    const std::string &name)
{
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
        return Status::Invalid(name + " dimension overflows size_t");
    return left * right;
}

} // namespace

std::size_t QueryWidth(const TransformerLayerConfig &layer)
{
    return layer.q_heads * layer.qk_head_width;
}

std::size_t KeyWidth(const TransformerLayerConfig &layer)
{
    return layer.kv_heads * layer.qk_head_width;
}

std::size_t ValueWidth(const TransformerLayerConfig &layer)
{
    return layer.kv_heads * layer.value_head_width;
}

std::size_t AttentionWidth(const TransformerLayerConfig &layer)
{
    return layer.q_heads * layer.value_head_width;
}

Result<void> ValidateLayerConfig(const TransformerLayerConfig &layer)
{
    if (layer.id.empty())
        return Status::Invalid("Transformer layer id is empty");
    if (layer.tokens == 0 || layer.model_width == 0 || layer.q_heads == 0 ||
        layer.kv_heads == 0 || layer.qk_head_width == 0 ||
        layer.value_head_width == 0 || layer.ffn_width == 0)
        return Status::Invalid(layer.id + ": all dimensions must be positive");
    if (layer.q_heads % layer.kv_heads != 0)
        return Status::Invalid(
            layer.id + ": q_heads must be divisible by kv_heads");
    if (!std::isfinite(layer.attention_norm_epsilon) ||
        layer.attention_norm_epsilon <= 0.0)
        return Status::Invalid(
            layer.id + ": attention_norm_epsilon must be finite and positive");

    const auto q = CheckedMultiply(layer.q_heads, layer.qk_head_width,
                                   layer.id + ".q_width");
    if (!q.ok()) return q.status();
    const auto k = CheckedMultiply(layer.kv_heads, layer.qk_head_width,
                                   layer.id + ".k_width");
    if (!k.ok()) return k.status();
    const auto v = CheckedMultiply(layer.kv_heads, layer.value_head_width,
                                   layer.id + ".v_width");
    if (!v.ok()) return v.status();
    const auto attention = CheckedMultiply(
        layer.q_heads, layer.value_head_width,
        layer.id + ".attention_width");
    if (!attention.ok()) return attention.status();
    return {};
}

Result<LayerWeightSchema> DeriveWeightSchema(
    const TransformerLayerConfig &layer)
{
    const auto valid = ValidateLayerConfig(layer);
    if (!valid.ok()) return valid.status();

    LayerWeightSchema result{
        {"Wq", layer.model_width, QueryWidth(layer), layer.qkv_bias},
        {"Wk", layer.model_width, KeyWidth(layer), layer.qkv_bias},
        {"Wv", layer.model_width, ValueWidth(layer), layer.qkv_bias},
        {"Wo", AttentionWidth(layer), layer.model_width, layer.output_bias},
        {"Wu", layer.model_width, layer.ffn_width, layer.ffn_bias},
        {"Wd", layer.ffn_width, layer.model_width, layer.ffn_bias},
        {},
    };
    result.vectors = {
        {"ln1.gamma", layer.model_width},
        {"ln1.beta", layer.model_width},
        {"ln2.gamma", layer.model_width},
        {"ln2.beta", layer.model_width},
    };
    if (layer.qkv_bias) {
        result.vectors.push_back({"bq", QueryWidth(layer)});
        result.vectors.push_back({"bk", KeyWidth(layer)});
        result.vectors.push_back({"bv", ValueWidth(layer)});
    }
    if (layer.output_bias)
        result.vectors.push_back({"bo", layer.model_width});
    if (layer.ffn_bias) {
        result.vectors.push_back({"bu", layer.ffn_width});
        result.vectors.push_back({"bd", layer.model_width});
    }
    return result;
}

Result<void> ValidateLayerSequence(
    const std::vector<TransformerLayerConfig> &layers)
{
    if (layers.empty())
        return Status::Invalid("model.layers must not be empty");
    for (std::size_t i = 0; i < layers.size(); ++i) {
        const auto valid = ValidateLayerConfig(layers[i]);
        if (!valid.ok()) return valid.status();
        for (std::size_t j = 0; j < i; ++j)
            if (layers[i].id == layers[j].id)
                return Status::Invalid("duplicate layer id: " + layers[i].id);
        if (i != 0 &&
            (layers[i - 1].tokens != layers[i].tokens ||
             layers[i - 1].model_width != layers[i].model_width))
            return Status::Invalid(
                "adjacent layers must have compatible hidden-state shapes: " +
                layers[i - 1].id + " -> " + layers[i].id);
        if (i != 0 &&
            layers[i - 1].attention_mask != layers[i].attention_mask)
            return Status::Invalid(
                "one public mask input requires all layers to use the same "
                "attention_mask kind");
    }
    return {};
}

std::string ToString(NormPlacement value)
{
    return value == NormPlacement::kPreNorm ? "pre" : "post";
}

std::string ToString(ActivationKind value)
{
    switch (value) {
    case ActivationKind::kGelu: return "gelu";
    case ActivationKind::kRelu: return "relu";
    case ActivationKind::kSilu: return "silu";
    }
    return "unknown";
}

std::string ToString(AttentionMaskKind value)
{
    switch (value) {
    case AttentionMaskKind::kNone: return "none";
    case AttentionMaskKind::kPublicPadding: return "public_padding";
    case AttentionMaskKind::kPublicCausal: return "public_causal";
    }
    return "unknown";
}

} // namespace hera
