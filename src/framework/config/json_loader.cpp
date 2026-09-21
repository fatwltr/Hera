#define BOOST_BIND_GLOBAL_PLACEHOLDERS
#include "hera/config.h"

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <set>
#include <sstream>
#include <stdexcept>

namespace hera {
namespace {

using boost::property_tree::ptree;

void RejectUnknown(const ptree &tree, const std::set<std::string> &allowed,
                   const std::string &path)
{
    for (const auto &entry : tree)
        if (!entry.first.empty() && allowed.count(entry.first) == 0)
            throw std::runtime_error(path + ": unknown field '" +
                                     entry.first + "'");
}

const ptree &RequiredChild(const ptree &tree, const std::string &key,
                           const std::string &path)
{
    const auto child = tree.get_child_optional(key);
    if (!child)
        throw std::runtime_error(path + ": missing field '" + key + "'");
    return *child;
}

template <typename T>
T Required(const ptree &tree, const std::string &key,
           const std::string &path)
{
    const auto value = tree.get_optional<T>(key);
    if (!value)
        throw std::runtime_error(path + ": missing or invalid field '" + key +
                                 "'");
    return *value;
}

void RequireArray(const ptree &tree, const std::string &path)
{
    for (const auto &entry : tree)
        if (!entry.first.empty())
            throw std::runtime_error(path + " must be a JSON array");
}

NormPlacement ParseNorm(const std::string &value)
{
    if (value == "pre") return NormPlacement::kPreNorm;
    if (value == "post") return NormPlacement::kPostNorm;
    throw std::runtime_error("unknown norm placement: " + value);
}

ActivationKind ParseActivation(const std::string &value)
{
    if (value == "gelu") return ActivationKind::kGelu;
    if (value == "relu") return ActivationKind::kRelu;
    if (value == "silu") return ActivationKind::kSilu;
    throw std::runtime_error("unknown activation: " + value);
}

AttentionMaskKind ParseMask(const std::string &value)
{
    if (value == "none") return AttentionMaskKind::kNone;
    if (value == "public_padding")
        return AttentionMaskKind::kPublicPadding;
    if (value == "public_causal") return AttentionMaskKind::kPublicCausal;
    throw std::runtime_error("unknown attention mask: " + value);
}

SecurityLevel ParseSecurity(const std::string &value)
{
    if (value == "none") return SecurityLevel::kNone;
    if (value == "tc128") return SecurityLevel::kTc128;
    throw std::runtime_error("unknown security level: " + value);
}

HeRegionId RequiredRegion(const ptree &tree, const std::string &key,
                          const std::string &path)
{
    auto result = ParseHeRegionId(Required<std::string>(tree, key, path));
    if (!result.ok()) throw std::runtime_error(result.status().message());
    return result.value();
}

Layout RequiredLayout(const ptree &tree, const std::string &key,
                      const std::string &path)
{
    auto result = ParseLayout(Required<std::string>(tree, key, path));
    if (!result.ok()) throw std::runtime_error(result.status().message());
    return result.value();
}

TransformerLayerConfig ParseLayer(const ptree &tree, std::size_t index)
{
    const std::string path = "model.layers[" + std::to_string(index) + "]";
    RejectUnknown(tree,
        {"id", "tokens", "model_width", "q_heads", "kv_heads",
         "qk_head_width", "value_head_width", "ffn_width", "norm",
         "activation", "attention_mask", "qkv_bias", "output_bias",
         "ffn_bias", "attention_residual", "ffn_residual",
         "attention_norm_epsilon", "attention_norm_rsqrt_bits"}, path);
    TransformerLayerConfig layer;
    layer.id = Required<std::string>(tree, "id", path);
    layer.tokens = Required<std::size_t>(tree, "tokens", path);
    layer.model_width = Required<std::size_t>(tree, "model_width", path);
    layer.q_heads = Required<std::size_t>(tree, "q_heads", path);
    layer.kv_heads = Required<std::size_t>(tree, "kv_heads", path);
    layer.qk_head_width = Required<std::size_t>(tree, "qk_head_width", path);
    layer.value_head_width =
        Required<std::size_t>(tree, "value_head_width", path);
    layer.ffn_width = Required<std::size_t>(tree, "ffn_width", path);
    layer.norm = ParseNorm(Required<std::string>(tree, "norm", path));
    layer.activation =
        ParseActivation(Required<std::string>(tree, "activation", path));
    layer.attention_mask =
        ParseMask(Required<std::string>(tree, "attention_mask", path));
    layer.qkv_bias = Required<bool>(tree, "qkv_bias", path);
    layer.output_bias = Required<bool>(tree, "output_bias", path);
    layer.ffn_bias = Required<bool>(tree, "ffn_bias", path);
    layer.attention_residual =
        Required<bool>(tree, "attention_residual", path);
    layer.ffn_residual = Required<bool>(tree, "ffn_residual", path);
    layer.attention_norm_epsilon = tree.get<double>(
        "attention_norm_epsilon", 1.0e-5);
    layer.attention_norm_rsqrt_bits = tree.get<int>(
        "attention_norm_rsqrt_bits", 20);
    return layer;
}

HeParameterConfig ParseProfile(const ptree &tree, const std::string &name)
{
    const std::string path = "he_parameter_profiles." + name;
    RejectUnknown(tree,
        {"polynomial_degree", "coefficient_modulus_bits",
         "default_scale_bits", "input_scale_bits", "output_scale_bits",
         "declared_depth", "terminal_prime_count",
         "require_exact_terminal_level", "security_level"}, path);
    HeParameterConfig profile;
    profile.polynomial_degree =
        Required<std::size_t>(tree, "polynomial_degree", path);
    const auto &bits = RequiredChild(tree, "coefficient_modulus_bits", path);
    RequireArray(bits, path + ".coefficient_modulus_bits");
    for (const auto &entry : bits)
        profile.coefficient_modulus_bits.push_back(
            entry.second.get_value<int>());
    profile.default_scale_bits =
        Required<int>(tree, "default_scale_bits", path);
    profile.input_scale_bits = Required<int>(tree, "input_scale_bits", path);
    profile.output_scale_bits =
        Required<int>(tree, "output_scale_bits", path);
    profile.declared_depth =
        Required<std::size_t>(tree, "declared_depth", path);
    profile.terminal_prime_count =
        Required<std::size_t>(tree, "terminal_prime_count", path);
    profile.require_exact_terminal_level =
        Required<bool>(tree, "require_exact_terminal_level", path);
    profile.security_level = ParseSecurity(
        Required<std::string>(tree, "security_level", path));
    return profile;
}

LayerExecutionConfig ParseLayerExecution(const ptree &tree,
                                         std::size_t index)
{
    const std::string path = "layers[" + std::to_string(index) + "]";
    RejectUnknown(tree, {"id", "he_regions"}, path);
    LayerExecutionConfig output;
    output.layer_id = Required<std::string>(tree, "id", path);
    const auto &regions = RequiredChild(tree, "he_regions", path);
    RejectUnknown(regions,
        {"region1", "region2", "region3", "region4", "region5"},
        path + ".he_regions");
    for (const auto &entry : regions) {
        auto parsed_id = ParseHeRegionId(entry.first);
        if (!parsed_id.ok()) throw std::runtime_error(parsed_id.status().message());
        const std::string region_path = path + ".he_regions." + entry.first;
        RejectUnknown(entry.second,
            {"profile", "input_layout", "output_layout"}, region_path);
        HeRegionConfig region;
        region.id = parsed_id.value();
        region.parameter_profile =
            Required<std::string>(entry.second, "profile", region_path);
        region.input_layout =
            RequiredLayout(entry.second, "input_layout", region_path);
        region.output_layout =
            RequiredLayout(entry.second, "output_layout", region_path);
        output.regions.emplace(region.id, std::move(region));
    }
    return output;
}

} // namespace

Result<HeraConfig> LoadHeraConfig(const std::filesystem::path &path)
{
    try {
        ptree root;
        boost::property_tree::read_json(path.string(), root);
        RejectUnknown(root,
            {"schema_version", "model", "numeric", "conversion",
             "softmax", "gelu", "he_parameter_profiles", "layers", "placement",
             "execution"},
            "root");

        HeraConfig config;
        config.schema_version = Required<int>(root, "schema_version", "root");

        const auto &model = RequiredChild(root, "model", "root");
        RejectUnknown(model, {"layers"}, "model");
        const auto &model_layers = RequiredChild(model, "layers", "model");
        RequireArray(model_layers, "model.layers");
        std::size_t index = 0;
        for (const auto &entry : model_layers)
            config.layers.push_back(ParseLayer(entry.second, index++));

        const auto &numeric = RequiredChild(root, "numeric", "root");
        RejectUnknown(numeric, {"mpc_ring_bits", "mpc_scale_bits"},
                      "numeric");
        config.mpc_numeric.ring_bits =
            Required<int>(numeric, "mpc_ring_bits", "numeric");
        config.mpc_numeric.scale_bits =
            Required<int>(numeric, "mpc_scale_bits", "numeric");

        if (const auto conversion = root.get_child_optional("conversion")) {
            RejectUnknown(*conversion,
                          {"statistical_security_bits", "fft_scale_bits",
                           "allow_internal_mod_switch"},
                          "conversion");
            config.conversion.statistical_security_bits = Required<int>(
                *conversion, "statistical_security_bits", "conversion");
            config.conversion.fft_scale_bits = Required<int>(
                *conversion, "fft_scale_bits", "conversion");
            config.conversion.allow_internal_mod_switch = Required<bool>(
                *conversion, "allow_internal_mod_switch", "conversion");
        }

        if (const auto softmax = root.get_child_optional("softmax")) {
            RejectUnknown(*softmax,
                          {"clip_threshold", "masked_shift",
                           "exp_divisor_bits", "exp_squarings",
                           "row_padding"},
                          "softmax");
            config.softmax.clip_threshold = Required<double>(
                *softmax, "clip_threshold", "softmax");
            config.softmax.masked_shift = Required<double>(
                *softmax, "masked_shift", "softmax");
            config.softmax.exp_divisor_bits = Required<int>(
                *softmax, "exp_divisor_bits", "softmax");
            config.softmax.exp_squarings = Required<std::size_t>(
                *softmax, "exp_squarings", "softmax");
            const auto padding = Required<std::string>(
                *softmax, "row_padding", "softmax");
            if (padding != "next_power_of_two")
                throw std::runtime_error(
                    "softmax.row_padding must be 'next_power_of_two'");
            config.softmax.pad_rows_to_power_of_two = true;
        }

        if (const auto gelu = root.get_child_optional("gelu")) {
            RejectUnknown(*gelu, {"threshold", "coefficients"}, "gelu");
            config.gelu.threshold = Required<double>(
                *gelu, "threshold", "gelu");
            const auto &coefficients = RequiredChild(
                *gelu, "coefficients", "gelu");
            RejectUnknown(coefficients, {"a", "b", "c", "d", "e"},
                          "gelu.coefficients");
            config.gelu.a = Required<double>(coefficients, "a",
                                             "gelu.coefficients");
            config.gelu.b = Required<double>(coefficients, "b",
                                             "gelu.coefficients");
            config.gelu.c = Required<double>(coefficients, "c",
                                             "gelu.coefficients");
            config.gelu.d = Required<double>(coefficients, "d",
                                             "gelu.coefficients");
            config.gelu.e = Required<double>(coefficients, "e",
                                             "gelu.coefficients");
        }

        const auto &profiles =
            RequiredChild(root, "he_parameter_profiles", "root");
        for (const auto &entry : profiles) {
            if (entry.first.empty())
                throw std::runtime_error(
                    "he_parameter_profiles must be a JSON object");
            config.he_parameter_profiles.emplace(
                entry.first, ParseProfile(entry.second, entry.first));
        }

        const auto &layers = RequiredChild(root, "layers", "root");
        RequireArray(layers, "layers");
        index = 0;
        for (const auto &entry : layers)
            config.layer_execution.push_back(
                ParseLayerExecution(entry.second, index++));

        const auto &placement = RequiredChild(root, "placement", "root");
        RejectUnknown(placement,
            {"ln_statistics", "attention_main", "softmax_polynomial",
             "attention_output_ln2", "ffn_activation"}, "placement");
        config.placement.ln_statistics =
            RequiredRegion(placement, "ln_statistics", "placement");
        config.placement.attention_main =
            RequiredRegion(placement, "attention_main", "placement");
        config.placement.softmax_polynomial =
            RequiredRegion(placement, "softmax_polynomial", "placement");
        config.placement.attention_output_ln2 = RequiredRegion(
            placement, "attention_output_ln2", "placement");
        config.placement.ffn_activation =
            RequiredRegion(placement, "ffn_activation", "placement");

        const auto &execution = RequiredChild(root, "execution", "root");
        RejectUnknown(execution, {"threads"}, "execution");
        config.threads =
            Required<std::size_t>(execution, "threads", "execution");

        const auto valid = ValidateHeraConfig(config);
        if (!valid.ok()) return valid.status();
        return config;
    } catch (const std::exception &error) {
        return Status::Invalid("cannot load HERA config '" + path.string() +
                               "': " + error.what());
    }
}

} // namespace hera
