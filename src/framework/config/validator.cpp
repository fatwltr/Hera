#include "hera/config.h"
#include "hera/detail/nonlinear_contracts.h"

#include <algorithm>
#include <cmath>
#include <set>

namespace hera {
namespace {

bool IsPowerOfTwo(std::size_t value)
{
    return value != 0 && (value & (value - 1)) == 0;
}

Result<void> ValidateProfile(const std::string &name,
                             const HeParameterConfig &profile)
{
    if (name.empty()) return Status::Invalid("HE profile name is empty");
    if (profile.polynomial_degree < 2048 ||
        !IsPowerOfTwo(profile.polynomial_degree))
        return Status::Invalid(name +
            ": polynomial_degree must be a power of two >= 2048");
    if (profile.coefficient_modulus_bits.empty())
        return Status::Invalid(name + ": coefficient modulus chain is empty");
    for (const int bits : profile.coefficient_modulus_bits)
        if (bits < 20 || bits > 60)
            return Status::Invalid(name +
                ": coefficient modulus bits must be in [20,60]");
    if (profile.default_scale_bits <= 0 || profile.input_scale_bits <= 0 ||
        profile.output_scale_bits <= 0)
        return Status::Invalid(name + ": scale bits must be positive");
    if (profile.terminal_prime_count == 0 ||
        profile.terminal_prime_count > profile.coefficient_modulus_bits.size())
        return Status::Invalid(name + ": invalid terminal_prime_count");
    if (profile.terminal_prime_count != 1)
        return Status::Invalid(
            name + ": CKKS-to-MPC requires terminal_prime_count=1");
    const auto usable_levels = profile.coefficient_modulus_bits.size() -
                               profile.terminal_prime_count;
    if (profile.declared_depth > usable_levels)
        return Status::Invalid(name +
            ": declared_depth exceeds the configured modulus chain");
    return {};
}

} // namespace

Result<void> ValidateHeraConfig(const HeraConfig &config)
{
    if (config.schema_version != 1)
        return Status::Invalid("only HERA config schema_version=1 is supported");
    const auto layers_valid = ValidateLayerSequence(config.layers);
    if (!layers_valid.ok()) return layers_valid.status();
    if (config.mpc_numeric.ring_bits < 2 ||
        config.mpc_numeric.ring_bits > 63 ||
        config.mpc_numeric.scale_bits < 0 ||
        config.mpc_numeric.scale_bits >= config.mpc_numeric.ring_bits)
        return Status::Invalid("invalid MPC numeric format");
    for (const auto &layer : config.layers) {
        const int variance_scale = std::max(config.mpc_numeric.scale_bits,
                                           layer.attention_norm_variance_scale_bits);
        if (variance_scale < 1 || variance_scale >= config.mpc_numeric.ring_bits - 1 ||
            layer.attention_norm_epsilon < std::ldexp(8.0, -variance_scale))
            return Status::Invalid(layer.id +
                ": LayerNorm variance precision must preserve epsilon and fit the MPC ring");
        if (layer.attention_norm_rsqrt_bits <=
                config.mpc_numeric.scale_bits + 1 ||
            layer.attention_norm_rsqrt_bits >
                config.mpc_numeric.ring_bits)
            return Status::Invalid(
                layer.id +
                ": attention_norm_rsqrt_bits must be greater than "
                "mpc_scale_bits+1 and no greater than mpc_ring_bits");
        auto rsqrt = detail::PlanLayerNormRsqrt(config.mpc_numeric.ring_bits,
            variance_scale, layer.attention_norm_rsqrt_bits, config.mpc_numeric.scale_bits);
        if (!rsqrt.ok()) return Status::Invalid(layer.id + ": " + rsqrt.status().message());
        auto softmax = detail::SoftmaxAccumulatorBits(layer.tokens,
            config.mpc_numeric.ring_bits, config.mpc_numeric.scale_bits);
        if (!softmax.ok()) return Status::Invalid(layer.id + ": " + softmax.status().message());
    }
    if (config.conversion.statistical_security_bits < 1 ||
        config.conversion.statistical_security_bits > 63)
        return Status::Invalid(
            "conversion.statistical_security_bits must be in [1,63]");
    if (config.conversion.fft_scale_bits < 1 ||
        config.conversion.fft_scale_bits > 40)
        return Status::Invalid(
            "conversion.fft_scale_bits must be in [1,40]");
    if (config.mpc_numeric.ring_bits +
            config.conversion.statistical_security_bits +
            config.conversion.fft_scale_bits >
        127)
        return Status::Invalid(
            "conversion working precision exceeds the 127-bit backend");
    if (config.he_parameter_profiles.empty())
        return Status::Invalid("he_parameter_profiles must not be empty");
    for (const auto &entry : config.he_parameter_profiles) {
        const auto valid = ValidateProfile(entry.first, entry.second);
        if (!valid.ok()) return valid.status();
        const auto &profile = entry.second;
        const int terminal_bits =
            profile.coefficient_modulus_bits.front();
        if (terminal_bits + config.conversion.statistical_security_bits +
                config.conversion.fft_scale_bits + 2 >
            127)
            return Status::Invalid(
                entry.first +
                ": terminal field plus statistical/FFT workspace exceeds 127 bits");
        const int largest_scale = std::max(
            profile.default_scale_bits,
            std::max(profile.input_scale_bits, profile.output_scale_bits));
        if (config.mpc_numeric.ring_bits +
                config.conversion.statistical_security_bits +
                std::max(0, largest_scale - config.mpc_numeric.scale_bits) >
            127)
            return Status::Invalid(
                entry.first +
                ": MPC-to-CKKS scale workspace exceeds 127 bits");
    }
    if (config.threads == 0)
        return Status::Invalid("execution.threads must be positive");
    if (!std::isfinite(config.softmax.clip_threshold) ||
        !std::isfinite(config.softmax.masked_shift) ||
        config.softmax.masked_shift >= config.softmax.clip_threshold)
        return Status::Invalid(
            "softmax.masked_shift must be finite and below clip_threshold");
    if (config.softmax.exp_divisor_bits < 1 ||
        config.softmax.exp_divisor_bits > 20 ||
        config.softmax.exp_squarings == 0 ||
        config.softmax.exp_squarings > 16)
        return Status::Invalid("invalid Softmax exponent schedule");
    if (!std::isfinite(config.gelu.threshold) ||
        config.gelu.threshold <= 0.0 ||
        !std::isfinite(config.gelu.a) || !std::isfinite(config.gelu.b) ||
        !std::isfinite(config.gelu.c) || !std::isfinite(config.gelu.d) ||
        !std::isfinite(config.gelu.e))
        return Status::Invalid(
            "GeLU threshold must be positive and all coefficients finite");
    if (config.layer_execution.size() != config.layers.size())
        return Status::Invalid(
            "layers must contain exactly one HE-region mapping per model layer");

    std::set<std::string> seen;
    for (const auto &execution : config.layer_execution) {
        if (!seen.insert(execution.layer_id).second)
            return Status::Invalid("duplicate layer execution id: " +
                                   execution.layer_id);
        bool model_layer_found = false;
        for (const auto &layer : config.layers)
            model_layer_found = model_layer_found ||
                                layer.id == execution.layer_id;
        if (!model_layer_found)
            return Status::Invalid("HE-region mapping references unknown layer: " +
                                   execution.layer_id);
        if (execution.regions.size() != 5)
            return Status::Invalid(execution.layer_id +
                                   ": exactly five HE regions are required");
        for (const auto &entry : execution.regions)
            if (config.he_parameter_profiles.count(
                    entry.second.parameter_profile) == 0)
                return Status::Invalid(
                    execution.layer_id + "." + ToString(entry.first) +
                    " references unknown profile: " +
                    entry.second.parameter_profile);
        const auto attention = execution.regions.find(HeRegionId::kRegion2);
        const auto output = execution.regions.find(HeRegionId::kRegion4);
        if (attention == execution.regions.end() ||
            output == execution.regions.end())
            return Status::Invalid(execution.layer_id +
                                   ": Region 2 and Region 4 are required");
        if (attention->second.parameter_profile != output->second.parameter_profile)
            return Status::Invalid(execution.layer_id +
                ": Region 4 must use the same parameter profile as Region 2");
    }
    for (const auto &layer : config.layers)
        if (seen.count(layer.id) == 0)
            return Status::Invalid("missing HE-region mapping for layer: " +
                                   layer.id);
    return {};
}

const LayerExecutionConfig *FindLayerExecution(
    const HeraConfig &config, const std::string &layer_id)
{
    for (const auto &layer : config.layer_execution)
        if (layer.layer_id == layer_id) return &layer;
    return nullptr;
}

const HeParameterConfig *FindRegionParameters(
    const HeraConfig &config, const std::string &layer_id,
    HeRegionId region)
{
    const auto *layer = FindLayerExecution(config, layer_id);
    if (!layer) return nullptr;
    const auto region_it = layer->regions.find(region);
    if (region_it == layer->regions.end()) return nullptr;
    const auto profile_it = config.he_parameter_profiles.find(
        region_it->second.parameter_profile);
    return profile_it == config.he_parameter_profiles.end()
        ? nullptr : &profile_it->second;
}

std::string ToString(SecurityLevel value)
{
    return value == SecurityLevel::kNone ? "none" : "tc128";
}

} // namespace hera
