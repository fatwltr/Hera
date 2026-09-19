#pragma once

#include "hera/tensor.h"

#include <filesystem>
#include <map>

namespace hera {

enum class SecurityLevel { kNone, kTc128 };

struct HeParameterConfig {
    std::size_t polynomial_degree{};
    std::vector<int> coefficient_modulus_bits;
    int default_scale_bits{};
    int input_scale_bits{};
    int output_scale_bits{};
    std::size_t declared_depth{};
    std::size_t terminal_prime_count{1};
    bool require_exact_terminal_level{true};
    SecurityLevel security_level{SecurityLevel::kTc128};
};

struct HeRegionConfig {
    HeRegionId id{HeRegionId::kRegion1};
    std::string parameter_profile;
    Layout input_layout{Layout::kSpatial};
    Layout output_layout{Layout::kSpatial};
};

struct LayerExecutionConfig {
    std::string layer_id;
    std::map<HeRegionId, HeRegionConfig> regions;
};

struct PlacementConfig {
    HeRegionId ln_statistics{HeRegionId::kRegion1};
    HeRegionId attention_main{HeRegionId::kRegion2};
    HeRegionId softmax_polynomial{HeRegionId::kRegion3};
    HeRegionId attention_output_ln2{HeRegionId::kRegion4};
    HeRegionId ffn_activation{HeRegionId::kRegion5};
};

// Parameters shared by every CKKS/MPC boundary. HE-side scales remain part
// of the selected HE profile; these values describe the sharing ring and the
// fixed-point canonical embedding used by the conversion protocol.
struct ConversionProtocolConfig {
    int statistical_security_bits{40};
    int fft_scale_bits{20};
    bool allow_internal_mod_switch{true};
};

struct SoftmaxConfig {
    double clip_threshold{-13.0};
    double masked_shift{-14.0};
    int exp_divisor_bits{6};
    std::size_t exp_squarings{6};
    // The physical HE representation is real-only. Each logical attention
    // row is padded to the next power of two before it is scattered to slots.
    bool pad_rows_to_power_of_two{true};
};

struct GeluConfig {
    double threshold{2.7};
    double a{0.020848611754127593};
    double b{-0.18352506127082727};
    double c{0.5410550166368381};
    double d{-0.03798164612714154};
    double e{0.001620808531841547};
};

struct HeraConfig {
    int schema_version{1};
    std::vector<TransformerLayerConfig> layers;
    NumericFormat mpc_numeric{43, 13};
    std::map<std::string, HeParameterConfig> he_parameter_profiles;
    std::vector<LayerExecutionConfig> layer_execution;
    PlacementConfig placement;
    ConversionProtocolConfig conversion;
    SoftmaxConfig softmax;
    GeluConfig gelu;
    std::size_t threads{1};
};

Result<HeraConfig> LoadHeraConfig(const std::filesystem::path &path);
Result<void> ValidateHeraConfig(const HeraConfig &config);
const LayerExecutionConfig *FindLayerExecution(
    const HeraConfig &config, const std::string &layer_id);
const HeParameterConfig *FindRegionParameters(
    const HeraConfig &config, const std::string &layer_id,
    HeRegionId region);

std::string ToString(SecurityLevel value);

} // namespace hera
