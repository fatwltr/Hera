#pragma once

#include "hera/config.h"

#include <map>
#include <string>
#include <vector>

namespace hera {

// Immutable geometry and level contract compiled once from a supported
// concrete runtime configuration, before generating evaluation keys.
struct CompiledLayerExecutionPlan {
    std::string layer_id;
    std::string statistics_profile;
    std::string attention_profile;
    std::string attention_output_profile;
    std::string softmax_profile;
    std::string activation_profile;
    std::size_t statistics_input_level{};
    int statistics_input_scale_bits{};
    std::string ffn_output_profile;
    std::size_t ffn_output_level{};
    int ffn_output_scale_bits{};
    std::size_t qk_projection_input_level{};
    std::size_t attention_scores_level{};
    std::size_t retained_value_level{2};
    std::size_t attention_value_input_level{};
    std::size_t attention_output_input_level{};
    // Wo output / residual consumption level. A normalized LN1 skip is
    // retained at L1 and mod-switched down to this level only for addition.
    std::size_t attention_residual_level{};
    Layout attention_residual_layout{Layout::kStrictColumn};
    std::vector<int> block2_rotation_steps;
};

struct CompiledTransformerExecutionPlan {
    std::vector<CompiledLayerExecutionPlan> layers;
    std::map<std::string, std::vector<int>> rotation_steps_by_profile;

    static Result<CompiledTransformerExecutionPlan> Compile(
        const HeraConfig &config);
};

} // namespace hera
