#include "hera/execution_plan.h"

#include "hera/block5_bridge.h"

#include <set>

namespace hera {

Result<CompiledTransformerExecutionPlan>
CompiledTransformerExecutionPlan::Compile(const HeraConfig &config)
{
    auto valid = ValidateHeraConfig(config);
    if (!valid.ok()) return valid.status();
    CompiledTransformerExecutionPlan plan;
    std::map<std::string, std::set<int>> rotations;
    for (std::size_t index = 0; index < config.layers.size(); ++index) {
        const auto &layer = config.layers[index];
        if (layer.norm != NormPlacement::kPreNorm)
            return Status::KernelUnavailable(
                "real HERA runtime currently supports Pre-LN only");
        if (layer.activation != ActivationKind::kGelu)
            return Status::KernelUnavailable(
                "real HERA runtime currently supports GeLU only");
        const auto *execution = FindLayerExecution(config, layer.id);
        if (!execution)
            return Status::Invalid(layer.id + ": HE placement is absent");
        const auto profile = [&](HeRegionId region) -> const std::string & {
            return execution->regions.at(region).parameter_profile;
        };
        const auto &r1 = profile(HeRegionId::kRegion1);
        const auto &r2 = profile(HeRegionId::kRegion2);
        const auto &r3 = profile(HeRegionId::kRegion3);
        const auto &r4 = profile(HeRegionId::kRegion4);
        const auto &r5 = profile(HeRegionId::kRegion5);
        if (r2 != r4)
            return Status::FailedPrecondition(
                layer.id + ": Region 2/4 must share one HE profile");
        const auto &p1 = config.he_parameter_profiles.at(r1);
        const auto &p2 = config.he_parameter_profiles.at(r2);
        const auto &p3 = config.he_parameter_profiles.at(r3);
        const auto &p5 = config.he_parameter_profiles.at(r5);
        const auto statistics_terminal =
            p1.coefficient_modulus_bits.size() - p1.terminal_prime_count;
        if (statistics_terminal < 3)
            return Status::FailedPrecondition(
                layer.id + ": Region 1 needs three statistics levels");
        auto checked = Block2LevelPlanner::Validate(p2);
        if (!checked.ok()) return checked.status();
        checked = AttentionLn2LevelPlanner::Validate(p2);
        if (!checked.ok()) return checked.status();
        checked = Block3LevelPlanner::Validate(p3, config.softmax);
        if (!checked.ok()) return checked.status();
        checked = FfnGeluLevelPlanner::Validate(p5);
        if (!checked.ok()) return checked.status();

        CompiledLayerExecutionPlan layer_plan;
        layer_plan.layer_id = layer.id;
        layer_plan.statistics_profile = r1;
        layer_plan.attention_profile = r2;
        layer_plan.softmax_profile = r3;
        layer_plan.activation_profile = r5;
        layer_plan.statistics_input_level = statistics_terminal - 3;
        layer_plan.attention_scores_level =
            p2.coefficient_modulus_bits.size() - p2.terminal_prime_count;
        layer_plan.qk_projection_input_level =
            layer_plan.attention_scores_level - 6;

        auto block2 = SealBlock2Executor::RequiredRotationSteps(layer, p2);
        if (!block2.ok()) return block2.status();
        layer_plan.block2_rotation_steps = block2.take_value();
        rotations[r2].insert(layer_plan.block2_rotation_steps.begin(),
                             layer_plan.block2_rotation_steps.end());
        auto attention = SealSciAttentionLn2Executor::RequiredRotationSteps(
            layer, p2);
        if (!attention.ok()) return attention.status();
        rotations[r2].insert(attention.value().begin(),
                             attention.value().end());
        auto ffn = SealSciFfnGeluExecutor::RequiredRotationSteps(layer, p5);
        if (!ffn.ok()) return ffn.status();
        rotations[r5].insert(ffn.value().begin(), ffn.value().end());
        auto bridge = SealSciBlock5BridgeExecutor::RequiredRotationSteps(
            config, index, false);
        if (!bridge.ok()) return bridge.status();
        for (const auto &entry : bridge.value())
            rotations[entry.first].insert(entry.second.begin(),
                                          entry.second.end());

        std::size_t stride = 1;
        while (stride < layer.tokens) stride <<= 1U;
        const auto columns = p1.polynomial_degree / 2 / stride;
        for (std::size_t step = 1; step < columns; step <<= 1U)
            rotations[r1].insert(static_cast<int>(step * stride));
        plan.layers.push_back(std::move(layer_plan));
    }
    for (auto &entry : rotations)
        plan.rotation_steps_by_profile.emplace(
            entry.first,
            std::vector<int>(entry.second.begin(), entry.second.end()));
    return plan;
}

} // namespace hera
