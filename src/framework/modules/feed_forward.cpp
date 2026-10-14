#include "hera/modules.h"
#include "hera/detail/level_schedule.h"

#include "module_utils.h"

namespace hera {

Result<TensorId> FeedForwardGraphModule::Build(
    GraphBuilder &graph, TensorId input, TensorId residual,
    const TransformerLayerConfig &layer, const GeluConfig &gelu,
    HeRegionId activation_region, HeRegionId output_region,
    bool /*has_next_layer*/,
    const std::string &prefix)
{
    try {
        using namespace module_internal;
        const auto source = EnsureRegion(
            graph, input, activation_region, Layout::kComplexStrictColumn,
            prefix + ".input");
        const TensorShape up_shape{{layer.tokens, layer.ffn_width}};
        auto up_spec = HeSpec(graph.Spec(source), up_shape, Layout::kStrictColumn,
                              activation_region);
        up_spec.planned_level =
            graph.Spec(source).planned_level.value_or(1) + 1;
        up_spec.planned_scale_bits =
            graph.ActiveParameters(activation_region).output_scale_bits;
        const auto up = graph.AddNode(
            prefix + ".Wu", OpKind::kMatMulCipherPlain, {source}, up_spec,
            MatMulAttributes{layer.id, "Wu", up_shape, up_spec.layout,
                             activation_region, false,
                             HeKernelKind::kFfnUpProjection});
        auto activation = ActivationGraphModule::Build(
            graph, up, layer.activation, gelu, activation_region,
            prefix + ".activation");
        if (!activation.ok()) return activation.status();

        // The staged executor deliberately ends GeLU in MPC. Wd is a later
        // HE region, so the graph keeps this boundary explicit instead of
        // pretending that the activation still contains a ciphertext.
        const auto &output_parameters = graph.ActiveParameters(output_region);
        const auto terminal_level =
            output_parameters.coefficient_modulus_bits.size() -
            output_parameters.terminal_prime_count;
        const bool direct_normalized = layer.norm == NormPlacement::kNormalizedResidual &&
            graph.ActiveProfile(HeRegionId::kRegion1) == graph.ActiveProfile(activation_region) &&
            graph.ActiveProfile(activation_region) == graph.ActiveProfile(output_region);
        const auto down_level = direct_normalized
            ? detail::SharedFfnOutputLevel(output_parameters)
            : layer.norm == NormPlacement::kPreNorm
            ? graph.Spec(residual).planned_level.value_or(terminal_level - 3)
            : terminal_level - 3;
        constexpr auto output_layout = Layout::kStrictColumn;
        const auto activation_he = MpcToHe(
            graph, activation.value(), output_region, Layout::kFfn,
            prefix + ".activation.mpc_to_he", down_level - 1,
            direct_normalized ? std::optional<int>{output_parameters.output_scale_bits} : std::nullopt);
        const TensorShape down_shape{{layer.tokens, layer.model_width}};
        auto down_spec = HeSpec(graph.Spec(activation_he), down_shape,
                                output_layout, output_region);
        down_spec.planned_level = down_level;
        down_spec.planned_scale_bits =
            direct_normalized ? output_parameters.output_scale_bits : output_parameters.default_scale_bits;
        const auto down = graph.AddNode(
            prefix + ".Wd", OpKind::kMatMulCipherPlain,
            {activation_he}, down_spec,
            MatMulAttributes{layer.id, "Wd", down_shape,
                             output_layout, output_region,
                             false, HeKernelKind::kFfnDownProjection});
        auto combined = down;
        if (layer.ffn_residual) {
            const auto residual_ready = layer.norm == NormPlacement::kNormalizedResidual && !direct_normalized
                ? MpcToHe(graph, HeToMpc(graph, residual, output_layout,
                            prefix + ".residual.he_to_mpc"),
                          output_region, output_layout, prefix + ".residual.mpc_to_he", down_level)
                : EnsureRegion(graph, residual, output_region,
                               output_layout, prefix + ".residual");
            combined = graph.AddNode(prefix + ".residual_add",
                                     OpKind::kElementwiseAdd,
                                     {down, residual_ready}, down_spec);
        }
        return combined;
    } catch (const std::exception &error) {
        return Status::Internal(prefix + ": " + error.what());
    }
}

} // namespace hera
