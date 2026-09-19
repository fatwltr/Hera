#include "hera/modules.h"

#include "module_utils.h"

namespace hera {

Result<TensorId> FeedForwardGraphModule::Build(
    GraphBuilder &graph, TensorId input, TensorId residual,
    const TransformerLayerConfig &layer, const GeluConfig &gelu,
    HeRegionId activation_region, HeRegionId output_region,
    bool has_next_layer,
    const std::string &prefix)
{
    try {
        using namespace module_internal;
        const auto source = EnsureRegion(
            graph, input, activation_region, Layout::kComplexStrictColumn,
            prefix + ".input");
        const TensorShape up_shape{{layer.tokens, layer.ffn_width}};
        auto up_spec = HeSpec(graph.Spec(source), up_shape, Layout::kFfn,
                              activation_region);
        up_spec.planned_level =
            graph.Spec(source).planned_level.value_or(1) + 1;
        up_spec.planned_scale_bits =
            graph.ActiveParameters(activation_region).default_scale_bits;
        const auto up = graph.AddNode(
            prefix + ".Wu", OpKind::kMatMulCipherPlain, {source}, up_spec,
            MatMulAttributes{layer.id, "Wu", up_shape, Layout::kFfn,
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
        const auto down_level = has_next_layer
            ? terminal_level - 1 : std::size_t{6};
        const auto activation_he = MpcToHe(
            graph, activation.value(), output_region, Layout::kFfn,
            prefix + ".activation.mpc_to_he", down_level - 1);
        const TensorShape down_shape{{layer.tokens, layer.model_width}};
        auto down_spec = HeSpec(graph.Spec(activation_he), down_shape,
                                Layout::kComplexStrictColumn, output_region);
        down_spec.planned_level = down_level;
        down_spec.planned_scale_bits =
            graph.ActiveParameters(output_region).default_scale_bits;
        const auto down = graph.AddNode(
            prefix + ".Wd", OpKind::kMatMulCipherPlain,
            {activation_he}, down_spec,
            MatMulAttributes{layer.id, "Wd", down_shape,
                             Layout::kComplexStrictColumn, output_region,
                             false, HeKernelKind::kFfnDownProjection});
        auto combined = down;
        if (layer.ffn_residual) {
            const auto residual_ready = EnsureRegion(
                graph, residual, output_region,
                Layout::kComplexStrictColumn, prefix + ".residual");
            combined = graph.AddNode(prefix + ".residual_add",
                                     OpKind::kElementwiseAdd,
                                     {down, residual_ready}, down_spec);
        }
        auto real_spec = down_spec;
        real_spec.layout = Layout::kStrictColumn;
        real_spec.planned_level = down_level + 1;
        return graph.AddNode(prefix + ".realify_complex",
                             OpKind::kHeRealifyComplex, {combined},
                             real_spec);
    } catch (const std::exception &error) {
        return Status::Internal(prefix + ": " + error.what());
    }
}

} // namespace hera
