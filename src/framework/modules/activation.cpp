#include "hera/modules.h"

#include "module_utils.h"

namespace hera {

Result<TensorId> ActivationGraphModule::Build(
    GraphBuilder &graph, TensorId input, ActivationKind activation,
    const GeluConfig &gelu, HeRegionId output_region,
    const std::string &prefix)
{
    try {
        using namespace module_internal;
        const auto &input_spec = graph.Spec(input);
        if (activation != ActivationKind::kGelu) {
            // Keep the architecture-level graph configurable for ReLU/SiLU.
            // Their concrete crypto kernels remain unavailable in the staged
            // FFN executor, but graph compilation/recording must not reject a
            // valid Transformer architecture.
            const auto name = activation == ActivationKind::kRelu
                ? "relu_backend_defined" : "silu_backend_defined";
            const auto approximate = graph.AddNode(
                prefix + ".backend_defined", OpKind::kPolynomial, {input},
                input_spec, PolynomialAttributes{name, {}});
            return HeToMpc(graph, approximate, Layout::kFfn,
                           prefix + ".he_to_mpc");
        }
        auto real_spec = HeSpec(input_spec, input_spec.shape,
                                Layout::kStrictColumn, output_region);
        real_spec.planned_level = input_spec.planned_level;
        real_spec.planned_scale_bits =
            graph.ActiveParameters(output_region).output_scale_bits;
        if (input_spec.layout != Layout::kStrictColumn)
            return Status::Invalid(prefix + ": GeLU requires real-column Wu output");
        const auto real = input;

        auto polynomial_spec = real_spec;
        const auto &parameters = graph.ActiveParameters(output_region);
        polynomial_spec.planned_level =
            parameters.coefficient_modulus_bits.size() -
            parameters.terminal_prime_count;
        const std::vector<double> f0{
            gelu.e, 0.5 - gelu.d, gelu.c, -gelu.b, gelu.a};
        const std::vector<double> f1{
            gelu.e, 0.5 + gelu.d, gelu.c, gelu.b, gelu.a};
        const auto negative = graph.AddNode(
            prefix + ".F0", OpKind::kPolynomial, {real}, polynomial_spec,
            PolynomialAttributes{"blb_gelu_negative", f0});
        const auto positive = graph.AddNode(
            prefix + ".F1", OpKind::kPolynomial, {real}, polynomial_spec,
            PolynomialAttributes{"blb_gelu_positive", f1});

        const auto x_share = HeToMpc(
            graph, real, Layout::kFfn, prefix + ".x.he_to_mpc");
        const auto f0_share = HeToMpc(
            graph, negative, Layout::kFfn, prefix + ".F0.he_to_mpc");
        const auto f1_share = HeToMpc(
            graph, positive, Layout::kFfn, prefix + ".F1.he_to_mpc");
        auto bit_spec = BooleanShareSpec(
            graph.Spec(x_share), input_spec.shape, Layout::kFfn);
        const auto b0 = graph.AddNode(
            prefix + ".cmp_lt_negative_threshold", OpKind::kCompare,
            {x_share}, bit_spec,
            MpcNonlinearAttributes{"gelu_x_lt_negative_threshold",
                                   -gelu.threshold});
        const auto b1 = graph.AddNode(
            prefix + ".cmp_lt_zero", OpKind::kCompare, {x_share}, bit_spec,
            MpcNonlinearAttributes{"gelu_x_lt_zero", 0.0});
        const auto b2 = graph.AddNode(
            prefix + ".cmp_gt_positive_threshold", OpKind::kCompare,
            {x_share}, bit_spec,
            MpcNonlinearAttributes{"gelu_x_gt_positive_threshold",
                                   gelu.threshold});
        const auto z0 = graph.AddNode(
            prefix + ".negative_interval", OpKind::kBooleanXor,
            {b0, b1}, bit_spec);
        const auto central_xor = graph.AddNode(
            prefix + ".central_interval_xor", OpKind::kBooleanXor,
            {b1, b2}, bit_spec);
        const auto z1 = graph.AddNode(
            prefix + ".central_interval", OpKind::kBooleanNot,
            {central_xor}, bit_spec);
        auto share_spec = ArithmeticShareSpec(
            graph.Spec(x_share), input_spec.shape, Layout::kFfn);
        const auto term0 = graph.AddNode(
            prefix + ".mux_F0", OpKind::kMux, {z0, f0_share}, share_spec,
            MpcNonlinearAttributes{"gelu_mux_F0", 0.0});
        const auto term1 = graph.AddNode(
            prefix + ".mux_F1", OpKind::kMux, {z1, f1_share}, share_spec,
            MpcNonlinearAttributes{"gelu_mux_F1", 0.0});
        const auto term2 = graph.AddNode(
            prefix + ".mux_positive_tail", OpKind::kMux,
            {b2, x_share}, share_spec,
            MpcNonlinearAttributes{"gelu_mux_positive_tail", 0.0});
        const auto sum01 = graph.AddNode(
            prefix + ".sum_polynomial_branches", OpKind::kMpcAdd,
            {term0, term1}, share_spec);
        return graph.AddNode(
            prefix + ".output", OpKind::kMpcAdd,
            {sum01, term2}, share_spec);
    } catch (const std::exception &error) {
        return Status::Internal(prefix + ": " + error.what());
    }
}

} // namespace hera
