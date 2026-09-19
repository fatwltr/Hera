#include "hera/modules.h"

#include "module_utils.h"

namespace hera {

Result<TensorId> LayerNormGraphModule::Build(
    GraphBuilder &graph, TensorId input, const LayerNormBuildConfig &config)
{
    try {
        using namespace module_internal;
        const auto is_second = config.prefix.find("ln2") != std::string::npos ||
                               config.prefix.find("post_ln2") != std::string::npos;
        const auto requested_layout = graph.Spec(input).layout ==
                Layout::kStrictColumn
            ? Layout::kStrictColumn : Layout::kSpatial;
        // Region-1 statistics consume real strict columns, but Block 2 has a
        // fixed dense-spatial producer contract. Do not propagate the
        // external input layout across the MPC refresh boundary.
        const auto finish_layout = is_second
            ? Layout::kComplexStrictColumn : Layout::kSpatial;
        auto source = EnsureRegion(graph, input, config.statistics_region,
                                   requested_layout,
                                   config.prefix + ".input");
        const auto &input_spec = graph.Spec(source);
        if (input_spec.shape.dimensions.size() != 2 ||
            input_spec.shape.dimensions[1] != config.width)
            return Status::Invalid(config.prefix +
                                   ": LayerNorm expects [tokens,width]");
        const auto tokens = input_spec.shape.dimensions[0];
        const TensorShape reduced_shape{{tokens, 1}};

        auto reduced_spec = HeSpec(input_spec, reduced_shape,
                                   Layout::kReduction,
                                   config.statistics_region);
        const auto row_sum = graph.AddNode(
            config.prefix + ".sum", OpKind::kReduceSum, {source}, reduced_spec,
            ReductionAttributes{1, true});
        const auto mean = graph.AddNode(
            config.prefix + ".mean", OpKind::kScalarMultiply, {row_sum},
            reduced_spec, ScalarAttributes{1.0 / static_cast<double>(config.width)});
        auto full_spec = HeSpec(input_spec, input_spec.shape, requested_layout,
                               config.statistics_region);
        const auto mean_full = graph.AddNode(
            config.prefix + ".mean_broadcast", OpKind::kBroadcast, {mean},
            full_spec, BroadcastAttributes{1, input_spec.shape});
        const auto negative_mean = graph.AddNode(
            config.prefix + ".negate_mean", OpKind::kScalarMultiply,
            {mean_full}, full_spec, ScalarAttributes{-1.0});
        const auto centered = graph.AddNode(
            config.prefix + ".center", OpKind::kElementwiseAdd,
            {source, negative_mean}, full_spec);
        const auto squared = graph.AddNode(
            config.prefix + ".square", OpKind::kElementwiseMultiply,
            {centered, centered}, full_spec);
        const auto square_sum = graph.AddNode(
            config.prefix + ".square_sum", OpKind::kReduceSum, {squared},
            reduced_spec, ReductionAttributes{1, true});
        const auto variance = graph.AddNode(
            config.prefix + ".variance", OpKind::kScalarMultiply,
            {square_sum}, reduced_spec,
            ScalarAttributes{1.0 / static_cast<double>(config.width)});
        const auto variance_epsilon = graph.AddNode(
            config.prefix + ".epsilon", OpKind::kScalarAdd, {variance},
            reduced_spec, ScalarAttributes{config.epsilon});

        const auto variance_share = HeToMpc(
            graph, variance_epsilon, Layout::kReduction,
            config.prefix + ".variance.he_to_mpc");
        auto inverse_spec = ArithmeticShareSpec(
            graph.Spec(variance_share), reduced_shape, Layout::kReduction);
        const auto inverse_share = graph.AddNode(
            config.prefix + ".rsqrt", OpKind::kReciprocalSqrt,
            {variance_share}, inverse_spec,
            MpcNonlinearAttributes{"layer_norm_inverse_stddev", 0.0});

        // centered is live across the nonlinear boundary. It is explicitly
        // refreshed into the finish region so no cross-context HE edge exists.
        const auto centered_share = HeToMpc(
            graph, centered, requested_layout,
            config.prefix + ".centered.he_to_mpc");
        const auto &finish_parameters =
            graph.ActiveParameters(config.finish_region);
        const auto finish_terminal =
            finish_parameters.coefficient_modulus_bits.size() -
            finish_parameters.terminal_prime_count;
        constexpr std::size_t kFfnGeluDepth = 6;
        if (is_second && finish_terminal < kFfnGeluDepth)
            return Status::FailedPrecondition(
                config.prefix + ": Region-5 suffix is shorter than six levels");
        const auto finish_start = is_second
            ? finish_terminal - kFfnGeluDepth : std::size_t{0};
        const auto centered_refreshed = MpcToHe(
            graph, centered_share, config.finish_region, finish_layout,
            config.prefix + ".centered.mpc_to_he", finish_start);
        const auto inverse_he = MpcToHe(
            graph, inverse_share, config.finish_region, Layout::kReduction,
            config.prefix + ".inverse.mpc_to_he", finish_start);
        auto inverse_full_spec = HeSpec(
            graph.Spec(inverse_he), input_spec.shape, finish_layout,
            config.finish_region);
        const auto inverse_full = graph.AddNode(
            config.prefix + ".inverse_broadcast", OpKind::kBroadcast,
            {inverse_he}, inverse_full_spec,
            BroadcastAttributes{1, input_spec.shape});
        // Block-2/5 executes centered*inverse followed by gamma with one lazy
        // rescale.  The intermediate therefore deliberately carries the
        // product scale while staying at the fresh level.
        auto product_spec = inverse_full_spec;
        product_spec.planned_level = finish_start;
        product_spec.planned_scale_bits =
            2 * graph.ActiveParameters(config.finish_region).input_scale_bits;
        const auto normalized = graph.AddNode(
            config.prefix + ".normalize", OpKind::kElementwiseMultiply,
            {centered_refreshed, inverse_full}, product_spec);
        const std::string parameter_prefix = is_second ? "ln2" : "ln1";
        auto affine_spec = inverse_full_spec;
        affine_spec.planned_level = finish_start + 1;
        affine_spec.planned_scale_bits =
            is_second
                ? graph.ActiveParameters(config.finish_region)
                      .default_scale_bits + 15
                : graph.ActiveParameters(config.finish_region)
                      .default_scale_bits;
        const auto scaled = graph.AddNode(
            config.prefix + ".gamma", OpKind::kElementwiseMultiplyPlain,
            {normalized}, affine_spec,
            ElementwisePlainAttributes{
                graph.ActiveLayer(), parameter_prefix + ".gamma",
                HeKernelKind::kLayerNormTail});
        return graph.AddNode(
            config.prefix + ".beta", OpKind::kElementwiseAddPlain,
            {scaled}, affine_spec,
            ElementwisePlainAttributes{
                graph.ActiveLayer(), parameter_prefix + ".beta",
                HeKernelKind::kLayerNormTail});
    } catch (const std::exception &error) {
        return Status::Internal(config.prefix + ": " + error.what());
    }
}

} // namespace hera
