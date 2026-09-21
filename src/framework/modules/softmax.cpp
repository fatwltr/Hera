#include "hera/modules.h"

#include "module_utils.h"

#include <cmath>

namespace hera {

Result<TensorId> SoftmaxGraphModule::Build(
    GraphBuilder &graph, TensorId scores,
    const TransformerLayerConfig &layer, const SoftmaxConfig &config,
    HeRegionId polynomial_region,
    const std::string &prefix)
{
    try {
        using namespace module_internal;
        const auto &scores_spec = graph.Spec(scores);
        if (scores_spec.domain != Domain::kHeCiphertext ||
            scores_spec.layout != Layout::kAttentionDiagonal ||
            scores_spec.shape != TensorShape{{layer.q_heads, layer.tokens,
                                               layer.tokens}})
            return Status::Invalid(
                prefix + ": expected Block-2 attention-diagonal scores");

        const auto diagonal_shares = HeToMpc(
            graph, scores, Layout::kAttentionDiagonal,
            prefix + ".scores.he_to_mpc");
        auto rows_spec = ArithmeticShareSpec(
            graph.Spec(diagonal_shares),
            {{layer.tokens, layer.q_heads, layer.tokens}},
            Layout::kSoftmaxRows);
        rows_spec.numeric.is_signed = true;
        const auto rows = graph.AddNode(
            prefix + ".scores.diagonal_to_rows",
            OpKind::kLocalShareRepack, {diagonal_shares}, rows_spec,
            RepackAttributes{Layout::kAttentionDiagonal,
                             Layout::kSoftmaxRows});

        const TensorShape reduction_shape{{layer.tokens, layer.q_heads, 1}};
        auto max_spec = ArithmeticShareSpec(
            rows_spec, reduction_shape, Layout::kReduction);
        max_spec.numeric.is_signed = true;
        const auto row_max = graph.AddNode(
            prefix + ".mpc_reduce_max", OpKind::kMpcReduceMax, {rows},
            max_spec,
            MpcNonlinearAttributes{"softmax_row_max", 0.0, 2});

        const auto shifted = graph.AddNode(
            prefix + ".prepare_masked_scores",
            OpKind::kMpcPrepareMaskedScores, {rows, row_max}, rows_spec,
            MpcNonlinearAttributes{"softmax_subtract_max_and_mask",
                                   config.masked_shift, 2});

        auto shifted_he_spec = HeSpec(
            rows_spec, rows_spec.shape, Layout::kSoftmaxRows,
            polynomial_region);
        shifted_he_spec.planned_level = 0;
        shifted_he_spec.planned_scale_bits =
            graph.ActiveParameters(polynomial_region).input_scale_bits;
        const auto shifted_he = graph.AddNode(
            prefix + ".shifted.mpc_to_block3", OpKind::kMpcToHe,
            {shifted}, shifted_he_spec,
            ConversionAttributes{Domain::kMpcArithmeticShare,
                                 Domain::kHeCiphertext, polynomial_region,
                                 Layout::kSoftmaxRows, rows_spec.numeric,
                                 std::size_t{0}});

        auto divided_spec = shifted_he_spec;
        divided_spec.planned_level = 1;
        divided_spec.planned_scale_bits =
            graph.ActiveParameters(polynomial_region).default_scale_bits;
        const auto divided = graph.AddNode(
            prefix + ".he_scale_1_over_64", OpKind::kScalarMultiply,
            {shifted_he}, divided_spec,
            ScalarAttributes{std::ldexp(
                1.0, -config.exp_divisor_bits)});
        const auto base = graph.AddNode(
            prefix + ".he_add_one", OpKind::kScalarAdd, {divided},
            divided_spec, ScalarAttributes{1.0});

        auto exponentials = base;
        for (std::size_t i = 0; i < config.exp_squarings; ++i) {
            auto square_spec = divided_spec;
            square_spec.planned_level = 2 + i;
            square_spec.numeric.is_signed = false;
            exponentials = graph.AddNode(
                prefix + ".he_square_" + std::to_string(i),
                OpKind::kElementwiseMultiply,
                {exponentials, exponentials}, square_spec);
        }

        auto exp_share_spec = ArithmeticShareSpec(
            graph.Spec(exponentials), rows_spec.shape,
            Layout::kSoftmaxRows);
        exp_share_spec.numeric.is_signed = false;
        const auto exp_shares = graph.AddNode(
            prefix + ".exp.he_to_mpc", OpKind::kHeToMpc,
            {exponentials}, exp_share_spec,
            ConversionAttributes{Domain::kHeCiphertext,
                                 Domain::kMpcArithmeticShare, std::nullopt,
                                 Layout::kSoftmaxRows,
                                 exp_share_spec.numeric, std::nullopt});

        auto clip_bits_spec = BooleanShareSpec(
            graph.Spec(shifted), rows_spec.shape, Layout::kSoftmaxRows);
        const auto clip_bits = graph.AddNode(
            prefix + ".threshold_compare", OpKind::kCompare, {shifted},
            clip_bits_spec,
            MpcNonlinearAttributes{"softmax_shifted_ge",
                                   config.clip_threshold});
        const auto clipped = graph.AddNode(
            prefix + ".threshold_mux", OpKind::kMux,
            {clip_bits, exp_shares}, exp_share_spec,
            MpcNonlinearAttributes{"softmax_clip_to_zero", 0.0});

        auto sum_spec = ArithmeticShareSpec(
            exp_share_spec, reduction_shape, Layout::kReduction);
        sum_spec.numeric.is_signed = false;
        const auto denominator = graph.AddNode(
            prefix + ".mpc_row_sum", OpKind::kMpcReduceSum, {clipped},
            sum_spec,
            MpcNonlinearAttributes{"softmax_row_sum", 0.0, 2});
        const auto reciprocal = graph.AddNode(
            prefix + ".mpc_reciprocal", OpKind::kReciprocal,
            {denominator}, sum_spec,
            MpcNonlinearAttributes{"softmax_row_reciprocal", 0.0});

        auto probability_spec = rows_spec;
        probability_spec.numeric.is_signed = false;
        return graph.AddNode(
            prefix + ".mpc_normalize", OpKind::kMpcMultiplyFixed,
            {clipped, reciprocal}, probability_spec,
            MpcNonlinearAttributes{"softmax_normalize", 0.0, 2,
                                   probability_spec.numeric.ring_bits,
                                   probability_spec.numeric.scale_bits});
    } catch (const std::exception &error) {
        return Status::Internal(prefix + ": " + error.what());
    }
}

} // namespace hera
