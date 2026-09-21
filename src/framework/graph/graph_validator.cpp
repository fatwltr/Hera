#include "hera/graph.h"

#include <set>

namespace hera {
namespace {

bool IsHeOp(OpKind kind)
{
    return (kind >= OpKind::kMatMulCipherPlain &&
            kind <= OpKind::kApplyPublicMask) ||
           kind == OpKind::kHeRealifyComplex;
}

Result<void> RequireInputs(const TransformerGraph &graph,
                           const GraphNode &node, std::size_t minimum,
                           std::size_t maximum)
{
    if (node.inputs.size() < minimum || node.inputs.size() > maximum)
        return Status::Invalid(node.name + ": invalid input count");
    for (const auto id : node.inputs)
        if (!graph.FindTensor(id))
            return Status::Invalid(node.name + ": missing input tensor");
    return {};
}

} // namespace

Result<void> ValidateGraph(const TransformerGraph &graph)
{
    const auto *input = graph.FindTensor(graph.input());
    const auto *output = graph.FindTensor(graph.output());
    if (!input || !output)
        return Status::Invalid("graph input or output is absent");
    if (input->domain != Domain::kHeCiphertext)
        return Status::Invalid("Transformer graph input must be HE ciphertext");
    if (output->domain != Domain::kHeCiphertext)
        return Status::Invalid("Transformer graph output must be HE ciphertext");

    std::set<TensorId> available{graph.input()};
    std::set<std::string> names;
    for (const auto &node : graph.nodes()) {
        if (!names.insert(node.name).second)
            return Status::Invalid("duplicate graph node name: " + node.name);
        for (const auto id : node.inputs)
            if (available.count(id) == 0)
                return Status::Invalid(node.name +
                    ": graph is not in topological order");
        const auto *out = graph.FindTensor(node.output);
        if (!out) return Status::Invalid(node.name + ": output is absent");
        const auto spec_valid = ValidateTensorSpec(*out, node.name + ".output");
        if (!spec_valid.ok()) return spec_valid.status();

        if (IsHeOp(node.kind)) {
            const auto count = RequireInputs(graph, node, 1, 2);
            if (!count.ok()) return count.status();
            if (out->domain != Domain::kHeCiphertext)
                return Status::Invalid(node.name + ": HE op has non-HE output");
            for (const auto id : node.inputs) {
                const auto *in = graph.FindTensor(id);
                if (in->domain != Domain::kHeCiphertext)
                    return Status::Invalid(node.name + ": HE op has non-HE input");
                if (in->he_parameter_profile != out->he_parameter_profile)
                    return Status::Invalid(node.name +
                        ": HE inputs and output must use one parameter profile");
            }
        } else if (node.kind == OpKind::kCompare) {
            const auto count = RequireInputs(graph, node, 1, 2);
            if (!count.ok()) return count.status();
            if (out->domain != Domain::kMpcBooleanShare)
                return Status::Invalid(node.name +
                                       ": compare output must be boolean shares");
            for (const auto id : node.inputs)
                if (graph.FindTensor(id)->domain !=
                    Domain::kMpcArithmeticShare)
                    return Status::Invalid(node.name +
                        ": compare inputs must be arithmetic shares");
        } else if (node.kind == OpKind::kMux) {
            const auto count = RequireInputs(graph, node, 2, 3);
            if (!count.ok()) return count.status();
            if (out->domain != Domain::kMpcArithmeticShare)
                return Status::Invalid(node.name +
                    ": MPC arithmetic op has an invalid output domain");
            if (graph.FindTensor(node.inputs[0])->domain !=
                Domain::kMpcBooleanShare)
                return Status::Invalid(node.name +
                                       ": mux selector must be boolean shares");
            for (std::size_t i = 1; i < node.inputs.size(); ++i)
                if (graph.FindTensor(node.inputs[i])->domain !=
                    Domain::kMpcArithmeticShare)
                    return Status::Invalid(node.name +
                        ": mux values must be arithmetic shares");
        } else if (node.kind == OpKind::kBooleanXor ||
                   node.kind == OpKind::kBooleanNot) {
            const auto expected = node.kind == OpKind::kBooleanXor ? 2U : 1U;
            const auto count = RequireInputs(graph, node, expected, expected);
            if (!count.ok()) return count.status();
            if (out->domain != Domain::kMpcBooleanShare)
                return Status::Invalid(node.name +
                    ": boolean operation output must be boolean shares");
            for (const auto id : node.inputs)
                if (graph.FindTensor(id)->domain != Domain::kMpcBooleanShare)
                    return Status::Invalid(node.name +
                        ": boolean operation input must be boolean shares");
        } else if (node.kind == OpKind::kMpcReduceMax ||
                   node.kind == OpKind::kMpcReduceSum) {
            const auto count = RequireInputs(graph, node, 1, 1);
            if (!count.ok()) return count.status();
            if (graph.FindTensor(node.inputs[0])->domain !=
                    Domain::kMpcArithmeticShare ||
                out->domain != Domain::kMpcArithmeticShare)
                return Status::Invalid(node.name +
                    ": MPC reduction requires arithmetic shares");
        } else if (node.kind == OpKind::kMpcPrepareMaskedScores ||
                   node.kind == OpKind::kMpcMultiplyFixed ||
                   node.kind == OpKind::kMpcAdd) {
            const auto count = RequireInputs(graph, node, 2, 2);
            if (!count.ok()) return count.status();
            if (graph.FindTensor(node.inputs[0])->domain !=
                    Domain::kMpcArithmeticShare ||
                graph.FindTensor(node.inputs[1])->domain !=
                    Domain::kMpcArithmeticShare ||
                out->domain != Domain::kMpcArithmeticShare)
                return Status::Invalid(node.name +
                    ": MPC arithmetic operation requires arithmetic shares");
        } else if (node.kind == OpKind::kReciprocal ||
                   node.kind == OpKind::kReciprocalSqrt) {
            const auto count = RequireInputs(graph, node, 1, 1);
            if (!count.ok()) return count.status();
            if (graph.FindTensor(node.inputs[0])->domain !=
                    Domain::kMpcArithmeticShare ||
                out->domain != Domain::kMpcArithmeticShare)
                return Status::Invalid(node.name +
                    ": reciprocal input/output must be arithmetic shares");
        } else if (node.kind == OpKind::kHeToMpc) {
            const auto count = RequireInputs(graph, node, 1, 1);
            if (!count.ok()) return count.status();
            if (graph.FindTensor(node.inputs[0])->domain !=
                    Domain::kHeCiphertext ||
                out->domain != Domain::kMpcArithmeticShare)
                return Status::Invalid(node.name +
                    ": HE-to-MPC domain contract is invalid");
        } else if (node.kind == OpKind::kMpcToHe) {
            const auto count = RequireInputs(graph, node, 1, 1);
            if (!count.ok()) return count.status();
            if (graph.FindTensor(node.inputs[0])->domain !=
                    Domain::kMpcArithmeticShare ||
                out->domain != Domain::kHeCiphertext)
                return Status::Invalid(node.name +
                    ": MPC-to-HE domain contract is invalid");
        } else if (node.kind == OpKind::kLocalShareRepack) {
            const auto count = RequireInputs(graph, node, 1, 1);
            if (!count.ok()) return count.status();
            if (graph.FindTensor(node.inputs[0])->domain != out->domain ||
                out->domain == Domain::kHeCiphertext ||
                out->domain == Domain::kPlaintext)
                return Status::Invalid(node.name +
                    ": local repack only supports MPC shares");
        }
        available.insert(node.output);
    }
    if (available.count(graph.output()) == 0)
        return Status::Invalid("graph output is not produced");
    return {};
}

} // namespace hera
