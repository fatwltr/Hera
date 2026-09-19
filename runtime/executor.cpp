#include "hera/private_transformer.h"

#include <map>
#include <sstream>

namespace hera {
namespace {

Result<void> ValidateWeight(const WeightHandle *actual,
                            const MatrixSpec &expected,
                            const std::string &layer_id)
{
    if (!actual || !actual->valid())
        return Status::Invalid(layer_id + "." + expected.name +
                               " weight handle is absent");
    if (actual->spec.rows != expected.rows ||
        actual->spec.columns != expected.columns ||
        actual->spec.has_bias != expected.has_bias)
        return Status::Invalid(layer_id + "." + expected.name +
                               " weight contract mismatch");
    return {};
}

Result<void> ValidateWeights(const HeraConfig &config,
                             const TransformerWeightHandles &weights)
{
    for (const auto &layer : config.layers) {
        auto schema = DeriveWeightSchema(layer);
        if (!schema.ok()) return schema.status();
        const MatrixSpec expected[] = {
            schema.value().wq, schema.value().wk, schema.value().wv,
            schema.value().wo, schema.value().wu, schema.value().wd};
        for (const auto &matrix : expected) {
            const auto valid = ValidateWeight(
                weights.Find(layer.id, matrix.name), matrix, layer.id);
            if (!valid.ok()) return valid.status();
        }
        for (const auto &plain : schema.value().vectors) {
            const auto *actual = weights.FindPlain(layer.id, plain.name);
            if (!actual || !actual->valid() ||
                actual->spec.name != plain.name ||
                actual->spec.length != plain.length)
                return Status::Invalid(layer.id + "." + plain.name +
                                       " plaintext parameter contract mismatch");
        }
    }
    return {};
}

Result<TensorHandle> MissingInput(const GraphNode &node, TensorId id)
{
    return Status::Internal(node.name + ": runtime tensor t" +
                            std::to_string(id) + " is unavailable");
}

Result<TensorHandle> Dispatch(
    const GraphNode &node, const TensorSpec &output_spec,
    const std::vector<TensorHandle> &inputs,
    const TransformerWeightHandles &weights,
    const PublicAttentionMask &mask, const BackendSet &backends)
{
    switch (node.kind) {
    case OpKind::kMatMulCipherPlain: {
        const auto &attr = std::get<MatMulAttributes>(node.attributes);
        const auto *weight = weights.Find(attr.layer_id, attr.weight_name);
        if (!weight)
            return Status::Invalid(node.name + ": weight is absent");
        return backends.he->MatMulCipherPlain(inputs.at(0), *weight, attr,
                                               output_spec);
    }
    case OpKind::kMatMulCipherCipher:
        return backends.he->MatMulCipherCipher(
            inputs.at(0), inputs.at(1),
            std::get<MatMulAttributes>(node.attributes), output_spec);
    case OpKind::kElementwiseAdd:
        return backends.he->Add(inputs.at(0), inputs.at(1), output_spec);
    case OpKind::kElementwiseMultiply:
        return backends.he->Multiply(inputs.at(0), inputs.at(1), output_spec);
    case OpKind::kElementwiseAddPlain: {
        const auto &attr = std::get<ElementwisePlainAttributes>(node.attributes);
        const auto *plain = weights.FindPlain(attr.layer_id,
                                              attr.parameter_name);
        if (!plain)
            return Status::Invalid(node.name + ": plaintext parameter is absent");
        return backends.he->AddPlain(inputs.at(0), *plain, attr, output_spec);
    }
    case OpKind::kElementwiseMultiplyPlain: {
        const auto &attr = std::get<ElementwisePlainAttributes>(node.attributes);
        const auto *plain = weights.FindPlain(attr.layer_id,
                                              attr.parameter_name);
        if (!plain)
            return Status::Invalid(node.name + ": plaintext parameter is absent");
        return backends.he->MultiplyPlain(inputs.at(0), *plain, attr,
                                           output_spec);
    }
    case OpKind::kScalarAdd:
        return backends.he->ScalarAdd(
            inputs.at(0), std::get<ScalarAttributes>(node.attributes),
            output_spec);
    case OpKind::kScalarMultiply:
        return backends.he->ScalarMultiply(
            inputs.at(0), std::get<ScalarAttributes>(node.attributes),
            output_spec);
    case OpKind::kReduceSum:
        return backends.he->ReduceSum(
            inputs.at(0), std::get<ReductionAttributes>(node.attributes),
            output_spec);
    case OpKind::kBroadcast:
        return backends.he->Broadcast(
            inputs.at(0), std::get<BroadcastAttributes>(node.attributes),
            output_spec);
    case OpKind::kPolynomial:
        return backends.he->Polynomial(
            inputs.at(0), std::get<PolynomialAttributes>(node.attributes),
            output_spec);
    case OpKind::kHeRealifyComplex:
        return backends.he->RealifyComplex(inputs.at(0), output_spec);
    case OpKind::kApplyPublicMask:
        return backends.he->ApplyPublicMask(
            inputs.at(0), mask,
            std::get<PublicMaskAttributes>(node.attributes), output_spec);
    case OpKind::kCompare:
        return backends.mpc->Compare(
            inputs, std::get<MpcNonlinearAttributes>(node.attributes),
            output_spec);
    case OpKind::kBooleanXor:
        return backends.mpc->BooleanXor(inputs.at(0), inputs.at(1),
                                        output_spec);
    case OpKind::kBooleanNot:
        return backends.mpc->BooleanNot(inputs.at(0), output_spec);
    case OpKind::kMux:
        return backends.mpc->Mux(
            inputs, std::get<MpcNonlinearAttributes>(node.attributes),
            output_spec);
    case OpKind::kMpcAdd:
        return backends.mpc->AddArithmeticShares(
            inputs.at(0), inputs.at(1), output_spec);
    case OpKind::kMpcReduceMax:
        return backends.mpc->ReduceMax(
            inputs.at(0), mask,
            std::get<MpcNonlinearAttributes>(node.attributes), output_spec);
    case OpKind::kMpcPrepareMaskedScores:
        return backends.mpc->PrepareMaskedScores(
            inputs.at(0), inputs.at(1), mask,
            std::get<MpcNonlinearAttributes>(node.attributes), output_spec);
    case OpKind::kMpcReduceSum:
        return backends.mpc->ReduceSum(
            inputs.at(0),
            std::get<MpcNonlinearAttributes>(node.attributes), output_spec);
    case OpKind::kMpcMultiplyFixed:
        return backends.mpc->MultiplyFixed(
            inputs.at(0), inputs.at(1),
            std::get<MpcNonlinearAttributes>(node.attributes), output_spec);
    case OpKind::kReciprocal:
        return backends.mpc->Reciprocal(
            inputs.at(0),
            std::get<MpcNonlinearAttributes>(node.attributes), output_spec);
    case OpKind::kReciprocalSqrt:
        return backends.mpc->ReciprocalSqrt(
            inputs.at(0),
            std::get<MpcNonlinearAttributes>(node.attributes), output_spec);
    case OpKind::kHeToMpc:
        return backends.conversion->HeToMpc(
            inputs.at(0), std::get<ConversionAttributes>(node.attributes),
            output_spec);
    case OpKind::kMpcToHe:
        return backends.conversion->MpcToHe(
            inputs.at(0), std::get<ConversionAttributes>(node.attributes),
            output_spec);
    case OpKind::kLocalShareRepack:
        return backends.packing->RepackLocalShares(
            inputs.at(0), std::get<RepackAttributes>(node.attributes),
            output_spec);
    }
    return Status::Internal(node.name + ": unknown operation");
}

void AppendDiagnostic(ExecutionReport *report, const GraphNode &node,
                      Status status)
{
    if (report)
        report->nodes.push_back(
            {node.id, node.name, node.kind, std::move(status)});
}

} // namespace

Result<TensorHandle> GraphExecutor::Execute(
    const TensorHandle &input, const TransformerWeightHandles &weights,
    const PublicAttentionMask &mask, ExecutionReport *report) const
{
    if (!context_)
        return Status::FailedPrecondition("executor has no HERA context");
    if (report) report->nodes.clear();
    const auto backend_valid = context_->backends().Validate();
    if (!backend_valid.ok()) return backend_valid.status();
    const auto *expected_input = context_->graph().FindTensor(
        context_->graph().input());
    if (!input.valid() || !expected_input ||
        !SameTensorContract(input.spec, *expected_input))
        return Status::Invalid(
            "encrypted hidden-state input does not match graph contract");
    const auto weight_valid = ValidateWeights(context_->config(), weights);
    if (!weight_valid.ok()) return weight_valid.status();

    std::map<TensorId, TensorHandle> values;
    values.emplace(context_->graph().input(), input);
    for (const auto &node : context_->graph().nodes()) {
        std::vector<TensorHandle> inputs;
        for (const auto id : node.inputs) {
            const auto it = values.find(id);
            if (it == values.end()) return MissingInput(node, id);
            inputs.push_back(it->second);
        }
        const auto *expected_output =
            context_->graph().FindTensor(node.output);
        if (!expected_output)
            return Status::Internal(node.name + ": output contract is absent");
        auto result = Dispatch(node, *expected_output, inputs, weights, mask,
                               context_->backends());
        if (!result.ok()) {
            auto status = Status(result.status().code(),
                node.name + " (" + ToString(node.kind) + "): " +
                result.status().message());
            AppendDiagnostic(report, node, status);
            return status;
        }
        if (!result.value().valid() ||
            !SatisfiesTensorContract(result.value().spec, *expected_output)) {
            auto status = Status::FailedPrecondition(
                node.name + ": backend returned a mismatched tensor contract");
            AppendDiagnostic(report, node, status);
            return status;
        }
        values.emplace(node.output, result.take_value());
        AppendDiagnostic(report, node, Status::Ok());
    }
    const auto output = values.find(context_->graph().output());
    if (output == values.end())
        return Status::Internal("graph output was not produced");
    return output->second;
}

Result<PrivateTransformer> PrivateTransformer::Compile(
    std::shared_ptr<HeraContext> context)
{
    if (!context)
        return Status::Invalid("PrivateTransformer requires a context");
    const auto graph_valid = ValidateGraph(context->graph());
    if (!graph_valid.ok()) return graph_valid.status();
    const auto backend_valid = context->backends().Validate();
    if (!backend_valid.ok()) return backend_valid.status();
    return PrivateTransformer(std::move(context));
}

Result<void> PrivateTransformer::Prepare(
    const TransformerWeightHandles &weights,
    PreparationReport *report) const
{
    auto valid = ValidateWeights(context_->config(), weights);
    if (!valid.ok()) return valid.status();
    if (context_->execution_runtime())
        return context_->execution_runtime()->Prepare(weights, report);
    if (report) *report = {};
    return {};
}

Result<HeCipherTensorHandle> PrivateTransformer::Evaluate(
    const EncryptedTransformerInput &input,
    const TransformerWeightHandles &weights, ExecutionReport *report) const
{
    if (!input.hidden_states.tensor.valid() ||
        input.hidden_states.tensor.spec.domain != Domain::kHeCiphertext)
        return Status::Invalid("PrivateTransformer input must be HE ciphertext");
    const auto &first_layer = context_->config().layers.front();
    if (input.attention_mask.kind != first_layer.attention_mask)
        return Status::Invalid("attention mask kind does not match config");
    if (input.attention_mask.kind != AttentionMaskKind::kNone) {
        const TensorShape expected{{first_layer.tokens, first_layer.tokens}};
        if (input.attention_mask.shape != expected)
            return Status::Invalid("attention mask shape must be [tokens,tokens]");
        if (input.attention_mask.kind == AttentionMaskKind::kPublicPadding) {
            const auto count = expected.ElementCount();
            if (!count.ok() ||
                input.attention_mask.additive_values.size() != count.value())
                return Status::Invalid(
                    "public padding mask must provide tokens*tokens values");
        }
    }
    GraphExecutor executor(context_);
    auto output = context_->execution_runtime()
        ? context_->execution_runtime()->Evaluate(
              input.hidden_states.tensor, weights, input.attention_mask, report)
        : executor.Execute(input.hidden_states.tensor, weights,
                           input.attention_mask, report);
    if (!output.ok()) return output.status();
    if (output.value().spec.domain != Domain::kHeCiphertext)
        return Status::Internal("Transformer output is not HE ciphertext");
    return HeCipherTensorHandle{output.take_value()};
}

} // namespace hera
