#include "hera/graph.h"

#include <sstream>
#include <stdexcept>

namespace hera {

const TensorSpec *TransformerGraph::FindTensor(TensorId id) const
{
    const auto it = tensors_.find(id);
    return it == tensors_.end() ? nullptr : &it->second;
}

std::string TransformerGraph::Dump() const
{
    std::ostringstream output;
    output << "HERA_GRAPH input=t" << input_ << " output=t" << output_
           << " nodes=" << nodes_.size() << '\n';
    for (const auto &node : nodes_) {
        const auto *spec = FindTensor(node.output);
        output << "n" << node.id << " " << node.name << " op="
               << ToString(node.kind) << " inputs=[";
        for (std::size_t i = 0; i < node.inputs.size(); ++i) {
            if (i) output << ',';
            output << 't' << node.inputs[i];
        }
        output << "] output=t" << node.output;
        if (spec) {
            output << " domain=" << ToString(spec->domain)
                   << " layout=" << ToString(spec->layout);
            if (spec->he_region)
                output << " region=" << ToString(*spec->he_region);
            if (spec->he_parameter_profile)
                output << " profile=" << *spec->he_parameter_profile;
            if (spec->planned_level)
                output << " level=" << *spec->planned_level;
            if (spec->planned_scale_bits)
                output << " scale_bits=" << *spec->planned_scale_bits;
            output << " shape=[";
            for (std::size_t i = 0; i < spec->shape.dimensions.size(); ++i) {
                if (i) output << ',';
                output << spec->shape.dimensions[i];
            }
            output << ']';
        }
        output << '\n';
    }
    return output.str();
}

void GraphBuilder::SetActiveLayer(const std::string &layer_id)
{
    if (!FindLayerExecution(config_, layer_id))
        throw std::invalid_argument("unknown active layer: " + layer_id);
    active_layer_ = layer_id;
}

const std::string &GraphBuilder::ActiveProfile(HeRegionId region) const
{
    const auto *layer = FindLayerExecution(config_, active_layer_);
    if (!layer) throw std::logic_error("graph builder has no active layer");
    const auto it = layer->regions.find(region);
    if (it == layer->regions.end())
        throw std::logic_error("active layer has no " + ToString(region));
    return it->second.parameter_profile;
}

const TransformerLayerConfig &GraphBuilder::ActiveLayerConfig() const
{
    for (const auto &layer : config_.layers)
        if (layer.id == active_layer_) return layer;
    throw std::logic_error("graph builder has no active layer config");
}

const HeParameterConfig &GraphBuilder::ActiveParameters(HeRegionId region) const
{
    const auto *parameters = FindRegionParameters(config_, active_layer_, region);
    if (!parameters)
        throw std::logic_error("active HE parameters are unavailable");
    return *parameters;
}

void GraphBuilder::CompleteHeContract(TensorSpec &spec, bool fresh_chain) const
{
    if (spec.domain != Domain::kHeCiphertext) return;
    if (!spec.he_region)
        throw std::logic_error("HE output has no region");
    spec.he_parameter_profile = ActiveProfile(*spec.he_region);
    const auto &parameters = ActiveParameters(*spec.he_region);
    if (!spec.planned_scale_bits)
        spec.planned_scale_bits = fresh_chain
            ? parameters.input_scale_bits : parameters.default_scale_bits;
    if (fresh_chain && !spec.planned_level) spec.planned_level = 0;
}

TensorId GraphBuilder::AddInput(std::string, TensorSpec spec)
{
    if (graph_.input_ != 0)
        throw std::logic_error("a graph can have only one hidden-state input");
    CompleteHeContract(spec, true);
    const auto id = next_tensor_id_++;
    graph_.input_ = id;
    graph_.tensors_.emplace(id, std::move(spec));
    return id;
}

TensorId GraphBuilder::AddNode(std::string name, OpKind kind,
                               std::vector<TensorId> inputs,
                               TensorSpec output_spec,
                               OpAttributes attributes)
{
    for (const auto input : inputs)
        if (graph_.tensors_.count(input) == 0)
            throw std::logic_error("node '" + name +
                                   "' references an unknown tensor");
    for (const auto &node : graph_.nodes_)
        if (node.name == name)
            throw std::logic_error("duplicate graph node name: " + name);
    CompleteHeContract(output_spec, kind == OpKind::kMpcToHe);
    const auto output = next_tensor_id_++;
    graph_.tensors_.emplace(output, std::move(output_spec));
    graph_.nodes_.push_back(
        {next_node_id_++, std::move(name), kind, std::move(inputs), output,
         std::move(attributes)});
    return output;
}

Result<TransformerGraph> GraphBuilder::Finish(TensorId output)
{
    if (graph_.tensors_.count(output) == 0)
        return Status::Invalid("graph output tensor does not exist");
    graph_.output_ = output;
    const auto valid = ValidateGraph(graph_);
    if (!valid.ok()) return valid.status();
    return std::move(graph_);
}

const TensorSpec &GraphBuilder::Spec(TensorId id) const
{
    const auto it = graph_.tensors_.find(id);
    if (it == graph_.tensors_.end())
        throw std::out_of_range("unknown tensor id");
    return it->second;
}

std::string ToString(OpKind value)
{
    switch (value) {
    case OpKind::kMatMulCipherPlain: return "matmul_cipher_plain";
    case OpKind::kMatMulCipherCipher: return "matmul_cipher_cipher";
    case OpKind::kElementwiseAdd: return "elementwise_add";
    case OpKind::kElementwiseMultiply: return "elementwise_multiply";
    case OpKind::kElementwiseAddPlain: return "elementwise_add_plain";
    case OpKind::kElementwiseMultiplyPlain: return "elementwise_multiply_plain";
    case OpKind::kScalarAdd: return "scalar_add";
    case OpKind::kScalarMultiply: return "scalar_multiply";
    case OpKind::kReduceSum: return "reduce_sum";
    case OpKind::kBroadcast: return "broadcast";
    case OpKind::kPolynomial: return "polynomial";
    case OpKind::kApplyPublicMask: return "apply_public_mask";
    case OpKind::kCompare: return "compare";
    case OpKind::kBooleanXor: return "boolean_xor";
    case OpKind::kBooleanNot: return "boolean_not";
    case OpKind::kMux: return "mux";
    case OpKind::kMpcReduceMax: return "mpc_reduce_max";
    case OpKind::kMpcPrepareMaskedScores: return "mpc_prepare_masked_scores";
    case OpKind::kMpcReduceSum: return "mpc_reduce_sum";
    case OpKind::kMpcMultiplyFixed: return "mpc_multiply_fixed";
    case OpKind::kMpcAdd: return "mpc_add";
    case OpKind::kReciprocal: return "reciprocal";
    case OpKind::kReciprocalSqrt: return "reciprocal_sqrt";
    case OpKind::kHeToMpc: return "he_to_mpc";
    case OpKind::kMpcToHe: return "mpc_to_he";
    case OpKind::kLocalShareRepack: return "local_share_repack";
    case OpKind::kHeRealifyComplex: return "he_realify_complex";
    }
    return "unknown";
}

} // namespace hera
