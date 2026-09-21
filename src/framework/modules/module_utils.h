#pragma once

#include "hera/modules.h"

#include <stdexcept>

namespace hera::module_internal {

inline TensorSpec HeSpec(const TensorSpec &source, TensorShape shape,
                         Layout layout, HeRegionId region)
{
    TensorSpec output = source;
    output.shape = std::move(shape);
    output.domain = Domain::kHeCiphertext;
    output.layout = layout;
    output.he_region = region;
    output.he_parameter_profile.reset();
    output.planned_level.reset();
    return output;
}

inline TensorSpec ArithmeticShareSpec(const TensorSpec &source,
                                      TensorShape shape, Layout layout)
{
    TensorSpec output = source;
    output.shape = std::move(shape);
    output.domain = Domain::kMpcArithmeticShare;
    output.layout = layout;
    output.he_region.reset();
    output.he_parameter_profile.reset();
    output.planned_level.reset();
    output.planned_scale_bits.reset();
    return output;
}

inline TensorSpec BooleanShareSpec(const TensorSpec &source,
                                   TensorShape shape, Layout layout)
{
    TensorSpec output = ArithmeticShareSpec(source, std::move(shape), layout);
    output.domain = Domain::kMpcBooleanShare;
    output.numeric = {1, 0};
    return output;
}

inline TensorId HeToMpc(GraphBuilder &graph, TensorId input, Layout layout,
                        const std::string &name)
{
    const auto &source = graph.Spec(input);
    auto output = ArithmeticShareSpec(source, source.shape, layout);
    ConversionAttributes attributes;
    attributes.source_domain = Domain::kHeCiphertext;
    attributes.target_domain = Domain::kMpcArithmeticShare;
    attributes.target_layout = layout;
    attributes.target_numeric = output.numeric;
    return graph.AddNode(name, OpKind::kHeToMpc, {input}, output, attributes);
}

inline TensorId MpcToHe(GraphBuilder &graph, TensorId input,
                        HeRegionId region, Layout layout,
                        const std::string &name,
                        std::optional<std::size_t> target_level = std::nullopt)
{
    const auto &source = graph.Spec(input);
    auto output = HeSpec(source, source.shape, layout, region);
    output.planned_level = target_level;
    ConversionAttributes attributes;
    attributes.source_domain = Domain::kMpcArithmeticShare;
    attributes.target_domain = Domain::kHeCiphertext;
    attributes.target_region = region;
    attributes.target_layout = layout;
    attributes.target_numeric = output.numeric;
    attributes.target_level = target_level;
    return graph.AddNode(name, OpKind::kMpcToHe, {input}, output, attributes);
}

inline TensorId EnsureRegion(GraphBuilder &graph, TensorId input,
                             HeRegionId region, Layout layout,
                             const std::string &prefix)
{
    const auto &source = graph.Spec(input);
    if (source.domain == Domain::kHeCiphertext &&
        source.he_parameter_profile == graph.ActiveProfile(region) &&
        source.layout == layout)
        return input;
    if (source.domain == Domain::kMpcArithmeticShare)
        return MpcToHe(graph, input, region, layout,
                       prefix + ".bridge.mpc_to_he");
    if (source.domain != Domain::kHeCiphertext)
        throw std::invalid_argument(
            prefix + ": only HE ciphertext or arithmetic shares can enter an HE region");
    const auto share = HeToMpc(graph, input, layout,
                               prefix + ".bridge.he_to_mpc");
    return MpcToHe(graph, share, region, layout,
                   prefix + ".bridge.mpc_to_he");
}

} // namespace hera::module_internal
