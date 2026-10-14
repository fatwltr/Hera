#include "hera/private_transformer.h"
#include "hera/detail/parallel_ranges.h"
#include "hera/detail/nonlinear_contracts.h"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <atomic>
#include <stdexcept>

#ifndef HERA_TEST_CONFIG_PATH
#define HERA_TEST_CONFIG_PATH "configs/figure10_smoke.json"
#endif

namespace {

void Require(bool condition, const std::string &message)
{
    if (!condition) throw std::runtime_error(message);
}

hera::TransformerWeightHandles MakeWeights(const hera::HeraConfig &config)
{
    hera::TransformerWeightHandles output;
    hera::WeightId id = 1;
    for (const auto &layer : config.layers) {
        auto schema = hera::DeriveWeightSchema(layer);
        Require(schema.ok(), schema.status().message());
        hera::LayerWeightHandles handles{
            layer.id,
            {id++, schema.value().wq}, {id++, schema.value().wk},
            {id++, schema.value().wv}, {id++, schema.value().wo},
            {id++, schema.value().wu}, {id++, schema.value().wd}, {}};
        for (const auto &plain : schema.value().vectors)
            handles.plain_parameters.push_back({id++, plain});
        output.layers.push_back(std::move(handles));
    }
    return output;
}

void TestShapeVariants(hera::HeraConfig config)
{
    config.layers[0].kv_heads = 2;
    auto gqa = hera::DeriveWeightSchema(config.layers[0]);
    Require(gqa.ok(), "GQA schema rejected");
    Require(gqa.value().wk.columns == 128, "GQA Wk width is wrong");
    Require(gqa.value().wo.rows == 512, "GQA Wo width is wrong");

    config.layers[0].kv_heads = 1;
    auto mqa = hera::DeriveWeightSchema(config.layers[0]);
    Require(mqa.ok(), "MQA schema rejected");
    Require(mqa.value().wv.columns == 64, "MQA Wv width is wrong");

    config.layers[0].q_heads = 7;
    config.layers[0].kv_heads = 2;
    Require(!hera::ValidateLayerConfig(config.layers[0]).ok(),
            "invalid grouped-head mapping was accepted");
}

void TestTwoLayers(hera::HeraConfig config)
{
    auto second = config.layers[0];
    second.id = "layer1";
    second.activation = hera::ActivationKind::kSilu;
    config.layers.push_back(second);
    auto execution = config.layer_execution[0];
    execution.layer_id = second.id;
    config.layer_execution.push_back(execution);
    auto graph = hera::BuildTransformerGraph(config);
    Require(graph.ok(), graph.status().message());
    Require(graph.value().FindTensor(graph.value().input())->domain ==
                hera::Domain::kHeCiphertext,
            "two-layer input is not HE");
    Require(graph.value().FindTensor(graph.value().output())->domain ==
                hera::Domain::kHeCiphertext,
            "two-layer output is not HE");
}

void TestNormalizedResidualGraph(hera::HeraConfig config)
{
    config.layers.front().norm = hera::NormPlacement::kNormalizedResidual;
    auto graph = hera::BuildTransformerGraph(config);
    Require(graph.ok(), graph.status().message());
    const auto prefix = config.layers.front().id;
    const auto node = [&](const std::string &name) -> const hera::GraphNode & {
        for (const auto &candidate : graph.value().nodes())
            if (candidate.name == name) return candidate;
        throw std::runtime_error("missing normalized residual graph node: " + name);
    };
    const auto &ln1 = node(prefix + ".ln1.beta");
    const auto &attention_add = node(prefix + ".attention_output.residual_add");
    Require(attention_add.inputs.back() == ln1.output,
            "Wo residual does not originate at complete LN1 output");
    const auto &ln2 = node(prefix + ".ln2.residual_beta");
    const bool shared_ffn = config.layer_execution.front().regions.at(hera::HeRegionId::kRegion1).parameter_profile ==
        config.layer_execution.front().regions.at(hera::HeRegionId::kRegion5).parameter_profile;
    const auto &consumer = node(prefix + (shared_ffn ? ".ffn.residual_add" : ".ffn.residual.he_to_mpc"));
    Require((shared_ffn ? consumer.inputs.back() : consumer.inputs.front()) == ln2.output &&
                graph.value().FindTensor(ln2.output)->layout == hera::Layout::kStrictColumn &&
                graph.value().FindTensor(ln2.output)->planned_level ==
                    graph.value().FindTensor(node(prefix + ".ln2.beta").output)->planned_level,
            "Wd residual does not use original LN2 affine at the existing tail level");
    Require(hera::ToString(config.layers.front().norm) == "normalized_residual",
            "normalized residual mode is not explicit in configuration");
    config.layers.front().ffn_residual = false;
    auto without_skip = hera::BuildTransformerGraph(config);
    Require(without_skip.ok(), "disabling the normalized FFN residual invalidated the graph");
}

void TestPostNorm(hera::HeraConfig config)
{
    config.layers[0].norm = hera::NormPlacement::kPostNorm;
    auto graph = hera::BuildTransformerGraph(config);
    Require(graph.ok(), graph.status().message());
    Require(graph.value().FindTensor(graph.value().output())->domain ==
                hera::Domain::kHeCiphertext,
            "post-norm output is not HE");
}

void TestRecordingExecution(hera::HeraConfig config)
{
    auto context = hera::HeraContext::Create(config);
    Require(context.ok(), context.status().message());
    auto recording = std::make_shared<hera::RecordingState>();
    Require(context.value()->BindBackends(
                hera::MakeRecordingBackends(recording)).ok(),
            "cannot bind recording backends");
    auto transformer = hera::PrivateTransformer::Compile(context.value());
    Require(transformer.ok(), transformer.status().message());
    const auto *input_spec = context.value()->graph().FindTensor(
        context.value()->graph().input());
    hera::EncryptedTransformerInput input;
    input.hidden_states.tensor = {77, *input_spec};
    input.attention_mask.kind = config.layers[0].attention_mask;
    input.attention_mask.shape = {
        {config.layers[0].tokens, config.layers[0].tokens}};
    hera::ExecutionReport report;
    auto output = transformer.value().Evaluate(
        input, MakeWeights(config), &report);
    Require(output.ok(), output.status().message());
    Require(output.value().tensor.spec.domain ==
                hera::Domain::kHeCiphertext,
            "recording output is not HE");
    Require(recording->records.size() == context.value()->graph().nodes().size(),
            "recording backend did not receive every graph node");
    Require(report.nodes.size() == context.value()->graph().nodes().size(),
            "execution report is incomplete");

    std::size_t nonlinear = 0;
    for (const auto &node : context.value()->graph().nodes()) {
        const auto *spec = context.value()->graph().FindTensor(node.output);
        if (node.kind == hera::OpKind::kCompare ||
            node.kind == hera::OpKind::kMux ||
            node.kind == hera::OpKind::kReciprocal ||
            node.kind == hera::OpKind::kReciprocalSqrt) {
            ++nonlinear;
            Require(spec->domain == hera::Domain::kMpcArithmeticShare ||
                        spec->domain == hera::Domain::kMpcBooleanShare,
                    "nonlinear node is not in MPC");
        }
    }
    Require(nonlinear != 0, "graph has no fine-grained MPC nodes");
}

void TestBlock2GraphContract(hera::HeraConfig config)
{
    auto graph = hera::BuildTransformerGraph(config);
    Require(graph.ok(), graph.status().message());
    bool gamma = false;
    bool beta = false;
    bool qk = false;
    std::size_t projections = 0;
    bool obsolete_scale = false;
    for (const auto &node : graph.value().nodes()) {
        if (node.name == "layer0.attention.Wq" ||
            node.name == "layer0.attention.Wk" ||
            node.name == "layer0.attention.Wv") {
            const auto *spec = graph.value().FindTensor(node.output);
            Require(spec && spec->planned_level == 2,
                    "Q/K/V projections do not share the L1 input boundary");
            const auto *source = graph.value().FindTensor(node.inputs.front());
            Require(source && source->layout == hera::Layout::kStrictColumn,
                    "Q/K/V projection input lost its feature-column layout");
            ++projections;
        }
        if (node.name == "layer0.ln1.gamma") {
            gamma = node.kind == hera::OpKind::kElementwiseMultiplyPlain;
            const auto *spec = graph.value().FindTensor(node.output);
            Require(spec && spec->planned_level == 1 &&
                        spec->layout == hera::Layout::kStrictColumn &&
                        spec->planned_scale_bits == 44,
                    "LayerNorm gamma has the wrong Block-2 contract");
        }
        if (node.name == "layer0.ln1.beta")
            beta = node.kind == hera::OpKind::kElementwiseAddPlain;
        if (node.name == "layer0.attention.qk") {
            qk = true;
            const auto *spec = graph.value().FindTensor(node.output);
            Require(spec && spec->planned_level == 6 &&
                        spec->planned_scale_bits == 44,
                    "QK has the wrong terminal contract");
        }
        if (node.name == "layer0.attention.scale") obsolete_scale = true;
    }
    Require(gamma && beta && qk && projections == 3,
            "Block-2 typed nodes are missing from the Transformer graph");
    Require(!obsolete_scale,
            "QK scale was not folded into the Wq plaintext plan");
}

void TestFfnGeluGraphContract(hera::HeraConfig config)
{
    auto graph = hera::BuildTransformerGraph(config);
    Require(graph.ok(), graph.status().message());
    const auto &profile = config.he_parameter_profiles.at(config.layer_execution.front().regions.at(hera::HeRegionId::kRegion5).parameter_profile);
    const auto terminal = profile.coefficient_modulus_bits.size() - profile.terminal_prime_count;
    const auto start = terminal - hera::kFfnSuffixDepth;
    bool ln2_complex = false;
    bool wu = false;
    bool realify = false;
    bool f0 = false;
    bool f1 = false;
    bool boolean_logic = false;
    bool output_share = false;
    bool wd_boundary = false;
    for (const auto &node : graph.value().nodes()) {
        const auto *spec = graph.value().FindTensor(node.output);
        if (node.name == "layer0.ln2.gamma") {
            ln2_complex = spec &&
                spec->layout == hera::Layout::kComplexStrictColumn &&
                spec->planned_level == start + 1 &&
                spec->planned_scale_bits == 60;
        } else if (node.name == "layer0.ffn.Wu") {
            const auto *attributes =
                std::get_if<hera::MatMulAttributes>(&node.attributes);
            wu = attributes &&
                attributes->kernel == hera::HeKernelKind::kFfnUpProjection &&
                spec && spec->planned_level == start + 2 &&
                spec->layout == hera::Layout::kStrictColumn &&
                spec->planned_scale_bits == 46;
        } else if (node.name == "layer0.ffn.activation.realify") {
            realify = true;
        } else if (node.name == "layer0.ffn.activation.F0") {
            f0 = node.kind == hera::OpKind::kPolynomial &&
                 spec && spec->planned_level == terminal;
        } else if (node.name == "layer0.ffn.activation.F1") {
            f1 = node.kind == hera::OpKind::kPolynomial;
        } else if (node.kind == hera::OpKind::kBooleanXor ||
                   node.kind == hera::OpKind::kBooleanNot) {
            boolean_logic = true;
        } else if (node.name == "layer0.ffn.activation.output") {
            output_share = node.kind == hera::OpKind::kMpcAdd && spec &&
                spec->domain == hera::Domain::kMpcArithmeticShare;
        } else if (node.name == "layer0.ffn.activation.mpc_to_he") {
            wd_boundary = node.kind == hera::OpKind::kMpcToHe;
        }
    }
    Require(ln2_complex && wu && !realify && f0 && f1 && boolean_logic &&
                output_share && wd_boundary,
            "typed LN2/Wu/GeLU/Wd boundary is incomplete");
}


void TestRejectIndependentAttentionOutput(hera::HeraConfig config)
{
    auto &regions = config.layer_execution.front().regions;
    const auto &attention = regions.at(hera::HeRegionId::kRegion2).parameter_profile;
    // Equal parameters under a different profile still mean different keys.
    config.he_parameter_profiles.emplace("independent_output",
        config.he_parameter_profiles.at(attention));
    regions.at(hera::HeRegionId::kRegion4).parameter_profile = "independent_output";
    auto valid = hera::ValidateHeraConfig(config);
    Require(!valid.ok() && valid.status().message().find("same parameter profile") !=
                std::string::npos,
            "independent Region 4 was not rejected during config validation");
    Require(!hera::BuildTransformerGraph(config).ok(),
            "independent Region 4 unexpectedly built a graph");
}


void TestSharedAttentionOutput(hera::HeraConfig config)
{
    auto &regions = config.layer_execution.front().regions;
    regions.at(hera::HeRegionId::kRegion4).parameter_profile =
        regions.at(hera::HeRegionId::kRegion2).parameter_profile;
    auto graph = hera::BuildTransformerGraph(config);
    Require(graph.ok(), graph.status().message());
    const auto wd_level = config.layers.front().norm == hera::NormPlacement::kNormalizedResidual &&
        regions.at(hera::HeRegionId::kRegion1).parameter_profile == regions.at(hera::HeRegionId::kRegion5).parameter_profile
            ? config.he_parameter_profiles.at(regions.at(hera::HeRegionId::kRegion5).parameter_profile).coefficient_modulus_bits.size() - 4 : 3;
    bool refreshed = false, wo = false, wd = false;
    for (const auto &node : graph.value().nodes()) {
        const auto *spec = graph.value().FindTensor(node.output);
        if (node.name == "layer0.attention_output.attention_value_output_region.bridge.mpc_to_he")
            refreshed = spec && spec->planned_level == 2;
        if (node.name == "layer0.attention_output.Wo")
            wo = spec && spec->planned_level == 3 && spec->layout == hera::Layout::kStrictColumn;
        if (node.name == "layer0.ffn.Wd")
            wd = spec && spec->planned_level == wd_level && spec->layout == hera::Layout::kStrictColumn;
        Require(node.name != "layer0.attention_output.realify_complex" &&
                    node.name != "layer0.ffn.realify_complex",
                "shared real residual graph retained a separate realification");
    }
    const auto *output = graph.value().FindTensor(graph.value().output());
    Require(refreshed && wo && wd && output && output->planned_level ==
                (config.layers.front().norm == hera::NormPlacement::kNormalizedResidual &&
                 config.layer_execution.front().regions.at(hera::HeRegionId::kRegion1).parameter_profile ==
                 config.layer_execution.front().regions.at(hera::HeRegionId::kRegion5).parameter_profile
                    ? config.he_parameter_profiles.at(regions.at(hera::HeRegionId::kRegion5).parameter_profile).coefficient_modulus_bits.size() - 4 : 3) &&
                output->layout == hera::Layout::kStrictColumn,
            "shared Region-2/4 graph contract is incomplete");
}

void TestUnavailableBackend(hera::HeraConfig config)
{
    auto context = hera::HeraContext::Create(config);
    Require(context.ok(), context.status().message());
    auto transformer = hera::PrivateTransformer::Compile(context.value());
    Require(transformer.ok(), transformer.status().message());
    const auto *input_spec = context.value()->graph().FindTensor(
        context.value()->graph().input());
    hera::EncryptedTransformerInput input;
    input.hidden_states.tensor = {88, *input_spec};
    input.attention_mask.kind = config.layers[0].attention_mask;
    input.attention_mask.shape = {
        {config.layers[0].tokens, config.layers[0].tokens}};
    auto result = transformer.value().Evaluate(input, MakeWeights(config));
    Require(!result.ok(), "unavailable backend unexpectedly executed");
    Require(result.status().code() == hera::StatusCode::kKernelUnavailable,
            "unavailable backend returned the wrong status code");
    Require(result.status().message().find(
                context.value()->graph().nodes().front().name) != std::string::npos,
            "unavailable error lacks the first node's context");
}

void TestConversionWorkspaceValidation(hera::HeraConfig config)
{
    auto &profile = config.he_parameter_profiles.begin()->second;
    profile.default_scale_bits = 100;
    profile.input_scale_bits = 100;
    profile.output_scale_bits = 100;
    Require(!hera::ValidateHeraConfig(config).ok(),
            "conversion workspace wider than 127 bits was accepted");
}

void TestLayerNormRsqrtWidthValidation(hera::HeraConfig config)
{
    const auto baseline = config;
    config.layers.front().attention_norm_rsqrt_bits =
        config.mpc_numeric.scale_bits + 1;
    Require(!hera::ValidateHeraConfig(config).ok(),
            "rsqrt ring without a signed integer bit was accepted");

    config.layers.front().attention_norm_rsqrt_bits =
        config.mpc_numeric.ring_bits + 1;
    Require(!hera::ValidateHeraConfig(config).ok(),
            "rsqrt ring wider than the MPC tensor was accepted");

    config.layers.front().attention_norm_rsqrt_bits =
        config.mpc_numeric.ring_bits;
    Require(hera::ValidateHeraConfig(config).ok(),
            "full-width rsqrt working ring was rejected");
    config.layers.front().attention_norm_variance_scale_bits = 0;
    Require(!hera::ValidateHeraConfig(config).ok(), "zero variance precision was accepted");
    config.layers.front().attention_norm_variance_scale_bits = config.mpc_numeric.ring_bits;
    Require(!hera::ValidateHeraConfig(config).ok(), "variance precision outside the ring was accepted");
    config.layers.front().attention_norm_variance_scale_bits = 24;
    config.layers.front().attention_norm_epsilon = 1e-8;
    Require(!hera::ValidateHeraConfig(config).ok(), "epsilon below the variance precision envelope was accepted");

    config = baseline;
    config.mpc_numeric.scale_bits = 20; // BLB's CKKS Transformer example.
    config.layers.front().attention_norm_rsqrt_bits = 27;
    Require(hera::ValidateHeraConfig(config).ok(), "BLB scale20 rsqrt configuration was rejected");
    for (int scale : {29, 30, 31}) {
        config = baseline;
        config.mpc_numeric.scale_bits = scale;
        config.layers.front().attention_norm_rsqrt_bits = scale + 3;
        auto rejected = hera::ValidateHeraConfig(config);
        Require(!rejected.ok() && rejected.status().message().find("LUT capacity") != std::string::npos,
                "unsupported SCI sqrt LUT was not rejected during config validation");
    }
    Require(hera::detail::PlanLayerNormRsqrt(63, 28, 40, 28).ok(),
            "SCI's supported scale28 LUT boundary was rejected");
    Require(!hera::detail::PlanLayerNormRsqrt(43, 28, 32, 28).ok(),
            "rsqrt output wider than the sharing ring was accepted");
    Require(!hera::detail::SoftmaxAccumulatorBits(128, 63, 28).ok(),
            "Softmax product wider than the 64-bit backend was accepted");
    Require(hera::detail::SoftmaxAccumulatorBits(2, 63, 28).ok(),
            "supported high-precision Softmax workspace was rejected");
}

void TestHeWorkerBudgetAndReuse()
{
    auto &pool = hera::detail::HeWorkerPool::Instance();
    pool.SetMemoryBudget(128);
    std::atomic<int> visited{0};
    for (int repetition = 0; repetition < 2; ++repetition) {
        auto done = hera::detail::ParallelRanges(
            4, 4, 1,
            [&](std::size_t begin, std::size_t end) -> hera::Result<void> {
                visited += static_cast<int>(end - begin);
                return {};
            }, 64);
        Require(done.ok(), done.status().message());
        Require(pool.WorkerThreadCount() == 1,
                "HE memory budget did not limit reusable worker count");
    }
    Require(visited == 8, "HE worker pool skipped an output range");
}

} // namespace

int main()
{
    try {
        auto config = hera::LoadHeraConfig(HERA_TEST_CONFIG_PATH);
        Require(config.ok(), config.status().message());
        TestShapeVariants(config.value());
        TestTwoLayers(config.value());
        TestNormalizedResidualGraph(config.value());
        TestPostNorm(config.value());
        TestRecordingExecution(config.value());
        TestBlock2GraphContract(config.value());
        TestFfnGeluGraphContract(config.value());
        TestRejectIndependentAttentionOutput(config.value());
        TestSharedAttentionOutput(config.value());
        TestUnavailableBackend(config.value());
        TestConversionWorkspaceValidation(config.value());
        TestLayerNormRsqrtWidthValidation(config.value());
        TestHeWorkerBudgetAndReuse();
        std::cout << "HERA_FRAMEWORK_STATUS=PASS\n";
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << "HERA_FRAMEWORK_STATUS=FAIL\n"
                  << "HERA_FRAMEWORK_ERROR=" << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
