#include "hera/private_transformer.h"
#include "hera/detail/parallel_ranges.h"

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
    bool obsolete_scale = false;
    for (const auto &node : graph.value().nodes()) {
        if (node.name == "layer0.ln1.gamma") {
            gamma = node.kind == hera::OpKind::kElementwiseMultiplyPlain;
            const auto *spec = graph.value().FindTensor(node.output);
            Require(spec && spec->planned_level == 1 &&
                        spec->planned_scale_bits == 44,
                    "LayerNorm gamma has the wrong Block-2 contract");
        }
        if (node.name == "layer0.ln1.beta")
            beta = node.kind == hera::OpKind::kElementwiseAddPlain;
        if (node.name == "layer0.attention.qk") {
            qk = true;
            const auto *spec = graph.value().FindTensor(node.output);
            Require(spec && spec->planned_level == 9 &&
                        spec->planned_scale_bits == 44,
                    "QK has the wrong terminal contract");
        }
        if (node.name == "layer0.attention.scale") obsolete_scale = true;
    }
    Require(gamma && beta && qk,
            "Block-2 typed nodes are missing from the Transformer graph");
    Require(!obsolete_scale,
            "QK scale was not folded into the Wq plaintext plan");
}

void TestFfnGeluGraphContract(hera::HeraConfig config)
{
    auto graph = hera::BuildTransformerGraph(config);
    Require(graph.ok(), graph.status().message());
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
                spec->planned_level == 2 &&
                spec->planned_scale_bits == 60;
        } else if (node.name == "layer0.ffn.Wu") {
            const auto *attributes =
                std::get_if<hera::MatMulAttributes>(&node.attributes);
            wu = attributes &&
                attributes->kernel == hera::HeKernelKind::kFfnUpProjection &&
                spec && spec->planned_level == 3;
        } else if (node.name == "layer0.ffn.activation.realify") {
            realify = node.kind == hera::OpKind::kHeRealifyComplex &&
                      spec && spec->planned_level == 4 &&
                      spec->planned_scale_bits == 46;
        } else if (node.name == "layer0.ffn.activation.F0") {
            f0 = node.kind == hera::OpKind::kPolynomial &&
                 spec && spec->planned_level == 7;
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
    Require(ln2_complex && wu && realify && f0 && f1 && boolean_logic &&
                output_share && wd_boundary,
            "typed LN2/Wu/GeLU/Wd boundary is incomplete");
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
    Require(result.status().message().find("layer0.ln1.sum") !=
                std::string::npos,
            "unavailable error lacks node context");
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
        TestPostNorm(config.value());
        TestRecordingExecution(config.value());
        TestBlock2GraphContract(config.value());
        TestFfnGeluGraphContract(config.value());
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
