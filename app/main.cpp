#include "hera/private_transformer.h"
#ifdef HERA_PRIVATE_RUNTIME_BUILT
#include "hera/private_runtime.h"
#endif

#include <cstdlib>
#include <iostream>

#ifndef HERA_DEFAULT_CONFIG_PATH
#define HERA_DEFAULT_CONFIG_PATH "configs/figure10_smoke.json"
#endif
namespace {

struct CliOptions {
    std::string config{HERA_DEFAULT_CONFIG_PATH};
    bool dry_run{};
    bool dump_graph{};
    bool record_execution{};
    std::string party;
    std::string ip{"127.0.0.1"};
    int port{20500};
    std::size_t threads{1};
    std::size_t he_threads{1};
    std::size_t conversion_workers{1};
    std::size_t he_memory_mib{512};
    std::string seal_compression{"none"};
    std::uint64_t input_seed{1};
    std::uint64_t weight_seed{2};
    bool virtual_weights{};
    std::size_t weight_template_count{1};
    std::size_t warmup{};
    std::size_t repeat{1};
    bool verify{};
};

hera::Result<CliOptions> ParseCli(int argc, char **argv)
{
    CliOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--dry-run") options.dry_run = true;
        else if (argument == "--dump-graph") options.dump_graph = true;
        else if (argument == "--record-execution")
            options.record_execution = true;
        else if (argument == "--config" && i + 1 < argc)
            options.config = argv[++i];
        else if (argument.rfind("--config=", 0) == 0)
            options.config = argument.substr(9);
        else if (argument.rfind("--party=", 0) == 0)
            options.party = argument.substr(8);
        else if (argument.rfind("--ip=", 0) == 0)
            options.ip = argument.substr(5);
        else if (argument.rfind("--port=", 0) == 0)
            options.port = std::stoi(argument.substr(7));
        else if (argument.rfind("--threads=", 0) == 0)
            options.threads = options.he_threads =
                options.conversion_workers = std::stoul(argument.substr(10));
        else if (argument.rfind("--he-threads=", 0) == 0)
            options.he_threads = std::stoul(argument.substr(13));
        else if (argument.rfind("--conversion-workers=", 0) == 0)
            options.conversion_workers = std::stoul(argument.substr(21));
        else if (argument.rfind("--he-memory-mib=", 0) == 0)
            options.he_memory_mib = std::stoul(argument.substr(16));
        else if (argument.rfind("--seal-compression=", 0) == 0)
            options.seal_compression = argument.substr(19);
        else if (argument.rfind("--input-seed=", 0) == 0)
            options.input_seed = std::stoull(argument.substr(13));
        else if (argument.rfind("--weight-seed=", 0) == 0)
            options.weight_seed = std::stoull(argument.substr(14));
        else if (argument == "--weight-mode=materialized")
            options.virtual_weights = false;
        else if (argument == "--weight-mode=virtual")
            options.virtual_weights = true;
        else if (argument.rfind("--weight-mode=", 0) == 0)
            return hera::Status::Invalid(
                "--weight-mode must be materialized or virtual");
        else if (argument.rfind("--weight-template-count=", 0) == 0)
            options.weight_template_count = std::stoul(argument.substr(24));
        else if (argument.rfind("--warmup=", 0) == 0)
            options.warmup = std::stoul(argument.substr(9));
        else if (argument.rfind("--repeat=", 0) == 0)
            options.repeat = std::stoul(argument.substr(9));
        else if (argument == "--verify") options.verify = true;
        else
            return hera::Status::Invalid("unknown CLI argument: " + argument);
    }
    return options;
}

hera::TransformerWeightHandles MakeWeightHandles(
    const hera::HeraConfig &config)
{
    hera::TransformerWeightHandles output;
    hera::WeightId next_id = 1;
    for (const auto &layer : config.layers) {
        const auto schema = hera::DeriveWeightSchema(layer).value();
        hera::LayerWeightHandles handles{
            layer.id,
            {next_id++, schema.wq}, {next_id++, schema.wk},
            {next_id++, schema.wv}, {next_id++, schema.wo},
            {next_id++, schema.wu}, {next_id++, schema.wd}, {}};
        for (const auto &plain : schema.vectors)
            handles.plain_parameters.push_back({next_id++, plain});
        output.layers.push_back(std::move(handles));
    }
    return output;
}

void PrintError(const hera::Status &status)
{
    std::cerr << "HERA_STATUS=ERROR\nHERA_ERROR=" << status.message() << '\n';
}

} // namespace

int main(int argc, char **argv)
{
    const auto options = ParseCli(argc, argv);
    if (!options.ok()) {
        PrintError(options.status());
        return EXIT_FAILURE;
    }
    auto context = hera::HeraContext::CreateFromJson(options.value().config);
    if (!context.ok()) {
        PrintError(context.status());
        return EXIT_FAILURE;
    }

    std::cout << "HERA_CONFIG_STATUS=PASS\n"
              << "HERA_LAYER_COUNT=" << context.value()->config().layers.size()
              << "\nHERA_GRAPH_NODE_COUNT="
              << context.value()->graph().nodes().size() << '\n';
    for (const auto &entry : context.value()->config().he_parameter_profiles) {
        std::cout << "HERA_HE_PROFILE=" << entry.first
                  << ",N=" << entry.second.polynomial_degree
                  << ",SCALE_BITS=" << entry.second.default_scale_bits
                  << ",DECLARED_DEPTH=" << entry.second.declared_depth
                  << ",SECURITY=" << hera::ToString(entry.second.security_level)
                  << '\n';
        if (entry.second.security_level == hera::SecurityLevel::kNone)
            std::cout << "HERA_SECURITY_WARNING=" << entry.first
                      << " uses security_level=none for research only\n";
    }
#ifdef HERA_SOFTMAX_TO_MPC_BUILT
    std::cout << "SOFTMAX_TO_MPC_READY=YES\n";
#else
    std::cout << "SOFTMAX_TO_MPC_READY=NO\n";
#endif
#ifdef HERA_ATTENTION_LN2_BUILT
    std::cout << "SOFTMAX_MPC_TO_HE_READY=YES\n"
              << "SOFTMAX_AV_INTEGRATION_READY=YES\n"
              << "ATTENTION_LN2_TO_MPC_READY=YES\n";
#else
    std::cout << "SOFTMAX_MPC_TO_HE_READY=NO\n"
              << "SOFTMAX_AV_INTEGRATION_READY=NO\n"
              << "ATTENTION_LN2_TO_MPC_READY=NO\n";
#endif
#ifdef HERA_FFN_GELU_BUILT
    std::cout << "FFN_GELU_TO_MPC_READY=YES\n";
#else
    std::cout << "FFN_GELU_TO_MPC_READY=NO\n";
#endif
#ifdef HERA_BLOCK5_BRIDGE_BUILT
    std::cout << "GELU_MPC_TO_HE_READY=YES\n"
              << "FFN_WD_INTEGRATION_READY=YES\n"
              << "FFN_RESIDUAL_DIRECT_ADD_READY=YES\n"
              << "CROSS_LAYER_BLOCK2_READY=YES\n";
#else
    std::cout << "GELU_MPC_TO_HE_READY=NO\n"
              << "FFN_WD_INTEGRATION_READY=NO\n"
              << "FFN_RESIDUAL_DIRECT_ADD_READY=NO\n"
              << "CROSS_LAYER_BLOCK2_READY=NO\n";
#endif
#ifdef HERA_PRIVATE_RUNTIME_BUILT
    std::cout << "HERA_END_TO_END_READY=YES\n";
#else
    std::cout << "HERA_END_TO_END_READY=NO\n";
#endif
    if (options.value().dump_graph)
        std::cout << context.value()->graph().Dump();
    if (options.value().dry_run && !options.value().record_execution) {
        std::cout << "HERA_DRY_RUN_STATUS=PASS\n";
        return EXIT_SUCCESS;
    }
#ifdef HERA_PRIVATE_RUNTIME_BUILT
    if (!options.value().party.empty()) {
        hera::PrivateRuntimeCliOptions runtime_options;
        runtime_options.config = options.value().config;
        runtime_options.party = options.value().party;
        runtime_options.ip = options.value().ip;
        runtime_options.port = options.value().port;
        runtime_options.he_threads = options.value().he_threads;
        runtime_options.conversion_workers =
            options.value().conversion_workers;
        runtime_options.he_memory_mib = options.value().he_memory_mib;
        runtime_options.seal_compression = options.value().seal_compression;
        runtime_options.input_seed = options.value().input_seed;
        runtime_options.weight_seed = options.value().weight_seed;
        runtime_options.virtual_weights = options.value().virtual_weights;
        runtime_options.weight_template_count =
            options.value().weight_template_count;
        runtime_options.warmup = options.value().warmup;
        runtime_options.repeat = options.value().repeat;
        runtime_options.verify = options.value().verify;
        return hera::RunPrivateRuntimeCli(runtime_options);
    }
#endif
    if (!options.value().record_execution) {
        std::cout << "HERA_BACKEND_STATUS=UNAVAILABLE\n";
        return EXIT_SUCCESS;
    }

    auto recording = std::make_shared<hera::RecordingState>();
    const auto bound = context.value()->BindBackends(
        hera::MakeRecordingBackends(recording));
    if (!bound.ok()) {
        PrintError(bound.status());
        return EXIT_FAILURE;
    }
    auto transformer = hera::PrivateTransformer::Compile(context.value());
    if (!transformer.ok()) {
        PrintError(transformer.status());
        return EXIT_FAILURE;
    }
    const auto *input_spec = context.value()->graph().FindTensor(
        context.value()->graph().input());
    hera::EncryptedTransformerInput input;
    input.hidden_states.tensor = {1, *input_spec};
    const auto &first = context.value()->config().layers.front();
    input.attention_mask.kind = first.attention_mask;
    if (first.attention_mask != hera::AttentionMaskKind::kNone)
        input.attention_mask.shape = {{first.tokens, first.tokens}};
    if (first.attention_mask == hera::AttentionMaskKind::kPublicPadding)
        input.attention_mask.additive_values.assign(
            first.tokens * first.tokens, 0.0);

    hera::ExecutionReport report;
    auto result = transformer.value().Evaluate(
        input, MakeWeightHandles(context.value()->config()), &report);
    if (!result.ok()) {
        PrintError(result.status());
        return EXIT_FAILURE;
    }
    std::cout << "HERA_RECORD_COUNT=" << recording->records.size()
              << "\nHERA_OUTPUT_DOMAIN="
              << hera::ToString(result.value().tensor.spec.domain)
              << "\nHERA_RECORDING_STATUS=PASS\n";
    return EXIT_SUCCESS;
}
