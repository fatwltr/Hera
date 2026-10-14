#pragma once

#include "hera/block5_bridge.h"
#include "hera/pre_ln_entry.h"
#include "hera/execution_plan.h"
#include "hera/private_transformer.h"
#include "hera/runtime_coordinator.h"

#include <map>
#include <filesystem>

namespace hera {

// Raw model data is deliberately an in-memory API. File formats and model
// importers are outside the cryptographic runtime.
struct LayerWeightData {
    std::vector<double> wq, wk, wv, wo, wu, wd;
    std::map<std::string, std::vector<double>> vectors;
};

struct TransformerWeightData {
    std::vector<LayerWeightData> layers;
};

// Concrete fused-stage implementation behind PrivateTransformer::Evaluate.
// Construction is explicit so applications can establish their preferred
// control channel and key exchange before binding the runtime to HeraContext.
class SealSciTransformerRuntime final : public ITransformerExecutionRuntime {
public:
    SealSciTransformerRuntime(
        const HeraConfig &config, HeContextRegistry &contexts,
        RuntimeTensorArena &arena, SealPlainParameterStore &parameters,
        CkksMpcConverter &converter, MpcSoftmaxRuntime &softmax_mpc,
        MpcLayerNormRuntime &layer_norm_mpc, MpcGeluRuntime &gelu_mpc,
        RuntimeCoordinator *coordinator = nullptr,
        std::shared_ptr<const CompiledTransformerExecutionPlan> plan = {});

    Result<void> Prepare(
        const TransformerWeightHandles &weights,
        PreparationReport *report) override;

    Result<TensorHandle> Evaluate(
        const TensorHandle &input,
        const TransformerWeightHandles &weights,
        const PublicAttentionMask &mask,
        ExecutionReport *report) override;

    static Result<std::map<std::string, std::vector<int>>>
    RequiredRotationSteps(const HeraConfig &config);

private:
    const HeraConfig &config_;
    HeContextRegistry &contexts_;
    RuntimeTensorArena &arena_;
    SealPlainParameterStore &parameters_;
    CkksMpcConverter &converter_;
    MpcSoftmaxRuntime &softmax_mpc_;
    MpcLayerNormRuntime &layer_norm_mpc_;
    MpcGeluRuntime &gelu_mpc_;
    RuntimeCoordinator *coordinator_{};
    std::shared_ptr<const CompiledTransformerExecutionPlan> plan_;
    bool prepared_{};
    std::uint64_t prepared_manifest_{};
};

Result<TransformerWeightHandles> RegisterTransformerWeights(
    const HeraConfig &config,
    const std::optional<TransformerWeightData> &server_data,
    SealPlainParameterStore &parameters);

TransformerWeightData MakeDeterministicTransformerWeights(
    const HeraConfig &config, std::uint64_t seed);

struct PrivateRuntimeCliOptions {
    std::filesystem::path config;
    std::string party{"local"};
    std::string ip{"127.0.0.1"};
    int port{20500};
    std::size_t he_threads{1};
    std::size_t conversion_workers{1};
    std::size_t conversion_batch_elements{};
    std::size_t mpc_workers{}; // zero selects min(HE workers, 50 socket lanes)
    std::size_t he_memory_mib{512};
    std::size_t projection_baby_steps{};
    std::size_t plaintext_cache_mib{256};
    std::size_t stage_timeout_seconds{30};
    std::string seal_compression{"zstd"};
    std::uint64_t input_seed{1};
    std::uint64_t weight_seed{2};
    bool virtual_weights{};
    std::size_t weight_template_count{1};
    std::size_t warmup{};
    std::size_t repeat{1};
    bool verify{};
};

int RunPrivateRuntimeCli(const PrivateRuntimeCliOptions &options);

} // namespace hera
