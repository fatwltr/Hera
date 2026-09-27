#pragma once

#include "hera/ffn_gelu.h"

#include <atomic>
#include <map>
#include <optional>

namespace hera {

struct Block5BridgeInputs {
    TensorHandle activation_shares;
    std::optional<TensorHandle> ffn_residual;
};

struct NextLayerBridgeOutputs {
    TensorHandle centered_shares;
    TensorHandle inverse_stddev_shares;
    Block2HeOutputs block2;
};

struct Block5BridgeOutputs {
    TensorHandle hidden_states;
    std::optional<NextLayerBridgeOutputs> next_layer;
};

struct Block5BridgeExecutionReport {
    std::size_t mpc_to_ckks_calls{};
    std::size_t ckks_to_mpc_calls{};
    std::size_t direct_residual_additions{};
    HeKernelOperationCounts he;
    AttentionLn2ExecutionReport layer_norm;
    Block2ExecutionReport next_block2;
    std::uint64_t conversion_communication_bytes{};
    std::uint64_t conversion_rounds{};
    double conversion_milliseconds{};
    std::vector<int> required_rotation_steps;
};

// Closes a Pre-LN Transformer layer without decrypting: GeLU shares are
// converted directly into the retained residual's HE context, projected by
// Wd, added to that residual and accurately realified. When a next layer is
// supplied, its Region-1 LayerNorm statistics and existing Block-2 executor
// are run in the same two-party call.
class SealSciBlock5BridgeExecutor {
public:
    SealSciBlock5BridgeExecutor(
        const HeraConfig &config,
        std::size_t current_layer_index,
        HeContextRegistry &contexts,
        SealCipherTensorStore &he_tensors,
        SealPlainParameterStore &parameters,
        CryptoTensorStore &mpc_tensors,
        CkksMpcConverter &converter,
        MpcLayerNormRuntime &mpc_layer_norm,
        TensorIdAllocator *ids = nullptr);

    static Result<std::map<std::string, std::vector<int>>>
    RequiredRotationSteps(const HeraConfig &config,
                          std::size_t current_layer_index,
                          bool include_next_layer = true);

    Result<Block5BridgeOutputs> Evaluate(
        const Block5BridgeInputs &inputs,
        const LayerWeightHandles &current_weights,
        const std::optional<LayerWeightHandles> &next_weights,
        Block5BridgeExecutionReport *report = nullptr);

private:
    const HeraConfig &config_;
    std::size_t current_layer_index_{};
    HeContextRegistry &contexts_;
    SealCipherTensorStore &he_tensors_;
    SealPlainParameterStore &parameters_;
    CryptoTensorStore &mpc_tensors_;
    CkksMpcConverter &converter_;
    MpcLayerNormRuntime &mpc_layer_norm_;
    TensorIdAllocator *ids_{};
    std::atomic<TensorId> next_id_{9000000};
};

} // namespace hera
