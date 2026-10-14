#pragma once

#include "hera/attention_ln2.h"
#include "hera/execution_plan.h"

namespace hera {

struct RuntimeTensorArena {
    TensorIdAllocator ids;
    SealCipherTensorStore he;
    CryptoTensorStore mpc;
};

struct PreLnEntryOutputs {
    Block2HeOutputs block2;
    TensorHandle attention_residual;
};

// Optional compatibility outputs, owned by the caller when requested.
struct PreLnEntryShareOutputs {
    TensorHandle centered_shares;
    TensorHandle inverse_stddev_shares;
};

struct PreLnEntryExecutionReport {
    std::size_t ckks_to_mpc_calls{};
    std::size_t mpc_to_ckks_calls{};
    std::uint64_t conversion_communication_bytes{};
    std::uint64_t conversion_rounds{};
    double conversion_milliseconds{};
    double mpc_milliseconds{};
    double he_milliseconds{};
    Block2ExecutionReport block2;
    AttentionLn2ExecutionReport statistics;
};

// Turns an encrypted hidden state into fresh Block-2 inputs and preserves an
// attention residual. Region 2/4 share a profile. Standard Pre-LN retains
// the original hidden; normalized_residual retains the complete LN1 output
// produced alongside the gamma-fused QKV branch, without extra depth.
class SealSciPreLnEntryExecutor {
public:
    SealSciPreLnEntryExecutor(
        const HeraConfig &config, std::size_t layer_index,
        HeContextRegistry &contexts, RuntimeTensorArena &arena,
        SealPlainParameterStore &parameters, CkksMpcConverter &converter,
        MpcLayerNormRuntime &mpc_layer_norm,
        const CompiledLayerExecutionPlan *plan = nullptr);

    SealSciPreLnEntryExecutor(
        const HeraConfig &config, std::size_t layer_index,
        HeContextRegistry &contexts, SealCipherTensorStore &he_tensors,
        CryptoTensorStore &mpc_tensors, SealPlainParameterStore &parameters,
        CkksMpcConverter &converter, MpcLayerNormRuntime &mpc_layer_norm,
        TensorIdAllocator *ids = nullptr,
        const CompiledLayerExecutionPlan *plan = nullptr);

    Result<PreLnEntryOutputs> Evaluate(
        const TensorHandle &hidden, const LayerWeightHandles &weights,
        PreLnEntryExecutionReport *report = nullptr,
        PreLnEntryShareOutputs *retained_shares = nullptr);

private:
    const HeraConfig &config_;
    std::size_t layer_index_{};
    HeContextRegistry &contexts_;
    SealCipherTensorStore &he_tensors_;
    CryptoTensorStore &mpc_tensors_;
    TensorIdAllocator *ids_{};
    TensorIdAllocator owned_ids_;
    SealPlainParameterStore &parameters_;
    CkksMpcConverter &converter_;
    MpcLayerNormRuntime &mpc_layer_norm_;
    const CompiledLayerExecutionPlan *plan_{};
};

} // namespace hera
