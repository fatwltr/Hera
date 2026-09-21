#pragma once

#include "hera/softmax.h"

#include <memory>

namespace hera {

class SharedMpcSession;

struct AttentionLn2Inputs {
    TensorHandle probability_shares;
    TensorHandle value;
    TensorHandle attention_residual;
};

struct AttentionLn2Outputs {
    TensorHandle centered_shares;
    TensorHandle inverse_stddev_shares;
    // Wo + attention residual before realification/LN2 statistics. This
    // ciphertext remains in the Region-2/4 shared context so Wd can target
    // its exact predecessor level and add the FFN skip without conversion.
    TensorHandle ffn_residual;
};

struct AttentionLn2ExecutionReport {
    std::size_t mpc_to_ckks_calls{};
    std::size_t ckks_to_mpc_calls{};
    std::size_t rotations{};
    std::size_t conjugations{};
    std::size_t ciphertext_plain_multiplications{};
    std::size_t ciphertext_multiplications{};
    std::size_t relinearizations{};
    std::size_t rescales{};
    std::size_t modulus_switches{};
    std::size_t reciprocal_square_roots{};
    std::uint64_t conversion_communication_bytes{};
    std::uint64_t conversion_rounds{};
    std::uint64_t mpc_communication_bytes{};
    std::uint64_t mpc_rounds{};
    double he_milliseconds{};
    double conversion_milliseconds{};
    double mpc_milliseconds{};
};

struct AttentionValuePlan {
    HePackingPlan probability_packing;
    HePackingPlan output_packing;
    PhysicalSlotMap probability_slot_map;
    std::vector<int> required_rotation_steps;
};

class AttentionLn2PackingPlanner {
public:
    static Result<AttentionValuePlan> AttentionValue(
        const TransformerLayerConfig &layer, std::size_t slot_count);
    static Result<HePackingPlan> ComplexStrictColumns(
        const TensorShape &shape, std::size_t slot_count);
    static Result<HePackingPlan> RealStrictColumns(
        const TensorShape &shape, std::size_t slot_count);
};

class AttentionLn2LevelPlanner {
public:
    static Result<void> Validate(const HeParameterConfig &parameters);
};

class MultiHeadAttentionValueKernel {
public:
    static Result<SealCipherTensor> Evaluate(
        const SealCipherTensor &probabilities,
        const SealCipherTensor &value,
        const TransformerLayerConfig &layer,
        const HeContext &context,
        AttentionLn2ExecutionReport *report = nullptr);
};

class AttentionOutputProjectionKernel {
public:
    static Result<SealCipherTensor> Evaluate(
        const SealCipherTensor &attention,
        const DenseMatrixParameter &weights,
        const std::optional<DenseVectorParameter> &bias,
        const TransformerLayerConfig &layer,
        const HeContext &context,
        AttentionLn2ExecutionReport *report = nullptr);
};

class AttentionResidualKernel {
public:
    // Mod-switches a same-context, higher-level residual to the Wo output's
    // exact parms_id and adds it. Packing and numerical scale must already
    // agree; this operation never rewrites scale metadata.
    static Result<SealCipherTensor> Add(
        SealCipherTensor projected,
        const SealCipherTensor &residual,
        const HeContext &context,
        AttentionLn2ExecutionReport *report = nullptr);
};

class ComplexRealifyKernel {
public:
    static Result<SealCipherTensor> Evaluate(
        const SealCipherTensor &input,
        const TransformerLayerConfig &layer,
        const HeContext &context,
        AttentionLn2ExecutionReport *report = nullptr);
};

struct LayerNormStatisticsOutputs {
    SealCipherTensor centered;
    SealCipherTensor variance;
};

class LayerNormStatisticsKernel {
public:
    static Result<LayerNormStatisticsOutputs> Evaluate(
        const SealCipherTensor &input,
        const TransformerLayerConfig &layer,
        const HeContext &context,
        AttentionLn2ExecutionReport *report = nullptr);
};

class MpcLayerNormRuntime {
public:
    MpcLayerNormRuntime(PartyRole role, std::string peer_address, int port);
    explicit MpcLayerNormRuntime(std::shared_ptr<SharedMpcSession> session);
    ~MpcLayerNormRuntime();
    MpcLayerNormRuntime(MpcLayerNormRuntime &&) noexcept;
    MpcLayerNormRuntime &operator=(MpcLayerNormRuntime &&) noexcept;
    MpcLayerNormRuntime(const MpcLayerNormRuntime &) = delete;
    MpcLayerNormRuntime &operator=(const MpcLayerNormRuntime &) = delete;

    Result<MpcShareTensor> ReciprocalSqrt(
        const MpcShareTensor &variance,
        int working_ring_bits,
        AttentionLn2ExecutionReport *report = nullptr);
    void ReleaseSession();
    void Cancel() noexcept;

private:
    class Impl;
    std::shared_ptr<SharedMpcSession> session_;
    std::shared_ptr<Impl> impl_;
};

class SealSciAttentionLn2Executor {
public:
    SealSciAttentionLn2Executor(
        const HeraConfig &config,
        const TransformerLayerConfig &layer,
        HeContextRegistry &contexts,
        SealCipherTensorStore &he_tensors,
        SealPlainParameterStore &parameters,
        CryptoTensorStore &mpc_tensors,
        CkksMpcConverter &converter,
        MpcLayerNormRuntime &mpc,
        TensorIdAllocator *ids = nullptr);

    static Result<std::vector<int>> RequiredRotationSteps(
        const TransformerLayerConfig &layer,
        const HeParameterConfig &parameters);

    Result<AttentionLn2Outputs> Evaluate(
        const AttentionLn2Inputs &inputs,
        const LayerWeightHandles &weights,
        AttentionLn2ExecutionReport *report = nullptr);

private:
    const HeraConfig &config_;
    const TransformerLayerConfig &layer_;
    HeContextRegistry &contexts_;
    SealCipherTensorStore &he_tensors_;
    SealPlainParameterStore &parameters_;
    CryptoTensorStore &mpc_tensors_;
    CkksMpcConverter &converter_;
    MpcLayerNormRuntime &mpc_;
    TensorIdAllocator *ids_{};
    std::atomic<TensorId> next_id_{7000000};
};

} // namespace hera
