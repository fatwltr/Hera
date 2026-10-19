#pragma once

#include "hera/blb_reference.h"

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
    // Pre-LN only: real-column Wo + attention residual before LN2.
    // Invalid in normalized_residual mode: its FFN skip is instead returned
    // by FfnGeluOutputs after the full original LN2 affine.
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
    int variance_scale_bits{}, rsqrt_input_ring_bits{}, rsqrt_output_ring_bits{};
    std::uint64_t conversion_communication_bytes{};
    std::uint64_t conversion_rounds{};
    std::uint64_t mpc_communication_bytes{};
    std::uint64_t mpc_rounds{};
    double he_milliseconds{};
    double conversion_milliseconds{};
    double mpc_milliseconds{};
    // A*V-only rotation counts; their sum equals this kernel's rotations.
    std::size_t attention_probability_rotations{};
    std::size_t attention_value_rotations{};
    std::size_t attention_recovery_rotations{};
    // Actual statistics workers; encoding count excludes cache hits.
    std::size_t layer_norm_plaintext_encodings{};
    std::size_t layer_norm_workers{};
    std::size_t reciprocal_sqrt_workers{};
    std::size_t attention_value_plaintext_encodings{};
    // Real Wo BSGS only; counts actual executed operations.
    std::size_t wo_baby_rotations{};
    std::size_t wo_giant_rotations{};
    std::size_t wo_output_conjugations{};
    std::size_t wo_phase_multiplications{};
    // Peak actual concurrency of the A/V preparation and giant products.
    std::size_t attention_preparation_workers{};
    std::size_t attention_product_workers{};
};

struct AttentionValuePlan {
    // Lane count is the largest power of two <= slots/(next_power_of_two(tokens)*tile_heads).
    // Consecutive attention diagonals share one slot's real/imaginary
    // components. One complex A ciphertext is one BSGS giant group, with up
    // to twice the lane count in baby steps. Paired replication shares one
    // conjugation per complex diagonal; odd components' -i extraction is
    // folded into V preprocessing to preserve the three-level kernel depth.
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
    // Legacy continuous suffix validator for standalone kernels without an AV refresh.
    static Result<void> Validate(const HeParameterConfig &parameters);
    // Refreshed Region-4 suffix: one Wo level followed by three statistics levels.
    static Result<void> ValidateOutput(const HeParameterConfig &parameters);
    static std::size_t OutputInputLevel(const HeParameterConfig &parameters);
};

class MultiHeadAttentionValueKernel {
public:
    static Result<void> Prepare(const TransformerLayerConfig &layer,
        const HeContext &context, std::size_t input_level,
        AttentionLn2ExecutionReport *report = nullptr);
    static Result<SealCipherTensor> Evaluate(
        const SealCipherTensor &probabilities,
        const SealCipherTensor &value,
        const TransformerLayerConfig &layer,
        const HeContext &context,
        AttentionLn2ExecutionReport *report = nullptr,
        std::size_t he_workers = 1);
};

struct AttentionOutputBsgsPlan {
    std::size_t token_stride{};
    std::size_t columns_per_ciphertext{};
    std::size_t input_ciphertexts{};
    std::size_t output_ciphertexts{};
    std::size_t baby_steps{};
    std::size_t giant_steps{};
    std::size_t rotation_upper_bound{};
};

class AttentionOutputProjectionKernel {
public:
    static Result<AttentionOutputBsgsPlan> RealColumnBsgs(
        const TransformerLayerConfig &layer, std::size_t slot_count);
    // Production real-output Wo only; excludes A*V and LN statistics keys.
    static Result<std::vector<int>> RequiredRealRotationSteps(
        const TransformerLayerConfig &layer, std::size_t slot_count);

    static Result<SealCipherTensor> Evaluate(
        const SealCipherTensor &attention,
        const DenseMatrixParameter &weights,
        const std::optional<DenseVectorParameter> &bias,
        const TransformerLayerConfig &layer,
        const HeContext &context,
        AttentionLn2ExecutionReport *report = nullptr,
        std::size_t he_workers = 1);

    // Production Wo: complex-input BSGS with two real-only weight channels.
    // A scale-1 fixed phase combines channels before each giant rotation;
    // shared babies and final realification consume one rescale in total.
    static Result<SealCipherTensor> EvaluateReal(
        const SealCipherTensor &attention,
        const DenseMatrixParameter &weights,
        const std::optional<DenseVectorParameter> &bias,
        const TransformerLayerConfig &layer,
        const HeContext &context,
        AttentionLn2ExecutionReport *report = nullptr,
        std::size_t he_workers = 1);
};

class AttentionResidualKernel {
public:
    // Mod-switches a same-context, higher-level real or complex residual to the Wo output's
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
    // Prepare bounded constant-plaintext caches from real ciphertexts or
    // metadata-only inputs. Execute on the same calling thread as Evaluate.
    static Result<void> Prepare(
        const SealCipherTensor &input,
        const TransformerLayerConfig &layer,
        const HeContext &context,
        AttentionLn2ExecutionReport *report = nullptr,
        std::optional<int> output_scale_bits = std::nullopt);
    // Uses the requested workers up to the input count and shared HE memory
    // budget. Nested calls on the HE pool execute serially.
    static Result<LayerNormStatisticsOutputs> Evaluate(
        const SealCipherTensor &input,
        const TransformerLayerConfig &layer,
        const HeContext &context,
        AttentionLn2ExecutionReport *report = nullptr,
        std::size_t he_workers = 1,
        std::optional<int> output_scale_bits = std::nullopt);
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

    // configured ring width describes the integer range at output precision.
    // Finer variance input raises working width; output precision can stay
    // global. A secure aggregate guard rejects zero/negative/out-of-range input.
    // BLB/SCI's inverse-sqrt LUT supports output scales 1..28; native
    // workspace and sharing-ring capacity are checked before session setup.
    Result<MpcShareTensor> ReciprocalSqrt(
        const MpcShareTensor &variance,
        int working_ring_bits,
        AttentionLn2ExecutionReport *report = nullptr,
        std::optional<int> output_scale_bits = std::nullopt);
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
        BlbReferenceConverter &converter,
        MpcLayerNormRuntime &mpc,
        TensorIdAllocator *ids = nullptr);

    static Result<std::vector<int>> RequiredRotationSteps(
        const TransformerLayerConfig &layer,
        const HeParameterConfig &parameters);
    static Result<std::vector<int>> RequiredOutputRotationSteps(
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
    BlbReferenceConverter &converter_;
    MpcLayerNormRuntime &mpc_;
    TensorIdAllocator *ids_{};
    std::atomic<TensorId> next_id_{7000000};
};

} // namespace hera
