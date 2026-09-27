#pragma once

#include "hera/attention_ln2.h"

#include <atomic>
#include <memory>

namespace hera {

class SharedMpcSession;

struct FfnGeluInputs {
    TensorHandle centered_shares;
    TensorHandle inverse_stddev_shares;
};

struct FfnGeluOutputs {
    TensorHandle activation_shares;
};

struct FfnGeluExecutionReport {
    std::size_t mpc_to_ckks_calls{};
    std::size_t ckks_to_mpc_calls{};
    std::size_t rotations{};
    std::size_t conjugations{};
    std::size_t ciphertext_plain_multiplications{};
    std::size_t ciphertext_multiplications{};
    std::size_t relinearizations{};
    std::size_t rescales{};
    std::size_t modulus_switches{};
    std::size_t comparisons{};
    std::size_t muxes{};
    std::uint64_t conversion_communication_bytes{};
    std::uint64_t conversion_rounds{};
    std::uint64_t mpc_communication_bytes{};
    std::uint64_t mpc_rounds{};
    double he_milliseconds{};
    double conversion_milliseconds{};
    double mpc_milliseconds{};
    std::vector<int> required_rotation_steps;
};

struct FfnGeluPackingPlan {
    HePackingPlan centered;
    HePackingPlan inverse_broadcast;
    HePackingPlan projected;
    HePackingPlan realified;
    std::size_t token_stride{};
    std::size_t columns_per_component{};
};

struct HeKernelOperationCounts {
    std::size_t rotations{};
    std::size_t conjugations{};
    std::size_t ciphertext_plain_multiplications{};
    std::size_t ciphertext_multiplications{};
    std::size_t relinearizations{};
    std::size_t rescales{};
    std::size_t modulus_switches{};
};

class FfnGeluPackingPlanner {
public:
    static Result<FfnGeluPackingPlan> Build(
        const TransformerLayerConfig &layer, std::size_t slot_count);
    static Result<HePackingPlan> ComplexColumns(
        const TensorShape &shape, std::size_t slot_count, Layout layout);
    static Result<HePackingPlan> RealColumns(
        const TensorShape &shape, std::size_t slot_count, Layout layout);
};

class FfnGeluLevelPlanner {
public:
    static Result<void> Validate(const HeParameterConfig &parameters);
};

struct GeluPolynomialOutputs {
    SealCipherTensor input;
    SealCipherTensor negative_branch;
    SealCipherTensor positive_branch;
};

class FfnLayerNormTailKernel {
public:
    static Result<SealCipherTensor> Evaluate(
        const SealCipherTensor &centered,
        const SealCipherTensor &inverse_stddev,
        const DenseVectorParameter &gamma,
        const DenseVectorParameter &beta,
        const TransformerLayerConfig &layer,
        const HeContext &context,
        FfnGeluExecutionReport *report = nullptr);
};

class FfnUpProjectionKernel {
public:
    static Result<SealCipherTensor> Evaluate(
        const SealCipherTensor &input,
        const DenseMatrixParameter &weights,
        const std::optional<DenseVectorParameter> &bias,
        const TransformerLayerConfig &layer,
        const HeContext &context,
        FfnGeluExecutionReport *report = nullptr,
        std::size_t he_workers = 1);
};

// Generic rectangular column-first ciphertext/plaintext projection used by
// both Wu and Wd. The input and output packing plans each map one logical
// value to exactly one real or imaginary CKKS lane.
class ComplexColumnProjectionKernel {
public:
    static Result<SealCipherTensor> Evaluate(
        const SealCipherTensor &input,
        const DenseMatrixParameter &weights,
        const std::optional<DenseVectorParameter> &bias,
        const HePackingPlan &output_packing,
        HeRegionId output_region,
        int output_scale_bits,
        const HeContext &context,
        HeKernelOperationCounts *counts = nullptr,
        std::size_t he_workers = 1);

    static Result<std::vector<int>> RequiredRotationSteps(
        const HePackingPlan &input_packing,
        const HePackingPlan &output_packing);
};

class ComplexColumnRealifyKernel {
public:
    static Result<SealCipherTensor> Evaluate(
        const SealCipherTensor &input,
        const HePackingPlan &real_output_packing,
        int output_scale_bits,
        const HeContext &context,
        HeKernelOperationCounts *counts = nullptr);
};

class FfnRealifyKernel {
public:
    static Result<SealCipherTensor> Evaluate(
        const SealCipherTensor &input,
        const TransformerLayerConfig &layer,
        const HeContext &context,
        FfnGeluExecutionReport *report = nullptr);
};

class GeluPolynomialKernel {
public:
    static Result<GeluPolynomialOutputs> Evaluate(
        const SealCipherTensor &input,
        const GeluConfig &config,
        const HeContext &context,
        FfnGeluExecutionReport *report = nullptr,
        std::size_t he_workers = 1);
};

class MpcGeluRuntime {
public:
    MpcGeluRuntime(PartyRole role, std::string peer_address, int port);
    explicit MpcGeluRuntime(std::shared_ptr<SharedMpcSession> session);
    ~MpcGeluRuntime();
    MpcGeluRuntime(MpcGeluRuntime &&) noexcept;
    MpcGeluRuntime &operator=(MpcGeluRuntime &&) noexcept;
    MpcGeluRuntime(const MpcGeluRuntime &) = delete;
    MpcGeluRuntime &operator=(const MpcGeluRuntime &) = delete;

    Result<MpcShareTensor> Evaluate(
        const MpcShareTensor &input,
        const MpcShareTensor &negative_branch,
        const MpcShareTensor &positive_branch,
        double threshold,
        FfnGeluExecutionReport *report = nullptr);
    void ReleaseSession();
    void Cancel() noexcept;

private:
    class Impl;
    std::shared_ptr<SharedMpcSession> session_;
    std::shared_ptr<Impl> impl_;
};

class SealSciFfnGeluExecutor {
public:
    SealSciFfnGeluExecutor(
        const HeraConfig &config,
        const TransformerLayerConfig &layer,
        HeContextRegistry &contexts,
        SealPlainParameterStore &parameters,
        CryptoTensorStore &tensors,
        CkksMpcConverter &converter,
        MpcGeluRuntime &mpc,
        TensorIdAllocator *ids = nullptr);

    static Result<std::vector<int>> RequiredRotationSteps(
        const TransformerLayerConfig &layer,
        const HeParameterConfig &parameters);

    Result<FfnGeluOutputs> Evaluate(
        const FfnGeluInputs &inputs,
        const LayerWeightHandles &weights,
        FfnGeluExecutionReport *report = nullptr);

private:
    const HeraConfig &config_;
    const TransformerLayerConfig &layer_;
    HeContextRegistry &contexts_;
    SealPlainParameterStore &parameters_;
    CryptoTensorStore &tensors_;
    CkksMpcConverter &converter_;
    MpcGeluRuntime &mpc_;
    TensorIdAllocator *ids_{};
    std::atomic<TensorId> next_id_{8000000};
};

} // namespace hera
