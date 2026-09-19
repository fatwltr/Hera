#pragma once

#include "hera/backend.h"
#include "hera/he_context_registry.h"

#include <atomic>
#include <complex>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <string_view>
#include <vector>

namespace hera {

enum class ComplexLane { kReal, kImaginary };

struct PhysicalSlotLocation {
    std::size_t ciphertext{};
    std::size_t slot{};
    ComplexLane component{ComplexLane::kReal};
    // For physically replicated GQA/MQA K/V values this identifies the query
    // head whose lane owns the copy. It is unset for non-replicated layouts.
    std::optional<std::size_t> packed_query_head;
};

// Complete logical-to-physical map for an HE tensor. Unlike the conversion
// PhysicalSlotMap, one logical value may intentionally have several physical
// locations (the GQA/MQA K/V replication case).
struct HePackingPlan {
    Layout layout{Layout::kSpatial};
    TensorShape logical_shape;
    std::size_t slot_count{};
    std::size_t ciphertext_count{};
    std::vector<std::vector<PhysicalSlotLocation>> logical_locations;

    Result<void> Validate() const;
    const std::vector<PhysicalSlotLocation> &Locations(
        std::size_t logical_index) const;
};

struct MultiHeadPackingPlan {
    static Result<HePackingPlan> DenseSpatial(
        const TensorShape &shape, std::size_t slot_count);
    static Result<HePackingPlan> ProjectionOutput(
        const TransformerLayerConfig &layer, Layout layout,
        std::size_t slot_count);
    // Spatial Transformer input replicated across query-head lanes. This is
    // the producer contract that makes W*E projection rotations independent
    // of the token index.
    static Result<HePackingPlan> ProjectionInput(
        const TransformerLayerConfig &layer, std::size_t slot_count);
    static Result<HePackingPlan> AttentionDiagonal(
        const TransformerLayerConfig &layer, std::size_t slot_count);
};

struct ComplexDiagonalProjectionPlan {
    std::size_t input_columns{};
    std::size_t output_columns{};
    Layout output_layout{Layout::kMultiHeadQ};
    bool physically_replicates_kv{};
    HePackingPlan output_packing;
    std::vector<int> required_rotation_steps;
};

// Compiles the logical elementary column transform E without materializing a
// dense E matrix. The concrete plaintext encoder uses output_packing as the
// W*E scatter destination.
class ComplexDiagonalProjectionPlanner {
public:
    static Result<ComplexDiagonalProjectionPlan> Build(
        const TransformerLayerConfig &layer, Layout output_layout,
        std::size_t slot_count);
};

struct SealCipherTensor {
    std::string parameter_profile;
    std::vector<seal::Ciphertext> ciphertexts;
    TensorSpec logical_spec;
    HePackingPlan physical_packing;

    Result<void> Validate(const HeContext &context) const;
};

class SealCipherTensorStore {
public:
    Result<void> Put(TensorId id, SealCipherTensor tensor);
    Result<SealCipherTensor> Get(TensorId id) const;
    Result<TensorHandle> Allocate(SealCipherTensor tensor);

private:
    mutable std::mutex mutex_;
    std::map<TensorId, SealCipherTensor> tensors_;
    std::atomic<TensorId> next_id_{4000000};
};

struct DenseMatrixParameter {
    MatrixSpec spec;
    std::vector<double> row_major;
};

struct DenseVectorParameter {
    VectorSpec spec;
    std::vector<double> values;
};

class SealPlainParameterStore {
public:
    Result<void> PutMatrix(const WeightHandle &handle,
                           std::vector<double> row_major);
    Result<void> PutVector(const PlainParameterHandle &handle,
                           std::vector<double> values);
    Result<DenseMatrixParameter> GetMatrix(WeightId id) const;
    Result<DenseVectorParameter> GetVector(PlainParameterId id) const;

    void BeginPlaintextPreparation();
    void FreezePlaintextCache();
    bool plaintext_cache_frozen() const noexcept;
    std::size_t encoded_plaintext_count() const;
    std::size_t encoded_plaintext_bytes() const;
    std::size_t plaintext_cache_hits() const;

    Result<seal::Plaintext> EncodeModelPlaintext(
        seal::CKKSEncoder &encoder, const std::vector<double> &values,
        const seal::parms_id_type &parms_id, double scale,
        std::string_view kernel_tag);
    Result<seal::Plaintext> EncodeModelPlaintext(
        seal::CKKSEncoder &encoder,
        const std::vector<std::complex<double>> &values,
        const seal::parms_id_type &parms_id, double scale,
        std::string_view kernel_tag);

private:
    mutable std::mutex mutex_;
    std::map<WeightId, DenseMatrixParameter> matrices_;
    std::map<PlainParameterId, DenseVectorParameter> vectors_;
    std::map<std::string, seal::Plaintext> encoded_plaintexts_;
    bool preparing_plaintexts_{};
    bool plaintext_cache_frozen_{};
    std::size_t encoded_plaintext_bytes_{};
    std::size_t plaintext_cache_hits_{};
};

// Runtime-only lexical binding used by the reusable kernels. Standalone
// kernel tests remain able to encode directly; the real transformer runtime
// binds a store during Prepare and Evaluate, where a frozen-cache miss is a
// hard error rather than an online encode fallback.
class ModelPlaintextCacheScope {
public:
    explicit ModelPlaintextCacheScope(
        SealPlainParameterStore *store, std::string key_namespace = {});
    ~ModelPlaintextCacheScope();
    ModelPlaintextCacheScope(const ModelPlaintextCacheScope &) = delete;
    ModelPlaintextCacheScope &operator=(const ModelPlaintextCacheScope &) = delete;
private:
    SealPlainParameterStore *previous_{};
    std::string previous_namespace_;
};

Result<seal::Plaintext> EncodeModelPlaintext(
    seal::CKKSEncoder &encoder, const std::vector<double> &values,
    const seal::parms_id_type &parms_id, double scale,
    std::string_view kernel_tag);
Result<seal::Plaintext> EncodeModelPlaintext(
    seal::CKKSEncoder &encoder,
    const std::vector<std::complex<double>> &values,
    const seal::parms_id_type &parms_id, double scale,
    std::string_view kernel_tag);

struct Block2HeInputs {
    // Both physical tensors use the same [tokens, model_width] dense spatial
    // packing. inverse_stddev is already broadcast across model_width.
    TensorHandle centered;
    TensorHandle inverse_stddev;
};

struct Block2HeOutputs {
    TensorHandle scaled_attention_scores;
    TensorHandle value;
};

struct Block2ExecutionReport {
    std::size_t rotations{};
    std::size_t conjugations{};
    std::size_t ciphertext_plain_multiplications{};
    std::size_t ciphertext_multiplications{};
    std::size_t relinearizations{};
    std::size_t rescales{};
    std::vector<int> required_rotation_steps;
};

class Block2LevelPlanner {
public:
    static Result<void> Validate(const HeParameterConfig &parameters);
};

class LayerNormTailKernel {
public:
    // Computes ((centered * broadcast(inverse_stddev)) * gamma) + beta.
    // Real and imaginary SIMD lanes are independent logical values. The
    // implementation consumes exactly one data level and never decrypts.
    static Result<SealCipherTensor> Evaluate(
        const SealCipherTensor &centered,
        const SealCipherTensor &inverse_stddev,
        const DenseVectorParameter &gamma,
        const DenseVectorParameter &beta,
        const TransformerLayerConfig &layer,
        const HeContext &context,
        Block2ExecutionReport *report = nullptr);
};

class ComplexDiagonalProjectionKernel {
public:
    ComplexDiagonalProjectionKernel(
        const TransformerLayerConfig &layer, const HeContext &context,
        Block2ExecutionReport *report = nullptr);
    ~ComplexDiagonalProjectionKernel();
    ComplexDiagonalProjectionKernel(ComplexDiagonalProjectionKernel &&) noexcept;
    ComplexDiagonalProjectionKernel &operator=(
        ComplexDiagonalProjectionKernel &&) noexcept;
    ComplexDiagonalProjectionKernel(
        const ComplexDiagonalProjectionKernel &) = delete;
    ComplexDiagonalProjectionKernel &operator=(
        const ComplexDiagonalProjectionKernel &) = delete;

    // Computes input[t,:] * W into Q/K/V-native complex multihead packing.
    // W is ordinary row-major [model_width, output_width]. Rotated input
    // ciphertexts are cached by this object and reused across Q, K, and V.
    Result<SealCipherTensor> Evaluate(
        const SealCipherTensor &input,
        const DenseMatrixParameter &weights,
        const std::optional<DenseVectorParameter> &bias,
        Layout output_layout);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

class MultiHeadQkKernel {
public:
    // Computes Q_h*K_h^T with GQA/MQA physical head mapping. The public
    // 1/sqrt(head_width) factor must already be folded into Q. Output uses
    // attention-diagonal packing at the terminal Block-2 level.
    static Result<SealCipherTensor> Evaluate(
        const SealCipherTensor &q, const SealCipherTensor &k,
        const TransformerLayerConfig &layer,
        const HeContext &context,
        Block2ExecutionReport *report = nullptr);
};

// True SEAL CKKS implementation of Figure-10 Block 2. It owns no secret key,
// creates no context, and never decrypts. All ciphertexts and evaluation keys
// come from the shared HeContextRegistry.
class SealBlock2Executor {
public:
    SealBlock2Executor(const HeraConfig &config,
                       const TransformerLayerConfig &layer,
                       HeContextRegistry &contexts,
                       SealCipherTensorStore &tensors,
                       SealPlainParameterStore &parameters);

    static Result<std::vector<int>> RequiredRotationSteps(
        const TransformerLayerConfig &layer,
        const HeParameterConfig &parameters);

    Result<Block2HeOutputs> Evaluate(
        const Block2HeInputs &inputs,
        const LayerWeightHandles &weights,
        Block2ExecutionReport *report = nullptr);

private:
    const HeraConfig &config_;
    const TransformerLayerConfig &layer_;
    HeContextRegistry &contexts_;
    SealCipherTensorStore &tensors_;
    SealPlainParameterStore &parameters_;
};

} // namespace hera
