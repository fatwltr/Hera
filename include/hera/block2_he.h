#pragma once

#include "hera/backend.h"
#include "hera/he_context_registry.h"

#include <atomic>
#include <complex>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string_view>
#include <tuple>
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
    void Erase(TensorId id) noexcept;
    std::size_t Size() const noexcept;

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

// Immutable model-specific projection program. The plaintext objects are
// shared with the prepared plaintext pool, so an online kernel needs neither
// to rebuild slot masks nor to hash their full contents for every product.
struct PreparedProjectionTerm {
    std::size_t source_ct{};
    ComplexLane source_component{ComplexLane::kReal};
    int baby_rotation{};
    std::shared_ptr<const seal::Plaintext> plaintext;
};

struct PreparedProjectionGroup {
    std::size_t target_ct{};
    int giant_rotation{};
    std::vector<PreparedProjectionTerm> terms;
};

struct PreparedProjectionPlan {
    std::string profile;
    seal::parms_id_type input_parms_id{};
    double input_scale{};
    HePackingPlan input_packing;
    HePackingPlan output_packing;
    std::vector<PreparedProjectionGroup> groups;
    std::vector<std::shared_ptr<const seal::Plaintext>> bias_plaintexts;
    bool bias_after_rescale{};
    bool reuse_baby_rotations{};
    int output_scale_bits{};
    double scale_tolerance{1e-3};
};

struct PreparedProjectionCounts {
    std::size_t rotations{};
    std::size_t conjugations{};
    std::size_t ciphertext_plain_multiplications{};
    std::size_t rescales{};
};

struct PreparedProjectionRotationCache {
    std::map<std::pair<std::size_t, ComplexLane>, seal::Ciphertext> bases;
    std::map<std::tuple<std::size_t, ComplexLane, int>, seal::Ciphertext> babies;
};

Result<std::vector<seal::Ciphertext>> EvaluatePreparedProjection(
    const PreparedProjectionPlan &plan, const SealCipherTensor &input,
    const HeContext &context, PreparedProjectionCounts *counts = nullptr,
    PreparedProjectionRotationCache *rotation_cache = nullptr);
Result<std::shared_ptr<const seal::Plaintext>> CompileModelPlaintext(
    seal::CKKSEncoder &encoder,
    const std::vector<std::complex<double>> &values,
    const seal::parms_id_type &parms_id, double scale,
    std::string_view kernel_tag);

bool SamePreparedPacking(const HePackingPlan &left,
                         const HePackingPlan &right);
Result<void> ValidatePreparedProjection(
    const PreparedProjectionPlan &plan, const SealCipherTensor &input,
    const HeContext &context);

class SealPlainParameterStore {
public:
    Result<void> PutMatrix(const WeightHandle &handle,
                           std::vector<double> row_major);
    Result<void> PutVector(const PlainParameterHandle &handle,
                           std::vector<double> values);
    Result<DenseMatrixParameter> GetMatrix(WeightId id) const;
    Result<DenseVectorParameter> GetVector(PlainParameterId id) const;
    // Stable immutable views for the online path. A dense model matrix must
    // not be copied for every inference or every projection invocation.
    Result<std::shared_ptr<const DenseMatrixParameter>> GetMatrixShared(
        WeightId id) const;
    Result<std::shared_ptr<const DenseVectorParameter>> GetVectorShared(
        PlainParameterId id) const;

    // Benchmark-only mode: compatible projection diagonals are mapped to a
    // bounded pool of encoded plaintext templates. This preserves HE call
    // counts and levels but intentionally does not preserve model semantics.
    Result<void> ConfigureVirtualWeightPlaintexts(
        std::size_t templates_per_compatibility_group);
    void BeginPlaintextPreparation();
    void FreezePlaintextCache();
    void AbortPlaintextPreparation() noexcept;
    bool preparing_plaintexts() const noexcept;
    bool plaintext_cache_frozen() const noexcept;
    std::size_t encoded_plaintext_count() const;
    std::size_t encoded_plaintext_bytes() const;
    std::size_t prepared_projection_count() const;
    std::size_t prepared_projection_hits() const;
    std::size_t plaintext_cache_hits() const;
    std::size_t virtual_weight_template_count() const noexcept;
    std::size_t virtual_weight_requests() const noexcept;
    std::size_t virtual_weight_cache_hits() const noexcept;

    Result<seal::Plaintext> EncodeModelPlaintext(
        seal::CKKSEncoder &encoder, const std::vector<double> &values,
        const seal::parms_id_type &parms_id, double scale,
        std::string_view kernel_tag);

    Result<std::shared_ptr<const seal::Plaintext>> PrepareModelPlaintext(
        seal::CKKSEncoder &encoder,
        const std::vector<std::complex<double>> &values,
        const seal::parms_id_type &parms_id, double scale,
        std::string_view kernel_tag);
    Result<void> PutPreparedProjection(
        std::string key, std::shared_ptr<const PreparedProjectionPlan> plan);
    Result<std::shared_ptr<const PreparedProjectionPlan>> GetPreparedProjection(
        std::string_view key) const;
    Result<seal::Plaintext> EncodeModelPlaintext(
        seal::CKKSEncoder &encoder,
        const std::vector<std::complex<double>> &values,
        const seal::parms_id_type &parms_id, double scale,
        std::string_view kernel_tag);

private:
    mutable std::mutex mutex_;
    std::map<WeightId, std::shared_ptr<const DenseMatrixParameter>> matrices_;
    std::map<PlainParameterId, std::shared_ptr<const DenseVectorParameter>> vectors_;
    std::map<std::string, std::shared_ptr<const seal::Plaintext>> encoded_plaintexts_;
    std::map<std::string, std::shared_ptr<const seal::Plaintext>> staged_plaintexts_;
    std::map<std::string, std::shared_ptr<const PreparedProjectionPlan>>
        prepared_projections_;
    std::map<std::string, std::shared_ptr<const PreparedProjectionPlan>>
        staged_projections_;
    bool preparing_plaintexts_{};
    bool plaintext_cache_frozen_{};
    std::size_t encoded_plaintext_bytes_{};
    std::size_t plaintext_cache_hits_{};
    std::size_t virtual_weight_template_count_{};
    std::size_t virtual_weight_requests_{};
    std::size_t virtual_weight_cache_hits_{};
    std::size_t staged_encoded_plaintext_bytes_{};
    std::size_t staged_plaintext_cache_hits_{};
    std::size_t staged_virtual_weight_requests_{};
    std::size_t staged_virtual_weight_cache_hits_{};
    mutable std::size_t prepared_projection_hits_{};
};

// Runtime-only lexical binding used by the reusable kernels. Standalone
// kernel tests remain able to encode directly; the real transformer runtime
// binds a store during Prepare and Evaluate, where a frozen-cache miss is a
// hard error rather than an online encode fallback.
class ModelPlaintextCacheScope {
public:
    static std::pair<SealPlainParameterStore *, std::string> Current();
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
    // Exact automorphism offsets for the standalone QK kernel. Step zero
    // requests complex conjugation; key generation is outside benchmarks.
    static Result<std::vector<int>> RequiredRotationSteps(
        const TransformerLayerConfig &layer, std::size_t slot_count);

    // Computes Q_h*K_h^T with GQA/MQA physical head mapping. The public
    // 1/sqrt(head_width) factor must already be folded into Q. Output uses
    // attention-diagonal packing at the terminal Block-2 level.
    static Result<SealCipherTensor> Evaluate(
        const SealCipherTensor &q, const SealCipherTensor &k,
        const TransformerLayerConfig &layer,
        const HeContext &context,
        Block2ExecutionReport *report = nullptr,
        std::size_t he_workers = 1);
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
                       SealPlainParameterStore &parameters,
                       const std::vector<int> *required_rotations = nullptr);

    static Result<std::vector<int>> RequiredRotationSteps(
        const TransformerLayerConfig &layer,
        const HeParameterConfig &parameters);

    // Variant for callers that provide a non-default spatial packing. The
    // returned set contains every exact offset used by Q/K/V projection and
    // QK; compiling it makes each logical rotation one SEAL key-switch.
    static Result<std::vector<int>> RequiredRotationSteps(
        const TransformerLayerConfig &layer,
        const HeParameterConfig &parameters,
        const HePackingPlan &input_packing);

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
    const std::vector<int> *required_rotations_{};
};

} // namespace hera
