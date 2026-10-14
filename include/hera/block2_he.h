#pragma once

#include "hera/backend.h"
#include "hera/he_context_registry.h"
#include "hera/detail/projection_plaintext.h"

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

enum class ProjectionPackingTarget { kQk, kAttentionValue };

struct ProjectionPackingDecision {
    ProjectionPackingTarget target{ProjectionPackingTarget::kQk};
    // First (largest) group, retained for simple diagnostics.
    std::size_t heads_per_tile{};
    // Ordered physical groups; prepared W/bias encodings record the full list.
    std::vector<std::size_t> head_groups;
};

struct MultiHeadPackingPlan {
    // Generic token-contiguous packing: one real value per CKKS slot.
    static Result<HePackingPlan> DenseSpatialReal(
        const TensorShape &shape, std::size_t slot_count);
    // LN1/Block-2 input/output: consecutive tokens within each feature,
    // padded to a power-of-two token stride; imaginary slots are unused.
    static Result<HePackingPlan> RealStrictColumns(
        const TensorShape &shape, std::size_t slot_count);
    // One real ciphertext repeats each token inverse across feature columns.
    static Result<HePackingPlan> TokenInverseBroadcast(
        std::size_t tokens, std::size_t slot_count);
    // Choose W*E destinations according to the downstream consumer. Q/K
    // use groups individually dividing the slot columns; V permits a physical tail.
    static Result<ProjectionPackingDecision> ProjectionDecision(
        const TransformerLayerConfig &layer, Layout layout,
        std::size_t slot_count);
    // Each head owns a whole token column: lane*(stride*heads)+head*stride+token.
    // V uses the largest power-of-two lane count fitting each head tile,
    // matching the A*V consumer; Q/K greedily consume full-column head groups.
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
    ProjectionPackingDecision packing_decision;
};

struct QkvColumnBsgsPlan {
    std::size_t token_stride{};
    std::size_t columns_per_ciphertext{};
    std::size_t baby_steps{};
    std::size_t giant_steps{};
    std::size_t input_ciphertexts{};
    std::size_t output_ciphertexts{}; // Sum of physical Q/K/V ciphertext counts.
    std::size_t rotation_upper_bound{};
};

// Compiles the logical elementary column transform E without materializing a
// dense E matrix. The concrete plaintext encoder uses output_packing as the
// W*E scatter destination.
class ComplexDiagonalProjectionPlanner {
public:
    // Minimize shared QKV babies plus giants over factor pairs Sb*Sg=C.
    static Result<QkvColumnBsgsPlan> ColumnBsgs(
        const TransformerLayerConfig &layer, std::size_t slot_count);
    static Result<ComplexDiagonalProjectionPlan> Build(
        const TransformerLayerConfig &layer, Layout output_layout,
        std::size_t slot_count);
};

struct SealCipherTensor {
    std::string parameter_profile;
    std::vector<seal::Ciphertext> ciphertexts;
    TensorSpec logical_spec;
    HePackingPlan physical_packing;
    // K projection emits conj(K) by negating imaginary weight/bias lanes.
    // This is a value convention, independent of the physical slot map.
    bool conjugated_complex{};

    Result<void> Validate(const HeContext &context) const;
};

class SealCipherTensorStore {
public:
    Result<void> Put(TensorId id, SealCipherTensor tensor);
    Result<SealCipherTensor> Get(TensorId id) const;
    Result<std::shared_ptr<const SealCipherTensor>> GetShared(TensorId id) const;
    Result<void> Rename(TensorId source, TensorId destination);
    Result<TensorHandle> Allocate(SealCipherTensor tensor);
    void Erase(TensorId id) noexcept;
    std::size_t Size() const noexcept;

private:
    mutable std::mutex mutex_;
    std::map<TensorId, std::shared_ptr<const SealCipherTensor>> tensors_;
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

// Tail parameters after moving gamma into Q/K/V: gamma is 1 for fused
// features, 0 for zero-gamma features; beta stays in the tail as beta/gamma.
struct GammaFusedLayerNormParameters {
    DenseVectorParameter gamma;
    DenseVectorParameter beta;
};

// Immutable model-specific projection program. The plaintext objects are
// shared with the prepared plaintext pool, so an online kernel needs neither
// to rebuild slot masks nor to hash their full contents for every product.
struct PreparedProjectionTerm {
    std::size_t source_ct{};
    ComplexLane source_component{ComplexLane::kReal};
    int baby_rotation{};
    ProjectionPlaintext plaintext;
};

struct PreparedProjectionGroup {
    std::size_t target_ct{};
    int giant_rotation{};
    std::vector<PreparedProjectionTerm> terms;
    // Real weight channel multiplied by the plan's fixed phase before the
    // giant rotation; shares the same input baby rotations as terms.
    std::vector<PreparedProjectionTerm> phase_terms{};
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
    // Real-only sources need no z +/- conj(z) component extraction.
    bool real_input{};
    // Use the whole complex source, with component coefficients in plaintexts.
    bool raw_complex_input{};
    // Model-independent X^(N/2), scale 1, with slots i*(-1)^slot.
    // Wo weight plaintexts in both channels remain real-only.
    std::shared_ptr<const seal::Plaintext> fixed_phase;
    // Plaintexts include 1/2; recover real output after rescale, before bias.
    bool realify_output{};
    bool conjugate_output{};
    // Present for QKV: identifies which W*E preprocessing was compiled.
    std::optional<ProjectionPackingDecision> packing_decision;
    int output_scale_bits{};
    double scale_tolerance{1e-3};
};

struct PreparedProjectionCounts {
    std::size_t rotations{};
    std::size_t conjugations{};
    std::size_t ciphertext_plain_multiplications{};
    std::size_t rescales{};
    std::size_t baby_rotations{};
    std::size_t giant_rotations{};
    std::size_t output_conjugations{};
    std::size_t phase_multiplications{};
};

void ConfigureProjectionRotationHoisting(bool enabled) noexcept;
bool ProjectionRotationHoistingEnabled() noexcept;
void ConfigureProjectionBabySteps(std::size_t steps) noexcept;
double ProjectionBsgsCost(std::size_t bases, std::size_t outputs, std::size_t columns,
    std::size_t baby, std::size_t slots, bool phase = false, bool extracted_bases = false);

struct PreparedProjectionRotationCache {
    std::map<std::pair<std::size_t, ComplexLane>, seal::Ciphertext> bases;
    std::map<std::tuple<std::size_t, ComplexLane, int>, seal::Ciphertext> babies;
#ifdef HERA_SEAL_ROTATION_HOISTING
    std::map<std::pair<std::size_t, ComplexLane>,
        std::shared_ptr<const seal::Evaluator::HoistedRotation>> hoisted;
#endif
};

// Opaque immutable LN/QK programs share the model preparation transaction.
struct PreparedBlock2Program {
    virtual ~PreparedBlock2Program() = default;
};

Result<std::vector<seal::Ciphertext>> EvaluatePreparedProjection(
    const PreparedProjectionPlan &plan, const SealCipherTensor &input,
    const HeContext &context, PreparedProjectionCounts *counts = nullptr,
    PreparedProjectionRotationCache *rotation_cache = nullptr,
    std::size_t he_workers = 1);
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

    // Offline: replace W by diag(gamma)*W and retain beta/gamma in LN1.
    // Zero-gamma rows keep original W, with residual gamma=0 and original beta.
    Result<void> FuseLayerNormGammaIntoQkv(const LayerWeightHandles &weights);
    Result<void> FuseLayerNormGammaIntoWu(const LayerWeightHandles &weights);
    Result<std::shared_ptr<const GammaFusedLayerNormParameters>>
    GetGammaFusedLayerNorm(WeightId id) const;

    // Benchmark-only mode: compatible projection diagonals are mapped to a
    // bounded pool of encoded plaintext templates. This preserves HE call
    // counts and levels but intentionally does not preserve model semantics.
    Result<void> ConfigureVirtualWeightPlaintexts(
        std::size_t templates_per_compatibility_group);
    Result<void> ConfigureProjectionPlaintextMemory(std::size_t bytes);
    std::size_t projection_resident_bytes() const;
    std::size_t projection_spilled_bytes() const;
    std::size_t projection_plaintext_loads() const;
    Result<ProjectionPlaintext> PrepareProjectionPlaintext(
        seal::CKKSEncoder &encoder, const std::vector<std::complex<double>> &values,
        const seal::parms_id_type &parms_id, double scale, std::string_view kernel_tag,
        std::shared_ptr<seal::SEALContext> context);
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
    Result<void> PutPreparedBlock2Program(
        std::string key, std::shared_ptr<const PreparedBlock2Program> program);
    Result<std::shared_ptr<const PreparedBlock2Program>> GetPreparedBlock2Program(
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
    struct LayerNormFusion {
        PlainParameterId gamma{};
        PlainParameterId beta{};
        std::shared_ptr<const GammaFusedLayerNormParameters> tail;
    };
    std::map<WeightId, LayerNormFusion> fused_qkv_;
    std::map<std::string, std::shared_ptr<const seal::Plaintext>> encoded_plaintexts_;
    std::map<std::string, std::shared_ptr<const seal::Plaintext>> staged_plaintexts_;
    std::map<std::string, std::shared_ptr<const PreparedProjectionPlan>>
        prepared_projections_;
    std::map<std::string, std::shared_ptr<const PreparedProjectionPlan>>
        staged_projections_;
    std::map<std::string, std::shared_ptr<const PreparedBlock2Program>>
        prepared_block2_programs_, staged_block2_programs_;
    std::shared_ptr<ProjectionPlaintextCache> projection_plaintexts_, staged_projection_plaintexts_;
    std::size_t projection_memory_budget_{256ULL << 20};
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

Result<ProjectionPlaintext> CompileProjectionPlaintext(
    seal::CKKSEncoder &encoder, const std::vector<std::complex<double>> &values,
    const seal::parms_id_type &parms_id, double scale, std::string_view kernel_tag,
    const HeContext &context);

struct Block2HeInputs {
    // centered uses real strict-column [tokens, model_width]; inverse_stddev
    // is a single real reduction [tokens,1] repeated in every feature column.
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
    // Computes residual_gamma * centered * inverse_stddev + adjusted_beta.
    // Gamma fusion is prepared offline; beta remains in the LayerNorm tail.
    // Requires real strict-column packing and preserves it in the output.
    // No conjugation or real/imaginary extraction; consumes one data level.
    // Optional original_affine/residual output reuses the same ct-ct product
    // and relinearization, with a parallel affine/rescale at that same level.
    static Result<SealCipherTensor> Evaluate(
        const SealCipherTensor &centered,
        const SealCipherTensor &inverse_stddev,
        const GammaFusedLayerNormParameters &affine,
        const TransformerLayerConfig &layer,
        const HeContext &context,
        Block2ExecutionReport *report = nullptr,
        const GammaFusedLayerNormParameters *original_affine = nullptr,
        SealCipherTensor *normalized_residual = nullptr,
        std::size_t he_workers = 1);
};

class ComplexDiagonalProjectionKernel {
public:
    // Keys for this projection alone, without LayerNorm or QK rotations.
    static Result<std::vector<int>> RequiredRotationSteps(
        const TransformerLayerConfig &layer,
        const HePackingPlan &input_packing, Layout output_layout);

    ComplexDiagonalProjectionKernel(
        const TransformerLayerConfig &layer, const HeContext &context,
        Block2ExecutionReport *report = nullptr, std::size_t he_workers = 1);
    ~ComplexDiagonalProjectionKernel();
    ComplexDiagonalProjectionKernel(ComplexDiagonalProjectionKernel &&) noexcept;
    ComplexDiagonalProjectionKernel &operator=(
        ComplexDiagonalProjectionKernel &&) noexcept;
    ComplexDiagonalProjectionKernel(
        const ComplexDiagonalProjectionKernel &) = delete;
    ComplexDiagonalProjectionKernel &operator=(
        const ComplexDiagonalProjectionKernel &) = delete;

    // Computes input[t,:] * W into Q/K/V-native complex multihead packing.
    // Input is real strict-column; W is ordinary row-major. The compiler
    // packs paired output columns as complex coefficients, folds 1/sqrt(2)
    // into Q/K weights and biases, folds 1/sqrt(head_width) into Q, and
    // negates K imaginary coefficients to emit conj(K) without a Galois call.
    // Baby rotations are cached by this object and shared across Q/K/V
    // for the same immutable source batch and level. Changing the batch,
    // parms_id or scale invalidates the cache. Giant groups are independent.
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
    static constexpr std::size_t kDepth = 4;
    // Compile fixed masks, routes and packing without HE arithmetic.
    // Requires an active model plaintext preparation scope.
    static Result<void> Prepare(const TransformerLayerConfig &layer,
                                const HeContext &context);
    struct TileBsgsPlan {
        std::size_t head_begin{}, heads_per_tile{}, lanes{}, input_groups{};
        std::size_t preprocessing_baby_steps{}, preprocessing_giant_steps{};
        std::size_t baby_steps{}, giant_steps{};
        std::size_t rotations{}, conjugations{}, relinearizations{};
        std::size_t ciphertext_multiplications{};
    };
    struct BsgsPlan {
        std::vector<TileBsgsPlan> head_tiles;
        std::size_t rotations{}, conjugations{}, relinearizations{};
        std::size_t ciphertext_multiplications{};
    };
    static Result<BsgsPlan> Plan(
        const TransformerLayerConfig &layer, std::size_t slot_count);
    // Exact automorphism offsets for the standalone QK kernel. Step zero
    // requests complex conjugation; key generation is outside benchmarks.
    static Result<std::vector<int>> RequiredRotationSteps(
        const TransformerLayerConfig &layer, std::size_t slot_count);

    // Computes Q_h*K_h^T with GQA/MQA physical head mapping. The public
    // 1/sqrt(head_width) factor must already be folded into Q, and both Q
    // and K must carry the projection compiler's 1/sqrt(2) factor. K must
    // have conjugated_complex=true, as emitted by the Wk compiler. Real
    // extraction then doubles the product without a runtime 1/2. Output uses
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
        Block2ExecutionReport *report = nullptr,
        SealCipherTensor *normalized_residual = nullptr);

private:
    const HeraConfig &config_;
    const TransformerLayerConfig &layer_;
    HeContextRegistry &contexts_;
    SealCipherTensorStore &tensors_;
    SealPlainParameterStore &parameters_;
    const std::vector<int> *required_rotations_{};
};

} // namespace hera
