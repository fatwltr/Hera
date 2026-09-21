#pragma once

#include "hera/block2_he.h"
#include "hera/conversion_backend.h"

#include <atomic>
#include <memory>
#include <string>

namespace hera {

class SharedMpcSession;

struct SoftmaxInputs {
    TensorHandle scaled_attention_scores;
    PublicAttentionMask mask;
};

struct SoftmaxOutputs {
    // 43-bit, scale-13, unsigned [tokens,q_heads,tokens] arithmetic shares.
    TensorHandle probability_shares;
};

struct SoftmaxExecutionReport {
    std::size_t ckks_to_mpc_calls{};
    std::size_t mpc_to_ckks_calls{};
    std::size_t comparisons{};
    std::size_t muxes{};
    std::size_t row_sums{};
    std::size_t reciprocals{};
    std::size_t secure_multiplications{};
    std::size_t ciphertext_plain_multiplications{};
    std::size_t ciphertext_multiplications{};
    std::size_t relinearizations{};
    std::size_t rescales{};
    std::size_t rotations{};
    std::uint64_t mpc_communication_bytes{};
    std::uint64_t mpc_rounds{};
    std::uint64_t conversion_communication_bytes{};
    std::uint64_t conversion_rounds{};
    std::uint64_t a_mask_bytes{};
    std::uint64_t z_mask_bytes{};
    std::size_t internal_mod_switches{};
    double ckks_to_mpc_milliseconds{};
    double mpc_to_ckks_milliseconds{};
    double mpc_milliseconds{};
    double he_milliseconds{};
};

class Block3LevelPlanner {
public:
    // /2^d consumes one level and every repeated square consumes one more.
    // The configured chain must end at one data prime after these operations.
    static Result<void> Validate(const HeParameterConfig &parameters,
                                 const SoftmaxConfig &softmax);
};

class SoftmaxPackingPlanner {
public:
    // Maps logical [row,head,column] values to real CKKS slots. Rows use a
    // power-of-two physical stride when requested; padding slots and every
    // imaginary lane remain zero. The mapping may span multiple ciphertexts.
    static Result<PhysicalSlotMap> RealRows(
        const TransformerLayerConfig &layer,
        std::size_t slots_per_ciphertext,
        bool pad_rows_to_power_of_two);
};

class SoftmaxExponentKernel {
public:
    // Computes (1 + x / 2^exp_divisor_bits)^(2^exp_squarings) on a
    // real-only CKKS batch. It starts at fresh level zero, reaches the
    // terminal single-prime level, and never decrypts or rotates.
    static Result<SealCipherBatch> Evaluate(
        const SealCipherBatch &shifted_scores,
        const HeContext &context,
        const SoftmaxConfig &config,
        SoftmaxExecutionReport *report = nullptr);
};

class MpcSoftmaxRuntime {
public:
    MpcSoftmaxRuntime(PartyRole role, std::string peer_address, int port);
    explicit MpcSoftmaxRuntime(std::shared_ptr<SharedMpcSession> session);
    ~MpcSoftmaxRuntime();
    MpcSoftmaxRuntime(MpcSoftmaxRuntime &&) noexcept;
    MpcSoftmaxRuntime &operator=(MpcSoftmaxRuntime &&) noexcept;
    MpcSoftmaxRuntime(const MpcSoftmaxRuntime &) = delete;
    MpcSoftmaxRuntime &operator=(const MpcSoftmaxRuntime &) = delete;

    // Releases only this runtime's algorithm wrapper.  The underlying shared
    // session remains alive so later sequential kernels reuse VOLE state.
    void ReleaseSession();
    void Cancel() noexcept;

    // Secure cmp/mux max, followed by local share subtraction. Publicly
    // masked entries are assigned config.masked_shift without communication.
    Result<MpcShareTensor> CenterAndMask(
        const MpcShareTensor &scores,
        const TransformerLayerConfig &layer,
        const PublicAttentionMask &mask,
        const SoftmaxConfig &config,
        SoftmaxExecutionReport *report = nullptr);

    // Secure threshold/mux, local row sum, exact fixed-point reciprocal, and
    // secure fixed multiplication/truncation. No result is reconstructed.
    Result<MpcShareTensor> ClipAndNormalize(
        const MpcShareTensor &shifted_scores,
        const MpcShareTensor &exponentials,
        const TransformerLayerConfig &layer,
        const PublicAttentionMask &mask,
        const SoftmaxConfig &config,
        SoftmaxExecutionReport *report = nullptr);

private:
    class Impl;
    std::shared_ptr<SharedMpcSession> session_;
    std::shared_ptr<Impl> impl_;
};

// Concrete staged Softmax. Block-2 scores enter as CKKS; probabilities leave
// as arithmetic shares. This class intentionally has no retained-V argument
// and performs no final MPC-to-CKKS conversion.
class SealSciSoftmaxExecutor {
public:
    SealSciSoftmaxExecutor(
        const HeraConfig &config,
        const TransformerLayerConfig &layer,
        HeContextRegistry &contexts,
        SealCipherTensorStore &he_tensors,
        CryptoTensorStore &mpc_tensors,
        CkksMpcConverter &converter,
        MpcSoftmaxRuntime &mpc,
        TensorIdAllocator *ids = nullptr);

    Result<SoftmaxOutputs> Evaluate(
        const SoftmaxInputs &inputs,
        SoftmaxExecutionReport *report = nullptr);

private:
    const HeraConfig &config_;
    const TransformerLayerConfig &layer_;
    HeContextRegistry &contexts_;
    SealCipherTensorStore &he_tensors_;
    CryptoTensorStore &mpc_tensors_;
    CkksMpcConverter &converter_;
    MpcSoftmaxRuntime &mpc_;
    TensorIdAllocator *ids_{};
    std::atomic<TensorId> next_id_{6000000};
};

} // namespace hera
