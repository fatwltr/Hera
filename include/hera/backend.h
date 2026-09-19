#pragma once

#include "hera/graph.h"

#include <memory>

namespace hera {

// Every backend call receives the graph-declared output contract. A concrete
// implementation must preserve it, report its actual level/scale, and must
// never decrypt an intermediate value.
class IHeBackend {
public:
    virtual ~IHeBackend() = default;

    // Ciphertext/plaintext matrix multiplication. The implementation must
    // support arbitrary compatible matrix dimensions, consume the declared
    // input packing, produce output_spec.layout, and account for rotations,
    // relinearization and rescale operations instead of hiding them.
    virtual Result<TensorHandle> MatMulCipherPlain(
        const TensorHandle &input, const WeightHandle &weight,
        const MatMulAttributes &attributes,
        const TensorSpec &output_spec) = 0;

    // Ciphertext/ciphertext matrix multiplication used by Q*K^T and A*V. Both
    // inputs must belong to a compatible HE context and level/scale plan.
    virtual Result<TensorHandle> MatMulCipherCipher(
        const TensorHandle &left, const TensorHandle &right,
        const MatMulAttributes &attributes,
        const TensorSpec &output_spec) = 0;

    virtual Result<TensorHandle> Add(
        const TensorHandle &left, const TensorHandle &right,
        const TensorSpec &output_spec) = 0;
    virtual Result<TensorHandle> Multiply(
        const TensorHandle &left, const TensorHandle &right,
        const TensorSpec &output_spec) = 0;
    virtual Result<TensorHandle> AddPlain(
        const TensorHandle &input, const PlainParameterHandle &plain,
        const ElementwisePlainAttributes &attributes,
        const TensorSpec &output_spec) = 0;
    virtual Result<TensorHandle> MultiplyPlain(
        const TensorHandle &input, const PlainParameterHandle &plain,
        const ElementwisePlainAttributes &attributes,
        const TensorSpec &output_spec) = 0;
    virtual Result<TensorHandle> ScalarAdd(
        const TensorHandle &input, const ScalarAttributes &attributes,
        const TensorSpec &output_spec) = 0;
    virtual Result<TensorHandle> ScalarMultiply(
        const TensorHandle &input, const ScalarAttributes &attributes,
        const TensorSpec &output_spec) = 0;
    virtual Result<TensorHandle> ReduceSum(
        const TensorHandle &input, const ReductionAttributes &attributes,
        const TensorSpec &output_spec) = 0;
    virtual Result<TensorHandle> Broadcast(
        const TensorHandle &input, const BroadcastAttributes &attributes,
        const TensorSpec &output_spec) = 0;
    virtual Result<TensorHandle> Polynomial(
        const TensorHandle &input, const PolynomialAttributes &attributes,
        const TensorSpec &output_spec) = 0;

    // Splits a complex SIMD payload into real-only ciphertexts without
    // decrypting it. The concrete backend must implement z+conj(z) and
    // (z-conj(z))*(-i), preserve the parms_id, and report the exact scale
    // change caused by the missing public division by two.
    virtual Result<TensorHandle> RealifyComplex(
        const TensorHandle &input, const TensorSpec &output_spec) = 0;

    // Adds a public padding/causal mask under encryption. Secret masks are not
    // part of this framework version and must not be smuggled through here.
    virtual Result<TensorHandle> ApplyPublicMask(
        const TensorHandle &input, const PublicAttentionMask &mask,
        const PublicMaskAttributes &attributes,
        const TensorSpec &output_spec) = 0;
};

class IMpcBackend {
public:
    virtual ~IMpcBackend() = default;

    // Produces secret boolean shares without revealing comparison results.
    virtual Result<TensorHandle> Compare(
        const std::vector<TensorHandle> &inputs,
        const MpcNonlinearAttributes &attributes,
        const TensorSpec &output_spec) = 0;
    virtual Result<TensorHandle> BooleanXor(
        const TensorHandle &left, const TensorHandle &right,
        const TensorSpec &output_spec) = 0;
    virtual Result<TensorHandle> BooleanNot(
        const TensorHandle &input, const TensorSpec &output_spec) = 0;

    // Secret selection. Public fallback constants are described by attributes
    // so the graph does not need a plaintext intermediate tensor.
    virtual Result<TensorHandle> Mux(
        const std::vector<TensorHandle> &inputs,
        const MpcNonlinearAttributes &attributes,
        const TensorSpec &output_spec) = 0;

    virtual Result<TensorHandle> AddArithmeticShares(
        const TensorHandle &left, const TensorHandle &right,
        const TensorSpec &output_spec) = 0;

    // Computes one maximum for every attention row using a comparison/mux
    // tree. The public mask removes invalid columns; an all-masked row is an
    // error and comparison results remain secret.
    virtual Result<TensorHandle> ReduceMax(
        const TensorHandle &input, const PublicAttentionMask &mask,
        const MpcNonlinearAttributes &attributes,
        const TensorSpec &output_spec) = 0;

    // Subtracts the secret row maximum and replaces publicly masked entries
    // by attributes.public_value. Output remains signed arithmetic shares.
    virtual Result<TensorHandle> PrepareMaskedScores(
        const TensorHandle &input, const TensorHandle &row_max,
        const PublicAttentionMask &mask,
        const MpcNonlinearAttributes &attributes,
        const TensorSpec &output_spec) = 0;

    // Local addition on each party's arithmetic shares. The backend must not
    // reconstruct a row and must retain enough integer bits for tokens terms.
    virtual Result<TensorHandle> ReduceSum(
        const TensorHandle &input,
        const MpcNonlinearAttributes &attributes,
        const TensorSpec &output_spec) = 0;

    // Secure fixed-point multiplication followed by an MPC truncation to the
    // output format; used for Softmax exp * reciprocal(sum).
    virtual Result<TensorHandle> MultiplyFixed(
        const TensorHandle &left, const TensorHandle &right,
        const MpcNonlinearAttributes &attributes,
        const TensorSpec &output_spec) = 0;

    virtual Result<TensorHandle> Reciprocal(
        const TensorHandle &input,
        const MpcNonlinearAttributes &attributes,
        const TensorSpec &output_spec) = 0;
    virtual Result<TensorHandle> ReciprocalSqrt(
        const TensorHandle &input,
        const MpcNonlinearAttributes &attributes,
        const TensorSpec &output_spec) = 0;
};

class IConversionBackend {
public:
    virtual ~IConversionBackend() = default;

    // Converts HE slots to arithmetic shares without reconstruction. Input
    // must be at a conversion-compatible terminal level and scale.
    virtual Result<TensorHandle> HeToMpc(
        const TensorHandle &ciphertext,
        const ConversionAttributes &attributes,
        const TensorSpec &output_spec) = 0;

    // Encrypts arithmetic shares into the target HE region and fresh modulus
    // chain without either party learning the combined clear value.
    virtual Result<TensorHandle> MpcToHe(
        const TensorHandle &share,
        const ConversionAttributes &attributes,
        const TensorSpec &output_spec) = 0;
};

class IPackingBackend {
public:
    virtual ~IPackingBackend() = default;

    // Reorders each party's local shares only. It must not communicate and
    // must preserve the logical tensor values and fixed-point format.
    virtual Result<TensorHandle> RepackLocalShares(
        const TensorHandle &input, const RepackAttributes &attributes,
        const TensorSpec &output_spec) = 0;
};

struct BackendSet {
    std::shared_ptr<IHeBackend> he;
    std::shared_ptr<IMpcBackend> mpc;
    std::shared_ptr<IConversionBackend> conversion;
    std::shared_ptr<IPackingBackend> packing;

    Result<void> Validate() const;
};

struct ExecutionRecord {
    NodeId node{};
    std::string node_name;
    OpKind operation{OpKind::kElementwiseAdd};
    std::vector<TensorId> inputs;
    TensorId output{};
};

struct RecordingState {
    std::vector<ExecutionRecord> records;
};

BackendSet MakeUnavailableBackends();
BackendSet MakeRecordingBackends(std::shared_ptr<RecordingState> state);

} // namespace hera
