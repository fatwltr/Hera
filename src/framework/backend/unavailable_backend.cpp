#include "hera/backend.h"

#include <sstream>

namespace hera {
namespace {

Status Unavailable(const std::string &operation,
                   const TensorSpec &output_spec)
{
    std::ostringstream message;
    message << "kernel unavailable: " << operation << " -> domain="
            << ToString(output_spec.domain) << " layout="
            << ToString(output_spec.layout) << " shape=[";
    for (std::size_t i = 0; i < output_spec.shape.dimensions.size(); ++i) {
        if (i) message << ',';
        message << output_spec.shape.dimensions[i];
    }
    message << ']';
    if (output_spec.he_region)
        message << " region=" << ToString(*output_spec.he_region);
    return Status::KernelUnavailable(message.str());
}

class UnavailableBackend final : public IHeBackend, public IMpcBackend,
                                 public IConversionBackend,
                                 public IPackingBackend {
public:
    Result<TensorHandle> MatMulCipherPlain(
        const TensorHandle &, const WeightHandle &,
        const MatMulAttributes &, const TensorSpec &out) override
    { return Unavailable("MatMulCipherPlain", out); }
    Result<TensorHandle> MatMulCipherCipher(
        const TensorHandle &, const TensorHandle &,
        const MatMulAttributes &, const TensorSpec &out) override
    { return Unavailable("MatMulCipherCipher", out); }
    Result<TensorHandle> Add(
        const TensorHandle &, const TensorHandle &,
        const TensorSpec &out) override
    { return Unavailable("Add", out); }
    Result<TensorHandle> Multiply(
        const TensorHandle &, const TensorHandle &,
        const TensorSpec &out) override
    { return Unavailable("Multiply", out); }
    Result<TensorHandle> AddPlain(
        const TensorHandle &, const PlainParameterHandle &,
        const ElementwisePlainAttributes &, const TensorSpec &out) override
    { return Unavailable("AddPlain", out); }
    Result<TensorHandle> MultiplyPlain(
        const TensorHandle &, const PlainParameterHandle &,
        const ElementwisePlainAttributes &, const TensorSpec &out) override
    { return Unavailable("MultiplyPlain", out); }
    Result<TensorHandle> ScalarAdd(
        const TensorHandle &, const ScalarAttributes &,
        const TensorSpec &out) override
    { return Unavailable("ScalarAdd", out); }
    Result<TensorHandle> ScalarMultiply(
        const TensorHandle &, const ScalarAttributes &,
        const TensorSpec &out) override
    { return Unavailable("ScalarMultiply", out); }
    Result<TensorHandle> ReduceSum(
        const TensorHandle &, const ReductionAttributes &,
        const TensorSpec &out) override
    { return Unavailable("ReduceSum", out); }
    Result<TensorHandle> Broadcast(
        const TensorHandle &, const BroadcastAttributes &,
        const TensorSpec &out) override
    { return Unavailable("Broadcast", out); }
    Result<TensorHandle> Polynomial(
        const TensorHandle &, const PolynomialAttributes &,
        const TensorSpec &out) override
    { return Unavailable("Polynomial", out); }
    Result<TensorHandle> RealifyComplex(
        const TensorHandle &, const TensorSpec &out) override
    { return Unavailable("RealifyComplex", out); }
    Result<TensorHandle> ApplyPublicMask(
        const TensorHandle &, const PublicAttentionMask &,
        const PublicMaskAttributes &,
        const TensorSpec &out) override
    { return Unavailable("ApplyPublicMask", out); }
    Result<TensorHandle> Compare(
        const std::vector<TensorHandle> &,
        const MpcNonlinearAttributes &, const TensorSpec &out) override
    { return Unavailable("Compare", out); }
    Result<TensorHandle> BooleanXor(
        const TensorHandle &, const TensorHandle &,
        const TensorSpec &out) override
    { return Unavailable("BooleanXor", out); }
    Result<TensorHandle> BooleanNot(
        const TensorHandle &, const TensorSpec &out) override
    { return Unavailable("BooleanNot", out); }
    Result<TensorHandle> Mux(
        const std::vector<TensorHandle> &,
        const MpcNonlinearAttributes &, const TensorSpec &out) override
    { return Unavailable("Mux", out); }
    Result<TensorHandle> AddArithmeticShares(
        const TensorHandle &, const TensorHandle &,
        const TensorSpec &out) override
    { return Unavailable("AddArithmeticShares", out); }
    Result<TensorHandle> ReduceMax(
        const TensorHandle &, const PublicAttentionMask &,
        const MpcNonlinearAttributes &, const TensorSpec &out) override
    { return Unavailable("MpcReduceMax", out); }
    Result<TensorHandle> PrepareMaskedScores(
        const TensorHandle &, const TensorHandle &,
        const PublicAttentionMask &, const MpcNonlinearAttributes &,
        const TensorSpec &out) override
    { return Unavailable("MpcPrepareMaskedScores", out); }
    Result<TensorHandle> ReduceSum(
        const TensorHandle &, const MpcNonlinearAttributes &,
        const TensorSpec &out) override
    { return Unavailable("MpcReduceSum", out); }
    Result<TensorHandle> MultiplyFixed(
        const TensorHandle &, const TensorHandle &,
        const MpcNonlinearAttributes &, const TensorSpec &out) override
    { return Unavailable("MpcMultiplyFixed", out); }
    Result<TensorHandle> Reciprocal(
        const TensorHandle &, const MpcNonlinearAttributes &,
        const TensorSpec &out) override
    { return Unavailable("Reciprocal", out); }
    Result<TensorHandle> ReciprocalSqrt(
        const TensorHandle &, const MpcNonlinearAttributes &,
        const TensorSpec &out) override
    { return Unavailable("ReciprocalSqrt", out); }
    Result<TensorHandle> HeToMpc(
        const TensorHandle &, const ConversionAttributes &,
        const TensorSpec &out) override
    { return Unavailable("HeToMpc", out); }
    Result<TensorHandle> MpcToHe(
        const TensorHandle &, const ConversionAttributes &,
        const TensorSpec &out) override
    { return Unavailable("MpcToHe", out); }
    Result<TensorHandle> RepackLocalShares(
        const TensorHandle &, const RepackAttributes &,
        const TensorSpec &out) override
    { return Unavailable("RepackLocalShares", out); }
};

} // namespace

Result<void> BackendSet::Validate() const
{
    if (!he || !mpc || !conversion || !packing)
        return Status::FailedPrecondition(
            "HE, MPC, conversion and packing backends must all be bound");
    return {};
}

BackendSet MakeUnavailableBackends()
{
    auto backend = std::make_shared<UnavailableBackend>();
    return {backend, backend, backend, backend};
}

} // namespace hera
