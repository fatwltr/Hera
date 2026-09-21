#include "hera/backend.h"

#include <atomic>

namespace hera {
namespace {

class RecordingBackend final : public IHeBackend, public IMpcBackend,
                               public IConversionBackend,
                               public IPackingBackend {
public:
    explicit RecordingBackend(std::shared_ptr<RecordingState> state)
        : state_(std::move(state)) {}

    Result<TensorHandle> MatMulCipherPlain(
        const TensorHandle &input, const WeightHandle &,
        const MatMulAttributes &, const TensorSpec &out) override
    { return Record(OpKind::kMatMulCipherPlain, {input}, out); }
    Result<TensorHandle> MatMulCipherCipher(
        const TensorHandle &left, const TensorHandle &right,
        const MatMulAttributes &, const TensorSpec &out) override
    { return Record(OpKind::kMatMulCipherCipher, {left, right}, out); }
    Result<TensorHandle> Add(
        const TensorHandle &left, const TensorHandle &right,
        const TensorSpec &out) override
    { return Record(OpKind::kElementwiseAdd, {left, right}, out); }
    Result<TensorHandle> Multiply(
        const TensorHandle &left, const TensorHandle &right,
        const TensorSpec &out) override
    { return Record(OpKind::kElementwiseMultiply, {left, right}, out); }
    Result<TensorHandle> AddPlain(
        const TensorHandle &input, const PlainParameterHandle &,
        const ElementwisePlainAttributes &, const TensorSpec &out) override
    { return Record(OpKind::kElementwiseAddPlain, {input}, out); }
    Result<TensorHandle> MultiplyPlain(
        const TensorHandle &input, const PlainParameterHandle &,
        const ElementwisePlainAttributes &, const TensorSpec &out) override
    { return Record(OpKind::kElementwiseMultiplyPlain, {input}, out); }
    Result<TensorHandle> ScalarAdd(
        const TensorHandle &input, const ScalarAttributes &,
        const TensorSpec &out) override
    { return Record(OpKind::kScalarAdd, {input}, out); }
    Result<TensorHandle> ScalarMultiply(
        const TensorHandle &input, const ScalarAttributes &,
        const TensorSpec &out) override
    { return Record(OpKind::kScalarMultiply, {input}, out); }
    Result<TensorHandle> ReduceSum(
        const TensorHandle &input, const ReductionAttributes &,
        const TensorSpec &out) override
    { return Record(OpKind::kReduceSum, {input}, out); }
    Result<TensorHandle> Broadcast(
        const TensorHandle &input, const BroadcastAttributes &,
        const TensorSpec &out) override
    { return Record(OpKind::kBroadcast, {input}, out); }
    Result<TensorHandle> Polynomial(
        const TensorHandle &input, const PolynomialAttributes &,
        const TensorSpec &out) override
    { return Record(OpKind::kPolynomial, {input}, out); }
    Result<TensorHandle> RealifyComplex(
        const TensorHandle &input, const TensorSpec &out) override
    { return Record(OpKind::kHeRealifyComplex, {input}, out); }
    Result<TensorHandle> ApplyPublicMask(
        const TensorHandle &input, const PublicAttentionMask &,
        const PublicMaskAttributes &,
        const TensorSpec &out) override
    { return Record(OpKind::kApplyPublicMask, {input}, out); }
    Result<TensorHandle> Compare(
        const std::vector<TensorHandle> &inputs,
        const MpcNonlinearAttributes &, const TensorSpec &out) override
    { return Record(OpKind::kCompare, inputs, out); }
    Result<TensorHandle> BooleanXor(
        const TensorHandle &left, const TensorHandle &right,
        const TensorSpec &out) override
    { return Record(OpKind::kBooleanXor, {left, right}, out); }
    Result<TensorHandle> BooleanNot(
        const TensorHandle &input, const TensorSpec &out) override
    { return Record(OpKind::kBooleanNot, {input}, out); }
    Result<TensorHandle> Mux(
        const std::vector<TensorHandle> &inputs,
        const MpcNonlinearAttributes &, const TensorSpec &out) override
    { return Record(OpKind::kMux, inputs, out); }
    Result<TensorHandle> AddArithmeticShares(
        const TensorHandle &left, const TensorHandle &right,
        const TensorSpec &out) override
    { return Record(OpKind::kMpcAdd, {left, right}, out); }
    Result<TensorHandle> ReduceMax(
        const TensorHandle &input, const PublicAttentionMask &,
        const MpcNonlinearAttributes &, const TensorSpec &out) override
    { return Record(OpKind::kMpcReduceMax, {input}, out); }
    Result<TensorHandle> PrepareMaskedScores(
        const TensorHandle &input, const TensorHandle &row_max,
        const PublicAttentionMask &, const MpcNonlinearAttributes &,
        const TensorSpec &out) override
    { return Record(OpKind::kMpcPrepareMaskedScores, {input, row_max}, out); }
    Result<TensorHandle> ReduceSum(
        const TensorHandle &input, const MpcNonlinearAttributes &,
        const TensorSpec &out) override
    { return Record(OpKind::kMpcReduceSum, {input}, out); }
    Result<TensorHandle> MultiplyFixed(
        const TensorHandle &left, const TensorHandle &right,
        const MpcNonlinearAttributes &, const TensorSpec &out) override
    { return Record(OpKind::kMpcMultiplyFixed, {left, right}, out); }
    Result<TensorHandle> Reciprocal(
        const TensorHandle &input, const MpcNonlinearAttributes &,
        const TensorSpec &out) override
    { return Record(OpKind::kReciprocal, {input}, out); }
    Result<TensorHandle> ReciprocalSqrt(
        const TensorHandle &input, const MpcNonlinearAttributes &,
        const TensorSpec &out) override
    { return Record(OpKind::kReciprocalSqrt, {input}, out); }
    Result<TensorHandle> HeToMpc(
        const TensorHandle &input, const ConversionAttributes &,
        const TensorSpec &out) override
    { return Record(OpKind::kHeToMpc, {input}, out); }
    Result<TensorHandle> MpcToHe(
        const TensorHandle &input, const ConversionAttributes &,
        const TensorSpec &out) override
    { return Record(OpKind::kMpcToHe, {input}, out); }
    Result<TensorHandle> RepackLocalShares(
        const TensorHandle &input, const RepackAttributes &,
        const TensorSpec &out) override
    { return Record(OpKind::kLocalShareRepack, {input}, out); }

private:
    Result<TensorHandle> Record(OpKind kind,
                                const std::vector<TensorHandle> &inputs,
                                const TensorSpec &output_spec)
    {
        TensorHandle output{next_id_.fetch_add(1), output_spec};
        ExecutionRecord record;
        record.operation = kind;
        for (const auto &input : inputs) record.inputs.push_back(input.id);
        record.output = output.id;
        state_->records.push_back(std::move(record));
        return output;
    }

    std::shared_ptr<RecordingState> state_;
    std::atomic<TensorId> next_id_{1000000};
};

} // namespace

BackendSet MakeRecordingBackends(std::shared_ptr<RecordingState> state)
{
    auto backend = std::make_shared<RecordingBackend>(std::move(state));
    return {backend, backend, backend, backend};
}

} // namespace hera
