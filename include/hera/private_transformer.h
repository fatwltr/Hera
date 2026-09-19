#pragma once

#include "hera/runtime.h"

namespace hera {

struct EncryptedTransformerInput {
    HeCipherTensorHandle hidden_states;
    PublicAttentionMask attention_mask;
};

class PrivateTransformer {
public:
    static Result<PrivateTransformer> Compile(
        std::shared_ptr<HeraContext> context);

    Result<void> Prepare(
        const TransformerWeightHandles &weights,
        PreparationReport *report = nullptr) const;

    Result<HeCipherTensorHandle> Evaluate(
        const EncryptedTransformerInput &input,
        const TransformerWeightHandles &weights,
        ExecutionReport *report = nullptr) const;

private:
    explicit PrivateTransformer(std::shared_ptr<HeraContext> context)
        : context_(std::move(context)) {}

    std::shared_ptr<HeraContext> context_;
};

} // namespace hera
