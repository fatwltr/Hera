#pragma once

#include "hera/model.h"

#include <cstdint>
#include <atomic>
#include <optional>
#include <string>
#include <vector>

namespace hera {

using TensorId = std::uint64_t;
using WeightId = std::uint64_t;
using PlainParameterId = std::uint64_t;

// A runtime-wide allocator keeps public tensor identities identical on both
// parties. Both parties execute the same stage schedule and therefore consume
// IDs in the same order; no secret-dependent branch may allocate an ID.
class TensorIdAllocator {
public:
    explicit TensorIdAllocator(TensorId first = 10000000)
        : next_(first) {}

    TensorId Next() noexcept { return next_.fetch_add(1); }
    TensorId Peek() const noexcept { return next_.load(); }

private:
    std::atomic<TensorId> next_;
};

enum class Domain {
    kHeCiphertext,
    kMpcArithmeticShare,
    kMpcBooleanShare,
    kPlaintext,
};

enum class Layout {
    kSpatial,
    kReduction,
    kMultiHeadQ,
    kMultiHeadK,
    kMultiHeadV,
    kAttentionDiagonal,
    kSoftmaxRows,
    // A row-wise Softmax matrix scattered by cyclic attention diagonal for
    // direct consumption by the encrypted A*V kernel.
    kSoftmaxAvDiagonal,
    kAttentionOutput,
    // Feature-major CKKS packing. kComplexStrictColumn uses real and
    // imaginary lanes independently; kStrictColumn is real-only.
    kComplexStrictColumn,
    kStrictColumn,
    kFfn,
};

enum class HeRegionId {
    kRegion1,
    kRegion2,
    kRegion3,
    kRegion4,
    kRegion5,
};

struct TensorShape {
    std::vector<std::size_t> dimensions;

    Result<std::size_t> ElementCount() const;
    bool operator==(const TensorShape &other) const noexcept
    {
        return dimensions == other.dimensions;
    }
    bool operator!=(const TensorShape &other) const noexcept
    {
        return !(*this == other);
    }
};

struct NumericFormat {
    int ring_bits{43};
    int scale_bits{13};
    // Two's-complement interpretation is part of an MPC tensor contract.
    // CKKS tensors keep this metadata so a later conversion cannot silently
    // reinterpret signed logits as unsigned probabilities (or vice versa).
    bool is_signed{true};
};

struct TensorSpec {
    TensorShape shape;
    Domain domain{Domain::kHeCiphertext};
    Layout layout{Layout::kSpatial};
    NumericFormat numeric;
    std::optional<HeRegionId> he_region;
    std::optional<std::string> he_parameter_profile;
    std::optional<std::size_t> planned_level;
    std::optional<int> planned_scale_bits;
};

struct TensorHandle {
    TensorId id{};
    TensorSpec spec;

    bool valid() const noexcept { return id != 0; }
};

struct HeCipherTensorHandle {
    TensorHandle tensor;
};

struct WeightHandle {
    WeightId id{};
    MatrixSpec spec;

    bool valid() const noexcept { return id != 0; }
};

struct PlainParameterHandle {
    PlainParameterId id{};
    VectorSpec spec;

    bool valid() const noexcept { return id != 0; }
};

struct LayerWeightHandles {
    std::string layer_id;
    WeightHandle wq;
    WeightHandle wk;
    WeightHandle wv;
    WeightHandle wo;
    WeightHandle wu;
    WeightHandle wd;
    std::vector<PlainParameterHandle> plain_parameters;
};

struct TransformerWeightHandles {
    std::vector<LayerWeightHandles> layers;

    const WeightHandle *Find(const std::string &layer_id,
                             const std::string &weight_name) const;
    const PlainParameterHandle *FindPlain(
        const std::string &layer_id,
        const std::string &parameter_name) const;
};

struct PublicAttentionMask {
    AttentionMaskKind kind{AttentionMaskKind::kNone};
    TensorShape shape;
    // Row-major additive mask values. Public causal masks may leave this
    // empty because they are derivable from shape; padding masks provide all
    // tokens*tokens entries.
    std::vector<double> additive_values;
};

Result<void> ValidateTensorSpec(const TensorSpec &spec,
                                const std::string &name);
bool SameTensorContract(const TensorSpec &left,
                        const TensorSpec &right) noexcept;
bool SatisfiesTensorContract(const TensorSpec &actual,
                             const TensorSpec &expected) noexcept;

std::string ToString(Domain value);
std::string ToString(Layout value);
std::string ToString(HeRegionId value);
Result<HeRegionId> ParseHeRegionId(const std::string &value);
Result<Layout> ParseLayout(const std::string &value);

} // namespace hera
