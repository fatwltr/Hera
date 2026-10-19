#pragma once

#include "hera/backend.h"

#include <filesystem>
#include <memory>

namespace hera {

struct NodeExecutionDiagnostic {
    NodeId node{};
    std::string name;
    OpKind operation{OpKind::kElementwiseAdd};
    Status status;
};

struct ExecutionReport {
    std::vector<NodeExecutionDiagnostic> nodes;
    // Disjoint computation intervals for this party. HE is explicitly timed;
    // conversion includes its HE operations, transport and peer waiting;
    // other nonlinear covers MPC calls. Work outside these intervals is
    // recorded separately, never attributed to HE by subtraction.
    struct CostBreakdown {
        double he_computation_milliseconds{};
        double he_mpc_conversion_milliseconds{};
        double other_nonlinear_milliseconds{};
        double local_overhead_milliseconds{};

        double total_milliseconds() const noexcept
        {
            return he_computation_milliseconds + he_mpc_conversion_milliseconds +
                   other_nonlinear_milliseconds;
        }

        double stage_milliseconds() const noexcept
        {
            return total_milliseconds() + local_overhead_milliseconds;
        }

        CostBreakdown &operator+=(const CostBreakdown &value) noexcept
        {
            he_computation_milliseconds += value.he_computation_milliseconds;
            he_mpc_conversion_milliseconds += value.he_mpc_conversion_milliseconds;
            other_nonlinear_milliseconds += value.other_nonlinear_milliseconds;
            local_overhead_milliseconds += value.local_overhead_milliseconds;
            return *this;
        }
    };
    struct StageMetrics {
        std::uint64_t communication_bytes{};
        std::uint64_t rounds{};
        std::size_t conversions{};
        std::size_t rotations{};
        std::size_t ciphertext_plain_multiplications{};
        std::size_t ciphertext_multiplications{};
        std::size_t relinearizations{};
        std::size_t rescales{};
        std::size_t mpc_operations{};
        std::size_t layer_norm_he_workers{};
        std::size_t layer_norm_mpc_workers{};
        std::size_t gelu_mpc_workers{};
        CostBreakdown costs;
        // Bytes sent during this stage's online calls, after session setup.
        std::uint64_t mpc_communication_bytes{};
        std::uint64_t conversion_communication_bytes{};
    };
    struct Stage {
        std::size_t layer{};
        std::string name;
        double milliseconds{};
        Status status;
        StageMetrics metrics;
    };
    std::vector<Stage> stages;
    double setup_milliseconds{};
    double online_milliseconds{};
    double output_transfer_milliseconds{};
    std::uint64_t communication_bytes{};
    std::uint64_t rounds{};
    CostBreakdown costs;
};

struct PreparationReport {
    double milliseconds{};
    std::size_t encoded_plaintext_count{};
    std::size_t encoded_bytes{};
    std::size_t cache_hits{};
    std::size_t virtual_weight_requests{};
    std::size_t virtual_weight_cache_hits{};
};

// High-level concrete runtimes may execute fused cryptographic stages while
// the typed graph remains the source of shape/domain contracts. Keeping this
// interface free of SEAL/SCI types preserves the framework-only build.
class ITransformerExecutionRuntime {
public:
    virtual ~ITransformerExecutionRuntime() = default;
    virtual Result<void> Prepare(
        const TransformerWeightHandles &weights,
        PreparationReport *report) = 0;
    virtual Result<TensorHandle> Evaluate(
        const TensorHandle &input,
        const TransformerWeightHandles &weights,
        const PublicAttentionMask &mask,
        ExecutionReport *report) = 0;
};

class HeraContext {
public:
    static Result<std::shared_ptr<HeraContext>> CreateFromJson(
        const std::filesystem::path &path);
    static Result<std::shared_ptr<HeraContext>> Create(HeraConfig config);

    Result<void> BindBackends(BackendSet backends);
    Result<void> BindExecutionRuntime(
        std::shared_ptr<ITransformerExecutionRuntime> runtime);

    const HeraConfig &config() const noexcept { return config_; }
    const TransformerGraph &graph() const noexcept { return graph_; }
    const BackendSet &backends() const noexcept { return backends_; }
    const std::shared_ptr<ITransformerExecutionRuntime> &execution_runtime()
        const noexcept { return execution_runtime_; }

private:
    HeraContext(HeraConfig config, TransformerGraph graph);

    HeraConfig config_;
    TransformerGraph graph_;
    BackendSet backends_;
    std::shared_ptr<ITransformerExecutionRuntime> execution_runtime_;
};

class GraphExecutor {
public:
    explicit GraphExecutor(std::shared_ptr<HeraContext> context)
        : context_(std::move(context)) {}

    Result<TensorHandle> Execute(
        const TensorHandle &input,
        const TransformerWeightHandles &weights,
        const PublicAttentionMask &mask,
        ExecutionReport *report = nullptr) const;

private:
    std::shared_ptr<HeraContext> context_;
};

} // namespace hera
