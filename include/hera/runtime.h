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
