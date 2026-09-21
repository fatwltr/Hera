#include "hera/runtime.h"

namespace hera {

Result<std::shared_ptr<HeraContext>> HeraContext::CreateFromJson(
    const std::filesystem::path &path)
{
    auto config = LoadHeraConfig(path);
    if (!config.ok()) return config.status();
    return Create(config.take_value());
}

Result<std::shared_ptr<HeraContext>> HeraContext::Create(HeraConfig config)
{
    const auto valid = ValidateHeraConfig(config);
    if (!valid.ok()) return valid.status();
    auto graph = BuildTransformerGraph(config);
    if (!graph.ok()) return graph.status();
    return std::shared_ptr<HeraContext>(
        new HeraContext(std::move(config), graph.take_value()));
}

HeraContext::HeraContext(HeraConfig config, TransformerGraph graph)
    : config_(std::move(config)), graph_(std::move(graph)),
      backends_(MakeUnavailableBackends())
{
}

Result<void> HeraContext::BindBackends(BackendSet backends)
{
    const auto valid = backends.Validate();
    if (!valid.ok()) return valid.status();
    backends_ = std::move(backends);
    return {};
}

Result<void> HeraContext::BindExecutionRuntime(
    std::shared_ptr<ITransformerExecutionRuntime> runtime)
{
    if (!runtime)
        return Status::Invalid("execution runtime must not be null");
    execution_runtime_ = std::move(runtime);
    return {};
}

} // namespace hera
