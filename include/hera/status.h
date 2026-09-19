#pragma once

#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace hera {

enum class StatusCode {
    kOk,
    kInvalidArgument,
    kNotFound,
    kFailedPrecondition,
    kKernelUnavailable,
    kCancelled,
    kDeadlineExceeded,
    kPeerAborted,
    kInternal,
};

class Status {
public:
    Status() = default;
    Status(StatusCode code, std::string message)
        : code_(code), message_(std::move(message)) {}

    static Status Ok() { return {}; }
    static Status Invalid(std::string message)
    {
        return {StatusCode::kInvalidArgument, std::move(message)};
    }
    static Status NotFound(std::string message)
    {
        return {StatusCode::kNotFound, std::move(message)};
    }
    static Status FailedPrecondition(std::string message)
    {
        return {StatusCode::kFailedPrecondition, std::move(message)};
    }
    static Status KernelUnavailable(std::string message)
    {
        return {StatusCode::kKernelUnavailable, std::move(message)};
    }
    static Status Cancelled(std::string message)
    {
        return {StatusCode::kCancelled, std::move(message)};
    }
    static Status DeadlineExceeded(std::string message)
    {
        return {StatusCode::kDeadlineExceeded, std::move(message)};
    }
    static Status PeerAborted(std::string message)
    {
        return {StatusCode::kPeerAborted, std::move(message)};
    }
    static Status Internal(std::string message)
    {
        return {StatusCode::kInternal, std::move(message)};
    }

    bool ok() const noexcept { return code_ == StatusCode::kOk; }
    StatusCode code() const noexcept { return code_; }
    const std::string &message() const noexcept { return message_; }

private:
    StatusCode code_{StatusCode::kOk};
    std::string message_;
};

template <typename T>
class Result {
public:
    Result(T value) : value_(std::move(value)) {}
    Result(Status status) : status_(std::move(status))
    {
        if (status_.ok())
            status_ = Status::Internal("an error Result cannot contain OK");
    }

    bool ok() const noexcept { return value_.has_value(); }
    const Status &status() const noexcept { return status_; }

    T &value()
    {
        if (!ok())
            throw std::logic_error("accessing failed Result: " + status_.message());
        return *value_;
    }
    const T &value() const
    {
        if (!ok())
            throw std::logic_error("accessing failed Result: " + status_.message());
        return *value_;
    }
    T take_value()
    {
        if (!ok())
            throw std::logic_error("accessing failed Result: " + status_.message());
        return std::move(*value_);
    }

private:
    std::optional<T> value_;
    Status status_;
};

template <>
class Result<void> {
public:
    Result() = default;
    Result(Status status) : status_(std::move(status)) {}

    bool ok() const noexcept { return status_.ok(); }
    const Status &status() const noexcept { return status_; }

private:
    Status status_;
};

} // namespace hera
