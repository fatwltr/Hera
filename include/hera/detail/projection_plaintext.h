#pragma once

#include <seal/plaintext.h>
#include <memory>
#include <string>

namespace hera {

class ProjectionPlaintextCache;

// Projection terms own immutable handles rather than pinning all model NTT
// plaintexts. Acquire pins one plaintext for the duration of an HE operation.
class ProjectionPlaintext {
public:
    ProjectionPlaintext() = default;
    ProjectionPlaintext(std::shared_ptr<const seal::Plaintext> plain)
        : direct_(std::move(plain)) {}
    explicit operator bool() const noexcept { return direct_ || cache_; }
    std::shared_ptr<const seal::Plaintext> Acquire() const;
    // Inspection convenience; the calling thread pins its most recent view.
    const seal::Plaintext &operator*() const;
    const seal::Plaintext *operator->() const { return &**this; }

private:
    friend class ProjectionPlaintextCache;
    ProjectionPlaintext(std::shared_ptr<ProjectionPlaintextCache> cache, std::string key)
        : cache_(std::move(cache)), key_(std::move(key)) {}
    std::shared_ptr<const seal::Plaintext> direct_;
    std::shared_ptr<ProjectionPlaintextCache> cache_;
    std::string key_;
};

} // namespace hera
