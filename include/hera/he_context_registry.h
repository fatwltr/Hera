#pragma once

#include "hera/config.h"

#include <seal/seal.h>

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace hera {

enum class PartyRole { kClient, kServer };

struct HeContext {
    std::string profile;
    HeParameterConfig parameters;
    std::shared_ptr<seal::SEALContext> context;
    seal::PublicKey public_key;
    seal::RelinKeys relinearization_keys;
    seal::GaloisKeys galois_keys;
    std::optional<seal::SecretKey> secret_key;

    bool has_secret_key() const noexcept { return secret_key.has_value(); }
    std::size_t slot_count() const noexcept
    {
        return parameters.polynomial_degree / 2;
    }
};

struct ProfilePublicMaterial {
    std::string profile;
    seal::PublicKey public_key;
    seal::RelinKeys relinearization_keys;
    seal::GaloisKeys galois_keys;
};

struct PublicContextBundle {
    std::vector<ProfilePublicMaterial> profiles;
};

struct SerializedProfilePublicMaterial {
    std::string profile;
    std::vector<std::uint8_t> public_key;
    std::vector<std::uint8_t> relinearization_keys;
    std::vector<std::uint8_t> galois_keys;
};

struct SerializedPublicContextBundle {
    std::uint32_t protocol_version{2};
    std::vector<SerializedProfilePublicMaterial> profiles;
};

class HeContextRegistry {
public:
    Result<void> CreateClientProfiles(const HeraConfig &config);
    Result<PublicContextBundle> ExportPublicBundle() const;
    Result<SerializedPublicContextBundle> ExportSerializedPublicBundle(
        seal::compr_mode_type compression =
            seal::compr_mode_type::none) const;
    Result<void> CreateServerProfiles(const HeraConfig &config,
                                      const PublicContextBundle &bundle);
    Result<void> CreateServerProfiles(
        const HeraConfig &config,
        const SerializedPublicContextBundle &bundle);

    // Called by the client before exporting the public bundle. The steps are
    // compiled from the HE graph; step zero requests complex conjugation.
    Result<void> CompileRotationKeys(
        std::string_view profile, const std::vector<int> &steps);

    const HeContext &Require(std::string_view profile,
                             PartyRole role) const;
    const HeContext *Find(std::string_view profile) const noexcept;
    PartyRole role() const noexcept { return role_; }
    bool initialized() const noexcept { return initialized_; }

private:
    PartyRole role_{PartyRole::kServer};
    bool initialized_{};
    std::map<std::string, HeContext> contexts_;
};

} // namespace hera
