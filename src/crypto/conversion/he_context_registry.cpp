#include "hera/he_context_registry.h"

#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <set>
#include <streambuf>
#include <cstring>
#include <limits>

namespace hera {
namespace {

std::shared_ptr<seal::SEALContext> BuildContext(
    const HeParameterConfig &profile)
{
    seal::EncryptionParameters parameters(seal::scheme_type::ckks);
    parameters.set_poly_modulus_degree(profile.polynomial_degree);
    // HeParameterConfig describes ciphertext data primes. SEAL reserves the
    // last coefficient modulus for key switching and removes it from
    // first_context_data(), so append an auxiliary special prime instead of
    // silently consuming the last configured data prime.
    auto modulus_bits = profile.coefficient_modulus_bits;
    modulus_bits.push_back(std::clamp(
        profile.coefficient_modulus_bits.front(), 30, 60));
    parameters.set_coeff_modulus(seal::CoeffModulus::Create(
        profile.polynomial_degree, modulus_bits));
    const auto security = profile.security_level == SecurityLevel::kTc128
                              ? seal::sec_level_type::tc128
                              : seal::sec_level_type::none;
    return std::make_shared<seal::SEALContext>(parameters, true, security);
}

} // namespace

class FixedVectorOutputBuffer final : public std::streambuf {
public:
    explicit FixedVectorOutputBuffer(std::vector<std::uint8_t> &bytes)
        : bytes_(bytes)
    {
        auto *begin = reinterpret_cast<char *>(bytes_.data());
        setp(begin, begin + bytes_.size());
    }

    std::size_t Written() const noexcept { return pptr() - pbase(); }

protected:
    std::streamsize xsputn(const char *source,
                           std::streamsize size) override
    {
        if (size < 0 || size > epptr() - pptr() ||
            size > std::numeric_limits<int>::max()) return 0;
        std::memcpy(pptr(), source, static_cast<std::size_t>(size));
        pbump(static_cast<int>(size));
        return size;
    }

    int_type overflow(int_type value) override
    {
        if (pptr() == epptr()) return traits_type::eof();
        *pptr() = traits_type::to_char_type(value);
        pbump(1);
        return value;
    }

private:
    std::vector<std::uint8_t> &bytes_;
};

class VectorInputBuffer final : public std::streambuf {
public:
    explicit VectorInputBuffer(const std::vector<std::uint8_t> &bytes)
    {
        auto *begin = reinterpret_cast<char *>(
            const_cast<std::uint8_t *>(bytes.data()));
        setg(begin, begin, begin + bytes.size());
    }
};

template <typename SealObject>
std::vector<std::uint8_t> Serialize(
    const SealObject &object, seal::compr_mode_type compression)
{
    const auto upper_bound = object.save_size(compression);
    if (upper_bound <= 0 || static_cast<std::uint64_t>(upper_bound) >
                                (std::uint64_t{1} << 34))
        throw std::runtime_error("public HE key serialization is too large");
    std::vector<std::uint8_t> bytes(
        static_cast<std::size_t>(upper_bound));
    FixedVectorOutputBuffer buffer(bytes);
    std::ostream stream(&buffer);
    stream.exceptions(std::ios::badbit | std::ios::failbit);
    object.save(stream, compression);
    bytes.resize(buffer.Written());
    return bytes;
}

template <typename SealObject>
SealObject Deserialize(const std::vector<std::uint8_t> &bytes,
                       const seal::SEALContext &context)
{
    VectorInputBuffer buffer(bytes);
    std::istream stream(&buffer);
    SealObject object;
    object.load(context, stream);
    return object;
}

Result<void> HeContextRegistry::CreateClientProfiles(const HeraConfig &config)
{
    if (initialized_)
        return Status::FailedPrecondition(
            "HE context registry is already initialized");
    const auto valid = ValidateHeraConfig(config);
    if (!valid.ok()) return valid.status();
    try {
        for (const auto &entry : config.he_parameter_profiles) {
            auto context = BuildContext(entry.second);
            if (!context->parameters_set())
                return Status::Invalid(entry.first +
                                       ": SEAL rejected HE parameters");
            seal::KeyGenerator keygen(*context);
            HeContext value;
            value.profile = entry.first;
            value.parameters = entry.second;
            value.context = std::move(context);
            value.secret_key = keygen.secret_key();
            keygen.create_public_key(value.public_key);
            keygen.create_relin_keys(value.relinearization_keys);
            // Conversion itself does not rotate. Rotation keys are populated
            // later by the HE backend's compiled rotation plan.
            contexts_.emplace(entry.first, std::move(value));
        }
    } catch (const std::exception &error) {
        contexts_.clear();
        return Status::Internal(std::string("creating client HE profiles: ") +
                                error.what());
    }
    role_ = PartyRole::kClient;
    initialized_ = true;
    return {};
}

Result<PublicContextBundle> HeContextRegistry::ExportPublicBundle() const
{
    if (!initialized_ || role_ != PartyRole::kClient)
        return Status::FailedPrecondition(
            "only an initialized client registry exports public keys");
    PublicContextBundle result;
    for (const auto &entry : contexts_) {
        ProfilePublicMaterial material;
        material.profile = entry.first;
        material.public_key = entry.second.public_key;
        material.relinearization_keys = entry.second.relinearization_keys;
        material.galois_keys = entry.second.galois_keys;
        result.profiles.push_back(std::move(material));
    }
    return result;
}

Result<SerializedPublicContextBundle>
HeContextRegistry::ExportSerializedPublicBundle(
    seal::compr_mode_type compression) const
{
    if (!initialized_ || role_ != PartyRole::kClient)
        return Status::FailedPrecondition(
            "only an initialized client registry exports public keys");
    try {
        SerializedPublicContextBundle result;
        for (const auto &entry : contexts_) {
            SerializedProfilePublicMaterial material;
            material.profile = entry.first;
            material.public_key = Serialize(entry.second.public_key,
                                            compression);
            material.relinearization_keys =
                Serialize(entry.second.relinearization_keys, compression);
            // An empty GaloisKeys object is valid for the conversion-only
            // build; a later HE kernel may populate it before export.
            if (entry.second.galois_keys.size() != 0)
                material.galois_keys = Serialize(entry.second.galois_keys,
                                                 compression);
            result.profiles.push_back(std::move(material));
        }
        return result;
    } catch (const std::exception &error) {
        return Status::Internal(std::string("serializing public HE bundle: ") +
                                error.what());
    }
}

Result<void> HeContextRegistry::CreateServerProfiles(
    const HeraConfig &config, const PublicContextBundle &bundle)
{
    if (initialized_)
        return Status::FailedPrecondition(
            "HE context registry is already initialized");
    const auto valid = ValidateHeraConfig(config);
    if (!valid.ok()) return valid.status();
    if (bundle.profiles.size() != config.he_parameter_profiles.size())
        return Status::Invalid("public HE profile bundle size mismatch");
    try {
        for (const auto &entry : config.he_parameter_profiles) {
            const ProfilePublicMaterial *material = nullptr;
            for (const auto &candidate : bundle.profiles)
                if (candidate.profile == entry.first) material = &candidate;
            if (!material)
                return Status::Invalid("public HE profile is absent: " +
                                       entry.first);
            auto context = BuildContext(entry.second);
            if (!context->parameters_set())
                return Status::Invalid(entry.first +
                                       ": SEAL rejected HE parameters");
            HeContext value;
            value.profile = entry.first;
            value.parameters = entry.second;
            value.context = std::move(context);
            if (material->public_key.parms_id() !=
                    value.context->key_parms_id() ||
                material->relinearization_keys.parms_id() !=
                    value.context->key_parms_id() ||
                (material->galois_keys.size() != 0 &&
                 material->galois_keys.parms_id() !=
                     value.context->key_parms_id()))
                return Status::Invalid(
                    entry.first + ": public key bundle/context mismatch");
            value.public_key = material->public_key;
            value.relinearization_keys = material->relinearization_keys;
            value.galois_keys = material->galois_keys;
            contexts_.emplace(entry.first, std::move(value));
        }
    } catch (const std::exception &error) {
        contexts_.clear();
        return Status::Internal(std::string("creating server HE profiles: ") +
                                error.what());
    }
    role_ = PartyRole::kServer;
    initialized_ = true;
    return {};
}

Result<void> HeContextRegistry::CreateServerProfiles(
    const HeraConfig &config, const SerializedPublicContextBundle &bundle)
{
    if (bundle.protocol_version != 2)
        return Status::FailedPrecondition(
            "public HE bundle protocol version mismatch");
    if (initialized_)
        return Status::FailedPrecondition(
            "HE context registry is already initialized");
    const auto valid = ValidateHeraConfig(config);
    if (!valid.ok()) return valid.status();
    if (bundle.profiles.size() != config.he_parameter_profiles.size())
        return Status::Invalid("serialized public HE profile count mismatch");
    try {
        for (const auto &entry : config.he_parameter_profiles) {
            const SerializedProfilePublicMaterial *material = nullptr;
            for (const auto &candidate : bundle.profiles)
                if (candidate.profile == entry.first) material = &candidate;
            if (!material)
                return Status::Invalid("serialized HE profile is absent: " +
                                       entry.first);
            auto context = BuildContext(entry.second);
            if (!context->parameters_set())
                return Status::Invalid(entry.first +
                                       ": SEAL rejected HE parameters");
            HeContext value;
            value.profile = entry.first;
            value.parameters = entry.second;
            value.context = std::move(context);
            value.public_key = Deserialize<seal::PublicKey>(
                material->public_key, *value.context);
            value.relinearization_keys = Deserialize<seal::RelinKeys>(
                material->relinearization_keys, *value.context);
            if (!material->galois_keys.empty())
                value.galois_keys = Deserialize<seal::GaloisKeys>(
                    material->galois_keys, *value.context);
            contexts_.emplace(entry.first, std::move(value));
        }
    } catch (const std::exception &error) {
        contexts_.clear();
        return Status::Internal(
            std::string("loading serialized public HE bundle: ") +
            error.what());
    }
    role_ = PartyRole::kServer;
    initialized_ = true;
    return {};
}

const HeContext *HeContextRegistry::Find(std::string_view profile) const noexcept
{
    const auto it = contexts_.find(std::string(profile));
    return it == contexts_.end() ? nullptr : &it->second;
}

Result<void> HeContextRegistry::CompileRotationKeys(
    std::string_view profile, const std::vector<int> &steps)
{
    if (!initialized_ || role_ != PartyRole::kClient)
        return Status::FailedPrecondition(
            "only an initialized client registry can compile Galois keys");
    const auto found = contexts_.find(std::string(profile));
    if (found == contexts_.end())
        return Status::NotFound("unknown HE profile: " + std::string(profile));
    if (!found->second.secret_key)
        return Status::FailedPrecondition("client HE profile has no secret key");
    try {
        std::set<int> unique(steps.begin(), steps.end());
        std::vector<int> normalized(unique.begin(), unique.end());
        seal::KeyGenerator keygen(*found->second.context,
                                  *found->second.secret_key);
        if (normalized.empty())
            found->second.galois_keys = seal::GaloisKeys{};
        else
            keygen.create_galois_keys(normalized,
                                      found->second.galois_keys);
        return {};
    } catch (const std::exception &error) {
        return Status::Invalid(std::string("cannot create Galois keys: ") +
                               error.what());
    }
}

const HeContext &HeContextRegistry::Require(std::string_view profile,
                                            PartyRole role) const
{
    if (!initialized_)
        throw std::logic_error("HE context registry is not initialized");
    if (role != role_)
        throw std::logic_error("HE context registry party-role mismatch");
    const auto *result = Find(profile);
    if (!result)
        throw std::invalid_argument("unknown HE parameter profile: " +
                                    std::string(profile));
    if (role == PartyRole::kClient && !result->secret_key)
        throw std::logic_error("client HE profile has no secret key");
    if (role == PartyRole::kServer && result->secret_key)
        throw std::logic_error("server HE profile unexpectedly has secret key");
    return *result;
}

} // namespace hera
