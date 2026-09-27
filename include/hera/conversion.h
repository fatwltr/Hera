#pragma once

#include "hera/he_context_registry.h"
#include "hera/tensor.h"
#include "hera/wide_ring.h"

#include <seal/seal.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace hera {

enum class ComplexComponent { kReal, kImaginary };

constexpr std::size_t kUnusedLogicalIndex =
    static_cast<std::size_t>(-1);

struct PhysicalSlotAssignment {
    std::size_t real_logical_index{kUnusedLogicalIndex};
    std::size_t imaginary_logical_index{kUnusedLogicalIndex};

    bool operator==(const PhysicalSlotAssignment &other) const noexcept
    {
        return real_logical_index == other.real_logical_index &&
               imaginary_logical_index == other.imaginary_logical_index;
    }
    bool operator!=(const PhysicalSlotAssignment &other) const noexcept
    {
        return !(*this == other);
    }
};

// Entries are ordered by ciphertext, then slot. Each logical index may occur
// at most once across both lanes. Unused physical lanes use
// kUnusedLogicalIndex and are encoded as zero.
struct PhysicalSlotMap {
    std::size_t ciphertext_count{};
    std::size_t slots_per_ciphertext{};
    std::size_t logical_element_count{};
    std::vector<PhysicalSlotAssignment> assignments;

    static PhysicalSlotMap DenseComplex(std::size_t logical_elements,
                                        std::size_t slots);
    static PhysicalSlotMap DenseReal(std::size_t logical_elements,
                                     std::size_t slots);
    Result<void> Validate() const;
};

struct ConversionConfig {
    int ring_bits{43};
    int scale_bits{13};
    int statistical_security_bits{40};
    int fft_scale_bits{20};
    bool allow_internal_mod_switch{true};
};

struct SealCipherBatch {
    std::string parameter_profile;
    std::vector<seal::Ciphertext> ciphertexts;
    TensorShape logical_shape;
    Layout logical_layout{Layout::kSpatial};
    PhysicalSlotMap slot_map;
};

struct MpcShareTensor {
    std::vector<std::uint64_t> local_share;
    TensorShape physical_shape;
    NumericFormat numeric;
    ComplexComponent component{ComplexComponent::kReal};

    Result<void> Validate() const;
};

struct ComplexMpcSharePair {
    MpcShareTensor real;
    MpcShareTensor imag;

    Result<void> Validate() const;
};

struct MpcToCkksTarget {
    std::string parameter_profile;
    std::optional<std::size_t> target_level;
    int scale_bits{};
    TensorShape logical_shape;
    Layout logical_layout{Layout::kSpatial};
    PhysicalSlotMap slot_map;
};

struct ConversionStats {
    std::uint64_t communication_bytes{};
    std::uint64_t rounds{};
    std::uint64_t a_mask_bytes{};
    std::uint64_t z_mask_bytes{};
    std::size_t internal_mod_switches{};
    std::size_t transform_cache_hits{};
    std::size_t transform_cache_misses{};
    std::size_t local_transform_tasks{};
    std::size_t local_transform_peak_inflight{};
    double milliseconds{};
};

class IProtocolTransport {
public:
    virtual ~IProtocolTransport() = default;
    virtual void Send(const void *data, std::size_t bytes) = 0;
    virtual void Receive(void *data, std::size_t bytes) = 0;
    virtual void Cancel() noexcept = 0;
    virtual std::uint64_t bytes() const noexcept = 0;
    virtual std::uint64_t rounds() const noexcept = 0;
};

// Length-delimited public-key bundle exchange for two-machine startup. No
// secret key material is accepted by this wire format.
Result<void> SendPublicContextBundle(
    IProtocolTransport &transport,
    const SerializedPublicContextBundle &bundle);
Result<SerializedPublicContextBundle> ReceivePublicContextBundle(
    IProtocolTransport &transport);

class IRandomStream {
public:
    virtual ~IRandomStream() = default;
    virtual void Fill(void *data, std::size_t bytes) = 0;
};

class IRandomSource {
public:
    virtual ~IRandomSource() = default;
    virtual Result<std::unique_ptr<IRandomStream>> OpenStream(
        const std::string &domain_label) = 0;
};

std::shared_ptr<IRandomSource> MakeProductionRandomSource();
std::shared_ptr<IRandomSource> MakeDeterministicRandomSourceForTesting(
    const std::vector<std::uint8_t> &seed);

// Direct two-party implementation of BLB Algorithms 1 and 2 with an
// additional independent Z_(2^ell) output-share refresh in CkksToMpc. The
// client is the only party allowed to own a decryption key. The server is the
// only party that may pass a ciphertext to CkksToMpc and the only party that
// receives the MpcToCkks result.
class CkksMpcConverter {
public:
    CkksMpcConverter(PartyRole role, ConversionConfig config,
                     HeContextRegistry &contexts,
                     std::shared_ptr<IProtocolTransport> transport,
                     std::shared_ptr<IRandomSource> randomness,
                     std::string peer_address, int worker_port,
                     std::size_t threads,
                     seal::compr_mode_type wire_compression =
                         seal::compr_mode_type::none);
    ~CkksMpcConverter();
    CkksMpcConverter(CkksMpcConverter &&) noexcept;
    CkksMpcConverter &operator=(CkksMpcConverter &&) noexcept;
    CkksMpcConverter(const CkksMpcConverter &) = delete;
    CkksMpcConverter &operator=(const CkksMpcConverter &) = delete;

    Result<ComplexMpcSharePair> CkksToMpc(
        const std::optional<SealCipherBatch> &server_input);
    Result<std::optional<SealCipherBatch>> MpcToCkks(
        const ComplexMpcSharePair &local_shares,
        const MpcToCkksTarget &target);

    ConversionStats stats() const noexcept;
    void Cancel() noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

std::shared_ptr<IProtocolTransport> MakeNetIoTransport(
    PartyRole role, const std::string &peer_address, int port);

} // namespace hera
