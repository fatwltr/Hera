#pragma once

#include "hera/he_context_registry.h"
#include "hera/tensor.h"
#include "hera/wide_ring.h"

#include <seal/seal.h>

#include <cstddef>
#include <array>
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
    // Public upper bound on coefficients per secure batch. Zero chooses
    // 16384 coefficients per worker. At least one
    // complete ciphertext is processed even when its degree exceeds this.
    // Both parties must use the same value; negotiated before OT setup.
    std::size_t batch_elements{};
    static constexpr std::size_t DefaultBatchElements(std::size_t workers)
    { return workers * 16384; }
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

    // Optional public producer contract for the reconstructed signed integer:
    // |x| < 2^bits, before interpreting its fixed-point scale. It is never
    // inferred from a party's local shares.
    std::optional<int> public_value_bound_bits{};

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
    // Explicit public contract: imaginary lanes are unused and BOTH parties'
    // imaginary shares are canonical zero. Negotiated before secure work.
    bool public_zero_imaginary{false};
    // Applies to both components, including physical padding. The producer
    // must establish the bound; peers negotiate it before entering SCI.
    // A valid bound without quarter-ring headroom uses the general path.
    std::optional<int> public_value_bound_bits{};
    // Per-ciphertext refinements for concatenated boundaries. Empty or null
    // entries use the scalar contract. Peers negotiate the entire vector.
    std::vector<std::optional<int>> public_ciphertext_bound_bits{};
    // Optional public slot periods for {real, imaginary}, per ciphertext.
    // Zero means no replication. Only mapped lanes participate; replicas
    // must have equal local shares on BOTH parties, checked before SCI.
    // Conversion extends each representative once, then copies its wide
    // share. This never reuses OT randomness between different values.
    std::vector<std::array<std::size_t, 2>> public_slot_periods{};
};

// Accumulated work durations in milliseconds. Secure field/ring stages include
// their SCI communication. Async local work can overlap other stages, so these
// values must not be summed to estimate end-to-end latency.
struct ConversionStageTimes {
    double preparation{};
    double handshake{};
    double sampling{};
    double mask_encode{};
    double field_to_ring{};
    double signed_extend{};
    double slot_encode{};
    double slot_decode{};
    double rns{};
    double he{};
    double ciphertext_transport{};
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
    std::size_t public_zero_imaginary_ciphertexts{};
    std::size_t packed_real_input_ciphertexts{};
    std::size_t packed_real_wire_ciphertexts{};
    std::size_t real_output_split_ciphertexts{};
    std::size_t signed_extension_batches{};
    std::size_t max_signed_extension_elements{};
    std::size_t bounded_signed_extension_elements{};
    std::size_t generic_signed_extension_elements{};
    std::size_t wide_b2a_elements{};
    std::size_t field_to_ring_batches{};
    std::size_t max_field_to_ring_elements{};
    std::size_t radix5_comparison_elements{};
    std::size_t correlated_comparison_elements{};
    std::size_t slot_extension_input_elements{};
    std::size_t slot_extension_unique_elements{};
    std::size_t slot_extension_padding_elements{};
    double milliseconds{};
    ConversionStageTimes ckks_to_mpc;
    ConversionStageTimes mpc_to_ckks;
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

// Two-party conversion based on BLB Algorithms 1 and 2. CkksToMpc independently
// samples coefficient-domain A and slot-domain Z_(2^ell) masks and adds their
// combined coefficient mask A + Encode(Z) before client decryption. The
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
                         seal::Serialization::compr_mode_default);
    ~CkksMpcConverter();
    CkksMpcConverter(CkksMpcConverter &&) noexcept;
    CkksMpcConverter &operator=(CkksMpcConverter &&) noexcept;
    CkksMpcConverter(const CkksMpcConverter &) = delete;
    CkksMpcConverter &operator=(const CkksMpcConverter &) = delete;

    // Both parties request the same optional fractional precision; the
    // existing handshake negotiates it. Other calls keep the global format.
    Result<ComplexMpcSharePair> CkksToMpc(
        const std::optional<SealCipherBatch> &server_input,
        std::optional<int> output_scale_bits = std::nullopt);
    Result<std::optional<SealCipherBatch>> MpcToCkks(
        const ComplexMpcSharePair &local_shares,
        const MpcToCkksTarget &target);
    // Pack real-only ciphertexts into both lanes during conversion, encoding
    // the split's 1/2 in RNS. Restore the requested real layout afterwards
    // using conjugation and scale-1 phases, without rescaling or losing bits
    // from the MPC shares. Existing complex ciphertexts pass through.
    Result<std::optional<SealCipherBatch>> MpcToCkksPacked(
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
