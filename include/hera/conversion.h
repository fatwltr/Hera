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
    // Public inference-domain precondition for EVERY C2M input coefficient,
    // including HE noise: |a_i| < floor(q/4). Never inferred from weights or
    // observed secret data. Both peers must opt into the same contract.
    bool public_c2m_quarter_range{false};
    // Public precondition for BOTH M2C extensions: reconstructed signed slot
    // integers and post-Encode coefficient integers satisfy |x| < 2^(k-2)
    // in their respective k-bit source rings, including fixed-point error.
    bool public_m2c_quarter_range{false};
    // Performance simulation only: execute the bounded protocol without
    // requiring/providing a public range proof. Results outside the protocol
    // domain have no numerical correctness guarantee. Both peers must agree.
    // The model CLI enables this only for virtual weights.
    bool benchmark_unchecked_range{false};
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
    // A missing/insufficient bound without an explicit quarter-range declaration is rejected.
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
    double slot_prepare{};
    double coefficient_extend{};
    double slot_encode{};
    double slot_decode{};
    double rns{};
    double he{};
    double ciphertext_transport{};
};

struct ConversionStats {
    std::uint64_t communication_bytes{};
    std::uint64_t rounds{};
    std::uint64_t field_to_ring_communication_bytes{};
    std::uint64_t signed_extension_communication_bytes{};
    // Public OT inventory, summed over workers. Single-worker benchmarks use
    // it to include refill costs; these are not secret OT choices or pads.
    std::uint64_t forward_ot_capacity{};
    std::uint64_t forward_ot_used{};
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
    std::size_t coefficient_extension_elements{};
    std::size_t bounded_coefficient_extension_elements{};
    std::size_t field_to_ring_batches{};
    std::size_t bounded_field_to_ring_elements{};
    std::size_t max_field_to_ring_elements{};
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

std::shared_ptr<IProtocolTransport> MakeNetIoTransport(
    PartyRole role, const std::string &peer_address, int port);

} // namespace hera
