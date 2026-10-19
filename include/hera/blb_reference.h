#pragma once

#include "hera/conversion.h"

namespace hera {

// Correctly prepared wide-ring slot shares at the configured MPC scale.
// The producer is responsible for securely obtaining them. Changing a narrow
// share's width label or zero-extending it does not satisfy this contract.
struct PreparedBlbSlots {
    std::vector<WideRingShare> real, imag;
    int ring_bits{};
    int scale_bits{};
};

// Engineered BLB Algorithms 1/2 converter. Batched ciphertexts, public lane
// maps, worker scheduling, transform/RNS caches and stage timing share one
// implementation. Only the quarter-range protocol is implemented; explicit
// benchmark_unchecked_range permits performance-only execution without a proof.
// CkksToMpc adds one
// uniform coefficient-domain mask A before client decryption and retains -A
// as the server's field share. A separate Encode(Z) mask is unnecessary:
// adding it would not change the uniform mask distribution, and Z has no
// later use. Only the client owns a decryption key. Only the server supplies
// CkksToMpc ciphertexts and receives the MpcToCkks result.
class BlbReferenceConverter {
public:
    BlbReferenceConverter(PartyRole role, ConversionConfig config,
                     HeContextRegistry &contexts,
                     std::shared_ptr<IProtocolTransport> transport,
                     std::shared_ptr<IRandomSource> randomness,
                     std::string peer_address, int worker_port,
                     std::size_t threads,
                     seal::compr_mode_type wire_compression =
                         seal::Serialization::compr_mode_default);
    ~BlbReferenceConverter();
    BlbReferenceConverter(BlbReferenceConverter &&) noexcept;
    BlbReferenceConverter &operator=(BlbReferenceConverter &&) noexcept;
    BlbReferenceConverter(const BlbReferenceConverter &) = delete;
    BlbReferenceConverter &operator=(const BlbReferenceConverter &) = delete;

    // Both parties request the same optional fractional precision; the
    // existing handshake negotiates it. Other calls keep the global format.
    Result<ComplexMpcSharePair> CkksToMpc(
        const std::optional<SealCipherBatch> &server_input,
        std::optional<int> output_scale_bits = std::nullopt,
        // Optional PUBLIC bound |a_i| < 2^bits on every true centered terminal
        // coefficient, including HE noise. Both peers must supply the same
        // producer-established bound. Missing/insufficient quarter-q
        // headroom is rejected before OT. Never infer this from shares.
        std::optional<int> public_coefficient_bound_bits = std::nullopt);
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

    // Algorithm 2 core: skips slot preparation, retaining coefficient R2F.
    // Shares cover target.slot_map's complete physical real/imaginary planes.
    Result<std::optional<SealCipherBatch>> MpcToCkksPrepared(
        const PreparedBlbSlots &local_shares, const MpcToCkksTarget &target);

    ConversionStats stats() const noexcept;
    void Cancel() noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hera
