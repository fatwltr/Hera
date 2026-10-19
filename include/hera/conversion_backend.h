#pragma once

#include "hera/backend.h"
#include "hera/blb_reference.h"

#include <map>
#include <atomic>
#include <mutex>
#include <optional>
#include <tuple>

namespace hera {

// Owns cryptographic payloads outside TensorHandle. TensorHandle remains an
// opaque graph-level ID and never exposes a ciphertext or an MPC share.
class CryptoTensorStore {
public:
    Result<void> PutCiphertext(TensorId id,
                               std::optional<SealCipherBatch> payload);
    Result<void> PutMpcShare(TensorId id, MpcShareTensor payload);

    Result<std::optional<SealCipherBatch>> GetCiphertext(TensorId id) const;
    Result<MpcShareTensor> GetMpcShare(TensorId id) const;
    void Erase(TensorId id) noexcept;
    std::size_t Size() const noexcept;

private:
    mutable std::mutex mutex_;
    std::map<TensorId, std::optional<SealCipherBatch>> ciphertexts_;
    std::map<TensorId, MpcShareTensor> mpc_shares_;
};

// Physical packing is a compiled property of an HE kernel. Producers should
// register their exact map here. Dense complex packing is used only when no
// specialized map is registered.
class ConversionPackingRegistry {
public:
    Result<void> Register(const TensorSpec &spec, PhysicalSlotMap map);
    Result<PhysicalSlotMap> Resolve(const TensorSpec &spec,
                                    std::size_t slots) const;

private:
    using Key = std::tuple<std::string, Layout, std::vector<std::size_t>>;
    static Result<Key> MakeKey(const TensorSpec &spec);
    std::map<Key, PhysicalSlotMap> maps_;
};

// Graph adapter for BlbReferenceConverter. HeToMpc compacts the physical real/imag
// planes into one logical arithmetic-share tensor; MpcToHe scatters it back
// according to the target HE producer's PhysicalSlotMap.
class SealSciConversionBackend final : public IConversionBackend {
public:
    SealSciConversionBackend(PartyRole role, HeContextRegistry &contexts,
                             BlbReferenceConverter &converter,
                             CryptoTensorStore &store,
                             ConversionPackingRegistry &packing);

    Result<TensorHandle> HeToMpc(
        const TensorHandle &ciphertext,
        const ConversionAttributes &attributes,
        const TensorSpec &output_spec) override;
    Result<TensorHandle> MpcToHe(
        const TensorHandle &share,
        const ConversionAttributes &attributes,
        const TensorSpec &output_spec) override;

private:
    PartyRole role_;
    HeContextRegistry &contexts_;
    BlbReferenceConverter &converter_;
    CryptoTensorStore &store_;
    ConversionPackingRegistry &packing_;
    std::atomic<TensorId> next_id_{2000000};
};

} // namespace hera
