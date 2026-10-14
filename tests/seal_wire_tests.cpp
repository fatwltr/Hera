#include "hera/seal_wire.h"

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void Require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}

class MemoryTransport final : public hera::IProtocolTransport {
public:
    void Send(const void *data, std::size_t size) override
    {
        Require(size <= hera::kSealWireChunkBytes, "wire send exceeded chunk bound");
        const auto *begin = static_cast<const std::uint8_t *>(data);
        data_.insert(data_.end(), begin, begin + size);
    }
    void Receive(void *data, std::size_t size) override
    {
        Require(size <= hera::kSealWireChunkBytes, "wire receive exceeded chunk bound");
        Require(size <= data_.size() - position_, "truncated wire frame");
        std::memcpy(data, data_.data() + position_, size);
        position_ += size;
    }
    void Cancel() noexcept override {}
    std::uint64_t bytes() const noexcept override { return data_.size(); }
    std::uint64_t rounds() const noexcept override { return 0; }

    const std::vector<std::uint8_t> &data() const { return data_; }
    std::size_t position() const { return position_; }

private:
    std::vector<std::uint8_t> data_;
    std::size_t position_{};
};

std::size_t CheckFrame(const MemoryTransport &wire, std::size_t begin,
                       std::size_t end, seal::compr_mode_type expected)
{
    std::size_t chunks = 0;
    bool terminated = false;
    while (begin < end) {
        Require(end - begin >= sizeof(std::uint32_t), "missing chunk length");
        std::uint32_t size;
        std::memcpy(&size, wire.data().data() + begin, sizeof(size));
        begin += sizeof(size);
        if (size == 0) {
            Require(begin == end, "trailing frame data");
            terminated = true;
            break;
        }
        Require(size <= hera::kSealWireChunkBytes && size <= end - begin,
                "invalid chunk size");
        if (chunks == 0) {
            Require(size >= sizeof(seal::Serialization::SEALHeader),
                    "missing SEAL header");
            seal::Serialization::SEALHeader header;
            std::memcpy(&header, wire.data().data() + begin, sizeof(header));
            Require(header.compr_mode == expected, "wrong compression in SEAL header");
        }
        ++chunks;
        begin += size;
    }
    Require(terminated, "missing frame terminator");
    return chunks;
}

void CheckCiphertext(const seal::Ciphertext &expected,
                     const seal::Ciphertext &actual)
{
    Require(actual.parms_id() == expected.parms_id() &&
                actual.scale() == expected.scale() &&
                actual.is_ntt_form() == expected.is_ntt_form() &&
                actual.size() == expected.size() &&
                actual.poly_modulus_degree() == expected.poly_modulus_degree() &&
                actual.coeff_modulus_size() == expected.coeff_modulus_size(),
            "ciphertext metadata changed during serialization");
    const auto words = expected.size() * expected.poly_modulus_degree() *
                       expected.coeff_modulus_size();
    Require(std::memcmp(actual.data(), expected.data(),
                        words * sizeof(std::uint64_t)) == 0,
            "ciphertext coefficients changed during serialization");
}

} // namespace

int main()
{
    try {
        seal::EncryptionParameters parameters(seal::scheme_type::ckks);
        parameters.set_poly_modulus_degree(16384);
        parameters.set_coeff_modulus(
            seal::CoeffModulus::Create(16384, {50, 40, 40, 50}));
        seal::SEALContext context(parameters);
        Require(context.parameters_set(), "invalid test parameters");
        seal::KeyGenerator keys(context);
        seal::CKKSEncoder encoder(context);
        seal::Plaintext plaintext;
        encoder.encode(std::vector<double>{0.125, -1.5, 2.0},
                       static_cast<double>(std::uint64_t{1} << 30), plaintext);
        seal::Encryptor encryptor(context, keys.secret_key());
        seal::Ciphertext input;
        encryptor.encrypt_symmetric(plaintext, input);

        // Consecutive frames exercise boundaries between compressed and raw
        // objects. Compare every coefficient: compression must be lossless.
        MemoryTransport wire;
        const auto raw_mode = seal::compr_mode_type::none;
        hera::SendSealObject(wire, input, raw_mode);
        const auto raw_end = wire.data().size();
        Require(CheckFrame(wire, 0, raw_end, raw_mode) > 1,
                "raw ciphertext did not exercise multiple chunks");
        auto restored = hera::ReceiveSealObject<seal::Ciphertext>(wire, context);
        CheckCiphertext(input, restored);
        Require(wire.position() == raw_end, "raw frame consumed incorrect bytes");

        hera::SendSealObject(wire, input);
        const auto default_end = wire.data().size();
        const auto default_mode = seal::Serialization::compr_mode_default;
        const auto default_chunks = CheckFrame(wire, raw_end, default_end, default_mode);
        restored = hera::ReceiveSealObject<seal::Ciphertext>(wire, context);
        CheckCiphertext(input, restored);
        Require(wire.position() == default_end,
                "default frame consumed incorrect bytes");
        if (default_mode != raw_mode) {
            Require(default_chunks > 1,
                    "compressed ciphertext did not exercise multiple chunks");
            Require(default_end - raw_end < raw_end,
                    "compression did not reduce ciphertext wire size");
        }

        hera::SendSealObject(wire, input, raw_mode);
        const auto final_end = wire.data().size();
        CheckFrame(wire, default_end, final_end, raw_mode);
        restored = hera::ReceiveSealObject<seal::Ciphertext>(wire, context);
        CheckCiphertext(input, restored);
        Require(wire.position() == final_end,
                "consecutive frame consumed incorrect bytes");

        bool rejected = false;
        try {
            hera::SendSealObject(wire, input,
                                 static_cast<seal::compr_mode_type>(255));
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        Require(rejected && wire.data().size() == final_end,
                "unsupported codec wrote a partial frame");

        std::cout << "SEAL_WIRE_RAW_BYTES=" << raw_end
                  << "\nSEAL_WIRE_DEFAULT_BYTES=" << default_end - raw_end
                  << "\nSEAL_WIRE_DEFAULT_MODE=" << static_cast<int>(default_mode)
                  << "\nHERA_SEAL_WIRE_STATUS=PASS\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "HERA_SEAL_WIRE_STATUS=FAIL: " << error.what() << '\n';
        return 1;
    }
}
