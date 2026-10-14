#include <seal/seal.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <vector>

int main()
{
    try {
#ifdef HERA_SEAL_ROTATION_HOISTING
        seal::EncryptionParameters parameters(seal::scheme_type::ckks);
        parameters.set_poly_modulus_degree(4096);
        parameters.set_coeff_modulus(seal::CoeffModulus::Create(4096,
            {60, 35, 60, 60, 60, 60, 60, 60}));
        seal::SEALContext context(parameters, true, seal::sec_level_type::none);
        seal::KeyGenerator generator(context);
        const std::vector<int> steps{1, -1, 15, -32, 1024};
        seal::GaloisKeys keys;
        generator.create_galois_keys(steps, keys);
        seal::CKKSEncoder encoder(context);
        seal::Encryptor encryptor(context, generator.secret_key());
        seal::Decryptor decryptor(context, generator.secret_key());
        seal::Evaluator evaluator(context);
        std::vector<std::complex<double>> values(encoder.slot_count());
        for (std::size_t i = 0; i < values.size(); ++i)
            values[i] = {0.2 * std::sin(i * .13), 0.1 * std::cos(i * .07)};
        seal::Plaintext plain;
        encoder.encode(values, std::ldexp(1., 44), plain);
        seal::Ciphertext input;
        encryptor.encrypt_symmetric(plain, input);
        auto data = context.first_context_data();
        for (std::size_t level = 0; data; data = data->next_context_data(), ++level) {
            if (level != 0 && level != 2 && level != 4 && level != 6) continue;
            seal::Ciphertext source;
            evaluator.mod_switch_to(input, data->parms_id(), source);
            seal::Evaluator::HoistedRotation prepared;
            evaluator.prepare_rotation(source, prepared);
            for (int step : steps) {
                seal::Ciphertext ordinary, hoisted;
                evaluator.rotate_vector(source, step, keys, ordinary);
                evaluator.rotate_vector_hoisted(prepared, step, keys, hoisted);
                std::vector<std::complex<double>> a, b;
                decryptor.decrypt(ordinary, plain); encoder.decode(plain, a);
                decryptor.decrypt(hoisted, plain); encoder.decode(plain, b);
                double error = 0;
                for (std::size_t i = 0; i < a.size(); ++i) error = std::max(error, std::abs(a[i] - b[i]));
                if (error > 1e-7 || ordinary.parms_id() != hoisted.parms_id() ||
                    ordinary.scale() != hoisted.scale() || ordinary.size() != hoisted.size())
                    throw std::runtime_error("hoisted rotation differs in values, level, scale or size");
            }
        }
        bool rejected = false;
        try {
            seal::Evaluator::HoistedRotation empty;
            seal::Ciphertext output;
            evaluator.rotate_vector_hoisted(empty, 1, keys, output);
        } catch (const std::invalid_argument &) { rejected = true; }
        if (!rejected) throw std::runtime_error("empty hoisting input was accepted");
        std::cout << "HERA_ROTATION_HOISTING_STATUS=PASS\n";
#else
        std::cout << "HERA_ROTATION_HOISTING_STATUS=SKIP_UNPATCHED_SEAL\n";
#endif
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "HERA_ROTATION_HOISTING_STATUS=FAIL: " << error.what() << '\n';
        return 1;
    }
}
