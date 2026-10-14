#pragma once

#include "hera/he_context_registry.h"

#include <stdexcept>

namespace hera::detail {

inline Result<std::size_t> LevelOf(const HeContext &context,
                                    seal::parms_id_type id)
{
    std::size_t level = 0;
    for (auto data = context.context->first_context_data(); data;
         data = data->next_context_data(), ++level)
        if (data->parms_id() == id) return level;
    return Status::Invalid("ciphertext parms_id is outside the HE profile");
}

inline Result<seal::parms_id_type> LevelId(const HeContext &context,
                                          std::size_t target)
{
    auto data = context.context->first_context_data();
    for (std::size_t level = 0; data && level < target; ++level)
        data = data->next_context_data();
    if (!data)
        return Status::FailedPrecondition("requested level exceeds the CKKS modulus chain");
    return data->parms_id();
}

inline double DroppedPrime(const HeContext &context, seal::parms_id_type id)
{
    const auto data = context.context->get_context_data(id);
    if (!data || !data->next_context_data())
        throw std::invalid_argument("operation needs another CKKS data level");
    return static_cast<double>(data->parms().coeff_modulus().back().value());
}

} // namespace hera::detail
