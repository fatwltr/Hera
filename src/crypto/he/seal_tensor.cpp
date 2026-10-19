#include "hera/block2_he.h"
#include "hera/detail/packing_utils.h"
#include "hera/detail/parallel_ranges.h"

#include <limits>
#include <cmath>
#include <cstring>
#include <sstream>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <tuple>

namespace hera {
namespace { std::atomic<bool> projection_hoisting{true}; std::atomic<std::size_t> projection_baby_steps{0}; }
void ConfigureProjectionRotationHoisting(bool enabled) noexcept { projection_hoisting.store(enabled); }
bool ProjectionRotationHoistingEnabled() noexcept { return projection_hoisting.load(); }
void ConfigureProjectionBabySteps(std::size_t steps) noexcept { projection_baby_steps.store(steps); }
double ProjectionBsgsCost(std::size_t bases, std::size_t outputs, std::size_t columns,
    std::size_t baby, std::size_t slots, bool phase, bool extracted_bases)
{
    const auto override = projection_baby_steps.load();
    if (override && baby != override) return std::numeric_limits<double>::infinity();
    const auto babies = bases * (baby - 1), giants = outputs * (columns / baby - 1);
#ifndef HERA_SEAL_ROTATION_HOISTING
    return static_cast<double>(babies + giants);
#else
    if (!ProjectionRotationHoistingEnabled()) return static_cast<double>(babies + giants);
    // Use a six-prime working-point estimate; exact hoisting memory is checked
    // online against the active chain. Hoisted baby rotations
    // avoid repeated conversion/NTT work; giant sums cannot share those digits.
    constexpr std::size_t q = 6;
    const auto ct_bytes = 2 * slots * q * 16;
    const auto hoist_bytes = ct_bytes + (q + 1) * q * 2 * slots * 8;
    const auto cache_bytes = babies * ct_bytes + (baby > 2 ? bases * hoist_bytes : 0) +
        (extracted_bases ? bases * ct_bytes : 0);
    const auto budget = detail::HeWorkerPool::Instance().MemoryBudget();
    const double penalty = cache_bytes > budget ? 1e12 + static_cast<double>(cache_bytes - budget) : 0.;
    return babies * (baby > 2 ? .55 : 1.) + giants +
        (phase ? outputs * (columns / baby) * .04 : 0.) + penalty;
#endif
}

namespace {

thread_local SealPlainParameterStore *g_model_plaintext_store = nullptr;
thread_local std::string g_model_plaintext_namespace;

// Use the same numerical policy for LN1/QKV and LN2/Wu. Small gains
// or large beta/gamma stay in the tail instead of amplifying PT error.
bool CanFuseLayerNormGain(double gain, double bias)
{
    return std::abs(gain) >= std::ldexp(1.0, -4) &&
           std::isfinite(bias / gain) &&
           std::abs(bias / gain) <= std::ldexp(1.0, 4);
}

template <typename Value>
std::string PlaintextKey(const std::vector<Value> &values,
                         const seal::parms_id_type &parms_id,
                         double scale, std::string_view tag)
{
    std::uint64_t hash = 1469598103934665603ULL;
    const auto mix = [&](const void *data, std::size_t bytes) {
        const auto *cursor = static_cast<const unsigned char *>(data);
        for (std::size_t i = 0; i < bytes; ++i) {
            hash ^= cursor[i];
            hash *= 1099511628211ULL;
        }
    };
    mix(tag.data(), tag.size());
    mix(parms_id.data(), parms_id.size() * sizeof(parms_id[0]));
    mix(&scale, sizeof(scale));
    if (!values.empty()) mix(values.data(), values.size() * sizeof(Value));
    std::ostringstream stream;
    stream << tag << ':' << std::hex << hash;
    return stream.str();
}

bool EndsWith(std::string_view value, std::string_view suffix)
{
    return value.size() >= suffix.size() &&
           value.substr(value.size() - suffix.size()) == suffix;
}

bool IsVirtualizableWeight(std::string_view tag)
{
    return EndsWith(tag, "block2.projection.diagonal") ||
           EndsWith(tag, "attention.wo.diagonal") ||
           EndsWith(tag, "attention.wo.bsgs.diagonal") ||
           EndsWith(tag, "ffn.projection.diagonal");
}

template <typename Value>
std::string VirtualPlaintextKey(const std::vector<Value> &values,
                                const seal::parms_id_type &parms_id,
                                double scale, std::size_t template_count)
{
    std::uint64_t content_hash = 1469598103934665603ULL;
    const auto mix_content = [&](const void *data, std::size_t bytes) {
        const auto *cursor = static_cast<const unsigned char *>(data);
        for (std::size_t i = 0; i < bytes; ++i) {
            content_hash ^= cursor[i];
            content_hash *= 1099511628211ULL;
        }
    };
    if (!values.empty())
        mix_content(values.data(), values.size() * sizeof(Value));

    std::uint64_t compatibility = 1469598103934665603ULL;
    const auto mix_compatibility = [&](const void *data, std::size_t bytes) {
        const auto *cursor = static_cast<const unsigned char *>(data);
        for (std::size_t i = 0; i < bytes; ++i) {
            compatibility ^= cursor[i];
            compatibility *= 1099511628211ULL;
        }
    };
    mix_compatibility(parms_id.data(),
                      parms_id.size() * sizeof(parms_id[0]));
    mix_compatibility(&scale, sizeof(scale));
    const auto value_count = values.size();
    mix_compatibility(&value_count, sizeof(value_count));
    const unsigned char value_kind =
        std::is_same<Value, double>::value ? 0 : 1;
    mix_compatibility(&value_kind, sizeof(value_kind));

    std::ostringstream stream;
    stream << "virtual-projection:" << std::hex << compatibility << ':'
           << (content_hash % template_count);
    return stream.str();
}

Result<std::size_t> CheckedProduct(std::size_t left, std::size_t right,
                                   const char *name)
{
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
        return Status::Invalid(std::string(name) + " overflows size_t");
    return left * right;
}

std::size_t Flat3(std::size_t a, std::size_t b, std::size_t c,
                  std::size_t b_size, std::size_t c_size)
{
    return (a * b_size + b) * c_size + c;
}

struct Tile {
    std::size_t head_begin{};
    std::size_t head_count{};
    std::size_t lanes{};
    std::size_t groups{};
    std::size_t ciphertext_begin{};
};

Result<std::vector<Tile>> MakeTiles(std::size_t tokens,
                                    std::size_t query_heads,
                                    std::size_t width,
                                    std::size_t slots,
                                    ProjectionPackingTarget target =
                                        ProjectionPackingTarget::kQk)
{
    if (tokens == 0 || query_heads == 0 || width == 0 || slots == 0)
        return Status::Invalid("multihead packing has a zero dimension");
    if (tokens > slots)
        return Status::Invalid(
            "one token column does not fit in a CKKS ciphertext");
    const std::size_t max_heads = target == ProjectionPackingTarget::kQk
        ? detail::QkHeadsPerTile(tokens, query_heads, slots)
        : slots / detail::TokenColumnStride(tokens);
    if (max_heads == 0)
        return Status::Invalid("no attention head fits in one ciphertext");
    std::vector<Tile> result;
    std::size_t ct_begin = 0;
    for (std::size_t begin = 0; begin < query_heads;) {
        const auto count = target == ProjectionPackingTarget::kQk
            ? detail::QkHeadsPerTile(tokens, query_heads - begin, slots)
            : std::min(max_heads, query_heads - begin);
        const auto block = detail::TokenColumnStride(tokens) * count;
        const auto capacity = slots / block;
        const auto lanes = target == ProjectionPackingTarget::kAttentionValue
            ? detail::FloorPowerOfTwo(capacity) : capacity;
        if (lanes == 0)
            return Status::Invalid("multihead tile has no reduction lane");
        const auto groups = (width + 2 * lanes - 1) / (2 * lanes);
        result.push_back({begin, count, lanes, groups, ct_begin});
        ct_begin += groups;
        begin += count;
    }
    return result;
}

} // namespace

std::pair<SealPlainParameterStore *, std::string>
ModelPlaintextCacheScope::Current()
{
    return {g_model_plaintext_store, g_model_plaintext_namespace};
}

ModelPlaintextCacheScope::ModelPlaintextCacheScope(
    SealPlainParameterStore *store, std::string key_namespace)
    : previous_(g_model_plaintext_store),
      previous_namespace_(g_model_plaintext_namespace)
{
    g_model_plaintext_store = store;
    g_model_plaintext_namespace = std::move(key_namespace);
}

ModelPlaintextCacheScope::~ModelPlaintextCacheScope()
{
    g_model_plaintext_store = previous_;
    g_model_plaintext_namespace = std::move(previous_namespace_);
}

bool SamePreparedPacking(const HePackingPlan &left,
                         const HePackingPlan &right)
{
    if (left.layout != right.layout ||
        left.logical_shape != right.logical_shape ||
        left.slot_count != right.slot_count ||
        left.ciphertext_count != right.ciphertext_count ||
        left.logical_locations.size() != right.logical_locations.size())
        return false;
    for (std::size_t i = 0; i < left.logical_locations.size(); ++i) {
        const auto &a = left.logical_locations[i];
        const auto &b = right.logical_locations[i];
        if (a.size() != b.size()) return false;
        for (std::size_t j = 0; j < a.size(); ++j)
            if (a[j].ciphertext != b[j].ciphertext ||
                a[j].slot != b[j].slot ||
                a[j].component != b[j].component ||
                a[j].packed_query_head != b[j].packed_query_head)
                return false;
    }
    return true;
}

Result<void> ValidatePreparedProjection(
    const PreparedProjectionPlan &plan, const SealCipherTensor &input,
    const HeContext &context)
{
    if (plan.real_input && plan.raw_complex_input)
        return Status::FailedPrecondition("prepared projection has conflicting input modes");
    if (plan.fixed_phase &&
        (!plan.raw_complex_input || !plan.realify_output ||
         plan.fixed_phase->parms_id() != plan.input_parms_id ||
         plan.fixed_phase->scale() != 1.0))
        return Status::FailedPrecondition("prepared projection fixed phase is invalid");
    if (plan.realify_output &&
        (!plan.bias_after_rescale || plan.output_packing.layout != Layout::kStrictColumn))
        return Status::FailedPrecondition("prepared real output needs post-rescale bias and real columns");
    if (plan.profile != context.profile ||
        input.parameter_profile != context.profile ||
        !SamePreparedPacking(plan.input_packing, input.physical_packing) ||
        input.ciphertexts.size() != plan.input_packing.ciphertext_count)
        return Status::FailedPrecondition(
            "prepared projection profile or input packing mismatch");
    if (plan.real_input)
        for (const auto &locations : input.physical_packing.logical_locations)
            if (locations.size() != 1 ||
                locations.front().component != ComplexLane::kReal)
                return Status::FailedPrecondition(
                    "prepared real projection has non-real input packing");
    for (const auto &ciphertext : input.ciphertexts)
        if (ciphertext.parms_id() != plan.input_parms_id ||
            !std::isfinite(ciphertext.scale()) ||
            std::abs(ciphertext.scale() / plan.input_scale - 1.0) > 1e-6)
            return Status::FailedPrecondition(
                "prepared projection input level or scale mismatch");
    return {};
}

Result<std::shared_ptr<const seal::Plaintext>> CompileModelPlaintext(
    seal::CKKSEncoder &encoder,
    const std::vector<std::complex<double>> &values,
    const seal::parms_id_type &parms_id, double scale,
    std::string_view kernel_tag)
{
    if (g_model_plaintext_store) {
        const auto tag = g_model_plaintext_namespace.empty()
            ? std::string(kernel_tag)
            : g_model_plaintext_namespace + "/" + std::string(kernel_tag);
        return g_model_plaintext_store->PrepareModelPlaintext(
            encoder, values, parms_id, scale, tag);
    }
    seal::Plaintext plain;
    encoder.encode(values, parms_id, scale, plain);
    return std::make_shared<const seal::Plaintext>(std::move(plain));
}

Result<std::vector<seal::Ciphertext>> EvaluatePreparedProjection(
    const PreparedProjectionPlan &plan, const SealCipherTensor &input,
    const HeContext &context, PreparedProjectionCounts *counts,
    PreparedProjectionRotationCache *rotation_cache, std::size_t he_workers)
{
    auto valid = ValidatePreparedProjection(plan, input, context);
    if (!valid.ok()) return valid.status();
    if (plan.bias_plaintexts.size() != plan.output_packing.ciphertext_count)
        return Status::Internal("prepared projection bias group count mismatch");
    try {
        seal::Evaluator evaluator(*context.context);
        PreparedProjectionRotationCache local_cache;
        auto &cache = rotation_cache ? *rotation_cache : local_cache;
        PreparedProjectionCounts setup_counts;
        // Build immutable bases before bounded parallel baby preparation.
        // Real bases and zero-offset babies alias the immutable input batch.
        auto base_for = [&](const PreparedProjectionTerm &term) -> const seal::Ciphertext & {
            if (plan.real_input || plan.raw_complex_input)
                return input.ciphertexts.at(term.source_ct);
            const auto key = std::make_pair(term.source_ct, term.source_component);
            auto found = cache.bases.find(key);
            if (found == cache.bases.end()) {
                seal::Ciphertext conjugated, extracted, imaginary;
                const auto &source = input.ciphertexts.at(term.source_ct);
                evaluator.complex_conjugate(source, context.galois_keys, conjugated);
                evaluator.add(source, conjugated, extracted);
                cache.bases.emplace(std::make_pair(term.source_ct, ComplexLane::kReal),
                                    std::move(extracted));
                evaluator.sub(source, conjugated, imaginary);
                cache.bases.emplace(std::make_pair(term.source_ct, ComplexLane::kImaginary),
                                    std::move(imaginary));
                found = cache.bases.find(key);
                ++setup_counts.conjugations;
            }
            return found->second;
        };
        using BabyKey = std::tuple<std::size_t, ComplexLane, int>;
        std::map<BabyKey, const seal::Ciphertext *> missing;
        std::vector<std::vector<const PreparedProjectionGroup *>> groups(
            plan.output_packing.ciphertext_count);
        for (const auto &group : plan.groups) {
            if (group.target_ct >= groups.size() ||
                (group.terms.empty() && group.phase_terms.empty()) ||
                (!group.phase_terms.empty() && !plan.fixed_phase))
                return Status::Internal("prepared projection group is invalid");
            groups[group.target_ct].push_back(&group);
            for (const auto *terms : {&group.terms, &group.phase_terms})
                for (const auto &term : *terms) {
                    if (!term.plaintext || term.source_ct >= input.ciphertexts.size() ||
                        ((plan.real_input || plan.raw_complex_input) &&
                         term.source_component != ComplexLane::kReal))
                        return Status::Internal("prepared projection term is invalid");
                    const auto &base = base_for(term);
                    if (!plan.reuse_baby_rotations || term.baby_rotation == 0) continue;
                    const auto key = std::make_tuple(term.source_ct, term.source_component,
                                                     term.baby_rotation);
                    if (cache.babies.find(key) != cache.babies.end()) continue;
                    missing.emplace(key, &base);
                }
        }
        std::vector<std::pair<BabyKey, const seal::Ciphertext *>> jobs(missing.begin(), missing.end());
#ifdef HERA_SEAL_ROTATION_HOISTING
        if (ProjectionRotationHoistingEnabled()) {
            const auto q = context.context->get_context_data(plan.input_parms_id)->parms().coeff_modulus().size();
            const auto n = context.parameters.polynomial_degree;
            const auto ct_bytes = n * q * 16;
            const auto hoist_bytes = ct_bytes + (q + 1) * q * n * 8;
            std::size_t bytes = (cache.bases.size() + cache.babies.size() + jobs.size()) * ct_bytes +
                cache.hoisted.size() * hoist_bytes;
            std::map<std::pair<std::size_t, ComplexLane>, std::size_t> uses;
            for (const auto &job : jobs) ++uses[{std::get<0>(job.first), std::get<1>(job.first)}];
            for (const auto &entry : uses) {
                if (entry.second < 2 || cache.hoisted.count(entry.first) ||
                    bytes + hoist_bytes + 4 * ct_bytes > detail::HeWorkerPool::Instance().MemoryBudget()) continue;
                const auto &base = plan.real_input || plan.raw_complex_input
                    ? input.ciphertexts[entry.first.first] : cache.bases.at(entry.first);
                auto prepared = std::make_shared<seal::Evaluator::HoistedRotation>();
                evaluator.prepare_rotation(base, *prepared);
                cache.hoisted.emplace(entry.first, std::move(prepared));
                bytes += hoist_bytes;
            }
        }
        const auto &hoisted = cache.hoisted;
#endif
        std::vector<seal::Ciphertext> babies(jobs.size());
        auto prepared_babies = detail::ParallelRanges(jobs.size(),
            he_workers, 1,
            [&](std::size_t begin, std::size_t end) -> Result<void> {
                seal::Evaluator worker(*context.context);
                for (std::size_t i = begin; i < end; ++i) {
#ifdef HERA_SEAL_ROTATION_HOISTING
                    const auto base = std::make_pair(std::get<0>(jobs[i].first), std::get<1>(jobs[i].first));
                    const auto found = hoisted.find(base);
                    if (found != hoisted.end())
                        worker.rotate_vector_hoisted(*found->second, std::get<2>(jobs[i].first),
                            context.galois_keys, babies[i]);
                    else
#endif
                        worker.rotate_vector(*jobs[i].second, std::get<2>(jobs[i].first),
                            context.galois_keys, babies[i]);
                }
                return {};
            }, detail::EstimatedHeWorkerBytes(context.parameters.polynomial_degree,
                context.parameters.coefficient_modulus_bits.size(), 3));
        if (!prepared_babies.ok()) return prepared_babies.status();
        for (std::size_t i = 0; i < jobs.size(); ++i)
            cache.babies.emplace(jobs[i].first, std::move(babies[i]));
        setup_counts.rotations += jobs.size();
        setup_counts.baby_rotations += jobs.size();
        std::vector<seal::Ciphertext> output(groups.size());
        std::vector<PreparedProjectionCounts> reports(groups.size());
        const double target = std::ldexp(1.0, plan.output_scale_bits);
        const auto &readonly_bases = cache.bases;
        const auto &readonly_babies = cache.babies;
        auto ready = detail::ParallelRanges(groups.size(),
            he_workers, 1,
            [&](std::size_t begin, std::size_t end) -> Result<void> {
                seal::Evaluator worker(*context.context);
                for (std::size_t ct = begin; ct < end; ++ct) {
                    auto &report = reports[ct];
                    bool first_group = true;
                    if (groups[ct].empty()) {
                        if (plan.bias_after_rescale) {
                            seal::Encryptor encryptor(*context.context, context.public_key);
                            const auto next = context.context->get_context_data(plan.input_parms_id)->next_context_data();
                            encryptor.encrypt_zero(next->parms_id(), output[ct]);
                            output[ct].scale() = target;
                            if (plan.bias_plaintexts[ct]) worker.add_plain_inplace(output[ct], *plan.bias_plaintexts[ct]);
                            continue;
                        }
                        // A zero matrix block still emits its bias at the
                        // same planned output level as ordinary blocks.
                        seal::Encryptor encryptor(*context.context, context.public_key);
                        encryptor.encrypt_zero(plan.input_parms_id, output[ct]);
                        const auto data = context.context->get_context_data(plan.input_parms_id);
                        output[ct].scale() = target * static_cast<double>(
                            data->parms().coeff_modulus().back().value());
                        first_group = false;
                    }
                    for (const auto *group : groups[ct]) {
                        auto sum_terms = [&](const std::vector<PreparedProjectionTerm> &terms) {
                            seal::Ciphertext sum;
                            bool first = true;
                            for (const auto &term : terms) {
                                const auto &base = (plan.real_input || plan.raw_complex_input)
                                    ? input.ciphertexts[term.source_ct]
                                    : readonly_bases.at({term.source_ct, term.source_component});
                                const seal::Ciphertext *baby = &base;
                                seal::Ciphertext temporary;
                                if (term.baby_rotation != 0) {
                                    if (plan.reuse_baby_rotations)
                                        baby = &readonly_babies.at({term.source_ct,
                                            term.source_component, term.baby_rotation});
                                    else {
                                        worker.rotate_vector(base, term.baby_rotation,
                                                             context.galois_keys, temporary);
                                        baby = &temporary;
                                        ++report.rotations;
                                        ++report.baby_rotations;
                                    }
                                }
                                seal::Ciphertext product;
                                const auto &plain = term.plaintext;
                                worker.multiply_plain(*baby, *plain, product);
                                ++report.ciphertext_plain_multiplications;
                                if (first) { sum = std::move(product); first = false; }
                                else worker.add_inplace(sum, product);
                            }
                            return sum;
                        };
                        auto sum = sum_terms(group->terms);
                        if (!group->phase_terms.empty()) {
                            auto phase_sum = sum_terms(group->phase_terms);
                            worker.multiply_plain_inplace(phase_sum, *plan.fixed_phase);
                            ++report.ciphertext_plain_multiplications;
                            ++report.phase_multiplications;
                            if (group->terms.empty()) sum = std::move(phase_sum);
                            else worker.add_inplace(sum, phase_sum);
                        }
                        if (group->giant_rotation != 0) {
                            worker.rotate_vector_inplace(sum, group->giant_rotation,
                                                         context.galois_keys);
                            ++report.rotations;
                            ++report.giant_rotations;
                        }
                        if (first_group) { output[ct] = std::move(sum); first_group = false; }
                        else worker.add_inplace(output[ct], sum);
                    }
                    if (!plan.bias_after_rescale && plan.bias_plaintexts[ct])
                        worker.add_plain_inplace(output[ct], *plan.bias_plaintexts[ct]);
                    worker.rescale_to_next_inplace(output[ct]);
                    if (!std::isfinite(output[ct].scale()) ||
                        std::abs(output[ct].scale() / target - 1.0) > plan.scale_tolerance)
                        return Status::FailedPrecondition("prepared projection output scale drift");
                    output[ct].scale() = target;
                    ++report.rescales;
                    if (plan.realify_output && !groups[ct].empty()) {
                        seal::Ciphertext conjugated;
                        worker.complex_conjugate(output[ct], context.galois_keys, conjugated);
                        worker.add_inplace(output[ct], conjugated);
                        ++report.conjugations;
                        ++report.output_conjugations;
                    }
                    if (plan.bias_after_rescale && plan.bias_plaintexts[ct])
                        worker.add_plain_inplace(output[ct], *plan.bias_plaintexts[ct]);
                }
                return {};
            }, detail::EstimatedHeWorkerBytes(context.parameters.polynomial_degree,
                context.parameters.coefficient_modulus_bits.size(), 8));
        if (!ready.ok()) return ready.status();
        if (counts) {
            counts->rotations += setup_counts.rotations;
            counts->conjugations += setup_counts.conjugations;
            counts->baby_rotations += setup_counts.baby_rotations;
            for (const auto &report : reports) {
                counts->rotations += report.rotations;
                counts->conjugations += report.conjugations;
                counts->ciphertext_plain_multiplications += report.ciphertext_plain_multiplications;
                counts->rescales += report.rescales;
                counts->baby_rotations += report.baby_rotations;
                counts->giant_rotations += report.giant_rotations;
                counts->output_conjugations += report.output_conjugations;
                counts->phase_multiplications += report.phase_multiplications;
            }
        }
        return output;
    } catch (const std::exception &error) {
        return Status::Internal(std::string("prepared projection failed: ") + error.what());
    }
}

template <typename Value>
Result<seal::Plaintext> EncodeCached(
    SealPlainParameterStore *store, seal::CKKSEncoder &encoder,
    const std::vector<Value> &values, const seal::parms_id_type &parms_id,
    double scale, std::string_view tag)
{
    if constexpr (std::is_same<Value, double>::value)
        return store->EncodeModelPlaintext(encoder, values, parms_id,
                                           scale, tag);
    else
        return store->EncodeModelPlaintext(encoder, values, parms_id,
                                           scale, tag);
}

Result<seal::Plaintext> EncodeModelPlaintext(
    seal::CKKSEncoder &encoder, const std::vector<double> &values,
    const seal::parms_id_type &parms_id, double scale,
    std::string_view kernel_tag)
{
    if (g_model_plaintext_store)
    {
        const auto tag = g_model_plaintext_namespace.empty()
            ? std::string(kernel_tag)
            : g_model_plaintext_namespace + "/" + std::string(kernel_tag);
        return EncodeCached(g_model_plaintext_store, encoder, values,
                            parms_id, scale, tag);
    }
    seal::Plaintext result;
    encoder.encode(values, parms_id, scale, result);
    return result;
}

Result<seal::Plaintext> EncodeModelPlaintext(
    seal::CKKSEncoder &encoder,
    const std::vector<std::complex<double>> &values,
    const seal::parms_id_type &parms_id, double scale,
    std::string_view kernel_tag)
{
    if (g_model_plaintext_store)
    {
        const auto tag = g_model_plaintext_namespace.empty()
            ? std::string(kernel_tag)
            : g_model_plaintext_namespace + "/" + std::string(kernel_tag);
        return EncodeCached(g_model_plaintext_store, encoder, values,
                            parms_id, scale, tag);
    }
    seal::Plaintext result;
    encoder.encode(values, parms_id, scale, result);
    return result;
}

Result<void> HePackingPlan::Validate() const
{
    const auto logical_count = logical_shape.ElementCount();
    if (!logical_count.ok()) return logical_count.status();
    if (slot_count == 0 || ciphertext_count == 0)
        return Status::Invalid("HE packing has an empty physical dimension");
    if (logical_locations.size() != logical_count.value())
        return Status::Invalid("HE packing logical coordinate count mismatch");
    const auto limit = std::numeric_limits<std::size_t>::max();
    if (ciphertext_count > limit / slot_count / 2)
        return Status::Invalid("HE packing physical coordinate count overflows");
    std::vector<unsigned char> occupied(ciphertext_count * slot_count * 2, 0);
    for (const auto &replicas : logical_locations) {
        if (replicas.empty())
            return Status::Invalid("HE packing omits a logical coordinate");
        for (const auto &location : replicas) {
            if (location.ciphertext >= ciphertext_count ||
                location.slot >= slot_count)
                return Status::Invalid("HE packing location is out of range");
            const auto index = 2 * (location.ciphertext * slot_count + location.slot) +
                (location.component == ComplexLane::kImaginary ? 1 : 0);
            if (occupied[index]++)
                return Status::Invalid(
                    "HE packing maps two logical coordinates to one lane");
        }
    }
    return {};
}

const std::vector<PhysicalSlotLocation> &HePackingPlan::Locations(
    std::size_t logical_index) const
{
    if (logical_index >= logical_locations.size())
        throw std::out_of_range("logical packing index is out of range");
    return logical_locations[logical_index];
}

Result<HePackingPlan> MultiHeadPackingPlan::DenseSpatialReal(
    const TensorShape &shape, std::size_t slot_count)
{
    auto count = shape.ElementCount();
    if (!count.ok()) return count.status();
    if (slot_count == 0)
        return Status::Invalid("real spatial packing has zero CKKS slots");
    HePackingPlan result;
    result.layout = Layout::kSpatial;
    result.logical_shape = shape;
    result.slot_count = slot_count;
    result.ciphertext_count = count.value() / slot_count +
                              (count.value() % slot_count != 0);
    result.logical_locations.resize(count.value());
    for (std::size_t logical = 0; logical < count.value(); ++logical)
        result.logical_locations[logical].push_back(
            {logical / slot_count, logical % slot_count,
             ComplexLane::kReal, std::nullopt});
    auto valid = result.Validate();
    if (!valid.ok()) return valid.status();
    return result;
}

Result<HePackingPlan> MultiHeadPackingPlan::TokenInverseBroadcast(
    std::size_t tokens, std::size_t slot_count)
{
    const auto stride = detail::TokenColumnStride(tokens);
    if (tokens == 0 || stride > slot_count || slot_count % stride != 0)
        return Status::Invalid("inverse broadcast token stride does not fit slots");
    HePackingPlan plan;
    plan.layout = Layout::kReduction;
    plan.logical_shape = {{tokens, 1}};
    plan.slot_count = slot_count;
    plan.ciphertext_count = 1;
    plan.logical_locations.resize(tokens);
    for (std::size_t token = 0; token < tokens; ++token)
        for (std::size_t column = 0; column < slot_count / stride; ++column)
            plan.logical_locations[token].push_back(
                {0, column * stride + token, ComplexLane::kReal, std::nullopt});
    return plan;
}

Result<HePackingPlan> MultiHeadPackingPlan::RealStrictColumns(
    const TensorShape &shape, std::size_t slot_count)
{
    if (shape.dimensions.size() != 2)
        return Status::Invalid("real strict-column shape must be [tokens,width]");
    const auto tokens = shape.dimensions[0];
    const auto width = shape.dimensions[1];
    auto count = shape.ElementCount();
    if (!count.ok()) return count.status();
    if (tokens == 0 || width == 0 || tokens > slot_count)
        return Status::Invalid("real strict-column dimensions do not fit CKKS slots");
    std::size_t stride = 1;
    while (stride < tokens) {
        if (stride > slot_count / 2)
            return Status::Invalid("padded token column exceeds CKKS slots");
        stride *= 2;
    }
    const auto columns = slot_count / stride;
    HePackingPlan result;
    result.layout = Layout::kStrictColumn;
    result.logical_shape = shape;
    result.slot_count = slot_count;
    result.ciphertext_count = width / columns + (width % columns != 0);
    result.logical_locations.resize(count.value());
    for (std::size_t token = 0; token < tokens; ++token)
        for (std::size_t feature = 0; feature < width; ++feature)
            result.logical_locations[token * width + feature].push_back(
                {feature / columns, (feature % columns) * stride + token,
                 ComplexLane::kReal, std::nullopt});
    auto valid = result.Validate();
    if (!valid.ok()) return valid.status();
    return result;
}

Result<ProjectionPackingDecision> MultiHeadPackingPlan::ProjectionDecision(
    const TransformerLayerConfig &layer, Layout layout, std::size_t slots)
{
    const auto valid = ValidateLayerConfig(layer);
    if (!valid.ok()) return valid.status();
    if (detail::TokenColumnStride(layer.tokens) > slots)
        return Status::Invalid("one token column does not fit in CKKS slots");
    // Q/K feed QK's regular column blocks. V feeds A*V, which handles tails.
    ProjectionPackingTarget target;
    if (layout == Layout::kMultiHeadQ || layout == Layout::kMultiHeadK)
        target = ProjectionPackingTarget::kQk;
    else if (layout == Layout::kMultiHeadV)
        target = ProjectionPackingTarget::kAttentionValue;
    else return Status::Invalid("projection decision requires Q, K, or V layout");
    auto tiles = MakeTiles(layer.tokens, layer.q_heads, 1, slots, target);
    if (!tiles.ok()) return tiles.status();
    ProjectionPackingDecision result;
    result.target = target;
    result.heads_per_tile = tiles.value().front().head_count;
    for (const auto &tile : tiles.value())
        result.head_groups.push_back(tile.head_count);
    return result;
}

Result<HePackingPlan> MultiHeadPackingPlan::ProjectionOutput(
    const TransformerLayerConfig &layer, Layout layout,
    std::size_t slot_count)
{
    const auto layer_valid = ValidateLayerConfig(layer);
    if (!layer_valid.ok()) return layer_valid.status();
    std::size_t logical_heads = 0;
    std::size_t width = 0;
    if (layout == Layout::kMultiHeadQ) {
        logical_heads = layer.q_heads;
        width = layer.qk_head_width;
    } else if (layout == Layout::kMultiHeadK) {
        logical_heads = layer.kv_heads;
        width = layer.qk_head_width;
    } else if (layout == Layout::kMultiHeadV) {
        logical_heads = layer.kv_heads;
        width = layer.value_head_width;
    } else {
        return Status::Invalid("projection output requires Q, K, or V layout");
    }
    auto decision = ProjectionDecision(layer, layout, slot_count);
    if (!decision.ok()) return decision.status();
    auto tiles = MakeTiles(layer.tokens, layer.q_heads, width, slot_count,
                           decision.value().target);
    if (!tiles.ok()) return tiles.status();
    auto count = CheckedProduct(layer.tokens, logical_heads,
                                "projection tokens*heads");
    if (!count.ok()) return count.status();
    count = CheckedProduct(count.value(), width,
                           "projection logical elements");
    if (!count.ok()) return count.status();

    HePackingPlan result;
    result.layout = layout;
    result.logical_shape = {{layer.tokens, logical_heads, width}};
    result.slot_count = slot_count;
    result.logical_locations.resize(count.value());
    const auto group_size = layer.q_heads / layer.kv_heads;
    for (const auto &tile : tiles.value()) {
        const auto block = detail::TokenColumnStride(layer.tokens) * tile.head_count;
        result.ciphertext_count = std::max(
            result.ciphertext_count, tile.ciphertext_begin + tile.groups);
        for (std::size_t local_head = 0; local_head < tile.head_count;
             ++local_head) {
            const auto query_head = tile.head_begin + local_head;
            const auto logical_head = layout == Layout::kMultiHeadQ
                                          ? query_head
                                          : query_head / group_size;
            for (std::size_t token = 0; token < layer.tokens; ++token) {
                for (std::size_t dimension = 0; dimension < width;
                     ++dimension) {
                    const auto pair_group = dimension / (2 * tile.lanes);
                    const auto in_pair = dimension % (2 * tile.lanes);
                    const auto lane = in_pair % tile.lanes;
                    const auto component = in_pair < tile.lanes
                        ? ComplexLane::kReal : ComplexLane::kImaginary;
                    const auto slot = lane * block +
                                      local_head * detail::TokenColumnStride(layer.tokens) + token;
                    const auto logical = Flat3(token, logical_head, dimension,
                                               logical_heads, width);
                    result.logical_locations[logical].push_back(
                        {tile.ciphertext_begin + pair_group, slot, component,
                         query_head});
                }
            }
        }
    }
    auto valid = result.Validate();
    if (!valid.ok()) return valid.status();
    return result;
}

Result<HePackingPlan> MultiHeadPackingPlan::ProjectionInput(
    const TransformerLayerConfig &layer, std::size_t slot_count)
{
    const auto layer_valid = ValidateLayerConfig(layer);
    if (!layer_valid.ok()) return layer_valid.status();
    auto tiles = MakeTiles(layer.tokens, layer.q_heads, layer.model_width,
                           slot_count);
    if (!tiles.ok()) return tiles.status();
    const auto count = layer.tokens * layer.model_width;
    HePackingPlan result;
    result.layout = Layout::kSpatial;
    result.logical_shape = {{layer.tokens, layer.model_width}};
    result.slot_count = slot_count;
    result.logical_locations.resize(count);
    for (const auto &tile : tiles.value()) {
        const auto block = detail::TokenColumnStride(layer.tokens) * tile.head_count;
        result.ciphertext_count = std::max(
            result.ciphertext_count, tile.ciphertext_begin + tile.groups);
        for (std::size_t local_head = 0; local_head < tile.head_count;
             ++local_head) {
            const auto query_head = tile.head_begin + local_head;
            for (std::size_t token = 0; token < layer.tokens; ++token) {
                for (std::size_t feature = 0; feature < layer.model_width;
                     ++feature) {
                    const auto pair_group = feature / (2 * tile.lanes);
                    const auto in_pair = feature % (2 * tile.lanes);
                    const auto lane = in_pair % tile.lanes;
                    const auto component = in_pair < tile.lanes
                        ? ComplexLane::kReal : ComplexLane::kImaginary;
                    const auto slot = lane * block +
                                      local_head * detail::TokenColumnStride(layer.tokens) + token;
                    result.logical_locations[
                        token * layer.model_width + feature].push_back(
                            {tile.ciphertext_begin + pair_group, slot,
                             component, query_head});
                }
            }
        }
    }
    auto valid = result.Validate();
    if (!valid.ok()) return valid.status();
    return result;
}

Result<HePackingPlan> MultiHeadPackingPlan::AttentionDiagonal(
    const TransformerLayerConfig &layer, std::size_t slot_count)
{
    const auto valid_layer = ValidateLayerConfig(layer);
    if (!valid_layer.ok()) return valid_layer.status();
    auto tiles = MakeTiles(layer.tokens, layer.q_heads, layer.tokens,
                           slot_count);
    if (!tiles.ok()) return tiles.status();
    const std::size_t count = layer.q_heads * layer.tokens * layer.tokens;
    HePackingPlan result;
    result.layout = Layout::kAttentionDiagonal;
    result.logical_shape = {{layer.q_heads, layer.tokens, layer.tokens}};
    result.slot_count = slot_count;
    result.logical_locations.resize(count);
    for (const auto &tile : tiles.value()) {
        const auto block = detail::TokenColumnStride(layer.tokens) * tile.head_count;
        result.ciphertext_count = std::max(
            result.ciphertext_count, tile.ciphertext_begin + tile.groups);
        for (std::size_t local_head = 0; local_head < tile.head_count;
             ++local_head) {
            const auto head = tile.head_begin + local_head;
            for (std::size_t row = 0; row < layer.tokens; ++row) {
                for (std::size_t column = 0; column < layer.tokens; ++column) {
                    const auto diagonal =
                        (column + layer.tokens - row) % layer.tokens;
                    const auto pair_group = diagonal / (2 * tile.lanes);
                    const auto in_pair = diagonal % (2 * tile.lanes);
                    const auto lane = in_pair % tile.lanes;
                    const auto component = in_pair < tile.lanes
                        ? ComplexLane::kReal : ComplexLane::kImaginary;
                    const auto slot = lane * block +
                                      local_head * detail::TokenColumnStride(layer.tokens) + row;
                    const auto logical = Flat3(head, row, column,
                                               layer.tokens, layer.tokens);
                    result.logical_locations[logical].push_back(
                        {tile.ciphertext_begin + pair_group, slot, component,
                         head});
                }
            }
        }
    }
    auto valid = result.Validate();
    if (!valid.ok()) return valid.status();
    return result;
}

Result<QkvColumnBsgsPlan> ComplexDiagonalProjectionPlanner::ColumnBsgs(
    const TransformerLayerConfig &layer, std::size_t slots)
{
    if (layer.tokens == 0 || layer.tokens > slots || layer.model_width == 0)
        return Status::Invalid("column BSGS has invalid input dimensions");
    QkvColumnBsgsPlan result;
    result.token_stride = detail::TokenColumnStride(layer.tokens);
    result.columns_per_ciphertext = slots / result.token_stride;
    if (slots % result.token_stride != 0)
        return Status::Invalid("column BSGS requires complete token columns");
    result.input_ciphertexts = (layer.model_width + result.columns_per_ciphertext - 1) /
                              result.columns_per_ciphertext;
    for (auto layout : {Layout::kMultiHeadQ, Layout::kMultiHeadK, Layout::kMultiHeadV}) {
        const auto width = layout == Layout::kMultiHeadV ? layer.value_head_width : layer.qk_head_width;
        auto decision = MultiHeadPackingPlan::ProjectionDecision(layer, layout, slots);
        if (!decision.ok()) return decision.status();
        auto tiles = MakeTiles(layer.tokens, layer.q_heads, width, slots, decision.value().target);
        if (!tiles.ok()) return tiles.status();
        for (const auto &tile : tiles.value()) result.output_ciphertexts += tile.groups;
    }
    result.rotation_upper_bound = std::numeric_limits<std::size_t>::max();
    double best_score = std::numeric_limits<double>::infinity();
    for (std::size_t baby = 1; baby <= result.columns_per_ciphertext; ++baby) {
        if (result.columns_per_ciphertext % baby != 0) continue;
        const auto giant = result.columns_per_ciphertext / baby;
        const auto cost = result.input_ciphertexts * (baby - 1) +
                          result.output_ciphertexts * (giant - 1);
        const auto score = ProjectionBsgsCost(result.input_ciphertexts, result.output_ciphertexts,
            result.columns_per_ciphertext, baby, slots);
        if (score < best_score) {
            best_score = score;
            result.baby_steps = baby;
            result.giant_steps = giant;
            result.rotation_upper_bound = cost;
        }
    }
    if (result.baby_steps == 0) return Status::Invalid("projection baby override must divide the slot columns");
    return result;
}

Result<ComplexDiagonalProjectionPlan>
ComplexDiagonalProjectionPlanner::Build(
    const TransformerLayerConfig &layer, Layout output_layout,
    std::size_t slot_count)
{
    auto packing = MultiHeadPackingPlan::ProjectionOutput(
        layer, output_layout, slot_count);
    if (!packing.ok()) return packing.status();
    std::size_t output_columns = 0;
    if (output_layout == Layout::kMultiHeadQ)
        output_columns = QueryWidth(layer);
    else if (output_layout == Layout::kMultiHeadK)
        output_columns = KeyWidth(layer);
    else if (output_layout == Layout::kMultiHeadV)
        output_columns = ValueWidth(layer);
    else
        return Status::Invalid(
            "complex diagonal projection requires Q, K, or V output");
    // Exact projection offsets depend on the caller's physical input
    // packing. SealBlock2Executor::RequiredRotationSteps computes them once
    // that packing is known; do not advertise a binary decomposition here.
    auto decision = MultiHeadPackingPlan::ProjectionDecision(
        layer, output_layout, slot_count);
    if (!decision.ok()) return decision.status();
    std::vector<int> rotations;
    return ComplexDiagonalProjectionPlan{
        layer.model_width, output_columns, output_layout,
        output_layout != Layout::kMultiHeadQ &&
            layer.q_heads != layer.kv_heads,
        packing.take_value(), std::move(rotations),
        decision.take_value()};
}

Result<void> SealCipherTensor::Validate(const HeContext &context) const
{
    if (conjugated_complex && logical_spec.layout != Layout::kMultiHeadK)
        return Status::Invalid("conjugated values require the native K layout");
    if (parameter_profile != context.profile)
        return Status::Invalid("ciphertext tensor/profile mismatch");
    if (logical_spec.domain != Domain::kHeCiphertext ||
        !logical_spec.he_parameter_profile ||
        *logical_spec.he_parameter_profile != parameter_profile)
        return Status::Invalid("ciphertext tensor has an invalid HE contract");
    if (physical_packing.logical_shape != logical_spec.shape ||
        physical_packing.layout != logical_spec.layout)
        return Status::Invalid("ciphertext tensor packing/contract mismatch");
    auto packing_valid = physical_packing.Validate();
    if (!packing_valid.ok()) return packing_valid.status();
    if (physical_packing.slot_count != context.slot_count() ||
        ciphertexts.size() != physical_packing.ciphertext_count)
        return Status::Invalid("ciphertext tensor physical size mismatch");
    for (const auto &ciphertext : ciphertexts)
        if (!context.context->get_context_data(ciphertext.parms_id()))
            return Status::Invalid("ciphertext belongs to another context");
    return {};
}

Result<void> SealCipherTensorStore::Put(TensorId id, SealCipherTensor tensor)
{
    if (id == 0) return Status::Invalid("ciphertext tensor ID is zero");
    std::lock_guard<std::mutex> lock(mutex_);
    if (!tensors_.emplace(id, std::make_shared<const SealCipherTensor>(std::move(tensor))).second)
        return Status::FailedPrecondition(
            "ciphertext tensor ID is already registered");
    return {};
}

Result<SealCipherTensor> SealCipherTensorStore::Get(TensorId id) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = tensors_.find(id);
    if (found == tensors_.end())
        return Status::NotFound("ciphertext tensor is not registered");
    return *found->second;
}

Result<std::shared_ptr<const SealCipherTensor>> SealCipherTensorStore::GetShared(TensorId id) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = tensors_.find(id);
    if (found == tensors_.end()) return Status::NotFound("ciphertext tensor is not registered");
    return found->second;
}

Result<void> SealCipherTensorStore::Rename(TensorId source, TensorId destination)
{
    if (destination == 0) return Status::Invalid("ciphertext tensor ID is zero");
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = tensors_.find(source);
    if (found == tensors_.end()) return Status::NotFound("ciphertext tensor is not registered");
    if (source == destination) return {};
    if (tensors_.find(destination) != tensors_.end())
        return Status::FailedPrecondition("ciphertext tensor ID is already registered");
    auto node = tensors_.extract(found);
    node.key() = destination;
    tensors_.insert(std::move(node));
    return {};
}

Result<TensorHandle> SealCipherTensorStore::Allocate(SealCipherTensor tensor)
{
    const auto id = next_id_.fetch_add(1);
    const auto spec = tensor.logical_spec;
    auto stored = Put(id, std::move(tensor));
    if (!stored.ok()) return stored.status();
    return TensorHandle{id, spec};
}

void SealCipherTensorStore::Erase(TensorId id) noexcept
{
    std::lock_guard<std::mutex> lock(mutex_);
    tensors_.erase(id);
}

std::size_t SealCipherTensorStore::Size() const noexcept
{
    std::lock_guard<std::mutex> lock(mutex_);
    return tensors_.size();
}

Result<void> SealPlainParameterStore::PutMatrix(
    const WeightHandle &handle, std::vector<double> row_major)
{
    if (!handle.valid()) return Status::Invalid("matrix handle ID is zero");
    if (handle.spec.rows != 0 &&
        handle.spec.columns > std::numeric_limits<std::size_t>::max() /
                                  handle.spec.rows)
        return Status::Invalid("matrix parameter size overflows size_t");
    if (row_major.size() != handle.spec.rows * handle.spec.columns)
        return Status::Invalid(handle.spec.name + ": matrix payload size mismatch");
    std::lock_guard<std::mutex> lock(mutex_);
    auto parameter = std::make_shared<const DenseMatrixParameter>(
        DenseMatrixParameter{handle.spec, std::move(row_major)});
    if (!matrices_.emplace(handle.id, std::move(parameter)).second)
        return Status::FailedPrecondition("matrix parameter ID already exists");
    return {};
}

Result<void> SealPlainParameterStore::PutVector(
    const PlainParameterHandle &handle, std::vector<double> values)
{
    if (!handle.valid()) return Status::Invalid("vector handle ID is zero");
    if (values.size() != handle.spec.length)
        return Status::Invalid(handle.spec.name + ": vector payload size mismatch");
    std::lock_guard<std::mutex> lock(mutex_);
    auto parameter = std::make_shared<const DenseVectorParameter>(
        DenseVectorParameter{handle.spec, std::move(values)});
    if (!vectors_.emplace(handle.id, std::move(parameter)).second)
        return Status::FailedPrecondition("vector parameter ID already exists");
    return {};
}

Result<DenseMatrixParameter> SealPlainParameterStore::GetMatrix(
    WeightId id) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = matrices_.find(id);
    if (found == matrices_.end())
        return Status::NotFound("matrix parameter payload is not registered");
    return *found->second;
}

Result<DenseVectorParameter> SealPlainParameterStore::GetVector(
    PlainParameterId id) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = vectors_.find(id);
    if (found == vectors_.end())
        return Status::NotFound("vector parameter payload is not registered");
    return *found->second;
}

Result<std::shared_ptr<const DenseMatrixParameter>>
SealPlainParameterStore::GetMatrixShared(WeightId id) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = matrices_.find(id);
    if (found == matrices_.end())
        return Status::NotFound("matrix parameter payload is not registered");
    return found->second;
}

Result<std::shared_ptr<const DenseVectorParameter>>
SealPlainParameterStore::GetVectorShared(PlainParameterId id) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = vectors_.find(id);
    if (found == vectors_.end())
        return Status::NotFound("vector parameter payload is not registered");
    return found->second;
}

Result<void> SealPlainParameterStore::FuseLayerNormGammaIntoQkv(
    const LayerWeightHandles &weights)
{
    const auto find_handle = [&](std::string_view name)
        -> const PlainParameterHandle * {
        for (const auto &handle : weights.plain_parameters)
            if (handle.spec.name == name) return &handle;
        return nullptr;
    };
    const auto *gamma_handle = find_handle("ln1.gamma");
    const auto *beta_handle = find_handle("ln1.beta");
    if (!gamma_handle || !beta_handle)
        return Status::Invalid("LN1 affine handles are absent");
    const WeightHandle *matrices[] = {&weights.wq, &weights.wk, &weights.wv};
    if (weights.wq.id == weights.wk.id || weights.wq.id == weights.wv.id ||
        weights.wk.id == weights.wv.id)
        return Status::Invalid("Q/K/V matrix IDs must be distinct");

    std::lock_guard<std::mutex> lock(mutex_);
    const auto gamma = vectors_.find(gamma_handle->id);
    const auto beta = vectors_.find(beta_handle->id);
    if (gamma == vectors_.end() || beta == vectors_.end())
        return Status::NotFound("LN1 affine parameters are not registered");
    std::size_t already_fused = 0;
    for (const auto *handle : matrices) {
        const auto matrix = matrices_.find(handle->id);
        if (matrix == matrices_.end())
            return Status::NotFound("Q/K/V matrix is not registered");
        const auto &original = *matrix->second;
        if (handle->spec.name != original.spec.name ||
            handle->spec.rows != original.spec.rows ||
            handle->spec.columns != original.spec.columns ||
            gamma->second->values.size() != original.spec.rows ||
            beta->second->values.size() != original.spec.rows)
            return Status::Invalid("LN1/QKV fusion parameter shape mismatch");
        const auto previous = fused_qkv_.find(handle->id);
        if (previous != fused_qkv_.end()) {
            if (previous->second.gamma != gamma_handle->id ||
                previous->second.beta != beta_handle->id)
                return Status::FailedPrecondition(
                    "Q/K/V matrix was fused with different affine parameters");
            ++already_fused;
        }
    }
    if (already_fused == 3) return {};
    if (already_fused != 0)
        return Status::FailedPrecondition("Q/K/V parameters are partially fused");
    if (preparing_plaintexts_ || plaintext_cache_frozen_ ||
        !encoded_plaintexts_.empty() || !prepared_projections_.empty())
        return Status::FailedPrecondition(
            "LN1/QKV fusion must precede plaintext preparation");

    auto tail = std::make_shared<GammaFusedLayerNormParameters>(
        GammaFusedLayerNormParameters{*gamma->second, *beta->second});
    for (std::size_t feature = 0; feature < tail->gamma.values.size(); ++feature) {
        const double gain = gamma->second->values[feature];
        if (!std::isfinite(gain) || !std::isfinite(tail->beta.values[feature]))
            return Status::Invalid("LN1 affine parameters must be finite");
        if (CanFuseLayerNormGain(gain, tail->beta.values[feature])) {
            tail->gamma.values[feature] = 1.0;
            tail->beta.values[feature] /= gain;
        }
    }
    std::map<WeightId, std::shared_ptr<const DenseMatrixParameter>> staged;
    std::map<WeightId, LayerNormFusion> staged_fusions;
    for (const auto *handle : matrices) {
        auto fused = std::make_shared<DenseMatrixParameter>(*matrices_.at(handle->id));
        for (std::size_t feature = 0; feature < fused->spec.rows; ++feature) {
            const double gain = gamma->second->values[feature];
            const double factor = CanFuseLayerNormGain(gain, beta->second->values[feature]) ? gain : 1.0;
            for (std::size_t column = 0; column < fused->spec.columns; ++column) {
                auto &value = fused->row_major[feature * fused->spec.columns + column];
                value *= factor;
                if (!std::isfinite(value)) return Status::Invalid("gamma-fused QKV weight overflow");
            }
        }
        staged.emplace(handle->id, std::move(fused));
        staged_fusions.emplace(handle->id, LayerNormFusion{
            gamma_handle->id, beta_handle->id, tail});
    }
    // Publish only after all three projections have validated successfully.
    for (auto &entry : staged)
        matrices_.at(entry.first) = std::move(entry.second);
    fused_qkv_.merge(staged_fusions);
    return {};
}

Result<void> SealPlainParameterStore::FuseLayerNormGammaIntoWu(
    const LayerWeightHandles &weights)
{
    const PlainParameterHandle *gh = nullptr, *bh = nullptr;
    for (const auto &handle : weights.plain_parameters) {
        if (handle.spec.name == "ln2.gamma") gh = &handle;
        if (handle.spec.name == "ln2.beta") bh = &handle;
    }
    if (!gh || !bh) return Status::Invalid("LN2 affine handles are absent");
    std::lock_guard<std::mutex> lock(mutex_);
    const auto previous = fused_qkv_.find(weights.wu.id);
    if (previous != fused_qkv_.end()) {
        if (previous->second.gamma != gh->id || previous->second.beta != bh->id)
            return Status::FailedPrecondition("Wu was fused with different LN2 parameters");
        return {};
    }
    const auto matrix = matrices_.find(weights.wu.id);
    const auto gamma = vectors_.find(gh->id), beta = vectors_.find(bh->id);
    if (matrix == matrices_.end() || gamma == vectors_.end() || beta == vectors_.end())
        return Status::NotFound("LN2/Wu parameters are not registered");
    const auto &original = *matrix->second;
    if (original.spec.rows != weights.wu.spec.rows ||
        original.spec.columns != weights.wu.spec.columns ||
        gamma->second->values.size() != original.spec.rows ||
        beta->second->values.size() != original.spec.rows)
        return Status::Invalid("LN2/Wu fusion shape mismatch");
    if (preparing_plaintexts_ || plaintext_cache_frozen_ ||
        !encoded_plaintexts_.empty() || !prepared_projections_.empty())
        return Status::FailedPrecondition("LN2/Wu fusion must precede plaintext preparation");
    auto fused = std::make_shared<DenseMatrixParameter>(original);
    auto tail = std::make_shared<GammaFusedLayerNormParameters>(
        GammaFusedLayerNormParameters{*gamma->second, *beta->second});
    for (std::size_t row = 0; row < original.spec.rows; ++row) {
        const double gain = gamma->second->values[row], offset = beta->second->values[row];
        if (!std::isfinite(gain) || !std::isfinite(offset))
            return Status::Invalid("LN2 affine parameters must be finite");
        // Retain small gains/large offsets in the tail: beta/gamma would
        // amplify the finite-precision Wu plaintext error.
        // A zero gain retains its original row and beta, with no division.
        if (!CanFuseLayerNormGain(gain, offset))
            continue;
        tail->gamma.values[row] = 1.0;
        tail->beta.values[row] = offset / gain;
        for (std::size_t col = 0; col < original.spec.columns; ++col) {
            auto &value = fused->row_major[row * original.spec.columns + col];
            value *= gain;
            if (!std::isfinite(value)) return Status::Invalid("gamma-fused Wu weight overflow");
        }
    }
    matrices_.at(weights.wu.id) = std::move(fused);
    fused_qkv_.emplace(weights.wu.id, LayerNormFusion{gh->id, bh->id, std::move(tail)});
    return {};
}

Result<std::shared_ptr<const GammaFusedLayerNormParameters>>
SealPlainParameterStore::GetGammaFusedLayerNorm(WeightId id) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = fused_qkv_.find(id);
    if (found == fused_qkv_.end())
        return Status::FailedPrecondition(
            "LayerNorm gamma must be fused before evaluation");
    return found->second.tail;
}

Result<void> SealPlainParameterStore::ConfigureVirtualWeightPlaintexts(
    std::size_t templates_per_compatibility_group)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (preparing_plaintexts_ || plaintext_cache_frozen_ ||
        !encoded_plaintexts_.empty())
        return Status::FailedPrecondition(
            "virtual weight mode must be configured before Prepare()");
    virtual_weight_template_count_ = templates_per_compatibility_group;
    return {};
}

void SealPlainParameterStore::BeginPlaintextPreparation()
{
    std::lock_guard<std::mutex> lock(mutex_);
    staged_plaintexts_.clear();
    staged_projections_.clear();
    staged_block2_programs_.clear();
    staged_encoded_plaintext_bytes_ = 0;
    staged_plaintext_cache_hits_ = 0;
    staged_virtual_weight_requests_ = 0;
    staged_virtual_weight_cache_hits_ = 0;
    preparing_plaintexts_ = true;
}

void SealPlainParameterStore::FreezePlaintextCache()
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!preparing_plaintexts_) return;
    encoded_plaintexts_.swap(staged_plaintexts_);
    staged_plaintexts_.clear();
    prepared_projections_.swap(staged_projections_);
    prepared_block2_programs_.swap(staged_block2_programs_);
    staged_projections_.clear();
    staged_block2_programs_.clear();
    prepared_projection_hits_ = 0;
    encoded_plaintext_bytes_ = staged_encoded_plaintext_bytes_;
    plaintext_cache_hits_ = staged_plaintext_cache_hits_;
    virtual_weight_requests_ = staged_virtual_weight_requests_;
    virtual_weight_cache_hits_ = staged_virtual_weight_cache_hits_;
    preparing_plaintexts_ = false;
    plaintext_cache_frozen_ = true;
}

void SealPlainParameterStore::AbortPlaintextPreparation() noexcept
{
    std::lock_guard<std::mutex> lock(mutex_);
    staged_plaintexts_.clear();
    staged_projections_.clear();
    staged_block2_programs_.clear();
    staged_encoded_plaintext_bytes_ = 0;
    staged_plaintext_cache_hits_ = 0;
    staged_virtual_weight_requests_ = 0;
    staged_virtual_weight_cache_hits_ = 0;
    preparing_plaintexts_ = false;
}

bool SealPlainParameterStore::plaintext_cache_frozen() const noexcept
{
    std::lock_guard<std::mutex> lock(mutex_);
    return plaintext_cache_frozen_;
}

bool SealPlainParameterStore::preparing_plaintexts() const noexcept
{
    std::lock_guard<std::mutex> lock(mutex_);
    return preparing_plaintexts_;
}

std::size_t SealPlainParameterStore::encoded_plaintext_count() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return preparing_plaintexts_ ? staged_plaintexts_.size() : encoded_plaintexts_.size();
}

std::size_t SealPlainParameterStore::encoded_plaintext_bytes() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return preparing_plaintexts_ ? staged_encoded_plaintext_bytes_
                                 : encoded_plaintext_bytes_;
}

std::size_t SealPlainParameterStore::prepared_projection_count() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return preparing_plaintexts_ ? staged_projections_.size()
                                 : prepared_projections_.size();
}

std::size_t SealPlainParameterStore::prepared_projection_hits() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return prepared_projection_hits_;
}

std::size_t SealPlainParameterStore::plaintext_cache_hits() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return preparing_plaintexts_ ? staged_plaintext_cache_hits_
                                 : plaintext_cache_hits_;
}

std::size_t SealPlainParameterStore::virtual_weight_requests() const noexcept
{
    std::lock_guard<std::mutex> lock(mutex_);
    return preparing_plaintexts_ ? staged_virtual_weight_requests_
                                 : virtual_weight_requests_;
}

std::size_t SealPlainParameterStore::virtual_weight_cache_hits() const noexcept
{
    std::lock_guard<std::mutex> lock(mutex_);
    return preparing_plaintexts_ ? staged_virtual_weight_cache_hits_
                                 : virtual_weight_cache_hits_;
}

Result<seal::Plaintext> SealPlainParameterStore::EncodeModelPlaintext(
    seal::CKKSEncoder &encoder, const std::vector<double> &values,
    const seal::parms_id_type &parms_id, double scale,
    std::string_view kernel_tag)
{
    std::lock_guard<std::mutex> lock(mutex_);
    const bool virtualized = virtual_weight_template_count_ != 0 &&
                             IsVirtualizableWeight(kernel_tag);
    const auto key = virtualized
        ? VirtualPlaintextKey(values, parms_id, scale,
                              virtual_weight_template_count_)
        : PlaintextKey(values, parms_id, scale, kernel_tag);
    auto &cache = preparing_plaintexts_ ? staged_plaintexts_
                                        : encoded_plaintexts_;
    if (virtualized) {
        if (preparing_plaintexts_) ++staged_virtual_weight_requests_;
        else ++virtual_weight_requests_;
    }
    const auto found = cache.find(key);
    if (found != cache.end()) {
        if (preparing_plaintexts_) {
            ++staged_plaintext_cache_hits_;
            if (virtualized) ++staged_virtual_weight_cache_hits_;
        } else {
            ++plaintext_cache_hits_;
            if (virtualized) ++virtual_weight_cache_hits_;
        }
        return *found->second;
    }
    if (!preparing_plaintexts_)
        return Status::FailedPrecondition(plaintext_cache_frozen_
            ? std::string("online model plaintext cache miss: ") +
                std::string(kernel_tag)
            : "model plaintext encode attempted outside Prepare()");
    seal::Plaintext plain;
    encoder.encode(values, parms_id, scale, plain);
    staged_encoded_plaintext_bytes_ +=
        plain.save_size(seal::compr_mode_type::none);
    cache.emplace(key, std::make_shared<const seal::Plaintext>(plain));
    return plain;
}

Result<seal::Plaintext> SealPlainParameterStore::EncodeModelPlaintext(
    seal::CKKSEncoder &encoder,
    const std::vector<std::complex<double>> &values,
    const seal::parms_id_type &parms_id, double scale,
    std::string_view kernel_tag)
{
    std::lock_guard<std::mutex> lock(mutex_);
    const bool virtualized = virtual_weight_template_count_ != 0 &&
                             IsVirtualizableWeight(kernel_tag);
    const auto key = virtualized
        ? VirtualPlaintextKey(values, parms_id, scale,
                              virtual_weight_template_count_)
        : PlaintextKey(values, parms_id, scale, kernel_tag);
    auto &cache = preparing_plaintexts_ ? staged_plaintexts_
                                        : encoded_plaintexts_;
    if (virtualized) {
        if (preparing_plaintexts_) ++staged_virtual_weight_requests_;
        else ++virtual_weight_requests_;
    }
    const auto found = cache.find(key);
    if (found != cache.end()) {
        if (preparing_plaintexts_) {
            ++staged_plaintext_cache_hits_;
            if (virtualized) ++staged_virtual_weight_cache_hits_;
        } else {
            ++plaintext_cache_hits_;
            if (virtualized) ++virtual_weight_cache_hits_;
        }
        return *found->second;
    }
    if (!preparing_plaintexts_)
        return Status::FailedPrecondition(plaintext_cache_frozen_
            ? std::string("online model plaintext cache miss: ") +
                std::string(kernel_tag)
            : "model plaintext encode attempted outside Prepare()");
    seal::Plaintext plain;
    encoder.encode(values, parms_id, scale, plain);
    staged_encoded_plaintext_bytes_ +=
        plain.save_size(seal::compr_mode_type::none);
    cache.emplace(key, std::make_shared<const seal::Plaintext>(plain));
    return plain;
}

Result<std::shared_ptr<const seal::Plaintext>>
SealPlainParameterStore::PrepareModelPlaintext(
    seal::CKKSEncoder &encoder,
    const std::vector<std::complex<double>> &values,
    const seal::parms_id_type &parms_id, double scale,
    std::string_view kernel_tag)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!preparing_plaintexts_)
        return Status::FailedPrecondition(
            "projection plaintext compilation requires Prepare()");
    const bool virtualized = virtual_weight_template_count_ != 0 &&
                             IsVirtualizableWeight(kernel_tag);
    const auto key = virtualized
        ? VirtualPlaintextKey(values, parms_id, scale,
                              virtual_weight_template_count_)
        : PlaintextKey(values, parms_id, scale, kernel_tag);
    if (virtualized) ++staged_virtual_weight_requests_;
    const auto found = staged_plaintexts_.find(key);
    if (found != staged_plaintexts_.end()) {
        ++staged_plaintext_cache_hits_;
        if (virtualized) ++staged_virtual_weight_cache_hits_;
        return found->second;
    }
    seal::Plaintext plain;
    encoder.encode(values, parms_id, scale, plain);
    staged_encoded_plaintext_bytes_ +=
        plain.save_size(seal::compr_mode_type::none);
    auto shared = std::make_shared<const seal::Plaintext>(std::move(plain));
    staged_plaintexts_.emplace(key, shared);
    return shared;
}

Result<void> SealPlainParameterStore::PutPreparedProjection(
    std::string key, std::shared_ptr<const PreparedProjectionPlan> plan)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!preparing_plaintexts_)
        return Status::FailedPrecondition(
            "projection plan compilation requires Prepare()");
    if (!plan || !staged_projections_.emplace(std::move(key), std::move(plan)).second)
        return Status::FailedPrecondition(
            "projection plan is null or already compiled");
    return {};
}

Result<std::shared_ptr<const PreparedProjectionPlan>>
SealPlainParameterStore::GetPreparedProjection(std::string_view key) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto &plans = preparing_plaintexts_ ? staged_projections_
                                               : prepared_projections_;
    const auto found = plans.find(std::string(key));
    if (found == plans.end())
        return Status::NotFound("prepared projection plan is absent: " +
                                std::string(key));
    if (!preparing_plaintexts_) ++prepared_projection_hits_;
    return found->second;
}

Result<void> SealPlainParameterStore::PutPreparedBlock2Program(
    std::string key, std::shared_ptr<const PreparedBlock2Program> program)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!preparing_plaintexts_)
        return Status::FailedPrecondition("Block-2 program compilation requires Prepare()");
    if (!program || !staged_block2_programs_.emplace(std::move(key), std::move(program)).second)
        return Status::FailedPrecondition("Block-2 program is null or already compiled");
    return {};
}

Result<std::shared_ptr<const PreparedBlock2Program>>
SealPlainParameterStore::GetPreparedBlock2Program(std::string_view key) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto &programs = preparing_plaintexts_ ? staged_block2_programs_ : prepared_block2_programs_;
    const auto found = programs.find(std::string(key));
    if (found == programs.end())
        return Status::NotFound("prepared Block-2 program is absent: " + std::string(key));
    return found->second;
}

} // namespace hera
