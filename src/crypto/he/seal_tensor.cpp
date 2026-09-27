#include "hera/block2_he.h"

#include <limits>
#include <cmath>
#include <cstring>
#include <sstream>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <tuple>

namespace hera {
namespace {

thread_local SealPlainParameterStore *g_model_plaintext_store = nullptr;
thread_local std::string g_model_plaintext_namespace;

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
                                    std::size_t slots)
{
    if (tokens == 0 || query_heads == 0 || width == 0 || slots == 0)
        return Status::Invalid("multihead packing has a zero dimension");
    if (tokens > slots)
        return Status::Invalid(
            "one token column does not fit in a CKKS ciphertext");
    const std::size_t max_heads = slots / tokens;
    if (max_heads == 0)
        return Status::Invalid("no attention head fits in one ciphertext");
    std::vector<Tile> result;
    std::size_t ct_begin = 0;
    for (std::size_t begin = 0; begin < query_heads;) {
        const auto count = std::min(max_heads, query_heads - begin);
        const auto block = tokens * count;
        const auto lanes = slots / block;
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
    if (plan.profile != context.profile ||
        input.parameter_profile != context.profile ||
        !SamePreparedPacking(plan.input_packing, input.physical_packing) ||
        input.ciphertexts.size() != plan.input_packing.ciphertext_count)
        return Status::FailedPrecondition(
            "prepared projection profile or input packing mismatch");
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
    PreparedProjectionRotationCache *rotation_cache)
{
    auto valid = ValidatePreparedProjection(plan, input, context);
    if (!valid.ok()) return valid.status();
    if (plan.bias_plaintexts.size() != plan.output_packing.ciphertext_count)
        return Status::Internal("prepared projection bias group count mismatch");
    try {
        seal::Evaluator evaluator(*context.context);
        PreparedProjectionRotationCache local_cache;
        auto &cache = rotation_cache ? *rotation_cache : local_cache;
        auto &bases = cache.bases;
        auto &babies = cache.babies;
        std::vector<seal::Ciphertext> output(
            plan.output_packing.ciphertext_count);
        std::vector<bool> initialized(output.size(), false);
        auto rotated = [&](const seal::Ciphertext &source, int step) {
            seal::Ciphertext result = source;
            const auto slots = static_cast<long long>(context.slot_count());
            long long canonical = static_cast<long long>(step) % slots;
            if (canonical > slots / 2) canonical -= slots;
            if (canonical <= -slots / 2) canonical += slots;
            if (canonical != 0) {
                evaluator.rotate_vector_inplace(result,
                                                static_cast<int>(canonical),
                                                context.galois_keys);
                if (counts) ++counts->rotations;
            }
            return result;
        };
        for (const auto &group : plan.groups) {
            if (group.target_ct >= output.size() || group.terms.empty())
                return Status::Internal("prepared projection group is invalid");
            seal::Ciphertext group_sum;
            bool first_term = true;
            for (const auto &term : group.terms) {
                if (!term.plaintext ||
                    term.source_ct >= input.ciphertexts.size())
                    return Status::Internal("prepared projection term is invalid");
                const auto base_key = std::make_pair(
                    term.source_ct, term.source_component);
                auto base = bases.find(base_key);
                if (base == bases.end()) {
                    const auto &source = input.ciphertexts[term.source_ct];
                    seal::Ciphertext conjugated, extracted;
                    evaluator.complex_conjugate(source, context.galois_keys,
                                                conjugated);
                    if (term.source_component == ComplexLane::kReal)
                        evaluator.add(source, conjugated, extracted);
                    else
                        evaluator.sub(source, conjugated, extracted);
                    base = bases.emplace(base_key, std::move(extracted)).first;
                    if (counts) ++counts->conjugations;
                }
                seal::Ciphertext baby;
                if (plan.reuse_baby_rotations) {
                    const auto baby_key = std::make_tuple(
                        term.source_ct, term.source_component,
                        term.baby_rotation);
                    auto saved = babies.find(baby_key);
                    if (saved == babies.end())
                        saved = babies.emplace(
                            baby_key,
                            rotated(base->second, term.baby_rotation)).first;
                    baby = saved->second;
                } else {
                    baby = rotated(base->second, term.baby_rotation);
                }
                seal::Ciphertext product;
                evaluator.multiply_plain(baby, *term.plaintext, product);
                if (counts) ++counts->ciphertext_plain_multiplications;
                if (first_term) {
                    group_sum = std::move(product);
                    first_term = false;
                } else {
                    evaluator.add_inplace(group_sum, product);
                }
            }
            if (group.giant_rotation != 0)
                group_sum = rotated(group_sum, group.giant_rotation);
            if (!initialized[group.target_ct]) {
                output[group.target_ct] = std::move(group_sum);
                initialized[group.target_ct] = true;
            } else {
                evaluator.add_inplace(output[group.target_ct], group_sum);
            }
        }
        const double target = std::ldexp(1.0, plan.output_scale_bits);
        for (std::size_t ct = 0; ct < output.size(); ++ct) {
            if (!initialized[ct])
                return Status::Internal(
                    "prepared projection output group is empty");
            if (!plan.bias_after_rescale && plan.bias_plaintexts[ct])
                evaluator.add_plain_inplace(output[ct],
                                            *plan.bias_plaintexts[ct]);
            evaluator.rescale_to_next_inplace(output[ct]);
            if (!std::isfinite(output[ct].scale()) ||
                std::abs(output[ct].scale() / target - 1.0) >
                    plan.scale_tolerance)
                return Status::FailedPrecondition(
                    "prepared projection output scale drift");
            output[ct].scale() = target;
            if (counts) ++counts->rescales;
            if (plan.bias_after_rescale && plan.bias_plaintexts[ct])
                evaluator.add_plain_inplace(output[ct],
                                            *plan.bias_plaintexts[ct]);
        }
        return output;
    } catch (const std::exception &error) {
        return Status::Internal(std::string("prepared projection failed: ") +
                                error.what());
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
    std::set<std::tuple<std::size_t, std::size_t, ComplexLane>> occupied;
    for (const auto &replicas : logical_locations) {
        if (replicas.empty())
            return Status::Invalid("HE packing omits a logical coordinate");
        for (const auto &location : replicas) {
            if (location.ciphertext >= ciphertext_count ||
                location.slot >= slot_count)
                return Status::Invalid("HE packing location is out of range");
            if (!occupied.emplace(location.ciphertext, location.slot,
                                  location.component).second)
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

Result<HePackingPlan> MultiHeadPackingPlan::DenseSpatial(
    const TensorShape &shape, std::size_t slot_count)
{
    auto count = shape.ElementCount();
    if (!count.ok()) return count.status();
    if (slot_count == 0)
        return Status::Invalid("dense spatial packing has zero CKKS slots");
    HePackingPlan result;
    result.layout = Layout::kSpatial;
    result.logical_shape = shape;
    result.slot_count = slot_count;
    result.ciphertext_count =
        (count.value() + 2 * slot_count - 1) / (2 * slot_count);
    result.logical_locations.resize(count.value());
    for (std::size_t logical = 0; logical < count.value(); ++logical) {
        const auto chunk = logical / (2 * slot_count);
        const auto local = logical % (2 * slot_count);
        result.logical_locations[logical].push_back(
            {chunk, local % slot_count,
             local < slot_count ? ComplexLane::kReal
                                : ComplexLane::kImaginary,
             std::nullopt});
    }
    auto valid = result.Validate();
    if (!valid.ok()) return valid.status();
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
    auto tiles = MakeTiles(layer.tokens, layer.q_heads, width, slot_count);
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
        const auto block = layer.tokens * tile.head_count;
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
                                      token * tile.head_count + local_head;
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
        const auto block = layer.tokens * tile.head_count;
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
                                      token * tile.head_count + local_head;
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
        const auto block = layer.tokens * tile.head_count;
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
                                      row * tile.head_count + local_head;
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
    std::vector<int> rotations;
    return ComplexDiagonalProjectionPlan{
        layer.model_width, output_columns, output_layout,
        output_layout != Layout::kMultiHeadQ &&
            layer.q_heads != layer.kv_heads,
        packing.take_value(), std::move(rotations)};
}

Result<void> SealCipherTensor::Validate(const HeContext &context) const
{
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
    if (!tensors_.emplace(id, std::move(tensor)).second)
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
    return found->second;
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
    staged_projections_.clear();
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
    return preparing_plaintexts_ ? staged_plaintexts_.size()
                                 : encoded_plaintexts_.size();
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

std::size_t SealPlainParameterStore::virtual_weight_template_count() const
    noexcept
{
    std::lock_guard<std::mutex> lock(mutex_);
    return virtual_weight_template_count_;
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

} // namespace hera
