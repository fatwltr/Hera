#include "hera/block2_he.h"

#include <limits>
#include <cstring>
#include <sstream>
#include <set>
#include <stdexcept>
#include <type_traits>

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
    std::vector<int> rotations;
    for (std::size_t power = 1; power <= slot_count / 2; power <<= 1U) {
        rotations.push_back(static_cast<int>(power));
        if (power != slot_count / 2)
            rotations.push_back(-static_cast<int>(power));
        if (power > slot_count / 4) break;
    }
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
    if (!matrices_.emplace(handle.id,
                           DenseMatrixParameter{handle.spec,
                                                std::move(row_major)}).second)
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
    if (!vectors_.emplace(handle.id,
                          DenseVectorParameter{handle.spec,
                                               std::move(values)}).second)
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
    return found->second;
}

Result<DenseVectorParameter> SealPlainParameterStore::GetVector(
    PlainParameterId id) const
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
    encoded_plaintexts_.clear();
    encoded_plaintext_bytes_ = 0;
    plaintext_cache_hits_ = 0;
    virtual_weight_requests_ = 0;
    virtual_weight_cache_hits_ = 0;
    preparing_plaintexts_ = true;
    plaintext_cache_frozen_ = false;
}

void SealPlainParameterStore::FreezePlaintextCache()
{
    std::lock_guard<std::mutex> lock(mutex_);
    preparing_plaintexts_ = false;
    plaintext_cache_frozen_ = true;
}

bool SealPlainParameterStore::plaintext_cache_frozen() const noexcept
{
    std::lock_guard<std::mutex> lock(mutex_);
    return plaintext_cache_frozen_;
}

std::size_t SealPlainParameterStore::encoded_plaintext_count() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return encoded_plaintexts_.size();
}

std::size_t SealPlainParameterStore::encoded_plaintext_bytes() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return encoded_plaintext_bytes_;
}

std::size_t SealPlainParameterStore::plaintext_cache_hits() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return plaintext_cache_hits_;
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
    return virtual_weight_requests_;
}

std::size_t SealPlainParameterStore::virtual_weight_cache_hits() const noexcept
{
    std::lock_guard<std::mutex> lock(mutex_);
    return virtual_weight_cache_hits_;
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
    if (virtualized) ++virtual_weight_requests_;
    const auto found = encoded_plaintexts_.find(key);
    if (found != encoded_plaintexts_.end()) {
        ++plaintext_cache_hits_;
        if (virtualized) ++virtual_weight_cache_hits_;
        return found->second;
    }
    if (plaintext_cache_frozen_)
        return Status::FailedPrecondition(
            std::string("online model plaintext cache miss: ") +
            std::string(kernel_tag));
    if (!preparing_plaintexts_)
        return Status::FailedPrecondition(
            "model plaintext encode attempted outside Prepare()");
    seal::Plaintext plain;
    encoder.encode(values, parms_id, scale, plain);
    encoded_plaintext_bytes_ += plain.save_size(seal::compr_mode_type::none);
    encoded_plaintexts_.emplace(key, plain);
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
    if (virtualized) ++virtual_weight_requests_;
    const auto found = encoded_plaintexts_.find(key);
    if (found != encoded_plaintexts_.end()) {
        ++plaintext_cache_hits_;
        if (virtualized) ++virtual_weight_cache_hits_;
        return found->second;
    }
    if (plaintext_cache_frozen_)
        return Status::FailedPrecondition(
            std::string("online model plaintext cache miss: ") +
            std::string(kernel_tag));
    if (!preparing_plaintexts_)
        return Status::FailedPrecondition(
            "model plaintext encode attempted outside Prepare()");
    seal::Plaintext plain;
    encoder.encode(values, parms_id, scale, plain);
    encoded_plaintext_bytes_ += plain.save_size(seal::compr_mode_type::none);
    encoded_plaintexts_.emplace(key, plain);
    return plain;
}

} // namespace hera
