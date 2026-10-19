#include "hera/block2_he.h"
#include "hera/detail/parallel_ranges.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <set>

namespace {

void Require(bool condition, const std::string &message)
{
    if (!condition) throw std::runtime_error(message);
}

hera::HeraConfig MakeConfig()
{
    hera::HeraConfig config;
    hera::TransformerLayerConfig layer;
    layer.id = "layer0";
    layer.tokens = 2;
    layer.model_width = 4;
    layer.q_heads = 2;
    layer.kv_heads = 1;
    layer.qk_head_width = 2;
    layer.value_head_width = 3;
    layer.ffn_width = 8;
    layer.norm = hera::NormPlacement::kPreNorm;
    layer.qkv_bias = true;
    config.layers = {layer};
    hera::HeParameterConfig profile;
    profile.polynomial_degree = 4096;
    profile.coefficient_modulus_bits = {60, 35, 60, 60, 60, 60, 60};
    profile.default_scale_bits = 44;
    profile.input_scale_bits = 44;
    profile.output_scale_bits = 44;
    profile.declared_depth = 6;
    profile.terminal_prime_count = 1;
    profile.require_exact_terminal_level = true;
    profile.security_level = hera::SecurityLevel::kNone;
    config.he_parameter_profiles.emplace("block2", profile);
    hera::LayerExecutionConfig execution;
    execution.layer_id = layer.id;
    for (const auto region : {hera::HeRegionId::kRegion1,
                              hera::HeRegionId::kRegion2,
                              hera::HeRegionId::kRegion3,
                              hera::HeRegionId::kRegion4,
                              hera::HeRegionId::kRegion5})
        execution.regions.emplace(
            region, hera::HeRegionConfig{region, "block2",
                                         hera::Layout::kSpatial,
                                         hera::Layout::kSpatial});
    config.layer_execution = {execution};
    config.threads = 1;
    return config;
}

hera::HePackingPlan MakeInterleavedSpatial(
    const hera::TensorShape &shape, std::size_t slots)
{
    const auto count = shape.ElementCount().value();
    hera::HePackingPlan plan;
    plan.layout = hera::Layout::kSpatial;
    plan.logical_shape = shape;
    plan.slot_count = slots;
    plan.ciphertext_count = (count + 2 * slots - 1) / (2 * slots);
    plan.logical_locations.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
        const auto ct = i / (2 * slots);
        const auto local = i % (2 * slots);
        plan.logical_locations[i].push_back(
            {ct, (local / 2) % slots,
             (local & 1U) == 0 ? hera::ComplexLane::kReal
                               : hera::ComplexLane::kImaginary,
             std::nullopt});
    }
    Require(plan.Validate().ok(), "interleaved spatial plan is invalid");
    return plan;
}

hera::SealCipherTensor Encrypt(
    const std::vector<double> &values, const hera::HePackingPlan &packing,
    const hera::HeContext &context, std::size_t level = 0)
{
    std::vector<std::vector<std::complex<double>>> slots(
        packing.ciphertext_count,
        std::vector<std::complex<double>>(packing.slot_count));
    for (std::size_t logical = 0; logical < values.size(); ++logical) {
        for (const auto &location : packing.Locations(logical)) {
            if (location.component == hera::ComplexLane::kReal)
                slots[location.ciphertext][location.slot].real(values[logical]);
            else
                slots[location.ciphertext][location.slot].imag((packing.layout == hera::Layout::kMultiHeadK ? -1.0 : 1.0) * values[logical]);
        }
    }
    seal::CKKSEncoder encoder(*context.context);
    seal::Encryptor encryptor(*context.context, *context.secret_key);
    std::vector<seal::Ciphertext> ciphertexts(packing.ciphertext_count);
    const double scale = std::ldexp(1.0, context.parameters.input_scale_bits);
    for (std::size_t i = 0; i < ciphertexts.size(); ++i) {
        seal::Plaintext plain;
        encoder.encode(slots[i], scale, plain);
        encryptor.encrypt_symmetric(plain, ciphertexts[i]);
        if (level != 0) {
            auto data = context.context->first_context_data();
            for (std::size_t step = 0; step < level && data; ++step)
                data = data->next_context_data();
            Require(static_cast<bool>(data), "invalid requested encryption level");
            seal::Evaluator evaluator(*context.context);
            evaluator.mod_switch_to_inplace(ciphertexts[i], data->parms_id());
        }
    }
    hera::TensorSpec spec;
    spec.shape = packing.logical_shape;
    spec.domain = hera::Domain::kHeCiphertext;
    spec.layout = packing.layout;
    spec.numeric = {43, 13};
    spec.he_region = hera::HeRegionId::kRegion2;
    spec.he_parameter_profile = context.profile;
    spec.planned_level = level;
    spec.planned_scale_bits = context.parameters.input_scale_bits;
    return {context.profile, std::move(ciphertexts), spec, packing,
            packing.layout == hera::Layout::kMultiHeadK};
}

std::vector<double> Decrypt(
    const hera::SealCipherTensor &tensor, const hera::HeContext &context);
double MaxError(const std::vector<double> &a,
                const std::vector<double> &b);

void TestParallelQkGroups(bool padded_tokens = false, bool split_heads = false)
{
    auto config = MakeConfig();
    auto &layer = config.layers.front();
    layer.tokens = split_heads ? 16 : (padded_tokens ? 3 : 32);
    layer.q_heads = split_heads ? 12 : 8;
    layer.kv_heads = split_heads ? 3 : 2;
    layer.qk_head_width = split_heads ? 64 : (padded_tokens ? 131 : 19);
    layer.qkv_bias = split_heads;
    if (split_heads) layer.model_width = layer.tokens;
    const auto &profile = config.he_parameter_profiles.at("block2");
    const auto slots = profile.polynomial_degree / 2;
    auto q_plan = hera::MultiHeadPackingPlan::ProjectionOutput(
        layer, hera::Layout::kMultiHeadQ, slots);
    auto k_plan = hera::MultiHeadPackingPlan::ProjectionOutput(
        layer, hera::Layout::kMultiHeadK, slots);
    Require(q_plan.ok() && k_plan.ok(), "parallel QK packing failed");
    if (split_heads)
        Require(q_plan.value().ciphertext_count == 3 &&
                    k_plan.value().ciphertext_count == 3,
                "12-head QK did not consume the 8+4 producer layout");
    // Direct QK cases only generate QK keys. The mixed-head case also needs
    // Wq/Wk keys to verify their producer layout, without unrelated V keys.
    auto steps = hera::MultiHeadQkKernel::RequiredRotationSteps(layer, slots);
    Require(steps.ok(), steps.status().message());
    auto x_plan = hera::MultiHeadPackingPlan::RealStrictColumns(
        {{layer.tokens, layer.model_width}}, slots);
    Require(x_plan.ok(), x_plan.status().message());
    if (split_heads) {
        std::set<int> keys(steps.value().begin(), steps.value().end());
        for (auto layout : {hera::Layout::kMultiHeadQ, hera::Layout::kMultiHeadK}) {
            auto projection_steps = hera::ComplexDiagonalProjectionKernel::RequiredRotationSteps(
                layer, x_plan.value(), layout);
            Require(projection_steps.ok(), projection_steps.status().message());
            keys.insert(projection_steps.value().begin(), projection_steps.value().end());
        }
        steps.value().assign(keys.begin(), keys.end());
    }
    hera::HeContextRegistry client;
    Require(client.CreateClientProfiles(config).ok(), "QK client setup failed");
    Require(client.CompileRotationKeys("block2", steps.value()).ok(),
            "QK rotation keys failed");
    auto bundle = client.ExportPublicBundle();
    Require(bundle.ok(), bundle.status().message());
    hera::HeContextRegistry server;
    Require(server.CreateServerProfiles(config, bundle.value()).ok(),
            "QK server setup failed");
    const auto &client_context = client.Require("block2", hera::PartyRole::kClient);
    const auto &server_context = server.Require("block2", hera::PartyRole::kServer);
    std::vector<double> q(layer.tokens * hera::QueryWidth(layer));
    std::vector<double> k(layer.tokens * hera::KeyWidth(layer));
    for (std::size_t i = 0; i < q.size(); ++i) {
        q[i] = (static_cast<int>(i % 11) - 5) * 0.003;
    }
    for (std::size_t i = 0; i < k.size(); ++i)
        k[i] = (static_cast<int>(i % 13) - 6) * 0.002;
    std::vector<double> expected(layer.q_heads * layer.tokens * layer.tokens);
    for (std::size_t head = 0; head < layer.q_heads; ++head)
        for (std::size_t row = 0; row < layer.tokens; ++row)
            for (std::size_t col = 0; col < layer.tokens; ++col)
                for (std::size_t dim = 0; dim < layer.qk_head_width; ++dim)
                    expected[(head * layer.tokens + row) * layer.tokens + col] +=
                        q[(row * layer.q_heads + head) * layer.qk_head_width + dim] *
                        k[(col * layer.kv_heads + head / (layer.q_heads / layer.kv_heads)) * layer.qk_head_width + dim];
    constexpr std::size_t kQkStartLevel = 2;
    for (auto &value : q) value /= std::sqrt(2.0);
    for (auto &value : k) value /= std::sqrt(2.0);
    hera::SealCipherTensor q_cipher, k_cipher;
    if (split_heads) {
        // A common real-column X produces both Q and K through preprocessed
        // complex weights, including bias and replicated GQA heads.
        std::vector<double> identity(layer.tokens * layer.model_width);
        for (std::size_t token = 0; token < layer.tokens; ++token)
            identity[token * layer.model_width + token] = 1.0;
        auto x = Encrypt(identity, x_plan.value(), client_context, kQkStartLevel - 1);
        hera::ComplexDiagonalProjectionKernel projection(layer, server_context);
        for (auto layout : {hera::Layout::kMultiHeadQ, hera::Layout::kMultiHeadK}) {
            const bool is_q = layout == hera::Layout::kMultiHeadQ;
            const auto &target = is_q ? q : k;
            const auto columns = is_q ? hera::QueryWidth(layer) : hera::KeyWidth(layer);
            const auto name = is_q ? "Wq" : "Wk";
            const auto factor = 1.0 / std::sqrt(2.0 * (is_q
                ? static_cast<double>(layer.qk_head_width) : 1.0));
            std::vector<double> weights(target.size()), bias(columns);
            for (std::size_t column = 0; column < columns; ++column)
                bias[column] = (static_cast<int>(column % 17) - 8) * 0.001;
            for (std::size_t logical = 0; logical < weights.size(); ++logical)
                weights[logical] = target[logical] / factor - bias[logical % columns];
            hera::DenseMatrixParameter matrix{{name, layer.model_width, columns, true}, weights};
            hera::DenseVectorParameter bias_parameter{{is_q ? "bq" : "bk", columns}, bias};
            auto output = projection.Evaluate(x, matrix, bias_parameter, layout);
            Require(output.ok(), output.status().message());
            Require(output.value().logical_spec.planned_level == kQkStartLevel &&
                        MaxError(Decrypt(output.value(), client_context), target) < 1e-4,
                    "mixed-head Wq/Wk projection changed values or the QK input level");
            (is_q ? q_cipher : k_cipher) = output.take_value();
        }
    } else {
        q_cipher = Encrypt(q, q_plan.value(), client_context, kQkStartLevel);
        k_cipher = Encrypt(k, k_plan.value(), client_context, kQkStartLevel);
    }
    hera::Block2ExecutionReport serial_report, parallel_report;
    auto serial = hera::MultiHeadQkKernel::Evaluate(
        q_cipher, k_cipher, layer, server_context, &serial_report, 1);
    hera::SealPlainParameterStore qk_cache;
    qk_cache.BeginPlaintextPreparation();
    {
        hera::ModelPlaintextCacheScope scope(&qk_cache, layer.id);
        auto compiled = hera::MultiHeadQkKernel::Prepare(layer, server_context);
        Require(compiled.ok(), compiled.status().message());
    }
    const auto compiled_mask_count = qk_cache.encoded_plaintext_count();
    {
        hera::ModelPlaintextCacheScope scope(&qk_cache, "another_layer");
        auto shared = hera::MultiHeadQkKernel::Prepare(layer, server_context);
        Require(shared.ok() && qk_cache.encoded_plaintext_count() == compiled_mask_count,
                "identical layers duplicated the QK program or its fixed plaintexts");
        auto original = qk_cache.GetPreparedBlock2Program(layer.id + "/block2/qk");
        auto duplicate = qk_cache.GetPreparedBlock2Program("another_layer/block2/qk");
        Require(original.ok() && duplicate.ok() && original.value() == duplicate.value(),
                "identical layers did not share their immutable QK program");
    }
    qk_cache.FreezePlaintextCache();
    const auto prepared_masks = qk_cache.encoded_plaintext_count();
    hera::Result<hera::SealCipherTensor> parallel = hera::Status::Internal("QK not evaluated");
    {
        hera::ModelPlaintextCacheScope scope(&qk_cache, layer.id);
        parallel = hera::MultiHeadQkKernel::Evaluate(
            q_cipher, k_cipher, layer, server_context, &parallel_report, 8);
        Require(qk_cache.encoded_plaintext_count() == prepared_masks,
                "frozen QK encoded more plaintexts online");
        auto incompatible_layer = layer;
        ++incompatible_layer.qk_head_width;
        Require(!hera::MultiHeadQkKernel::Evaluate(q_cipher, k_cipher,
                    incompatible_layer, server_context).ok(),
                "QK reused a prepared program for a different head width");
    }
    Require(serial.ok() && parallel.ok(),
            serial.ok() ? parallel.status().message() : serial.status().message());
    if (split_heads) {
        auto incompatible = k_cipher;
        std::swap(incompatible.physical_packing.logical_locations[0],
                  incompatible.physical_packing.logical_locations[1]);
        auto rejected = hera::MultiHeadQkKernel::Evaluate(
            q_cipher, incompatible, layer, server_context);
        Require(!rejected.ok() && rejected.status().message().find(
                    "head-group packing decision") != std::string::npos,
                "QK accepted a different producer packing with the same CT count");
    }
    const auto bsgs = hera::MultiHeadQkKernel::Plan(layer, slots);
    Require(bsgs.ok(), bsgs.status().message());
    std::cout << "HERA_QK_PLANNED_KS="
              << bsgs.value().rotations + bsgs.value().conjugations +
                     bsgs.value().relinearizations << '\n';
    auto wrong_convention = k_cipher;
    wrong_convention.conjugated_complex = false;
    auto convention_rejected = hera::MultiHeadQkKernel::Evaluate(
        q_cipher, wrong_convention, layer, server_context);
    Require(!convention_rejected.ok(), "QK accepted unconjugated K");
    auto wrong_level = q_cipher;
    seal::Evaluator bad_evaluator(*client_context.context);
    for (auto &ct : wrong_level.ciphertexts) bad_evaluator.mod_switch_to_next_inplace(ct);
    auto level_rejected = hera::MultiHeadQkKernel::Evaluate(
        wrong_level, k_cipher, layer, server_context);
    Require(!level_rejected.ok(), "QK accepted a mismatched modulus suffix");
    auto wrong_scale = q_cipher;
    wrong_scale.ciphertexts.front().scale() *= 2;
    Require(!hera::MultiHeadQkKernel::Evaluate(
                wrong_scale, k_cipher, layer, server_context).ok(),
            "QK accepted a mismatched scale");
    auto serial_values = Decrypt(serial.value(), client_context);
    auto parallel_values = Decrypt(parallel.value(), client_context);
    Require(MaxError(serial_values, expected) < 1e-4 &&
                MaxError(parallel_values, expected) < 1e-4,
            "QK paired dimensions/GQA/complex score packing differ from plaintext");
    Require(serial.value().logical_spec.planned_level == 6,
            "four-level QK did not reach the terminal boundary");
    std::cout << "HERA_QK_PAIRED_DIMENSIONS_MAX_ERROR="
              << MaxError(serial_values, expected) << '\n';
    Require(MaxError(serial_values, parallel_values) < 1e-4,
            "parallel QK differs from serial QK");
    const auto same_counts = [&](const hera::Block2ExecutionReport &other) {
        return serial_report.rotations == other.rotations &&
            serial_report.conjugations == other.conjugations &&
            serial_report.ciphertext_plain_multiplications == other.ciphertext_plain_multiplications &&
            serial_report.ciphertext_multiplications == other.ciphertext_multiplications &&
            serial_report.relinearizations == other.relinearizations &&
            serial_report.rescales == other.rescales;
    };
    Require(same_counts(parallel_report) && parallel_report.qk_preparation_workers > 1,
            "QK parallel preparation changed operation counts or stayed serial");
    if (padded_tokens) {
        auto &pool = hera::detail::HeWorkerPool::Instance();
        const auto budget = pool.MemoryBudget();
        pool.SetMemoryBudget(1);
        hera::Block2ExecutionReport constrained;
        auto limited = hera::MultiHeadQkKernel::Evaluate(
            q_cipher, k_cipher, layer, server_context, &constrained, 8);
        pool.SetMemoryBudget(budget);
        Require(limited.ok(), limited.status().message());
        Require(constrained.qk_preparation_workers == 1 &&
                    constrained.qk_product_workers == 1 && same_counts(constrained) &&
                    MaxError(Decrypt(limited.value(), client_context), expected) < 1e-4,
                "memory-constrained QK changed the result or operation counts");
        hera::Block2ExecutionReport nested_report;
        auto task = hera::detail::SubmitLocalTask(2, [&] {
            return hera::MultiHeadQkKernel::Evaluate(
                q_cipher, k_cipher, layer, server_context, &nested_report, 8);
        });
        auto nested = task.get();
        Require(nested.ok(), nested.status().message());
        Require(nested_report.qk_preparation_workers == 1 &&
                    nested_report.qk_product_workers == 1 && same_counts(nested_report) &&
                    MaxError(Decrypt(nested.value(), client_context), expected) < 1e-4,
                "nested QK scheduling changed its result or failed to serialize");
    }
    std::cout << "HERA_QK_SERIAL_KS="
              << serial_report.rotations + serial_report.conjugations +
                     serial_report.relinearizations << '\n'
              << "HERA_QK_PARALLEL_KS="
              << parallel_report.rotations + parallel_report.conjugations +
                     parallel_report.relinearizations << '\n';
}

std::vector<double> Decrypt(
    const hera::SealCipherTensor &tensor, const hera::HeContext &context)
{
    seal::Decryptor decryptor(*context.context, *context.secret_key);
    seal::CKKSEncoder encoder(*context.context);
    std::vector<std::vector<std::complex<double>>> decoded(
        tensor.ciphertexts.size());
    for (std::size_t i = 0; i < tensor.ciphertexts.size(); ++i) {
        seal::Plaintext plain;
        decryptor.decrypt(tensor.ciphertexts[i], plain);
        encoder.decode(plain, decoded[i]);
    }
    std::vector<double> result(tensor.physical_packing.logical_locations.size());
    for (std::size_t logical = 0; logical < result.size(); ++logical) {
        const auto &location = tensor.physical_packing.Locations(logical).front();
        const auto value = decoded[location.ciphertext][location.slot];
        result[logical] = location.component == hera::ComplexLane::kReal
            ? value.real() : (tensor.conjugated_complex ? -value.imag() : value.imag());
    }
    return result;
}

hera::LayerWeightHandles RegisterWeights(
    const hera::TransformerLayerConfig &layer,
    hera::SealPlainParameterStore &store,
    std::vector<double> &wq, std::vector<double> &wk,
    std::vector<double> &wv, bool fuse = true,
    const std::vector<double> &gamma_override = {})
{
    const auto schema = hera::DeriveWeightSchema(layer).value();
    hera::PlainParameterId id = 1;
    hera::LayerWeightHandles handles;
    handles.layer_id = layer.id;
    handles.wq = {id++, schema.wq};
    handles.wk = {id++, schema.wk};
    handles.wv = {id++, schema.wv};
    handles.wo = {id++, schema.wo};
    handles.wu = {id++, schema.wu};
    handles.wd = {id++, schema.wd};
    wq.resize(schema.wq.rows * schema.wq.columns);
    wk.resize(schema.wk.rows * schema.wk.columns);
    wv.resize(schema.wv.rows * schema.wv.columns);
    for (std::size_t i = 0; i < wq.size(); ++i)
        wq[i] = (static_cast<int>(i % 7) - 3) * 0.015;
    for (std::size_t i = 0; i < wk.size(); ++i)
        wk[i] = (static_cast<int>(i % 5) - 2) * 0.02;
    for (std::size_t i = 0; i < wv.size(); ++i)
        wv[i] = (static_cast<int>(i % 9) - 4) * 0.01;
    Require(store.PutMatrix(handles.wq, wq).ok(), "cannot register Wq");
    Require(store.PutMatrix(handles.wk, wk).ok(), "cannot register Wk");
    Require(store.PutMatrix(handles.wv, wv).ok(), "cannot register Wv");
    for (const auto &spec : schema.vectors) {
        hera::PlainParameterHandle handle{id++, spec};
        std::vector<double> values(spec.length, 0.0);
        if (spec.name == "ln1.gamma")
            for (std::size_t i = 0; i < values.size(); ++i)
                values[i] = 0.9 + 0.05 * static_cast<double>(i);
        if (spec.name == "ln1.gamma" && !gamma_override.empty())
            values = gamma_override;
        else if (spec.name == "ln1.beta")
            for (std::size_t i = 0; i < values.size(); ++i)
                values[i] = -0.02 + 0.01 * static_cast<double>(i);
        else if (spec.name == "bq" || spec.name == "bk" ||
                 spec.name == "bv")
            for (std::size_t i = 0; i < values.size(); ++i)
                values[i] = (static_cast<int>(i % 3) - 1) * 0.004;
        Require(store.PutVector(handle, values).ok(),
                "cannot register vector " + spec.name);
        handles.plain_parameters.push_back(handle);
    }
    if (fuse) {
        auto fused = store.FuseLayerNormGammaIntoQkv(handles);
        Require(fused.ok(), fused.status().message());
    }
    return handles;
}

std::vector<double> MatMul(const std::vector<double> &x,
                           std::size_t rows, std::size_t inner,
                           const std::vector<double> &w,
                           std::size_t columns)
{
    std::vector<double> result(rows * columns);
    for (std::size_t row = 0; row < rows; ++row)
        for (std::size_t column = 0; column < columns; ++column)
            for (std::size_t k = 0; k < inner; ++k)
                result[row * columns + column] +=
                    x[row * inner + k] * w[k * columns + column];
    return result;
}

double MaxError(const std::vector<double> &a,
                const std::vector<double> &b)
{
    Require(a.size() == b.size(), "error vectors have different sizes");
    double result = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        result = std::max(result, std::abs(a[i] - b[i]));
    return result;
}

std::size_t ActualLevel(const hera::HeContext &context,
                        seal::parms_id_type id)
{
    std::size_t level = 0;
    for (auto data = context.context->first_context_data(); data;
         data = data->next_context_data(), ++level)
        if (data->parms_id() == id) return level;
    throw std::runtime_error("ciphertext has an unknown parms_id");
}

void TestPackingVariants(const hera::TransformerLayerConfig &base,
                         std::size_t slots)
{
    for (const std::size_t kv_heads : {std::size_t{4}, std::size_t{2},
                                       std::size_t{1}}) {
        auto layer = base;
        layer.q_heads = 4;
        layer.kv_heads = kv_heads;
        auto k = hera::MultiHeadPackingPlan::ProjectionOutput(
            layer, hera::Layout::kMultiHeadK, slots);
        Require(k.ok(), "MHA/GQA/MQA K packing rejected");
        for (const auto &locations : k.value().logical_locations) {
            Require(locations.size() == layer.q_heads / layer.kv_heads,
                    "K physical head replication count is wrong");
        }
    }

    auto smoke = base;
    smoke.tokens = 32;
    smoke.model_width = 512;
    smoke.q_heads = 8;
    smoke.kv_heads = 2;
    smoke.qk_head_width = 70;
    smoke.value_head_width = 63;
    auto input = hera::MultiHeadPackingPlan::ProjectionInput(
        smoke, 16384);
    Require(input.ok(), "smoke projection-input packing rejected");
    Require(input.value().ciphertext_count == 4,
            "smoke projection input should use four ciphertexts");
    Require(input.value().Locations(0).size() == smoke.q_heads,
            "spatial input was not replicated across query heads");
    const auto imag_feature = 64;
    const auto imag_logical = imag_feature;
    Require(input.value().Locations(imag_logical).front().component ==
                hera::ComplexLane::kImaginary,
            "projection input did not fill imaginary lanes after real lanes");
    auto q = hera::MultiHeadPackingPlan::ProjectionOutput(
        smoke, hera::Layout::kMultiHeadQ, 16384);
    auto v = hera::MultiHeadPackingPlan::ProjectionOutput(
        smoke, hera::Layout::kMultiHeadV, 16384);
    Require(q.ok() && v.ok(), "tail multihead projection packing rejected");
    Require(q.value().Validate().ok() && v.value().Validate().ok(),
            "tail multihead projection packing is not bijective");
}

void TestPowerOfTwoValueProjection()
{
    auto config = MakeConfig();
    auto &layer = config.layers.front();
    layer.tokens = 32;
    layer.q_heads = 3;
    layer.kv_heads = 1;
    layer.qk_head_width = 12;
    layer.value_head_width = 18;
    auto &profile = config.he_parameter_profiles.at("block2");
    // This regression only evaluates one projection level. Avoid allocating
    // the full Block-2 key-switch chain for its standalone Wv check.
    profile.polynomial_degree = 2048;
    profile.coefficient_modulus_bits = {60, 60};
    profile.declared_depth = 1;
    const auto slots = profile.polynomial_degree / 2;
    // Capacity 1024/(32*3)=10 rounds down to eight for V; Q/K instead
    // use one-head groups. Dimensions 8 and 16 cross V's component/CT edges.
    auto v_plan = hera::MultiHeadPackingPlan::ProjectionOutput(
        layer, hera::Layout::kMultiHeadV, slots);
    Require(v_plan.ok() && v_plan.value().ciphertext_count == 2 &&
                v_plan.value().Locations(8).front().component ==
                    hera::ComplexLane::kImaginary &&
                v_plan.value().Locations(16).front().ciphertext == 1,
            "Wv power-of-two projection packing is incorrect");
    hera::HeContextRegistry client;
    Require(client.CreateClientProfiles(config, false).ok(),
            "power-of-two Wv client context failed");
    const auto packing = hera::MultiHeadPackingPlan::RealStrictColumns(
        {{layer.tokens, layer.model_width}}, slots).value();
    auto steps = hera::ComplexDiagonalProjectionKernel::RequiredRotationSteps(
        layer, packing, hera::Layout::kMultiHeadV);
    Require(steps.ok(), steps.status().message());
    Require(steps.value().size() < slots,
            "standalone Wv still requests all rotation keys");
    Require(client.CompileRotationKeys("block2", steps.value()).ok(),
            "power-of-two Wv keys failed");
    auto bundle = client.ExportPublicBundle();
    Require(bundle.ok(), bundle.status().message());
    hera::HeContextRegistry server;
    Require(server.CreateServerProfiles(config, bundle.value()).ok(),
            "power-of-two Wv server context failed");
    const auto &client_context = client.Require("block2", hera::PartyRole::kClient);
    const auto &server_context = server.Require("block2", hera::PartyRole::kServer);
    Require(server_context.relinearization_keys.size() == 0 &&
                server_context.galois_keys.size() == steps.value().size(),
            "standalone Wv received unused evaluation keys");
    std::vector<double> values(layer.tokens * layer.model_width);
    std::vector<double> weights(layer.model_width * hera::ValueWidth(layer));
    for (std::size_t i = 0; i < values.size(); ++i)
        values[i] = std::sin(static_cast<double>(i + 1) * 0.13) * 0.2;
    for (std::size_t i = 0; i < weights.size(); ++i)
        weights[i] = (static_cast<int>(i % 9) - 4) * 0.03;
    hera::DenseMatrixParameter wv{
        {"Wv", layer.model_width, hera::ValueWidth(layer), true}, weights};
    auto input = Encrypt(values, packing, client_context);
    hera::ComplexDiagonalProjectionKernel projection(layer, server_context);
    auto output = projection.Evaluate(
        input, wv, std::nullopt, hera::Layout::kMultiHeadV);
    Require(output.ok(), output.status().message());
    const auto expected = MatMul(values, layer.tokens, layer.model_width,
                                weights, hera::ValueWidth(layer));
    Require(MaxError(Decrypt(output.value(), client_context), expected) < 1e-4,
            "power-of-two Wv projection differs from plaintext");

    hera::SealPlainParameterStore parameters;
    parameters.BeginPlaintextPreparation();
    {
        hera::ModelPlaintextCacheScope scope(&parameters, layer.id);
        auto metadata = input;
        for (auto &ct : metadata.ciphertexts) {
            const auto id = ct.parms_id();
            const auto scale = ct.scale();
            ct = seal::Ciphertext();
            ct.parms_id() = id;
            ct.scale() = scale;
        }
        auto prepared = projection.Evaluate(
            metadata, wv, std::nullopt, hera::Layout::kMultiHeadV);
        Require(prepared.ok(), prepared.status().message());
    }
    parameters.FreezePlaintextCache();
    {
        hera::ModelPlaintextCacheScope scope(&parameters, layer.id);
        auto cached = projection.Evaluate(
            input, wv, std::nullopt, hera::Layout::kMultiHeadV);
        Require(cached.ok(), cached.status().message());
        Require(MaxError(Decrypt(cached.value(), client_context), expected) < 1e-4 &&
                    cached.value().physical_packing.Locations(8).front()
                        .component == hera::ComplexLane::kImaginary,
                "prepared Wv did not preserve power-of-two packing/numerics");
    }
}


// Cross the input ciphertext boundary and both output component boundaries.
// Q/K use full-capacity lanes while V retains its power-of-two lane contract.
void TestColumnBsgsPlanning()
{
    // Retain a regression for the rotation-minimizing policy; the online
    // default can additionally account for hoisted baby work and residency.
    hera::ConfigureProjectionRotationHoisting(false);
    auto layer = MakeConfig().layers.front();
    layer.tokens = 128;
    layer.model_width = 1024;
    layer.q_heads = layer.kv_heads = 16;
    layer.qk_head_width = layer.value_head_width = 64;
    constexpr std::size_t slots = 16384;
    auto plan = hera::ComplexDiagonalProjectionPlanner::ColumnBsgs(layer, slots);
    Require(plan.ok(), plan.status().message());
    const auto &p = plan.value();
    Require(p.token_stride == 128 && p.columns_per_ciphertext == 128 &&
                p.input_ciphertexts == 8 && p.output_ciphertexts == 12 &&
                p.baby_steps == 16 && p.giant_steps == 8 &&
                p.rotation_upper_bound == 204,
            "BERT-large QKV BSGS did not minimize the shared rotation count");
    const auto qk = hera::MultiHeadQkKernel::Plan(layer, slots);
    Require(qk.ok(), qk.status().message());
    std::cout << "HERA_BERT_LARGE_QK_PLANNED_KS="
              << qk.value().rotations + qk.value().conjugations +
                     qk.value().relinearizations << '\n';
    std::set<int> keys;
    auto input = hera::MultiHeadPackingPlan::RealStrictColumns(
        {{layer.tokens, layer.model_width}}, slots).value();
    for (auto layout : {hera::Layout::kMultiHeadQ, hera::Layout::kMultiHeadK,
                        hera::Layout::kMultiHeadV}) {
        auto output = hera::MultiHeadPackingPlan::ProjectionOutput(layer, layout, slots);
        Require(output.ok() && output.value().ciphertext_count == 4,
                "BERT-large projection output count changed");
        const auto &packing = output.value();
        // Columns h=0,d=0; h=1,d=0; h=0,d=1. Tokens remain adjacent.
        Require(packing.Locations(64).front().slot == 128 &&
                    packing.Locations(1).front().slot == 16 * 128 &&
                    packing.Locations(1024).front().slot == 1 &&
                    packing.Locations(8).front().slot == 0 &&
                    packing.Locations(8).front().component == hera::ComplexLane::kImaginary,
                "multihead packing is not a permutation of whole token columns");
        auto steps = hera::ComplexDiagonalProjectionKernel::RequiredRotationSteps(
            layer, input, layout);
        Require(steps.ok(), steps.status().message());
        keys.insert(steps.value().begin(), steps.value().end());
    }
    Require(keys.size() == 22 && keys.count(0) == 0,
            "BERT-large projection requires more than 22 direct rotation keys");
    for (std::size_t baby = 1; baby < 16; ++baby)
        Require(keys.count(static_cast<int>(128 * baby)) == 1,
                "BERT-large baby rotation key is missing");
    std::cout << "HERA_BERT_LARGE_QKV_BABY_KS=120\n"
              << "HERA_BERT_LARGE_QKV_GIANT_KS=84\n"
              << "HERA_BERT_LARGE_QKV_TOTAL_KS=204\n";

    layer.model_width = 768;
    layer.q_heads = layer.kv_heads = 12;
    for (const auto layout : {hera::Layout::kMultiHeadQ, hera::Layout::kMultiHeadK,
                             hera::Layout::kMultiHeadV}) {
        auto decision = hera::MultiHeadPackingPlan::ProjectionDecision(layer, layout, slots);
        Require(decision.ok(), decision.status().message());
        const bool is_v = layout == hera::Layout::kMultiHeadV;
        Require(decision.value().heads_per_tile == (is_v ? 12 : 8) &&
                    decision.value().head_groups == (is_v
                        ? std::vector<std::size_t>{12}
                        : std::vector<std::size_t>{8, 4}) &&
                    decision.value().target == (is_v
                        ? hera::ProjectionPackingTarget::kAttentionValue
                        : hera::ProjectionPackingTarget::kQk),
                "12-head W*E preprocessing used the wrong downstream policy");
        auto output = hera::MultiHeadPackingPlan::ProjectionOutput(layer, layout, slots);
        Require(output.ok() && output.value().ciphertext_count == (is_v ? 4 : 3),
                "12-head Q/K/V ciphertext counts disagree with their policies");
        if (!is_v) {
            const auto &p = output.value();
            Require(p.Locations(4 * 64).front().ciphertext == 0 &&
                        p.Locations(4 * 64).front().slot == 4 * 128 &&
                        p.Locations(16).front().component == hera::ComplexLane::kImaginary &&
                        p.Locations(32).front().ciphertext == 1 &&
                        p.Locations(32).front().component == hera::ComplexLane::kReal &&
                        p.Locations(8 * 64).front().ciphertext == 2 &&
                        p.Locations(8 * 64).front().slot == 0 &&
                        p.Locations(8 * 64 + 32).front().component == hera::ComplexLane::kImaginary,
                    "Q/K did not compile columns into 8+4 complex groups");
            std::vector<unsigned char> occupied(p.ciphertext_count * slots * 2, 0);
            for (const auto &locations : p.logical_locations)
                for (const auto &location : locations)
                    occupied[(location.ciphertext * slots + location.slot) * 2 +
                             (location.component == hera::ComplexLane::kImaginary)] = 1;
            Require(std::all_of(occupied.begin(), occupied.end(), [](auto v) { return v == 1; }),
                    "BERT-base Q/K preprocessing left a slot component empty");
        }
    }
    auto base_plan = hera::ComplexDiagonalProjectionPlanner::ColumnBsgs(layer, slots);
    Require(base_plan.ok() && base_plan.value().output_ciphertexts == 10 &&
                base_plan.value().baby_steps == 16 && base_plan.value().giant_steps == 8 &&
                base_plan.value().rotation_upper_bound == 160,
            "QKV BSGS did not account for different Q/K and V head groups");
    auto base_qk = hera::MultiHeadQkKernel::Plan(layer, slots);
    Require(base_qk.ok() && base_qk.value().head_tiles.size() == 2,
            "QK did not plan both mixed-size head tiles");
    const auto &first = base_qk.value().head_tiles[0];
    const auto &last = base_qk.value().head_tiles[1];
    Require(first.head_begin == 0 && first.heads_per_tile == 8 &&
                first.input_groups == 2 && last.head_begin == 8 &&
                last.heads_per_tile == 4 && last.input_groups == 1,
            "QK feature partitions differ from the Wq/Wk destinations");
    auto scores = hera::MultiHeadPackingPlan::AttentionDiagonal(layer, slots);
    Require(scores.ok() && scores.value().ciphertext_count == 6 &&
                scores.value().Locations(8 * 128 * 128).front().ciphertext == 4 &&
                scores.value().Locations(8 * 128 * 128).front().slot == 0,
            "attention scores did not retain the 8+4 head ordering");
    std::cout << "HERA_BERT_BASE_QK_PLANNED_KS="
              << base_qk.value().rotations + base_qk.value().conjugations +
                     base_qk.value().relinearizations << '\n';
    std::cout << "HERA_BERT_BASE_QK_HEAD_GROUPS=8,4\n"
              << "HERA_BERT_BASE_V_HEAD_GROUP=12\n"
              << "HERA_BERT_BASE_QKV_TOTAL_KS=160\n";
    hera::ConfigureProjectionRotationHoisting(true);
}

void TestRealInputComplexProjection(bool padded_tokens = false, bool split_heads = false)
{
    auto config = MakeConfig();
    auto &layer = config.layers.front();
    layer.tokens = split_heads ? 8 : (padded_tokens ? 3 : 4);
    layer.model_width = 300;
    layer.q_heads = split_heads ? 12 : 2;
    layer.kv_heads = split_heads ? 3 : 1;
    layer.qk_head_width = split_heads ? 64 : (padded_tokens ? 181 : 131);
    layer.value_head_width = split_heads ? 64 : 129;
    auto &profile = config.he_parameter_profiles.at("block2");
    profile.polynomial_degree = 2048;
    profile.coefficient_modulus_bits = {60, 60, 60};
    profile.declared_depth = 2;
    const auto packing = hera::MultiHeadPackingPlan::RealStrictColumns(
        {{layer.tokens, layer.model_width}}, profile.polynomial_degree / 2).value();
    Require(packing.ciphertext_count == (split_heads ? 3 : 2),
            "projection must span multiple real inputs");
    std::set<int> keys;
    for (auto layout : {hera::Layout::kMultiHeadQ, hera::Layout::kMultiHeadK,
                        hera::Layout::kMultiHeadV}) {
        auto steps = hera::ComplexDiagonalProjectionKernel::RequiredRotationSteps(
            layer, packing, layout);
        Require(steps.ok(), steps.status().message());
        keys.insert(steps.value().begin(), steps.value().end());
    }
    Require(keys.count(0) == 0, "real projection requested a conjugation key");
    hera::HeContextRegistry client;
    Require(client.CreateClientProfiles(config, false).ok(), "projection context failed");
    Require(client.CompileRotationKeys("block2", {keys.begin(), keys.end()}).ok(),
            "real projection key compilation failed");
    auto bundle = client.ExportPublicBundle();
    Require(bundle.ok(), bundle.status().message());
    hera::HeContextRegistry server;
    Require(server.CreateServerProfiles(config, bundle.value()).ok(),
            "projection server failed");
    const auto &client_context = client.Require("block2", hera::PartyRole::kClient);
    const auto &server_context = server.Require("block2", hera::PartyRole::kServer);
    std::vector<double> values(layer.tokens * layer.model_width);
    for (std::size_t i = 0; i < values.size(); ++i)
        values[i] = 0.02 * std::sin(0.1 * static_cast<double>(i));
    auto input = Encrypt(values, packing, client_context);
    hera::SealPlainParameterStore store;
    double maximum_error = 0;
    std::size_t total_rotations = 0;
    hera::Block2ExecutionReport shared_direct_report, shared_cached_report;
    hera::Block2ExecutionReport shared_final_report, cached_final_report;
    hera::ComplexDiagonalProjectionKernel shared_direct(
        layer, server_context, &shared_direct_report);
    hera::ComplexDiagonalProjectionKernel shared_cached(
        layer, server_context, &shared_cached_report);
    for (auto layout : {hera::Layout::kMultiHeadQ, hera::Layout::kMultiHeadK,
                        hera::Layout::kMultiHeadV}) {
        const bool is_q = layout == hera::Layout::kMultiHeadQ;
        const bool is_v = layout == hera::Layout::kMultiHeadV;
        const auto columns = is_q ? hera::QueryWidth(layer) :
            (is_v ? hera::ValueWidth(layer) : hera::KeyWidth(layer));
        const std::string name = is_q ? "Wq" : (is_v ? "Wv" : "Wk");
        std::vector<double> weights(layer.model_width * columns), bias(columns);
        for (std::size_t i = 0; i < weights.size(); ++i)
            weights[i] = (static_cast<int>((i * 7) % 23) - 11) * 0.002;
        for (std::size_t i = 0; i < bias.size(); ++i)
            bias[i] = (static_cast<int>(i % 11) - 5) * 0.003;
        hera::DenseMatrixParameter matrix{{name, layer.model_width, columns, true}, weights};
        hera::DenseVectorParameter bias_parameter{{"b" + name.substr(1), columns}, bias};
        auto expected = MatMul(values, layer.tokens, layer.model_width, weights, columns);
        const double factor = is_v ? 1.0 : 1.0 / std::sqrt(
            2.0 * (is_q ? static_cast<double>(layer.qk_head_width) : 1.0));
        for (std::size_t i = 0; i < expected.size(); ++i)
            expected[i] = (expected[i] + bias[i % columns]) * factor;
        hera::Block2ExecutionReport report;
        hera::ComplexDiagonalProjectionKernel direct(layer, server_context, &report);
        auto output = direct.Evaluate(input, matrix, bias_parameter, layout);
        Require(output.ok(), output.status().message());
        maximum_error = std::max(maximum_error,
            MaxError(Decrypt(output.value(), client_context), expected));
        Require(report.conjugations == 0 && report.ciphertext_multiplications == 0 &&
                    output.value().logical_spec.planned_level == 1,
                "real projection performed input extraction or used extra depth");
        Require(output.value().conjugated_complex == (layout == hera::Layout::kMultiHeadK),
                "projection omitted the K conjugation value convention");
        total_rotations += report.rotations;
        auto shared = shared_direct.Evaluate(input, matrix, bias_parameter, layout);
        Require(shared.ok(), shared.status().message());
        maximum_error = std::max(maximum_error,
            MaxError(Decrypt(shared.value(), client_context), expected));
        Require(MaxError(Decrypt(shared.value(), client_context),
                         Decrypt(output.value(), client_context)) < 1e-4,
                "shared baby rotations changed projection numerics");
        const auto &plan = output.value().physical_packing;
        if (split_heads)
            Require(plan.ciphertext_count == (is_v ? 4 : 3),
                    "real projection selected the wrong preprocessed W destinations");
        bool has_imaginary = false;
        for (const auto &locations : plan.logical_locations)
            for (const auto &location : locations)
                has_imaginary |= location.component == hera::ComplexLane::kImaginary;
        Require(has_imaginary, "complex weight test lacks imaginary output columns");
        // Check every GQA replica, not just the first logical location.
        seal::Decryptor decryptor(*client_context.context, *client_context.secret_key);
        seal::CKKSEncoder encoder(*client_context.context);
        for (std::size_t ct = 0; ct < output.value().ciphertexts.size(); ++ct) {
            seal::Plaintext plain;
            std::vector<std::complex<double>> slots;
            decryptor.decrypt(output.value().ciphertexts[ct], plain);
            encoder.decode(plain, slots);
            for (std::size_t logical = 0; logical < expected.size(); ++logical)
                for (const auto &location : plan.Locations(logical)) {
                    if (location.ciphertext != ct) continue;
                    const double actual = location.component == hera::ComplexLane::kReal
                        ? slots[location.slot].real() :
                            (output.value().conjugated_complex ? -1.0 : 1.0) * slots[location.slot].imag();
                    maximum_error = std::max(maximum_error,
                                            std::abs(actual - expected[logical]));
                }
        }
        if (!padded_tokens && !split_heads) {
            auto zero = matrix;
            std::fill(zero.row_major.begin(), zero.row_major.end(), 0.0);
            hera::Block2ExecutionReport constant_counts;
            hera::ComplexDiagonalProjectionKernel constant_projection(
                layer, server_context, &constant_counts);
            auto constant = constant_projection.Evaluate(input, zero, bias_parameter, layout);
            Require(constant.ok(), constant.status().message());
            std::vector<double> expected_bias(layer.tokens * columns);
            for (std::size_t i = 0; i < expected_bias.size(); ++i)
                expected_bias[i] = bias[i % columns] * factor;
            Require(MaxError(Decrypt(constant.value(), client_context), expected_bias) < 1e-6 &&
                        constant.value().logical_spec.planned_level == 1 &&
                        constant.value().conjugated_complex == (!is_q && !is_v),
                    "zero QKV projection changed bias, level or K convention");
            Require(constant_counts.rotations == 0 && constant_counts.conjugations == 0,
                    "zero QKV projection performed unnecessary key switching");
            auto unbiased = constant_projection.Evaluate(input, zero, std::nullopt, layout);
            Require(unbiased.ok() && MaxError(Decrypt(unbiased.value(), client_context),
                        std::vector<double>(expected_bias.size())) < 1e-6,
                    "zero QKV projection without bias did not emit zero");
        }
        store.BeginPlaintextPreparation();
        {
            hera::ModelPlaintextCacheScope scope(&store, layer.id);
            auto metadata = input;
            for (auto &ct : metadata.ciphertexts) {
                const auto id = ct.parms_id();
                const auto scale = ct.scale();
                ct = seal::Ciphertext();
                ct.parms_id() = id;
                ct.scale() = scale;
            }
            hera::ComplexDiagonalProjectionKernel prepare(layer, server_context);
            auto compiled = prepare.Evaluate(metadata, matrix, bias_parameter, layout);
            Require(compiled.ok(), compiled.status().message());
        }
        store.FreezePlaintextCache();
        {
            hera::ModelPlaintextCacheScope scope(&store, layer.id);
            hera::Block2ExecutionReport cached_report;
            hera::ComplexDiagonalProjectionKernel cached(layer, server_context, &cached_report, 2);
            auto result = cached.Evaluate(input, matrix, bias_parameter, layout);
            Require(result.ok(), result.status().message());
            maximum_error = std::max(maximum_error,
                MaxError(Decrypt(result.value(), client_context), expected));
            Require(cached_report.conjugations == 0 &&
                        cached_report.rotations == report.rotations &&
                        cached_report.rescales == report.rescales,
                    "prepared real projection changed extraction/operation counts");
            auto reused = shared_cached.Evaluate(input, matrix, bias_parameter, layout);
            Require(reused.ok(), reused.status().message());
            maximum_error = std::max(maximum_error,
                MaxError(Decrypt(reused.value(), client_context), expected));
            Require(store.prepared_projection_hits() == 2, "prepared projection cache missed");
            if (split_heads && is_q) {
                auto other_layer = layer;
                other_layer.q_heads = 6;
                other_layer.qk_head_width = 128; // Same W shape, different W*E.
                hera::ComplexDiagonalProjectionKernel other(other_layer, server_context);
                auto wrong = other.Evaluate(input, matrix, bias_parameter, layout);
                Require(!wrong.ok() && wrong.status().message().find(
                            "packing decision mismatch") != std::string::npos,
                        "prepared W*E from a different head policy was reused");
                // The first group still has eight heads; only the tail list
                // differs. Checking just heads_per_tile cannot reject this.
                auto source_plan = store.GetPreparedProjection(layer.id + "/block2/" + name);
                Require(source_plan.ok(), source_plan.status().message());
                auto stale_plan = std::make_shared<hera::PreparedProjectionPlan>(*source_plan.value());
                stale_plan->packing_decision->head_groups = {8, 2, 2};
                hera::SealPlainParameterStore stale_store;
                stale_store.BeginPlaintextPreparation();
                Require(stale_store.PutPreparedProjection(
                            layer.id + "/block2/" + name, stale_plan).ok(),
                        "cannot stage the stale mixed-head projection");
                stale_store.FreezePlaintextCache();
                hera::ModelPlaintextCacheScope stale_scope(&stale_store, layer.id);
                auto stale = cached.Evaluate(input, matrix, bias_parameter, layout);
                Require(!stale.ok() && stale.status().message().find(
                            "packing decision mismatch") != std::string::npos,
                        "prepared Wq with the same first group but stale tail was reused");
            }
        }
        shared_final_report = shared_direct_report;
        cached_final_report = shared_cached_report;
        if (is_v) {
            auto changed_values = values;
            for (std::size_t i = 0; i < changed_values.size(); ++i)
                changed_values[i] += (static_cast<int>(i % 7) - 3) * 0.005;
            auto changed_expected = MatMul(changed_values, layer.tokens,
                layer.model_width, weights, columns);
            for (std::size_t i = 0; i < changed_expected.size(); ++i)
                changed_expected[i] += bias[i % columns];
            auto changed_input = Encrypt(changed_values, packing, client_context);
            auto changed = shared_direct.Evaluate(
                changed_input, matrix, bias_parameter, layout);
            Require(changed.ok(), changed.status().message());
            maximum_error = std::max(maximum_error,
                MaxError(Decrypt(changed.value(), client_context), changed_expected));
            {
                hera::ModelPlaintextCacheScope scope(&store, layer.id);
                auto changed_cached = shared_cached.Evaluate(
                    changed_input, matrix, bias_parameter, layout);
                Require(changed_cached.ok(), changed_cached.status().message());
                maximum_error = std::max(maximum_error,
                    MaxError(Decrypt(changed_cached.value(), client_context), changed_expected));
            }
            seal::Evaluator evaluator(*server_context.context);
            seal::CKKSEncoder encoder(*server_context.context);
            // Keep the same buffers, level and scale while changing the source.
            // Cached baby ciphertexts and hoisted digits must both be refreshed.
            for (std::size_t ct = 0; ct < changed_input.ciphertexts.size(); ++ct) {
                std::vector<double> delta(packing.slot_count);
                for (const auto &locations : packing.logical_locations)
                    for (const auto &location : locations)
                        if (location.ciphertext == ct) delta[location.slot] = 0.01;
                seal::Plaintext plain;
                encoder.encode(delta, changed_input.ciphertexts[ct].parms_id(),
                    changed_input.ciphertexts[ct].scale(), plain);
                evaluator.add_plain_inplace(changed_input.ciphertexts[ct], plain);
            }
            for (auto &value : changed_values) value += 0.01;
            changed_expected = MatMul(changed_values, layer.tokens,
                layer.model_width, weights, columns);
            for (std::size_t i = 0; i < changed_expected.size(); ++i)
                changed_expected[i] += bias[i % columns];
            auto mutated = shared_direct.Evaluate(changed_input, matrix, bias_parameter, layout);
            Require(mutated.ok(), mutated.status().message());
            maximum_error = std::max(maximum_error,
                MaxError(Decrypt(mutated.value(), client_context), changed_expected));
            {
                hera::ModelPlaintextCacheScope scope(&store, layer.id);
                auto mutated_cached = shared_cached.Evaluate(changed_input, matrix, bias_parameter, layout);
                Require(mutated_cached.ok(), mutated_cached.status().message());
                maximum_error = std::max(maximum_error,
                    MaxError(Decrypt(mutated_cached.value(), client_context), changed_expected));
            }
            for (auto &ct : changed_input.ciphertexts)
                evaluator.mod_switch_to_next_inplace(ct);
            changed_input.logical_spec.planned_level = 1;
            auto switched = shared_direct.Evaluate(
                changed_input, matrix, bias_parameter, layout);
            Require(switched.ok() && switched.value().logical_spec.planned_level == 2,
                    "projection reused a rotation at the wrong modulus level");
            maximum_error = std::max(maximum_error,
                MaxError(Decrypt(switched.value(), client_context), changed_expected));
        }
        auto invalid = input;
        invalid.physical_packing = MakeInterleavedSpatial(input.logical_spec.shape,
                                                         packing.slot_count);
        Require(!direct.Evaluate(invalid, matrix, bias_parameter, layout).ok(),
                "old complex X packing was accepted");
        auto token_contiguous = hera::MultiHeadPackingPlan::DenseSpatialReal(
            input.logical_spec.shape, packing.slot_count).value();
        Require(!direct.Evaluate(Encrypt(values, token_contiguous, client_context),
                                 matrix, bias_parameter, layout).ok(),
                "projection silently accepted token-contiguous X");
    }
    Require(maximum_error < 1e-4, "real X times complex W differs from plaintext");
    Require(shared_final_report.rotations < total_rotations,
            "projection BSGS did not reuse its baby rotations");
    // Check sharing against independent evaluation, allowing the cost policy
    // to select different splits as hoisting and memory budgets change.
    Require(shared_final_report.rotations < total_rotations &&
                shared_final_report.conjugations == 0 &&
                cached_final_report.conjugations == 0 &&
                shared_final_report.rotations == cached_final_report.rotations &&
                shared_final_report.rescales == cached_final_report.rescales &&
                shared_final_report.ciphertext_plain_multiplications ==
                    cached_final_report.ciphertext_plain_multiplications,
            "shared QKV did not reuse baby rotations");
    std::cout << "HERA_QKV_INDEPENDENT_ROTATIONS=" << total_rotations << '\n'
              << "HERA_QKV_SHARED_ROTATIONS=" << shared_final_report.rotations << '\n'
              << "HERA_REAL_X_COMPLEX_W_MAX_ERROR=" << maximum_error << '\n'
              << "HERA_REAL_X_COMPLEX_W_ROTATIONS=" << total_rotations << '\n'
              << "HERA_REAL_X_COMPLEX_W_CONJUGATIONS=0\n";
}

void TestQkvParameterFusion()
{
    for (bool with_bias : {false, true}) {
        hera::SealPlainParameterStore store;
        hera::LayerWeightHandles handles;
        handles.layer_id = "fusion";
        handles.wq = {1, {"Wq", 3, 4, with_bias}};
        handles.wk = {2, {"Wk", 3, 2, with_bias}};
        handles.wv = {3, {"Wv", 3, 3, with_bias}};
        handles.plain_parameters = {
            {4, {"ln1.gamma", 3}}, {5, {"ln1.beta", 3}}};
        const std::vector<double> gamma{-1.25, 0.0, 0.75};
        const std::vector<double> beta{0.3, -0.2, 0.5};
        Require(store.PutVector(handles.plain_parameters[0], gamma).ok() &&
                    store.PutVector(handles.plain_parameters[1], beta).ok(),
                "cannot register fusion affine vectors");
        const hera::WeightHandle *matrices[] = {
            &handles.wq, &handles.wk, &handles.wv};
        const char *bias_names[] = {"bq", "bk", "bv"};
        std::vector<std::vector<double>> originals(3), original_biases(3);
        for (std::size_t index = 0; index < 3; ++index) {
            const auto &handle = *matrices[index];
            auto &values = originals[index];
            values.resize(handle.spec.rows * handle.spec.columns);
            for (std::size_t i = 0; i < values.size(); ++i)
                values[i] = (static_cast<int>(i % 7) - 3) * 0.11;
            // Leave Wv absent first, to check validation is atomic.
            if (index != 2)
                Require(store.PutMatrix(handle, values).ok(),
                        "cannot register fusion matrix");
            original_biases[index].resize(handle.spec.columns, 0.0);
            if (with_bias) {
                hera::PlainParameterHandle bias_handle{
                    6 + index, {bias_names[index], handle.spec.columns}};
                for (std::size_t j = 0; j < handle.spec.columns; ++j)
                    original_biases[index][j] = (j + 1) * 0.04;
                Require(store.PutVector(bias_handle, original_biases[index]).ok(),
                        "cannot register fusion bias");
                handles.plain_parameters.push_back(bias_handle);
            }
        }
        auto original_q = store.GetMatrixShared(handles.wq.id).value();
        Require(!store.GetGammaFusedLayerNorm(handles.wq.id).ok(),
                "unfused matrix was accepted");
        Require(!store.FuseLayerNormGammaIntoQkv(handles).ok() &&
                    store.GetMatrixShared(handles.wq.id).value() == original_q,
                "failed fusion partially replaced the model");
        Require(store.PutMatrix(handles.wv, originals[2]).ok(),
                "cannot register remaining fusion matrix");
        auto fused = store.FuseLayerNormGammaIntoQkv(handles);
        Require(fused.ok(), fused.status().message());
        for (std::size_t index = 0; index < 3; ++index) {
            const auto &handle = *matrices[index];
            const auto matrix = store.GetMatrixShared(handle.id).value();
            const auto tail = store.GetGammaFusedLayerNorm(handle.id).value();
            const std::vector<double> input{-0.3, 0.6, 0.2, 0.1, -0.7, 0.4};
            auto affine_input = input;
            for (std::size_t i = 0; i < input.size(); ++i)
                affine_input[i] = input[i] * gamma[i % 3] + beta[i % 3];
            auto expected = MatMul(affine_input, 2, 3, originals[index],
                                   handle.spec.columns);
            auto tail_input = input;
            for (std::size_t i = 0; i < input.size(); ++i)
                tail_input[i] = input[i] * tail->gamma.values[i % 3] +
                                tail->beta.values[i % 3];
            auto actual = MatMul(tail_input, 2, 3, matrix->row_major,
                                 handle.spec.columns);
            for (std::size_t i = 0; i < expected.size(); ++i) {
                expected[i] += original_biases[index][i % handle.spec.columns];
                actual[i] += original_biases[index][i % handle.spec.columns];
            }
            Require(MaxError(actual, expected) < 1e-14,
                    "fused projection differs from explicit LN affine + W");
        }
        const auto once = store.GetMatrixShared(handles.wq.id).value();
        Require(store.FuseLayerNormGammaIntoQkv(handles).ok() &&
                    store.GetMatrixShared(handles.wq.id).value() == once,
                "repeated preprocessing fused gamma twice");
        auto conflicting = handles;
        conflicting.plain_parameters[0].id = 20;
        Require(store.PutVector(conflicting.plain_parameters[0], gamma).ok(),
                "cannot register conflicting gamma");
        Require(!store.FuseLayerNormGammaIntoQkv(conflicting).ok(),
                "conflicting fusion parameters were accepted");
        if (with_bias)
            for (std::size_t index = 0; index < 3; ++index)
                Require(store.GetVector(6 + index).value().values ==
                            original_biases[index],
                        "gamma fusion modified the original projection bias");
    }
    hera::SealPlainParameterStore late_store;
    auto layer = MakeConfig().layers.front();
    std::vector<double> wq, wk, wv;
    auto weights = RegisterWeights(layer, late_store, wq, wk, wv, false);
    late_store.BeginPlaintextPreparation();
    Require(!late_store.FuseLayerNormGammaIntoQkv(weights).ok(),
            "fusion after plaintext preparation started was accepted");
    late_store.AbortPlaintextPreparation();
    Require(late_store.FuseLayerNormGammaIntoQkv(weights).ok(),
            "offline fusion after aborted preparation failed");
}

void TestRealLayerNormTail()
{
    auto config = MakeConfig();
    auto &layer = config.layers.front();
    layer.tokens = 3;
    layer.model_width = 769;
    auto &profile = config.he_parameter_profiles.at("block2");
    profile.polynomial_degree = 2048;
    profile.coefficient_modulus_bits = {60, 60, 60};
    profile.declared_depth = 1;
    profile.input_scale_bits = profile.default_scale_bits = 40;
    profile.output_scale_bits = 40;
    hera::HeContextRegistry client;
    auto created = client.CreateClientProfiles(config);
    Require(created.ok(), created.status().message());
    // This isolated tail requires no Galois keys at all.
    auto bundle = client.ExportPublicBundle();
    Require(bundle.ok(), bundle.status().message());
    hera::HeContextRegistry server;
    Require(server.CreateServerProfiles(config, bundle.value()).ok(),
            "real LN server context failed");
    const auto &client_context = client.Require("block2", hera::PartyRole::kClient);
    const auto &context = server.Require("block2", hera::PartyRole::kServer);
    Require(context.galois_keys.size() == 0 && !context.has_secret_key(),
            "real LN tail should need neither Galois keys nor a secret key");
    const auto packing = hera::MultiHeadPackingPlan::RealStrictColumns(
        {{layer.tokens, layer.model_width}}, context.slot_count()).value();
    Require(packing.ciphertext_count == 4 && packing.layout == hera::Layout::kStrictColumn,
            "real column LN input did not span the expected four ciphertexts");
    for (std::size_t logical = 0; logical < packing.logical_locations.size(); ++logical) {
        const auto &location = packing.Locations(logical).front();
        Require(location.component == hera::ComplexLane::kReal &&
                    location.ciphertext == (logical % layer.model_width) / 256 &&
                    location.slot == ((logical % layer.model_width) % 256) * 4 +
                                     logical / layer.model_width,
                "real-only LN packing used an imaginary lane or wrong slot");
    }
    std::vector<double> centered(layer.tokens * layer.model_width), inverse(centered.size());
    hera::GammaFusedLayerNormParameters affine{
        {{"ln1.gamma", layer.model_width}, std::vector<double>(layer.model_width)},
        {{"ln1.beta", layer.model_width}, std::vector<double>(layer.model_width)}};
    for (std::size_t feature = 0; feature < layer.model_width; ++feature) {
        affine.gamma.values[feature] = feature % 5 == 0 ? 0.0 : 1.0;
        affine.beta.values[feature] = (static_cast<int>(feature % 7) - 3) * 0.01;
    }
    std::vector<double> expected(centered.size());
    for (std::size_t logical = 0; logical < centered.size(); ++logical) {
        const auto feature = logical % layer.model_width;
        centered[logical] = std::sin(logical * 0.07) * 0.2;
        inverse[logical] = 0.8 + (logical / layer.model_width) * 0.15;
        expected[logical] = centered[logical] * inverse[logical] *
            affine.gamma.values[feature] + affine.beta.values[feature];
    }
    auto centered_ct = Encrypt(centered, packing, client_context);
    auto inverse_ct = Encrypt(inverse, packing, client_context);
    hera::SealCipherTensorStore view_store;
    Require(view_store.Put(71, centered_ct).ok(), "store input failed");
    auto view = view_store.GetShared(71);
    Require(view.ok(), view.status().message());
    const auto data = view.value()->ciphertexts.front().data();
    Require(view_store.Rename(71, 72).ok() && !view_store.Get(71).ok(),
            "store rename did not move its identifier");
    Require(view_store.GetShared(72).value()->ciphertexts.front().data() == data,
            "store rename copied ciphertext coefficients");
    Require(view_store.Put(73, inverse_ct).ok() && !view_store.Rename(72, 73).ok() &&
                view_store.GetShared(72).ok(), "store rename collision lost its source");
    view_store.Erase(72);
    Require(view.value()->ciphertexts.front().data() == data &&
                MaxError(Decrypt(*view.value(), client_context), centered) < 1e-4,
            "erasing a store entry invalidated a live immutable view");
    hera::Block2ExecutionReport report;
    auto output = hera::LayerNormTailKernel::Evaluate(
        centered_ct, inverse_ct, affine, layer, context, &report);
    Require(output.ok(), output.status().message());
    const double error = MaxError(Decrypt(output.value(), client_context), expected);
    auto broadcast_plan = hera::MultiHeadPackingPlan::TokenInverseBroadcast(layer.tokens, context.slot_count());
    Require(broadcast_plan.ok(), broadcast_plan.status().message());
    std::vector<double> token_inverse(layer.tokens);
    for (std::size_t t = 0; t < layer.tokens; ++t) token_inverse[t] = inverse[t * layer.model_width];
    auto broadcast_ct = Encrypt(token_inverse, broadcast_plan.value(), client_context);
    hera::Block2ExecutionReport broadcast_report;
    auto broadcast_output = hera::LayerNormTailKernel::Evaluate(centered_ct, broadcast_ct,
        affine, layer, context, &broadcast_report, nullptr, nullptr, 2);
    Require(broadcast_output.ok(), broadcast_output.status().message());
    Require(broadcast_ct.ciphertexts.size() == 1 &&
                MaxError(Decrypt(broadcast_output.value(), client_context), expected) < 5e-3 &&
                broadcast_report.ciphertext_multiplications == report.ciphertext_multiplications &&
                broadcast_output.value().logical_spec.planned_level == output.value().logical_spec.planned_level,
            "single inverse broadcast changed LN1 values, multiplication count or depth");
    auto original = affine;
    std::vector<double> residual_expected(centered.size());
    for (std::size_t f = 0; f < layer.model_width; ++f) {
        original.gamma.values[f] = f % 17 == 0 ? 0.0 : (f % 2 ? -1.3 : 0.7);
        original.beta.values[f] = (static_cast<int>(f % 9) - 4) * 0.03;
    }
    for (std::size_t i = 0; i < centered.size(); ++i) {
        const auto f = i % layer.model_width;
        residual_expected[i] = centered[i] * inverse[i] * original.gamma.values[f] + original.beta.values[f];
    }
    hera::SealCipherTensor residual;
    hera::Block2ExecutionReport dual_report;
    auto dual = hera::LayerNormTailKernel::Evaluate(
        centered_ct, inverse_ct, affine, layer, context, &dual_report, &original, &residual);
    Require(dual.ok(), dual.status().message());
    Require(MaxError(Decrypt(dual.value(), client_context), expected) < 5e-3 &&
                MaxError(Decrypt(residual, client_context), residual_expected) < 5e-3 &&
                residual.logical_spec.planned_level == output.value().logical_spec.planned_level &&
                dual_report.ciphertext_multiplications == report.ciphertext_multiplications &&
                dual_report.relinearizations == report.relinearizations,
            "LN1 original-affine residual changed projection, repeated product, or added depth");
    hera::SealPlainParameterStore dual_store;
    dual_store.BeginPlaintextPreparation();
    {
        hera::ModelPlaintextCacheScope scope(&dual_store, "dual-ln1");
        auto ready = hera::LayerNormTailKernel::Evaluate(centered_ct, inverse_ct, affine,
            layer, context, nullptr, &original, &residual);
        Require(ready.ok(), ready.status().message());
    }
    dual_store.FreezePlaintextCache();
    {
        hera::ModelPlaintextCacheScope scope(&dual_store, "dual-ln1");
        auto ready = hera::LayerNormTailKernel::Evaluate(centered_ct, inverse_ct, affine,
            layer, context, nullptr, &original, &residual);
        Require(ready.ok() && MaxError(Decrypt(residual, client_context), residual_expected) < 5e-3,
                "prepared LN1 residual differs from original affine oracle");
    }
    Require(error < 5e-3 && output.value().logical_spec.planned_level == 1 &&
                report.ciphertext_multiplications == 4 &&
                report.ciphertext_plain_multiplications == 4 &&
                report.relinearizations == 4 && report.rescales == 4 &&
                report.conjugations == 0 && report.rotations == 0,
            "real LN tail changed values, levels or expected operation counts");
    Require(hera::SamePreparedPacking(output.value().physical_packing, packing),
            "real LN output did not preserve its real-only packing");
    double imaginary_error = 0.0, padding_error = 0.0;
    std::vector<std::vector<bool>> occupied(packing.ciphertext_count,
                                          std::vector<bool>(packing.slot_count, false));
    for (const auto &locations : packing.logical_locations)
        for (const auto &location : locations)
            occupied[location.ciphertext][location.slot] = true;
    seal::Decryptor decryptor(*client_context.context, *client_context.secret_key);
    seal::CKKSEncoder encoder(*client_context.context);
    for (std::size_t ct = 0; ct < output.value().ciphertexts.size(); ++ct) {
        seal::Plaintext plain;
        decryptor.decrypt(output.value().ciphertexts[ct], plain);
        std::vector<std::complex<double>> decoded;
        encoder.decode(plain, decoded);
        for (std::size_t slot = 0; slot < decoded.size(); ++slot) {
            imaginary_error = std::max(imaginary_error, std::abs(decoded[slot].imag()));
            if (!occupied[ct][slot])
                padding_error = std::max(padding_error, std::abs(decoded[slot]));
        }
    }
    Require(imaginary_error < 5e-3 && padding_error < 5e-3,
            "real LN output populated imaginary lanes or padding");
    auto complex_packing = MakeInterleavedSpatial(
        {{layer.tokens, layer.model_width}}, context.slot_count());
    auto rejected = hera::LayerNormTailKernel::Evaluate(
        Encrypt(centered, complex_packing, client_context),
        Encrypt(inverse, complex_packing, client_context), affine, layer, context);
    Require(!rejected.ok(), "real LN silently accepted imaginary-lane input");
    const auto token_contiguous = hera::MultiHeadPackingPlan::DenseSpatialReal(
        {{layer.tokens, layer.model_width}}, context.slot_count()).value();
    auto wrong_order = hera::LayerNormTailKernel::Evaluate(
        Encrypt(centered, token_contiguous, client_context),
        Encrypt(inverse, token_contiguous, client_context), affine, layer, context);
    Require(!wrong_order.ok(), "LN tail silently accepted token-contiguous input");
    hera::SealPlainParameterStore cache;
    cache.BeginPlaintextPreparation();
    {
        hera::ModelPlaintextCacheScope scope(&cache, layer.id);
        auto meta_centered = centered_ct, meta_inverse = inverse_ct;
        for (auto *tensor : {&meta_centered, &meta_inverse})
            for (auto &ct : tensor->ciphertexts) {
                const auto id = ct.parms_id();
                const auto scale = ct.scale();
                ct = seal::Ciphertext();
                ct.parms_id() = id;
                ct.scale() = scale;
            }
        hera::Block2ExecutionReport preparation_report;
        auto prepared = hera::LayerNormTailKernel::Evaluate(
            meta_centered, meta_inverse, affine, layer, context, &preparation_report);
        Require(prepared.ok(), prepared.status().message());
        Require(preparation_report.ciphertext_multiplications == 0 &&
                    preparation_report.rescales == 0 &&
                    preparation_report.conjugations == 0,
                "real LN preparation performed HE arithmetic");
    }
    cache.FreezePlaintextCache();
    const auto encoded_count = cache.encoded_plaintext_count();
    {
        hera::ModelPlaintextCacheScope scope(&cache, layer.id);
        auto cached = hera::LayerNormTailKernel::Evaluate(
            centered_ct, inverse_ct, affine, layer, context);
        Require(cached.ok(), cached.status().message());
        Require(MaxError(Decrypt(cached.value(), client_context),
                         Decrypt(output.value(), client_context)) < 1e-4 &&
                    cache.encoded_plaintext_count() == encoded_count,
                "real LN frozen cache changed the result or encoded more plaintexts");
    }
    // Aborting replacement preparation keeps the active immutable program.
    cache.BeginPlaintextPreparation();
    auto replacement = affine;
    replacement.beta.values.front() += 0.1;
    {
        hera::ModelPlaintextCacheScope scope(&cache, layer.id);
        auto staged = hera::LayerNormTailKernel::Evaluate(
            centered_ct, inverse_ct, replacement, layer, context);
        Require(staged.ok(), staged.status().message());
    }
    cache.AbortPlaintextPreparation();
    {
        hera::ModelPlaintextCacheScope scope(&cache, layer.id);
        Require(!hera::LayerNormTailKernel::Evaluate(
                    centered_ct, inverse_ct, replacement, layer, context).ok(),
                "LN reused a frozen program with changed affine parameters");
        auto restored = hera::LayerNormTailKernel::Evaluate(
            centered_ct, inverse_ct, affine, layer, context);
        Require(restored.ok() && MaxError(Decrypt(restored.value(), client_context),
                    Decrypt(output.value(), client_context)) < 1e-4 &&
                    cache.encoded_plaintext_count() == encoded_count,
                "aborted replacement lost the active LN program");
    }
    auto scaled_centered = centered_ct, scaled_inverse = inverse_ct;
    scaled_centered.ciphertexts.front().scale() *= 1.0009;
    scaled_inverse.ciphertexts.front().scale() *= 1.0009;
    auto centered_values = Decrypt(scaled_centered, client_context);
    auto inverse_values = Decrypt(scaled_inverse, client_context);
    std::vector<double> scaled_expected(expected.size());
    for (std::size_t logical = 0; logical < scaled_expected.size(); ++logical)
        scaled_expected[logical] = centered_values[logical] * inverse_values[logical] *
            affine.gamma.values[logical % layer.model_width] + affine.beta.values[logical % layer.model_width];
    auto scaled_output = hera::LayerNormTailKernel::Evaluate(
        scaled_centered, scaled_inverse, affine, layer, context);
    Require(scaled_output.ok() && MaxError(Decrypt(scaled_output.value(), client_context),
                scaled_expected) < 5e-3, "LN compilation lost per-ciphertext scale compensation");
    auto zero_gain = affine;
    std::fill(zero_gain.gamma.values.begin(), zero_gain.gamma.values.end(), 0.0);
    auto constant = hera::LayerNormTailKernel::Evaluate(
        centered_ct, inverse_ct, zero_gain, layer, context);
    Require(constant.ok(), constant.status().message());
    for (std::size_t logical = 0; logical < expected.size(); ++logical)
        expected[logical] = zero_gain.beta.values[logical % layer.model_width];
    Require(MaxError(Decrypt(constant.value(), client_context), expected) < 1e-4,
            "real LN zero gains did not preserve beta");
    std::cout << "HERA_REAL_LN_TAIL_MAX_ERROR=" << error << '\n'
              << "HERA_REAL_LN_TAIL_IMAG_MAX_ERROR=" << imaginary_error << '\n'
              << "HERA_REAL_LN_TAIL_PADDING_MAX_ERROR=" << padding_error << '\n'
              << "HERA_REAL_LN_TAIL_STATUS=PASS\n";
}

void TestBlock2(bool qkv_bias, bool mixed_gamma = false)
{
    auto config = MakeConfig();
    config.layers.front().qkv_bias = qkv_bias;
    Require(hera::ValidateHeraConfig(config).ok(), "test config is invalid");
    const auto &layer = config.layers.front();
    const auto &profile = config.he_parameter_profiles.at("block2");
    auto short_profile = profile;
    short_profile.coefficient_modulus_bits.resize(6);
    short_profile.declared_depth = 5;
    Require(!hera::Block2LevelPlanner::Validate(short_profile).ok(),
            "short Block-2 modulus chain was accepted");
    TestPackingVariants(layer, profile.polynomial_degree / 2);

    hera::HeContextRegistry client;
    Require(client.CreateClientProfiles(config).ok(),
            "cannot create client HE context");
    const auto packing = hera::MultiHeadPackingPlan::RealStrictColumns(
        {{layer.tokens, layer.model_width}}, profile.polynomial_degree / 2).value();
    auto steps = hera::SealBlock2Executor::RequiredRotationSteps(
        layer, profile, packing);
    Require(steps.ok(), steps.status().message());
    Require(std::any_of(steps.value().begin(), steps.value().end(), [](int step) {
                const auto magnitude = static_cast<unsigned>(std::abs(step));
                return magnitude != 0 && (magnitude & (magnitude - 1)) != 0;
            }),
            "exact non-power-of-two projection rotation key is absent");
    Require(client.CompileRotationKeys("block2", steps.value()).ok(),
            "cannot compile Block 2 Galois keys");
    auto bundle = client.ExportPublicBundle();
    Require(bundle.ok(), bundle.status().message());
    hera::HeContextRegistry server;
    Require(server.CreateServerProfiles(config, bundle.value()).ok(),
            "cannot create server HE context");
    Require(!server.Require("block2", hera::PartyRole::kServer).has_secret_key(),
            "server unexpectedly owns the secret key");

    const auto &client_context =
        client.Require("block2", hera::PartyRole::kClient);
    const std::vector<double> centered{-0.35, 0.20, 0.10, -0.15,
                                        0.30, -0.25, 0.05, 0.40};
    const std::vector<double> inverse{1.20, 1.20, 1.20, 1.20,
                                      0.85, 0.85, 0.85, 0.85};
    auto centered_ct = Encrypt(centered, packing, client_context);
    auto inverse_ct = Encrypt(inverse, packing, client_context);

    hera::SealCipherTensorStore tensors;
    auto centered_handle = tensors.Allocate(std::move(centered_ct));
    auto inverse_handle = tensors.Allocate(std::move(inverse_ct));
    Require(centered_handle.ok() && inverse_handle.ok(),
            "cannot register Block 2 inputs");
    hera::SealPlainParameterStore parameters;
    std::vector<double> wq, wk, wv;
    const std::vector<double> gamma_override = mixed_gamma
        ? std::vector<double>{0.0, -0.75, 1.25, 0.0} : std::vector<double>{};
    auto weights = RegisterWeights(layer, parameters, wq, wk, wv, true,
                                   gamma_override);
    const auto &server_context =
        server.Require("block2", hera::PartyRole::kServer);
    const auto affine = parameters.GetGammaFusedLayerNorm(weights.wq.id).value();
    hera::Block2ExecutionReport normalization_report;
    auto normalization = hera::LayerNormTailKernel::Evaluate(
        tensors.Get(centered_handle.value().id).value(),
        tensors.Get(inverse_handle.value().id).value(), *affine, layer, server_context,
        &normalization_report);
    Require(normalization.ok(), normalization.status().message());
    std::vector<double> expected_normalization(centered.size());
    for (std::size_t i = 0; i < centered.size(); ++i)
        expected_normalization[i] = centered[i] * inverse[i] *
            affine->gamma.values[i % layer.model_width] +
            affine->beta.values[i % layer.model_width];
    const auto normalization_error = MaxError(
        Decrypt(normalization.value(), client_context), expected_normalization);
    std::cout << "HERA_BLOCK2_NORMALIZATION_MAX_ERROR="
              << normalization_error << '\n';
    // This profile drops a 60-bit prime with 44-bit inputs: the lane masks
    // have only ~16 scale bits. Use the existing Block-2 absolute error
    // budget here; final V/scores and cache-equivalence bounds stay unchanged.
    Require(normalization_error < 5e-3,
            "LayerNorm dynamic normalization exceeds the error budget");
    Require(normalization.value().logical_spec.planned_level == 1 &&
                normalization_report.ciphertext_multiplications == 1 &&
                normalization_report.ciphertext_plain_multiplications == 1 &&
                normalization_report.rescales == 1 &&
                normalization_report.conjugations == 0,
            "LayerNorm tail changed its multiplication counts or depth");
    auto zero_gain = *affine;
    std::fill(zero_gain.gamma.values.begin(), zero_gain.gamma.values.end(), 0.0);
    auto constant = hera::LayerNormTailKernel::Evaluate(
        tensors.Get(centered_handle.value().id).value(),
        tensors.Get(inverse_handle.value().id).value(), zero_gain, layer,
        server_context);
    Require(constant.ok(), constant.status().message());
    std::vector<double> expected_constant(centered.size());
    for (std::size_t i = 0; i < expected_constant.size(); ++i)
        expected_constant[i] = zero_gain.beta.values[i % layer.model_width];
    Require(MaxError(Decrypt(constant.value(), client_context), expected_constant) < 1e-4,
            "zero gamma did not produce the LayerNorm beta constant");
    hera::SealBlock2Executor executor(config, layer, server, tensors,
                                       parameters);
    hera::Block2ExecutionReport report;
    auto output = executor.Evaluate(
        {centered_handle.value(), inverse_handle.value()}, weights, &report);
    Require(output.ok(), output.status().message());

    auto value_payload = tensors.Get(output.value().value.id);
    auto score_payload = tensors.Get(output.value().scaled_attention_scores.id);
    Require(value_payload.ok() && score_payload.ok(),
            "Block 2 outputs are absent from the tensor store");
    auto actual_v = Decrypt(value_payload.value(), client_context);
    auto actual_scores = Decrypt(score_payload.value(), client_context);

    std::vector<double> normalized(centered.size());
    for (std::size_t i = 0; i < normalized.size(); ++i) {
        const auto feature = i % layer.model_width;
        const double gamma = mixed_gamma ? gamma_override[feature]
            : 0.9 + 0.05 * static_cast<double>(feature);
        const double beta = -0.02 + 0.01 * static_cast<double>(feature);
        normalized[i] = centered[i] * inverse[i] * gamma + beta;
    }
    auto expected_q = MatMul(normalized, layer.tokens, layer.model_width,
                            wq, hera::QueryWidth(layer));
    for (std::size_t row = 0; row < layer.tokens; ++row)
        for (std::size_t column = 0; column < hera::QueryWidth(layer);
             ++column) {
            expected_q[row * hera::QueryWidth(layer) + column] +=
                (qkv_bias ? (static_cast<int>(column % 3) - 1) * 0.004 : 0.0);
            expected_q[row * hera::QueryWidth(layer) + column] /=
                std::sqrt(static_cast<double>(layer.qk_head_width));
        }
    auto expected_k = MatMul(normalized, layer.tokens, layer.model_width,
                            wk, hera::KeyWidth(layer));
    auto expected_v = MatMul(normalized, layer.tokens, layer.model_width,
                            wv, hera::ValueWidth(layer));
    for (std::size_t row = 0; row < layer.tokens; ++row) {
        for (std::size_t column = 0; column < hera::KeyWidth(layer); ++column)
            expected_k[row * hera::KeyWidth(layer) + column] +=
                (qkv_bias ? (static_cast<int>(column % 3) - 1) * 0.004 : 0.0);
        for (std::size_t column = 0; column < hera::ValueWidth(layer); ++column)
            expected_v[row * hera::ValueWidth(layer) + column] +=
                (qkv_bias ? (static_cast<int>(column % 3) - 1) * 0.004 : 0.0);
    }
    std::vector<double> expected_scores(layer.q_heads * layer.tokens *
                                        layer.tokens);
    for (std::size_t head = 0; head < layer.q_heads; ++head)
        for (std::size_t row = 0; row < layer.tokens; ++row)
            for (std::size_t column = 0; column < layer.tokens; ++column)
                for (std::size_t d = 0; d < layer.qk_head_width; ++d) {
                    const auto q_index = row * hera::QueryWidth(layer) +
                                         head * layer.qk_head_width + d;
                    const auto kv_head = head /
                        (layer.q_heads / layer.kv_heads);
                    const auto k_index = column * hera::KeyWidth(layer) +
                                         kv_head * layer.qk_head_width + d;
                    expected_scores[(head * layer.tokens + row) *
                                    layer.tokens + column] +=
                        expected_q[q_index] * expected_k[k_index];
                }
    Require(MaxError(actual_v, expected_v) < 5e-3,
            "V projection exceeds CKKS error budget");
    Require(MaxError(actual_scores, expected_scores) < 2e-2,
            "QK scores exceed CKKS error budget");
    Require(output.value().value.spec.planned_level == 2,
            "V output has the wrong planned level");
    Require(output.value().scaled_attention_scores.spec.planned_level == 6,
            "attention scores have the wrong planned level");
    Require(ActualLevel(server_context,
                        value_payload.value().ciphertexts.front().parms_id()) == 2,
            "V ciphertext is not actually at level 2");
    Require(ActualLevel(server_context,
                        score_payload.value().ciphertexts.front().parms_id()) == 6,
            "score ciphertext is not actually at terminal level 6");
    Require(std::abs(std::log2(
                value_payload.value().ciphertexts.front().scale()) - 44.0) < 0.01 &&
            std::abs(std::log2(
                score_payload.value().ciphertexts.front().scale()) - 44.0) < 0.01,
            "Block-2 ciphertext scale does not match the profile");
    Require(report.ciphertext_multiplications != 0 &&
            report.ciphertext_plain_multiplications != 0 &&
            report.relinearizations != 0 && report.rescales != 0,
            "Block 2 operation report is incomplete");

    // Compile from metadata-only ciphertexts: Prepare must not encrypt zeros
    // or execute QK, rotations, or any other HE arithmetic.
    parameters.BeginPlaintextPreparation();
    {
        hera::ModelPlaintextCacheScope scope(&parameters, layer.id);
        auto meta_centered = tensors.Get(centered_handle.value().id);
        auto meta_inverse = tensors.Get(inverse_handle.value().id);
        Require(meta_centered.ok() && meta_inverse.ok(),
                "Block 2 metadata inputs are absent");
        for (auto *tensor : {&meta_centered.value(), &meta_inverse.value()})
            for (auto &ct : tensor->ciphertexts) {
                const auto id = ct.parms_id();
                const auto scale = ct.scale();
                ct = seal::Ciphertext();
                ct.parms_id() = id;
                ct.scale() = scale;
            }
        auto normalized_meta = hera::LayerNormTailKernel::Evaluate(
            meta_centered.value(), meta_inverse.value(), *affine, layer, server_context);
        Require(normalized_meta.ok(), normalized_meta.status().message());
        auto qk_meta = normalized_meta.value();
        const auto terminal = profile.coefficient_modulus_bits.size() -
                              profile.terminal_prime_count;
        const auto qk_input_level = terminal - hera::MultiHeadQkKernel::kDepth - 1;
        auto qk_context_data = server_context.context->first_context_data();
        for (std::size_t level = 0; level < qk_input_level; ++level)
            qk_context_data = qk_context_data->next_context_data();
        Require(static_cast<bool>(qk_context_data),
                "Block 2 test QK level is absent");
        for (auto &ct : qk_meta.ciphertexts)
            ct.parms_id() = qk_context_data->parms_id();
        qk_meta.logical_spec.planned_level = qk_input_level;
        hera::ComplexDiagonalProjectionKernel qk_projection(
            layer, server_context);
        hera::ComplexDiagonalProjectionKernel value_projection(
            layer, server_context);
        const auto bias = [&](std::string_view name)
            -> std::optional<hera::DenseVectorParameter> {
            if (!qkv_bias) return std::nullopt;
            for (const auto &handle : weights.plain_parameters)
                if (handle.spec.name == name) return parameters.GetVector(handle.id).value();
            throw std::runtime_error("projection bias missing");
        };
        auto q = qk_projection.Evaluate(
            qk_meta, parameters.GetMatrix(weights.wq.id).value(),
            bias("bq"),
            hera::Layout::kMultiHeadQ);
        auto k = qk_projection.Evaluate(
            qk_meta, parameters.GetMatrix(weights.wk.id).value(),
            bias("bk"),
            hera::Layout::kMultiHeadK);
        auto v = value_projection.Evaluate(
            normalized_meta.value(),
            parameters.GetMatrix(weights.wv.id).value(),
            bias("bv"),
            hera::Layout::kMultiHeadV);
        Require(q.ok() && k.ok() && v.ok(),
                q.ok() ? (k.ok() ? v.status().message()
                                 : k.status().message())
                       : q.status().message());
        auto qk_compiled = hera::MultiHeadQkKernel::Prepare(layer, server_context);
        Require(qk_compiled.ok(), qk_compiled.status().message());
    }
    parameters.FreezePlaintextCache();
    Require(parameters.prepared_projection_count() == 3,
            "Prepare did not compile Wq, Wk and Wv projection plans");
    hera::Block2ExecutionReport cached_report;
    hera::Result<hera::Block2HeOutputs> cached =
        hera::Status::Internal("cached Block 2 was not evaluated");
    {
        hera::ModelPlaintextCacheScope scope(&parameters, layer.id);
        cached = executor.Evaluate(
            {centered_handle.value(), inverse_handle.value()}, weights,
            &cached_report);
    }
    Require(cached.ok(), cached.status().message());
    Require(parameters.prepared_projection_hits() == 3,
            "online Block 2 did not use all prepared projection plans");
    auto cached_value = tensors.Get(cached.value().value.id);
    auto cached_scores = tensors.Get(cached.value().scaled_attention_scores.id);
    Require(cached_value.ok() && cached_scores.ok(),
            "cached Block 2 outputs are absent");
    Require(MaxError(Decrypt(cached_value.value(), client_context), actual_v) <
                1e-4 &&
            MaxError(Decrypt(cached_scores.value(), client_context),
                     actual_scores) < 1e-4,
            "prepared Block 2 differs from the direct HE path");
    Require(cached_report.rotations == report.rotations &&
                cached_report.ciphertext_plain_multiplications ==
                    report.ciphertext_plain_multiplications &&
                cached_report.rescales == report.rescales,
            "prepared Block 2 changed HE operation counts");
    std::cout << "HERA_BLOCK2_FUSED_QKV_BIAS=" << qkv_bias << '\n'
              << "HERA_BLOCK2_MIXED_GAMMA=" << mixed_gamma << '\n'
              << "HERA_BLOCK2_FUSED_VALUE_MAX_ERROR="
              << MaxError(actual_v, expected_v) << '\n'
              << "HERA_BLOCK2_FUSED_SCORE_MAX_ERROR="
              << MaxError(actual_scores, expected_scores) << '\n';
}

} // namespace

int main()
{
    try {
        TestColumnBsgsPlanning();
        TestRealLayerNormTail();
        TestRealInputComplexProjection();
        TestRealInputComplexProjection(true);
        TestRealInputComplexProjection(false, true);
        TestQkvParameterFusion();
        TestBlock2(true);
        TestBlock2(false);
        TestBlock2(true, true);
        TestPowerOfTwoValueProjection();
        TestParallelQkGroups();
        TestParallelQkGroups(true);
        TestParallelQkGroups(false, true);
        std::cout << "HERA_BLOCK2_HE_STATUS=PASS\n";
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << "HERA_BLOCK2_HE_STATUS=FAIL\n"
                  << "HERA_BLOCK2_HE_ERROR=" << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
