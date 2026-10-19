#include "hera/ffn_gelu.h"
#include "hera/detail/level_schedule.h"
#include "hera/detail/parallel_ranges.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace {
using Clock = std::chrono::steady_clock;
using Tensor = hera::SealCipherTensor;
using Counts = hera::HeKernelOperationCounts;

void Require(bool condition, const std::string &message)
{
    if (!condition) throw std::runtime_error(message);
}

template <class T> T Take(hera::Result<T> result)
{
    Require(result.ok(), result.status().message());
    return result.take_value();
}

void Check(hera::Result<void> result)
{
    Require(result.ok(), result.status().message());
}

std::size_t Product(std::size_t a, std::size_t b)
{
    Require(b == 0 || a <= std::numeric_limits<std::size_t>::max() / b,
            "size overflow");
    return a * b;
}

struct Options {
    std::string op, projection{"wq"}, config{HERA_DEFAULT_CONFIG_PATH};
    std::size_t layer{}, threads{1}, warmup{1}, repeat{3};
    std::size_t memory_mib{512};
    bool virtual_weights{};
};

void Usage()
{
    std::cout << "Usage: hera_he_benchmark --op=qk|xw|av [--config=FILE]\n"
              << "  --projection=wq|wk|wv|wo|wu|wd  XW only; default wq\n"
              << "  --layer=0 --threads=1 --warmup=1 --repeat=3\n"
              << "  --weight-mode=materialized|virtual  XW only; default materialized\n"
              << "  --he-memory-mib=512\n"
              << "Dimensions, HE profiles and input levels follow the selected layer.\n"
              << "KS = rotations + conjugations + relinearizations, per invocation.\n";
}

Options Parse(int argc, char **argv)
{
    Options o;
    bool projection_set = false, weight_mode_set = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        const auto eq = arg.find('=');
        Require(arg.rfind("--", 0) == 0 && eq != std::string::npos,
                "expected --name=value: " + arg);
        const auto name = arg.substr(0, eq), value = arg.substr(eq + 1);
        Require(!value.empty(), "empty option: " + name);
        if (name == "--op") o.op = value;
        else if (name == "--config") o.config = value;
        else if (name == "--projection") { o.projection = value; projection_set = true; }
        else if (name == "--weight-mode") {
            Require(value == "materialized" || value == "virtual", "invalid weight mode");
            o.virtual_weights = value == "virtual";
            weight_mode_set = true;
        } else {
            std::size_t n{};
            const auto parsed = std::from_chars(value.data(), value.data() + value.size(), n);
            Require(parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size(),
                    "invalid integer for " + name);
            Require(n > 0 || name == "--warmup" || name == "--layer",
                    name + " must be positive");
            if (name == "--layer") o.layer = n;
            else if (name == "--threads") o.threads = n;
            else if (name == "--warmup") o.warmup = n;
            else if (name == "--repeat") o.repeat = n;
            else if (name == "--he-memory-mib") o.memory_mib = n;
            else throw std::runtime_error("unknown option: " + name);
        }
    }
    Require(o.op == "qk" || o.op == "xw" || o.op == "av", "--op must be qk, xw or av");
    const std::vector<std::string> projections{"wq", "wk", "wv", "wo", "wu", "wd"};
    Require(std::find(projections.begin(), projections.end(), o.projection) != projections.end(),
            "unknown XW projection");
    Require(o.op == "xw" || (!projection_set && !weight_mode_set),
            "--projection and --weight-mode require --op=xw");
    return o;
}

// Only instantiate the selected profile, without changing its primes or security level.
struct Plan {
    hera::HeraConfig config;
    hera::TransformerLayerConfig layer;
    std::string profile;
    hera::HeRegionId region{hera::HeRegionId::kRegion2};
    hera::Layout projection_layout{hera::Layout::kMultiHeadQ};
    hera::HePackingPlan first, second, output;
    std::vector<int> rotations;
    std::size_t level{}, depth{}, input_width{}, output_width{};
    int scale_bits{}, output_scale_bits{};
    bool bias{};
    double output_factor{1.0};
};

Plan MakePlan(const Options &o)
{
    Plan p;
    auto original = Take(hera::LoadHeraConfig(o.config));
    Check(hera::ValidateHeraConfig(original));
    Require(o.layer < original.layers.size(), "--layer index is out of range");
    p.layer = original.layers[o.layer];
    const auto &l = p.layer;
    const auto *execution = hera::FindLayerExecution(original, l.id);
    Require(execution != nullptr, "missing layer execution config");
    const bool shared_skip = l.norm == hera::NormPlacement::kNormalizedResidual &&
        execution->regions.at(hera::HeRegionId::kRegion1).parameter_profile ==
        execution->regions.at(hera::HeRegionId::kRegion5).parameter_profile;
    if (o.op == "xw" && o.projection == "wo") p.region = hera::HeRegionId::kRegion4;
    if (o.op == "xw" && o.projection == "wu") p.region = hera::HeRegionId::kRegion5;
    if (o.op == "xw" && o.projection == "wd")
        p.region = shared_skip ? hera::HeRegionId::kRegion5 : hera::HeRegionId::kRegion4;
    p.profile = execution->regions.at(p.region).parameter_profile;
    const auto profile = original.he_parameter_profiles.at(p.profile);
    const auto slots = profile.polynomial_degree / 2;
    const auto terminal = profile.coefficient_modulus_bits.size() - profile.terminal_prime_count;
    p.scale_bits = p.output_scale_bits = profile.default_scale_bits;
    p.depth = o.op == "qk" ? hera::MultiHeadQkKernel::kDepth : (o.op == "av" ? 3 : 1);
    Require(terminal >= p.depth, "profile has insufficient levels for this kernel");
    if (o.op == "qk") {
        p.level = terminal - p.depth;
        p.first = Take(hera::MultiHeadPackingPlan::ProjectionOutput(l, hera::Layout::kMultiHeadQ, slots));
        p.second = Take(hera::MultiHeadPackingPlan::ProjectionOutput(l, hera::Layout::kMultiHeadK, slots));
        p.rotations = Take(hera::MultiHeadQkKernel::RequiredRotationSteps(l, slots));
    } else if (o.op == "av") {
        p.level = hera::detail::AttentionValueInputLevel(profile);
        auto av = Take(hera::AttentionLn2PackingPlanner::AttentionValue(l, slots));
        p.first = std::move(av.probability_packing);
        p.rotations = std::move(av.required_rotation_steps);
        p.second = Take(hera::MultiHeadPackingPlan::ProjectionOutput(l, hera::Layout::kMultiHeadV, slots));
    } else if (o.projection == "wq" || o.projection == "wk" || o.projection == "wv") {
        Check(hera::Block2LevelPlanner::Validate(profile));
        p.level = o.projection == "wv" ? 1 : terminal - hera::MultiHeadQkKernel::kDepth - 1;
        p.input_width = l.model_width;
        p.projection_layout = o.projection == "wq" ? hera::Layout::kMultiHeadQ :
            (o.projection == "wk" ? hera::Layout::kMultiHeadK : hera::Layout::kMultiHeadV);
        p.output_width = o.projection == "wq" ? hera::QueryWidth(l) :
            (o.projection == "wk" ? hera::KeyWidth(l) : hera::ValueWidth(l));
        p.bias = l.qkv_bias;
        p.output_factor = o.projection == "wq" ? 1.0 / std::sqrt(2.0 * l.qk_head_width) :
            (o.projection == "wk" ? 1.0 / std::sqrt(2.0) : 1.0);
        p.first = Take(hera::MultiHeadPackingPlan::RealStrictColumns({{l.tokens, l.model_width}}, slots));
        p.rotations = Take(hera::ComplexDiagonalProjectionKernel::RequiredRotationSteps(l, p.first, p.projection_layout));
    } else if (o.projection == "wo") {
        Check(hera::AttentionLn2LevelPlanner::ValidateOutput(profile));
        p.level = hera::AttentionLn2LevelPlanner::OutputInputLevel(profile);
        p.input_width = hera::AttentionWidth(l);
        p.output_width = l.model_width;
        p.bias = l.output_bias;
        p.first = Take(hera::AttentionLn2PackingPlanner::AttentionValue(l, slots)).output_packing;
        p.rotations = Take(hera::AttentionOutputProjectionKernel::RequiredRealRotationSteps(l, slots));
    } else {
        const bool up = o.projection == "wu";
        if (up) Check(hera::FfnGeluLevelPlanner::Validate(profile));
        Require(terminal >= (up ? hera::kFfnSuffixDepth : 4), "profile has insufficient FFN suffix");
        p.level = up ? terminal - hera::kFfnSuffixDepth + 1 : terminal - 4;
        p.scale_bits = up ? profile.default_scale_bits + 15 :
            (shared_skip ? profile.output_scale_bits : profile.default_scale_bits);
        p.output_scale_bits = up || shared_skip ? profile.output_scale_bits : profile.default_scale_bits;
        p.input_width = up ? l.model_width : l.ffn_width;
        p.output_width = up ? l.ffn_width : l.model_width;
        p.bias = l.ffn_bias;
        p.first = Take(hera::FfnGeluPackingPlanner::ComplexColumns(
            {{l.tokens, p.input_width}}, slots, up ? hera::Layout::kComplexStrictColumn : hera::Layout::kFfn));
        p.output = Take(hera::FfnGeluPackingPlanner::RealColumns(
            {{l.tokens, p.output_width}}, slots, hera::Layout::kStrictColumn));
        p.rotations = Take(hera::ComplexColumnProjectionKernel::RequiredRotationSteps(p.first, p.output));
    }
    Require(p.level <= terminal - p.depth, "kernel input level exceeds its suffix");
    p.config = original;
    p.config.layers = {l};
    auto isolated_execution = *execution;
    for (auto &entry : isolated_execution.regions) entry.second.parameter_profile = p.profile;
    p.config.layer_execution = {std::move(isolated_execution)};
    p.config.he_parameter_profiles = {{p.profile, profile}};
    return p;
}

seal::parms_id_type LevelId(const hera::HeContext &context, std::size_t level)
{
    auto data = context.context->first_context_data();
    while (level-- && data) data = data->next_context_data();
    Require(static_cast<bool>(data), "requested HE level is absent");
    return data->parms_id();
}

Tensor Encrypt(const std::vector<double> &values, const hera::HePackingPlan &packing,
               const Plan &p, const hera::HeContext &context)
{
    Require(values.size() == packing.logical_locations.size(), "input/packing size mismatch");
    std::vector<std::vector<std::complex<double>>> slots(packing.ciphertext_count,
        std::vector<std::complex<double>>(packing.slot_count));
    const bool conjugated = packing.layout == hera::Layout::kMultiHeadK;
    for (std::size_t i = 0; i < values.size(); ++i)
        for (const auto &place : packing.Locations(i)) {
            auto &slot = slots.at(place.ciphertext).at(place.slot);
            if (place.component == hera::ComplexLane::kReal) slot.real(values[i]);
            else slot.imag(conjugated ? -values[i] : values[i]);
        }
    seal::CKKSEncoder encoder(*context.context);
    seal::Encryptor encryptor(*context.context, *context.secret_key);
    Tensor result;
    result.parameter_profile = context.profile;
    result.physical_packing = packing;
    result.conjugated_complex = conjugated;
    auto &spec = result.logical_spec;
    spec.shape = packing.logical_shape;
    spec.domain = hera::Domain::kHeCiphertext;
    spec.layout = packing.layout;
    spec.he_region = p.region;
    spec.he_parameter_profile = context.profile;
    spec.planned_level = p.level;
    spec.planned_scale_bits = p.scale_bits;
    result.ciphertexts.resize(packing.ciphertext_count);
    for (std::size_t ct = 0; ct < result.ciphertexts.size(); ++ct) {
        seal::Plaintext plain;
        encoder.encode(slots[ct], LevelId(context, p.level), std::ldexp(1.0, p.scale_bits), plain);
        encryptor.encrypt_symmetric(plain, result.ciphertexts[ct]);
    }
    Check(result.Validate(context));
    return result;
}

std::vector<double> Decrypt(const Tensor &tensor, const hera::HeContext &context)
{
    seal::Decryptor decryptor(*context.context, *context.secret_key);
    seal::CKKSEncoder encoder(*context.context);
    std::vector<std::vector<std::complex<double>>> slots(tensor.ciphertexts.size());
    for (std::size_t ct = 0; ct < slots.size(); ++ct) {
        seal::Plaintext plain;
        decryptor.decrypt(tensor.ciphertexts[ct], plain);
        encoder.decode(plain, slots[ct]);
    }
    std::vector<double> values(tensor.physical_packing.logical_locations.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        const auto &place = tensor.physical_packing.Locations(i).front();
        const auto slot = slots.at(place.ciphertext).at(place.slot);
        values[i] = place.component == hera::ComplexLane::kReal ? slot.real() :
            (tensor.conjugated_complex ? -slot.imag() : slot.imag());
        Require(std::isfinite(values[i]), "nonfinite output");
    }
    return values;
}

template <class Report> Counts Count(const Report &r)
{
    return {r.rotations, r.conjugations, r.ciphertext_plain_multiplications,
            r.ciphertext_multiplications, r.relinearizations, r.rescales, 0};
}

std::size_t Ks(const Counts &c) { return c.rotations + c.conjugations + c.relinearizations; }

bool SameCounts(const Counts &a, const Counts &b)
{
    return a.rotations == b.rotations && a.conjugations == b.conjugations &&
        a.relinearizations == b.relinearizations && a.rescales == b.rescales &&
        a.ciphertext_plain_multiplications == b.ciphertext_plain_multiplications &&
        a.ciphertext_multiplications == b.ciphertext_multiplications;
}

struct Fixture {
    std::vector<double> left, right;
    hera::DenseMatrixParameter weights;
    std::optional<hera::DenseVectorParameter> bias;
    Tensor first, second;
};

Fixture MakeFixture(const Options &o, const Plan &p, const hera::HeContext &client)
{
    Fixture f;
    const auto &l = p.layer;
    f.left.resize(p.first.logical_locations.size());
    f.right.resize(p.second.logical_locations.size());
    for (std::size_t i = 0; i < f.left.size(); ++i) f.left[i] = 0.1 * std::sin((i + 1) * 0.17);
    for (std::size_t i = 0; i < f.right.size(); ++i) f.right[i] = 0.1 * std::cos((i + 2) * 0.13);
    if (o.op == "av") {
        for (std::size_t row = 0; row < l.tokens; ++row)
            for (std::size_t h = 0; h < l.q_heads; ++h) {
                const auto offset = (row * l.q_heads + h) * l.tokens;
                double sum = 0;
                for (std::size_t col = 0; col < l.tokens; ++col)
                    sum += f.left[offset + col] = 1.0 + (row * 3 + h * 2 + col * 5) % 11;
                for (std::size_t col = 0; col < l.tokens; ++col) f.left[offset + col] /= sum;
            }
    }
    auto encoded_left = f.left, encoded_right = f.right;
    if (o.op == "qk") {
        for (auto &x : encoded_left) x /= std::sqrt(2.0 * l.qk_head_width);
        for (auto &x : encoded_right) x /= std::sqrt(2.0);
    }
    f.first = Encrypt(encoded_left, p.first, p, client);
    if (o.op != "xw") f.second = Encrypt(encoded_right, p.second, p, client);
    else {
        f.weights.spec = {o.projection, p.input_width, p.output_width, true};
        f.weights.row_major.resize(Product(p.input_width, p.output_width));
        const double factor = 0.1 / std::sqrt(static_cast<double>(p.input_width));
        for (std::size_t i = 0; i < f.weights.row_major.size(); ++i)
            f.weights.row_major[i] = factor * std::sin((i + 3) * 0.07);
        if (p.bias) {
            f.bias = hera::DenseVectorParameter{{"bias", p.output_width}, std::vector<double>(p.output_width)};
            for (std::size_t i = 0; i < p.output_width; ++i) f.bias->values[i] = 0.01 * std::cos(i * 0.11);
        }
    }
    return f;
}

double Reference(std::size_t index, const Options &o, const Plan &p, const Fixture &f)
{
    const auto &l = p.layer;
    double expected = 0;
    if (o.op == "qk") {
        const auto h = index / (l.tokens * l.tokens);
        const auto row = (index / l.tokens) % l.tokens, col = index % l.tokens;
        const auto kh = h / (l.q_heads / l.kv_heads);
        for (std::size_t d = 0; d < l.qk_head_width; ++d)
            expected += f.left[(row * l.q_heads + h) * l.qk_head_width + d] *
                        f.right[(col * l.kv_heads + kh) * l.qk_head_width + d];
        return expected / std::sqrt(static_cast<double>(l.qk_head_width));
    }
    if (o.op == "av") {
        const auto row = index / (l.q_heads * l.value_head_width);
        const auto h = (index / l.value_head_width) % l.q_heads, d = index % l.value_head_width;
        const auto kh = h / (l.q_heads / l.kv_heads);
        for (std::size_t col = 0; col < l.tokens; ++col)
            expected += f.left[(row * l.q_heads + h) * l.tokens + col] *
                        f.right[(col * l.kv_heads + kh) * l.value_head_width + d];
        return expected;
    }
    const auto row = index / p.output_width, col = index % p.output_width;
    expected = f.bias ? f.bias->values[col] : 0;
    for (std::size_t k = 0; k < p.input_width; ++k)
        expected += f.left[row * p.input_width + k] * f.weights.row_major[k * p.output_width + col];
    return expected * p.output_factor;
}

void Run(const Options &o)
{
    hera::detail::HeWorkerPool::Instance().SetMemoryBudget(Product(o.memory_mib, 1ULL << 20));
    const auto p = MakePlan(o);
    hera::HeContextRegistry client, server;
    Check(client.CreateClientProfiles(p.config, o.op != "xw"));
    Check(client.CompileRotationKeys(p.profile, p.rotations));
    {
        auto bundle = Take(client.ExportPublicBundle());
        Check(server.CreateServerProfiles(p.config, bundle));
    }
    Check(client.ReleaseClientEvaluationKeys());
    const auto &cc = client.Require(p.profile, hera::PartyRole::kClient);
    const auto &sc = server.Require(p.profile, hera::PartyRole::kServer);
    Require(!sc.has_secret_key(), "evaluator must not hold a secret key");
    const auto f = MakeFixture(o, p, cc);
    auto elapsed = [](Clock::time_point start) {
        return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    };
    hera::SealPlainParameterStore cache;
    if (o.virtual_weights) Check(cache.ConfigureVirtualWeightPlaintexts(1));
    auto evaluate = [&](Counts &counts) -> hera::Result<Tensor> {
        hera::ModelPlaintextCacheScope scope(&cache, p.layer.id);
        hera::Block2ExecutionReport b2;
        hera::AttentionLn2ExecutionReport attention;
        hera::FfnGeluExecutionReport ffn;
        hera::Result<Tensor> result = hera::Status::Internal("unselected kernel");
        if (o.op == "qk") {
            result = hera::MultiHeadQkKernel::Evaluate(f.first, f.second, p.layer, sc, &b2, o.threads);
            counts = Count(b2);
        } else if (o.op == "av") {
            result = hera::MultiHeadAttentionValueKernel::Evaluate(f.first, f.second, p.layer, sc, &attention, o.threads);
            counts = Count(attention);
        } else if (o.projection == "wq" || o.projection == "wk" || o.projection == "wv") {
            // Fresh per invocation: never reuse ciphertext baby rotations from an earlier run.
            hera::ComplexDiagonalProjectionKernel kernel(p.layer, sc, &b2, o.threads);
            result = kernel.Evaluate(f.first, f.weights, f.bias, p.projection_layout);
            counts = Count(b2);
        } else if (o.projection == "wo") {
            result = hera::AttentionOutputProjectionKernel::EvaluateReal(f.first, f.weights, f.bias, p.layer, sc, &attention, o.threads);
            counts = Count(attention);
        } else if (o.projection == "wu") {
            result = hera::FfnUpProjectionKernel::Evaluate(f.first, f.weights, f.bias, p.layer, sc, &ffn, o.threads);
            counts = Count(ffn);
        } else {
            result = hera::ComplexColumnProjectionKernel::Evaluate(f.first, f.weights, f.bias, p.output,
                p.region, p.output_scale_bits, sc, &counts, o.threads);
        }
        return result;
    };
    cache.BeginPlaintextPreparation();
    {
        hera::ModelPlaintextCacheScope scope(&cache, p.layer.id);
        if (o.op == "qk") Check(hera::MultiHeadQkKernel::Prepare(p.layer, sc));
        else if (o.op == "av") Check(hera::MultiHeadAttentionValueKernel::Prepare(p.layer, sc, p.level));
        else { Counts unused; Take(evaluate(unused)); }
    }
    cache.FreezePlaintextCache();
    const auto prepared_count = cache.encoded_plaintext_count();
    for (std::size_t i = 0; i < o.warmup; ++i) { Counts unused; Take(evaluate(unused)); }
    Counts operations;
    Tensor output;
    std::vector<double> samples;
    samples.reserve(o.repeat);
    for (std::size_t i = 0; i < o.repeat; ++i) {
        Counts counts;
        const auto start = Clock::now();
        auto result = evaluate(counts);
        const auto ms = elapsed(start);
        output = Take(std::move(result));
        if (i != 0) Require(SameCounts(counts, operations), "operation counts changed between repetitions");
        operations = counts;
        samples.push_back(ms);
    }
    Require(cache.encoded_plaintext_count() == prepared_count, "online evaluation encoded new plaintexts");
    Check(output.Validate(cc));
    Require(output.logical_spec.planned_level == p.level + p.depth, "unexpected output level");
    for (const auto &ct : output.ciphertexts)
        Require(ct.parms_id() == LevelId(cc, p.level + p.depth), "output parms_id differs from planned level");
    const auto actual = Decrypt(output, cc);
    std::size_t checked = 0;
    double error = 0;
    if (!o.virtual_weights) {
        // Full check for tiny cases; bounded, evenly spaced samples for large matrices.
        checked = std::min<std::size_t>(256, actual.size());
        Require(checked > 0, "empty kernel output");
        for (std::size_t j = 0; j < checked; ++j) {
            const auto i = checked == 1 ? 0 : j * (actual.size() - 1) / (checked - 1);
            error = std::max(error, std::abs(actual[i] - Reference(i, o, p, f)));
        }
        Require(std::isfinite(error) && error < 1e-3, "plaintext oracle error exceeds 1e-3: " + std::to_string(error));
    }
    if (o.op == "qk") {
        const auto planned = Take(hera::MultiHeadQkKernel::Plan(p.layer, sc.slot_count()));
        Require(Ks(operations) == planned.rotations + planned.conjugations + planned.relinearizations,
                "executed QK KS differs from its plan");
    }
    const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    std::cout << std::fixed << std::setprecision(3)
              << "HERA_BENCHMARK=" << o.op
              << (o.op == "xw" ? "." + o.projection : "") << '\n'
              << "HERA_KS_PER_RUN=" << Ks(operations) << '\n'
              << "HERA_MEAN_MS=" << mean << '\n'
              << "HERA_BENCHMARK_STATUS=PASS\n";
}
} // namespace

int main(int argc, char **argv)
{
    if (argc == 2 && std::string(argv[1]) == "--help") { Usage(); return 0; }
    try { Run(Parse(argc, argv)); return 0; }
    catch (const std::exception &error) {
        std::cerr << "HERA_BENCHMARK_STATUS=FAIL " << error.what() << '\n';
        return 1;
    }
}
