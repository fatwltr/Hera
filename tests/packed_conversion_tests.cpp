#include "hera/conversion.h"
#include "hera/detail/he_packing.h"
#include "hera/detail/conversion_lanes.h"
#include "hera/seal_wire.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <sys/wait.h>
#include <unistd.h>

namespace {
void Check(bool ok, const std::string &message) { if (!ok) throw std::runtime_error(message); }

hera::HeraConfig Config()
{
    hera::HeraConfig config;
    config.layers.push_back(hera::TransformerLayerConfig{
        "layer", 2, 4, 1, 1, 4, 4, 8, hera::NormPlacement::kPreNorm,
        hera::ActivationKind::kGelu, hera::AttentionMaskKind::kNone,
        false, false, false, true, true});
    hera::LayerExecutionConfig execution;
    execution.layer_id = "layer";
    for (const auto region : {hera::HeRegionId::kRegion1, hera::HeRegionId::kRegion2,
            hera::HeRegionId::kRegion3, hera::HeRegionId::kRegion4, hera::HeRegionId::kRegion5})
        execution.regions.emplace(region, hera::HeRegionConfig{region,"test",hera::Layout::kSpatial,hera::Layout::kSpatial});
    config.layer_execution.push_back(std::move(execution));
    config.mpc_numeric = {43, 13, true};
    config.conversion = {40, 20, true};
    config.he_parameter_profiles.emplace("test", hera::HeParameterConfig{
        4096, {60, 40, 40, 60}, 40, 40, 40, 3, 1, true, hera::SecurityLevel::kNone});
    return config;
}

void Run(hera::PartyRole role, hera::HeContextRegistry &registry)
{
    const bool server = role == hera::PartyRole::kServer;
    const auto &context = registry.Require("test", role);
    const auto slots = context.slot_count();
    auto transport = hera::MakeNetIoTransport(role, "127.0.0.1", 17900);
    hera::CkksMpcConverter converter(role, {43,13,40,20,true,4096}, registry,
        transport, hera::MakeProductionRandomSource(), "127.0.0.1", 17920, 3);
    constexpr auto mask = (std::uint64_t{1} << 43) - 1;
    double maximum = 0;
    // Odd dense tail, sparse singleton, one logical value, full singleton,
    // then a mixed complex/real frame with mixed public range contracts.
    for (int fixture = 0; fixture < 10; ++fixture) {
        auto map = hera::PhysicalSlotMap::DenseReal(
            fixture == 0 ? 2 * slots + 5 : fixture == 1 ? 7 : fixture == 2 ? 1
                : fixture == 3 || fixture == 5 || fixture == 6 || fixture == 8 ? slots : fixture == 9 ? 7 : 3 * slots, slots);
        if (fixture == 8) {
            for (std::size_t slot = 0; slot < slots; ++slot)
                map.assignments[slot] = {2 * slot, 2 * slot + 1};
            map.logical_element_count = 2 * slots;
        }
        if (fixture == 4) {
            const auto offset = map.logical_element_count;
            ++map.ciphertext_count;
            map.assignments.resize(4 * slots);
            for (std::size_t slot = 0; slot < slots; ++slot) {
                map.assignments[3 * slots + slot] = {offset + 2 * slot, offset + 2 * slot + 1};
            }
            map.logical_element_count += 2 * slots;
        }
        auto planned = hera::detail::PlanRealConversion(map);
        Check(planned.ok(), planned.status().message());
        std::size_t imaginary = 0;
        for (const auto &entry : planned.value().map.assignments)
            imaginary += entry.imaginary_logical_index != hera::kUnusedLogicalIndex;
        Check(fixture == 2 || imaginary > 0, "packed boundary failed to use imaginary data");
        const hera::TensorShape physical{{map.ciphertext_count, slots}};
        hera::ComplexMpcSharePair shares{
            {std::vector<std::uint64_t>(map.ciphertext_count * slots),physical,{43,13,true},hera::ComplexComponent::kReal},
            {std::vector<std::uint64_t>(map.ciphertext_count * slots),physical,{43,13,true},hera::ComplexComponent::kImaginary}};
        std::vector<std::complex<double>> expected(map.ciphertext_count * slots);
        for (std::size_t i = 0; i < map.assignments.size(); ++i) {
            for (int component = 0; component < 2; ++component) {
                const auto logical = component ? map.assignments[i].imaginary_logical_index : map.assignments[i].real_logical_index;
                if (logical == hera::kUnusedLogicalIndex) continue;
                // Includes zero, negative values, and odd least-significant
                // bits: integer division by two would fail this oracle.
                const auto period = fixture == 6 ? 1 : fixture == 8 && component ? 8 : 4;
                const auto key = fixture >= 5 ? (i / slots) * 101 + (i % slots) % period : logical;
                const auto value = static_cast<std::int64_t>(key % 101) - 50;
                const auto random = (0x71826354abcdefULL + 104729 * key + 31337 * component) & mask;
                auto &plane = component ? shares.imag : shares.real;
                plane.local_share[i] = server ? random : (static_cast<std::uint64_t>(value) - random) & mask;
                if (component) expected[i].imag(std::ldexp(static_cast<double>(value), -13));
                else expected[i].real(std::ldexp(static_cast<double>(value), -13));
            }
        }
        const auto original = shares;
        hera::MpcToCkksTarget target{"test", 1, 40, {{map.logical_element_count}},hera::Layout::kSpatial,map};
        if (fixture == 4) target.public_ciphertext_bound_bits = {7, std::nullopt, 7, std::nullopt};
        if (fixture >= 5) {
            target.public_slot_periods.resize(map.ciphertext_count);
            for (auto &periods : target.public_slot_periods)
                periods = {fixture == 6 ? std::size_t{1} : std::size_t{4}, fixture == 8 ? std::size_t{8} : std::size_t{0}};
        }
        const auto before = converter.stats();
        auto converted = converter.MpcToCkksPacked(shares, target);
        Check(converted.ok(), converted.status().message());
        Check(shares.real.local_share == original.real.local_share && shares.imag.local_share == original.imag.local_share,
            "packed conversion mutated its input shares");
        const auto after = converter.stats();
        Check(after.correlated_comparison_elements > before.correlated_comparison_elements,
            "M2C did not exercise the correlated comparison path");
        Check(after.radix5_comparison_elements == before.radix5_comparison_elements,
            "M2C unexpectedly changed its comparison radix");
        Check(after.packed_real_wire_ciphertexts - before.packed_real_wire_ciphertexts == (fixture == 8 ? 0 : fixture == 0 || fixture == 4 || fixture == 7 ? 2 : 1),
            "packed conversion did not reduce real wire ciphertext count");
        if (fixture >= 5) {
            const auto unique = after.slot_extension_unique_elements - before.slot_extension_unique_elements;
            const auto dense = after.slot_extension_input_elements - before.slot_extension_input_elements;
            Check(unique <= 16 && unique < dense, "public broadcast replicas were not deduplicated");
        }
        if (server) {
            Check(converted.value() && converted.value()->ciphertexts.size() == map.ciphertext_count,
                "packed conversion did not restore the requested layout");
            auto id = context.context->first_context_data()->next_context_data()->parms_id();
            for (const auto &ct : converted.value()->ciphertexts) {
                Check(ct.parms_id() == id && ct.scale() == std::ldexp(1.,40), "real split consumed a level or changed scale");
                hera::SendSealObject(*transport, ct);
            }
        } else {
            seal::Decryptor decryptor(*context.context, *context.secret_key);
            seal::CKKSEncoder encoder(*context.context);
            for (std::size_t ct = 0; ct < map.ciphertext_count; ++ct) {
                auto encrypted = hera::ReceiveSealObject<seal::Ciphertext>(*transport, *context.context);
                seal::Plaintext plain; decryptor.decrypt(encrypted, plain);
                std::vector<std::complex<double>> decoded; encoder.decode(plain, decoded);
                for (std::size_t slot = 0; slot < slots; ++slot) {
                    const auto i = ct * slots + slot;
                    maximum = std::max({maximum, std::abs(decoded[slot].real() - expected[i].real()),
                        std::abs(decoded[slot].imag() - expected[i].imag())});
                }
            }
            Check(maximum <= std::ldexp(4.,-13), "packed M2C reconstruction exceeds four ULP");
        }
    }
    // Invalid contracts must be rejected on both peers before any extension.
    for (int fixture = 0; fixture < 5; ++fixture) {
        auto map = hera::PhysicalSlotMap::DenseReal(slots, slots);
        const hera::TensorShape physical{{1, slots}};
        hera::ComplexMpcSharePair shares{
            {std::vector<std::uint64_t>(slots), physical, {43,13,true}, hera::ComplexComponent::kReal},
            {std::vector<std::uint64_t>(slots), physical, {43,13,true}, hera::ComplexComponent::kImaginary}};
        hera::MpcToCkksTarget target{"test", 1, 40, {{slots}}, hera::Layout::kSpatial, map};
        target.public_slot_periods = {{4, 0}};
        if (fixture == 0) target.public_slot_periods[0][0] = 3;
        if (fixture == 1) target.public_slot_periods.resize(2);
        if (fixture == 2 && server) shares.real.local_share[4] = 1;
        if (fixture == 3 && server) target.public_slot_periods[0][0] = 8;
        if (fixture == 4) target.public_slot_periods[0][0] = 2 * slots;
        const auto before = converter.stats().generic_signed_extension_elements;
        auto result = converter.MpcToCkksPacked(shares, target);
        Check(!result.ok(), "invalid or mismatched period contract was accepted");
        Check(before == converter.stats().generic_signed_extension_elements, "invalid period entered extension");
    }
    if (!server) std::cout << "PACKED_M2C_MAX_ULP=" << std::ldexp(maximum,13) << '\n';
}
}

int main()
{
    try {
        // Exact pre-encoding invariant, independent of FFT root approximation.
        // Doubling half-encoded coefficients must reproduce ordinary encoding
        // even at the full signed-ring extrema and for random wide shares.
        const std::int64_t edges[] = {0,1,-1,(std::int64_t{1} << 42)-1,
            -(std::int64_t{1} << 42),-(std::int64_t{1} << 42)+1};
        for (auto value : edges) {
            auto wide = hera::WideFromLow(static_cast<std::uint64_t>(value < 0 ? -value : value),83);
            if (value < 0) wide = hera::WideNegate(wide,83);
            for (int shift : {1,11,27}) {
                const auto normal = hera::WideShiftLeft(wide,shift,83+shift);
                const auto half = hera::WideShiftLeft(wide,shift-1,82+shift);
                const auto restored = hera::WideShiftLeft(half,1,83+shift);
                Check(normal.low == restored.low && normal.high == restored.high,
                    "half encoding discarded an integer bit");
            }
        }
        hera::HeContextRegistry client, server;
        auto config = Config();
        auto created = client.CreateClientProfiles(config, false);
        Check(created.ok(), created.status().message());
        Check(client.CompileRotationKeys("test", {0, -4, 1024}).ok(), "packed conversion keys failed");
        auto bundle = client.ExportPublicBundle(); Check(bundle.ok(), bundle.status().message());
        Check(server.CreateServerProfiles(config,bundle.value()).ok(), "packed conversion server setup failed");
        const auto child = fork(); Check(child >= 0, "packed conversion fork failed");
        if (child == 0) {
            try { Run(hera::PartyRole::kClient,client); std::cout.flush(); _exit(0); }
            catch (const std::exception &error) { std::cerr << error.what() << '\n'; _exit(2); }
        }
        Run(hera::PartyRole::kServer,server);
        int status{}; waitpid(child,&status,0);
        Check(WIFEXITED(status) && WEXITSTATUS(status)==0, "packed conversion child failed");
        std::cout << "HERA_PACKED_CONVERSION_STATUS=PASS\n";
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
