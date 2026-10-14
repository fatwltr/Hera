set(HERA_CRYPTO_ROOT "${PROJECT_SOURCE_DIR}/src/crypto")

if(NOT (HERA_ENABLE_CONVERSION OR HERA_ENABLE_HE))
    return()
endif()

if(HERA_ENABLE_CONVERSION)
    # SCI's exported package performs a C-language RDSEED feature probe.
    enable_language(C)
endif()
find_package(SEAL 4.4.5 EXACT CONFIG REQUIRED)
find_package(Threads REQUIRED)

add_library(hera_seal_runtime STATIC
    ${HERA_CRYPTO_ROOT}/conversion/he_context_registry.cpp
    ${HERA_CRYPTO_ROOT}/he/he_packing.cpp
    ${HERA_CRYPTO_ROOT}/he/seal_tensor.cpp)
hera_configure_target(hera_seal_runtime)
target_link_libraries(hera_seal_runtime PUBLIC
    hera_framework SEAL::seal Threads::Threads)

if(HERA_ENABLE_CONVERSION)
    find_package(OpenSSL REQUIRED COMPONENTS Crypto)
    find_package(SCI CONFIG REQUIRED)
    get_filename_component(HERA_SCI_PREFIX "${SCI_DIR}/../../.." ABSOLUTE)
    set(HERA_SCI_MANIFEST "${HERA_SCI_PREFIX}/share/hera-sci/manifest.txt")
    if(NOT EXISTS "${HERA_SCI_MANIFEST}")
        message(FATAL_ERROR
            "HERA requires the patched HERA-SCI package; run "
            "scripts/bootstrap_dependencies.sh first")
    endif()
    file(READ "${HERA_SCI_MANIFEST}" HERA_SCI_MANIFEST_CONTENT)
    if(NOT HERA_SCI_MANIFEST_CONTENT MATCHES "(^|\n)HERA_SCI_ABI=3(\n|$)")
        message(FATAL_ERROR "HERA-SCI ABI 3 is required")
    endif()
    if(NOT HERA_SCI_MANIFEST_CONTENT MATCHES
       "(^|\n)SEAL_VERSION=4\\.4\\.5(\n|$)")
        message(FATAL_ERROR "HERA-SCI must use exact Microsoft SEAL 4.4.5")
    endif()
    if(NOT HERA_SCI_MANIFEST_CONTENT MATCHES
       "(^|\n)OT_BACKEND=VOLE_SILENT_OT(\n|$)")
        message(FATAL_ERROR "HERA-SCI must use the VOLE/Silent-OT backend")
    endif()
    if(NOT HERA_SCI_MANIFEST_CONTENT MATCHES
       "(^|\n)OT_EXPANSION=UNROLLED_AES_PREFIX_PAD(\n|$)")
        message(FATAL_ERROR "Rebuild HERA-SCI using scripts/bootstrap_dependencies.sh for optimized OT expansion")
    endif()
    if(NOT HERA_SCI_MANIFEST_CONTENT MATCHES
       "(^|\n)MPC_MEMORY=REUSABLE_SLAB_V1(\n|$)")
        message(FATAL_ERROR "Rebuild HERA-SCI using scripts/bootstrap_dependencies.sh for comparison workspace reuse")
    endif()
    if(NOT HERA_SCI_MANIFEST_CONTENT MATCHES
       "(^|\n)MPC_COMPARISON=CORRELATED_AND_V1(\n|$)")
        message(FATAL_ERROR "Rebuild HERA-SCI using scripts/bootstrap_dependencies.sh for correlated conversion comparisons")
    endif()
    # This package's SilentOT facade is backed by FerretCOT. Keep the ABI
    # and exported USE_VOLE=1 definition together rather than switching only
    # application-side macros.
    set(HERA_OT_BACKEND "Ferret" CACHE INTERNAL "Validated SCI OT backend" FORCE)
    message(STATUS "HERA OT backend: Ferret (VOLE/Silent OT)")

    add_library(hera_conversion STATIC
        ${HERA_CRYPTO_ROOT}/conversion/wide_ring.cpp
        ${HERA_CRYPTO_ROOT}/conversion/wide_b2a.cpp
        ${HERA_CRYPTO_ROOT}/conversion/conversion_types.cpp
        ${HERA_CRYPTO_ROOT}/conversion/random_source.cpp
        ${HERA_CRYPTO_ROOT}/conversion/random_sampling.cpp
        ${HERA_CRYPTO_ROOT}/conversion/mpc_session.cpp
        ${HERA_CRYPTO_ROOT}/conversion/netio_transport.cpp
        ${HERA_CRYPTO_ROOT}/conversion/protocol_wire.cpp
        ${HERA_CRYPTO_ROOT}/conversion/fixed_ckks_transform.cpp
        ${HERA_CRYPTO_ROOT}/conversion/ckks_mpc_converter.cpp
        ${HERA_CRYPTO_ROOT}/conversion/conversion_backend.cpp)
    hera_configure_target(hera_conversion WIDE_INTEGER)
    target_include_directories(hera_conversion PRIVATE
        ${HERA_CRYPTO_ROOT}/conversion)
    target_link_libraries(hera_conversion PUBLIC
        hera_framework hera_seal_runtime SEAL::seal Threads::Threads
        OpenSSL::Crypto SCI::SCI-FloatingPoint)

    if(BUILD_TESTING)
        hera_add_test_executable(hera_conversion_compare_tests
            ${PROJECT_SOURCE_DIR}/tests/conversion_compare_tests.cpp
            WIDE_INTEGER LIBRARIES hera_conversion
            PASS "HERA_CONVERSION_COMPARE_STATUS=PASS" TIMEOUT 180
            RESOURCE hera_conversion_ports)
        hera_add_test_executable(hera_packed_conversion_tests
            ${PROJECT_SOURCE_DIR}/tests/packed_conversion_tests.cpp
            WIDE_INTEGER LIBRARIES hera_conversion
            PASS "HERA_PACKED_CONVERSION_STATUS=PASS" TIMEOUT 180
            RESOURCE hera_conversion_ports)
        hera_add_test_executable(hera_ot_expansion_tests
            ${PROJECT_SOURCE_DIR}/tests/ot_expansion_tests.cpp
            WIDE_INTEGER LIBRARIES SCI::SCI-FloatingPoint Threads::Threads
            PASS "HERA_OT_EXPANSION_STATUS=PASS")
        hera_add_test_executable(hera_mpc_session_tests
            ${PROJECT_SOURCE_DIR}/tests/mpc_session_tests.cpp
            WIDE_INTEGER LIBRARIES hera_conversion
            PASS "HERA_MPC_SESSION_STATUS=PASS" TIMEOUT 120
            RESOURCE hera_mpc_session_ports)
        hera_add_test_executable(hera_seal_wire_tests
            ${PROJECT_SOURCE_DIR}/tests/seal_wire_tests.cpp
            LIBRARIES hera_seal_runtime
            PASS "HERA_SEAL_WIRE_STATUS=PASS")
        hera_add_test_executable(hera_conversion_unit_tests
            ${PROJECT_SOURCE_DIR}/tests/conversion_unit_tests.cpp
            WIDE_INTEGER LIBRARIES hera_conversion
            PASS "HERA_CONVERSION_UNIT_STATUS=PASS")
        target_include_directories(hera_conversion_unit_tests PRIVATE
            ${HERA_CRYPTO_ROOT}/conversion)

        hera_add_test_executable(hera_wide_b2a_tests
            ${PROJECT_SOURCE_DIR}/tests/wide_b2a_tests.cpp
            WIDE_INTEGER LIBRARIES hera_conversion
            PASS "HERA_WIDE_B2A_STATUS=PASS" TIMEOUT 120
            RESOURCE hera_conversion_ports)
        target_include_directories(hera_wide_b2a_tests PRIVATE
            ${HERA_CRYPTO_ROOT}/conversion)

        hera_add_test_executable(hera_vole_ot_tests
            ${PROJECT_SOURCE_DIR}/tests/vole_ot_tests.cpp
            WIDE_INTEGER LIBRARIES SCI::SCI-FloatingPoint Threads::Threads
            PASS "HERA_VOLE_OT_STATUS=PASS" TIMEOUT 120
            RESOURCE hera_conversion_ports)
        hera_add_test_executable(hera_conversion_protocol_tests
            ${PROJECT_SOURCE_DIR}/tests/conversion_protocol_tests.cpp
            WIDE_INTEGER LIBRARIES hera_conversion
            PASS "HERA_CONVERSION_PROTOCOL_STATUS=PASS" TIMEOUT 180
            RESOURCE hera_conversion_ports)
        target_include_directories(hera_conversion_protocol_tests PRIVATE
            ${HERA_CRYPTO_ROOT}/conversion)
        add_test(NAME hera_conversion_protocol_uncompressed_tests
            COMMAND hera_conversion_protocol_tests --uncompressed)
        set_tests_properties(hera_conversion_protocol_uncompressed_tests PROPERTIES
            TIMEOUT 180 RESOURCE_LOCK hera_conversion_ports
            PASS_REGULAR_EXPRESSION "HERA_CONVERSION_PROTOCOL_STATUS=PASS")
        add_test(NAME hera_conversion_protocol_small_batch_tests
            COMMAND hera_conversion_protocol_tests --small-batches)
        set_tests_properties(hera_conversion_protocol_small_batch_tests PROPERTIES
            TIMEOUT 180 RESOURCE_LOCK hera_conversion_ports
            PASS_REGULAR_EXPRESSION "HERA_CONVERSION_PROTOCOL_STATUS=PASS")
    endif()

    add_executable(hera_complex_conversion
        ${PROJECT_SOURCE_DIR}/app/conversion_main.cpp)
    target_link_libraries(hera_complex_conversion PRIVATE hera_conversion)
    hera_configure_target(hera_complex_conversion WIDE_INTEGER)
    add_executable(hera_bench_conversion
        ${PROJECT_SOURCE_DIR}/app/conversion_main.cpp)
    target_link_libraries(hera_bench_conversion PRIVATE hera_conversion)
    hera_configure_target(hera_bench_conversion WIDE_INTEGER)
    target_compile_definitions(hera_bench_conversion PRIVATE
        HERA_CONVERSION_BENCHMARK=1)
    if(BUILD_TESTING)
        set(HERA_TEST_SEAL_COMPRESSION none)
        if(SEAL_USE_ZSTD)
            set(HERA_TEST_SEAL_COMPRESSION zstd)
        elseif(SEAL_USE_ZLIB)
            set(HERA_TEST_SEAL_COMPRESSION zlib)
        endif()
        add_test(NAME hera_bench_conversion_smoke
            COMMAND hera_bench_conversion --party=local
                --config=${PROJECT_SOURCE_DIR}/configs/conversion_smoke.json
                --profile=conversion_test --port=24920 --threads=1
                --warmup=1 --iterations=1
                --seal-compression=${HERA_TEST_SEAL_COMPRESSION})
        set_tests_properties(hera_bench_conversion_smoke PROPERTIES
            TIMEOUT 240 RESOURCE_LOCK hera_bench_conversion_ports
            PASS_REGULAR_EXPRESSION "HERA_BENCHMARK_STATUS=PASS")
        file(READ "${PROJECT_SOURCE_DIR}/configs/conversion_smoke.json" _hera_dense_conversion_config)
        foreach(_hera_scale input_scale_bits output_scale_bits default_scale_bits)
            string(JSON _hera_dense_conversion_config SET "${_hera_dense_conversion_config}"
                he_parameter_profiles conversion_test ${_hera_scale} 24)
        endforeach()
        file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/conversion_dense_test.json" "${_hera_dense_conversion_config}")
        add_test(NAME hera_bench_conversion_dense_batch_tests
            COMMAND hera_bench_conversion --party=local
                --config=${CMAKE_CURRENT_BINARY_DIR}/conversion_dense_test.json
                --profile=conversion_test --port=24920 --threads=3
                --ciphertexts=5 --dense --batch-elements=8192
                --warmup=1 --iterations=2
                --seal-compression=${HERA_TEST_SEAL_COMPRESSION})
        set_tests_properties(hera_bench_conversion_dense_batch_tests PROPERTIES
            TIMEOUT 240 RESOURCE_LOCK hera_bench_conversion_ports
            PASS_REGULAR_EXPRESSION "HERA_BENCHMARK_STATUS=PASS")
        add_test(NAME hera_complex_conversion_smoke
            COMMAND hera_complex_conversion --party=local
                --config=${PROJECT_SOURCE_DIR}/configs/conversion_smoke.json
                --profile=conversion_test --port=19450 --threads=1
                --seal-compression=${HERA_TEST_SEAL_COMPRESSION})
        set_tests_properties(hera_complex_conversion_smoke PROPERTIES
            TIMEOUT 180 RESOURCE_LOCK hera_conversion_ports
            PASS_REGULAR_EXPRESSION "HERA_COMPLEX_CONVERSION_STATUS=PASS")
        add_test(NAME hera_complex_conversion_uncompressed_smoke
            COMMAND hera_complex_conversion --party=local
                --config=${PROJECT_SOURCE_DIR}/configs/conversion_smoke.json
                --profile=conversion_test --port=19450 --threads=1
                --seal-compression=none)
        set_tests_properties(hera_complex_conversion_uncompressed_smoke PROPERTIES
            TIMEOUT 180 RESOURCE_LOCK hera_conversion_ports
            PASS_REGULAR_EXPRESSION "HERA_COMPLEX_CONVERSION_STATUS=PASS")
    endif()
endif()

if(HERA_ENABLE_HE)
    add_library(hera_he STATIC ${HERA_CRYPTO_ROOT}/he/block2_he.cpp)
    hera_configure_target(hera_he)
    target_link_libraries(hera_he PUBLIC
        hera_framework hera_seal_runtime SEAL::seal Threads::Threads)
    add_executable(hera_bench_qk
        ${PROJECT_SOURCE_DIR}/tests/he_kernel_benchmark.cpp)
    hera_configure_target(hera_bench_qk)
    target_compile_definitions(hera_bench_qk PRIVATE HERA_BENCHMARK_QK=1)
    target_link_libraries(hera_bench_qk PRIVATE hera_he)
    if(BUILD_TESTING)
        add_test(NAME hera_bench_qk_smoke
            COMMAND hera_bench_qk
                --tokens=4 --heads=2 --width=2 --degree=4096
                --warmup=0 --iterations=1 --he-threads=1)
        set_tests_properties(hera_bench_qk_smoke PROPERTIES
            TIMEOUT 180
            PASS_REGULAR_EXPRESSION "HERA_BENCHMARK_STATUS=PASS")
        hera_add_test_executable(hera_block2_he_tests
            ${PROJECT_SOURCE_DIR}/tests/block2_he_tests.cpp
            LIBRARIES hera_he
            DEFINITIONS
                HERA_TEST_CONFIG_PATH="${PROJECT_SOURCE_DIR}/configs/figure10_smoke.json"
            PASS "HERA_BLOCK2_HE_STATUS=PASS" TIMEOUT 300)
        add_test(NAME hera_layer_norm_tail_tests
            COMMAND hera_block2_he_tests --layer-norm-tail-only)
        set_tests_properties(hera_layer_norm_tail_tests PROPERTIES
            TIMEOUT 120 PASS_REGULAR_EXPRESSION "HERA_REAL_LN_TAIL_STATUS=PASS")
    endif()
endif()

if(HERA_ENABLE_SOFTMAX)
    target_compile_definitions(hera_private_inference PRIVATE
        HERA_SOFTMAX_TO_MPC_BUILT=1)
    add_library(hera_softmax STATIC ${HERA_CRYPTO_ROOT}/he/softmax.cpp)
    hera_configure_target(hera_softmax WIDE_INTEGER)
    target_link_libraries(hera_softmax PUBLIC
        hera_he hera_conversion SCI::SCI-FloatingPoint SEAL::seal
        Threads::Threads)
    if(BUILD_TESTING)
        hera_add_test_executable(hera_softmax_tests
            ${PROJECT_SOURCE_DIR}/tests/softmax_tests.cpp
            WIDE_INTEGER LIBRARIES hera_softmax
            DEFINITIONS
                HERA_TEST_CONFIG_PATH="${PROJECT_SOURCE_DIR}/configs/figure10_smoke.json"
            PASS "HERA_SOFTMAX_STATUS=PASS" TIMEOUT 300
            RESOURCE hera_softmax_ports)
        add_test(NAME hera_softmax_mpc_tests COMMAND hera_softmax_tests)
        set_tests_properties(hera_softmax_mpc_tests PROPERTIES
            TIMEOUT 180 RESOURCE_LOCK hera_softmax_ports
            ENVIRONMENT "HERA_SOFTMAX_MPC_ONLY=1"
            PASS_REGULAR_EXPRESSION "HERA_SOFTMAX_MPC_STATUS=PASS")
    endif()
endif()

if(HERA_ENABLE_ATTENTION_LN2)
    target_compile_definitions(hera_private_inference PRIVATE
        HERA_ATTENTION_LN2_BUILT=1)
    add_library(hera_attention_ln2 STATIC
        ${HERA_CRYPTO_ROOT}/he/attention_ln2.cpp)
    if(BUILD_TESTING)
        hera_add_test_executable(hera_rotation_hoisting_tests
            ${PROJECT_SOURCE_DIR}/tests/rotation_hoisting_tests.cpp
            LIBRARIES SEAL::seal PASS "HERA_ROTATION_HOISTING_STATUS=(PASS|SKIP_UNPATCHED_SEAL)" TIMEOUT 60)
    endif()
    add_executable(hera_bench_ln_rsqrt
        ${PROJECT_SOURCE_DIR}/tests/layer_norm_rsqrt_benchmark.cpp)
    hera_configure_target(hera_bench_ln_rsqrt WIDE_INTEGER)
    target_link_libraries(hera_bench_ln_rsqrt PRIVATE hera_attention_ln2)
    hera_configure_target(hera_attention_ln2 WIDE_INTEGER)
    target_link_libraries(hera_attention_ln2 PUBLIC
        hera_softmax hera_conversion SCI::SCI-FloatingPoint SEAL::seal
        Threads::Threads)
    # Standalone HE-only timing boundaries. Setup, input packing/encryption,
    # output decryption, and correctness checks are outside the timed region.
    add_executable(hera_bench_attention_v
        ${PROJECT_SOURCE_DIR}/tests/he_kernel_benchmark.cpp)
    hera_configure_target(hera_bench_attention_v)
    target_link_libraries(hera_bench_attention_v PRIVATE
        hera_attention_ln2 hera_he)
    add_executable(hera_bench_ln2_statistics
        ${PROJECT_SOURCE_DIR}/tests/layer_norm_statistics_benchmark.cpp)
    hera_configure_target(hera_bench_ln2_statistics WIDE_INTEGER)
    target_link_libraries(hera_bench_ln2_statistics PRIVATE hera_attention_ln2)
    if(BUILD_TESTING)
        add_test(NAME hera_bench_attention_v_smoke
            COMMAND hera_bench_attention_v
                --tokens=4 --heads=2 --width=2 --degree=4096
                --warmup=0 --iterations=1 --he-threads=1)
        set_tests_properties(hera_bench_attention_v_smoke PROPERTIES
            TIMEOUT 180
            PASS_REGULAR_EXPRESSION "HERA_BENCHMARK_STATUS=PASS")
        add_test(NAME hera_ln_rsqrt_precision_tests COMMAND hera_bench_ln_rsqrt
            --tokens=8 --degree=16384 --fixture=edges --warmup=0 --iterations=1 --port=26100)
        set_tests_properties(hera_ln_rsqrt_precision_tests PROPERTIES TIMEOUT 180
            RESOURCE_LOCK hera_ln_precision_ports PASS_REGULAR_EXPRESSION "CHECK=PASS")
        add_test(NAME hera_ln_statistics_rsqrt_precision_tests COMMAND hera_bench_ln_rsqrt
            --tokens=8 --degree=16384 --fixture=statistics --warmup=0 --iterations=1 --port=26100)
        set_tests_properties(hera_ln_statistics_rsqrt_precision_tests PROPERTIES TIMEOUT 180
            RESOURCE_LOCK hera_ln_precision_ports PASS_REGULAR_EXPRESSION "CHECK=PASS")
        add_test(NAME hera_ln_rsqrt_quantum_tests COMMAND hera_bench_ln_rsqrt
            --tokens=8 --degree=16384 --fixture=quantum --warmup=0 --iterations=1 --port=26100)
        set_tests_properties(hera_ln_rsqrt_quantum_tests PROPERTIES TIMEOUT 180
            RESOURCE_LOCK hera_ln_precision_ports PASS_REGULAR_EXPRESSION "CHECK=PASS")
        foreach(invalid_fixture invalid-zero invalid-range invalid-negative)
            add_test(NAME hera_ln_rsqrt_${invalid_fixture}_tests COMMAND hera_bench_ln_rsqrt
                --tokens=8 --degree=16384 --fixture=${invalid_fixture} --warmup=0 --iterations=1 --port=26100)
            set_tests_properties(hera_ln_rsqrt_${invalid_fixture}_tests PROPERTIES TIMEOUT 180
                RESOURCE_LOCK hera_ln_precision_ports PASS_REGULAR_EXPRESSION "CHECK=PASS")
        endforeach()
        hera_add_test_executable(hera_attention_ln2_tests
            ${PROJECT_SOURCE_DIR}/tests/attention_ln2_tests.cpp
            WIDE_INTEGER LIBRARIES hera_attention_ln2
            PASS "ATTENTION_LN2_STATUS=PASS" TIMEOUT 900
            RESOURCE hera_attention_ln2_ports)
        add_test(NAME hera_attention_ln2_fused_tests
            COMMAND hera_attention_ln2_tests --fused-attention)
        set_tests_properties(hera_attention_ln2_fused_tests PROPERTIES
            TIMEOUT 900 PASS_REGULAR_EXPRESSION "ATTENTION_LN2_STATUS=PASS"
            RESOURCE_LOCK hera_attention_ln2_ports)
        hera_add_test_executable(hera_ln_rsqrt_contract_tests
            ${PROJECT_SOURCE_DIR}/tests/layer_norm_rsqrt_contract_tests.cpp
            WIDE_INTEGER LIBRARIES hera_attention_ln2
            PASS "RSQRT_CONTRACT_STATUS=PASS" TIMEOUT 180
            RESOURCE hera_ln_precision_ports)
    endif()
endif()

if(HERA_ENABLE_FFN_GELU)
    target_compile_definitions(hera_private_inference PRIVATE
        HERA_FFN_GELU_BUILT=1)
    add_library(hera_ffn_gelu STATIC ${HERA_CRYPTO_ROOT}/he/ffn_gelu.cpp)
    hera_configure_target(hera_ffn_gelu WIDE_INTEGER)
    target_link_libraries(hera_ffn_gelu PUBLIC
        hera_he hera_conversion SCI::SCI-FloatingPoint SEAL::seal
        Threads::Threads)
    add_executable(hera_bench_he_functions
        ${PROJECT_SOURCE_DIR}/tests/functional_he_benchmark.cpp)
    hera_configure_target(hera_bench_he_functions WIDE_INTEGER)
    target_link_libraries(hera_bench_he_functions PRIVATE
        hera_ffn_gelu hera_attention_ln2 hera_softmax)
    add_executable(hera_bench_mpc_functions
        ${PROJECT_SOURCE_DIR}/tests/nonlinear_mpc_benchmark.cpp)
    hera_configure_target(hera_bench_mpc_functions WIDE_INTEGER)
    target_link_libraries(hera_bench_mpc_functions PRIVATE
        hera_ffn_gelu hera_softmax)
    if(BUILD_TESTING)
        foreach(kernel qkv wo wu wd softmax-exp gelu-poly)
            add_test(NAME hera_bench_he_${kernel}_smoke
                COMMAND hera_bench_he_functions --kernel=${kernel}
                    --warmup=0 --iterations=1 --he-threads=1)
            set_tests_properties(hera_bench_he_${kernel}_smoke PROPERTIES
                TIMEOUT 240
                PASS_REGULAR_EXPRESSION "HERA_BENCHMARK_STATUS=PASS")
        endforeach()
        foreach(kernel softmax-max softmax-normalize gelu-select)
            add_test(NAME hera_bench_mpc_${kernel}_smoke
                COMMAND hera_bench_mpc_functions --kernel=${kernel}
                    --warmup=1 --iterations=1 --port=24900)
            set_tests_properties(hera_bench_mpc_${kernel}_smoke PROPERTIES
                TIMEOUT 240 RESOURCE_LOCK hera_bench_mpc_ports
                PASS_REGULAR_EXPRESSION "HERA_BENCHMARK_STATUS=PASS")
        endforeach()
        foreach(kernel softmax-max softmax-normalize gelu-select)
            add_test(NAME hera_bench_mpc_${kernel}_parallel
                COMMAND hera_bench_mpc_functions --kernel=${kernel}
                    --tokens=17 --heads=3 --width=4097 --workers=3
                    --warmup=1 --iterations=1 --port=24900)
            set_tests_properties(hera_bench_mpc_${kernel}_parallel PROPERTIES
                TIMEOUT 240 RESOURCE_LOCK hera_bench_mpc_ports
                PASS_REGULAR_EXPRESSION "HERA_BENCHMARK_STATUS=PASS")
        endforeach()
        hera_add_test_executable(hera_ffn_gelu_tests
            ${PROJECT_SOURCE_DIR}/tests/ffn_gelu_tests.cpp
            WIDE_INTEGER LIBRARIES hera_ffn_gelu
            PASS "FFN_GELU_STATUS=PASS" TIMEOUT 900
            RESOURCE hera_ffn_gelu_ports)
    endif()
endif()

if(HERA_ENABLE_BLOCK5_BRIDGE)
    target_compile_definitions(hera_private_inference PRIVATE
        HERA_BLOCK5_BRIDGE_BUILT=1 HERA_PRIVATE_RUNTIME_BUILT=1)
    add_library(hera_block5_bridge STATIC
        ${HERA_CRYPTO_ROOT}/he/block5_bridge.cpp
        ${HERA_CRYPTO_ROOT}/he/pre_ln_entry.cpp)
    hera_configure_target(hera_block5_bridge WIDE_INTEGER)
    target_link_libraries(hera_block5_bridge PUBLIC
        hera_ffn_gelu hera_attention_ln2 hera_he hera_conversion
        SCI::SCI-FloatingPoint SEAL::seal Threads::Threads)

    add_library(hera_private_runtime STATIC
        ${HERA_CRYPTO_ROOT}/runtime/execution_plan.cpp
        ${HERA_CRYPTO_ROOT}/runtime/private_runtime.cpp
        ${HERA_CRYPTO_ROOT}/runtime/runtime_coordinator.cpp)
    hera_configure_target(hera_private_runtime WIDE_INTEGER)
    target_link_libraries(hera_private_runtime PUBLIC
        hera_block5_bridge hera_ffn_gelu hera_attention_ln2 hera_softmax
        hera_he hera_conversion SCI::SCI-FloatingPoint SEAL::seal
        Threads::Threads)
    # Keep application parsing/reporting out of the reusable runtime library.
    target_sources(hera_private_inference PRIVATE
        ${PROJECT_SOURCE_DIR}/app/private_runtime_cli.cpp)
    target_link_libraries(hera_private_inference PRIVATE hera_private_runtime)

    if(BUILD_TESTING)
        hera_add_test_executable(hera_private_runtime_tests
            ${PROJECT_SOURCE_DIR}/tests/private_runtime_tests.cpp
            WIDE_INTEGER LIBRARIES hera_private_runtime
            PASS "HERA_PRIVATE_RUNTIME_STATUS=PASS" TIMEOUT 900
            RESOURCE hera_private_runtime_ports)
        add_test(NAME hera_private_runtime_extended_ffn_chain_tests COMMAND hera_private_runtime_tests --extended-ffn-chain)
        set_tests_properties(hera_private_runtime_extended_ffn_chain_tests PROPERTIES TIMEOUT 900
        PASS_REGULAR_EXPRESSION "HERA_PRIVATE_RUNTIME_STATUS=PASS" RESOURCE_LOCK hera_private_runtime_ports)
        set_tests_properties(hera_private_runtime_tests PROPERTIES RESOURCE_LOCK hera_private_runtime_ports)
        foreach(bridge_variant explicit-bridge explicit-bridge-extended-ffn-chain explicit-bridge-no-next-skip fused-attention)
            add_test(NAME hera_private_runtime_${bridge_variant}_tests
                COMMAND hera_private_runtime_tests --${bridge_variant})
            set_tests_properties(hera_private_runtime_${bridge_variant}_tests PROPERTIES
                TIMEOUT 900 PASS_REGULAR_EXPRESSION "HERA_PRIVATE_RUNTIME_STATUS=PASS"
                RESOURCE_LOCK hera_private_runtime_ports)
        endforeach()
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
            "${PROJECT_SOURCE_DIR}/configs/runtime_tiny.json")
        file(READ "${PROJECT_SOURCE_DIR}/configs/runtime_tiny.json" _hera_cli_pre_config)
        string(JSON _hera_cli_pre_config SET "${_hera_cli_pre_config}" model layers 0 norm "\"pre\"")
        # Standard Pre-LN statistics use input scale38; the compact F profile
        # uses scale46 statistics only in normalized-residual mode.
        string(JSON _hera_cli_pre_config SET "${_hera_cli_pre_config}"
            he_parameter_profiles runtime_pre_statistics [=[{
                "polynomial_degree": 2048,
                "coefficient_modulus_bits": [60,60,60,60,60],
                "default_scale_bits": 38, "input_scale_bits": 38, "output_scale_bits": 38,
                "declared_depth": 4, "terminal_prime_count": 1,
                "require_exact_terminal_level": true, "security_level": "none"
            }]=])
        string(JSON _hera_cli_pre_config SET "${_hera_cli_pre_config}"
            layers 0 he_regions region1 profile "\"runtime_pre_statistics\"")
        file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/runtime_tiny_pre_test.json" "${_hera_cli_pre_config}")
        add_test(NAME hera_private_cli_pre_tests COMMAND hera_private_inference
            --config=${CMAKE_CURRENT_BINARY_DIR}/runtime_tiny_pre_test.json
            --party=local --port=28100 --he-threads=4 --conversion-workers=2 --verify)
        set_tests_properties(hera_private_cli_pre_tests PROPERTIES TIMEOUT 900
            PASS_REGULAR_EXPRESSION "HERA_PRIVATE_INFERENCE_STATUS=PASS"
            RESOURCE_LOCK hera_private_runtime_ports)
        hera_add_test_executable(hera_runtime_coordinator_tests
            ${PROJECT_SOURCE_DIR}/tests/runtime_coordinator_tests.cpp
            WIDE_INTEGER LIBRARIES hera_private_runtime
            PASS "HERA_RUNTIME_COORDINATOR_STATUS=PASS" TIMEOUT 20
            RESOURCE hera_private_runtime_ports)
        hera_add_test_executable(hera_block5_bridge_tests
            ${PROJECT_SOURCE_DIR}/tests/block5_bridge_tests.cpp
            WIDE_INTEGER LIBRARIES hera_block5_bridge
            DEFINITIONS
                HERA_TEST_CONFIG_PATH="${PROJECT_SOURCE_DIR}/configs/figure10_smoke.json"
            PASS "BLOCK5_BRIDGE_STATUS=PASS" TIMEOUT 300
            RESOURCE hera_block5_ports)
        add_test(NAME hera_block5_normalized_next_tests
            COMMAND hera_block5_bridge_tests --normalized-next)
        set_tests_properties(hera_block5_normalized_next_tests PROPERTIES
            TIMEOUT 300 PASS_REGULAR_EXPRESSION "BLOCK5_BRIDGE_STATUS=PASS"
            RESOURCE_LOCK hera_block5_ports)
    endif()
endif()
