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

    add_library(hera_conversion STATIC
        ${HERA_CRYPTO_ROOT}/conversion/wide_ring.cpp
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
        hera_add_test_executable(hera_conversion_unit_tests
            ${PROJECT_SOURCE_DIR}/tests/conversion_unit_tests.cpp
            WIDE_INTEGER LIBRARIES hera_conversion
            PASS "HERA_CONVERSION_UNIT_STATUS=PASS")
        target_include_directories(hera_conversion_unit_tests PRIVATE
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
    endif()

    add_executable(hera_complex_conversion
        ${PROJECT_SOURCE_DIR}/app/conversion_main.cpp)
    target_link_libraries(hera_complex_conversion PRIVATE hera_conversion)
    hera_configure_target(hera_complex_conversion WIDE_INTEGER)
    if(BUILD_TESTING)
        add_test(NAME hera_complex_conversion_smoke
            COMMAND hera_complex_conversion --party=local
                --config=${PROJECT_SOURCE_DIR}/configs/conversion_smoke.json
                --profile=conversion_test --port=19450 --threads=1)
        set_tests_properties(hera_complex_conversion_smoke PROPERTIES
            TIMEOUT 180 RESOURCE_LOCK hera_conversion_ports
            PASS_REGULAR_EXPRESSION "HERA_COMPLEX_CONVERSION_STATUS=PASS")
    endif()
endif()

if(HERA_ENABLE_HE)
    add_library(hera_he STATIC ${HERA_CRYPTO_ROOT}/he/block2_he.cpp)
    hera_configure_target(hera_he)
    target_link_libraries(hera_he PUBLIC
        hera_framework hera_seal_runtime SEAL::seal Threads::Threads)
    if(BUILD_TESTING)
        hera_add_test_executable(hera_block2_he_tests
            ${PROJECT_SOURCE_DIR}/tests/block2_he_tests.cpp
            LIBRARIES hera_he
            DEFINITIONS
                HERA_TEST_CONFIG_PATH="${PROJECT_SOURCE_DIR}/configs/figure10_smoke.json"
            PASS "HERA_BLOCK2_HE_STATUS=PASS" TIMEOUT 300)
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
    hera_configure_target(hera_attention_ln2 WIDE_INTEGER)
    target_link_libraries(hera_attention_ln2 PUBLIC
        hera_softmax hera_conversion SCI::SCI-FloatingPoint SEAL::seal
        Threads::Threads)
    if(BUILD_TESTING)
        hera_add_test_executable(hera_attention_ln2_tests
            ${PROJECT_SOURCE_DIR}/tests/attention_ln2_tests.cpp
            WIDE_INTEGER LIBRARIES hera_attention_ln2
            PASS "ATTENTION_LN2_STATUS=PASS" TIMEOUT 900
            RESOURCE hera_attention_ln2_ports)
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
    if(BUILD_TESTING)
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
        ${HERA_CRYPTO_ROOT}/he/block5_bridge.cpp)
    hera_configure_target(hera_block5_bridge WIDE_INTEGER)
    target_link_libraries(hera_block5_bridge PUBLIC
        hera_ffn_gelu hera_attention_ln2 hera_he hera_conversion
        SCI::SCI-FloatingPoint SEAL::seal Threads::Threads)

    add_library(hera_private_runtime STATIC
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
    endif()
endif()
