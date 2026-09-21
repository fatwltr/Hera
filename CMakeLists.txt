cmake_minimum_required(VERSION 3.21)

project(Hera
    VERSION 0.1.0
    DESCRIPTION "Hybrid CKKS/MPC private Transformer inference research framework"
    LANGUAGES CXX)

include(CTest)

option(HERA_ENABLE_FULL_RUNTIME
       "Build every cryptographic stage and the end-to-end runtime" OFF)

set(HERA_FRAMEWORK_SOURCES
    config/json_loader.cpp
    config/validator.cpp
    core/model.cpp
    core/tensor.cpp
    graph/graph.cpp
    graph/graph_validator.cpp
    graph/transformer_graph_builder.cpp
    modules/layer_norm.cpp
    modules/attention.cpp
    modules/softmax.cpp
    modules/activation.cpp
    modules/feed_forward.cpp
    runtime/hera_context.cpp
    runtime/executor.cpp
    backend/unavailable_backend.cpp
    backend/recording_backend.cpp)

add_library(hera_framework STATIC ${HERA_FRAMEWORK_SOURCES})
target_include_directories(hera_framework PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(hera_framework PUBLIC cxx_std_17)
target_compile_options(hera_framework PRIVATE -Wall -Wextra -Wpedantic)

add_executable(hera_private_inference app/main.cpp)
target_link_libraries(hera_private_inference PRIVATE hera_framework)
target_compile_options(hera_private_inference PRIVATE
    -Wall -Wextra -Wpedantic)
target_compile_definitions(hera_private_inference PRIVATE
    HERA_DEFAULT_CONFIG_PATH="${CMAKE_CURRENT_SOURCE_DIR}/configs/figure10_smoke.json")

add_executable(hera_framework_tests tests/framework_tests.cpp)
target_link_libraries(hera_framework_tests PRIVATE hera_framework)
target_compile_options(hera_framework_tests PRIVATE
    -Wall -Wextra -Wpedantic)
target_compile_definitions(hera_framework_tests PRIVATE
    HERA_TEST_CONFIG_PATH="${CMAKE_CURRENT_SOURCE_DIR}/configs/figure10_smoke.json")
add_test(NAME hera_framework_tests COMMAND hera_framework_tests)
set_tests_properties(hera_framework_tests PROPERTIES
    PASS_REGULAR_EXPRESSION "HERA_FRAMEWORK_STATUS=PASS")

option(HERA_ENABLE_CONVERSION
       "Build the independent SEAL/HERA-SCI complex conversion protocol" OFF)
option(HERA_ENABLE_HE
       "Build the SEAL CKKS implementation of HERA Block 2" OFF)
option(HERA_ENABLE_SOFTMAX
       "Build the staged SEAL/HERA-SCI Softmax implementation" OFF)
option(HERA_ENABLE_ATTENTION_LN2
       "Build Attention x V through the LayerNorm reciprocal-sqrt boundary" OFF)
option(HERA_ENABLE_FFN_GELU
       "Build LayerNorm tail through the GeLU MPC-share boundary" OFF)
option(HERA_ENABLE_BLOCK5_BRIDGE
       "Build Wd/residual and optional next-layer Block-2 bridge" OFF)

if(HERA_ENABLE_FULL_RUNTIME)
    set(HERA_ENABLE_BLOCK5_BRIDGE ON CACHE BOOL
        "Build Wd/residual and optional next-layer Block-2 bridge" FORCE)
endif()

if(HERA_ENABLE_BLOCK5_BRIDGE)
    set(HERA_ENABLE_FFN_GELU ON)
    set(HERA_ENABLE_ATTENTION_LN2 ON)
endif()

if(HERA_ENABLE_SOFTMAX OR HERA_ENABLE_ATTENTION_LN2 OR HERA_ENABLE_FFN_GELU OR
   HERA_ENABLE_BLOCK5_BRIDGE)
    set(HERA_ENABLE_CONVERSION ON)
    set(HERA_ENABLE_HE ON)
endif()

if(HERA_ENABLE_ATTENTION_LN2 OR HERA_ENABLE_FFN_GELU)
    set(HERA_ENABLE_SOFTMAX ON)
endif()

if(HERA_ENABLE_CONVERSION OR HERA_ENABLE_HE)
    # SCI's exported package performs a C-language RDSEED feature probe.
    if(HERA_ENABLE_CONVERSION)
        enable_language(C)
    endif()
    # Keep package selection controlled by SEAL_DIR so the conversion module
    # can be built against a compatible installed SEAL release.
    find_package(SEAL 4.4.5 EXACT CONFIG REQUIRED)
    find_package(Threads REQUIRED)

    add_library(hera_seal_runtime STATIC
        conversion/he_context_registry.cpp
        he/seal_tensor.cpp)
    target_include_directories(hera_seal_runtime PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_features(hera_seal_runtime PUBLIC cxx_std_17)
    target_link_libraries(hera_seal_runtime PUBLIC
        hera_framework SEAL::seal Threads::Threads)
    target_compile_options(hera_seal_runtime PRIVATE
        -Wall -Wextra -Wpedantic)
endif()

if(HERA_ENABLE_CONVERSION)
    find_package(OpenSSL REQUIRED COMPONENTS Crypto)
    find_package(SCI CONFIG REQUIRED)
    get_filename_component(HERA_SCI_PREFIX "${SCI_DIR}/../../.." ABSOLUTE)
    set(HERA_SCI_MANIFEST
        "${HERA_SCI_PREFIX}/share/hera-sci/manifest.txt")
    if(NOT EXISTS "${HERA_SCI_MANIFEST}")
        message(FATAL_ERROR
            "HERA conversion/nonlinear targets require the unified patched "
            "HERA-SCI package. Run scripts/prepare_hera_sci.sh and point "
            "SCI_DIR at its install/lib/cmake/SCI directory.")
    endif()
    file(READ "${HERA_SCI_MANIFEST}" HERA_SCI_MANIFEST_CONTENT)
    if(NOT HERA_SCI_MANIFEST_CONTENT MATCHES "(^|\n)HERA_SCI_ABI=3(\n|$)")
        message(FATAL_ERROR
            "The selected HERA-SCI package has an incompatible ABI; "
            "HERA_SCI_ABI=3 is required.")
    endif()
    if(NOT HERA_SCI_MANIFEST_CONTENT MATCHES "(^|\n)SEAL_VERSION=4\\.4\\.5(\n|$)")
        message(FATAL_ERROR
            "The selected HERA-SCI package was not built against exact "
            "Microsoft SEAL 4.4.5.")
    endif()
    if(NOT HERA_SCI_MANIFEST_CONTENT MATCHES
       "(^|\n)OT_BACKEND=VOLE_SILENT_OT(\n|$)")
        message(FATAL_ERROR
            "The selected HERA-SCI package does not use the required "
            "VOLE/Silent-OT backend.")
    endif()

    add_library(hera_conversion STATIC
        conversion/wide_ring.cpp
        conversion/conversion_types.cpp
        conversion/random_source.cpp
        conversion/random_sampling.cpp
        conversion/mpc_session.cpp
        conversion/netio_transport.cpp
        conversion/protocol_wire.cpp
        conversion/fixed_ckks_transform.cpp
        conversion/ckks_mpc_converter.cpp
        conversion/conversion_backend.cpp)
    target_include_directories(hera_conversion
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/conversion)
    target_compile_features(hera_conversion PUBLIC cxx_std_17)
    target_link_libraries(hera_conversion PUBLIC
        hera_framework hera_seal_runtime SEAL::seal Threads::Threads
        OpenSSL::Crypto SCI::SCI-FloatingPoint)
    target_compile_options(hera_conversion PRIVATE
        -Wall -Wextra -Wpedantic -Wno-pedantic)

    add_executable(hera_conversion_unit_tests
        tests/conversion_unit_tests.cpp)
    target_include_directories(hera_conversion_unit_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/conversion)
    target_link_libraries(hera_conversion_unit_tests PRIVATE hera_conversion)
    target_compile_options(hera_conversion_unit_tests PRIVATE
        -Wall -Wextra -Wpedantic -Wno-pedantic)
    add_test(NAME hera_conversion_unit_tests
             COMMAND hera_conversion_unit_tests)
    set_tests_properties(hera_conversion_unit_tests PROPERTIES
        PASS_REGULAR_EXPRESSION "HERA_CONVERSION_UNIT_STATUS=PASS")

    add_executable(hera_vole_ot_tests tests/vole_ot_tests.cpp)
    target_link_libraries(hera_vole_ot_tests PRIVATE
        SCI::SCI-FloatingPoint Threads::Threads)
    target_compile_options(hera_vole_ot_tests PRIVATE
        -Wall -Wextra -Wpedantic -Wno-pedantic)
    add_test(NAME hera_vole_ot_tests COMMAND hera_vole_ot_tests)
    set_tests_properties(hera_vole_ot_tests PROPERTIES
        TIMEOUT 120
        RESOURCE_LOCK hera_conversion_ports
        PASS_REGULAR_EXPRESSION "HERA_VOLE_OT_STATUS=PASS")

    add_executable(hera_conversion_protocol_tests
        tests/conversion_protocol_tests.cpp)
    target_link_libraries(hera_conversion_protocol_tests PRIVATE
        hera_conversion)
    target_compile_options(hera_conversion_protocol_tests PRIVATE
        -Wall -Wextra -Wpedantic -Wno-pedantic)
    add_test(NAME hera_conversion_protocol_tests
             COMMAND hera_conversion_protocol_tests)
    set_tests_properties(hera_conversion_protocol_tests PROPERTIES
        TIMEOUT 180
        RESOURCE_LOCK hera_conversion_ports
        PASS_REGULAR_EXPRESSION "HERA_CONVERSION_PROTOCOL_STATUS=PASS")

    add_executable(hera_complex_conversion app/conversion_main.cpp)
    target_link_libraries(hera_complex_conversion PRIVATE hera_conversion)
    target_compile_options(hera_complex_conversion PRIVATE
        -Wall -Wextra -Wpedantic -Wno-pedantic)
    add_test(NAME hera_complex_conversion_smoke
        COMMAND hera_complex_conversion --party=local
            --config=${CMAKE_CURRENT_SOURCE_DIR}/configs/conversion_smoke.json
            --profile=conversion_test --port=19450 --threads=1)
    set_tests_properties(hera_complex_conversion_smoke PROPERTIES
        TIMEOUT 180
        RESOURCE_LOCK hera_conversion_ports
        PASS_REGULAR_EXPRESSION "HERA_COMPLEX_CONVERSION_STATUS=PASS")
endif()

if(HERA_ENABLE_HE)
    add_library(hera_he STATIC he/block2_he.cpp)
    target_include_directories(hera_he PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_features(hera_he PUBLIC cxx_std_17)
    target_link_libraries(hera_he PUBLIC
        hera_framework hera_seal_runtime SEAL::seal Threads::Threads)
    target_compile_options(hera_he PRIVATE
        -Wall -Wextra -Wpedantic)

    add_executable(hera_block2_he_tests tests/block2_he_tests.cpp)
    target_link_libraries(hera_block2_he_tests PRIVATE hera_he)
    target_compile_options(hera_block2_he_tests PRIVATE
        -Wall -Wextra -Wpedantic)
    target_compile_definitions(hera_block2_he_tests PRIVATE
        HERA_TEST_CONFIG_PATH="${CMAKE_CURRENT_SOURCE_DIR}/configs/figure10_smoke.json")
    add_test(NAME hera_block2_he_tests COMMAND hera_block2_he_tests)
    set_tests_properties(hera_block2_he_tests PROPERTIES
        TIMEOUT 300
        PASS_REGULAR_EXPRESSION "HERA_BLOCK2_HE_STATUS=PASS")
endif()

if(HERA_ENABLE_SOFTMAX)
    target_compile_definitions(hera_private_inference PRIVATE
        HERA_SOFTMAX_TO_MPC_BUILT=1)
    add_library(hera_softmax STATIC he/softmax.cpp)
    target_include_directories(hera_softmax PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_features(hera_softmax PUBLIC cxx_std_17)
    target_link_libraries(hera_softmax PUBLIC
        hera_he hera_conversion SCI::SCI-FloatingPoint
        SEAL::seal Threads::Threads)
    target_compile_options(hera_softmax PRIVATE
        -Wall -Wextra -Wpedantic -Wno-pedantic)

    add_executable(hera_softmax_tests tests/softmax_tests.cpp)
    target_link_libraries(hera_softmax_tests PRIVATE hera_softmax)
    target_compile_options(hera_softmax_tests PRIVATE
        -Wall -Wextra -Wpedantic -Wno-pedantic)
    target_compile_definitions(hera_softmax_tests PRIVATE
        HERA_TEST_CONFIG_PATH="${CMAKE_CURRENT_SOURCE_DIR}/configs/figure10_smoke.json")
    add_test(NAME hera_softmax_tests COMMAND hera_softmax_tests)
    set_tests_properties(hera_softmax_tests PROPERTIES
        TIMEOUT 300
        RESOURCE_LOCK hera_softmax_ports
        PASS_REGULAR_EXPRESSION "HERA_SOFTMAX_STATUS=PASS")
    add_test(NAME hera_softmax_mpc_tests COMMAND hera_softmax_tests)
    set_tests_properties(hera_softmax_mpc_tests PROPERTIES
        TIMEOUT 180
        RESOURCE_LOCK hera_softmax_ports
        ENVIRONMENT "HERA_SOFTMAX_MPC_ONLY=1"
        PASS_REGULAR_EXPRESSION "HERA_SOFTMAX_MPC_STATUS=PASS")
endif()

if(HERA_ENABLE_ATTENTION_LN2)
    target_compile_definitions(hera_private_inference PRIVATE
        HERA_ATTENTION_LN2_BUILT=1)
    add_library(hera_attention_ln2 STATIC he/attention_ln2.cpp)
    target_include_directories(hera_attention_ln2 PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_features(hera_attention_ln2 PUBLIC cxx_std_17)
    target_link_libraries(hera_attention_ln2 PUBLIC
        hera_softmax hera_conversion SCI::SCI-FloatingPoint
        SEAL::seal Threads::Threads)
    target_compile_options(hera_attention_ln2 PRIVATE
        -Wall -Wextra -Wpedantic -Wno-pedantic)

    add_executable(hera_attention_ln2_tests
        tests/attention_ln2_tests.cpp)
    target_link_libraries(hera_attention_ln2_tests PRIVATE
        hera_attention_ln2)
    target_compile_options(hera_attention_ln2_tests PRIVATE
        -Wall -Wextra -Wpedantic -Wno-pedantic)
    add_test(NAME hera_attention_ln2_tests
             COMMAND hera_attention_ln2_tests)
    set_tests_properties(hera_attention_ln2_tests PROPERTIES
        TIMEOUT 900
        RESOURCE_LOCK hera_attention_ln2_ports
        PASS_REGULAR_EXPRESSION "ATTENTION_LN2_STATUS=PASS")
endif()

if(HERA_ENABLE_FFN_GELU)
    target_compile_definitions(hera_private_inference PRIVATE
        HERA_FFN_GELU_BUILT=1)
    add_library(hera_ffn_gelu STATIC he/ffn_gelu.cpp)
    target_include_directories(hera_ffn_gelu PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_features(hera_ffn_gelu PUBLIC cxx_std_17)
    target_link_libraries(hera_ffn_gelu PUBLIC
        hera_he hera_conversion SCI::SCI-FloatingPoint
        SEAL::seal Threads::Threads)
    target_compile_options(hera_ffn_gelu PRIVATE
        -Wall -Wextra -Wpedantic -Wno-pedantic)

    add_executable(hera_ffn_gelu_tests tests/ffn_gelu_tests.cpp)
    target_link_libraries(hera_ffn_gelu_tests PRIVATE hera_ffn_gelu)
    target_compile_options(hera_ffn_gelu_tests PRIVATE
        -Wall -Wextra -Wpedantic -Wno-pedantic)
    add_test(NAME hera_ffn_gelu_tests COMMAND hera_ffn_gelu_tests)
    set_tests_properties(hera_ffn_gelu_tests PROPERTIES
        TIMEOUT 900
        RESOURCE_LOCK hera_ffn_gelu_ports
        PASS_REGULAR_EXPRESSION "FFN_GELU_STATUS=PASS")
endif()

if(HERA_ENABLE_BLOCK5_BRIDGE)
    target_compile_definitions(hera_private_inference PRIVATE
        HERA_BLOCK5_BRIDGE_BUILT=1)
    add_library(hera_block5_bridge STATIC he/block5_bridge.cpp)
    target_include_directories(hera_block5_bridge PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_features(hera_block5_bridge PUBLIC cxx_std_17)
    target_link_libraries(hera_block5_bridge PUBLIC
        hera_ffn_gelu hera_attention_ln2 hera_he hera_conversion
        SCI::SCI-FloatingPoint SEAL::seal Threads::Threads)
    target_compile_options(hera_block5_bridge PRIVATE
        -Wall -Wextra -Wpedantic -Wno-pedantic)

    add_library(hera_private_runtime STATIC
        runtime/private_runtime.cpp
        runtime/runtime_coordinator.cpp
        app/private_runtime_cli.cpp)
    target_include_directories(hera_private_runtime PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_features(hera_private_runtime PUBLIC cxx_std_17)
    target_link_libraries(hera_private_runtime PUBLIC
        hera_block5_bridge hera_ffn_gelu hera_attention_ln2 hera_softmax
        hera_he hera_conversion SCI::SCI-FloatingPoint SEAL::seal
        Threads::Threads)
    target_compile_options(hera_private_runtime PRIVATE
        -Wall -Wextra -Wpedantic -Wno-pedantic)
    target_link_libraries(hera_private_inference PRIVATE hera_private_runtime)
    target_compile_definitions(hera_private_inference PRIVATE
        HERA_PRIVATE_RUNTIME_BUILT=1)

    add_executable(hera_private_runtime_tests
        tests/private_runtime_tests.cpp)
    target_link_libraries(hera_private_runtime_tests PRIVATE
        hera_private_runtime)
    target_compile_options(hera_private_runtime_tests PRIVATE
        -Wall -Wextra -Wpedantic -Wno-pedantic)
    add_test(NAME hera_private_runtime_tests
             COMMAND hera_private_runtime_tests)
    set_tests_properties(hera_private_runtime_tests PROPERTIES
        TIMEOUT 900
        RESOURCE_LOCK hera_private_runtime_ports
        PASS_REGULAR_EXPRESSION "HERA_PRIVATE_RUNTIME_STATUS=PASS")

    add_executable(hera_runtime_coordinator_tests
        tests/runtime_coordinator_tests.cpp)
    target_link_libraries(hera_runtime_coordinator_tests PRIVATE
        hera_private_runtime)
    target_compile_options(hera_runtime_coordinator_tests PRIVATE
        -Wall -Wextra -Wpedantic -Wno-pedantic)
    add_test(NAME hera_runtime_coordinator_tests
             COMMAND hera_runtime_coordinator_tests)
    set_tests_properties(hera_runtime_coordinator_tests PROPERTIES
        TIMEOUT 20
        RESOURCE_LOCK hera_private_runtime_ports
        PASS_REGULAR_EXPRESSION "HERA_RUNTIME_COORDINATOR_STATUS=PASS")

    add_executable(hera_block5_bridge_tests
        tests/block5_bridge_tests.cpp)
    target_link_libraries(hera_block5_bridge_tests PRIVATE
        hera_block5_bridge)
    target_compile_options(hera_block5_bridge_tests PRIVATE
        -Wall -Wextra -Wpedantic -Wno-pedantic)
    target_compile_definitions(hera_block5_bridge_tests PRIVATE
        HERA_TEST_CONFIG_PATH="${CMAKE_CURRENT_SOURCE_DIR}/configs/figure10_smoke.json")
    add_test(NAME hera_block5_bridge_tests
             COMMAND hera_block5_bridge_tests)
    set_tests_properties(hera_block5_bridge_tests PROPERTIES
        TIMEOUT 300
        RESOURCE_LOCK hera_block5_ports
        PASS_REGULAR_EXPRESSION "BLOCK5_BRIDGE_STATUS=PASS")
endif()
