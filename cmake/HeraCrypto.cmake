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
endif()

if(HERA_ENABLE_HE)
    add_library(hera_he STATIC ${HERA_CRYPTO_ROOT}/he/block2_he.cpp)
    hera_configure_target(hera_he)
    target_link_libraries(hera_he PUBLIC
        hera_framework hera_seal_runtime SEAL::seal Threads::Threads)
endif()

if(HERA_ENABLE_SOFTMAX)
    target_compile_definitions(hera_private_inference PRIVATE
        HERA_SOFTMAX_TO_MPC_BUILT=1)
    add_library(hera_softmax STATIC ${HERA_CRYPTO_ROOT}/he/softmax.cpp)
    hera_configure_target(hera_softmax WIDE_INTEGER)
    target_link_libraries(hera_softmax PUBLIC
        hera_he hera_conversion SCI::SCI-FloatingPoint SEAL::seal
        Threads::Threads)
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
endif()

if(HERA_ENABLE_FFN_GELU)
    target_compile_definitions(hera_private_inference PRIVATE
        HERA_FFN_GELU_BUILT=1)
    add_library(hera_ffn_gelu STATIC ${HERA_CRYPTO_ROOT}/he/ffn_gelu.cpp)
    hera_configure_target(hera_ffn_gelu WIDE_INTEGER)
    target_link_libraries(hera_ffn_gelu PUBLIC
        hera_he hera_conversion SCI::SCI-FloatingPoint SEAL::seal
        Threads::Threads)

    # Manual kernel measurements; kept outside the default correctness suite.
    if(TARGET hera_attention_ln2)
        add_executable(hera_he_benchmark ${PROJECT_SOURCE_DIR}/app/he_benchmark.cpp)
        hera_configure_target(hera_he_benchmark WIDE_INTEGER)
        target_link_libraries(hera_he_benchmark PRIVATE hera_ffn_gelu hera_attention_ln2)
        target_compile_definitions(hera_he_benchmark PRIVATE
            HERA_DEFAULT_CONFIG_PATH="${PROJECT_SOURCE_DIR}/configs/runtime_tiny.json")
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

endif()
