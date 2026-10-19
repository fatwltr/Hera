# Full-runtime suite: core correctness/contract/end-to-end tests.

hera_add_test_executable(hera_framework_tests
    ${PROJECT_SOURCE_DIR}/tests/framework_tests.cpp
    LIBRARIES hera_framework
    DEFINITIONS HERA_TEST_CONFIG_PATH="${PROJECT_SOURCE_DIR}/configs/figure10_smoke.json"
    PASS "HERA_FRAMEWORK_STATUS=PASS")

if(TARGET hera_conversion)
    foreach(kind unit compare carry protocol)
        string(TOUPPER "${kind}" kind_upper)
        hera_add_test_executable(hera_conversion_${kind}_tests
            ${PROJECT_SOURCE_DIR}/tests/conversion_${kind}_tests.cpp
            WIDE_INTEGER LIBRARIES hera_conversion
            PASS "HERA_CONVERSION_${kind_upper}_STATUS=PASS" TIMEOUT 180
            RESOURCE hera_conversion_ports)
        target_include_directories(hera_conversion_${kind}_tests PRIVATE ${HERA_CRYPTO_ROOT}/conversion)
    endforeach()
    hera_add_test_executable(hera_blb_reference_tests
        ${PROJECT_SOURCE_DIR}/tests/blb_reference_tests.cpp
        WIDE_INTEGER LIBRARIES hera_conversion
        PASS "HERA_BLB_REFERENCE_STATUS=PASS" TIMEOUT 180 RESOURCE hera_conversion_ports)
    target_include_directories(hera_blb_reference_tests PRIVATE ${HERA_CRYPTO_ROOT}/conversion)
    hera_add_test_executable(hera_packed_conversion_tests
        ${PROJECT_SOURCE_DIR}/tests/packed_conversion_tests.cpp
        WIDE_INTEGER LIBRARIES hera_conversion
        PASS "HERA_PACKED_CONVERSION_STATUS=PASS" TIMEOUT 180 RESOURCE hera_conversion_ports)
    hera_add_test_executable(hera_mpc_session_tests
        ${PROJECT_SOURCE_DIR}/tests/mpc_session_tests.cpp
        WIDE_INTEGER LIBRARIES hera_conversion
        PASS "HERA_MPC_SESSION_STATUS=PASS" TIMEOUT 120 RESOURCE hera_mpc_session_ports)
    set(HERA_TEST_SEAL_COMPRESSION none)
    if(SEAL_USE_ZSTD)
        set(HERA_TEST_SEAL_COMPRESSION zstd)
    elseif(SEAL_USE_ZLIB)
        set(HERA_TEST_SEAL_COMPRESSION zlib)
    endif()
endif()

if(TARGET hera_he)
    hera_add_test_executable(hera_block2_he_tests
        ${PROJECT_SOURCE_DIR}/tests/block2_he_tests.cpp
        LIBRARIES hera_he
        DEFINITIONS HERA_TEST_CONFIG_PATH="${PROJECT_SOURCE_DIR}/configs/figure10_smoke.json"
        PASS "HERA_BLOCK2_HE_STATUS=PASS" TIMEOUT 300)
endif()

if(TARGET hera_softmax)
    hera_add_test_executable(hera_softmax_tests
        ${PROJECT_SOURCE_DIR}/tests/softmax_tests.cpp
        WIDE_INTEGER LIBRARIES hera_softmax
        DEFINITIONS HERA_TEST_CONFIG_PATH="${PROJECT_SOURCE_DIR}/configs/figure10_smoke.json"
        PASS "HERA_SOFTMAX_STATUS=PASS" TIMEOUT 300 RESOURCE hera_softmax_ports)
    add_test(NAME hera_softmax_mpc_tests COMMAND hera_softmax_tests)
    set_tests_properties(hera_softmax_mpc_tests PROPERTIES
        TIMEOUT 180 RESOURCE_LOCK hera_softmax_ports
        ENVIRONMENT "HERA_SOFTMAX_MPC_ONLY=1" PASS_REGULAR_EXPRESSION "HERA_SOFTMAX_MPC_STATUS=PASS")
endif()

if(TARGET hera_attention_ln2)
    hera_add_test_executable(hera_attention_ln2_tests
        ${PROJECT_SOURCE_DIR}/tests/attention_ln2_tests.cpp
        WIDE_INTEGER LIBRARIES hera_attention_ln2
        PASS "ATTENTION_LN2_STATUS=PASS" TIMEOUT 900 RESOURCE hera_attention_ln2_ports)
    hera_add_test_executable(hera_ln_rsqrt_contract_tests
        ${PROJECT_SOURCE_DIR}/tests/layer_norm_rsqrt_contract_tests.cpp
        WIDE_INTEGER LIBRARIES hera_attention_ln2
        PASS "RSQRT_CONTRACT_STATUS=PASS" TIMEOUT 180 RESOURCE hera_ln_precision_ports)
endif()

if(TARGET hera_ffn_gelu)
    hera_add_test_executable(hera_ffn_gelu_tests
        ${PROJECT_SOURCE_DIR}/tests/ffn_gelu_tests.cpp
        WIDE_INTEGER LIBRARIES hera_ffn_gelu
        PASS "FFN_GELU_STATUS=PASS" TIMEOUT 900 RESOURCE hera_ffn_gelu_ports)
endif()

if(TARGET hera_private_runtime)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    hera_add_test_executable(hera_private_runtime_tests
        ${PROJECT_SOURCE_DIR}/tests/private_runtime_tests.cpp
        WIDE_INTEGER LIBRARIES hera_private_runtime
        PASS "HERA_PRIVATE_RUNTIME_STATUS=PASS" TIMEOUT 900 RESOURCE hera_private_runtime_ports)
    # SCI must execute both communication directions even when OpenMP creates
    # only one thread (common on constrained VMs or with OMP_DYNAMIC enabled).
    add_test(NAME hera_private_runtime_single_thread_tests COMMAND hera_private_runtime_tests)
    set_tests_properties(hera_private_runtime_single_thread_tests PROPERTIES
        TIMEOUT 900 RESOURCE_LOCK hera_private_runtime_ports
        ENVIRONMENT "OMP_NUM_THREADS=1;OMP_THREAD_LIMIT=1;OMP_DYNAMIC=TRUE"
        PASS_REGULAR_EXPRESSION "HERA_PRIVATE_RUNTIME_STATUS=PASS")
    hera_add_test_executable(hera_runtime_coordinator_tests
        ${PROJECT_SOURCE_DIR}/tests/runtime_coordinator_tests.cpp
        WIDE_INTEGER LIBRARIES hera_private_runtime
        PASS "HERA_RUNTIME_COORDINATOR_STATUS=PASS" TIMEOUT 20 RESOURCE hera_private_runtime_ports)
    add_test(NAME hera_private_cli_tests COMMAND ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/tests/check_local_cli.py $<TARGET_FILE:hera_private_inference>
        --config=${PROJECT_SOURCE_DIR}/configs/runtime_tiny.json
        --party=local --port=28100 --threads=2 --verify --seal-compression=${HERA_TEST_SEAL_COMPRESSION})
    set_tests_properties(hera_private_cli_tests PROPERTIES TIMEOUT 900
        PASS_REGULAR_EXPRESSION "HERA_PRIVATE_INFERENCE_STATUS=PASS" RESOURCE_LOCK hera_private_runtime_ports)
    add_test(NAME hera_private_cli_virtual_tests COMMAND ${Python3_EXECUTABLE}
        ${PROJECT_SOURCE_DIR}/tests/check_local_cli.py $<TARGET_FILE:hera_private_inference>
        --config=${PROJECT_SOURCE_DIR}/configs/runtime_tiny.json
        --party=local --weight-mode=virtual --port=27300 --threads=2 --verify
        --seal-compression=${HERA_TEST_SEAL_COMPRESSION})
    set_tests_properties(hera_private_cli_virtual_tests PROPERTIES TIMEOUT 900
        PASS_REGULAR_EXPRESSION "HERA_PRIVATE_INFERENCE_STATUS=PASS"
        RESOURCE_LOCK hera_private_runtime_ports)
endif()

# Keep direct CTest/preset runs reproducible on servers with many logical CPUs.
get_property(hera_tests DIRECTORY PROPERTY TESTS)
foreach(test_name IN LISTS hera_tests)
    if(NOT test_name STREQUAL "hera_private_runtime_single_thread_tests")
        set_property(TEST ${test_name} APPEND PROPERTY ENVIRONMENT
            "OMP_NUM_THREADS=2" "OMP_THREAD_LIMIT=2"
            "OMP_MAX_ACTIVE_LEVELS=1" "OMP_DYNAMIC=FALSE")
    endif()
    set_property(TEST ${test_name} APPEND PROPERTY FAIL_REGULAR_EXPRESSION "STATUS=FAIL")
endforeach()
