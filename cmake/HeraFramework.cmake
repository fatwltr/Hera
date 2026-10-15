set(HERA_FRAMEWORK_ROOT "${PROJECT_SOURCE_DIR}/src/framework")

add_library(hera_framework STATIC
    ${HERA_FRAMEWORK_ROOT}/config/json_loader.cpp
    ${HERA_FRAMEWORK_ROOT}/config/validator.cpp
    ${HERA_FRAMEWORK_ROOT}/core/model.cpp
    ${HERA_FRAMEWORK_ROOT}/core/tensor.cpp
    ${HERA_FRAMEWORK_ROOT}/graph/graph.cpp
    ${HERA_FRAMEWORK_ROOT}/graph/graph_validator.cpp
    ${HERA_FRAMEWORK_ROOT}/graph/transformer_graph_builder.cpp
    ${HERA_FRAMEWORK_ROOT}/modules/layer_norm.cpp
    ${HERA_FRAMEWORK_ROOT}/modules/attention.cpp
    ${HERA_FRAMEWORK_ROOT}/modules/softmax.cpp
    ${HERA_FRAMEWORK_ROOT}/modules/activation.cpp
    ${HERA_FRAMEWORK_ROOT}/modules/feed_forward.cpp
    ${HERA_FRAMEWORK_ROOT}/runtime/hera_context.cpp
    ${HERA_FRAMEWORK_ROOT}/runtime/executor.cpp
    ${HERA_FRAMEWORK_ROOT}/backend/unavailable_backend.cpp
    ${HERA_FRAMEWORK_ROOT}/backend/recording_backend.cpp)
hera_configure_target(hera_framework)

add_executable(hera_private_inference ${PROJECT_SOURCE_DIR}/app/main.cpp)
target_link_libraries(hera_private_inference PRIVATE hera_framework)
hera_configure_target(hera_private_inference)
target_compile_definitions(hera_private_inference PRIVATE
    HERA_DEFAULT_CONFIG_PATH="${PROJECT_SOURCE_DIR}/configs/figure10_smoke.json")
