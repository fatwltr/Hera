cmake_minimum_required(VERSION 3.21)

project(Hera
    VERSION 0.1.0
    DESCRIPTION "Hybrid CKKS/MPC private Transformer inference research framework"
    LANGUAGES CXX)

include(CTest)

option(HERA_ENABLE_FULL_RUNTIME
       "Build every cryptographic stage and the end-to-end runtime" OFF)
option(HERA_ENABLE_CONVERSION
       "Build the independent SEAL/HERA-SCI conversion protocol" OFF)
option(HERA_ENABLE_HE
       "Build the SEAL CKKS Block-2 implementation" OFF)
option(HERA_ENABLE_SOFTMAX "Build the staged Softmax implementation" OFF)
option(HERA_ENABLE_ATTENTION_LN2
       "Build Attention x V through LayerNorm reciprocal-sqrt" OFF)
option(HERA_ENABLE_FFN_GELU
       "Build LayerNorm tail through the GeLU MPC boundary" OFF)
option(HERA_ENABLE_BLOCK5_BRIDGE
       "Build Wd/residual and the cross-layer bridge" OFF)

# Focused component switches are kept for kernel development. Normal users
# only need HERA_ENABLE_FULL_RUNTIME.
if(HERA_ENABLE_FULL_RUNTIME)
    set(HERA_ENABLE_BLOCK5_BRIDGE ON)
endif()
if(HERA_ENABLE_BLOCK5_BRIDGE)
    set(HERA_ENABLE_FFN_GELU ON)
    set(HERA_ENABLE_ATTENTION_LN2 ON)
endif()
if(HERA_ENABLE_ATTENTION_LN2 OR HERA_ENABLE_FFN_GELU)
    set(HERA_ENABLE_SOFTMAX ON)
endif()
if(HERA_ENABLE_SOFTMAX OR HERA_ENABLE_ATTENTION_LN2 OR HERA_ENABLE_FFN_GELU OR
   HERA_ENABLE_BLOCK5_BRIDGE)
    set(HERA_ENABLE_CONVERSION ON)
    set(HERA_ENABLE_HE ON)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(HeraHelpers)
include(HeraFramework)
include(HeraCrypto)
if(BUILD_TESTING)
    include("${PROJECT_SOURCE_DIR}/cmake/HeraTests.cmake")
endif()
