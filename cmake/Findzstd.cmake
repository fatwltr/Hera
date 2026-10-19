# Ubuntu 22.04's libzstd-dev provides headers/a static archive but no CMake
# package config. This module also ships with our generated SEAL package.
find_path(zstd_INCLUDE_DIR NAMES zstd.h)
find_library(zstd_LIBRARY NAMES libzstd.a)
if(zstd_INCLUDE_DIR)
    foreach(component MAJOR MINOR RELEASE)
        file(STRINGS "${zstd_INCLUDE_DIR}/zstd.h" version_line
            REGEX "^#define ZSTD_VERSION_${component} +[0-9]+")
        string(REGEX MATCH "[0-9]+$" zstd_${component} "${version_line}")
    endforeach()
    set(zstd_VERSION "${zstd_MAJOR}.${zstd_MINOR}.${zstd_RELEASE}")
endif()
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(zstd
    REQUIRED_VARS zstd_INCLUDE_DIR zstd_LIBRARY VERSION_VAR zstd_VERSION)
if(zstd_FOUND AND NOT TARGET zstd::libzstd_static)
    add_library(zstd::libzstd_static STATIC IMPORTED)
    set_target_properties(zstd::libzstd_static PROPERTIES
        IMPORTED_LOCATION "${zstd_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${zstd_INCLUDE_DIR}")
endif()
mark_as_advanced(zstd_INCLUDE_DIR zstd_LIBRARY)
