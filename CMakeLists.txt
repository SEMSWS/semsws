cmake_minimum_required(VERSION 3.20)

# C++ standard MUST be set before project() / find_package() so that every
# imported target (MFEM, adios2, HDF5) and every downstream .cpp inherits
# C++17. MFEM's dfem/ headers use CTAD + std::string_view / std::variant
# and fail to parse under C++14 on NVHPC.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(SEMSWS LANGUAGES C CXX VERSION 0.1.0)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
if(NOT DEFINED CMAKE_INSTALL_RPATH)
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "CMAKE_BUILD_TYPE not set; defaulting to Release.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING
        "Build type (Debug | Release | RelWithDebInfo | MinSizeRel)" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

include(GNUInstallDirs)

add_subdirectory(src)

option(SEMSWS_BUILD_TESTS "Build C++ unit tests" ON)
if(SEMSWS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test/unit)
endif()
