# Minimum required CMake version
cmake_minimum_required(VERSION 3.20)

# Set compiler-specific optimization flags. -fopt-info-vec-optimized prints
# a line for every successful auto-vectorisation, which floods CI and Spack
# build logs; gate it behind an opt-in toggle.
option(SEM_VEC_INFO "Enable GCC -fopt-info-vec-optimized output" OFF)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "-O2 -march=native")
    if(SEM_VEC_INFO)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopt-info-vec-optimized")
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC" OR CMAKE_CXX_COMPILER_ID STREQUAL "PGI")
    set(CMAKE_CXX_FLAGS "-O2 -tp=native")
else()
    set(CMAKE_CXX_FLAGS "-O2")
endif()

# Note: OpenMP is intentionally NOT used (conflicts with MFEM's device
# abstraction and with MPI on multi-GPU nodes).

# yaml-cpp for YAML configuration file parsing.
#
# yaml-cpp 0.9.0 or newer is required. 0.9.0 ships the `#include <cstdint>`
# fix (jbeder/yaml-cpp@7b469b4) that older 0.7/0.8 are missing — without
# that include, manual builds on GCC 15+ fail with `uint16_t was not
# declared`. The version requirement also doubles as protection against
# unrelated system installs (e.g. /mnt/c/msys64/mingw64 yaml-cpp on WSL2
# typically ships 0.6/0.7 and is rejected here by version, so its include
# tree never enters the SEMSWS compile path).
#
# Spack provides yaml-cpp@0.9.0 via the spack-repo/packages/yaml-cpp shadow;
# manual builds need to install it themselves (`apt`, source build, etc.).
find_package(yaml-cpp 0.9.0 REQUIRED)
message(STATUS "yaml-cpp via find_package: ${yaml-cpp_DIR}")

# =============================================================================
# Find MFEM. Requires a CMake-built MFEM that ships MFEMConfig.cmake.
# =============================================================================
find_package(MFEM CONFIG REQUIRED PATHS ${MFEM_DIR})
# SEMSWS uses <mfem.hpp> exclusively; the umbrella only pulls in forall.hpp
# and linalg/tensor.hpp (transitively dual.hpp) starting in MFEM 4.9.
if(MFEM_VERSION AND MFEM_VERSION VERSION_LESS 4.9)
    message(FATAL_ERROR
        "SEMSWS requires MFEM >= 4.9. Found MFEM ${MFEM_VERSION}.")
endif()

if(MFEM_USE_UMPIRE)
    find_package(umpire CONFIG REQUIRED)
    message(STATUS "umpire via find_package: ${umpire_DIR}")
endif()

# =============================================================================
# Detect MFEM precision (single or double)
# MFEM exports MFEM_USE_SINGLE and MFEM_USE_DOUBLE from MFEMConfig.cmake
# =============================================================================
if(MFEM_USE_SINGLE)
    set(SEM_PRECISION "single")
    message(STATUS "MFEM precision: SINGLE (real_t = float)")
    message(STATUS "  -> SEMSWS will use single precision for all computations")
elseif(MFEM_USE_DOUBLE)
    set(SEM_PRECISION "double")
    message(STATUS "MFEM precision: DOUBLE (real_t = double)")
    message(STATUS "  -> SEMSWS will use double precision for all computations")
else()
    # Fallback: assume double precision
    set(SEM_PRECISION "double")
    message(WARNING "Could not detect MFEM precision, assuming DOUBLE")
endif()

#find MPI package
find_package(MPI REQUIRED)

# =============================================================================
# ADIOS2 (required for FWI kernel/model I/O)
# =============================================================================
find_package(adios2 REQUIRED)
message(STATUS "ADIOS2: Found (FWI kernel/model I/O via BP5)")

# HDF5 (C API only — no C++ bindings required; keeps dependencies portable
# across distros where libhdf5-cpp-dev ships separately). Must be the same
# HDF5 that MFEM was built against; point HDF5_ROOT at that prefix.
find_package(HDF5 REQUIRED COMPONENTS C)
# Cray-HDF5 fallback: cray-hdf5's h5cc wrapper relies on CPATH and doesn't
# emit -I flags, so FindHDF5 can fail to populate HDF5_INCLUDE_DIRS even when
# the library is found. Recover the include dir by resolving <libdir>/../include
# from the located libhdf5 (e.g. /opt/cray/pe/hdf5/.../lib → .../include).
if(NOT HDF5_INCLUDE_DIRS AND HDF5_C_LIBRARY_hdf5)
    get_filename_component(_sem_hdf5_libdir "${HDF5_C_LIBRARY_hdf5}" DIRECTORY)
    get_filename_component(_sem_hdf5_root  "${_sem_hdf5_libdir}"     DIRECTORY)
    if(EXISTS "${_sem_hdf5_root}/include/hdf5.h")
        set(HDF5_INCLUDE_DIRS "${_sem_hdf5_root}/include")
        message(STATUS "HDF5 include dir not reported by FindHDF5; "
                       "recovered from library path: ${HDF5_INCLUDE_DIRS}")
    endif()
endif()
if(NOT HDF5_INCLUDE_DIRS)
    message(FATAL_ERROR
        "find_package(HDF5) succeeded but HDF5_INCLUDE_DIRS is empty. "
        "Re-run cmake with -DHDF5_ROOT=<prefix containing include/hdf5.h>.")
endif()
message(STATUS "HDF5 include dir: ${HDF5_INCLUDE_DIRS}")
message(STATUS "HDF5 libraries:   ${HDF5_LIBRARIES}")

set(GSLIB_DIR ${GSLIB_DIR})
message("gslib dir is ${GSLIB_DIR}")

message(STATUS "MFEM_DIR: ${MFEM_DIR}")
get_target_property(_sem_mfem_inc mfem INTERFACE_INCLUDE_DIRECTORIES)
message(STATUS "mfem includes: ${_sem_mfem_inc}")

message(STATUS "MPI include directories: ${MPI_INCLUDE_PATH}")
message(STATUS "MPI libraries: ${MPI_LIBRARIES}")

# =============================================================================
# GPU configuration (CUDA or HIP, following MFEM's approach)
# When MFEM is built with CUDA/HIP, we compile ALL sources with nvcc/hipcc
# =============================================================================

# Option to force CPU-only build even when MFEM has GPU support
# This is useful because nvcc generates slower host code than g++
# Use: cmake .. -DSEM_FORCE_CPU=ON
option(SEM_FORCE_CPU "Force CPU-only build even if MFEM has CUDA/HIP" OFF)

# =============================================================================
# Profiling support
# Enable with: cmake .. -DSEM_ENABLE_PROFILING=ON
# =============================================================================
option(SEM_ENABLE_PROFILING "Enable per-step timing profiler (adds MFEM_DEVICE_SYNC overhead)" OFF)
option(SEM_BUILD_COUPLING_PROBES "Build fluid-solid coupling probe executables (dev-only)" OFF)

# GPU-aware MPI assertion. Default OFF: SetGPUAwareMPI is not called, so the
# fluid-solid coupling stages buffers through host (correct on any MPI).
# Enable only when the linked MPI is verified GPU-aware (CUDA-aware OpenMPI,
# Cray MPICH with MPICH_GPU_SUPPORT_ENABLED=1, etc.).
option(SEM_USE_GPU_AWARE_MPI "Assert that the linked MPI is GPU-aware (CUDA/ROCm)." OFF)

if(SEM_ENABLE_PROFILING)
    message(STATUS "")
    message(STATUS "========================================")
    message(STATUS "SEM profiling: ENABLED")
    message(STATUS "  -> Per-region timing with GPU sync")
    message(STATUS "  -> Output at simulation end")
    message(STATUS "  -> Disable for production runs")
    message(STATUS "========================================")
    message(STATUS "")
    add_compile_definitions(SEM_ENABLE_PROFILING)
else()
    message(STATUS "SEM profiling: DISABLED (zero overhead)")
endif()

if(SEM_FORCE_CPU AND (MFEM_USE_CUDA OR MFEM_USE_HIP))
    message(STATUS "")
    message(STATUS "========================================")
    message(STATUS "SEM_FORCE_CPU=ON: Building CPU-optimized version")
    message(STATUS "  -> GPU kernels will NOT be available")
    message(STATUS "  -> Use 'device: cpu' in YAML config")
    message(STATUS "========================================")
    message(STATUS "")
    set(SEM_USE_CUDA OFF)
    set(SEM_USE_HIP OFF)
elseif(MFEM_USE_HIP)
    # =========================================================================
    # HIP configuration (AMD GPU via hipcc)
    # hipcc compiles .cpp files with HIP device support automatically.
    # No need to change source language (unlike CUDA/nvcc).
    # MFEM_FORALL macros generate HIP kernels when compiled with hipcc.
    # =========================================================================

    # Set HIP architectures if not specified
    if(NOT DEFINED AMDGPU_TARGETS)
        set(AMDGPU_TARGETS "gfx90a" CACHE STRING "AMD GPU architectures")
    endif()

    message(STATUS "")
    message(STATUS "========================================")
    message(STATUS "MFEM HIP support: ENABLED")
    message(STATUS "  -> All sources compiled with hipcc")
    message(STATUS "  -> AMD GPU target: ${AMDGPU_TARGETS}")
    message(STATUS "========================================")
    message(STATUS "")

    set(SEM_USE_HIP ON)
    set(SEM_USE_CUDA OFF)
    # ROCm 6.4+ hip headers require __HIP_PLATFORM_AMD__ to be defined when
    # invoked from a non-hipcc compiler (e.g. crayCC under a Spack build).
    # hipcc would set this automatically; here we declare it explicitly so
    # consumers like mfem.hpp -> hip_runtime.h compile correctly under any
    # CXX compiler that links against HIP runtime.
    add_compile_definitions(__HIP_PLATFORM_AMD__=1)
    # When CMAKE_CXX_COMPILER is hipcc, the HIP runtime (libamdhip64.so) is
    # added automatically; under crayCC / clang-direct compiles we need to
    # link it explicitly so MFEM's hipMalloc / __hipPushCallConfiguration
    # references resolve at executable link time.
    find_package(hip REQUIRED CONFIG)
    link_libraries(hip::host)
elseif(MFEM_USE_CUDA)
    # Enable CUDA language
    enable_language(CUDA)

    # Locate the CUDAToolkit::* imported targets (CUDA::cudart, CUDA::cusparse,
    # etc.). MFEM-static builds reference cudaStreamSynchronize / cudaMalloc
    # from libmfem.a, but the static archive carries no DT_NEEDED metadata so
    # the consumer link line must add libcudart explicitly. enable_language
    # alone doesn't expose CUDA::cudart — that requires this find_package.
    find_package(CUDAToolkit REQUIRED)

    # Set CUDA host compiler to match CXX compiler
    if (NOT CMAKE_CUDA_HOST_COMPILER)
        set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    endif()

    # Set CUDA standard to match C++ standard.
    # Note: CMAKE_CUDA_STANDARD alone is NOT always sufficient — on some
    # NVHPC toolchains nvcc ignores it for the host pass. Append the flag
    # explicitly as well so nvcc forwards `-std=c++17` to the host
    # compiler and MFEM's dfem headers (which use CTAD +
    # std::string_view / std::variant) parse correctly.
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17")

    # Set CUDA architectures if not specified (default to sm_70, sm_80)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "70;80" CACHE STRING "CUDA architectures")
    endif()

    # =========================================================================
    # CUDA Architecture Validation
    # With MFEM_USE_SINGLE (float precision), CUDA architectures < 60 cause
    # atomicAdd conflict: CUDA already provides atomicAdd(float*, float) but
    # MFEM's backends.hpp tries to redefine it for __CUDA_ARCH__ < 600.
    # Require sm_60+ (Pascal or later) to avoid this issue.
    # =========================================================================
    set(MIN_CUDA_ARCH 60)
    foreach(arch ${CMAKE_CUDA_ARCHITECTURES})
        # Extract numeric part (handle formats like "60", "70-real", "80-virtual")
        string(REGEX MATCH "^[0-9]+" arch_num "${arch}")
        if(arch_num)
            if(arch_num LESS ${MIN_CUDA_ARCH})
                message(FATAL_ERROR
                    "CUDA architecture ${arch} (sm_${arch_num}) is not supported!\n"
                    "Minimum required: sm_${MIN_CUDA_ARCH} (Pascal or later)\n"
                    "Reason: MFEM with single precision (MFEM_USE_SINGLE) has atomicAdd "
                    "conflict on architectures < sm_60.\n"
                    "Fix: Use -DCMAKE_CUDA_ARCHITECTURES=60 or higher (e.g., 70, 80, 90)")
            endif()
        endif()
    endforeach()
    message(STATUS "CUDA architecture validation: PASSED (all >= sm_${MIN_CUDA_ARCH})")

    # CUDA flags for extended lambda and relaxed constexpr (required by MFEM)
    if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")
    endif()

    message(STATUS "MFEM CUDA support: ENABLED")
    message(STATUS "  -> All sources will be compiled with nvcc")
    message(STATUS "  -> CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    message(STATUS "  -> CUDA host compiler: ${CMAKE_CUDA_HOST_COMPILER}")

    set(SEM_USE_CUDA ON)
    set(SEM_USE_HIP OFF)
else()
    message(STATUS "MFEM GPU support: DISABLED (CPU-only build)")
    set(SEM_USE_CUDA OFF)
    set(SEM_USE_HIP OFF)
endif()

# =============================================================================
# libCEED configuration (optional, for portable GPU/CPU kernels)
# =============================================================================

option(SEM_USE_LIBCEED "Enable libCEED-based integrators for GPU/CPU portability" OFF)

if(SEM_USE_LIBCEED)
    # Try to find libCEED
    find_package(PkgConfig)
    if(PkgConfig_FOUND)
        pkg_check_modules(CEED IMPORTED_TARGET ceed)
    endif()

    # Also try CEED_DIR if specified
    if(NOT CEED_FOUND AND DEFINED CEED_DIR)
        find_path(CEED_INCLUDE_DIR ceed/ceed.h
            HINTS ${CEED_DIR}/include
            NO_DEFAULT_PATH
        )
        find_library(CEED_LIBRARY ceed
            HINTS ${CEED_DIR}/lib ${CEED_DIR}/lib64
            NO_DEFAULT_PATH
        )
        if(CEED_INCLUDE_DIR AND CEED_LIBRARY)
            set(CEED_FOUND TRUE)
            set(CEED_INCLUDE_DIRS ${CEED_INCLUDE_DIR})
            set(CEED_LIBRARIES ${CEED_LIBRARY})
            message(STATUS "libCEED found at ${CEED_DIR}")
        endif()
    endif()

    if(CEED_FOUND)
        message(STATUS "")
        message(STATUS "========================================")
        message(STATUS "libCEED support: ENABLED")
        message(STATUS "  -> CeedElasticIntegrator3D available")
        message(STATUS "  -> GPU/CPU portable kernels enabled")
        if(CEED_INCLUDE_DIRS)
            message(STATUS "  -> Include: ${CEED_INCLUDE_DIRS}")
        endif()
        if(CEED_LIBRARIES)
            message(STATUS "  -> Library: ${CEED_LIBRARIES}")
        endif()
        message(STATUS "========================================")
        message(STATUS "")
    else()
        message(WARNING "libCEED not found! SEM_USE_LIBCEED will be disabled.")
        message(WARNING "  To enable, install libCEED and set CEED_DIR or PKG_CONFIG_PATH")
        set(SEM_USE_LIBCEED OFF)
    endif()
else()
    message(STATUS "libCEED support: DISABLED")
    message(STATUS "  -> Use -DSEM_USE_LIBCEED=ON and -DCEED_DIR=/path/to/libceed to enable")
endif()

# =============================================================================
# Modular SEM Libraries
# =============================================================================
# Architecture: Layered dependency structure
#
#   sem_config ──────┬──────────────────────────────────────┐
#       │            │                                      │
#       ▼            ▼                                      ▼
#   sem_material  sem_srcrecv                          sem_util
#       │            │                                      │
#       └────────────┼──────────────────────────────────────┘
#                    ▼
#               sem_integ
#                    │
#                    ▼
#              sem_operator
#                    │
#                    ▼
#             sem_simulation
#
# =============================================================================

# Common include directories for all libraries
set(SEM_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${MFEM_INCLUDE_DIRS}
    ${MPI_INCLUDE_PATH}
    ${HDF5_INCLUDE_DIRS}
)

# Common link libraries for all libraries.
# Append CUDA::cudart on CUDA builds so every sem_* target picks up the
# CUDA runtime that libmfem.a needs (mem_manager.o references
# cudaStreamSynchronize, cudaMalloc, ...).
set(SEM_LINK_LIBS ${MFEM_LIBRARIES} ${MPI_LIBRARIES})
if(SEM_USE_CUDA)
    list(APPEND SEM_LINK_LIBS CUDA::cudart)
endif()

# Static MFEM (libmfem.a) built with MPI leaves METIS_PartGraph* symbols
# unresolved in mesh.cpp.o; MFEMConfig.cmake does not propagate the METIS
# link, so consumers must add it themselves. Shared MFEM hides this
# because libmfem.so encodes libmetis in its DT_NEEDED entries.
#
# Not gated on MFEM_USE_METIS — MFEMConfig.cmake does not export that
# variable (only MFEM_USE_METIS_5, which means "METIS is version 5+",
# not "METIS is in use"). MFEM_USE_MPI=ON in MFEM's CMake always pulls
# METIS in, and SEMSWS itself REQUIREs MPI a few lines above, so the
# unconditional find_library here is correct.
find_library(SEM_METIS_LIB
    NAMES metis
    HINTS ${MFEM_DIR}/lib
          ${MFEM_DIR}/../metis/lib
          /usr/lib/x86_64-linux-gnu)
if(SEM_METIS_LIB)
    message(STATUS "SEMSWS: appending METIS to SEM_LINK_LIBS: ${SEM_METIS_LIB}")
    list(APPEND SEM_LINK_LIBS ${SEM_METIS_LIB})
else()
    message(WARNING
        "libmetis was not found via find_library. Static MFEM builds will\n"
        "  fail to link with undefined references to METIS_PartGraphKway /\n"
        "  METIS_PartGraphVKway. Pass -DSEM_METIS_LIB=<full path to\n"
        "  libmetis.so|.a> or add the install prefix to CMAKE_PREFIX_PATH.")
endif()

# Helper function to create a SEM library
function(add_sem_library name)
    cmake_parse_arguments(ARG "" "" "SOURCES;DEPS;EXTRA_LIBS" ${ARGN})

    # CUDA compilation if enabled (HIP uses hipcc as CXX compiler, no language change needed)
    if(SEM_USE_CUDA)
        set_source_files_properties(${ARG_SOURCES} PROPERTIES LANGUAGE CUDA)
    endif()

    add_library(${name} STATIC ${ARG_SOURCES})
    target_include_directories(${name} PUBLIC ${SEM_INCLUDE_DIRS})
    target_link_libraries(${name} PUBLIC ${SEM_LINK_LIBS} ${ARG_DEPS} ${ARG_EXTRA_LIBS})
    target_compile_features(${name} PUBLIC cxx_std_17)
    # cxx_std_17 above only propagates to CXX compilation; nvcc still
    # needs the CUDA-language standard to be pinned explicitly.
    # Required when MFEM's dfem/ headers pull in CTAD + std::string_view /
    # std::variant on NVHPC nvc++, which otherwise compiles at C++14.
    # Use CUDA_STANDARD property rather than `cuda_std_17` compile feature:
    # CMake's CUDA feature table is empty for several nvcc/Clang-CUDA
    # versions, and a PUBLIC cuda_std_17 requirement then propagates to
    # pure-CXX consumers (e.g. test/unit executables) and aborts with
    # "No known features for CUDA compiler".
    if(SEM_USE_CUDA)
        set_target_properties(${name} PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
            CUDA_EXTENSIONS OFF)
    endif()

    if(SEM_USE_CUDA)
        target_compile_definitions(${name} PUBLIC SEM_USE_CUDA)
    elseif(SEM_USE_HIP)
        target_compile_definitions(${name} PUBLIC SEM_USE_HIP)
    endif()

    if(SEM_USE_GPU_AWARE_MPI)
        target_compile_definitions(${name} PUBLIC SEM_USE_GPU_AWARE_MPI)
    endif()
endfunction()

# =============================================================================
# Layer 0: Revolve (Optimal checkpointing scheduler, no MFEM dependency)
# =============================================================================
set(SEM_REVOLVE_SOURCES
    revolve/Revolve.cpp
)
add_sem_library(sem_revolve
    SOURCES ${SEM_REVOLVE_SOURCES}
)

# =============================================================================
# Layer 1: Configuration (YAML parsing only)
# =============================================================================
set(SEM_CONFIG_SOURCES
    config/YamlConfig.cpp
    config/ConfigLoaders.cpp
    config/HDF5SourceReceiverReader.cpp
)
add_sem_library(sem_config
    SOURCES ${SEM_CONFIG_SOURCES}
    EXTRA_LIBS yaml-cpp::yaml-cpp ${HDF5_LIBRARIES}
)

# =============================================================================
# Layer 2: Material System
# =============================================================================
set(SEM_MATERIAL_SOURCES
    material/MaterialField.cpp
    material/InterpolatingCoefficient.cpp
    material/MaterialFactory.cpp
    # Isotropic Elastic module
    material/isotropic_elastic/IsotropicElasticMaterial.cpp
    material/isotropic_elastic/IsotropicElasticIO.cpp
    material/isotropic_elastic/IsotropicElasticLoader.cpp
    material/isotropic_elastic/IsotropicElasticBuilder.cpp
    # Isotropic Acoustic module
    material/isotropic_acoustic/IsotropicAcousticMaterial.cpp
    material/isotropic_acoustic/IsotropicAcousticIO.cpp
    material/isotropic_acoustic/IsotropicAcousticLoader.cpp
    material/isotropic_acoustic/IsotropicAcousticBuilder.cpp
    # Attenuation coefficients (used by ApplyAttenuationCorrection in material)
    integ/attenuation/AttenuationCoeffs.cpp
    io/ADIOS2IO.cpp
)
add_sem_library(sem_material
    SOURCES ${SEM_MATERIAL_SOURCES}
    DEPS sem_config
    EXTRA_LIBS adios2::cxx11_mpi
)

# =============================================================================
# Layer 2: Source/Receiver System
# =============================================================================
set(SEM_SRCRECV_SOURCES
    srcrecv/source/SourceTimeFunction.cpp
    srcrecv/source/SourceLocation.cpp
    srcrecv/source/SourceCollection.cpp
    srcrecv/source/SourceFactory.cpp
    srcrecv/receiver/ReceiverData.cpp
    srcrecv/receiver/ReceiverLocation.cpp
    srcrecv/receiver/ReceiverRecord.cpp
    srcrecv/receiver/ReceiverIO.cpp
    srcrecv/receiver/ReceiverFactory.cpp
    srcrecv/inversion/ObservedData.cpp
    srcrecv/inversion/HDF5ObservedReader.cpp
    srcrecv/inversion/HDF5SourceReceiverWriter.cpp
    srcrecv/inversion/ObservedResampler.cpp
    srcrecv/inversion/AdjointSource.cpp
    srcrecv/SrcRecvLoaders.cpp
)
add_sem_library(sem_srcrecv
    SOURCES ${SEM_SRCRECV_SOURCES}
    DEPS sem_config sem_material
    EXTRA_LIBS ${HDF5_LIBRARIES}
)

# =============================================================================
# Layer 2: Utilities (PointFinder etc.)
# =============================================================================
set(SEM_UTIL_SOURCES
    util/PointFinder.cpp
    util/MemoryReport.cpp
)
add_sem_library(sem_util
    SOURCES ${SEM_UTIL_SOURCES}
)

# =============================================================================
# Layer 3: Integrators (Geometry, DOF ordering, kernels)
# =============================================================================
set(SEM_INTEG_SOURCES
    # Core framework
    integ/core/SEMGeometry.cpp
    integ/core/SEMDofOrdering.cpp
    # Stiffness integrator factory (free functions, one per physics/dim)
    integ/StiffnessIntegratorFactory.cpp
    # Mass integrators
    integ/mass/SEMMassIntegrator2D.cpp
    integ/mass/SEMMassIntegrator3D.cpp
    # Attenuation models
    integ/attenuation/AttenuationModel.cpp
    integ/attenuation/GeneralizedZener.cpp
    # Stiffness kernels (template specializations - heavy compilation)
    integ/kernels/IsotropicElasticKernels2D.cpp
    integ/kernels/IsotropicElasticKernels3D.cpp
    integ/kernels/IsotropicAcousticKernels2D.cpp
    integ/kernels/IsotropicAcousticKernels3D.cpp
    integ/kernels/Visco_IsotropicElasticKernels2D.cpp
    integ/kernels/Visco_IsotropicElasticKernels3D.cpp
    integ/kernels/Visco_IsotropicAcousticKernels2D.cpp
    integ/kernels/Visco_IsotropicAcousticKernels3D.cpp
)
add_sem_library(sem_integ
    SOURCES ${SEM_INTEG_SOURCES}
    DEPS sem_material
)

# Apply -maxrregcount to 3D elastic kernel for register optimization
# This limits registers to 40, potentially improving occupancy at the cost of spills
# if(SEM_USE_CUDA)
#     set_source_files_properties(
#         integ/kernels/Visco_IsotropicElasticKernels3D.cpp
#         PROPERTIES COMPILE_FLAGS "--maxrregcount=40"
#     )
# endif()

# =============================================================================
# Layer 3.5: FWI (Sensitivity kernels)
# =============================================================================
set(SEM_FWI_SOURCES
    fwi/IsotropicAcousticSensitivity2D.cpp
    fwi/IsotropicAcousticSensitivityAD2D.cpp
    fwi/IsotropicAcousticSensitivity3D.cpp
    fwi/IsotropicAcousticSensitivityAD3D.cpp
    fwi/IsotropicElasticSensitivity2D.cpp
    fwi/IsotropicElasticSensitivityAD2D.cpp
    fwi/IsotropicElasticSensitivity3D.cpp
    fwi/IsotropicElasticSensitivityAD3D.cpp
    fwi/CheckpointStorage.cpp
    fwi/HelmholtzSmoothing.cpp
    fwi/SensitivityKernelFactory.cpp
)
add_sem_library(sem_fwi
    SOURCES ${SEM_FWI_SOURCES}
    DEPS sem_integ sem_material
)

# =============================================================================
# Layer 4: Operators (Wave equation discretization)
# =============================================================================
set(SEM_OPERATOR_SOURCES
    operator/WaveOperator.cpp
    operator/WaveOperatorBase2D.cpp
    operator/WaveOperatorBase3D.cpp
    operator/damping/CerjanABC.cpp
    operator/elastic/ElasticityOperator2D.cpp
    operator/elastic/ElasticityOperator3D.cpp
    operator/acoustic/AcousticOperator2D.cpp
    operator/acoustic/AcousticOperator3D.cpp
    operator/OperatorFactory.cpp
)
add_sem_library(sem_operator
    SOURCES ${SEM_OPERATOR_SOURCES}
    DEPS sem_integ sem_material
)

# =============================================================================
# Layer 5: IO (Wavefield output strategies)
# =============================================================================
set(SEM_IO_SOURCES
    io/WavefieldWriter.cpp
    io/ParaViewWavefieldWriter.cpp
    io/GMTWavefieldWriter.cpp
    io/WavefieldWriterFactory.cpp
    io/MaterialWriter.cpp
    io/FieldVizWriter.cpp
)
add_sem_library(sem_io
    SOURCES ${SEM_IO_SOURCES}
    DEPS sem_material sem_util
)

# =============================================================================
# Layer 6: Simulation (High-level interface, time integration)
# =============================================================================
set(SEM_SIMULATION_SOURCES
    # New decomposed components
    simulation/SimulationComponents.cpp
    simulation/SimulationRunner.cpp
    simulation/SimulationIONew.cpp
    simulation/SimulationFacade.cpp
    simulation/SimulationFacadeSetup.cpp
    # ForwardSimulation extends SimulationFacade
    simulation/ForwardSimulation.cpp
    # AdjointSimulation for FWI
    simulation/AdjointSimulation.cpp
    # Time integrator
    timeinteg/TimeIntegrator.cpp
)
add_sem_library(sem_simulation
    SOURCES ${SEM_SIMULATION_SOURCES}
    DEPS sem_operator sem_srcrecv sem_util sem_io sem_fwi sem_revolve
    EXTRA_LIBS ${HDF5_LIBRARIES}
)

# =============================================================================
# Layer 7: Coupling (fluid-solid)
# =============================================================================
set(SEM_COUPLING_SOURCES
    coupling/CoupledSimulationFacade.cpp
    coupling/FluidSolidInterface.cpp
)
add_sem_library(sem_coupling
    SOURCES ${SEM_COUPLING_SOURCES}
    DEPS sem_simulation sem_operator sem_material sem_config
)

# =============================================================================
# Convenience target: links all SEM libraries
# =============================================================================
add_library(sem INTERFACE)
target_link_libraries(sem INTERFACE
    sem_simulation
    sem_coupling
    sem_operator
    sem_integ
    sem_fwi
    sem_srcrecv
    sem_material
    sem_config
    sem_util
    sem_io
    sem_revolve
)

# =============================================================================
# Main executables
# =============================================================================

# ForwardSimulation test with custom progress callback
add_executable(semsws semsws.cpp)
if(SEM_USE_CUDA)
    set_source_files_properties(semsws.cpp PROPERTIES LANGUAGE CUDA)
endif()
if(SEM_USE_HIP)
    target_compile_definitions(semsws PRIVATE SEM_USE_HIP)
endif()
if(SEM_USE_GPU_AWARE_MPI)
    target_compile_definitions(semsws PRIVATE SEM_USE_GPU_AWARE_MPI)
endif()
target_link_libraries(semsws PRIVATE sem)

# =============================================================================
# Mesh Partitioning Tool
# =============================================================================
# Serial tool to pre-partition meshes for memory-efficient loading
# Does not require MPI at runtime, but links against MFEM
add_executable(partition_mesh tools/partition_mesh.cpp)
target_link_libraries(partition_mesh PRIVATE sem)

# Mesh refinement tool (MPI-parallel, config-based, all material formats)
add_executable(refine_mesh tools/refine_mesh.cpp)
target_link_libraries(refine_mesh PRIVATE sem)

# Mesh quality evaluation tool (MPI-parallel, config-based)
add_executable(evaluate_mesh tools/evaluate_mesh.cpp)
target_link_libraries(evaluate_mesh PRIVATE sem)

# Standalone material->BP export tool (MPI-parallel).
# Materializes a YAML-described material on its mesh's GLL points and writes
# vp/vs/rho [/qkappa/qmu] .bp files into a target directory. Used by the
# Python driver as a one-shot pre-pass so multi-shot runs can read the same
# BPs concurrently via material.format=adios2.
add_executable(semsws_export_model tools/export_model.cpp)
target_link_libraries(semsws_export_model PRIVATE sem)

# =============================================================================
# Kernel / mesh helper tools
# =============================================================================

# Collect tool sources that need CUDA compilation.
set(SEM_TOOL_SOURCES
    tools/smooth_kernel.cpp
    tools/combine_mesh.cpp
    tools/refine_mesh.cpp
    tools/evaluate_mesh.cpp
    tools/sem_viz.cpp
    tools/sum_kernels.cpp
    tools/mask.cpp
    tools/precondition_gradient.cpp
    tools/clip_field.cpp
    tools/scale_field.cpp
    tools/lbfgs_direction.cpp
    tools/update_model.cpp
    tools/max_abs.cpp
)
if(SEM_USE_CUDA)
    set_source_files_properties(${SEM_TOOL_SOURCES} PROPERTIES LANGUAGE CUDA)
endif()
# HIP: no language change needed, hipcc handles .cpp files directly

# Helmholtz smoothing tool for kernels
add_executable(smooth_kernel tools/smooth_kernel.cpp)
target_link_libraries(smooth_kernel PRIVATE sem)

# MPI tool to combine partitioned mesh files for GLVis visualization
# Requires MPI at runtime (mpirun -np N)
add_executable(combine_mesh tools/combine_mesh.cpp)
target_link_libraries(combine_mesh PRIVATE sem)

# sem_viz — offline .bp → paraview/glvis/gmt converter
add_executable(sem_viz tools/sem_viz.cpp)
target_link_libraries(sem_viz PRIVATE sem)

# =============================================================================
# FWI gradient / optimizer post-processing tools
# =============================================================================
# Per-shot kernel reduction
add_executable(sum_kernels tools/sum_kernels.cpp)
target_link_libraries(sum_kernels PRIVATE sem)

# Boundary / attribute masking of MaterialFields
add_executable(mask tools/mask.cpp)
target_link_libraries(mask PRIVATE sem)

# Pseudo-Hessian preconditioning
add_executable(precondition_gradient tools/precondition_gradient.cpp)
target_link_libraries(precondition_gradient PRIVATE sem)

# Outlier clipping (percentile / absolute / of_max)
add_executable(clip_field tools/clip_field.cpp)
target_link_libraries(clip_field PRIVATE sem)

# Scalar scale of a MaterialField
add_executable(scale_field tools/scale_field.cpp)
target_link_libraries(scale_field PRIVATE sem)

# L-BFGS direction (with secant-pair history on disk)
add_executable(lbfgs_direction tools/lbfgs_direction.cpp)
target_link_libraries(lbfgs_direction PRIVATE sem)

# Steepest descent / accepted-step model update with bounds
add_executable(update_model tools/update_model.cpp)
target_link_libraries(update_model PRIVATE sem)

# Max |field| reduction (writes scalar to stdout / file)
add_executable(max_abs tools/max_abs.cpp)
target_link_libraries(max_abs PRIVATE sem)

# Fluid-solid coupling probe executables — dev-only tools consumed by
# the setup-phase pytest modules (test_facade_setup, test_interface_dof,
# test_gmsh_ingestion, etc.). End users running the quantitative
# coupling regression suite don't need them, so gate behind
# -DSEM_BUILD_COUPLING_PROBES=ON to avoid adding ~3 extra binaries to
# the default build.
if(SEM_BUILD_COUPLING_PROBES)
    # GMSH attribute probe: verify GMSH Physical Volume attribute
    # inheritance and ParSubMesh::CreateFromDomain.
    add_executable(coupling_probe_gmsh tools/coupling_probe_gmsh.cpp)
    target_link_libraries(coupling_probe_gmsh PRIVATE sem)

    # Facade setup probe:
    add_executable(coupling_probe_facade tools/coupling_probe_facade.cpp)
    target_link_libraries(coupling_probe_facade PRIVATE sem)

    # Interface probe: exercises FluidSolidInterface::Setup and the
    # Extract* gather paths against an analytical field.
    add_executable(coupling_probe_interface tools/coupling_probe_interface.cpp)
    target_link_libraries(coupling_probe_interface PRIVATE sem)
endif()

# =============================================================================
# Install rules (Spack / cmake --install)
#
# Phase 1 of the packaging plan: install user-facing binaries and the public
# header tree only. The static layer libraries (sem_*) ship under lib/ as a
# convenience but are not exported as imported targets — downstream linking
# against semsws as a C++ library is out of scope until Phase 4. Until then,
# `find_package(SEMSWS)` is intentionally not provided.
# =============================================================================

set(SEMSWS_LIB_TARGETS
    sem_revolve sem_config sem_material sem_srcrecv sem_util sem_integ
    sem_fwi sem_operator sem_io sem_simulation sem_coupling
)

set(SEMSWS_BIN_TARGETS
    semsws
    partition_mesh refine_mesh evaluate_mesh semsws_export_model
    smooth_kernel combine_mesh sem_viz
    sum_kernels mask precondition_gradient clip_field scale_field
    lbfgs_direction update_model max_abs
)

if(SEM_BUILD_COUPLING_PROBES)
    list(APPEND SEMSWS_BIN_TARGETS
        coupling_probe_gmsh coupling_probe_facade coupling_probe_interface)
endif()

install(TARGETS ${SEMSWS_LIB_TARGETS}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(TARGETS ${SEMSWS_BIN_TARGETS}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/semsws
    FILES_MATCHING PATTERN "*.hpp"
)
