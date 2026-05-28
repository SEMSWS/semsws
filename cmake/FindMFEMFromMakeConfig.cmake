if(NOT MFEM_DIR)
    message(FATAL_ERROR "FindMFEMFromMakeConfig: MFEM_DIR is not set")
endif()

set(_mfem_config_mk "${MFEM_DIR}/share/mfem/config.mk")
if(NOT EXISTS "${_mfem_config_mk}")
    if(EXISTS "${MFEM_DIR}/config.mk")
        set(_mfem_config_mk "${MFEM_DIR}/config.mk")
    else()
        message(FATAL_ERROR
            "FindMFEMFromMakeConfig: no config.mk under ${MFEM_DIR}")
    endif()
endif()

file(READ "${_mfem_config_mk}" _mfem_config_mk_text)

function(_mfem_mkconf_get name out_var)
    if(_mfem_config_mk_text MATCHES "(^|\n)[ \t]*${name}[ \t]*=[ \t]*([^\n]*)")
        string(STRIP "${CMAKE_MATCH_2}" _v)
        set(${out_var} "${_v}" PARENT_SCOPE)
    else()
        set(${out_var} "" PARENT_SCOPE)
    endif()
endfunction()

function(_mfem_mkconf_expand input out_var)
    set(_s "${input}")
    set(_round 0)
    while(_s MATCHES "\\$\\(([A-Za-z_][A-Za-z0-9_]*)\\)" AND _round LESS 16)
        set(_v "${CMAKE_MATCH_1}")
        set(_replacement "${${_v}}")
        string(REPLACE "$(${_v})" "${_replacement}" _s "${_s}")
        math(EXPR _round "${_round} + 1")
    endwhile()
    set(${out_var} "${_s}" PARENT_SCOPE)
endfunction()

_mfem_mkconf_get(MFEM_VERSION       MFEM_VERSION_INT)
_mfem_mkconf_get(MFEM_INC_DIR       MFEM_INC_DIR)
_mfem_mkconf_get(MFEM_LIB_DIR       MFEM_LIB_DIR)
_mfem_mkconf_get(MFEM_EXT_LIBS      MFEM_EXT_LIBS)
_mfem_mkconf_get(MFEM_LIBS          MFEM_LIBS_RAW)
_mfem_mkconf_get(MFEM_TPLFLAGS      MFEM_TPLFLAGS_RAW)
_mfem_mkconf_get(MFEM_USE_MPI       MFEM_USE_MPI_STR)
_mfem_mkconf_get(MFEM_USE_CUDA      MFEM_USE_CUDA_STR)
_mfem_mkconf_get(MFEM_USE_HIP       MFEM_USE_HIP_STR)
_mfem_mkconf_get(MFEM_USE_SINGLE    MFEM_USE_SINGLE_STR)
_mfem_mkconf_get(MFEM_USE_DOUBLE    MFEM_USE_DOUBLE_STR)

_mfem_mkconf_expand("${MFEM_LIBS_RAW}"     MFEM_LIBS_EXPANDED)
_mfem_mkconf_expand("${MFEM_TPLFLAGS_RAW}" MFEM_TPLFLAGS_EXPANDED)
_mfem_mkconf_expand("${MFEM_EXT_LIBS}"     MFEM_EXT_LIBS_EXPANDED)

foreach(_flag MPI CUDA HIP SINGLE DOUBLE)
    set(_v "${MFEM_USE_${_flag}_STR}")
    if(_v STREQUAL "YES")
        set(MFEM_USE_${_flag} TRUE)
    else()
        set(MFEM_USE_${_flag} FALSE)
    endif()
endforeach()
if(NOT MFEM_USE_SINGLE AND NOT MFEM_USE_DOUBLE)
    set(MFEM_USE_DOUBLE TRUE)
endif()

if(MFEM_VERSION_INT MATCHES "^([0-9]+)0?([0-9])0?([0-9])$")
    set(MFEM_VERSION "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}")
else()
    set(MFEM_VERSION "${MFEM_VERSION_INT}")
endif()

if(NOT MFEM_INC_DIR)
    set(MFEM_INC_DIR "${MFEM_DIR}/include")
endif()
if(NOT MFEM_LIB_DIR)
    set(MFEM_LIB_DIR "${MFEM_DIR}/lib")
endif()
set(MFEM_INCLUDE_DIRS "${MFEM_INC_DIR}" "${MFEM_INC_DIR}/mfem")

find_library(MFEM_LIBRARY
    NAMES mfem
    HINTS "${MFEM_LIB_DIR}"
    NO_DEFAULT_PATH
)
if(NOT MFEM_LIBRARY)
    message(FATAL_ERROR
        "FindMFEMFromMakeConfig: libmfem not found in ${MFEM_LIB_DIR}")
endif()

if(NOT TARGET mfem)
    add_library(mfem INTERFACE IMPORTED)
endif()
set_target_properties(mfem PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${MFEM_INCLUDE_DIRS}"
)
target_link_libraries(mfem INTERFACE "${MFEM_LIBRARY}")
if(MFEM_EXT_LIBS_EXPANDED)
    # GNU ld is single-pass: EXT_LIBS (-lz, -lhypre, ...) must come AFTER
    # libmfem.a, so push via target_link_libraries (not target_link_options).
    separate_arguments(_mfem_ext_libs UNIX_COMMAND "${MFEM_EXT_LIBS_EXPANDED}")
    target_link_libraries(mfem INTERFACE ${_mfem_ext_libs})
endif()
if(MFEM_TPLFLAGS_EXPANDED)
    string(REGEX MATCHALL "-I[^ \t]+" _tpl_includes "${MFEM_TPLFLAGS_EXPANDED}")
    foreach(_inc ${_tpl_includes})
        string(SUBSTRING "${_inc}" 2 -1 _inc_path)
        target_include_directories(mfem INTERFACE "${_inc_path}")
    endforeach()
endif()

set(MFEM_FOUND TRUE)
set(MFEM_LIBRARIES mfem)

message(STATUS "MFEM resolved via config.mk fallback")
message(STATUS "  version:     ${MFEM_VERSION}")
message(STATUS "  include:     ${MFEM_INC_DIR}")
message(STATUS "  library:     ${MFEM_LIBRARY}")
message(STATUS "  precision:   single=${MFEM_USE_SINGLE} double=${MFEM_USE_DOUBLE}")
message(STATUS "  GPU:         CUDA=${MFEM_USE_CUDA} HIP=${MFEM_USE_HIP}")
