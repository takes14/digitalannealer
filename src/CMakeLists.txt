cmake_minimum_required(VERSION 3.18)
project(digitalannealer LANGUAGES CXX CUDA)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find packages
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# Build/performance tuning knobs (can be overridden with -D)
if(NOT DEFINED DA_ENABLE_IPO)
    option(DA_ENABLE_IPO "Enable C++ IPO/LTO for release builds when supported" ON)
endif()
if(NOT DEFINED DA_AUTO_DETECT_CUDA_ARCH)
    option(
        DA_AUTO_DETECT_CUDA_ARCH
        "Detect CUDA compute capability via nvidia-smi when CMAKE_CUDA_ARCHITECTURES is unset"
        ON
    )
endif()
set(
    DA_DEFAULT_CUDA_ARCHITECTURES "75;80;86;89;90"
    CACHE STRING
    "Fallback CUDA architectures when auto-detection is unavailable"
)

# Resolve pybind11 installed via Python package managers (e.g. uv/pip)
if(NOT pybind11_DIR)
    execute_process(
        COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
        OUTPUT_VARIABLE _pybind11_cmakedir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _pybind11_cmakedir_result
    )
    if(_pybind11_cmakedir_result EQUAL 0 AND EXISTS "${_pybind11_cmakedir}")
        list(PREPEND CMAKE_PREFIX_PATH "${_pybind11_cmakedir}")
    endif()
endif()

find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Resolve CUDA architectures.
# We prefer runtime detection (nvidia-smi) because CMake's "native" can resolve to old
# defaults depending on toolchain configuration.
function(da_detect_cuda_architectures out_var)
    if(NOT DA_AUTO_DETECT_CUDA_ARCH)
        set(${out_var} "" PARENT_SCOPE)
        return()
    endif()

    find_program(_da_nvidia_smi nvidia-smi)
    if(NOT _da_nvidia_smi)
        set(${out_var} "" PARENT_SCOPE)
        return()
    endif()

    execute_process(
        COMMAND "${_da_nvidia_smi}" --query-gpu=compute_cap --format=csv,noheader
        OUTPUT_VARIABLE _da_compute_caps
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _da_compute_caps_result
        ERROR_QUIET
        TIMEOUT 10
    )
    if(NOT _da_compute_caps_result EQUAL 0 OR _da_compute_caps STREQUAL "")
        set(${out_var} "" PARENT_SCOPE)
        return()
    endif()

    string(REPLACE "\n" ";" _da_compute_caps_list "${_da_compute_caps}")
    set(_da_detected_cuda_arches "")
    foreach(_da_cap IN LISTS _da_compute_caps_list)
        string(STRIP "${_da_cap}" _da_cap)
        if(_da_cap MATCHES "^([0-9]+)\\.([0-9]+)$")
            list(APPEND _da_detected_cuda_arches "${CMAKE_MATCH_1}${CMAKE_MATCH_2}")
        endif()
    endforeach()

    list(REMOVE_DUPLICATES _da_detected_cuda_arches)
    set(${out_var} "${_da_detected_cuda_arches}" PARENT_SCOPE)
endfunction()

set(_da_should_set_cuda_arches OFF)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(_da_should_set_cuda_arches ON)
elseif(CMAKE_CUDA_ARCHITECTURES STREQUAL "52")
    # CMake/NVCC defaults can resolve to sm_52 on modern toolchains; override that default.
    set(_da_should_set_cuda_arches ON)
endif()

if(_da_should_set_cuda_arches)
    da_detect_cuda_architectures(_da_detected_cuda_arches)
    if(_da_detected_cuda_arches)
        set(CMAKE_CUDA_ARCHITECTURES "${_da_detected_cuda_arches}" CACHE STRING "" FORCE)
        message(STATUS "Auto-detected CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
    elseif(DA_DEFAULT_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "${DA_DEFAULT_CUDA_ARCHITECTURES}" CACHE STRING "" FORCE)
        message(STATUS "Using fallback CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
    elseif(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
        set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "" FORCE)
        message(STATUS "Using CMAKE_CUDA_ARCHITECTURES=native")
    else()
        set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90 CACHE STRING "" FORCE)
        message(STATUS "Using legacy CMAKE_CUDA_ARCHITECTURES=75;80;86;89;90")
    endif()
endif()

# Enable IPO for the pybind11 module only when both C++ and CUDA toolchains support it.
# _core links against da_kernels (CUDA device objects); if the CUDA linker path does not
# support LTO flags the final link step will fail even when C++ IPO alone is fine.
set(_da_enable_ipo_release OFF)
if(DA_ENABLE_IPO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT _da_ipo_cxx_ok OUTPUT _da_ipo_cxx_err LANGUAGES CXX)
    check_ipo_supported(RESULT _da_ipo_cuda_ok OUTPUT _da_ipo_cuda_err LANGUAGES CUDA)
    if(_da_ipo_cxx_ok AND _da_ipo_cuda_ok)
        set(_da_enable_ipo_release ON)
    elseif(NOT _da_ipo_cxx_ok)
        message(STATUS "C++ IPO disabled (${CMAKE_CXX_COMPILER_ID}): ${_da_ipo_cxx_err}")
    else()
        message(STATUS "CUDA IPO disabled: ${_da_ipo_cuda_err}")
    endif()
endif()

# Kernel library
add_library(da_kernels STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/da_update_dense.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/da_update_sparse.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/pt_exchange.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/engine/da_engine.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/engine/pt_engine.cu
)
target_include_directories(da_kernels PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(da_kernels PUBLIC
    CUDA::cudart
    CUDA::curand
)
set_target_properties(da_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# ---- CUDA performance flags (Release builds only) ----
# --use_fast_math: single-instruction __logf/__expf in hot kernel loops
# --expt-relaxed-constexpr: allow constexpr in device code
target_compile_options(da_kernels PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# Python module
pybind11_add_module(_core
    ${CMAKE_CURRENT_SOURCE_DIR}/bindings/main.cpp
)
target_include_directories(_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(_core PRIVATE
    da_kernels
    CUDA::cudart
    CUDA::curand
)
if(_da_enable_ipo_release)
    set_property(TARGET _core PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

# Install the Python module
install(TARGETS _core DESTINATION digitalannealer)
