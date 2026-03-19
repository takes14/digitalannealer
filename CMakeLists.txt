cmake_minimum_required(VERSION 3.18)
project(digitalannealer LANGUAGES CXX CUDA)

# Build/performance tuning knobs (can be overridden with -D)
option(DA_ENABLE_IPO "Enable C++ IPO/LTO for release builds when supported" ON)
option(
    DA_AUTO_DETECT_CUDA_ARCH
    "Detect CUDA compute capability via nvidia-smi when CMAKE_CUDA_ARCHITECTURES is unset"
    ON
)
set(
    DA_DEFAULT_CUDA_ARCHITECTURES "75;80;86;89;90"
    CACHE STRING
    "Fallback CUDA architectures when auto-detection is unavailable"
)

# Main build is in src/
add_subdirectory(src)

# Optional: Build C++ tests
option(BUILD_TESTS "Build C++ tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(googletest)

    add_subdirectory(tests/cpp)
endif()
