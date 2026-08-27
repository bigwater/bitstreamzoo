cmake_minimum_required(VERSION 3.24)  # CMAKE_CUDA_ARCHITECTURES=native

# Auto-detect CUDA
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    project(bsim_suite LANGUAGES CXX CUDA)
    set(HAVE_CUDA TRUE)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    find_package(CUDAToolkit QUIET)
else()
    project(bsim_suite LANGUAGES CXX)
    set(HAVE_CUDA FALSE)
    message(STATUS "CUDA not found - building CPU targets only")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to Release build for optimized performance
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG" CACHE STRING "" FORCE)

# ── nlohmann/json via FetchContent (single copy) ─────────────────────
include(FetchContent)
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)

# ── OpenMP ────────────────────────────────────────────────────────────
find_package(OpenMP REQUIRED)

# ── Shared OBJECT library (compiled once) ─────────────────────────────
add_library(bsim_common OBJECT
    src/lexer.cpp
    src/parser.cpp
    src/ast_liveness.cpp
)
target_include_directories(bsim_common PUBLIC src)
target_link_libraries(bsim_common PUBLIC nlohmann_json::nlohmann_json)

# ── CPU targets ───────────────────────────────────────────────────────
add_library(bitstream_ops STATIC src/bitstream_ops.cpp)
target_compile_options(bitstream_ops PRIVATE -mavx512f -mavx512bw -mavx512vl -mavx512vpopcntdq)
target_link_libraries(bitstream_ops PRIVATE OpenMP::OpenMP_CXX)

add_executable(bsim
    src/main.cpp
)
target_include_directories(bsim PRIVATE src)
target_link_libraries(bsim PRIVATE bsim_common
    nlohmann_json::nlohmann_json bitstream_ops OpenMP::OpenMP_CXX)

add_executable(test_ops src/test_ops.cpp)
target_link_libraries(test_ops PRIVATE bitstream_ops OpenMP::OpenMP_CXX)

# ── GPU targets (conditional) ────────────────────────────────────────
if(HAVE_CUDA)
    add_executable(bsim_cuda
        src/main.cu
    )
    target_include_directories(bsim_cuda PRIVATE src)
    target_link_libraries(bsim_cuda PRIVATE bsim_common
        nlohmann_json::nlohmann_json)
    set_target_properties(bsim_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    add_executable(test_kernels src/test_kernels.cu)
    target_include_directories(test_kernels PRIVATE src)
    set_target_properties(test_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
