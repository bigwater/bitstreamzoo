#pragma once

#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

// ── Error checking ─────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            throw std::runtime_error(                                           \
                std::string("CUDA error at ") + __FILE__ + ":" +                \
                std::to_string(__LINE__) + ": " + cudaGetErrorString(err));      \
        }                                                                       \
    } while (0)

// ── Word-size traits ────────────────────────────────────────────────
template<int BITS> struct WordTraits;

template<> struct WordTraits<64> {
    using word_t = uint64_t;
    using carry_t = __uint128_t;
    static constexpr int WORD_BITS = 64;
    static constexpr int CARRY_SHIFT = 64;
    static __device__ int popcount(uint64_t x) { return __popcll(x); }
    static constexpr const char* typedef_str = "typedef unsigned long long word_t;\n";
    static constexpr const char* zero_lit = "0ULL";
    static constexpr const char* ones_lit = "~0ULL";
    static constexpr const char* one_lit = "1ULL";
};

template<> struct WordTraits<32> {
    using word_t = uint32_t;
    using carry_t = uint64_t;
    static constexpr int WORD_BITS = 32;
    static constexpr int CARRY_SHIFT = 32;
    static __device__ int popcount(uint32_t x) { return __popc(x); }
    static constexpr const char* typedef_str = "typedef unsigned int word_t;\n";
    static constexpr const char* zero_lit = "0U";
    static constexpr const char* ones_lit = "~0U";
    static constexpr const char* one_lit = "1U";
};

// ── Types ──────────────────────────────────────────────────────────
static constexpr int BLOCK_SIZE = 256;

// Number of words needed for bitlength bit positions
template<int BITS>
inline int64_t num_words(int64_t bitlength) {
    return (bitlength + WordTraits<BITS>::WORD_BITS - 1) / WordTraits<BITS>::WORD_BITS;
}

// Grid size for N words
inline int grid_size(int64_t N) {
    return static_cast<int>((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
}

// ── GPU stream: a bitstream stored as an array of words ──────────
template<int BITS>
struct GpuStream {
    using word_t = typename WordTraits<BITS>::word_t;

    word_t* data = nullptr;
    int64_t n_words = 0;
    bool owned = true;   // false for slices of bulk allocations

    // Compatibility with interpreter.inl interface
    word_t* data_ptr() { return data; }
    const word_t* data_ptr() const { return data; }
    explicit operator bool() const { return data != nullptr; }

    void alloc(int nw) {
        n_words = nw;
        CUDA_CHECK(cudaMalloc(&data, sizeof(word_t) * nw));
        CUDA_CHECK(cudaMemset(data, 0, sizeof(word_t) * nw));
        owned = true;
    }

    void free() {
        if (data && owned) {
            CUDA_CHECK(cudaFree(data));
        }
        data = nullptr;
    }

    void zero() {
        CUDA_CHECK(cudaMemset(data, 0, sizeof(word_t) * n_words));
    }

    void ones() {
        CUDA_CHECK(cudaMemset(data, 0xFF, sizeof(word_t) * n_words));
    }

    void copy_from_host(const word_t* host, int nw) {
        CUDA_CHECK(cudaMemcpy(data, host, sizeof(word_t) * nw,
                              cudaMemcpyHostToDevice));
    }

    void copy_to_host(word_t* host, int nw) const {
        CUDA_CHECK(cudaMemcpy(host, data, sizeof(word_t) * nw,
                              cudaMemcpyDeviceToHost));
    }

    void copy_from(const GpuStream& src) {
        CUDA_CHECK(cudaMemcpy(data, src.data, sizeof(word_t) * n_words,
                              cudaMemcpyDeviceToDevice));
    }
};
