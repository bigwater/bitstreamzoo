#pragma once

#include "common.cuh"

// ── Trivial per-word kernels ───────────────────────────────────────

template<int BITS>
__global__ void kern_not(typename WordTraits<BITS>::word_t* dst,
                         const typename WordTraits<BITS>::word_t* src, int64_t N) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] = ~src[i];
}

template<int BITS>
__global__ void kern_and(typename WordTraits<BITS>::word_t* dst,
                         const typename WordTraits<BITS>::word_t* a,
                         const typename WordTraits<BITS>::word_t* b, int64_t N) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] = a[i] & b[i];
}

template<int BITS>
__global__ void kern_or(typename WordTraits<BITS>::word_t* dst,
                        const typename WordTraits<BITS>::word_t* a,
                        const typename WordTraits<BITS>::word_t* b, int64_t N) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] = a[i] | b[i];
}

template<int BITS>
__global__ void kern_xor(typename WordTraits<BITS>::word_t* dst,
                         const typename WordTraits<BITS>::word_t* a,
                         const typename WordTraits<BITS>::word_t* b, int64_t N) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] = a[i] ^ b[i];
}

template<int BITS>
__global__ void kern_copy(typename WordTraits<BITS>::word_t* dst,
                          const typename WordTraits<BITS>::word_t* src, int64_t N) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] = src[i];
}

template<int BITS>
__global__ void kern_fill(typename WordTraits<BITS>::word_t* dst,
                          typename WordTraits<BITS>::word_t val, int64_t N) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] = val;
}

// ── Shift kernels ──────────────────────────────────────────────────

template<int BITS>
__global__ void kern_shl(typename WordTraits<BITS>::word_t* dst,
                         const typename WordTraits<BITS>::word_t* src,
                         int64_t k, int64_t N) {
    using word_t = typename WordTraits<BITS>::word_t;
    constexpr int WB = WordTraits<BITS>::WORD_BITS;
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int64_t word_shift = k / WB;
    int bit_shift = static_cast<int>(k % WB);

    int64_t lo_idx = i - word_shift;
    int64_t hi_idx = lo_idx - 1;

    word_t lo = (lo_idx >= 0 && lo_idx < N) ? src[lo_idx] : static_cast<word_t>(0);
    word_t hi = (hi_idx >= 0 && hi_idx < N) ? src[hi_idx] : static_cast<word_t>(0);

    if (bit_shift == 0) {
        dst[i] = lo;
    } else {
        dst[i] = (lo << bit_shift) | (hi >> (WB - bit_shift));
    }
}

template<int BITS>
__global__ void kern_shr(typename WordTraits<BITS>::word_t* dst,
                         const typename WordTraits<BITS>::word_t* src,
                         int64_t k, int64_t N) {
    using word_t = typename WordTraits<BITS>::word_t;
    constexpr int WB = WordTraits<BITS>::WORD_BITS;
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int64_t word_shift = k / WB;
    int bit_shift = static_cast<int>(k % WB);

    int64_t lo_idx = i + word_shift;
    int64_t hi_idx = lo_idx + 1;

    word_t lo = (lo_idx >= 0 && lo_idx < N) ? src[lo_idx] : static_cast<word_t>(0);
    word_t hi = (hi_idx >= 0 && hi_idx < N) ? src[hi_idx] : static_cast<word_t>(0);

    if (bit_shift == 0) {
        dst[i] = lo;
    } else {
        dst[i] = (lo >> bit_shift) | (hi << (WB - bit_shift));
    }
}

// ── Addition (per-word, no carry) ──────────────────────────────────

template<int BITS>
__global__ void kern_add(typename WordTraits<BITS>::word_t* dst,
                         const typename WordTraits<BITS>::word_t* a,
                         const typename WordTraits<BITS>::word_t* b, int64_t N) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] = a[i] + b[i];
}

// ── Popcount ───────────────────────────────────────────────────────

template<int BITS>
__global__ void kern_popcount_per_word(unsigned long long* partial,
                                       const typename WordTraits<BITS>::word_t* src,
                                       int64_t N, int64_t lw,
                                       typename WordTraits<BITS>::word_t last_mask) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        if (i >= lw) { partial[i] = 0; return; }
        typename WordTraits<BITS>::word_t w = src[i];
        if (i == lw - 1) w &= last_mask;  // exclude sub-word padding
        partial[i] = WordTraits<BITS>::popcount(w);
    }
}

