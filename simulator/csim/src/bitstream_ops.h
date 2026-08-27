#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>

namespace bs {

// Allocate a zeroed stream buffer, cache-line aligned at both ends.
//
// On AMD Zen4, `rep stos` silently skips the last partial cache line of a
// large buffer under cache pressure, and glibc's memset dispatches to
// `rep stosb` above a size threshold. A buffer is only out of reach of the
// erratum when it ends on a cache-line boundary, which needs the size AND the
// base address aligned. Rounding the size alone left every buffer starting at
// malloc's 16-byte alignment, so each one still ended 16 bytes into a line.
//
// This is defense in depth, not a fix. It only covers the buffers we own:
// std::vector, std::string and the JSON writer allocate large buffers on the
// same run and their memset calls are not ours to change. The mitigation that
// actually holds is the glibc tunable set in benchmark/base.py (and exported
// manually for direct runs), which keeps memset off the `rep stosb` path
// entirely.
inline uint64_t* alloc_stream(size_t n_words) {
    size_t aligned = (n_words + 7) & ~size_t(7);  // round up to 8 words
    size_t bytes = aligned * 8;
    if (bytes == 0) bytes = 64;
    void* p = std::aligned_alloc(64, bytes);
    std::memset(p, 0, bytes);
    return static_cast<uint64_t*>(p);
}
inline void free_stream(uint64_t* p) { std::free(p); }

// Function pointer types for generic Kogge-Stone
using BinOpFn  = void(*)(uint64_t*, const uint64_t*, const uint64_t*, size_t);
using ShiftFn  = void(*)(uint64_t*, const uint64_t*, size_t, size_t);

// Generic Kogge-Stone carry-lookahead adder (used by all variants)
void kogge_stone_add(uint64_t* sum, const uint64_t* a, const uint64_t* b,
                     size_t n_words, BinOpFn fn_and, BinOpFn fn_or,
                     BinOpFn fn_xor, ShiftFn fn_shl);

// Ripple-carry reference (for testing correctness)
void vec_add_reference(uint64_t* sum, const uint64_t* a, const uint64_t* b, size_t n);

// ==================== Scalar ====================
namespace scalar {
void vec_and(uint64_t* dst, const uint64_t* a, const uint64_t* b, size_t n);
void vec_or (uint64_t* dst, const uint64_t* a, const uint64_t* b, size_t n);
void vec_xor(uint64_t* dst, const uint64_t* a, const uint64_t* b, size_t n);
void vec_not(uint64_t* dst, const uint64_t* a, size_t n);
void vec_shl(uint64_t* dst, const uint64_t* src, size_t k, size_t n);
void vec_shr(uint64_t* dst, const uint64_t* src, size_t k, size_t n);
void vec_add(uint64_t* sum, const uint64_t* a, const uint64_t* b, size_t n_words);
int64_t vec_popcount(const uint64_t* a, size_t n);
bool vec_is_nonzero(const uint64_t* a, size_t n);
}

// ==================== Scalar (no auto-vectorization) ====================
namespace novec {
void vec_and(uint64_t* dst, const uint64_t* a, const uint64_t* b, size_t n);
void vec_or (uint64_t* dst, const uint64_t* a, const uint64_t* b, size_t n);
void vec_xor(uint64_t* dst, const uint64_t* a, const uint64_t* b, size_t n);
void vec_not(uint64_t* dst, const uint64_t* a, size_t n);
void vec_shl(uint64_t* dst, const uint64_t* src, size_t k, size_t n);
void vec_shr(uint64_t* dst, const uint64_t* src, size_t k, size_t n);
void vec_add(uint64_t* sum, const uint64_t* a, const uint64_t* b, size_t n_words);
int64_t vec_popcount(const uint64_t* a, size_t n);
bool vec_is_nonzero(const uint64_t* a, size_t n);
}

// ==================== AVX-512 ====================
namespace avx512 {
void vec_and(uint64_t* dst, const uint64_t* a, const uint64_t* b, size_t n);
void vec_or (uint64_t* dst, const uint64_t* a, const uint64_t* b, size_t n);
void vec_xor(uint64_t* dst, const uint64_t* a, const uint64_t* b, size_t n);
void vec_not(uint64_t* dst, const uint64_t* a, size_t n);
void vec_shl(uint64_t* dst, const uint64_t* src, size_t k, size_t n);
void vec_shr(uint64_t* dst, const uint64_t* src, size_t k, size_t n);
void vec_add(uint64_t* sum, const uint64_t* a, const uint64_t* b, size_t n_words);
int64_t vec_popcount(const uint64_t* a, size_t n);
bool vec_is_nonzero(const uint64_t* a, size_t n);
}

// ==================== OpenMP ====================
namespace omp {
void vec_and(uint64_t* dst, const uint64_t* a, const uint64_t* b, size_t n);
void vec_or (uint64_t* dst, const uint64_t* a, const uint64_t* b, size_t n);
void vec_xor(uint64_t* dst, const uint64_t* a, const uint64_t* b, size_t n);
void vec_not(uint64_t* dst, const uint64_t* a, size_t n);
void vec_shl(uint64_t* dst, const uint64_t* src, size_t k, size_t n);
void vec_shr(uint64_t* dst, const uint64_t* src, size_t k, size_t n);
void vec_add(uint64_t* sum, const uint64_t* a, const uint64_t* b, size_t n_words);
int64_t vec_popcount(const uint64_t* a, size_t n);
bool vec_is_nonzero(const uint64_t* a, size_t n);
}

// ==================== AVX-512 + OpenMP ====================
namespace avx512_omp {
void vec_and(uint64_t* dst, const uint64_t* a, const uint64_t* b, size_t n);
void vec_or (uint64_t* dst, const uint64_t* a, const uint64_t* b, size_t n);
void vec_xor(uint64_t* dst, const uint64_t* a, const uint64_t* b, size_t n);
void vec_not(uint64_t* dst, const uint64_t* a, size_t n);
void vec_shl(uint64_t* dst, const uint64_t* src, size_t k, size_t n);
void vec_shr(uint64_t* dst, const uint64_t* src, size_t k, size_t n);
void vec_add(uint64_t* sum, const uint64_t* a, const uint64_t* b, size_t n_words);
int64_t vec_popcount(const uint64_t* a, size_t n);
bool vec_is_nonzero(const uint64_t* a, size_t n);
}

} // namespace bs
