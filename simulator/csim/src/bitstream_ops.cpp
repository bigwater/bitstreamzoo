#include "bitstream_ops.h"
#include <immintrin.h>
#include <algorithm>

namespace bs {

// ============================================================
// Generic Kogge-Stone carry-lookahead adder
// ============================================================
// Cached temp buffers for Kogge-Stone to avoid repeated alloc/free.
// Repeated aligned_alloc/free of large buffers (>30 MB) causes heap
// corruption in programs with many additions (game_of_life, montgomery_mul).
static thread_local uint64_t* ks_gen  = nullptr;
static thread_local uint64_t* ks_prop = nullptr;
static thread_local uint64_t* ks_tmp  = nullptr;
static thread_local size_t    ks_cap  = 0;

static void ks_ensure(size_t n_words) {
    if (n_words <= ks_cap) return;
    if (ks_gen)  free_stream(ks_gen);
    if (ks_prop) free_stream(ks_prop);
    if (ks_tmp)  free_stream(ks_tmp);
    ks_gen  = alloc_stream(n_words);
    ks_prop = alloc_stream(n_words);
    ks_tmp  = alloc_stream(n_words);
    ks_cap  = n_words;
}

void kogge_stone_add(uint64_t* sum, const uint64_t* a, const uint64_t* b,
                     size_t n_words, BinOpFn fn_and, BinOpFn fn_or,
                     BinOpFn fn_xor, ShiftFn fn_shl) {
    if (n_words == 0) return;

    ks_ensure(n_words);
    uint64_t* gen  = ks_gen;
    uint64_t* prop = ks_prop;
    uint64_t* tmp  = ks_tmp;

    fn_and(gen, a, b, n_words);      // gen  = a & b
    fn_xor(prop, a, b, n_words);     // prop = a ^ b

    // Parallel prefix: after round j, carries propagate 2^(j+1) positions
    size_t n_bits = n_words * 64;
    for (size_t shift = 1; shift < n_bits; shift <<= 1) {
        fn_shl(tmp, gen, shift, n_words);   // tmp = gen << shift
        fn_and(tmp, prop, tmp, n_words);    // tmp = prop & (gen << shift)
        fn_or(gen, gen, tmp, n_words);      // gen = gen | tmp
        fn_shl(tmp, prop, shift, n_words);  // tmp = prop << shift
        fn_and(prop, prop, tmp, n_words);   // prop = prop & tmp
    }

    fn_shl(tmp, gen, 1, n_words);    // carry_in = gen << 1
    fn_xor(sum, a, b, n_words);      // sum = a ^ b
    fn_xor(sum, sum, tmp, n_words);  // sum = (a ^ b) ^ carry_in
    // Buffers kept for reuse (freed at thread exit)
}

// Ripple-carry reference (always correct, for testing)
void vec_add_reference(uint64_t* sum, const uint64_t* a, const uint64_t* b, size_t n) {
    uint64_t carry = 0;
    for (size_t i = 0; i < n; i++) {
        __uint128_t t = (__uint128_t)a[i] + b[i] + carry;
        sum[i] = (uint64_t)t;
        carry = (uint64_t)(t >> 64);
    }
}

// ============================================================
// Macros to reduce repetition for pointwise ops
// ============================================================

#define DEF_BIN_SCALAR(ns, name, op) \
    void ns::name(uint64_t* dst, const uint64_t* a, const uint64_t* b, size_t n) { \
        for (size_t i = 0; i < n; i++) dst[i] = a[i] op b[i]; \
    }

#define DEF_BIN_AVX512(ns, name, intr, op) \
    void ns::name(uint64_t* dst, const uint64_t* a, const uint64_t* b, size_t n) { \
        size_t i = 0; \
        for (; i + 8 <= n; i += 8) { \
            __m512i va = _mm512_loadu_si512(a + i); \
            __m512i vb = _mm512_loadu_si512(b + i); \
            _mm512_storeu_si512(dst + i, intr(va, vb)); \
        } \
        for (; i < n; i++) dst[i] = a[i] op b[i]; \
    }

#define DEF_BIN_OMP(ns, name, op) \
    void ns::name(uint64_t* dst, const uint64_t* a, const uint64_t* b, size_t n) { \
        _Pragma("omp parallel for schedule(static)") \
        for (size_t i = 0; i < n; i++) dst[i] = a[i] op b[i]; \
    }

#define DEF_BIN_AVX512_OMP(ns, name, intr, op) \
    void ns::name(uint64_t* dst, const uint64_t* a, const uint64_t* b, size_t n) { \
        _Pragma("omp parallel for schedule(static)") \
        for (size_t i = 0; i < n; i += 8) { \
            if (i + 8 <= n) { \
                __m512i va = _mm512_loadu_si512(a + i); \
                __m512i vb = _mm512_loadu_si512(b + i); \
                _mm512_storeu_si512(dst + i, intr(va, vb)); \
            } else { \
                for (size_t j = i; j < n; j++) dst[j] = a[j] op b[j]; \
            } \
        } \
    }

// ============================================================
// Scalar implementations
// ============================================================

DEF_BIN_SCALAR(scalar, vec_and, &)
DEF_BIN_SCALAR(scalar, vec_or,  |)
DEF_BIN_SCALAR(scalar, vec_xor, ^)

void scalar::vec_not(uint64_t* dst, const uint64_t* a, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = ~a[i];
}

void scalar::vec_shl(uint64_t* dst, const uint64_t* src, size_t k, size_t n) {
    if (n == 0) return;
    if (k == 0) { if (dst != src) std::memcpy(dst, src, n * 8); return; }

    size_t woff = k / 64;
    size_t boff = k % 64;

    if (woff >= n) { std::memset(dst, 0, n * 8); return; }

    if (boff == 0) {
        for (size_t i = n; i-- > woff; )
            dst[i] = src[i - woff];
    } else {
        for (size_t i = n; i-- > woff + 1; )
            dst[i] = (src[i - woff] << boff) | (src[i - woff - 1] >> (64 - boff));
        dst[woff] = src[0] << boff;
    }
    std::memset(dst, 0, woff * 8);
}

void scalar::vec_shr(uint64_t* dst, const uint64_t* src, size_t k, size_t n) {
    if (n == 0) return;
    if (k == 0) { if (dst != src) std::memcpy(dst, src, n * 8); return; }

    size_t woff = k / 64;
    size_t boff = k % 64;

    if (woff >= n) { std::memset(dst, 0, n * 8); return; }

    size_t limit = n - woff;
    if (boff == 0) {
        for (size_t i = 0; i < limit; i++)
            dst[i] = src[i + woff];
    } else {
        for (size_t i = 0; i < limit - 1; i++)
            dst[i] = (src[i + woff] >> boff) | (src[i + woff + 1] << (64 - boff));
        dst[limit - 1] = src[n - 1] >> boff;
    }
    std::memset(dst + limit, 0, woff * 8);
}

void scalar::vec_add(uint64_t* sum, const uint64_t* a, const uint64_t* b, size_t n_words) {
    kogge_stone_add(sum, a, b, n_words,
                    scalar::vec_and, scalar::vec_or, scalar::vec_xor, scalar::vec_shl);
}

int64_t scalar::vec_popcount(const uint64_t* a, size_t n) {
    int64_t count = 0;
    for (size_t i = 0; i < n; i++) count += __builtin_popcountll(a[i]);
    return count;
}

bool scalar::vec_is_nonzero(const uint64_t* a, size_t n) {
    for (size_t i = 0; i < n; i++) if (a[i]) return true;
    return false;
}

// ============================================================
// Scalar (no auto-vectorization) — guaranteed no SIMD
// ============================================================

#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize")

DEF_BIN_SCALAR(novec, vec_and, &)
DEF_BIN_SCALAR(novec, vec_or,  |)
DEF_BIN_SCALAR(novec, vec_xor, ^)

void novec::vec_not(uint64_t* dst, const uint64_t* a, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = ~a[i];
}

void novec::vec_shl(uint64_t* dst, const uint64_t* src, size_t k, size_t n) {
    if (n == 0) return;
    if (k == 0) { if (dst != src) std::memcpy(dst, src, n * 8); return; }
    size_t woff = k / 64, boff = k % 64;
    if (woff >= n) { std::memset(dst, 0, n * 8); return; }
    if (boff == 0) {
        for (size_t i = n; i-- > woff; ) dst[i] = src[i - woff];
    } else {
        for (size_t i = n; i-- > woff + 1; )
            dst[i] = (src[i - woff] << boff) | (src[i - woff - 1] >> (64 - boff));
        dst[woff] = src[0] << boff;
    }
    std::memset(dst, 0, woff * 8);
}

void novec::vec_shr(uint64_t* dst, const uint64_t* src, size_t k, size_t n) {
    if (n == 0) return;
    if (k == 0) { if (dst != src) std::memcpy(dst, src, n * 8); return; }
    size_t woff = k / 64, boff = k % 64;
    if (woff >= n) { std::memset(dst, 0, n * 8); return; }
    size_t limit = n - woff;
    if (boff == 0) {
        for (size_t i = 0; i < limit; i++) dst[i] = src[i + woff];
    } else {
        for (size_t i = 0; i < limit - 1; i++)
            dst[i] = (src[i + woff] >> boff) | (src[i + woff + 1] << (64 - boff));
        dst[limit - 1] = src[n - 1] >> boff;
    }
    std::memset(dst + limit, 0, woff * 8);
}

void novec::vec_add(uint64_t* sum, const uint64_t* a, const uint64_t* b, size_t n_words) {
    kogge_stone_add(sum, a, b, n_words,
                    novec::vec_and, novec::vec_or, novec::vec_xor, novec::vec_shl);
}

int64_t novec::vec_popcount(const uint64_t* a, size_t n) {
    int64_t count = 0;
    for (size_t i = 0; i < n; i++) count += __builtin_popcountll(a[i]);
    return count;
}

bool novec::vec_is_nonzero(const uint64_t* a, size_t n) {
    for (size_t i = 0; i < n; i++) if (a[i]) return true;
    return false;
}

#pragma GCC pop_options

// ============================================================
// AVX-512 implementations
// ============================================================

DEF_BIN_AVX512(avx512, vec_and, _mm512_and_si512, &)
DEF_BIN_AVX512(avx512, vec_or,  _mm512_or_si512,  |)
DEF_BIN_AVX512(avx512, vec_xor, _mm512_xor_si512, ^)

void avx512::vec_not(uint64_t* dst, const uint64_t* a, size_t n) {
    __m512i ones = _mm512_set1_epi64(-1LL);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512i va = _mm512_loadu_si512(a + i);
        _mm512_storeu_si512(dst + i, _mm512_xor_si512(va, ones));
    }
    for (; i < n; i++) dst[i] = ~a[i];
}

void avx512::vec_shl(uint64_t* dst, const uint64_t* src, size_t k, size_t n) {
    // Use novec implementation which handles dst==src aliasing correctly
    // via backwards iteration. The original AVX-512 forward loop was buggy
    // when dst==src: it overwrote src[0] before the loop read it.
    novec::vec_shl(dst, src, k, n);
}

void avx512::vec_shr(uint64_t* dst, const uint64_t* src, size_t k, size_t n) {
    if (n == 0) return;
    if (k == 0) { if (dst != src) std::memcpy(dst, src, n * 8); return; }

    size_t woff = k / 64;
    size_t boff = k % 64;

    if (woff >= n) { std::memset(dst, 0, n * 8); return; }

    size_t limit = n - woff;

    if (boff == 0) {
        size_t i = 0;
        for (; i + 8 <= limit; i += 8) {
            __m512i v = _mm512_loadu_si512(src + woff + i);
            _mm512_storeu_si512(dst + i, v);
        }
        for (; i < limit; i++) dst[i] = src[i + woff];
    } else {
        __m128i rsh = _mm_cvtsi64_si128(boff);
        __m128i lsh = _mm_cvtsi64_si128(64 - boff);

        size_t rem = limit - 1;
        size_t i = 0;
        for (; i + 8 <= rem; i += 8) {
            __m512i lo = _mm512_loadu_si512(src + woff + i);
            __m512i hi = _mm512_loadu_si512(src + woff + i + 1);
            __m512i r = _mm512_or_si512(
                _mm512_srl_epi64(lo, rsh),
                _mm512_sll_epi64(hi, lsh));
            _mm512_storeu_si512(dst + i, r);
        }
        for (; i < rem; i++)
            dst[i] = (src[i + woff] >> boff) | (src[i + woff + 1] << (64 - boff));
        dst[limit - 1] = src[n - 1] >> boff;
    }
    std::memset(dst + limit, 0, woff * 8);
}

void avx512::vec_add(uint64_t* sum, const uint64_t* a, const uint64_t* b, size_t n_words) {
    kogge_stone_add(sum, a, b, n_words,
                    avx512::vec_and, avx512::vec_or, avx512::vec_xor, avx512::vec_shl);
}

int64_t avx512::vec_popcount(const uint64_t* a, size_t n) {
    int64_t total = 0;
    size_t i = 0;
    // Use VPOPCNTQ (avx512_vpopcntdq)
    __m512i acc = _mm512_setzero_si512();
    for (; i + 8 <= n; i += 8) {
        __m512i v = _mm512_loadu_si512(a + i);
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(v));
    }
    // Horizontal sum of 8 x int64
    total = _mm512_reduce_add_epi64(acc);
    for (; i < n; i++) total += __builtin_popcountll(a[i]);
    return total;
}

bool avx512::vec_is_nonzero(const uint64_t* a, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512i v = _mm512_loadu_si512(a + i);
        if (_mm512_test_epi64_mask(v, v)) return true;
    }
    for (; i < n; i++) if (a[i]) return true;
    return false;
}

// ============================================================
// OpenMP implementations
// ============================================================

DEF_BIN_OMP(omp, vec_and, &)
DEF_BIN_OMP(omp, vec_or,  |)
DEF_BIN_OMP(omp, vec_xor, ^)

void omp::vec_not(uint64_t* dst, const uint64_t* a, size_t n) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) dst[i] = ~a[i];
}

void omp::vec_shl(uint64_t* dst, const uint64_t* src, size_t k, size_t n) {
    // Use novec (backwards iteration) to handle dst==src aliasing correctly.
    // The OMP parallel-for version has race conditions when dst==src.
    novec::vec_shl(dst, src, k, n);
}

void omp::vec_shr(uint64_t* dst, const uint64_t* src, size_t k, size_t n) {
    if (n == 0) return;
    if (k == 0) { if (dst != src) std::memcpy(dst, src, n * 8); return; }

    size_t woff = k / 64;
    size_t boff = k % 64;

    if (woff >= n) { std::memset(dst, 0, n * 8); return; }

    size_t limit = n - woff;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) {
        if (i >= limit) {
            dst[i] = 0;
        } else if (boff == 0) {
            dst[i] = src[i + woff];
        } else if (i == limit - 1) {
            dst[i] = src[n - 1] >> boff;
        } else {
            dst[i] = (src[i + woff] >> boff) | (src[i + woff + 1] << (64 - boff));
        }
    }
}

void omp::vec_add(uint64_t* sum, const uint64_t* a, const uint64_t* b, size_t n_words) {
    kogge_stone_add(sum, a, b, n_words,
                    omp::vec_and, omp::vec_or, omp::vec_xor, omp::vec_shl);
}

int64_t omp::vec_popcount(const uint64_t* a, size_t n) {
    int64_t total = 0;
    #pragma omp parallel for schedule(static) reduction(+:total)
    for (size_t i = 0; i < n; i++) total += __builtin_popcountll(a[i]);
    return total;
}

bool omp::vec_is_nonzero(const uint64_t* a, size_t n) {
    // Reduction on bool (any nonzero)
    bool found = false;
    #pragma omp parallel for schedule(static) reduction(||:found)
    for (size_t i = 0; i < n; i++) found = found || (a[i] != 0);
    return found;
}

// ============================================================
// AVX-512 + OpenMP implementations
// ============================================================

DEF_BIN_AVX512_OMP(avx512_omp, vec_and, _mm512_and_si512, &)
DEF_BIN_AVX512_OMP(avx512_omp, vec_or,  _mm512_or_si512,  |)
DEF_BIN_AVX512_OMP(avx512_omp, vec_xor, _mm512_xor_si512, ^)

void avx512_omp::vec_not(uint64_t* dst, const uint64_t* a, size_t n) {
    __m512i ones = _mm512_set1_epi64(-1LL);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i += 8) {
        if (i + 8 <= n) {
            __m512i va = _mm512_loadu_si512(a + i);
            _mm512_storeu_si512(dst + i, _mm512_xor_si512(va, ones));
        } else {
            for (size_t j = i; j < n; j++) dst[j] = ~a[j];
        }
    }
}

void avx512_omp::vec_shl(uint64_t* dst, const uint64_t* src, size_t k, size_t n) {
    // Use novec (backwards iteration) to handle dst==src aliasing correctly.
    novec::vec_shl(dst, src, k, n);
}

void avx512_omp::vec_shr(uint64_t* dst, const uint64_t* src, size_t k, size_t n) {
    if (n == 0) return;
    if (k == 0) { if (dst != src) std::memcpy(dst, src, n * 8); return; }

    size_t woff = k / 64;
    size_t boff = k % 64;

    if (woff >= n) { std::memset(dst, 0, n * 8); return; }

    size_t limit = n - woff;

    if (boff == 0) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < limit; i += 8) {
            if (i + 8 <= limit) {
                _mm512_storeu_si512(dst + i, _mm512_loadu_si512(src + woff + i));
            } else {
                for (size_t j = i; j < limit; j++) dst[j] = src[j + woff];
            }
        }
    } else {
        size_t rem = limit - 1;
        __m128i rsh = _mm_cvtsi64_si128(boff);
        __m128i lsh = _mm_cvtsi64_si128(64 - boff);

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < rem; i += 8) {
            if (i + 8 <= rem) {
                __m512i lo = _mm512_loadu_si512(src + woff + i);
                __m512i hi = _mm512_loadu_si512(src + woff + i + 1);
                _mm512_storeu_si512(dst + i,
                    _mm512_or_si512(_mm512_srl_epi64(lo, rsh),
                                    _mm512_sll_epi64(hi, lsh)));
            } else {
                for (size_t j = i; j < rem; j++)
                    dst[j] = (src[j + woff] >> boff) | (src[j + woff + 1] << (64 - boff));
            }
        }
        dst[limit - 1] = src[n - 1] >> boff;
    }
    std::memset(dst + limit, 0, woff * 8);
}

void avx512_omp::vec_add(uint64_t* sum, const uint64_t* a, const uint64_t* b, size_t n_words) {
    kogge_stone_add(sum, a, b, n_words,
                    avx512_omp::vec_and, avx512_omp::vec_or,
                    avx512_omp::vec_xor, avx512_omp::vec_shl);
}

int64_t avx512_omp::vec_popcount(const uint64_t* a, size_t n) {
    int64_t total = 0;
    #pragma omp parallel for schedule(static) reduction(+:total)
    for (size_t i = 0; i < n; i += 8) {
        if (i + 8 <= n) {
            __m512i v = _mm512_loadu_si512(a + i);
            __m512i pc = _mm512_popcnt_epi64(v);
            total += _mm512_reduce_add_epi64(pc);
        } else {
            for (size_t j = i; j < n; j++) total += __builtin_popcountll(a[j]);
        }
    }
    return total;
}

bool avx512_omp::vec_is_nonzero(const uint64_t* a, size_t n) {
    bool found = false;
    #pragma omp parallel for schedule(static) reduction(||:found)
    for (size_t i = 0; i < n; i += 8) {
        if (i + 8 <= n) {
            __m512i v = _mm512_loadu_si512(a + i);
            found = found || _mm512_test_epi64_mask(v, v);
        } else {
            for (size_t j = i; j < n; j++) found = found || (a[j] != 0);
        }
    }
    return found;
}

#undef DEF_BIN_SCALAR
#undef DEF_BIN_AVX512
#undef DEF_BIN_OMP
#undef DEF_BIN_AVX512_OMP

} // namespace bs
