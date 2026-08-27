#include "bitstream_ops.h"
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <stdexcept>
#include <random>
#include <chrono>
#include <functional>
#include <vector>

// ============================================================
// Test framework
// ============================================================

static int g_pass = 0, g_fail = 0;

#define TEST(name) static void test_##name()
#define RUN(name) do { \
    printf("  %-50s", #name); fflush(stdout); \
    try { test_##name(); printf("PASS\n"); g_pass++; } \
    catch (const std::exception& e) { printf("FAIL: %s\n", e.what()); g_fail++; } \
    catch (...) { printf("FAIL (unknown)\n"); g_fail++; } \
} while(0)

#define CHECK(cond) do { if (!(cond)) { \
    char buf[256]; snprintf(buf, sizeof(buf), "CHECK failed at line %d: %s", __LINE__, #cond); \
    throw std::runtime_error(buf); \
}} while(0)

#define CHECK_EQ(a, b) do { if ((a) != (b)) { \
    char buf[256]; snprintf(buf, sizeof(buf), "CHECK_EQ failed at line %d: 0x%lx != 0x%lx", \
        __LINE__, (uint64_t)(a), (uint64_t)(b)); \
    throw std::runtime_error(buf); \
}} while(0)

using namespace bs;

// ============================================================
// Helpers
// ============================================================

// Fill with deterministic pattern
static void fill_pattern(uint64_t* arr, size_t n, uint64_t seed) {
    for (size_t i = 0; i < n; i++)
        arr[i] = seed * 0x0123456789ABCDEFULL + i * 0xFEDCBA9876543210ULL;
}

// Fill random
static void fill_random(uint64_t* arr, size_t n, std::mt19937_64& rng) {
    for (size_t i = 0; i < n; i++) arr[i] = rng();
}

// Compare arrays
static bool arrays_equal(const uint64_t* a, const uint64_t* b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            fprintf(stderr, "  mismatch at word %zu: 0x%016lx vs 0x%016lx\n", i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

// Type for binary pointwise ops
using BinOp = void(*)(uint64_t*, const uint64_t*, const uint64_t*, size_t);
using UnOp  = void(*)(uint64_t*, const uint64_t*, size_t);
using ShOp  = void(*)(uint64_t*, const uint64_t*, size_t, size_t);

// Test a binary pointwise op against a scalar reference
static void check_binary(BinOp fn, uint64_t(*ref)(uint64_t, uint64_t), size_t n) {
    uint64_t* a = alloc_stream(n);
    uint64_t* b = alloc_stream(n);
    uint64_t* dst = alloc_stream(n);
    fill_pattern(a, n, 1);
    fill_pattern(b, n, 2);
    fn(dst, a, b, n);
    for (size_t i = 0; i < n; i++)
        CHECK_EQ(dst[i], ref(a[i], b[i]));
    free_stream(a); free_stream(b); free_stream(dst);
}

// Test NOT against scalar reference
static void check_not(UnOp fn, size_t n) {
    uint64_t* a = alloc_stream(n);
    uint64_t* dst = alloc_stream(n);
    fill_pattern(a, n, 3);
    fn(dst, a, n);
    for (size_t i = 0; i < n; i++)
        CHECK_EQ(dst[i], ~a[i]);
    free_stream(a); free_stream(dst);
}

// Test shift against scalar reference
static void check_shift(ShOp fn, ShOp ref_fn, size_t k, size_t n) {
    uint64_t* src = alloc_stream(n);
    uint64_t* dst = alloc_stream(n);
    uint64_t* expected = alloc_stream(n);
    fill_pattern(src, n, 4);
    ref_fn(expected, src, k, n);
    fn(dst, src, k, n);
    CHECK(arrays_equal(dst, expected, n));
    free_stream(src); free_stream(dst); free_stream(expected);
}

// ============================================================
// Pointwise binary op tests
// ============================================================

static auto ref_and = [](uint64_t a, uint64_t b) { return a & b; };
static auto ref_or  = [](uint64_t a, uint64_t b) { return a | b; };
static auto ref_xor = [](uint64_t a, uint64_t b) { return a ^ b; };

// Test all 4 variants at multiple sizes
#define TEST_BINARY_ALL(opname, ref_fn) \
    TEST(opname##_all) { \
        for (size_t n : {1, 7, 8, 15, 16, 100, 1000}) { \
            check_binary(scalar::vec_##opname, ref_fn, n); \
            check_binary(avx512::vec_##opname, ref_fn, n); \
            check_binary(omp::vec_##opname, ref_fn, n); \
            check_binary(avx512_omp::vec_##opname, ref_fn, n); \
        } \
    }

TEST_BINARY_ALL(and, ref_and)
TEST_BINARY_ALL(or,  ref_or)
TEST_BINARY_ALL(xor, ref_xor)

TEST(not_all) {
    for (size_t n : {1, 7, 8, 15, 16, 100, 1000}) {
        check_not(scalar::vec_not, n);
        check_not(avx512::vec_not, n);
        check_not(omp::vec_not, n);
        check_not(avx512_omp::vec_not, n);
    }
}

// ============================================================
// Shift tests
// ============================================================

// Test SHL at various shift amounts, all variants vs scalar reference
TEST(shl_all) {
    for (size_t n : {1, 8, 15, 100}) {
        for (size_t k : {0UL, 1UL, 31UL, 63UL, 64UL, 65UL, 127UL, 128UL, 200UL}) {
            // Skip shifts larger than total bits for small n
            check_shift(scalar::vec_shl,     scalar::vec_shl, k, n);
            check_shift(avx512::vec_shl,     scalar::vec_shl, k, n);
            check_shift(omp::vec_shl,        scalar::vec_shl, k, n);
            check_shift(avx512_omp::vec_shl, scalar::vec_shl, k, n);
        }
    }
}

TEST(shr_all) {
    for (size_t n : {1, 8, 15, 100}) {
        for (size_t k : {0UL, 1UL, 31UL, 63UL, 64UL, 65UL, 127UL, 128UL, 200UL}) {
            check_shift(scalar::vec_shr,     scalar::vec_shr, k, n);
            check_shift(avx512::vec_shr,     scalar::vec_shr, k, n);
            check_shift(omp::vec_shr,        scalar::vec_shr, k, n);
            check_shift(avx512_omp::vec_shr, scalar::vec_shr, k, n);
        }
    }
}

// SHL carry propagation: shift a pattern where MSB of each word is set
TEST(shl_carry) {
    size_t n = 4;
    uint64_t* src = alloc_stream(n);
    uint64_t* dst = alloc_stream(n);
    src[0] = 0x8000000000000000ULL;  // MSB of word 0
    src[1] = 0;
    src[2] = 0x8000000000000000ULL;
    src[3] = 0;

    scalar::vec_shl(dst, src, 1, n);
    CHECK_EQ(dst[0], 0ULL);
    CHECK_EQ(dst[1], 1ULL);  // Carry from word 0
    CHECK_EQ(dst[2], 0ULL);
    CHECK_EQ(dst[3], 1ULL);  // Carry from word 2

    free_stream(src); free_stream(dst);
}

// SHR carry propagation: shift a pattern where LSB of each word is set
TEST(shr_carry) {
    size_t n = 4;
    uint64_t* src = alloc_stream(n);
    uint64_t* dst = alloc_stream(n);
    src[0] = 0;
    src[1] = 1;  // LSB of word 1
    src[2] = 0;
    src[3] = 1;

    scalar::vec_shr(dst, src, 1, n);
    CHECK_EQ(dst[0], 0x8000000000000000ULL);  // Carry from word 1
    CHECK_EQ(dst[1], 0ULL);
    CHECK_EQ(dst[2], 0x8000000000000000ULL);  // Carry from word 3
    CHECK_EQ(dst[3], 0ULL);

    free_stream(src); free_stream(dst);
}

// ============================================================
// ADD tests (Kogge-Stone vs ripple-carry reference)
// ============================================================

using AddFn = void(*)(uint64_t*, const uint64_t*, const uint64_t*, size_t);

static void check_add(AddFn fn, size_t n) {
    uint64_t* a = alloc_stream(n);
    uint64_t* b = alloc_stream(n);
    uint64_t* got = alloc_stream(n);
    uint64_t* expected = alloc_stream(n);

    std::mt19937_64 rng(42 + n);
    fill_random(a, n, rng);
    fill_random(b, n, rng);

    vec_add_reference(expected, a, b, n);
    fn(got, a, b, n);
    CHECK(arrays_equal(got, expected, n));

    free_stream(a); free_stream(b); free_stream(got); free_stream(expected);
}

TEST(add_all) {
    for (size_t n : {1, 2, 4, 8, 15, 16, 100}) {
        check_add(scalar::vec_add, n);
        check_add(avx512::vec_add, n);
        check_add(omp::vec_add, n);
        check_add(avx512_omp::vec_add, n);
    }
}

// ADD edge case: no carry
TEST(add_no_carry) {
    size_t n = 2;
    uint64_t* a = alloc_stream(n);
    uint64_t* b = alloc_stream(n);
    uint64_t* sum = alloc_stream(n);
    a[0] = 1; a[1] = 2;
    b[0] = 3; b[1] = 4;
    scalar::vec_add(sum, a, b, n);
    CHECK_EQ(sum[0], 4ULL);
    CHECK_EQ(sum[1], 6ULL);
    free_stream(a); free_stream(b); free_stream(sum);
}

// ADD edge case: carry within word
TEST(add_carry_word) {
    size_t n = 2;
    uint64_t* a = alloc_stream(n);
    uint64_t* b = alloc_stream(n);
    uint64_t* sum = alloc_stream(n);
    a[0] = 0xFFFFFFFFFFFFFFFFULL; a[1] = 0;
    b[0] = 1; b[1] = 0;
    scalar::vec_add(sum, a, b, n);
    CHECK_EQ(sum[0], 0ULL);
    CHECK_EQ(sum[1], 1ULL);  // Carry into word 1
    free_stream(a); free_stream(b); free_stream(sum);
}

// ADD edge case: carry chain across multiple words
TEST(add_carry_chain) {
    size_t n = 4;
    uint64_t* a = alloc_stream(n);
    uint64_t* b = alloc_stream(n);
    uint64_t* sum = alloc_stream(n);
    uint64_t* expected = alloc_stream(n);
    a[0] = ~0ULL; a[1] = ~0ULL; a[2] = ~0ULL; a[3] = 0;
    b[0] = 1; b[1] = 0; b[2] = 0; b[3] = 0;
    vec_add_reference(expected, a, b, n);
    scalar::vec_add(sum, a, b, n);
    CHECK(arrays_equal(sum, expected, n));
    CHECK_EQ(sum[0], 0ULL);
    CHECK_EQ(sum[1], 0ULL);
    CHECK_EQ(sum[2], 0ULL);
    CHECK_EQ(sum[3], 1ULL);  // Carry propagated through 3 words
    free_stream(a); free_stream(b); free_stream(sum); free_stream(expected);
}

// ADD with many random cases
TEST(add_random_large) {
    size_t n = 1000;
    std::mt19937_64 rng(12345);
    for (int trial = 0; trial < 10; trial++) {
        uint64_t* a = alloc_stream(n);
        uint64_t* b = alloc_stream(n);
        uint64_t* got = alloc_stream(n);
        uint64_t* expected = alloc_stream(n);
        fill_random(a, n, rng);
        fill_random(b, n, rng);
        vec_add_reference(expected, a, b, n);

        scalar::vec_add(got, a, b, n);
        CHECK(arrays_equal(got, expected, n));

        avx512::vec_add(got, a, b, n);
        CHECK(arrays_equal(got, expected, n));

        omp::vec_add(got, a, b, n);
        CHECK(arrays_equal(got, expected, n));

        avx512_omp::vec_add(got, a, b, n);
        CHECK(arrays_equal(got, expected, n));

        free_stream(a); free_stream(b); free_stream(got); free_stream(expected);
    }
}

// ============================================================
// Popcount tests
// ============================================================

TEST(popcount_all) {
    for (size_t n : {1, 7, 8, 15, 100}) {
        uint64_t* a = alloc_stream(n);
        fill_pattern(a, n, 5);
        int64_t expected = scalar::vec_popcount(a, n);

        // Verify scalar against per-word builtin
        int64_t manual = 0;
        for (size_t i = 0; i < n; i++) manual += __builtin_popcountll(a[i]);
        CHECK_EQ(expected, manual);

        CHECK_EQ(avx512::vec_popcount(a, n), expected);
        CHECK_EQ(omp::vec_popcount(a, n), expected);
        CHECK_EQ(avx512_omp::vec_popcount(a, n), expected);
        free_stream(a);
    }
}

TEST(popcount_known) {
    uint64_t* a = alloc_stream(2);
    a[0] = 0; a[1] = 0;
    CHECK_EQ(scalar::vec_popcount(a, 2), 0);
    a[0] = 1; a[1] = 0;
    CHECK_EQ(scalar::vec_popcount(a, 2), 1);
    a[0] = ~0ULL; a[1] = ~0ULL;
    CHECK_EQ(scalar::vec_popcount(a, 2), 128);
    free_stream(a);
}

// ============================================================
// Is_nonzero tests
// ============================================================

TEST(is_nonzero_all) {
    for (size_t n : {1, 8, 15, 100}) {
        uint64_t* a = alloc_stream(n);  // All zeros
        CHECK(!scalar::vec_is_nonzero(a, n));
        CHECK(!avx512::vec_is_nonzero(a, n));
        CHECK(!omp::vec_is_nonzero(a, n));
        CHECK(!avx512_omp::vec_is_nonzero(a, n));

        a[n / 2] = 1;  // Set one bit
        CHECK(scalar::vec_is_nonzero(a, n));
        CHECK(avx512::vec_is_nonzero(a, n));
        CHECK(omp::vec_is_nonzero(a, n));
        CHECK(avx512_omp::vec_is_nonzero(a, n));

        free_stream(a);
    }
}

// ============================================================
// Cross-variant consistency (all variants produce same result)
// ============================================================

TEST(cross_variant_consistency) {
    size_t n = 500;
    std::mt19937_64 rng(99);
    uint64_t* a = alloc_stream(n);
    uint64_t* b = alloc_stream(n);
    uint64_t* r_scalar = alloc_stream(n);
    uint64_t* r_avx512 = alloc_stream(n);
    uint64_t* r_omp = alloc_stream(n);
    uint64_t* r_simd_omp = alloc_stream(n);
    fill_random(a, n, rng);
    fill_random(b, n, rng);

    // AND
    scalar::vec_and(r_scalar, a, b, n);
    avx512::vec_and(r_avx512, a, b, n);
    omp::vec_and(r_omp, a, b, n);
    avx512_omp::vec_and(r_simd_omp, a, b, n);
    CHECK(arrays_equal(r_scalar, r_avx512, n));
    CHECK(arrays_equal(r_scalar, r_omp, n));
    CHECK(arrays_equal(r_scalar, r_simd_omp, n));

    // SHL
    scalar::vec_shl(r_scalar, a, 37, n);
    avx512::vec_shl(r_avx512, a, 37, n);
    omp::vec_shl(r_omp, a, 37, n);
    avx512_omp::vec_shl(r_simd_omp, a, 37, n);
    CHECK(arrays_equal(r_scalar, r_avx512, n));
    CHECK(arrays_equal(r_scalar, r_omp, n));
    CHECK(arrays_equal(r_scalar, r_simd_omp, n));

    // SHR
    scalar::vec_shr(r_scalar, a, 71, n);
    avx512::vec_shr(r_avx512, a, 71, n);
    omp::vec_shr(r_omp, a, 71, n);
    avx512_omp::vec_shr(r_simd_omp, a, 71, n);
    CHECK(arrays_equal(r_scalar, r_avx512, n));
    CHECK(arrays_equal(r_scalar, r_omp, n));
    CHECK(arrays_equal(r_scalar, r_simd_omp, n));

    // ADD
    scalar::vec_add(r_scalar, a, b, n);
    avx512::vec_add(r_avx512, a, b, n);
    omp::vec_add(r_omp, a, b, n);
    avx512_omp::vec_add(r_simd_omp, a, b, n);
    CHECK(arrays_equal(r_scalar, r_avx512, n));
    CHECK(arrays_equal(r_scalar, r_omp, n));
    CHECK(arrays_equal(r_scalar, r_simd_omp, n));

    free_stream(a); free_stream(b);
    free_stream(r_scalar); free_stream(r_avx512);
    free_stream(r_omp); free_stream(r_simd_omp);
}

// ============================================================
// Main
// ============================================================

int main() {
    printf("=== Bitstream Ops Unit Tests ===\n\n");

    printf("--- Pointwise Binary ---\n");
    RUN(and_all);
    RUN(or_all);
    RUN(xor_all);

    printf("\n--- NOT ---\n");
    RUN(not_all);

    printf("\n--- Shift Left ---\n");
    RUN(shl_all);
    RUN(shl_carry);

    printf("\n--- Shift Right ---\n");
    RUN(shr_all);
    RUN(shr_carry);

    printf("\n--- ADD (Kogge-Stone) ---\n");
    RUN(add_all);
    RUN(add_no_carry);
    RUN(add_carry_word);
    RUN(add_carry_chain);
    RUN(add_random_large);

    printf("\n--- Popcount ---\n");
    RUN(popcount_all);
    RUN(popcount_known);

    printf("\n--- Is_nonzero ---\n");
    RUN(is_nonzero_all);

    printf("\n--- Cross-variant Consistency ---\n");
    RUN(cross_variant_consistency);

    printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
