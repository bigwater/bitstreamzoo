#include "add_carry.cuh"
#include "kernels.cuh"
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ── Helpers ─────────────────────────────────────────────────────────

static int g_pass = 0, g_fail = 0;

#define CHECK_EQ(a, b, msg) do { \
    if ((a) != (b)) { \
        fprintf(stderr, "  FAIL: %s (got %llu, expected %llu)\n", \
                msg, (unsigned long long)(a), (unsigned long long)(b)); \
        ++g_fail; \
    } else { \
        ++g_pass; \
    } \
} while (0)

// CUDA event timer: measures GPU time between start/stop
struct GpuTimer {
    cudaEvent_t start, stop;
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }
    void begin() { CUDA_CHECK(cudaEventRecord(start)); }
    float end_ms() {
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
    ~GpuTimer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
};

template<int BITS>
struct TestBuf {
    using word_t = typename WordTraits<BITS>::word_t;
    word_t* d_ptr = nullptr;
    int n;
    TestBuf(int n) : n(n) {
        CUDA_CHECK(cudaMalloc(&d_ptr, sizeof(word_t) * n));
    }
    void from_host(const std::vector<word_t>& v) {
        CUDA_CHECK(cudaMemcpy(d_ptr, v.data(), sizeof(word_t) * v.size(),
                              cudaMemcpyHostToDevice));
    }
    std::vector<word_t> to_host() {
        std::vector<word_t> h(n);
        CUDA_CHECK(cudaMemcpy(h.data(), d_ptr, sizeof(word_t) * n,
                              cudaMemcpyDeviceToHost));
        return h;
    }
    ~TestBuf() { if (d_ptr) cudaFree(d_ptr); }
};

// Reference: serial addition with carry propagation
template<typename word_t>
std::vector<word_t> ref_add(const std::vector<word_t>& a, const std::vector<word_t>& b) {
    int N = a.size();
    std::vector<word_t> result(N);
    word_t carry = 0;
    for (int i = 0; i < N; ++i) {
        word_t s = a[i] + b[i];
        word_t c1 = (s < a[i]) ? 1 : 0;
        word_t s2 = s + carry;
        word_t c2 = (s2 < s) ? 1 : 0;
        result[i] = s2;
        carry = c1 | c2;
    }
    return result;
}

// ── Test: kern_not ──────────────────────────────────────────────────

template<int BITS>
float test_kern_not() {
    using word_t = typename WordTraits<BITS>::word_t;
    constexpr int N = 3;

    std::vector<word_t> src = {0, ~word_t(0), word_t(0xA5)};
    TestBuf<BITS> ds(N), dd(N);
    ds.from_host(src);

    GpuTimer t;
    t.begin();
    kern_not<BITS><<<1, N>>>(dd.d_ptr, ds.d_ptr, N);
    float ms = t.end_ms();

    auto r = dd.to_host();
    CHECK_EQ(r[0], ~word_t(0), "not[0]");
    CHECK_EQ(r[1], word_t(0), "not[1]");
    CHECK_EQ(r[2], word_t(~word_t(0xA5)), "not[2]");
    return ms;
}

// ── Test: kern_and, kern_or, kern_xor ───────────────────────────────

template<int BITS>
float test_kern_logic() {
    using word_t = typename WordTraits<BITS>::word_t;
    constexpr int N = 2;

    std::vector<word_t> a = {0xFF, 0x0F};
    std::vector<word_t> b = {0xF0, 0xF0};

    TestBuf<BITS> da(N), db(N), dd(N);
    da.from_host(a); db.from_host(b);

    GpuTimer t;
    t.begin();
    kern_and<BITS><<<1, N>>>(dd.d_ptr, da.d_ptr, db.d_ptr, N);
    kern_or<BITS><<<1, N>>>(dd.d_ptr, da.d_ptr, db.d_ptr, N);
    kern_xor<BITS><<<1, N>>>(dd.d_ptr, da.d_ptr, db.d_ptr, N);
    float ms = t.end_ms();

    // Verify last one (xor)
    auto r = dd.to_host();
    CHECK_EQ(r[0], word_t(0x0F), "xor[0]");
    CHECK_EQ(r[1], word_t(0xFF), "xor[1]");

    // Also verify and/or
    kern_and<BITS><<<1, N>>>(dd.d_ptr, da.d_ptr, db.d_ptr, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    r = dd.to_host();
    CHECK_EQ(r[0], word_t(0xF0), "and[0]");
    CHECK_EQ(r[1], word_t(0x00), "and[1]");

    kern_or<BITS><<<1, N>>>(dd.d_ptr, da.d_ptr, db.d_ptr, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    r = dd.to_host();
    CHECK_EQ(r[0], word_t(0xFF), "or[0]");
    CHECK_EQ(r[1], word_t(0xFF), "or[1]");
    return ms;
}

// ── Test: kern_add (per-word, no carry) ─────────────────────────────

template<int BITS>
float test_kern_add() {
    using word_t = typename WordTraits<BITS>::word_t;
    constexpr int N = 4;

    std::vector<word_t> a = {3, 7, ~word_t(0), 100};
    std::vector<word_t> b = {5, 2, 1, 200};

    TestBuf<BITS> da(N), db(N), dd(N);
    da.from_host(a); db.from_host(b);

    GpuTimer t;
    t.begin();
    kern_add<BITS><<<1, N>>>(dd.d_ptr, da.d_ptr, db.d_ptr, N);
    float ms = t.end_ms();

    auto r = dd.to_host();
    CHECK_EQ(r[0], word_t(8), "add[0]=3+5");
    CHECK_EQ(r[1], word_t(9), "add[1]=7+2");
    CHECK_EQ(r[2], word_t(0), "add[2]=0xFFF..F+1 wraps");
    CHECK_EQ(r[3], word_t(300), "add[3]=100+200");
    return ms;
}

// ── Test: kern_shl / kern_shr ───────────────────────────────────────

template<int BITS>
float test_kern_shift() {
    using word_t = typename WordTraits<BITS>::word_t;
    constexpr int WB = WordTraits<BITS>::WORD_BITS;
    constexpr int N = 4;

    std::vector<word_t> src = {1, 0, 0, 0};
    TestBuf<BITS> ds(N), dd(N);
    ds.from_host(src);

    GpuTimer t;
    t.begin();
    kern_shl<BITS><<<1, N>>>(dd.d_ptr, ds.d_ptr, 1, N);
    kern_shl<BITS><<<1, N>>>(dd.d_ptr, ds.d_ptr, WB, N);
    kern_shr<BITS><<<1, N>>>(dd.d_ptr, ds.d_ptr, 1, N);
    float ms = t.end_ms();

    // Verify SHL by 1
    kern_shl<BITS><<<1, N>>>(dd.d_ptr, ds.d_ptr, 1, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto r = dd.to_host();
    CHECK_EQ(r[0], word_t(2), "shl1[0]");
    CHECK_EQ(r[1], word_t(0), "shl1[1]");

    // SHL by WORD_BITS
    kern_shl<BITS><<<1, N>>>(dd.d_ptr, ds.d_ptr, WB, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    r = dd.to_host();
    CHECK_EQ(r[0], word_t(0), "shl_WB[0]");
    CHECK_EQ(r[1], word_t(1), "shl_WB[1]");

    // SHR by 1 on [2, 0, 0, 0]
    std::vector<word_t> src2 = {2, 0, 0, 0};
    ds.from_host(src2);
    kern_shr<BITS><<<1, N>>>(dd.d_ptr, ds.d_ptr, 1, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    r = dd.to_host();
    CHECK_EQ(r[0], word_t(1), "shr1[0]");

    // SHR by WORD_BITS on [0, 1, 0, 0]
    std::vector<word_t> src3 = {0, 1, 0, 0};
    ds.from_host(src3);
    kern_shr<BITS><<<1, N>>>(dd.d_ptr, ds.d_ptr, WB, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    r = dd.to_host();
    CHECK_EQ(r[0], word_t(1), "shr_WB[0]");
    CHECK_EQ(r[1], word_t(0), "shr_WB[1]");
    return ms;
}

// ── Test: kern_popcount_per_word ────────────────────────────────────

template<int BITS>
float test_kern_popcount() {
    using word_t = typename WordTraits<BITS>::word_t;
    constexpr int N = 4;

    std::vector<word_t> src = {0, 1, ~word_t(0), word_t(0xFF)};
    TestBuf<BITS> ds(N);
    ds.from_host(src);

    unsigned long long* d_partial;
    CUDA_CHECK(cudaMalloc(&d_partial, sizeof(unsigned long long) * N));

    GpuTimer t;
    t.begin();
    // lw=N, last_mask=all-ones -> no padding mask: full per-word popcount
    kern_popcount_per_word<BITS><<<1, N>>>(d_partial, ds.d_ptr, N,
                                           (int64_t)N, ~word_t(0));
    float ms = t.end_ms();

    std::vector<unsigned long long> h(N);
    CUDA_CHECK(cudaMemcpy(h.data(), d_partial, sizeof(unsigned long long) * N,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_partial));

    CHECK_EQ(h[0], 0ULL, "popcount[0]=0");
    CHECK_EQ(h[1], 1ULL, "popcount[1]=1");
    CHECK_EQ(h[2], (unsigned long long)WordTraits<BITS>::WORD_BITS, "popcount[2]=all ones");
    CHECK_EQ(h[3], 8ULL, "popcount[3]=0xFF");
    return ms;
}

// ── Test: carry propagation correctness ─────────────────────────────

// Helper: run kern_add + gpu_add_carry_propagate and verify against ref_add
template<int BITS>
float verify_add_carry(const std::vector<typename WordTraits<BITS>::word_t>& a,
                       const std::vector<typename WordTraits<BITS>::word_t>& b,
                       const char* label) {
    using word_t = typename WordTraits<BITS>::word_t;
    int N = a.size();
    auto expected = ref_add<word_t>(a, b);

    TestBuf<BITS> da(N), db(N), dd(N);
    da.from_host(a); db.from_host(b);

    int G = grid_size(N);
    kern_add<BITS><<<G, BLOCK_SIZE>>>(dd.d_ptr, da.d_ptr, db.d_ptr, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    AddCarryBufs<BITS> bufs;
    GpuTimer t;
    t.begin();
    gpu_add_carry_propagate<BITS>(dd.d_ptr, da.d_ptr, N, bufs);
    float ms = t.end_ms();

    auto r = dd.to_host();
    bool ok = true;
    int first_fail = -1;
    for (int i = 0; i < N; ++i) {
        if (r[i] != expected[i]) {
            if (first_fail < 0) first_fail = i;
            ok = false;
        }
    }
    if (ok) {
        ++g_pass;
    } else {
        fprintf(stderr, "  FAIL: %s (first mismatch at [%d]: got %llu, expected %llu)\n",
                label, first_fail,
                (unsigned long long)r[first_fail],
                (unsigned long long)expected[first_fail]);
        ++g_fail;
    }
    bufs.free();
    return ms;
}

template<int BITS>
float test_carry_single_word() {
    using word_t = typename WordTraits<BITS>::word_t;
    // N=1: no carry propagation needed, just kern_add
    std::vector<word_t> a = {~word_t(0)};
    std::vector<word_t> b = {1};

    TestBuf<BITS> da(1), db(1), dd(1);
    da.from_host(a); db.from_host(b);

    GpuTimer t;
    t.begin();
    kern_add<BITS><<<1, 1>>>(dd.d_ptr, da.d_ptr, db.d_ptr, 1);
    float ms = t.end_ms();

    auto r = dd.to_host();
    CHECK_EQ(r[0], word_t(0), "single_word wraps to 0");
    return ms;
}

template<int BITS>
float test_carry_basic() {
    using word_t = typename WordTraits<BITS>::word_t;
    // a = [0xFFF..F, 0, 0, 0], b = [1, 0, 0, 0] -> [0, 1, 0, 0]
    std::vector<word_t> a = {~word_t(0), 0, 0, 0};
    std::vector<word_t> b = {1, 0, 0, 0};
    return verify_add_carry<BITS>(a, b, "carry_basic");
}

template<int BITS>
float test_carry_chain() {
    using word_t = typename WordTraits<BITS>::word_t;
    // Long carry chain: a = [0xFFF..F x3, 0], b = [1, 0, 0, 0] -> [0, 0, 0, 1]
    std::vector<word_t> a = {~word_t(0), ~word_t(0), ~word_t(0), 0};
    std::vector<word_t> b = {1, 0, 0, 0};
    return verify_add_carry<BITS>(a, b, "carry_chain");
}

template<int BITS>
float test_carry_no_carry() {
    using word_t = typename WordTraits<BITS>::word_t;
    std::vector<word_t> a = {1, 2, 3, 4};
    std::vector<word_t> b = {10, 20, 30, 40};
    return verify_add_carry<BITS>(a, b, "carry_no_carry");
}

template<int BITS>
float test_carry_all_ones() {
    using word_t = typename WordTraits<BITS>::word_t;
    // All-ones + 1: carry through all 8 words
    constexpr int N = 8;
    std::vector<word_t> a(N, ~word_t(0));
    std::vector<word_t> b(N, 0);
    b[0] = 1;
    return verify_add_carry<BITS>(a, b, "carry_all_ones_8");
}

template<int BITS>
float test_carry_multiblock() {
    using word_t = typename WordTraits<BITS>::word_t;
    // N=512, all-ones + 1: carry across 2 thread blocks
    constexpr int N = 512;
    std::vector<word_t> a(N, ~word_t(0));
    std::vector<word_t> b(N, 0);
    b[0] = 1;
    return verify_add_carry<BITS>(a, b, "carry_multiblock_512");
}

template<int BITS>
float test_carry_multiblock_mixed() {
    using word_t = typename WordTraits<BITS>::word_t;
    // Mixed pattern: carries at block boundaries
    constexpr int N = 1024;
    std::vector<word_t> a(N, 0);
    std::vector<word_t> b(N, 0);
    for (int p : {0, 255, 256, 511, 512}) {
        if (p < N) a[p] = ~word_t(0);
    }
    b[0] = 2;
    b[255] = 1;
    b[512] = 1;
    return verify_add_carry<BITS>(a, b, "carry_mixed_1024");
}

// ── Scaling benchmarks ──────────────────────────────────────────────
// Run each op at various sizes and report timing.
// Flag anomalies: time should scale roughly with N / BLOCK_SIZE.

struct BenchRow {
    const char* op;
    int bits;
    int N;
    float gpu_ms;      // GPU kernel time (CUDA events)
    float wall_ms;     // Wall clock including host overhead
    bool correct;
};

static std::vector<BenchRow> g_bench;

// Helper: run a single kernel, return GPU time (CUDA events) and wall time
template<int BITS>
void bench_simple_op(const char* op_name, int N) {
    using word_t = typename WordTraits<BITS>::word_t;
    int G = grid_size(N);

    TestBuf<BITS> da(N), db(N), dd(N);
    {
        std::vector<word_t> va(N), vb(N);
        for (int i = 0; i < N; ++i) {
            va[i] = static_cast<word_t>(i * 17 + 3);
            vb[i] = static_cast<word_t>(i * 31 + 7);
        }
        da.from_host(va); db.from_host(vb);
    }

    // Warm up
    kern_and<BITS><<<G, BLOCK_SIZE>>>(dd.d_ptr, da.d_ptr, db.d_ptr, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer t;
    auto wall0 = std::chrono::high_resolution_clock::now();
    t.begin();

    if (strcmp(op_name, "AND") == 0)
        kern_and<BITS><<<G, BLOCK_SIZE>>>(dd.d_ptr, da.d_ptr, db.d_ptr, N);
    else if (strcmp(op_name, "OR") == 0)
        kern_or<BITS><<<G, BLOCK_SIZE>>>(dd.d_ptr, da.d_ptr, db.d_ptr, N);
    else if (strcmp(op_name, "XOR") == 0)
        kern_xor<BITS><<<G, BLOCK_SIZE>>>(dd.d_ptr, da.d_ptr, db.d_ptr, N);
    else if (strcmp(op_name, "NOT") == 0)
        kern_not<BITS><<<G, BLOCK_SIZE>>>(dd.d_ptr, da.d_ptr, N);
    else if (strcmp(op_name, "ADD_raw") == 0)
        kern_add<BITS><<<G, BLOCK_SIZE>>>(dd.d_ptr, da.d_ptr, db.d_ptr, N);
    else if (strcmp(op_name, "SHL") == 0)
        kern_shl<BITS><<<G, BLOCK_SIZE>>>(dd.d_ptr, da.d_ptr, 1, N);

    float ms = t.end_ms();
    auto wall1 = std::chrono::high_resolution_clock::now();
    float wms = std::chrono::duration<float, std::milli>(wall1 - wall0).count();
    g_bench.push_back({op_name, BITS, N, ms, wms, true});
}

// Bench ADD+carry: measures full pipeline and verifies correctness
template<int BITS>
void bench_add_carry(int N) {
    using word_t = typename WordTraits<BITS>::word_t;
    int G = grid_size(N);

    // Use patterns that generate carries
    std::vector<word_t> va(N), vb(N);
    for (int i = 0; i < N; ++i) {
        va[i] = ~word_t(0) - static_cast<word_t>(i % 3);
        vb[i] = static_cast<word_t>(i % 5 + 1);
    }
    auto expected = ref_add<word_t>(va, vb);

    TestBuf<BITS> da(N), db(N), dd(N);
    da.from_host(va); db.from_host(vb);
    CUDA_CHECK(cudaDeviceSynchronize());

    AddCarryBufs<BITS> bufs;

    // Warm up the full carry path
    kern_add<BITS><<<G, BLOCK_SIZE>>>(dd.d_ptr, da.d_ptr, db.d_ptr, N);
    if (N > 1) gpu_add_carry_propagate<BITS>(dd.d_ptr, da.d_ptr, N, bufs);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Re-upload input (carry modified dd)
    da.from_host(va); db.from_host(vb);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed run
    GpuTimer t;
    auto wall0 = std::chrono::high_resolution_clock::now();
    t.begin();
    kern_add<BITS><<<G, BLOCK_SIZE>>>(dd.d_ptr, da.d_ptr, db.d_ptr, N);
    if (N > 1) gpu_add_carry_propagate<BITS>(dd.d_ptr, da.d_ptr, N, bufs);
    float ms = t.end_ms();
    auto wall1 = std::chrono::high_resolution_clock::now();
    float wms = std::chrono::duration<float, std::milli>(wall1 - wall0).count();

    // Verify
    auto r = dd.to_host();
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        if (r[i] != expected[i]) { ok = false; break; }
    }
    if (!ok) {
        ++g_fail;
        fprintf(stderr, "  FAIL: ADD+carry correctness at N=%d\n", N);
    } else {
        ++g_pass;
    }
    g_bench.push_back({"ADD+carry", BITS, N, ms, wms, ok});
    bufs.free();
}

// ── Print timing table ──────────────────────────────────────────────

void print_bench_table() {
    printf("\n%-10s %4s %10s %10s %10s %s\n",
           "Op", "Bits", "N", "GPU(ms)", "Wall(ms)", "Status");
    printf("--------------------------------------------------------------\n");
    for (const auto& r : g_bench) {
        printf("%-10s %4d %10d %10.4f %10.4f %s\n",
               r.op, r.bits, r.N, r.gpu_ms, r.wall_ms,
               r.correct ? "OK" : "FAIL");
    }
}

// Check for timing anomalies:
// 1. wall_ms >> gpu_ms for ADD+carry means host-side overhead is large
// 2. ADD+carry should scale sub-linearly vs N (parallel prefix)
// 3. Simple ops (AND/OR/XOR/NOT) should have similar timing at same N
void check_anomalies() {
    printf("\n--- Anomaly checks ---\n");
    bool any_anomaly = false;

    for (const auto& r : g_bench) {
        // Check: wall >> gpu means host overhead dominates
        if (r.gpu_ms > 0.001f && r.wall_ms > r.gpu_ms * 10.0f) {
            printf("  WARNING: %s<%d> N=%d: wall=%.4fms >> gpu=%.4fms (%.1fx host overhead)\n",
                   r.op, r.bits, r.N, r.wall_ms, r.gpu_ms, r.wall_ms / r.gpu_ms);
            any_anomaly = true;
        }
    }

    // Check: ADD+carry vs AND at same N. ADD+carry should be no more than ~50x of AND
    // (it does 3 kernels + host memcpy, but at large N the kernels dominate)
    for (const auto& rc : g_bench) {
        if (strcmp(rc.op, "ADD+carry") != 0) continue;
        for (const auto& ra : g_bench) {
            if (strcmp(ra.op, "AND") != 0 || ra.bits != rc.bits || ra.N != rc.N) continue;
            float ratio = (ra.gpu_ms > 0.001f) ? rc.gpu_ms / ra.gpu_ms : 0;
            if (ratio > 50.0f) {
                printf("  WARNING: ADD+carry<%d> N=%d is %.1fx slower than AND (gpu: %.4f vs %.4f ms)\n",
                       rc.bits, rc.N, ratio, rc.gpu_ms, ra.gpu_ms);
                any_anomaly = true;
            }
        }
    }

    // Check: carry wall_ms at largest N should not dwarf carry gpu_ms
    // This would indicate the host-side block scan is a bottleneck
    for (const auto& r : g_bench) {
        if (strcmp(r.op, "ADD+carry") != 0) continue;
        float overhead = r.wall_ms - r.gpu_ms;
        if (r.N >= 100000 && overhead > r.gpu_ms * 2.0f) {
            printf("  WARNING: ADD+carry<%d> N=%d host overhead %.4fms >> gpu %.4fms\n",
                   r.bits, r.N, overhead, r.gpu_ms);
            any_anomaly = true;
        }
    }

    if (!any_anomaly) {
        printf("  No anomalies detected.\n");
    }
}

// ── Main ────────────────────────────────────────────────────────────

int main() {
    // Warm up GPU
    {
        void* tmp;
        cudaMalloc(&tmp, 1024);
        cudaFree(tmp);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    printf("=== Correctness Tests (32-bit) ===\n");
    float ms;
    ms = test_kern_not<32>();     printf("  kern_not<32>:       %.4f ms\n", ms);
    ms = test_kern_logic<32>();   printf("  kern_logic<32>:     %.4f ms\n", ms);
    ms = test_kern_add<32>();     printf("  kern_add<32>:       %.4f ms\n", ms);
    ms = test_kern_shift<32>();   printf("  kern_shift<32>:     %.4f ms\n", ms);
    ms = test_kern_popcount<32>();printf("  kern_popcount<32>:  %.4f ms\n", ms);
    ms = test_carry_single_word<32>(); printf("  carry_single<32>:   %.4f ms\n", ms);
    ms = test_carry_basic<32>();  printf("  carry_basic<32>:    %.4f ms\n", ms);
    ms = test_carry_chain<32>();  printf("  carry_chain<32>:    %.4f ms\n", ms);
    ms = test_carry_no_carry<32>();printf("  carry_no_carry<32>: %.4f ms\n", ms);
    ms = test_carry_all_ones<32>();printf("  carry_all_ones<32>: %.4f ms\n", ms);
    ms = test_carry_multiblock<32>();printf("  carry_multiblk<32>: %.4f ms\n", ms);
    ms = test_carry_multiblock_mixed<32>();printf("  carry_mixed<32>:    %.4f ms\n", ms);

    printf("\n=== Correctness Tests (64-bit) ===\n");
    ms = test_kern_not<64>();     printf("  kern_not<64>:       %.4f ms\n", ms);
    ms = test_kern_logic<64>();   printf("  kern_logic<64>:     %.4f ms\n", ms);
    ms = test_kern_add<64>();     printf("  kern_add<64>:       %.4f ms\n", ms);
    ms = test_kern_shift<64>();   printf("  kern_shift<64>:     %.4f ms\n", ms);
    ms = test_kern_popcount<64>();printf("  kern_popcount<64>:  %.4f ms\n", ms);
    ms = test_carry_single_word<64>(); printf("  carry_single<64>:   %.4f ms\n", ms);
    ms = test_carry_basic<64>();  printf("  carry_basic<64>:    %.4f ms\n", ms);
    ms = test_carry_chain<64>();  printf("  carry_chain<64>:    %.4f ms\n", ms);
    ms = test_carry_no_carry<64>();printf("  carry_no_carry<64>: %.4f ms\n", ms);
    ms = test_carry_all_ones<64>();printf("  carry_all_ones<64>: %.4f ms\n", ms);
    ms = test_carry_multiblock<64>();printf("  carry_multiblk<64>: %.4f ms\n", ms);
    ms = test_carry_multiblock_mixed<64>();printf("  carry_mixed<64>:    %.4f ms\n", ms);

    // Scaling benchmarks at various sizes
    printf("\n=== Scaling Benchmarks ===\n");
    int sizes[] = {1, 256, 512, 1024, 10000, 100000, 1000000};
    const char* ops[] = {"AND", "OR", "XOR", "NOT", "ADD_raw", "SHL", "ADD+carry"};

    for (int bits : {32, 64}) {
        for (int N : sizes) {
            for (const char* op : ops) {
                if (N == 1 && strcmp(op, "ADD+carry") == 0) continue; // N=1 has no carry
                if (strcmp(op, "ADD+carry") == 0) {
                    if (bits == 32) bench_add_carry<32>(N);
                    else            bench_add_carry<64>(N);
                } else {
                    if (bits == 32) bench_simple_op<32>(op, N);
                    else            bench_simple_op<64>(op, N);
                }
            }
        }
    }

    print_bench_table();
    check_anomalies();

    printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
