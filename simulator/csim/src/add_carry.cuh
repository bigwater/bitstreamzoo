#pragma once

#include "common.cuh"
#include <cub/device/device_scan.cuh>

// ── (gen, prop) pair for carry propagation ──────────────────────────

template<typename word_t>
struct GenProp {
    word_t gen;
    word_t prop;
};

// Associative combine operator for (gen, prop) pairs:
//   (g_hi, p_hi) o (g_lo, p_lo) = (g_hi | (p_hi & g_lo), p_hi & p_lo)
template<typename word_t>
struct GenPropCombine {
    __host__ __device__ __forceinline__
    GenProp<word_t> operator()(const GenProp<word_t>& lo,
                               const GenProp<word_t>& hi) const {
        return {hi.gen | (hi.prop & lo.gen), hi.prop & lo.prop};
    }
};

// ── Temporary buffers for parallel carry propagation ────────────────

template<int BITS>
struct AddCarryBufs {
    using word_t = typename WordTraits<BITS>::word_t;
    using gp_t   = GenProp<word_t>;

    gp_t*  gp_in  = nullptr;   // N elements (input to scan)
    gp_t*  gp_out = nullptr;   // N elements (scanned output)
    void*  cub_tmp = nullptr;  // CUB temporary storage
    size_t cub_tmp_bytes = 0;
    int    capacity = 0;

    void ensure(int N) {
        if (N <= capacity) return;
        free();
        CUDA_CHECK(cudaMalloc(&gp_in,  sizeof(gp_t) * N));
        CUDA_CHECK(cudaMalloc(&gp_out, sizeof(gp_t) * N));
        capacity = N;

        // Query CUB temp storage size
        GenPropCombine<word_t> op;
        cub_tmp_bytes = 0;
        cub::DeviceScan::InclusiveScan(nullptr, cub_tmp_bytes,
                                       gp_in, gp_out, op, N);
        CUDA_CHECK(cudaMalloc(&cub_tmp, cub_tmp_bytes));
    }

    void free() {
        if (gp_in)  { CUDA_CHECK(cudaFree(gp_in));  gp_in = nullptr; }
        if (gp_out) { CUDA_CHECK(cudaFree(gp_out)); gp_out = nullptr; }
        if (cub_tmp){ CUDA_CHECK(cudaFree(cub_tmp));cub_tmp = nullptr; }
        capacity = 0;
        cub_tmp_bytes = 0;
    }
};

// ── Kernel: compute per-element (gen, prop) from sum and a ──────────
//
// gen[i]  = (sum[i] < a[i])       — word i overflowed
// prop[i] = (sum[i] == all-ones)  — word i propagates incoming carry

template<int BITS>
__global__ void kern_gen_prop(
    GenProp<typename WordTraits<BITS>::word_t>* gp,
    const typename WordTraits<BITS>::word_t* sum,
    const typename WordTraits<BITS>::word_t* a,
    int N)
{
    using word_t = typename WordTraits<BITS>::word_t;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    word_t s = sum[i];
    gp[i] = {
        (s < a[i]) ? static_cast<word_t>(1) : static_cast<word_t>(0),
        (s == ~static_cast<word_t>(0)) ? static_cast<word_t>(1) : static_cast<word_t>(0)
    };
}

// ── Kernel: apply carry-in to each word ─────────────────────────────
//
// After inclusive scan, scanned[i-1].gen gives the carry into word i.

template<int BITS>
__global__ void kern_carry_apply(
    typename WordTraits<BITS>::word_t* sum,
    const GenProp<typename WordTraits<BITS>::word_t>* scanned,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    // carry_in[0] = 0; carry_in[i] = scanned[i-1].gen for i > 0
    if (i > 0 && scanned[i - 1].gen) {
        sum[i] += 1;
    }
}

// ── Host orchestrator: GPU-parallel carry propagation ───────────────
//
// Given sum[i] = a[i] + b[i] (raw per-word, no carry), propagate
// carries across all N words using CUB inclusive prefix scan on
// (generate, propagate) pairs.
//
// Both the interpreter and codegen backends call this function.

template<int BITS>
void gpu_add_carry_propagate(
    typename WordTraits<BITS>::word_t* sum,
    const typename WordTraits<BITS>::word_t* a,
    int N,
    AddCarryBufs<BITS>& bufs)
{
    using word_t = typename WordTraits<BITS>::word_t;

    bufs.ensure(N);
    int G = grid_size(N);

    // Step 1: compute per-element (gen, prop)
    kern_gen_prop<BITS><<<G, BLOCK_SIZE>>>(bufs.gp_in, sum, a, N);
    CUDA_CHECK(cudaGetLastError());

    // Step 2: inclusive prefix scan with CUB (entirely on GPU)
    GenPropCombine<word_t> op;
    CUDA_CHECK(cub::DeviceScan::InclusiveScan(
        bufs.cub_tmp, bufs.cub_tmp_bytes,
        bufs.gp_in, bufs.gp_out, op, N));

    // Step 3: apply carry-in to each word
    kern_carry_apply<BITS><<<G, BLOCK_SIZE>>>(sum, bufs.gp_out, N);
    CUDA_CHECK(cudaGetLastError());
}
