#pragma once

#include "common.cuh"

// Reusable CUDA event-based timer for measuring GPU kernel time.
struct GpuTimer {
    cudaEvent_t start, stop;

    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }

    void begin() { cudaEventRecord(start); }

    // Records stop event, synchronizes, returns elapsed time in ms.
    float end_ms() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }

    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
};
