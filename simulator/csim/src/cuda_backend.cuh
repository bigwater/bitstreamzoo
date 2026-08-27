#pragma once
// CUDA data plane: wraps GPU kernels behind the backend interface.
// Stream = GpuStream<BITS> (device pointer + word count + ownership flag).

#include "common.cuh"
#include "kernels.cuh"
#include "add_carry.cuh"
#include <cub/cub.cuh>
#include <cstdint>
#include <cstring>
#include <vector>

// Padding past bitlength is left dirty on purpose; only the bounded
// reductions (popcount/is_nonzero) read across the whole word array.

template<int BITS>
class CudaBackend {
    using word_t = typename WordTraits<BITS>::word_t;

public:
    using Stream = GpuStream<BITS>;

    CudaBackend(size_t n_words, size_t bitlength)
        : n_words_(static_cast<int64_t>((n_words + 7) & ~size_t(7))),
          bitlength_(static_cast<int64_t>(bitlength)) {}

    ~CudaBackend() {
        if (d_zero_buf_) cudaFree(d_zero_buf_);
        if (d_ones_buf_) cudaFree(d_ones_buf_);
        add_bufs_.free();
        nonzero_bufs_.free();
        popcount_bufs_.free();
    }

    size_t n_words() const { return static_cast<size_t>(n_words_); }
    size_t bitlength() const { return static_cast<size_t>(bitlength_); }

    // ── Buffer management ──
    Stream alloc_stream() {
        Stream s;
        s.alloc(static_cast<int>(n_words_));
        n_mallocs_++;
        return s;
    }
    void free_stream(Stream& s) { s.free(); }
    void copy_stream(Stream& dst, const Stream& src) { dst.copy_from(src); }
    void zero_stream(Stream& dst) { dst.zero(); }
    void ones_stream(Stream& dst) {
        dst.ones();
    }

    // ── Bitwise operations ──
    void op_not(Stream& dst, const Stream& src) {
        kern_not<BITS><<<grid_size(), BLOCK_SIZE>>>(dst.data, src.data, n_words_);
        n_kernel_launches_++;
    }
    void op_and(Stream& dst, const Stream& a, const Stream& b) {
        kern_and<BITS><<<grid_size(), BLOCK_SIZE>>>(dst.data, a.data, b.data, n_words_);
        n_kernel_launches_++;
    }
    void op_or(Stream& dst, const Stream& a, const Stream& b) {
        kern_or<BITS><<<grid_size(), BLOCK_SIZE>>>(dst.data, a.data, b.data, n_words_);
        n_kernel_launches_++;
    }
    void op_xor(Stream& dst, const Stream& a, const Stream& b) {
        kern_xor<BITS><<<grid_size(), BLOCK_SIZE>>>(dst.data, a.data, b.data, n_words_);
        n_kernel_launches_++;
    }
    void op_add(Stream& dst, const Stream& a, const Stream& b) {
        kern_add<BITS><<<grid_size(), BLOCK_SIZE>>>(dst.data, a.data, b.data, n_words_);
        n_kernel_launches_++;
        if (n_words_ > 1)
            gpu_add_carry_propagate<BITS>(dst.data, a.data, n_words_, add_bufs_);
    }
    void op_shl(Stream& dst, const Stream& src, size_t amt) {
        kern_shl<BITS><<<grid_size(), BLOCK_SIZE>>>(dst.data, src.data,
            static_cast<int64_t>(amt), n_words_);
        n_kernel_launches_++;
    }
    void op_shr(Stream& dst, const Stream& src, size_t amt) {
        kern_shr<BITS><<<grid_size(), BLOCK_SIZE>>>(dst.data, src.data,
            static_cast<int64_t>(amt), n_words_);
        n_kernel_launches_++;
    }

    // ── Reductions ──
    int64_t op_popcount(const Stream& src) {
        if (bitlength_ == 0) { n_reductions_++; return 0; }
        // Per-word popcount, bounded to the logical bitlength.
        if (!popcount_per_word_buf_ || popcount_buf_cap_ < n_words_) {
            if (popcount_per_word_buf_) cudaFree(popcount_per_word_buf_);
            cudaMalloc(&popcount_per_word_buf_, n_words_ * sizeof(unsigned long long));
            popcount_buf_cap_ = n_words_;
        }
        int64_t lw = (bitlength_ + BITS - 1) / BITS;
        word_t last_mask = (bitlength_ % BITS)
            ? ((word_t(1) << (bitlength_ % BITS)) - 1) : ~word_t(0);
        kern_popcount_per_word<BITS><<<grid_size(), BLOCK_SIZE>>>(
            popcount_per_word_buf_, src.data, n_words_, lw, last_mask);

        // CUB sum reduction
        if (popcount_bufs_.capacity < n_words_) {
            popcount_bufs_.free();
            cudaMalloc(&popcount_bufs_.d_result, sizeof(unsigned long long));
            size_t tmp = 0;
            cub::DeviceReduce::Sum(nullptr, tmp, popcount_per_word_buf_,
                (unsigned long long*)popcount_bufs_.d_result, static_cast<int>(n_words_));
            popcount_bufs_.tmp_bytes = tmp;
            cudaMalloc(&popcount_bufs_.d_tmp, tmp);
            popcount_bufs_.capacity = n_words_;
        }
        cub::DeviceReduce::Sum(popcount_bufs_.d_tmp, popcount_bufs_.tmp_bytes,
            popcount_per_word_buf_, (unsigned long long*)popcount_bufs_.d_result,
            static_cast<int>(n_words_));

        unsigned long long result;
        cudaMemcpy(&result, popcount_bufs_.d_result, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        n_reductions_++;
        return result;
    }

    bool op_is_nonzero(const Stream& src) {
        if (bitlength_ == 0) { n_reductions_++; return false; }
        int64_t lw  = (bitlength_ + BITS - 1) / BITS;
        int64_t rem = bitlength_ % BITS;
        int64_t full = (rem == 0) ? lw : (lw - 1);  // fully-valid words
        if (nonzero_bufs_.capacity < n_words_) {
            nonzero_bufs_.free();
            cudaMalloc(&nonzero_bufs_.d_result, sizeof(word_t));
            size_t tmp = 0;
            cub::DeviceReduce::Max(nullptr, tmp, src.data,
                (word_t*)nonzero_bufs_.d_result, static_cast<int>(n_words_));
            nonzero_bufs_.tmp_bytes = tmp;
            cudaMalloc(&nonzero_bufs_.d_tmp, tmp);
            nonzero_bufs_.capacity = n_words_;
        }
        word_t result = 0;
        if (full > 0) {
            cub::DeviceReduce::Max(nonzero_bufs_.d_tmp, nonzero_bufs_.tmp_bytes,
                src.data, (word_t*)nonzero_bufs_.d_result, static_cast<int>(full));
            cudaMemcpy(&result, nonzero_bufs_.d_result, sizeof(word_t),
                       cudaMemcpyDeviceToHost);
        }
        n_reductions_++;
        if (result != 0) return true;
        if (rem != 0) {  // masked final logical word
            word_t lastw = 0;
            cudaMemcpy(&lastw, src.data + (lw - 1), sizeof(word_t),
                       cudaMemcpyDeviceToHost);
            if (lastw & ((word_t(1) << rem) - 1)) return true;
        }
        return false;
    }

    // ── Host I/O ──
    void load_from_words(Stream& dst, const uint64_t* src, size_t count) {
        std::vector<uint64_t> host(src, src + count);
        host.resize(n_words_ * BITS / 64, 0);  // pad to aligned size
        if constexpr (BITS == 64) {
            cudaMemcpy(dst.data, host.data(), n_words_ * sizeof(word_t),
                       cudaMemcpyHostToDevice);
        } else {
            // 32-bit: split each uint64 into two uint32
            std::vector<uint32_t> words32(n_words_);
            for (int64_t i = 0; i < n_words_ / 2 && i < (int64_t)host.size(); i++) {
                words32[2*i] = static_cast<uint32_t>(host[i]);
                words32[2*i+1] = static_cast<uint32_t>(host[i] >> 32);
            }
            cudaMemcpy(dst.data, words32.data(), n_words_ * sizeof(word_t),
                       cudaMemcpyHostToDevice);
        }
    }

    std::vector<uint64_t> store_to_words(const Stream& src) {
        if constexpr (BITS == 64) {
            std::vector<uint64_t> host(n_words_);
            cudaMemcpy(host.data(), src.data, n_words_ * sizeof(word_t),
                       cudaMemcpyDeviceToHost);
            return host;
        } else {
            std::vector<uint32_t> words32(n_words_);
            cudaMemcpy(words32.data(), src.data, n_words_ * sizeof(word_t),
                       cudaMemcpyDeviceToHost);
            std::vector<uint64_t> host(n_words_ / 2);
            for (size_t i = 0; i < host.size(); i++)
                host[i] = words32[2*i] | ((uint64_t)words32[2*i+1] << 32);
            return host;
        }
    }

    // Stats
    int n_kernel_launches() const { return n_kernel_launches_; }
    int n_mallocs() const { return n_mallocs_; }
    int n_reductions() const { return n_reductions_; }

private:
    int64_t n_words_;
    int64_t bitlength_;
    int n_kernel_launches_ = 0;
    int n_mallocs_ = 0;
    int n_reductions_ = 0;

    // Carry propagation buffers
    AddCarryBufs<BITS> add_bufs_;

    // CUB reduction buffers
    struct ReduceBufs {
        void* d_tmp = nullptr;
        size_t tmp_bytes = 0;
        void* d_result = nullptr;
        int64_t capacity = 0;
        void free() {
            if (d_tmp) { cudaFree(d_tmp); d_tmp = nullptr; }
            if (d_result) { cudaFree(d_result); d_result = nullptr; }
            tmp_bytes = 0; capacity = 0;
        }
    };
    ReduceBufs nonzero_bufs_;
    ReduceBufs popcount_bufs_;
    unsigned long long* popcount_per_word_buf_ = nullptr;
    int64_t popcount_buf_cap_ = 0;

    // Constant buffers
    word_t* d_zero_buf_ = nullptr;
    word_t* d_ones_buf_ = nullptr;

    static constexpr int BLOCK_SIZE = 256;
    int grid_size() const { return static_cast<int>((n_words_ + BLOCK_SIZE - 1) / BLOCK_SIZE); }
};
