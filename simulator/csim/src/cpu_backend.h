#pragma once
// CPU data plane: wraps bitstream_ops functions behind the backend interface.
// Stream uses unique_ptr for ownership — move-only, no accidental copies.

#include "bitstream_ops.h"
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

namespace bs {

class CpuBackend {
public:
    // Stream buffer with unique ownership. Move-only — compiler rejects copies.
    struct Stream {
        using Ptr = std::unique_ptr<uint64_t[], decltype(&bs::free_stream)>;
        Ptr buf{nullptr, bs::free_stream};
        size_t n_words = 0;

        Stream() = default;
        Stream(uint64_t* p, size_t nw) : buf(p, bs::free_stream), n_words(nw) {}

        // Move OK, copy forbidden (compiler-enforced)
        Stream(Stream&&) = default;
        Stream& operator=(Stream&&) = default;

        uint64_t* data() { return buf.get(); }
        const uint64_t* data() const { return buf.get(); }
        explicit operator bool() const { return buf != nullptr; }
    };

    enum class Variant { SCALAR, SIMD, SIMD_OMP };
    enum class AddMode { RIPPLE, KOGGE_STONE };

    CpuBackend(size_t n_words, size_t bitlength,
               Variant variant = Variant::SIMD,
               AddMode add_mode = AddMode::RIPPLE)
        : n_words_((n_words + 7) & ~size_t(7)), bitlength_(bitlength)
    {
        if (variant == Variant::SIMD_OMP) {
            fn_and_ = avx512_omp::vec_and;  fn_or_  = avx512_omp::vec_or;
            fn_xor_ = avx512_omp::vec_xor;  fn_not_ = avx512_omp::vec_not;
            fn_shl_ = avx512_omp::vec_shl;  fn_shr_ = avx512_omp::vec_shr;
            fn_popcount_   = avx512_omp::vec_popcount;
            fn_is_nonzero_ = avx512_omp::vec_is_nonzero;
        } else if (variant == Variant::SIMD) {
            fn_and_ = avx512::vec_and;  fn_or_  = avx512::vec_or;
            fn_xor_ = avx512::vec_xor;  fn_not_ = avx512::vec_not;
            fn_shl_ = avx512::vec_shl;  fn_shr_ = avx512::vec_shr;
            fn_popcount_   = avx512::vec_popcount;
            fn_is_nonzero_ = avx512::vec_is_nonzero;
        } else {
            fn_and_ = novec::vec_and;  fn_or_  = novec::vec_or;
            fn_xor_ = novec::vec_xor;  fn_not_ = novec::vec_not;
            fn_shl_ = novec::vec_shl;  fn_shr_ = novec::vec_shr;
            fn_popcount_   = novec::vec_popcount;
            fn_is_nonzero_ = novec::vec_is_nonzero;
        }
        fn_add_ = (add_mode == AddMode::RIPPLE)      ? vec_add_reference
                 : (variant == Variant::SIMD_OMP)    ? avx512_omp::vec_add
                 : (variant == Variant::SIMD)        ? avx512::vec_add
                                                     : novec::vec_add;
    }

    size_t n_words() const { return n_words_; }
    size_t bitlength() const { return bitlength_; }

    // Backend-stat accessors. The CPU backend has no kernels and no
    // explicit malloc tracking, so these always return 0. The CUDA
    // backend overrides them with real counts. Both are exposed so
    // the interpreter can copy them into Result uniformly.
    int n_kernel_launches() const { return 0; }
    int n_mallocs() const { return 0; }

    // ── Buffer management ──
    Stream alloc_stream() {
        return Stream(bs::alloc_stream(n_words_), n_words_);
    }
    void free_stream(Stream& s) {
        s.buf.reset();
        s.n_words = 0;
    }
    void copy_stream(Stream& dst, const Stream& src) {
        std::memcpy(dst.data(), src.data(), n_words_ * 8);
    }
    void zero_stream(Stream& dst) {
        std::memset(dst.data(), 0, n_words_ * 8);
    }
    void ones_stream(Stream& dst) {
        std::memset(dst.data(), 0xFF, n_words_ * 8);
        // Padding bits past bitlength are left dirty on purpose.  The
        // only operations that read across the whole word array
        // (popcount/is_nonzero) bound themselves to the logical
        // bitlength, so padding is never observed.
    }

    // ── Bitwise operations ──
    void op_not(Stream& dst, const Stream& src) { fn_not_(dst.data(), src.data(), n_words_); }
    void op_and(Stream& dst, const Stream& a, const Stream& b) { fn_and_(dst.data(), a.data(), b.data(), n_words_); }
    void op_or (Stream& dst, const Stream& a, const Stream& b) { fn_or_ (dst.data(), a.data(), b.data(), n_words_); }
    void op_xor(Stream& dst, const Stream& a, const Stream& b) { fn_xor_(dst.data(), a.data(), b.data(), n_words_); }
    void op_add(Stream& dst, const Stream& a, const Stream& b) { fn_add_(dst.data(), a.data(), b.data(), n_words_); }
    void op_shl(Stream& dst, const Stream& src, size_t amt) { fn_shl_(dst.data(), src.data(), amt, n_words_); }
    void op_shr(Stream& dst, const Stream& src, size_t amt) { fn_shr_(dst.data(), src.data(), amt, n_words_); }

    // ── Reductions (bounded to the logical bitlength) ──
    // Iterate only the logical words (excludes alignment-pad words)
    // and mask the final partial word (excludes sub-word padding), so
    // the result is correct regardless of dirty padding.
    int64_t op_popcount(const Stream& src) {
        const uint64_t* a = src.data();
        if (bitlength_ == 0) return 0;
        size_t lw = (bitlength_ + 63) / 64;
        size_t rem = bitlength_ % 64;
        if (rem == 0) return fn_popcount_(a, lw);
        int64_t c = (lw > 1) ? fn_popcount_(a, lw - 1) : 0;
        uint64_t last = a[lw - 1] & ((uint64_t(1) << rem) - 1);
        return c + __builtin_popcountll(last);
    }
    bool op_is_nonzero(const Stream& src) {
        const uint64_t* a = src.data();
        if (bitlength_ == 0) return false;
        size_t lw = (bitlength_ + 63) / 64;
        size_t rem = bitlength_ % 64;
        if (rem == 0) return fn_is_nonzero_(a, lw);
        if (lw > 1 && fn_is_nonzero_(a, lw - 1)) return true;
        return (a[lw - 1] & ((uint64_t(1) << rem) - 1)) != 0;
    }

    // ── Host I/O ──
    void load_from_words(Stream& dst, const uint64_t* src, size_t count) {
        size_t copy_n = std::min(count, n_words_);
        std::memcpy(dst.data(), src, copy_n * 8);
        if (copy_n < n_words_)
            std::memset(dst.data() + copy_n, 0, (n_words_ - copy_n) * 8);
    }
    std::vector<uint64_t> store_to_words(const Stream& src) {
        return std::vector<uint64_t>(src.data(), src.data() + n_words_);
    }

private:
    size_t n_words_;
    size_t bitlength_;
    BinOpFn fn_and_, fn_or_, fn_xor_;
    void(*fn_not_)(uint64_t*, const uint64_t*, size_t);
    ShiftFn fn_shl_, fn_shr_;
    void(*fn_add_)(uint64_t*, const uint64_t*, const uint64_t*, size_t);
    int64_t(*fn_popcount_)(const uint64_t*, size_t);
    bool(*fn_is_nonzero_)(const uint64_t*, size_t);
};

} // namespace bs
