#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

// Dense bitset for liveness analysis.
// For 1.24M variables, each LiveSet is ~155 KB (fits in L2 cache).
class LiveSet {
public:
    LiveSet() = default;
    explicit LiveSet(uint32_t n_vars)
        : bits_((n_vars + 63) / 64, 0), n_vars_(n_vars) {}

    void set(uint32_t id) { bits_[id / 64] |= (1ULL << (id % 64)); }
    void clear(uint32_t id) { bits_[id / 64] &= ~(1ULL << (id % 64)); }
    bool test(uint32_t id) const { return (bits_[id / 64] >> (id % 64)) & 1; }

    // *this |= other
    void union_with(const LiveSet& other) {
        for (size_t i = 0; i < bits_.size(); ++i)
            bits_[i] |= other.bits_[i];
    }

    // *this &= ~other
    void difference(const LiveSet& other) {
        for (size_t i = 0; i < bits_.size(); ++i)
            bits_[i] &= ~other.bits_[i];
    }

    bool equals(const LiveSet& other) const {
        return bits_ == other.bits_;
    }

    // Iterate over set bits, calling f(id) for each
    template<typename F>
    void for_each_set(F&& f) const {
        for (size_t w = 0; w < bits_.size(); ++w) {
            uint64_t word = bits_[w];
            while (word) {
                int bit = __builtin_ctzll(word);
                f(static_cast<uint32_t>(w * 64 + bit));
                word &= word - 1;
            }
        }
    }

    void clear_all() {
        std::memset(bits_.data(), 0, bits_.size() * sizeof(uint64_t));
    }

    uint32_t n_vars() const { return n_vars_; }

private:
    std::vector<uint64_t> bits_;
    uint32_t n_vars_ = 0;
};
