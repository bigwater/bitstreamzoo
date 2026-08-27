#pragma once
// Utility functions for converting between hex strings, byte arrays,
// and uint64_t word arrays — replacing GMP for I/O operations.

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace bsdata {

// Convert a hex string (with optional "0x" prefix) to a little-endian
// uint64_t word array. Returns exactly n_words elements (zero-padded).
inline std::vector<uint64_t> hex_to_words(const std::string& hex, size_t n_words) {
    std::vector<uint64_t> words(n_words, 0);

    // Strip "0x" or "0X" prefix
    size_t start = 0;
    bool negative = false;
    if (hex.size() >= 2 && hex[0] == '0' && (hex[1] == 'x' || hex[1] == 'X'))
        start = 2;
    if (hex.size() > start && hex[start] == '-') {
        negative = true;
        ++start;
    }

    const char* h = hex.c_str() + start;
    size_t hlen = hex.size() - start;

    // Parse from LSB: hex string is MSB-first, so we read from the RIGHT
    for (size_t wi = 0; wi < n_words && hlen > 0; ++wi) {
        uint64_t word = 0;
        // Take up to 16 hex chars from the right
        size_t chars = (hlen >= 16) ? 16 : hlen;
        size_t pos = hlen - chars;
        for (size_t ci = 0; ci < chars; ++ci) {
            char c = h[pos + ci];
            uint64_t nib;
            if (c >= '0' && c <= '9') nib = c - '0';
            else if (c >= 'a' && c <= 'f') nib = c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') nib = c - 'A' + 10;
            else nib = 0;
            word = (word << 4) | nib;
        }
        words[wi] = word;
        hlen -= chars;
    }

    // Handle negative: two's complement (for -1 = all-ones)
    if (negative) {
        // negate: ~words + 1
        uint64_t carry = 1;
        for (size_t i = 0; i < n_words; ++i) {
            uint64_t w = ~words[i];
            uint64_t sum = w + carry;
            carry = (sum < w) ? 1 : 0;
            words[i] = sum;
        }
    }

    return words;
}

// Convert a little-endian uint64_t word array to a hex string "0x...".
// The output omits leading zeros (except "0x0" for zero).
inline std::string words_to_hex(const uint64_t* data, size_t n_words) {
    if (n_words == 0) return "0x0";

    // Find the highest non-zero word
    size_t top = n_words;
    while (top > 0 && data[top - 1] == 0) --top;
    if (top == 0) return "0x0";

    // Build hex string from MSB to LSB
    std::string result = "0x";

    // Top word: no leading zeros
    char buf[17];
    snprintf(buf, sizeof(buf), "%lx", data[top - 1]);
    result += buf;

    // Remaining words: zero-padded to 16 chars
    for (size_t i = top - 1; i > 0; --i) {
        snprintf(buf, sizeof(buf), "%016lx", data[i - 1]);
        result += buf;
    }

    return result;
}

// Load little-endian bytes from a binary buffer into a uint64_t word array.
// This is the replacement for mpz_import(val, stride, -1, 1, 0, 0, binary).
inline std::vector<uint64_t> bytes_to_words(const uint8_t* data, size_t n_bytes, size_t n_words) {
    std::vector<uint64_t> words(n_words, 0);
    // Copy as many complete words as possible
    size_t full_words = n_bytes / 8;
    if (full_words > n_words) full_words = n_words;
    std::memcpy(words.data(), data, full_words * 8);
    // Handle remaining bytes (partial last word)
    size_t remaining = n_bytes - full_words * 8;
    if (remaining > 0 && full_words < n_words) {
        uint64_t last = 0;
        std::memcpy(&last, data + full_words * 8, remaining);
        words[full_words] = last;
    }
    return words;
}

// Mask a word array to bitlength bits (zero out excess bits in top word).
inline void mask_words(uint64_t* data, size_t n_words, int64_t bitlength) {
    if (bitlength <= 0 || n_words == 0) return;
    size_t rem = bitlength % 64;
    if (rem != 0)
        data[n_words - 1] &= (uint64_t(1) << rem) - 1;
}

// Convert a JSON integer (small value) to a word array.
inline std::vector<uint64_t> int_to_words(int64_t val, size_t n_words) {
    std::vector<uint64_t> words(n_words, 0);
    if (val >= 0) {
        words[0] = static_cast<uint64_t>(val);
    } else {
        // Negative: fill with all-ones, set low word
        for (size_t i = 0; i < n_words; ++i)
            words[i] = ~uint64_t(0);
        // Actually for -1, all words should be 0xFFFFFFFFFFFFFFFF
        // For other negatives: two's complement
        words[0] = static_cast<uint64_t>(val);  // sign-extends in 2's complement
        for (size_t i = 1; i < n_words; ++i)
            words[i] = ~uint64_t(0);  // sign extension
    }
    return words;
}

} // namespace bsdata
