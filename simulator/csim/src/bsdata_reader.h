#pragma once
/**
 * Shared bsdata reader for C++ and CUDA backends.
 *
 * Reads .bsdata binary format: UTF-8 JSON header + NUL + raw binary data.
 * JSON header contains metadata and references to binary sections:
 *   {"_bin": [offset, nbytes]}         - single stream (LE packed bits)
 *   {"_array_bin": [count, offset, stride]} - array of count streams
 * Small values are stored inline as JSON integers.
 *
 * All stream data is stored as vector<uint64_t> word arrays (no GMP dependency).
 * Conversion helpers are in word_utils.h.
 */

#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>
#include <nlohmann/json.hpp>
#include "word_utils.h"

namespace bsdata {

using json = nlohmann::json;

// ── bsdata file structures ──

struct BsdataInput {
    int64_t bitlength = 0;
    std::unordered_map<std::string, int64_t> params;
    std::unordered_map<std::string, std::vector<uint64_t>> inputs;
    std::map<std::string, std::map<int, std::vector<uint64_t>>> input_arrays;
    int iterations = 1;
    std::map<std::string, std::string> feedback;
    // Broadcast array metadata: key=array name, value[i]=true if all-ones, false if all-zeros
    std::map<std::string, std::vector<bool>> broadcast_arrays;
};

/// A multi-case bsdata file contains multiple independent test cases
struct BsdataMultiCase {
    std::vector<BsdataInput> cases;
};

// ── Internal helpers ──

namespace detail {

/// Compute n_words from bitlength (64-bit words)
inline size_t n_words_from_bitlength(int64_t bitlength) {
    if (bitlength <= 0) return 1;
    return static_cast<size_t>((bitlength + 63) / 64);
}

/// Convert a JSON value (hex string or integer) to a word array
inline std::vector<uint64_t> json_to_words(const json& j, size_t n_words) {
    if (j.is_string()) {
        const std::string& s = j.get_ref<const std::string&>();
        return hex_to_words(s, n_words);
    }
    if (j.is_number_integer()) {
        if (j.is_number_unsigned())
            return int_to_words(static_cast<int64_t>(j.get<uint64_t>()), n_words);
        return int_to_words(j.get<int64_t>(), n_words);
    }
    return std::vector<uint64_t>(n_words, 0);
}

/// Decode a single value from the JSON header, resolving _bin references
inline std::vector<uint64_t> decode_value(const json& v, const uint8_t* binary,
                                           size_t binary_len, size_t n_words) {
    if (v.is_object()) {
        if (v.contains("_bin")) {
            auto& ref = v["_bin"];
            size_t offset = ref[0].get<size_t>();
            size_t nbytes = ref[1].get<size_t>();
            if (offset + nbytes <= binary_len) {
                return bytes_to_words(binary + offset, nbytes, n_words);
            }
            return std::vector<uint64_t>(n_words, 0);
        }
        // Shouldn't normally appear for scalar values in bsdata
        return std::vector<uint64_t>(n_words, 0);
    }
    return json_to_words(v, n_words);
}

/// Decode a section of inputs/expected from the JSON header
inline std::unordered_map<std::string, std::vector<uint64_t>>
decode_inputs(const json& section, const uint8_t* binary, size_t binary_len,
              size_t n_words) {
    std::unordered_map<std::string, std::vector<uint64_t>> result;
    for (auto& [k, v] : section.items()) {
        if (v.is_object() && (v.contains("_array_bin") || v.contains("_bin") ||
                              v.contains("_broadcast_pack"))) {
            if (v.contains("_bin")) {
                result[k] = decode_value(v, binary, binary_len, n_words);
            }
            // Skip _array_bin and _broadcast_pack entries (handled by decode_arrays)
        } else if (v.is_array()) {
            // Skip arrays (handled by decode_arrays)
        } else {
            result[k] = decode_value(v, binary, binary_len, n_words);
        }
    }
    return result;
}

/// Decode array inputs from the JSON header
inline std::map<std::string, std::map<int, std::vector<uint64_t>>>
decode_arrays(const json& section, const uint8_t* binary, size_t binary_len,
              size_t n_words) {
    std::map<std::string, std::map<int, std::vector<uint64_t>>> result;
    for (auto& [k, v] : section.items()) {
        if (v.is_object() && v.contains("_broadcast_pack")) {
            // Broadcast arrays: DON'T allocate full-size vectors here.
            // The interpreter uses only the boolean metadata from
            // decode_broadcast_metadata(). Allocating n_words-sized vectors
            // for each broadcast element wastes gigabytes at large bitlengths
            // and causes memory pressure that triggers corruption.
            // Skip — the interpreter handles broadcast via shared buffers.
        } else if (v.is_object() && v.contains("_array_bin")) {
            auto& ref = v["_array_bin"];
            int count = ref[0].get<int>();
            size_t offset = ref[1].get<size_t>();
            size_t stride = ref[2].get<size_t>();
            std::map<int, std::vector<uint64_t>> arr;
            for (int i = 0; i < count; ++i) {
                size_t start = offset + static_cast<size_t>(i) * stride;
                if (start + stride <= binary_len) {
                    arr[i] = bytes_to_words(binary + start, stride, n_words);
                } else {
                    arr[i] = std::vector<uint64_t>(n_words, 0);
                }
            }
            result[k] = std::move(arr);
        } else if (v.is_array()) {
            // Inline array (small values stored as JSON list)
            std::map<int, std::vector<uint64_t>> arr;
            for (size_t i = 0; i < v.size(); ++i)
                arr[static_cast<int>(i)] = json_to_words(v[i], n_words);
            result[k] = std::move(arr);
        } else if (v.is_object() && !v.contains("_bin")) {
            // Dict-style array {index: value}
            std::map<int, std::vector<uint64_t>> arr;
            for (auto& [idx, val] : v.items())
                arr[std::stoi(idx)] = decode_value(val, binary, binary_len, n_words);
            result[k] = std::move(arr);
        }
    }
    return result;
}

/// Extract broadcast metadata from a JSON section (parallel to decode_arrays)
inline std::map<std::string, std::vector<bool>>
decode_broadcast_metadata(const json& section, const uint8_t* binary, size_t binary_len) {
    std::map<std::string, std::vector<bool>> result;
    for (auto& [k, v] : section.items()) {
        if (v.is_object() && v.contains("_broadcast_pack")) {
            auto& ref = v["_broadcast_pack"];
            int count = ref[0].get<int>();
            size_t offset = ref[1].get<size_t>();
            std::vector<bool> bvec(count);
            for (int i = 0; i < count; ++i) {
                size_t byte_idx = offset + (i / 8);
                int bit_idx = i % 8;
                bvec[i] = (byte_idx < binary_len) &&
                           ((binary[byte_idx] >> bit_idx) & 1);
            }
            result[k] = std::move(bvec);
        }
    }
    return result;
}

/// Parse a single case from a JSON object
inline BsdataInput parse_case(const json& layer, const uint8_t* binary,
                               size_t binary_len) {
    BsdataInput input;
    input.bitlength = layer.value("bitlength", layer.value("n_vectors", int64_t(0)));
    size_t n_words = n_words_from_bitlength(input.bitlength);

    if (layer.contains("params")) {
        for (auto& [k, v] : layer["params"].items())
            input.params[k] = v.get<int64_t>();
    }
    if (layer.contains("inputs")) {
        input.inputs = decode_inputs(layer["inputs"], binary, binary_len, n_words);
        // Also check for arrays in the inputs section
        auto arrays = decode_arrays(layer["inputs"], binary, binary_len, n_words);
        for (auto& [k, v] : arrays)
            input.input_arrays[k] = std::move(v);
        auto bcast = decode_broadcast_metadata(layer["inputs"], binary, binary_len);
        for (auto& [k, v] : bcast)
            input.broadcast_arrays[k] = std::move(v);
    }
    if (layer.contains("input_arrays")) {
        auto arrays = decode_arrays(layer["input_arrays"], binary, binary_len, n_words);
        for (auto& [k, v] : arrays)
            input.input_arrays[k] = std::move(v);
        auto bcast = decode_broadcast_metadata(layer["input_arrays"], binary, binary_len);
        for (auto& [k, v] : bcast)
            input.broadcast_arrays[k] = std::move(v);
    }
    input.iterations = layer.value("iterations", 1);
    if (layer.contains("feedback")) {
        for (auto& [out_name, in_name] : layer["feedback"].items())
            input.feedback[out_name] = in_name.get<std::string>();
    }
    return input;
}

} // namespace detail

// ── Public API ──

/// Read a .bsdata file and return a BsdataInput with inputs/params/input_arrays.
/// For multi-layer files, returns the first layer (use read_bsdata_multi for all).
inline BsdataInput read_bsdata(const std::string& path) {
    std::ifstream fs(path, std::ios::binary);
    if (!fs) {
        throw std::runtime_error("Cannot open bsdata file: " + path);
    }

    // Read entire file
    std::vector<uint8_t> raw((std::istreambuf_iterator<char>(fs)),
                              std::istreambuf_iterator<char>());

    // Find NUL separator
    auto sep_it = std::find(raw.begin(), raw.end(), 0);
    if (sep_it == raw.end()) {
        throw std::runtime_error("Invalid bsdata file (no NUL separator): " + path);
    }
    size_t sep_pos = std::distance(raw.begin(), sep_it);

    // Parse JSON header
    std::string header_str(raw.begin(), raw.begin() + sep_pos);
    json header = json::parse(header_str);

    // Binary section starts after NUL
    const uint8_t* binary = raw.data() + sep_pos + 1;
    size_t binary_len = raw.size() - sep_pos - 1;

    // Check for multi-case key (accept both "cases" and legacy "layers")
    const char* multi_key = nullptr;
    if (header.contains("cases")) multi_key = "cases";
    else if (header.contains("layers")) multi_key = "layers";

    if (multi_key) {
        if (header[multi_key].empty()) {
            return BsdataInput{};
        }
        return detail::parse_case(header[multi_key][0], binary, binary_len);
    }

    return detail::parse_case(header, binary, binary_len);
}

/// Read a multi-case .bsdata file and return all cases.
inline BsdataMultiCase read_bsdata_multi(const std::string& path) {
    std::ifstream fs(path, std::ios::binary);
    if (!fs) {
        throw std::runtime_error("Cannot open bsdata file: " + path);
    }

    std::vector<uint8_t> raw((std::istreambuf_iterator<char>(fs)),
                              std::istreambuf_iterator<char>());

    auto sep_it = std::find(raw.begin(), raw.end(), 0);
    if (sep_it == raw.end()) {
        throw std::runtime_error("Invalid bsdata file (no NUL separator): " + path);
    }
    size_t sep_pos = std::distance(raw.begin(), sep_it);

    std::string header_str(raw.begin(), raw.begin() + sep_pos);
    json header = json::parse(header_str);

    const uint8_t* binary = raw.data() + sep_pos + 1;
    size_t binary_len = raw.size() - sep_pos - 1;

    // Accept both "cases" and legacy "layers" key
    const char* multi_key = nullptr;
    if (header.contains("cases")) multi_key = "cases";
    else if (header.contains("layers")) multi_key = "layers";

    BsdataMultiCase result;
    if (multi_key) {
        for (auto& entry : header[multi_key]) {
            result.cases.push_back(
                detail::parse_case(entry, binary, binary_len));
        }
    } else {
        result.cases.push_back(
            detail::parse_case(header, binary, binary_len));
    }
    return result;
}

/// Parse JSON input from stdin (the legacy path).
/// Returns a BsdataInput populated from the JSON object.
inline BsdataInput parse_json_input(const json& input_json) {
    BsdataInput result;
    result.bitlength = input_json.value("bitlength", input_json.value("n_vectors", int64_t(0)));
    size_t n_words = detail::n_words_from_bitlength(result.bitlength);

    if (input_json.contains("inputs")) {
        for (auto& [k, v] : input_json["inputs"].items())
            result.inputs[k] = detail::json_to_words(v, n_words);
    }
    if (input_json.contains("params")) {
        for (auto& [k, v] : input_json["params"].items())
            result.params[k] = v.get<int64_t>();
    }
    if (input_json.contains("input_arrays")) {
        for (auto& [name, arr] : input_json["input_arrays"].items()) {
            std::map<int, std::vector<uint64_t>> m;
            if (arr.is_array()) {
                for (size_t i = 0; i < arr.size(); ++i)
                    m[static_cast<int>(i)] = detail::json_to_words(arr[i], n_words);
            } else if (arr.is_object()) {
                for (auto& [idx, val] : arr.items())
                    m[std::stoi(idx)] = detail::json_to_words(val, n_words);
            }
            result.input_arrays[name] = std::move(m);
        }
    }
    result.iterations = input_json.value("iterations", 1);
    if (input_json.contains("feedback")) {
        for (auto& [out_name, in_name] : input_json["feedback"].items())
            result.feedback[out_name] = in_name.get<std::string>();
    }
    return result;
}

/// Dump a BsdataInput as JSON (for --dump-input flag)
inline json dump_input(const BsdataInput& input) {
    json j;
    j["bitlength"] = input.bitlength;

    json inputs_j = json::object();
    for (const auto& [k, v] : input.inputs)
        inputs_j[k] = words_to_hex(v.data(), v.size());
    j["inputs"] = inputs_j;

    j["params"] = input.params;

    json arrays_j = json::object();
    for (const auto& [name, arr] : input.input_arrays) {
        json arr_j = json::object();
        for (const auto& [idx, val] : arr)
            arr_j[std::to_string(idx)] = words_to_hex(val.data(), val.size());
        arrays_j[name] = arr_j;
    }
    j["input_arrays"] = arrays_j;

    return j;
}

} // namespace bsdata
