#include "interpreter.h"
#include "cpu_backend.h"
#include "parser.h"
#include "ast_liveness.h"  // for count_ops_ast()
#include "bsdata_reader.h"
#include "word_utils.h"
#include <chrono>
#include <climits>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>

using json = nlohmann::json;
using hrc = std::chrono::high_resolution_clock;

static double ms_between(hrc::time_point a, hrc::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

static json run_case(const bs::Program& prog, bsdata::BsdataInput& bsinput,
                     bool trace,
                     const std::string& backend_name,
                     const std::string& add_mode_name,
                     double parse_ms,
                     bool reuse_mem = false) {
    int64_t bitlength = bsinput.bitlength;
    auto& inputs = bsinput.inputs;
    auto& params = bsinput.params;
    auto& input_arrays = bsinput.input_arrays;
    int iterations = bsinput.iterations;
    auto& feedback = bsinput.feedback;
    auto& broadcast_arrays = bsinput.broadcast_arrays;

    int64_t static_ops_64 = count_ops_ast(prog, params);
    int static_ops = static_cast<int>(std::min(static_ops_64, (int64_t)INT_MAX));

    size_t n_words = 0;
    if (bitlength > 0) {
        n_words = static_cast<size_t>((bitlength + 63) / 64);
    } else if (!inputs.empty()) {
        size_t max_words = 1;
        for (const auto& [_, v] : inputs)
            if (v.size() > max_words) max_words = v.size();
        n_words = max_words;
    }
    if (n_words == 0) n_words = 1;

    // ── Set up backend and interpreter ──
    auto variant = (backend_name == "simd_omp")
        ? bs::CpuBackend::Variant::SIMD_OMP
        : (backend_name == "simd" || backend_name == "gmp")
        ? bs::CpuBackend::Variant::SIMD
        : bs::CpuBackend::Variant::SCALAR;
    auto add_mode = (add_mode_name == "kogge-stone")
        ? bs::CpuBackend::AddMode::KOGGE_STONE
        : bs::CpuBackend::AddMode::RIPPLE;

    using Interp = bs::Interpreter<bs::CpuBackend>;
    Interp::Result raw_result;
    double total_exec_ms = 0.0;
    auto run_inputs = inputs;

    for (int iter = 0; iter < iterations; ++iter) {
        bs::CpuBackend backend(n_words, static_cast<size_t>(bitlength), variant, add_mode);
        Interp interp(std::move(backend), reuse_mem);
        // Wall-clock the whole interp.run() so the reported CPU time
        // covers the same window as the CUDA backend in main.cu: input
        // load, execution, and output collection. The interpreter's
        // internal result.exec_ms stops before output collection, so
        // timing the call here keeps the two backends' end-to-end
        // runtime directly comparable.
        auto t_iter_start = std::chrono::high_resolution_clock::now();
        raw_result = interp.run(prog, run_inputs, params, input_arrays, broadcast_arrays);
        auto t_iter_end = std::chrono::high_resolution_clock::now();
        total_exec_ms += std::chrono::duration<double, std::milli>(
            t_iter_end - t_iter_start).count();
        for (const auto& [out_name, in_name] : feedback) {
            auto sit = raw_result.streams.find(out_name);
            if (sit != raw_result.streams.end())
                run_inputs[in_name] = sit->second;
        }
    }

    int dynamic_ops = raw_result.op_count * iterations;
    double avg_exec_ms = total_exec_ms;

    // ── Build output JSON ──
    json outputs_json = json::object();
    for (const auto& name : prog.outputs) {
        auto sit = raw_result.streams.find(name);
        if (sit != raw_result.streams.end()) {
            std::vector<uint64_t> masked = sit->second;
            bsdata::mask_words(masked.data(), masked.size(), bitlength);
            outputs_json[name] = bsdata::words_to_hex(masked.data(), masked.size());
        }
        auto ait = raw_result.arrays.find(name);
        if (ait != raw_result.arrays.end()) {
            json arr_json = json::object();
            for (const auto& [idx, val] : ait->second) {
                std::vector<uint64_t> masked = val;
                bsdata::mask_words(masked.data(), masked.size(), bitlength);
                arr_json[std::to_string(idx)] = bsdata::words_to_hex(masked.data(), masked.size());
            }
            outputs_json[name] = arr_json;
        }
    }
    for (const auto& name : prog.output_int_names) {
        auto it = raw_result.ints.find(name);
        if (it != raw_result.ints.end())
            outputs_json[name] = it->second;
    }

    json timing;
    timing["parse_ms"] = parse_ms;
    timing["convert_ms"] = 0.0;
    timing["liveness_ms"] = raw_result.liveness_ms;
    timing["compile_ms"] = 0.0;
    timing["alloc_ms"] = 0.0;
    timing["exec_ms"] = avg_exec_ms;
    // CPU backend has no GPU kernels; kernel_ms == exec_ms by convention.
    timing["kernel_ms"] = avg_exec_ms;
    timing["warmup_ms"] = 0.0;
    timing["runs"] = 1;

    json output;
    output["outputs"] = outputs_json;
    output["op_count"] = dynamic_ops;
    output["exec_ms"] = avg_exec_ms;
    output["static_ops"] = static_ops;
    output["dynamic_ops"] = dynamic_ops;
    output["n_words"] = n_words;
    output["n_variables"] = raw_result.n_variables;
    output["max_live"] = raw_result.max_live;

    size_t peak_bytes = static_cast<size_t>(raw_result.n_variables) * n_words * 8;
    size_t reuse_bytes = static_cast<size_t>(raw_result.max_live) * n_words * 8;
    // CPU backend has no kernels and no explicit malloc tracking;
    // these come from CpuBackend::n_kernel_launches() / n_mallocs(),
    // both of which return 0. Routing through the Result struct keeps
    // the JSON schema consistent with bsim_cuda.
    output["n_kernel_launches"] = raw_result.n_kernel_launches;
    output["n_mallocs"] = raw_result.n_mallocs;
    output["peak_bytes"] = peak_bytes;
    output["reuse_bytes"] = reuse_bytes;

    double int32_gops = 0.0;
    if (avg_exec_ms > 0)
        int32_gops = static_cast<double>(dynamic_ops) * n_words * 2.0
                     / (avg_exec_ms * 1e-3) / 1e9;
    output["int32_gops"] = int32_gops;

    json op_mix_json;
    for (const auto& [op, cnt] : raw_result.op_mix)
        op_mix_json[op] = cnt * iterations;
    output["op_mix"] = op_mix_json;
    output["timing"] = timing;
    output["backend"] = backend_name;

    return output;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: bsim program.bs [--trace] [--backend simd|simd_omp|scalar] "
                     "[--add ripple|kogge-stone] [--input file.bsdata ...] [--dump-input] "
                     "[--reuse-mem]\n"
                     "  Without --input: reads JSON from stdin\n"
                     "  With --input: reads .bsdata file(s) directly\n"
                     "  Multiple --input flags run all in one process\n"
                     "  --reuse-mem: enable register reuse (liveness-based buffer sharing)\n";
        return 1;
    }

    std::string bs_path = argv[1];
    bool trace = false;
    bool dump_input = false;
    bool reuse_mem = false;
    std::string backend_name = "simd";
    std::string add_mode_name = "ripple";
    std::vector<std::string> input_files;
    for (int i = 2; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--trace") {
            trace = true;
        } else if (arg == "--backend" && i + 1 < argc) {
            backend_name = argv[++i];
        } else if (arg == "--add" && i + 1 < argc) {
            add_mode_name = argv[++i];
        } else if (arg == "--input" && i + 1 < argc) {
            input_files.push_back(argv[++i]);
        } else if (arg == "--dump-input") {
            dump_input = true;
        } else if (arg == "--reuse-mem") {
            reuse_mem = true;
        }
    }

    std::ifstream fs(bs_path);
    if (!fs) {
        std::cerr << "Error: cannot open " << bs_path << "\n";
        return 1;
    }
    std::stringstream ss;
    ss << fs.rdbuf();
    std::string source = ss.str();

    struct CaseInfo {
        bsdata::BsdataInput input;
        std::string name;
    };
    std::vector<CaseInfo> cases;

    if (!input_files.empty()) {
        for (const auto& file : input_files) {
            try {
                auto multi = bsdata::read_bsdata_multi(file);
                if (multi.cases.size() == 1) {
                    cases.push_back({std::move(multi.cases[0]), file});
                } else {
                    for (size_t i = 0; i < multi.cases.size(); ++i) {
                        cases.push_back({std::move(multi.cases[i]),
                                         file + "[" + std::to_string(i) + "]"});
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "Error reading bsdata: " << e.what() << "\n";
                return 1;
            }
        }
    } else {
        json input_json;
        try {
            std::cin >> input_json;
        } catch (const std::exception& e) {
            std::cerr << "JSON parse error: " << e.what() << "\n";
            return 1;
        }
        cases.push_back({bsdata::parse_json_input(input_json), ""});
    }

    if (dump_input) {
        if (cases.size() == 1) {
            std::cout << bsdata::dump_input(cases[0].input).dump(2) << "\n";
        } else {
            json arr = json::array();
            for (auto& c : cases)
                arr.push_back(bsdata::dump_input(c.input));
            std::cout << arr.dump(2) << "\n";
        }
        return 0;
    }

    auto t_parse0 = hrc::now();
    bs::Program prog;
    try {
        prog = bs::parse(source);
    } catch (const std::exception& e) {
        std::cerr << "Parse error: " << e.what() << "\n";
        return 1;
    }
    double parse_ms = ms_between(t_parse0, hrc::now());

    bool batch = (cases.size() > 1);
    json results = json::array();
    for (auto& c : cases) {
        json result = run_case(prog, c.input, trace,
                               backend_name, add_mode_name,
                               parse_ms, reuse_mem);
        if (batch) result["name"] = c.name;
        results.push_back(std::move(result));
    }

    if (batch)
        std::cout << results.dump() << "\n";
    else
        std::cout << results[0].dump() << "\n";

    return 0;
}
