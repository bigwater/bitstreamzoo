#include "interpreter.h"
#include "cuda_backend.cuh"
#include "gpu_timer.cuh"
#include "parser.h"
#include "ast_liveness.h"
#include "bsdata_reader.h"
#include "word_utils.h"
#include <chrono>
#include <climits>
#include <cstring>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>

using json = nlohmann::json;
using hrc = std::chrono::high_resolution_clock;

static double ms_between(hrc::time_point a, hrc::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

template<int BITS>
static json run_case(const bs::Program& prog,
                     bsdata::BsdataInput& bsinput,
                     bool trace, bool reuse_mem, bool warmup,
                     double parse_ms) {
    int64_t bitlength = bsinput.bitlength;
    auto& inputs = bsinput.inputs;
    auto& params = bsinput.params;
    auto& input_arrays = bsinput.input_arrays;
    int iterations = bsinput.iterations;
    auto& feedback = bsinput.feedback;
    auto& broadcast_arrays = bsinput.broadcast_arrays;

    int64_t ops64 = count_ops_ast(prog, params);
    int static_ops = static_cast<int>(std::min(ops64, (int64_t)INT_MAX));

    // Compute n_words for BITS-wide words
    size_t n_words = 0;
    if (bitlength > 0) {
        n_words = static_cast<size_t>((bitlength + BITS - 1) / BITS);
    } else if (!inputs.empty()) {
        size_t max_words = 1;
        for (const auto& [_, v] : inputs)
            if (v.size() > max_words) max_words = v.size();
        n_words = max_words;
    }
    if (n_words == 0) n_words = 1;

    using Backend = CudaBackend<BITS>;
    using Interp = bs::Interpreter<Backend>;

    typename Interp::Result raw_result;
    double total_exec_ms = 0.0;
    double total_kernel_ms = 0.0;  // GPU-side kernel time across iterations
    int total_kernel_launches = 0;
    int total_mallocs = 0;
    auto run_inputs = inputs;

    // Optional CUDA context warmup before the timed loop. The first
    // CUDA call in a process triggers context creation and lazy
    // kernel-module loading, a fixed per-process cost (hundreds of ms)
    // that a process pays once and that is not intrinsic to
    // per-operation execution. One throwaway interp.run() on a private
    // copy of the inputs absorbs it, so the timed exec_ms below
    // reflects the actual per-operation overhead: warm allocations,
    // H2D/D2H copies, and kernel launches. The warmup result and its
    // feedback are discarded; the timed loop starts from the original
    // inputs.
    //
    // The warmup is opt-in (--warmup). It must stay off for external
    // profilers (Nsight): they attribute every kernel in the process,
    // so a warmup run would double the kernel counts and times such a
    // profile sees.
    if (warmup) {
        Backend warm_backend(n_words, static_cast<size_t>(bitlength));
        Interp warm_interp(std::move(warm_backend), reuse_mem);
        auto warm_inputs = inputs;
        warm_interp.run(prog, warm_inputs, params, input_arrays,
                        broadcast_arrays);
    }

    for (int iter = 0; iter < iterations; ++iter) {
        Backend backend(n_words, static_cast<size_t>(bitlength));
        Interp interp(std::move(backend), reuse_mem);

        // Measure host wall time and GPU active time over the SAME
        // window so the two are directly comparable. Both timers
        // wrap exactly the interp.run() call, which on the CUDA
        // backend includes input H2D copies, kernel launches, and
        // output D2H copies.
        //
        // - exec_ms (host wall, std::chrono): how long the host was
        //   blocked. For the CUDA backend this is dominated by sync
        //   waits on cudaMemcpy and by queue-depth throttling when
        //   the launch queue fills.
        //
        // - kernel_ms (GPU side, cudaEventElapsedTime): how long the
        //   GPU was actively processing work in the default stream
        //   between the start and stop event markers. cudaEventRecord
        //   queues the markers in stream order, so the elapsed time
        //   covers all kernels and memcpy operations between them,
        //   *including* GPU idle gaps that occur between kernels.
        //
        // GpuTimer.end_ms() calls cudaEventSynchronize(stop) before
        // reading the elapsed time, so we are guaranteed that the
        // GPU has reached the stop event — all queued work between
        // the markers has finished. Without that synchronize the
        // elapsed time would be undefined.
        //
        // Expected ordering: exec_ms >= kernel_ms in steady state.
        // The difference exec_ms - kernel_ms is host-side overhead
        // (cudaMemcpy host stalls, queue-depth throttling). When
        // exec_ms == kernel_ms the GPU is fully utilized; when
        // exec_ms >> kernel_ms there is significant host overhead
        // or GPU idleness.
        auto t_iter_start = hrc::now();
        GpuTimer kernel_timer;
        kernel_timer.begin();
        raw_result = interp.run(prog, run_inputs, params, input_arrays, broadcast_arrays);
        double iter_kernel_ms = kernel_timer.end_ms();
        // end_ms() blocked until cudaEventSynchronize(stop), so the
        // GPU is now drained. Capture host wall time after the sync
        // so exec_ms covers the whole interp.run including the async
        // tail; this makes exec_ms strictly >= kernel_ms.
        auto t_iter_end = hrc::now();
        double iter_exec_ms = std::chrono::duration<double, std::milli>(
            t_iter_end - t_iter_start).count();

        total_kernel_ms += iter_kernel_ms;
        total_exec_ms   += iter_exec_ms;
        total_kernel_launches += raw_result.n_kernel_launches;
        total_mallocs += raw_result.n_mallocs;

        for (const auto& [out_name, in_name] : feedback) {
            auto sit = raw_result.streams.find(out_name);
            if (sit != raw_result.streams.end())
                run_inputs[in_name] = sit->second;
        }
    }

    int dynamic_ops = raw_result.op_count * iterations;
    double avg_exec_ms = total_exec_ms;

    // Build output JSON
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

    json output;
    output["outputs"] = outputs_json;
    output["op_count"] = dynamic_ops;
    output["exec_ms"] = avg_exec_ms;
    output["static_ops"] = static_ops;
    output["dynamic_ops"] = dynamic_ops;
    output["n_words"] = n_words;
    output["n_variables"] = raw_result.n_variables;
    output["max_live"] = raw_result.max_live;

    size_t peak_bytes = static_cast<size_t>(raw_result.n_variables) * n_words * sizeof(typename WordTraits<BITS>::word_t);
    size_t reuse_bytes = static_cast<size_t>(raw_result.max_live) * n_words * sizeof(typename WordTraits<BITS>::word_t);
    output["n_kernel_launches"] = total_kernel_launches;
    output["n_mallocs"] = total_mallocs;
    output["peak_bytes"] = peak_bytes;
    output["reuse_bytes"] = reuse_bytes;

    double int32_gops = 0.0;
    if (avg_exec_ms > 0)
        int32_gops = static_cast<double>(dynamic_ops) * n_words * 2.0
                     / (avg_exec_ms * 1e-3) / 1e9;
    output["int32_gops"] = int32_gops;

    json timing;
    timing["parse_ms"] = parse_ms;
    timing["liveness_ms"] = raw_result.liveness_ms;
    timing["compile_ms"] = 0.0;
    timing["alloc_ms"] = 0.0;
    timing["exec_ms"] = avg_exec_ms;
    timing["kernel_ms"] = total_kernel_ms;
    timing["warmup_ms"] = 0.0;
    timing["runs"] = 1;
    output["timing"] = timing;

    return output;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: bsim_cuda program.bs [--trace] [--word-bits=32|64] [--reuse-mem] "
                     "[--warmup] [--input file.bsdata ...] [--dump-input]\n";
        return 1;
    }

    std::string bs_path = argv[1];
    bool trace = false;
    bool dump_input = false;
    bool reuse_mem = false;
    bool warmup = false;
    int word_bits = 64;
    std::vector<std::string> input_files;
    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--trace") {
            trace = true;
        } else if (std::string(argv[i]) == "--reuse-mem") {
            reuse_mem = true;
        } else if (std::string(argv[i]) == "--warmup") {
            warmup = true;
        } else if (std::strncmp(argv[i], "--word-bits=", 12) == 0) {
            word_bits = std::atoi(argv[i] + 12);
        } else if (std::string(argv[i]) == "--input" && i + 1 < argc) {
            input_files.push_back(argv[++i]);
        } else if (std::string(argv[i]) == "--dump-input") {
            dump_input = true;
        }
    }

    if (word_bits != 32 && word_bits != 64) {
        std::cerr << "Error: --word-bits must be 32 or 64\n";
        return 1;
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

    try {
        bool batch = (cases.size() > 1);
        json results = json::array();
        for (auto& c : cases) {
            json result;
            if (word_bits == 32)
                result = run_case<32>(prog, c.input, trace, reuse_mem, warmup, parse_ms);
            else
                result = run_case<64>(prog, c.input, trace, reuse_mem, warmup, parse_ms);
            if (batch) result["name"] = c.name;
            results.push_back(std::move(result));
        }
        if (batch)
            std::cout << results.dump() << "\n";
        else
            std::cout << results[0].dump() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
