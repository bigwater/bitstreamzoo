#pragma once
// Unified interpreter control plane — parameterized by Backend.
// Backend provides: Stream type, alloc/free, bitwise ops, I/O.
// Control plane provides: AST walk, statement dispatch, register reuse, I/O.
//
// The key design: streams are stored in a flat vector<Stream> (stream_pool_),
// indexed by integer. Variable names map to indices via var_index_.
// Rehashing var_index_ does NOT invalidate stream_pool_ references.

#include "ast.h"
#include "ast_liveness.h"
#include <chrono>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <deque>
#include <vector>

namespace bs {

class RuntimeError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

template<typename Backend>
class Interpreter {
public:
    using Stream = typename Backend::Stream;

    struct Result {
        std::unordered_map<std::string, std::vector<uint64_t>> streams;
        std::map<std::string, std::map<int, std::vector<uint64_t>>> arrays;
        std::unordered_map<std::string, int64_t> ints;
        int op_count = 0;
        int n_variables = 0;
        int max_live = 0;
        double liveness_ms = 0.0;
        // Host-side wall time of run() — for the CPU backend this is the
        // actual compute time; for the CUDA backend this is the host
        // submission time only (kernels run async). Use kernel_ms below
        // for the true GPU-side time.
        double exec_ms = 0.0;
        // GPU-side kernel time, measured by the *caller* (e.g. main.cu)
        // using cudaEvent timing around the run() call. The interpreter
        // itself cannot measure this because it doesn't know about CUDA
        // events at the template level. Defaults to exec_ms when no
        // CUDA timing is available, so JSON consumers always see a
        // non-zero value.
        double kernel_ms = 0.0;
        // Backend stats copied at the end of run() so callers can
        // query them after the backend is moved into the interpreter.
        int n_kernel_launches = 0;
        int n_mallocs = 0;
        std::unordered_map<std::string, int> op_mix;
    };

    Interpreter(Backend backend, bool reuse_mem = false)
        : backend_(std::move(backend)), reuse_mem_(reuse_mem) {}

    // unique_ptr in Stream handles cleanup automatically
    ~Interpreter() = default;

    Result run(const Program& prog,
               const std::unordered_map<std::string, std::vector<uint64_t>>& inputs,
               const std::unordered_map<std::string, int64_t>& params,
               const std::map<std::string, std::map<int, std::vector<uint64_t>>>& input_arrays,
               const std::map<std::string, std::vector<bool>>& broadcast_arrays = {});

private:
    Backend backend_;
    bool reuse_mem_;

    // Stream storage: flat pool + name→index map.
    // stream_pool_ holds all stream buffers. var_index_ maps variable names
    // to indices into stream_pool_. Rehashing var_index_ never invalidates
    // stream_pool_ references — this is the core fix for the memory corruption.
    // deque, not vector: push_back must not invalidate references into the
    // pool. exec_stmt takes a reference to the destination stream and only
    // then evaluates the RHS, whose operand lookup can allocate a new slot.
    // With a vector that allocation can reallocate the pool and leave the
    // destination reference dangling.
    std::deque<Stream> stream_pool_;
    std::unordered_map<std::string, int> var_index_;

    // Array storage (arrays are separate — not part of register reuse)
    std::map<std::string, std::map<int, Stream>> arrays_;

    // Integer environment
    std::unordered_map<std::string, int64_t> int_env_;

    // Shared constant streams
    Stream zero_stream_;
    Stream ones_stream_;
    bool consts_valid_ = false;

    // Broadcast metadata
    std::map<std::string, std::vector<bool>> broadcast_meta_;

    // Liveness (cached across runs)
    LivenessInfo liveness_;
    const LivenessInfo* active_liveness_ = nullptr;

    // Op counting
    int op_count_ = 0;
    std::unordered_map<std::string, int> op_mix_;

    static constexpr int MAX_WHILE_ITERS = 10000;

    // Core methods
    int64_t eval_int(const IntExpr& expr);
    Stream& get_var(const std::string& name);
    const Stream& get_operand(const Expr& expr);
    void exec_rhs(const Expr& expr, Stream& dst);
    void exec_stmt(const Stmt& stmt);
    void count_op(const std::string& op);
};

} // namespace bs

// Template implementation
#include "interpreter.inl"
