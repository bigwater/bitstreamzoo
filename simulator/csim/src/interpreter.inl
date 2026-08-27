// interpreter.inl — Template implementation for Interpreter<Backend>.
// Modeled after pythonsim/interpreter.py: simple, sequential, correct.

#include <cstring>
#include <functional>

namespace bs {

template<typename B>
void Interpreter<B>::count_op(const std::string& op) {
    ++op_count_;
    ++op_mix_[op];
}

// ── Integer evaluation (identical to Python's _eval_int) ──

template<typename B>
int64_t Interpreter<B>::eval_int(const IntExpr& expr) {
    return std::visit([this](auto&& e) -> int64_t {
        using T = std::decay_t<decltype(e)>;
        if constexpr (std::is_same_v<T, IntLitExpr>) {
            return e.value;
        } else if constexpr (std::is_same_v<T, IntVarExpr>) {
            auto it = int_env_.find(e.name);
            if (it == int_env_.end())
                throw RuntimeError("undefined int variable: " + e.name);
            return it->second;
        } else if constexpr (std::is_same_v<T, std::shared_ptr<IntBinExpr>>) {
            int64_t l = eval_int(e->left);
            int64_t r = eval_int(e->right);
            if (e->op == "+") return l + r;
            if (e->op == "-") return l - r;
            if (e->op == "*") return l * r;
            throw RuntimeError("unknown int binop: " + e->op);
        } else if constexpr (std::is_same_v<T, std::shared_ptr<PopcountExpr>>) {
            const auto& var = std::get<VarExpr>(e->arg);
            auto& val = get_var(var.name);
            count_op("popcount");
            return backend_.op_popcount(val);
        } else {
            throw RuntimeError("unknown int expr type");
        }
    }, expr);
}

// ── Stream variable lookup ──
// Returns reference into stream_pool_ (flat vector).

template<typename B>
auto Interpreter<B>::get_var(const std::string& name) -> Stream& {
    auto it = var_index_.find(name);
    if (it != var_index_.end())
        return stream_pool_[it->second];
    // Not found — allocate a new slot (only without reuse_mem)
    int idx = static_cast<int>(stream_pool_.size());
    stream_pool_.push_back(backend_.alloc_stream());
    var_index_[name] = idx;
    return stream_pool_[idx];
}

// ── Read-only operand access ──

template<typename B>
auto Interpreter<B>::get_operand(const Expr& expr) -> const Stream& {
    return std::visit([this](auto&& e) -> const Stream& {
        using T = std::decay_t<decltype(e)>;
        if constexpr (std::is_same_v<T, VarExpr>) {
            return get_var(e.name);
        } else if constexpr (std::is_same_v<T, ConstExpr>) {
            return (e.name == "ZERO") ? zero_stream_ : ones_stream_;
        } else if constexpr (std::is_same_v<T, ArrayAccessExpr>) {
            int idx = const_cast<Interpreter*>(this)->eval_int(e.index);
            auto bit = broadcast_meta_.find(e.name);
            if (bit != broadcast_meta_.end()) {
                if (idx >= 0 && idx < (int)bit->second.size())
                    return bit->second[idx] ? ones_stream_ : zero_stream_;
                return zero_stream_;
            }
            auto ait = arrays_.find(e.name);
            if (ait == arrays_.end())
                throw RuntimeError("undefined array: " + e.name);
            auto it = ait->second.find(idx);
            return (it != ait->second.end()) ? it->second : zero_stream_;
        } else {
            throw RuntimeError("get_operand: not a primary expression");
            return zero_stream_;
        }
    }, expr);
}

// ── Execute RHS expression into destination stream ──

template<typename B>
void Interpreter<B>::exec_rhs(const Expr& expr, Stream& dst) {
    std::visit([this, &dst](auto&& e) {
        using T = std::decay_t<decltype(e)>;

        if constexpr (std::is_same_v<T, VarExpr>) {
            auto& src = get_var(e.name);
            if (&dst != &src)  // skip self-copy (register reuse alias)
                backend_.copy_stream(dst, src);

        } else if constexpr (std::is_same_v<T, ConstExpr>) {
            if (e.name == "ZERO")
                backend_.zero_stream(dst);
            else
                backend_.ones_stream(dst);

        } else if constexpr (std::is_same_v<T, ArrayAccessExpr>) {
            int idx = eval_int(e.index);
            auto ait = arrays_.find(e.name);
            if (ait != arrays_.end()) {
                auto it = ait->second.find(idx);
                if (it != ait->second.end()) {
                    backend_.copy_stream(dst, it->second);
                    return;
                }
            }
            auto bit = broadcast_meta_.find(e.name);
            if (bit != broadcast_meta_.end() && idx >= 0 && idx < (int)bit->second.size()) {
                if (bit->second[idx])
                    backend_.ones_stream(dst);
                else
                    backend_.zero_stream(dst);
                return;
            }
            backend_.zero_stream(dst);

        } else if constexpr (std::is_same_v<T, std::shared_ptr<UnaryExpr>>) {
            const Stream& src = get_operand(e->operand);
            backend_.op_not(dst, src);
            count_op(e->op);

        } else if constexpr (std::is_same_v<T, std::shared_ptr<BinExpr>>) {
            const Stream& lhs = get_operand(e->left);
            const Stream& rhs = get_operand(e->right);
            if (e->op == "&")      backend_.op_and(dst, lhs, rhs);
            else if (e->op == "|") backend_.op_or(dst, lhs, rhs);
            else if (e->op == "^") backend_.op_xor(dst, lhs, rhs);
            else if (e->op == "+") backend_.op_add(dst, lhs, rhs);
            else throw RuntimeError("unknown binop: " + e->op);
            count_op(e->op);

        } else if constexpr (std::is_same_v<T, std::shared_ptr<ShiftExpr>>) {
            const Stream& src = get_operand(e->stream);
            int64_t amt = eval_int(e->amount);
            if (e->op == "<<")
                backend_.op_shl(dst, src, static_cast<size_t>(amt));
            else
                backend_.op_shr(dst, src, static_cast<size_t>(amt));
            count_op(e->op);

        } else {
            throw RuntimeError("unknown stream expression type");
        }
    }, expr);
}

// ── Statement dispatch ──

template<typename B>
void Interpreter<B>::exec_stmt(const Stmt& stmt) {
    std::visit([this](auto&& s) {
        using T = std::decay_t<decltype(s)>;

        if constexpr (std::is_same_v<T, LocalDecl>) {
            auto& buf = get_var(s.name);
            if (s.init_expr.has_value()) {
                exec_rhs(*s.init_expr, buf);
            } else {
                backend_.zero_stream(buf);
            }

        } else if constexpr (std::is_same_v<T, ArrayDecl>) {
            int size = eval_int(s.size_expr);
            auto& arr = arrays_[s.name];
            for (int i = 0; i < size; ++i) {
                auto it = arr.find(i);
                if (it != arr.end()) {
                    backend_.zero_stream(it->second);
                } else {
                    arr[i] = backend_.alloc_stream();
                }
            }

        } else if constexpr (std::is_same_v<T, Assign>) {
            auto& buf = get_var(s.target);
            exec_rhs(s.expr, buf);

        } else if constexpr (std::is_same_v<T, ArrayAssign>) {
            int idx = eval_int(s.index);
            auto& slot = arrays_[s.target][idx];
            if (!slot) slot = backend_.alloc_stream();
            exec_rhs(s.expr, slot);

        } else if constexpr (std::is_same_v<T, std::shared_ptr<IfStmt>>) {
            const auto& var = std::get<VarExpr>(s->cond);
            auto& cond = get_var(var.name);
            if (backend_.op_is_nonzero(cond)) {
                for (const auto& sub : s->body) exec_stmt(sub);
            }

        } else if constexpr (std::is_same_v<T, std::shared_ptr<WhileStmt>>) {
            const auto& var = std::get<VarExpr>(s->cond);
            for (int iter = 0; iter < MAX_WHILE_ITERS; ++iter) {
                auto& cond = get_var(var.name);
                if (!backend_.op_is_nonzero(cond)) return;
                for (const auto& sub : s->body) exec_stmt(sub);
            }
            throw RuntimeError("while loop exceeded max iterations");

        } else if constexpr (std::is_same_v<T, std::shared_ptr<ForStmt>>) {
            int lo = eval_int(s->lo);
            int hi = eval_int(s->hi);
            for (int i = lo; i < hi; ++i) {
                int_env_[s->var] = i;
                for (const auto& sub : s->body) exec_stmt(sub);
            }

        } else if constexpr (std::is_same_v<T, IntLocalDecl>) {
            int_env_[s.name] = eval_int(s.init_expr);

        } else if constexpr (std::is_same_v<T, IntAssign>) {
            int_env_[s.target] = eval_int(s.expr);

        } else {
            throw RuntimeError("unknown statement type");
        }
    }, stmt);
}

// ── Main entry point ──

template<typename B>
auto Interpreter<B>::run(
    const Program& prog,
    const std::unordered_map<std::string, std::vector<uint64_t>>& inputs,
    const std::unordered_map<std::string, int64_t>& params,
    const std::map<std::string, std::map<int, std::vector<uint64_t>>>& input_arrays,
    const std::map<std::string, std::vector<bool>>& broadcast_arrays) -> Result
{
    using hrc = std::chrono::high_resolution_clock;

    // ── Reset state ──
    stream_pool_.clear();  // unique_ptrs auto-free their buffers
    var_index_.clear();
    arrays_.clear();       // unique_ptrs auto-free
    int_env_.clear();
    op_count_ = 0;
    op_mix_.clear();
    broadcast_meta_ = broadcast_arrays;

    // ── Constant streams ──
    if (!consts_valid_) {
        zero_stream_ = backend_.alloc_stream();  // all zeros
        ones_stream_ = backend_.alloc_stream();
        backend_.ones_stream(ones_stream_);       // all ones, masked to bitlength
        consts_valid_ = true;
    }

    // ── Liveness analysis ──
    double liveness_ms = 0.0;
    active_liveness_ = nullptr;
    if (reuse_mem_) {
        if (liveness_.n_variables > 0) {
            active_liveness_ = &liveness_;
        } else {
            auto t0 = hrc::now();
            liveness_ = analyze_liveness_ast(prog, params);
            liveness_ms = std::chrono::duration<double, std::milli>(
                hrc::now() - t0).count();
            active_liveness_ = &liveness_;
        }
        // Pre-allocate the register pool. stream_pool_ is a deque, so growth
        // never invalidates references into it and no reserve is needed.
        for (int i = 0; i < active_liveness_->n_registers; ++i)
            stream_pool_.push_back(backend_.alloc_stream());
        for (auto& [name, reg] : active_liveness_->var_to_reg)
            var_index_[name] = reg;
    }

    // ── Load params ──
    for (const auto& [k, v] : params)
        int_env_[k] = v;

    auto t_exec_start = hrc::now();

    // ── Load inputs ──
    for (const auto& name : prog.inputs) {
        auto it = inputs.find(name);
        if (it != inputs.end()) {
            auto& buf = get_var(name);
            backend_.load_from_words(buf, it->second.data(), it->second.size());
        } else {
            auto bit = broadcast_arrays.find(name);
            if (bit != broadcast_arrays.end()) {
                // Broadcast — handled via get_operand
            } else {
                auto ait = input_arrays.find(name);
                if (ait != input_arrays.end()) {
                    for (const auto& [idx, val] : ait->second) {
                        auto& slot = arrays_[name][idx];
                        if (!slot) slot = backend_.alloc_stream();
                        backend_.load_from_words(slot, val.data(), val.size());
                    }
                } else {
                    auto& buf = get_var(name);
                    backend_.zero_stream(buf);
                }
            }
        }
    }

    // ── Execute ──
    for (const auto& stmt : prog.stmts)
        exec_stmt(stmt);

    auto t_exec_end = hrc::now();

    // ── Collect outputs ──
    Result result;
    result.op_count = op_count_;
    result.op_mix = op_mix_;
    result.liveness_ms = liveness_ms;
    result.exec_ms = std::chrono::duration<double, std::milli>(
        t_exec_end - t_exec_start).count();
    // Copy backend stats into Result so callers can read them after
    // the backend has been moved into the interpreter. For the CPU
    // backend both are 0 (no kernels). For the CUDA backend they
    // reflect the per-launch counter incremented in cuda_backend.cuh.
    result.n_kernel_launches = backend_.n_kernel_launches();
    result.n_mallocs = backend_.n_mallocs();
    // Default kernel_ms to exec_ms; main.cu overrides this with a
    // cudaEvent-based measurement for the CUDA backend.
    result.kernel_ms = result.exec_ms;

    if (active_liveness_) {
        result.n_variables = active_liveness_->n_variables;
        result.max_live = active_liveness_->max_live;
    } else {
        int nv = static_cast<int>(var_index_.size());
        for (const auto& [_, arr] : arrays_)
            nv += static_cast<int>(arr.size());
        result.n_variables = nv;
        result.max_live = 0;
    }

    for (const auto& name : prog.outputs) {
        auto vi = var_index_.find(name);
        if (vi != var_index_.end())
            result.streams[name] = backend_.store_to_words(stream_pool_[vi->second]);
        auto ait = arrays_.find(name);
        if (ait != arrays_.end())
            for (auto& [idx, s] : ait->second)
                result.arrays[name][idx] = backend_.store_to_words(s);
    }
    for (const auto& name : prog.output_int_names) {
        auto it = int_env_.find(name);
        if (it != int_env_.end())
            result.ints[name] = it->second;
    }

    return result;
}

} // namespace bs
