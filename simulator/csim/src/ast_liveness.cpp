#include "ast_liveness.h"
#include "liveset.h"  // for LiveSet (used in backward dataflow)
#include <algorithm>
#include <climits>
#include <set>
#include <unordered_set>

// ── Dense variable indexing (same approach as liveness.cpp) ─────────

struct VarIdx {
    std::unordered_map<std::string, uint32_t> name_to_id;
    std::vector<std::string> id_to_name;
    uint32_t n_vars = 0;

    void reserve(size_t expected) {
        name_to_id.reserve(expected);
        id_to_name.reserve(expected);
    }

    uint32_t add(const std::string& name) {
        auto [it, inserted] = name_to_id.emplace(name, n_vars);
        if (inserted) {
            id_to_name.push_back(name);
            return n_vars++;
        }
        return it->second;
    }

    uint32_t find(const std::string& name) const {
        auto it = name_to_id.find(name);
        return it != name_to_id.end() ? it->second : UINT32_MAX;
    }
};

// ── Helpers ─────────────────────────────────────────────────────────

// Is this a scalar stream variable (not an array, not a constant)?
static inline bool is_scalar_stream(const std::string& name,
                                    const std::unordered_set<std::string>& array_names) {
    return !name.empty() && name != "ZERO" && name != "ONES"
        && array_names.count(name) == 0;
}

// Extract scalar stream variable names used in an expression.
// In 3-address form, operands are leaves, so this is trivial.
static void vars_in_expr(const bs::Expr& expr,
                         const std::unordered_set<std::string>& array_names,
                         VarIdx& idx,
                         std::vector<uint32_t>& out) {
    std::visit([&](auto&& e) {
        using T = std::decay_t<decltype(e)>;
        if constexpr (std::is_same_v<T, bs::VarExpr>) {
            if (is_scalar_stream(e.name, array_names)) {
                uint32_t id = idx.find(e.name);
                if (id != UINT32_MAX) out.push_back(id);
            }
        } else if constexpr (std::is_same_v<T, bs::ConstExpr>) {
            // no stream vars
        } else if constexpr (std::is_same_v<T, bs::ArrayAccessExpr>) {
            // array element — not tracked in var_to_reg
        } else if constexpr (std::is_same_v<T, std::shared_ptr<bs::UnaryExpr>>) {
            vars_in_expr(e->operand, array_names, idx, out);
        } else if constexpr (std::is_same_v<T, std::shared_ptr<bs::BinExpr>>) {
            vars_in_expr(e->left, array_names, idx, out);
            vars_in_expr(e->right, array_names, idx, out);
        } else if constexpr (std::is_same_v<T, std::shared_ptr<bs::ShiftExpr>>) {
            vars_in_expr(e->stream, array_names, idx, out);
        }
    }, expr);
}

// Check if a PopcountExpr references a scalar stream var
static void vars_in_int_expr(const bs::IntExpr& expr,
                              const std::unordered_set<std::string>& array_names,
                              VarIdx& idx,
                              std::vector<uint32_t>& out) {
    std::visit([&](auto&& e) {
        using T = std::decay_t<decltype(e)>;
        if constexpr (std::is_same_v<T, std::shared_ptr<bs::PopcountExpr>>) {
            vars_in_expr(e->arg, array_names, idx, out);
        } else if constexpr (std::is_same_v<T, std::shared_ptr<bs::IntBinExpr>>) {
            vars_in_int_expr(e->left, array_names, idx, out);
            vars_in_int_expr(e->right, array_names, idx, out);
        }
        // IntLitExpr, IntVarExpr: no stream vars
    }, expr);
}

// ── Step 1: Collect all scalar stream variable names ────────────────

static void collect_vars(const std::vector<bs::Stmt>& stmts,
                         const std::unordered_set<std::string>& array_names,
                         VarIdx& idx) {
    for (const auto& stmt : stmts) {
        std::visit([&](auto&& s) {
            using T = std::decay_t<decltype(s)>;
            if constexpr (std::is_same_v<T, bs::Assign>) {
                if (is_scalar_stream(s.target, array_names))
                    idx.add(s.target);
                // Also collect vars from expr (for inputs used but never assigned)
                std::vector<uint32_t> dummy;
                // We add vars seen in expressions in case they are only referenced
                std::visit([&](auto&& e) {
                    using E = std::decay_t<decltype(e)>;
                    if constexpr (std::is_same_v<E, bs::VarExpr>) {
                        if (is_scalar_stream(e.name, array_names))
                            idx.add(e.name);
                    } else if constexpr (std::is_same_v<E, std::shared_ptr<bs::UnaryExpr>>) {
                        // operand
                        if (auto* v = std::get_if<bs::VarExpr>(&e->operand))
                            if (is_scalar_stream(v->name, array_names))
                                idx.add(v->name);
                    } else if constexpr (std::is_same_v<E, std::shared_ptr<bs::BinExpr>>) {
                        if (auto* v = std::get_if<bs::VarExpr>(&e->left))
                            if (is_scalar_stream(v->name, array_names))
                                idx.add(v->name);
                        if (auto* v = std::get_if<bs::VarExpr>(&e->right))
                            if (is_scalar_stream(v->name, array_names))
                                idx.add(v->name);
                    } else if constexpr (std::is_same_v<E, std::shared_ptr<bs::ShiftExpr>>) {
                        if (auto* v = std::get_if<bs::VarExpr>(&e->stream))
                            if (is_scalar_stream(v->name, array_names))
                                idx.add(v->name);
                    }
                }, s.expr);
            } else if constexpr (std::is_same_v<T, bs::LocalDecl>) {
                if (is_scalar_stream(s.name, array_names))
                    idx.add(s.name);
                if (s.init_expr.has_value()) {
                    std::visit([&](auto&& e) {
                        using E = std::decay_t<decltype(e)>;
                        if constexpr (std::is_same_v<E, bs::VarExpr>) {
                            if (is_scalar_stream(e.name, array_names))
                                idx.add(e.name);
                        }
                    }, *s.init_expr);
                }
            } else if constexpr (std::is_same_v<T, bs::ArrayAssign>) {
                // collect vars used in expr
                std::visit([&](auto&& e) {
                    using E = std::decay_t<decltype(e)>;
                    if constexpr (std::is_same_v<E, bs::VarExpr>) {
                        if (is_scalar_stream(e.name, array_names))
                            idx.add(e.name);
                    }
                }, s.expr);
            } else if constexpr (std::is_same_v<T, std::shared_ptr<bs::IfStmt>>) {
                // cond var
                if (auto* v = std::get_if<bs::VarExpr>(&s->cond))
                    if (is_scalar_stream(v->name, array_names))
                        idx.add(v->name);
                collect_vars(s->body, array_names, idx);
            } else if constexpr (std::is_same_v<T, std::shared_ptr<bs::WhileStmt>>) {
                if (auto* v = std::get_if<bs::VarExpr>(&s->cond))
                    if (is_scalar_stream(v->name, array_names))
                        idx.add(v->name);
                collect_vars(s->body, array_names, idx);
            } else if constexpr (std::is_same_v<T, std::shared_ptr<bs::ForStmt>>) {
                collect_vars(s->body, array_names, idx);
            }
            // IntLocalDecl, IntAssign, ArrayDecl: no scalar stream vars
        }, stmt);
    }
}

// ── Step 2: Forward walk for PC assignment + first_def / last_use ───

// Reusable scratch buffer to avoid per-statement heap allocation
static thread_local std::vector<uint32_t> g_uses_scratch;

static void ast_forward_walk(const std::vector<bs::Stmt>& stmts,
                             const std::unordered_set<std::string>& array_names,
                             int& pc,
                             std::vector<int>& first_def,
                             std::vector<int>& last_use,
                             VarIdx& idx) {
    for (const auto& stmt : stmts) {
        std::visit([&](auto&& s) {
            using T = std::decay_t<decltype(s)>;
            if constexpr (std::is_same_v<T, bs::Assign>) {
                // Uses (reuse scratch buffer to avoid 1.24M heap allocations)
                auto& uses = g_uses_scratch;
                uses.clear();
                vars_in_expr(s.expr, array_names, idx, uses);
                for (uint32_t id : uses) {
                    if (first_def[id] == INT_MAX) first_def[id] = 0;
                    if (pc > last_use[id]) last_use[id] = pc;
                }
                // Def
                if (is_scalar_stream(s.target, array_names)) {
                    uint32_t did = idx.find(s.target);
                    if (did != UINT32_MAX) {
                        if (first_def[did] == INT_MAX) first_def[did] = pc;
                        if (pc > last_use[did]) last_use[did] = pc;
                    }
                }
                ++pc;
            } else if constexpr (std::is_same_v<T, bs::LocalDecl>) {
                if (s.init_expr.has_value()) {
                    auto& uses = g_uses_scratch; uses.clear();
                    vars_in_expr(*s.init_expr, array_names, idx, uses);
                    for (uint32_t id : uses) {
                        if (first_def[id] == INT_MAX) first_def[id] = 0;
                        if (pc > last_use[id]) last_use[id] = pc;
                    }
                }
                if (is_scalar_stream(s.name, array_names)) {
                    uint32_t did = idx.find(s.name);
                    if (did != UINT32_MAX) {
                        if (first_def[did] == INT_MAX) first_def[did] = pc;
                        if (pc > last_use[did]) last_use[did] = pc;
                    }
                }
                ++pc;
            } else if constexpr (std::is_same_v<T, bs::ArrayAssign>) {
                auto& uses = g_uses_scratch; uses.clear();
                vars_in_expr(s.expr, array_names, idx, uses);
                for (uint32_t id : uses) {
                    if (first_def[id] == INT_MAX) first_def[id] = 0;
                    if (pc > last_use[id]) last_use[id] = pc;
                }
                ++pc;
            } else if constexpr (std::is_same_v<T, std::shared_ptr<bs::IfStmt>>) {
                if (auto* v = std::get_if<bs::VarExpr>(&s->cond)) {
                    if (is_scalar_stream(v->name, array_names)) {
                        uint32_t cid = idx.find(v->name);
                        if (cid != UINT32_MAX) {
                            if (first_def[cid] == INT_MAX) first_def[cid] = 0;
                            if (pc > last_use[cid]) last_use[cid] = pc;
                        }
                    }
                }
                ast_forward_walk(s->body, array_names, pc, first_def, last_use, idx);
            } else if constexpr (std::is_same_v<T, std::shared_ptr<bs::WhileStmt>>) {
                if (auto* v = std::get_if<bs::VarExpr>(&s->cond)) {
                    if (is_scalar_stream(v->name, array_names)) {
                        uint32_t cid = idx.find(v->name);
                        if (cid != UINT32_MAX) {
                            if (first_def[cid] == INT_MAX) first_def[cid] = 0;
                            if (pc > last_use[cid]) last_use[cid] = pc;
                        }
                    }
                }
                ast_forward_walk(s->body, array_names, pc, first_def, last_use, idx);
            } else if constexpr (std::is_same_v<T, std::shared_ptr<bs::ForStmt>>) {
                ast_forward_walk(s->body, array_names, pc, first_def, last_use, idx);
            } else if constexpr (std::is_same_v<T, bs::IntLocalDecl>) {
                // popcount arg may reference a stream var
                auto& uses = g_uses_scratch; uses.clear();
                vars_in_int_expr(s.init_expr, array_names, idx, uses);
                for (uint32_t id : uses) {
                    if (first_def[id] == INT_MAX) first_def[id] = 0;
                    if (pc > last_use[id]) last_use[id] = pc;
                }
                ++pc;
            } else if constexpr (std::is_same_v<T, bs::IntAssign>) {
                auto& uses = g_uses_scratch; uses.clear();
                vars_in_int_expr(s.expr, array_names, idx, uses);
                for (uint32_t id : uses) {
                    if (first_def[id] == INT_MAX) first_def[id] = 0;
                    if (pc > last_use[id]) last_use[id] = pc;
                }
                ++pc;
            }
            // ArrayDecl: no PC increment (no stream computation)
        }, stmt);
    }
}

// ── Step 2b: Count AST statements (for PC range) ───────────────────

static int count_ast_stmts(const std::vector<bs::Stmt>& stmts) {
    int count = 0;
    for (const auto& stmt : stmts) {
        std::visit([&](auto&& s) {
            using T = std::decay_t<decltype(s)>;
            if constexpr (std::is_same_v<T, bs::Assign> ||
                          std::is_same_v<T, bs::LocalDecl> ||
                          std::is_same_v<T, bs::ArrayAssign> ||
                          std::is_same_v<T, bs::IntLocalDecl> ||
                          std::is_same_v<T, bs::IntAssign>) {
                ++count;
            } else if constexpr (std::is_same_v<T, std::shared_ptr<bs::IfStmt>>) {
                count += count_ast_stmts(s->body);
            } else if constexpr (std::is_same_v<T, std::shared_ptr<bs::WhileStmt>>) {
                count += count_ast_stmts(s->body);
            } else if constexpr (std::is_same_v<T, std::shared_ptr<bs::ForStmt>>) {
                count += count_ast_stmts(s->body);
            }
        }, stmt);
    }
    return count;
}

// ── Step 3: Detect whether all statements are straight-line ────────

static bool has_control_flow(const std::vector<bs::Stmt>& stmts) {
    for (const auto& stmt : stmts) {
        bool cf = std::visit([](auto&& s) -> bool {
            using T = std::decay_t<decltype(s)>;
            if constexpr (std::is_same_v<T, std::shared_ptr<bs::IfStmt>> ||
                          std::is_same_v<T, std::shared_ptr<bs::WhileStmt>> ||
                          std::is_same_v<T, std::shared_ptr<bs::ForStmt>>) {
                return true;
            }
            return false;
        }, stmt);
        if (cf) return true;
    }
    return false;
}

// ── Step 3b: Backward dataflow on AST (for programs with control flow) ──

static LiveSet ast_backward_dataflow(
    const std::vector<bs::Stmt>& stmts,
    const LiveSet& live_out,
    const std::unordered_set<std::string>& array_names,
    VarIdx& idx,
    std::vector<int>& first_def,
    std::vector<int>& last_use,
    int pc_end)
{
    LiveSet live = live_out;
    int pc = pc_end;

    for (int i = static_cast<int>(stmts.size()) - 1; i >= 0; --i) {
        const auto& stmt = stmts[i];
        std::visit([&](auto&& s) {
            using T = std::decay_t<decltype(s)>;
            if constexpr (std::is_same_v<T, bs::Assign>) {
                --pc;
                // Kill dst
                if (is_scalar_stream(s.target, array_names)) {
                    uint32_t did = idx.find(s.target);
                    if (did != UINT32_MAX) live.clear(did);
                }
                // Gen uses
                auto& uses = g_uses_scratch; uses.clear();
                vars_in_expr(s.expr, array_names, idx, uses);
                for (uint32_t id : uses) live.set(id);
            } else if constexpr (std::is_same_v<T, bs::LocalDecl>) {
                --pc;
                // Kill name
                if (is_scalar_stream(s.name, array_names)) {
                    uint32_t did = idx.find(s.name);
                    if (did != UINT32_MAX) live.clear(did);
                }
                // Gen uses from init
                if (s.init_expr.has_value()) {
                    auto& uses = g_uses_scratch; uses.clear();
                    vars_in_expr(*s.init_expr, array_names, idx, uses);
                    for (uint32_t id : uses) live.set(id);
                }
            } else if constexpr (std::is_same_v<T, bs::ArrayAssign>) {
                --pc;
                // No kill (array element)
                auto& uses = g_uses_scratch; uses.clear();
                vars_in_expr(s.expr, array_names, idx, uses);
                for (uint32_t id : uses) live.set(id);
            } else if constexpr (std::is_same_v<T, bs::IntLocalDecl>) {
                --pc;
                // popcount may reference stream vars
                auto& uses = g_uses_scratch; uses.clear();
                vars_in_int_expr(s.init_expr, array_names, idx, uses);
                for (uint32_t id : uses) live.set(id);
            } else if constexpr (std::is_same_v<T, bs::IntAssign>) {
                --pc;
                auto& uses = g_uses_scratch; uses.clear();
                vars_in_int_expr(s.expr, array_names, idx, uses);
                for (uint32_t id : uses) live.set(id);
            } else if constexpr (std::is_same_v<T, std::shared_ptr<bs::IfStmt>>) {
                int n = count_ast_stmts(s->body);
                int body_start = pc - n;

                // Analyze body with current live-out
                LiveSet body_in = ast_backward_dataflow(
                    s->body, live, array_names, idx, first_def, last_use, pc);

                // live_in(if) = body_in ∪ live_out ∪ {cond}
                body_in.union_with(live);
                if (auto* v = std::get_if<bs::VarExpr>(&s->cond)) {
                    if (is_scalar_stream(v->name, array_names)) {
                        uint32_t cid = idx.find(v->name);
                        if (cid != UINT32_MAX) body_in.set(cid);
                    }
                }

                // Extend intervals
                body_in.for_each_set([&](uint32_t id) {
                    if (body_start < first_def[id]) first_def[id] = body_start;
                });
                live.for_each_set([&](uint32_t id) {
                    if (pc > last_use[id]) last_use[id] = pc;
                });

                live = body_in;
                pc = body_start;
            } else if constexpr (std::is_same_v<T, std::shared_ptr<bs::WhileStmt>>) {
                int n = count_ast_stmts(s->body);
                int body_start = pc - n;

                uint32_t cid = UINT32_MAX;
                if (auto* v = std::get_if<bs::VarExpr>(&s->cond)) {
                    if (is_scalar_stream(v->name, array_names))
                        cid = idx.find(v->name);
                }

                // Iterate to fixpoint
                LiveSet body_in(idx.n_vars);
                for (int iter = 0; iter < 100; ++iter) {
                    LiveSet body_out = live;
                    body_out.union_with(body_in);  // back edge
                    if (cid != UINT32_MAX) body_out.set(cid);

                    LiveSet new_in = ast_backward_dataflow(
                        s->body, body_out, array_names, idx,
                        first_def, last_use, pc);

                    if (new_in.equals(body_in)) break;
                    body_in = new_in;
                }

                if (cid != UINT32_MAX) body_in.set(cid);

                // Extend intervals
                body_in.for_each_set([&](uint32_t id) {
                    if (body_start < first_def[id]) first_def[id] = body_start;
                });
                LiveSet final_out = live;
                final_out.union_with(body_in);
                if (cid != UINT32_MAX) final_out.set(cid);
                final_out.for_each_set([&](uint32_t id) {
                    if (pc > last_use[id]) last_use[id] = pc;
                });

                live = body_in;
                pc = body_start;
            } else if constexpr (std::is_same_v<T, std::shared_ptr<bs::ForStmt>>) {
                int n = count_ast_stmts(s->body);
                int body_start = pc - n;

                // Treat identically to while — iterative fixpoint with back-edge.
                // The loop variable is an int, not a stream.
                LiveSet body_in(idx.n_vars);
                for (int iter = 0; iter < 100; ++iter) {
                    LiveSet body_out = live;
                    body_out.union_with(body_in);  // back edge

                    LiveSet new_in = ast_backward_dataflow(
                        s->body, body_out, array_names, idx,
                        first_def, last_use, pc);

                    if (new_in.equals(body_in)) break;
                    body_in = new_in;
                }

                // Extend intervals
                body_in.for_each_set([&](uint32_t id) {
                    if (body_start < first_def[id]) first_def[id] = body_start;
                });
                LiveSet final_out = live;
                final_out.union_with(body_in);
                final_out.for_each_set([&](uint32_t id) {
                    if (pc > last_use[id]) last_use[id] = pc;
                });

                live = body_in;
                pc = body_start;
            }
            // ArrayDecl: no PC change, no liveness effect
        }, stmt);
    }

    return live;
}

// ── Step 5: Linear scan register allocation ─────────────────────────
// Same algorithm as liveness.cpp Phase 4 (Poletto & Sarkar, PLDI 1999).

static void ast_allocate_registers(
    const VarIdx& idx,
    const std::vector<int>& first_def,
    const std::vector<int>& last_use,
    int total_stmts,
    LivenessInfo& info)
{
    struct Interval {
        uint32_t id;
        int start, end;
    };

    std::vector<Interval> intervals;
    intervals.reserve(idx.n_vars);
    for (uint32_t i = 0; i < idx.n_vars; ++i) {
        int s = first_def[i] != INT_MAX ? first_def[i] : 0;
        int e = last_use[i] >= 0 ? last_use[i] : total_stmts;
        intervals.push_back({i, s, e});
    }

    std::sort(intervals.begin(), intervals.end(),
              [](const Interval& a, const Interval& b) {
                  return a.start < b.start ||
                         (a.start == b.start && a.end < b.end);
              });

    std::set<std::pair<int, int>> active;  // (end, reg_id)
    std::set<int> free_regs;
    int next_reg = 0;
    int max_live = 0;

    for (const auto& iv : intervals) {
        while (!active.empty() && active.begin()->first < iv.start) {
            free_regs.insert(active.begin()->second);
            active.erase(active.begin());
        }

        int reg;
        if (!free_regs.empty()) {
            reg = *free_regs.begin();
            free_regs.erase(free_regs.begin());
        } else {
            reg = next_reg++;
        }

        info.var_to_reg[idx.id_to_name[iv.id]] = reg;
        active.insert({iv.end, reg});

        int cur = static_cast<int>(active.size());
        if (cur > max_live) max_live = cur;

    }

    info.max_live = max_live;
    info.n_registers = max_live;
}

// ── Main entry point ───────────────────────────────────────────────

LivenessInfo analyze_liveness_ast(
    const bs::Program& prog,
    const std::unordered_map<std::string, int64_t>& /* params */)
{
    LivenessInfo info;

    // Step 1: Collect all scalar stream variable names
    VarIdx idx;
    // Estimate: ~1 variable per statement (straight-line code)
    idx.reserve(prog.stmts.size());

    // Convert array_names to unordered_set for O(1) lookups
    // (critical for programs with 1M+ ops like regex_large)
    std::unordered_set<std::string> array_names(
        prog.array_names.begin(), prog.array_names.end());

    // Inputs first (non-array)
    for (const auto& name : prog.inputs) {
        if (array_names.count(name) == 0)
            idx.add(name);
    }
    // Outputs (non-array)
    for (const auto& name : prog.outputs) {
        if (array_names.count(name) == 0)
            idx.add(name);
    }
    // Walk AST for all other variables
    collect_vars(prog.stmts, array_names, idx);

    info.n_variables = static_cast<int>(idx.n_vars);
    if (idx.n_vars == 0) {
        info.max_live = 0;
        info.n_registers = 0;
        return info;
    }

    // Step 2: Forward walk for PC assignment + first_def/last_use
    std::vector<int> first_def(idx.n_vars, INT_MAX);
    std::vector<int> last_use(idx.n_vars, -1);
    int pc = 0;

    // Inputs defined at PC 0
    for (const auto& name : prog.inputs) {
        uint32_t id = idx.find(name);
        if (id != UINT32_MAX) first_def[id] = 0;
    }
    // Outputs live from PC 0 (their register must not be reused before first assign)
    for (const auto& name : prog.outputs) {
        uint32_t id = idx.find(name);
        if (id != UINT32_MAX) first_def[id] = 0;
    }

    ast_forward_walk(prog.stmts, array_names, pc, first_def, last_use, idx);
    int total_stmts = pc;

    // Step 3: Backward dataflow (only if control flow present)
    if (has_control_flow(prog.stmts)) {
        // Build output ID set for live_out initialization
        LiveSet live_out(idx.n_vars);
        for (const auto& name : prog.outputs) {
            uint32_t id = idx.find(name);
            if (id != UINT32_MAX) live_out.set(id);
        }
        ast_backward_dataflow(prog.stmts, live_out, array_names, idx,
                              first_def, last_use, total_stmts);
    }

    // Step 4: Output extension — outputs have last_use = total_stmts (never freed)
    for (const auto& name : prog.outputs) {
        uint32_t id = idx.find(name);
        if (id != UINT32_MAX) last_use[id] = total_stmts;
    }

    // Export last_use
    for (uint32_t i = 0; i < idx.n_vars; ++i) {
        if (last_use[i] >= 0)
            info.last_use[idx.id_to_name[i]] = last_use[i];
    }

    // Step 5: Linear scan register allocation
    ast_allocate_registers(idx, first_def, last_use, total_stmts, info);

    return info;
}

// ── Lightweight AST-based op counter (no materialization) ──────────

namespace {

class OpCounter {
public:
    OpCounter(const std::unordered_map<std::string, int64_t>& params)
        : params_(params) {}

    int64_t count(const bs::Program& prog) {
        int_env_ = params_;
        int64_t total = 0;
        for (const auto& stmt : prog.stmts)
            total += count_stmt(stmt);
        return total;
    }

private:
    std::unordered_map<std::string, int64_t> params_;
    std::unordered_map<std::string, int64_t> int_env_;

    int64_t eval_int(const bs::IntExpr& expr) {
        return std::visit([this](auto&& e) -> int64_t {
            using T = std::decay_t<decltype(e)>;
            if constexpr (std::is_same_v<T, bs::IntLitExpr>) {
                return e.value;
            } else if constexpr (std::is_same_v<T, bs::IntVarExpr>) {
                auto it = int_env_.find(e.name);
                if (it != int_env_.end()) return it->second;
                auto pit = params_.find(e.name);
                if (pit != params_.end()) return pit->second;
                return 0;
            } else if constexpr (std::is_same_v<T, std::shared_ptr<bs::IntBinExpr>>) {
                int64_t l = eval_int(e->left);
                int64_t r = eval_int(e->right);
                if (e->op == "+") return l + r;
                if (e->op == "-") return l - r;
                if (e->op == "*") return l * r;
                return 0;
            } else {
                return 0;
            }
        }, expr);
    }

    bool is_compute(const bs::Expr& expr) {
        return std::visit([](auto&& e) -> bool {
            using T = std::decay_t<decltype(e)>;
            if constexpr (std::is_same_v<T, bs::VarExpr>) return false;
            if constexpr (std::is_same_v<T, bs::ConstExpr>) return false;
            if constexpr (std::is_same_v<T, bs::ArrayAccessExpr>) return false;
            return true;
        }, expr);
    }

    int64_t count_stmt(const bs::Stmt& stmt) {
        return std::visit([this](auto&& s) -> int64_t {
            using T = std::decay_t<decltype(s)>;
            if constexpr (std::is_same_v<T, bs::LocalDecl>) {
                return (s.init_expr.has_value() && is_compute(*s.init_expr)) ? 1 : 0;
            } else if constexpr (std::is_same_v<T, bs::Assign>) {
                return is_compute(s.expr) ? 1 : 0;
            } else if constexpr (std::is_same_v<T, bs::ArrayAssign>) {
                return is_compute(s.expr) ? 1 : 0;
            } else if constexpr (std::is_same_v<T, bs::ArrayDecl>) {
                return 0;
            } else if constexpr (std::is_same_v<T, bs::IntLocalDecl>) {
                return std::holds_alternative<std::shared_ptr<bs::PopcountExpr>>(s.init_expr) ? 1 : 0;
            } else if constexpr (std::is_same_v<T, bs::IntAssign>) {
                return std::holds_alternative<std::shared_ptr<bs::PopcountExpr>>(s.expr) ? 1 : 0;
            } else if constexpr (std::is_same_v<T, std::shared_ptr<bs::ForStmt>>) {
                int64_t lo = eval_int(s->lo);
                int64_t hi = eval_int(s->hi);
                if (hi <= lo) return 0;
                auto saved = int_env_;
                int_env_[s->var] = lo;
                int64_t body_ops = 0;
                for (const auto& sub : s->body)
                    body_ops += count_stmt(sub);
                int_env_ = saved;
                return body_ops * (hi - lo);
            } else if constexpr (std::is_same_v<T, std::shared_ptr<bs::IfStmt>>) {
                int64_t body_ops = 0;
                for (const auto& sub : s->body) body_ops += count_stmt(sub);
                return body_ops;
            } else if constexpr (std::is_same_v<T, std::shared_ptr<bs::WhileStmt>>) {
                int64_t body_ops = 0;
                for (const auto& sub : s->body) body_ops += count_stmt(sub);
                return body_ops;
            } else {
                return 0;
            }
        }, stmt);
    }
};

} // namespace

int64_t count_ops_ast(const bs::Program& prog,
                      const std::unordered_map<std::string, int64_t>& params) {
    OpCounter counter(params);
    return counter.count(prog);
}
