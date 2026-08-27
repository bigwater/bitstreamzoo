#pragma once

#include <memory>
#include <optional>
#include <set>
#include <string>
#include <variant>
#include <vector>

namespace bs {

// ── Forward declarations for recursive types ────────────────────
struct IntBinExpr;
struct PopcountExpr;
struct UnaryExpr;
struct BinExpr;
struct ShiftExpr;
struct IfStmt;
struct WhileStmt;
struct ForStmt;

// ── Integer leaf types (must be complete before IntExpr) ────────
struct IntLitExpr { int value; };
struct IntVarExpr { std::string name; };

// IntExpr variant — recursive types via shared_ptr
using IntExpr = std::variant<
    IntLitExpr,
    IntVarExpr,
    std::shared_ptr<IntBinExpr>,
    std::shared_ptr<PopcountExpr>
>;

// ── Stream leaf types (must be complete before Expr) ────────────
struct VarExpr { std::string name; };
struct ConstExpr { std::string name; };  // "ZERO" or "ONES"
struct ArrayAccessExpr { std::string name; IntExpr index; };

// Expr variant — recursive types via shared_ptr
using Expr = std::variant<
    VarExpr,
    ConstExpr,
    ArrayAccessExpr,
    std::shared_ptr<UnaryExpr>,
    std::shared_ptr<BinExpr>,
    std::shared_ptr<ShiftExpr>
>;

// ── Now define recursive expression types (use complete Expr/IntExpr) ──
struct UnaryExpr { std::string op; Expr operand; };   // "~"
struct BinExpr { std::string op; Expr left, right; };  // "&", "|", "^", "+"
struct ShiftExpr { std::string op; Expr stream; IntExpr amount; }; // "<<", ">>"
struct IntBinExpr { std::string op; IntExpr left, right; }; // "+", "-", "*"
struct PopcountExpr { Expr arg; };

// ── Statements ──────────────────────────────────────────────────
struct Assign { std::string target; Expr expr; };
struct ArrayAssign { std::string target; IntExpr index; Expr expr; };
struct LocalDecl { std::string name; std::optional<Expr> init_expr; };
struct ArrayDecl { std::string name; IntExpr size_expr; };
struct IntLocalDecl { std::string name; IntExpr init_expr; };
struct IntAssign { std::string target; IntExpr expr; };

using Stmt = std::variant<
    Assign,
    ArrayAssign,
    std::shared_ptr<IfStmt>,
    std::shared_ptr<WhileStmt>,
    std::shared_ptr<ForStmt>,
    LocalDecl,
    ArrayDecl,
    IntLocalDecl,
    IntAssign
>;

struct IfStmt { Expr cond; std::vector<Stmt> body; };
struct WhileStmt { Expr cond; std::vector<Stmt> body; };
struct ForStmt { std::string var; IntExpr lo, hi; std::vector<Stmt> body; };

// ── Program ─────────────────────────────────────────────────────
struct Program {
    std::vector<Stmt> stmts;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<std::string> params;
    std::vector<std::string> output_int_names;
    std::set<std::string> array_names;
};

} // namespace bs
