#include "parser.h"
#include <stdexcept>

namespace bs {

// ── Helpers ─────────────────────────────────────────────────────

Parser::Parser(std::vector<Token> tokens) : tokens_(std::move(tokens)) {}

const Token& Parser::peek() const { return tokens_[pos_]; }

Token Parser::advance() {
    Token tok = tokens_[pos_];
    ++pos_;
    return tok;
}

Token Parser::expect(TT tt) {
    Token tok = advance();
    if (tok.type != tt) {
        throw ParseError("line " + std::to_string(tok.line) +
                          ": expected " + tt_name(tt) + ", got " +
                          tt_name(tok.type) + " ('" + tok.value + "')");
    }
    return tok;
}

bool Parser::at(TT t) const { return peek().type == t; }
bool Parser::at(TT t1, TT t2) const { auto t = peek().type; return t == t1 || t == t2; }
bool Parser::at(TT t1, TT t2, TT t3) const {
    auto t = peek().type; return t == t1 || t == t2 || t == t3;
}

// ── Top-level ───────────────────────────────────────────────────

Program Parser::parse_program() {
    Program prog;

    while (!at(TT::EOF_TOK)) {
        if (at(TT::INPUT)) {
            parse_input_decl(prog);
        } else if (at(TT::OUTPUT)) {
            // Could be "output stream ..." or "output int ..."
            size_t saved = pos_;
            advance(); // consume OUTPUT
            if (at(TT::INT)) {
                advance(); // consume INT
                std::string name = expect(TT::IDENT).value;
                int_names_.insert(name);
                output_int_names_.push_back(name);
                // output int vars need initialization
                prog.stmts.push_back(IntLocalDecl{name, IntLitExpr{0}});
            } else {
                pos_ = saved;
                parse_output_decl(prog);
            }
        } else if (at(TT::PARAM)) {
            parse_param_decl(prog);
        } else if (at(TT::STREAM)) {
            Stmt d = parse_local_or_array_decl();
            prog.stmts.push_back(std::move(d));
        } else if (at(TT::INT)) {
            prog.stmts.push_back(parse_int_local_decl());
        } else if (at(TT::IF, TT::WHILE, TT::FOR)) {
            prog.stmts.push_back(parse_control_stmt());
        } else if (at(TT::IDENT)) {
            prog.stmts.push_back(parse_assignment());
        } else {
            const Token& tok = peek();
            throw ParseError("line " + std::to_string(tok.line) +
                              ": unexpected token " + tt_name(tok.type) +
                              " ('" + tok.value + "')");
        }
    }

    prog.output_int_names = output_int_names_;
    prog.array_names = array_names_;
    return prog;
}

void Parser::parse_input_decl(Program& prog) {
    expect(TT::INPUT);
    expect(TT::STREAM);
    std::string name = expect(TT::IDENT).value;
    if (at(TT::LBRACKET)) {
        advance();
        parse_int_expr(); // size (consumed, not stored separately)
        expect(TT::RBRACKET);
        array_names_.insert(name);
    }
    prog.inputs.push_back(name);
}

void Parser::parse_output_decl(Program& prog) {
    expect(TT::OUTPUT);
    expect(TT::STREAM);
    std::string name = expect(TT::IDENT).value;
    if (at(TT::LBRACKET)) {
        advance();
        IntExpr size = parse_int_expr();
        expect(TT::RBRACKET);
        array_names_.insert(name);
        // Emit ArrayDecl so the interpreter pre-allocates the output array
        // before execution begins (avoids lazy allocation of large buffers
        // during execution, which triggers heap corruption at large bitlengths).
        prog.stmts.push_back(ArrayDecl{name, std::move(size)});
    }
    prog.outputs.push_back(name);
}

void Parser::parse_param_decl(Program& prog) {
    expect(TT::PARAM);
    expect(TT::INT);
    std::string name = expect(TT::IDENT).value;
    int_names_.insert(name);
    prog.params.push_back(name);
}

Stmt Parser::parse_local_or_array_decl() {
    expect(TT::STREAM);
    std::string name = expect(TT::IDENT).value;

    // Array declaration: stream foo[N]
    if (at(TT::LBRACKET)) {
        advance();
        IntExpr size = parse_int_expr();
        expect(TT::RBRACKET);
        array_names_.insert(name);
        return ArrayDecl{name, std::move(size)};
    }

    // Optional initializer
    std::optional<Expr> init;
    if (at(TT::ASSIGN)) {
        advance();
        init = parse_stream_expr();
    }
    return LocalDecl{name, std::move(init)};
}

Stmt Parser::parse_int_local_decl() {
    expect(TT::INT);
    std::string name = expect(TT::IDENT).value;
    int_names_.insert(name);
    expect(TT::ASSIGN);
    IntExpr init = parse_int_expr();
    return IntLocalDecl{name, std::move(init)};
}

// ── Statements ──────────────────────────────────────────────────

Stmt Parser::parse_assignment() {
    std::string name = expect(TT::IDENT).value;

    // Array element: foo[i] = expr
    if (at(TT::LBRACKET)) {
        advance();
        IntExpr idx = parse_int_expr();
        expect(TT::RBRACKET);
        expect(TT::ASSIGN);
        Expr expr = parse_stream_expr();
        return ArrayAssign{name, std::move(idx), std::move(expr)};
    }

    expect(TT::ASSIGN);
    // Int variable?
    if (int_names_.count(name)) {
        IntExpr expr = parse_int_expr();
        return IntAssign{name, std::move(expr)};
    }
    Expr expr = parse_stream_expr();
    return Assign{name, std::move(expr)};
}

Stmt Parser::parse_control_stmt() {
    if (at(TT::IF)) return parse_if();
    if (at(TT::WHILE)) return parse_while();
    return parse_for();
}

Stmt Parser::parse_if() {
    expect(TT::IF);
    expect(TT::LPAREN);
    Expr cond = parse_stream_expr();
    expect(TT::RPAREN);
    auto body = parse_block();
    return std::make_shared<IfStmt>(IfStmt{std::move(cond), std::move(body)});
}

Stmt Parser::parse_while() {
    expect(TT::WHILE);
    expect(TT::LPAREN);
    Expr cond = parse_stream_expr();
    expect(TT::RPAREN);
    auto body = parse_block();
    return std::make_shared<WhileStmt>(WhileStmt{std::move(cond), std::move(body)});
}

Stmt Parser::parse_for() {
    expect(TT::FOR);
    std::string var = expect(TT::IDENT).value;
    int_names_.insert(var);
    expect(TT::IN);
    IntExpr lo = parse_int_expr();
    expect(TT::DOTDOT);
    IntExpr hi = parse_int_expr();
    auto body = parse_block();
    return std::make_shared<ForStmt>(ForStmt{var, std::move(lo), std::move(hi), std::move(body)});
}

std::vector<Stmt> Parser::parse_block() {
    expect(TT::LBRACE);
    std::vector<Stmt> stmts;
    while (!at(TT::RBRACE)) {
        if (at(TT::STREAM)) {
            stmts.push_back(parse_local_or_array_decl());
        } else if (at(TT::INT)) {
            stmts.push_back(parse_int_local_decl());
        } else if (at(TT::IF, TT::WHILE, TT::FOR)) {
            stmts.push_back(parse_control_stmt());
        } else if (at(TT::IDENT)) {
            stmts.push_back(parse_assignment());
        } else {
            const Token& tok = peek();
            throw ParseError("line " + std::to_string(tok.line) +
                              ": unexpected token in block: " + tt_name(tok.type) +
                              " ('" + tok.value + "')");
        }
    }
    expect(TT::RBRACE);
    return stmts;
}

// ── Stream expressions (precedence climbing) ────────────────────
// Precedence (high→low): ~ (unary), << >>, &, +, ^, |

Expr Parser::parse_stream_expr() { return parse_or(); }

Expr Parser::parse_or() {
    Expr left = parse_xor();
    while (at(TT::PIPE)) {
        advance();
        Expr right = parse_xor();
        left = std::make_shared<BinExpr>(BinExpr{"|", std::move(left), std::move(right)});
    }
    return left;
}

Expr Parser::parse_xor() {
    Expr left = parse_add();
    while (at(TT::CARET)) {
        advance();
        Expr right = parse_add();
        left = std::make_shared<BinExpr>(BinExpr{"^", std::move(left), std::move(right)});
    }
    return left;
}

Expr Parser::parse_add() {
    Expr left = parse_and();
    while (at(TT::PLUS)) {
        advance();
        Expr right = parse_and();
        left = std::make_shared<BinExpr>(BinExpr{"+", std::move(left), std::move(right)});
    }
    return left;
}

Expr Parser::parse_and() {
    Expr left = parse_shift();
    while (at(TT::AMP)) {
        advance();
        Expr right = parse_shift();
        left = std::make_shared<BinExpr>(BinExpr{"&", std::move(left), std::move(right)});
    }
    return left;
}

Expr Parser::parse_shift() {
    Expr left = parse_unary();
    while (at(TT::LSHIFT, TT::RSHIFT)) {
        std::string op = advance().value;
        IntExpr amount = parse_int_expr();
        left = std::make_shared<ShiftExpr>(ShiftExpr{op, std::move(left), std::move(amount)});
    }
    return left;
}

Expr Parser::parse_unary() {
    if (at(TT::TILDE)) {
        advance();
        Expr operand = parse_unary();
        return std::make_shared<UnaryExpr>(UnaryExpr{"~", std::move(operand)});
    }
    return parse_primary();
}

Expr Parser::parse_primary() {
    if (at(TT::ZERO)) { advance(); return ConstExpr{"ZERO"}; }
    if (at(TT::ONES)) { advance(); return ConstExpr{"ONES"}; }
    if (at(TT::LPAREN)) {
        advance();
        Expr expr = parse_stream_expr();
        expect(TT::RPAREN);
        return expr;
    }
    if (at(TT::IDENT)) {
        std::string name = advance().value;
        if (at(TT::LBRACKET)) {
            advance();
            IntExpr idx = parse_int_expr();
            expect(TT::RBRACKET);
            return ArrayAccessExpr{name, std::move(idx)};
        }
        return VarExpr{name};
    }
    const Token& tok = peek();
    throw ParseError("line " + std::to_string(tok.line) +
                      ": expected stream expression, got " + tt_name(tok.type) +
                      " ('" + tok.value + "')");
}

// ── Integer expressions ─────────────────────────────────────────

IntExpr Parser::parse_int_expr() { return parse_int_add(); }

IntExpr Parser::parse_int_add() {
    IntExpr left = parse_int_mul();
    while (at(TT::PLUS, TT::MINUS)) {
        std::string op = advance().value;
        IntExpr right = parse_int_mul();
        left = std::make_shared<IntBinExpr>(IntBinExpr{op, std::move(left), std::move(right)});
    }
    return left;
}

IntExpr Parser::parse_int_mul() {
    IntExpr left = parse_int_primary();
    while (at(TT::STAR)) {
        advance();
        IntExpr right = parse_int_primary();
        left = std::make_shared<IntBinExpr>(IntBinExpr{"*", std::move(left), std::move(right)});
    }
    return left;
}

IntExpr Parser::parse_int_primary() {
    if (at(TT::INT_LIT)) {
        Token tok = advance();
        return IntLitExpr{std::stoi(tok.value)};
    }
    if (at(TT::LPAREN)) {
        advance();
        IntExpr expr = parse_int_expr();
        expect(TT::RPAREN);
        return expr;
    }
    if (at(TT::POPCOUNT)) {
        advance();
        expect(TT::LPAREN);
        Expr arg = parse_stream_expr();
        expect(TT::RPAREN);
        return std::make_shared<PopcountExpr>(PopcountExpr{std::move(arg)});
    }
    if (at(TT::IDENT)) {
        std::string name = advance().value;
        return IntVarExpr{name};
    }
    const Token& tok = peek();
    throw ParseError("line " + std::to_string(tok.line) +
                      ": expected integer expression, got " + tt_name(tok.type) +
                      " ('" + tok.value + "')");
}

// ── 3-address validation ────────────────────────────────────────

static bool is_primary(const Expr& e) {
    return std::holds_alternative<VarExpr>(e) ||
           std::holds_alternative<ConstExpr>(e) ||
           std::holds_alternative<ArrayAccessExpr>(e);
}

static void validate_stream_rhs(const Expr& e, const std::string& ctx) {
    if (is_primary(e)) return;
    if (auto p = std::get_if<std::shared_ptr<UnaryExpr>>(&e)) {
        if (!is_primary((*p)->operand))
            throw ThreeAddrError(ctx + ": unary operand must be a primary");
        return;
    }
    if (auto p = std::get_if<std::shared_ptr<BinExpr>>(&e)) {
        if (!is_primary((*p)->left))
            throw ThreeAddrError(ctx + ": binary left operand must be a primary");
        if (!is_primary((*p)->right))
            throw ThreeAddrError(ctx + ": binary right operand must be a primary");
        return;
    }
    if (auto p = std::get_if<std::shared_ptr<ShiftExpr>>(&e)) {
        if (!is_primary((*p)->stream))
            throw ThreeAddrError(ctx + ": shift operand must be a primary");
        return;
    }
    throw ThreeAddrError(ctx + ": unknown expression type in stream RHS");
}

static void validate_int_rhs(const IntExpr& e, const std::string& ctx) {
    if (auto p = std::get_if<std::shared_ptr<PopcountExpr>>(&e)) {
        if (!std::holds_alternative<VarExpr>((*p)->arg))
            throw ThreeAddrError(ctx + ": popcount arg must be a variable");
    }
}

static void validate_stmt(const Stmt& s, const std::string& ctx);

static void validate_stmts(const std::vector<Stmt>& stmts, const std::string& ctx) {
    for (const auto& s : stmts) validate_stmt(s, ctx);
}

static void validate_stmt(const Stmt& s, const std::string& ctx) {
    if (auto p = std::get_if<LocalDecl>(&s)) {
        if (p->init_expr)
            validate_stream_rhs(*p->init_expr, ctx + ": stream " + p->name);
    } else if (auto p = std::get_if<Assign>(&s)) {
        validate_stream_rhs(p->expr, ctx + ": " + p->target);
    } else if (auto p = std::get_if<ArrayAssign>(&s)) {
        validate_stream_rhs(p->expr, ctx + ": " + p->target + "[...]");
    } else if (auto p = std::get_if<IntLocalDecl>(&s)) {
        validate_int_rhs(p->init_expr, ctx + ": int " + p->name);
    } else if (auto p = std::get_if<IntAssign>(&s)) {
        validate_int_rhs(p->expr, ctx + ": " + p->target);
    } else if (auto p = std::get_if<std::shared_ptr<IfStmt>>(&s)) {
        if (!std::holds_alternative<VarExpr>((*p)->cond))
            throw ThreeAddrError(ctx + ": if condition must be a variable");
        validate_stmts((*p)->body, ctx);
    } else if (auto p = std::get_if<std::shared_ptr<WhileStmt>>(&s)) {
        if (!std::holds_alternative<VarExpr>((*p)->cond))
            throw ThreeAddrError(ctx + ": while condition must be a variable");
        validate_stmts((*p)->body, ctx);
    } else if (auto p = std::get_if<std::shared_ptr<ForStmt>>(&s)) {
        validate_stmts((*p)->body, ctx);
    }
    // ArrayDecl, IntLocalDecl without popcount: no additional checks
}

void validate_3addr(const Program& prog) {
    std::string ctx = prog.inputs.empty() ? "<program>" : prog.inputs[0];
    for (const auto& s : prog.stmts) validate_stmt(s, ctx);
}

// ── Public API ──────────────────────────────────────────────────

Program parse(const std::string& source) {
    auto tokens = tokenize(source);
    Parser parser(std::move(tokens));
    Program prog = parser.parse_program();
    validate_3addr(prog);
    return prog;
}

} // namespace bs
