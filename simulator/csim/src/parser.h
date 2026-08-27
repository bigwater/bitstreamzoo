#pragma once

#include "ast.h"
#include "lexer.h"
#include <set>
#include <stdexcept>

namespace bs {

class ParseError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class ThreeAddrError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class Parser {
public:
    explicit Parser(std::vector<Token> tokens);
    Program parse_program();

private:
    std::vector<Token> tokens_;
    size_t pos_ = 0;
    std::set<std::string> int_names_;
    std::set<std::string> array_names_;
    std::vector<std::string> output_int_names_;

    const Token& peek() const;
    Token advance();
    Token expect(TT tt);
    bool at(TT t) const;
    bool at(TT t1, TT t2) const;
    bool at(TT t1, TT t2, TT t3) const;

    // Declarations
    void parse_input_decl(Program& prog);
    void parse_output_decl(Program& prog);
    void parse_param_decl(Program& prog);
    Stmt parse_local_or_array_decl();
    Stmt parse_int_local_decl();

    // Statements
    Stmt parse_assignment();
    Stmt parse_control_stmt();
    Stmt parse_if();
    Stmt parse_while();
    Stmt parse_for();
    std::vector<Stmt> parse_block();

    // Stream expressions (precedence climbing)
    Expr parse_stream_expr();
    Expr parse_or();
    Expr parse_xor();
    Expr parse_add();
    Expr parse_and();
    Expr parse_shift();
    Expr parse_unary();
    Expr parse_primary();

    // Integer expressions
    IntExpr parse_int_expr();
    IntExpr parse_int_add();
    IntExpr parse_int_mul();
    IntExpr parse_int_primary();
};

Program parse(const std::string& source);
void validate_3addr(const Program& prog);

} // namespace bs
