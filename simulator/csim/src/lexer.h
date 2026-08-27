#pragma once

#include <string>
#include <vector>

namespace bs {

enum class TT {
    // Keywords
    INPUT, OUTPUT, PARAM, STREAM, INT, IF, WHILE, FOR, IN, ZERO, ONES, POPCOUNT,
    // Identifiers and literals
    IDENT, INT_LIT,
    // Delimiters
    LBRACE, RBRACE, LPAREN, RPAREN, LBRACKET, RBRACKET,
    // Operators
    ASSIGN, TILDE, AMP, PIPE, CARET, LSHIFT, RSHIFT, PLUS, MINUS, STAR,
    // Misc
    DOTDOT, EOF_TOK
};

struct Token {
    TT type;
    std::string value;
    int line;
};

std::vector<Token> tokenize(const std::string& source);

const char* tt_name(TT t);

} // namespace bs
