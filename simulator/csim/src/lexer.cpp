#include "lexer.h"
#include <stdexcept>
#include <unordered_map>

namespace bs {

static const std::unordered_map<std::string, TT> KEYWORDS = {
    {"input", TT::INPUT}, {"output", TT::OUTPUT}, {"param", TT::PARAM},
    {"stream", TT::STREAM}, {"int", TT::INT}, {"if", TT::IF},
    {"while", TT::WHILE}, {"for", TT::FOR}, {"in", TT::IN},
    {"ZERO", TT::ZERO}, {"ONES", TT::ONES}, {"popcount", TT::POPCOUNT},
};

const char* tt_name(TT t) {
    switch (t) {
        case TT::INPUT: return "INPUT";
        case TT::OUTPUT: return "OUTPUT";
        case TT::PARAM: return "PARAM";
        case TT::STREAM: return "STREAM";
        case TT::INT: return "INT";
        case TT::IF: return "IF";
        case TT::WHILE: return "WHILE";
        case TT::FOR: return "FOR";
        case TT::IN: return "IN";
        case TT::ZERO: return "ZERO";
        case TT::ONES: return "ONES";
        case TT::POPCOUNT: return "POPCOUNT";
        case TT::IDENT: return "IDENT";
        case TT::INT_LIT: return "INT_LIT";
        case TT::LBRACE: return "LBRACE";
        case TT::RBRACE: return "RBRACE";
        case TT::LPAREN: return "LPAREN";
        case TT::RPAREN: return "RPAREN";
        case TT::LBRACKET: return "LBRACKET";
        case TT::RBRACKET: return "RBRACKET";
        case TT::ASSIGN: return "ASSIGN";
        case TT::TILDE: return "TILDE";
        case TT::AMP: return "AMP";
        case TT::PIPE: return "PIPE";
        case TT::CARET: return "CARET";
        case TT::LSHIFT: return "LSHIFT";
        case TT::RSHIFT: return "RSHIFT";
        case TT::PLUS: return "PLUS";
        case TT::MINUS: return "MINUS";
        case TT::STAR: return "STAR";
        case TT::DOTDOT: return "DOTDOT";
        case TT::EOF_TOK: return "EOF";
    }
    return "?";
}

static bool is_ident_start(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

static bool is_ident_char(char c) {
    return is_ident_start(c) || (c >= '0' && c <= '9');
}

static bool is_digit(char c) { return c >= '0' && c <= '9'; }

std::vector<Token> tokenize(const std::string& src) {
    std::vector<Token> tokens;
    int line = 1;
    size_t i = 0;
    size_t n = src.size();

    while (i < n) {
        char c = src[i];

        // Whitespace
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            if (c == '\n') ++line;
            ++i;
            continue;
        }

        // Line comment
        if (c == '/' && i + 1 < n && src[i + 1] == '/') {
            i += 2;
            while (i < n && src[i] != '\n') ++i;
            continue;
        }

        // Two-char operators
        if (c == '<' && i + 1 < n && src[i + 1] == '<') {
            tokens.push_back({TT::LSHIFT, "<<", line});
            i += 2; continue;
        }
        if (c == '>' && i + 1 < n && src[i + 1] == '>') {
            tokens.push_back({TT::RSHIFT, ">>", line});
            i += 2; continue;
        }
        if (c == '.' && i + 1 < n && src[i + 1] == '.') {
            tokens.push_back({TT::DOTDOT, "..", line});
            i += 2; continue;
        }

        // Single-char tokens
        switch (c) {
            case '{': tokens.push_back({TT::LBRACE, "{", line}); ++i; continue;
            case '}': tokens.push_back({TT::RBRACE, "}", line}); ++i; continue;
            case '(': tokens.push_back({TT::LPAREN, "(", line}); ++i; continue;
            case ')': tokens.push_back({TT::RPAREN, ")", line}); ++i; continue;
            case '[': tokens.push_back({TT::LBRACKET, "[", line}); ++i; continue;
            case ']': tokens.push_back({TT::RBRACKET, "]", line}); ++i; continue;
            case '=': tokens.push_back({TT::ASSIGN, "=", line}); ++i; continue;
            case '~': tokens.push_back({TT::TILDE, "~", line}); ++i; continue;
            case '&': tokens.push_back({TT::AMP, "&", line}); ++i; continue;
            case '|': tokens.push_back({TT::PIPE, "|", line}); ++i; continue;
            case '^': tokens.push_back({TT::CARET, "^", line}); ++i; continue;
            case '+': tokens.push_back({TT::PLUS, "+", line}); ++i; continue;
            case '-': tokens.push_back({TT::MINUS, "-", line}); ++i; continue;
            case '*': tokens.push_back({TT::STAR, "*", line}); ++i; continue;
        }

        // Integer literal
        if (is_digit(c)) {
            size_t start = i;
            while (i < n && is_digit(src[i])) ++i;
            tokens.push_back({TT::INT_LIT, src.substr(start, i - start), line});
            continue;
        }

        // Identifier or keyword
        if (is_ident_start(c)) {
            size_t start = i;
            while (i < n && is_ident_char(src[i])) ++i;
            std::string word = src.substr(start, i - start);
            auto it = KEYWORDS.find(word);
            TT tt = (it != KEYWORDS.end()) ? it->second : TT::IDENT;
            tokens.push_back({tt, std::move(word), line});
            continue;
        }

        throw std::runtime_error("line " + std::to_string(line) +
                                 ": unexpected character '" + std::string(1, c) + "'");
    }

    tokens.push_back({TT::EOF_TOK, "", line});
    return tokens;
}

} // namespace bs
