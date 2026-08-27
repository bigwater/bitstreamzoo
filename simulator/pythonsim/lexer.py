"""Tokenizer for the Bitstream DSL."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto


class TT(Enum):
    """Token types."""
    # Keywords
    INPUT = auto()
    OUTPUT = auto()
    PARAM = auto()
    STREAM = auto()
    INT = auto()
    IF = auto()
    WHILE = auto()
    FOR = auto()
    IN = auto()
    ZERO = auto()
    ONES = auto()
    POPCOUNT = auto()
    # Identifiers and literals
    IDENT = auto()
    INT_LIT = auto()
    # Delimiters
    LBRACE = auto()
    RBRACE = auto()
    LPAREN = auto()
    RPAREN = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    # Operators
    ASSIGN = auto()
    TILDE = auto()
    AMP = auto()
    PIPE = auto()
    CARET = auto()
    LSHIFT = auto()
    RSHIFT = auto()
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    # Misc
    DOTDOT = auto()
    EOF = auto()


KEYWORDS = {
    "input": TT.INPUT,
    "output": TT.OUTPUT,
    "param": TT.PARAM,
    "stream": TT.STREAM,
    "int": TT.INT,
    "if": TT.IF,
    "while": TT.WHILE,
    "for": TT.FOR,
    "in": TT.IN,
    "ZERO": TT.ZERO,
    "ONES": TT.ONES,
    "popcount": TT.POPCOUNT,
}


@dataclass
class Token:
    type: TT
    value: str
    line: int


# Token patterns in priority order
_TOKEN_SPEC = [
    ("COMMENT", r"//[^\n]*"),
    ("WS", r"[ \t\r\n]+"),
    ("DOTDOT", r"\.\."),
    ("LSHIFT", r"<<"),
    ("RSHIFT", r">>"),
    ("INT_LIT", r"[0-9]+"),
    ("IDENT", r"[a-zA-Z_][a-zA-Z0-9_]*"),
    ("LBRACE", r"\{"),
    ("RBRACE", r"\}"),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("LBRACKET", r"\["),
    ("RBRACKET", r"\]"),
    ("ASSIGN", r"="),
    ("TILDE", r"~"),
    ("AMP", r"&"),
    ("PIPE", r"\|"),
    ("CARET", r"\^"),
    ("PLUS", r"\+"),
    ("MINUS", r"-"),
    ("STAR", r"\*"),
]

_TOKEN_RE = re.compile("|".join(f"(?P<{name}>{pat})" for name, pat in _TOKEN_SPEC))

_SIMPLE_MAP = {
    "DOTDOT": TT.DOTDOT,
    "LSHIFT": TT.LSHIFT,
    "RSHIFT": TT.RSHIFT,
    "INT_LIT": TT.INT_LIT,
    "LBRACE": TT.LBRACE,
    "RBRACE": TT.RBRACE,
    "LPAREN": TT.LPAREN,
    "RPAREN": TT.RPAREN,
    "LBRACKET": TT.LBRACKET,
    "RBRACKET": TT.RBRACKET,
    "ASSIGN": TT.ASSIGN,
    "TILDE": TT.TILDE,
    "AMP": TT.AMP,
    "PIPE": TT.PIPE,
    "CARET": TT.CARET,
    "PLUS": TT.PLUS,
    "MINUS": TT.MINUS,
    "STAR": TT.STAR,
}


def tokenize(source: str) -> list[Token]:
    """Tokenize a .bs source string into a list of Tokens."""
    tokens: list[Token] = []
    line = 1
    for m in _TOKEN_RE.finditer(source):
        kind = m.lastgroup
        value = m.group()
        # Track line numbers
        line += value.count("\n")
        # Skip whitespace and comments
        if kind in ("COMMENT", "WS"):
            continue
        if kind == "IDENT":
            tt = KEYWORDS.get(value, TT.IDENT)
        elif kind == "INT_LIT":
            tt = TT.INT_LIT
        else:
            tt = _SIMPLE_MAP[kind]
        tokens.append(Token(tt, value, line))
    tokens.append(Token(TT.EOF, "", line))
    return tokens
