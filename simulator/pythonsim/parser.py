"""AST nodes and recursive-descent parser for the Bitstream DSL."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

from .lexer import Token, TT, tokenize

# ── AST Nodes ──────────────────────────────────────────────────────

# Declarations
@dataclass
class InputDecl:
    name: str

@dataclass
class OutputDecl:
    name: str

@dataclass
class ParamDecl:
    name: str

@dataclass
class LocalDecl:
    name: str
    init_expr: Expr | None = None  # None → default ZERO

@dataclass
class ArrayDecl:
    name: str
    size_expr: IntExpr

# Stream expressions
@dataclass
class VarExpr:
    name: str

@dataclass
class ArrayAccessExpr:
    name: str
    index: IntExpr

@dataclass
class ConstExpr:
    name: str  # "ZERO" or "ONES"

@dataclass
class UnaryExpr:
    op: str  # "~"
    operand: Expr

@dataclass
class BinExpr:
    op: str  # "&", "|", "^", "+"
    left: Expr
    right: Expr

@dataclass
class ShiftExpr:
    op: str  # "<<", ">>"
    stream: Expr
    amount: IntExpr

# Integer expressions
@dataclass
class IntLitExpr:
    value: int

@dataclass
class IntVarExpr:
    name: str

@dataclass
class IntBinExpr:
    op: str  # "+", "-", "*"
    left: IntExpr
    right: IntExpr

@dataclass
class PopcountExpr:
    """popcount(stream_expr) — stream-to-int reduction."""
    arg: Expr

# Integer-typed declarations and assignments
@dataclass
class IntLocalDecl:
    """int x = int_expr"""
    name: str
    init_expr: IntExpr

@dataclass
class IntAssign:
    """x = int_expr  (where x is an int variable)"""
    target: str
    expr: IntExpr

# Statements
@dataclass
class Assign:
    target: str
    expr: Expr

@dataclass
class ArrayAssign:
    target: str
    index: IntExpr
    expr: Expr

@dataclass
class IfStmt:
    cond: Expr
    body: list[Stmt]

@dataclass
class WhileStmt:
    cond: Expr
    body: list[Stmt]

@dataclass
class ForStmt:
    var: str
    lo: IntExpr
    hi: IntExpr
    body: list[Stmt]

# Top-level program
@dataclass
class Program:
    decls: list[Decl]
    stmts: list[Stmt]
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    params: list[str] = field(default_factory=list)
    output_int_names: list[str] = field(default_factory=list)

# Type aliases
Expr = Union[VarExpr, ArrayAccessExpr, ConstExpr, UnaryExpr, BinExpr, ShiftExpr]
IntExpr = Union[IntLitExpr, IntVarExpr, IntBinExpr, PopcountExpr]
Stmt = Union[Assign, ArrayAssign, IfStmt, WhileStmt, ForStmt, LocalDecl, ArrayDecl, IntLocalDecl, IntAssign]
Decl = Union[InputDecl, OutputDecl, ParamDecl, LocalDecl, ArrayDecl]


# ── Parser ─────────────────────────────────────────────────────────

class ParseError(Exception):
    pass


class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0
        # Track which names are declared as int (params + loop vars + int locals)
        self.int_names: set[str] = set()
        # Track which names are stream arrays
        self.array_names: set[str] = set()
        # Track output int names
        self.output_int_names: list[str] = []

    def peek(self) -> Token:
        return self.tokens[self.pos]

    def advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect(self, tt: TT) -> Token:
        tok = self.advance()
        if tok.type != tt:
            raise ParseError(f"line {tok.line}: expected {tt.name}, got {tok.type.name} ({tok.value!r})")
        return tok

    def at(self, *types: TT) -> bool:
        return self.peek().type in types

    # ── Top-level ────────────────────────────────────────────────

    def parse_program(self) -> Program:
        decls: list[Decl] = []
        stmts: list[Stmt] = []
        inputs, outputs, params = [], [], []

        while not self.at(TT.EOF):
            if self.at(TT.INPUT):
                d = self._parse_input_decl()
                decls.append(d)
                inputs.append(d.name)
            elif self.at(TT.OUTPUT):
                # Could be "output stream ..." or "output int ..."
                saved = self.pos
                self.advance()  # consume OUTPUT
                if self.at(TT.INT):
                    self.advance()  # consume INT
                    name = self.expect(TT.IDENT).value
                    self.int_names.add(name)
                    self.output_int_names.append(name)
                    # output int vars also need initialization
                    stmts.append(IntLocalDecl(name, IntLitExpr(0)))
                else:
                    self.pos = saved  # put back
                    d = self._parse_output_decl()
                    decls.append(d)
                    outputs.append(d.name)
            elif self.at(TT.PARAM):
                d = self._parse_param_decl()
                decls.append(d)
                params.append(d.name)
            elif self.at(TT.STREAM):
                d = self._parse_local_or_array_decl()
                decls.append(d)
                stmts.append(d)
            elif self.at(TT.INT):
                # int local: int x = int_expr
                d = self._parse_int_local_decl()
                stmts.append(d)
            elif self.at(TT.IF, TT.WHILE, TT.FOR):
                stmts.append(self._parse_control_stmt())
            elif self.at(TT.IDENT):
                stmts.append(self._parse_assignment())
            else:
                tok = self.peek()
                raise ParseError(f"line {tok.line}: unexpected token {tok.type.name} ({tok.value!r})")

        return Program(decls, stmts, inputs, outputs, params,
                       output_int_names=self.output_int_names)

    def _parse_input_decl(self) -> InputDecl:
        self.expect(TT.INPUT)
        self.expect(TT.STREAM)
        name = self.expect(TT.IDENT).value
        if self.at(TT.LBRACKET):
            # input stream arr[N] — treat as array
            self.advance()
            size = self._parse_int_expr()
            self.expect(TT.RBRACKET)
            self.array_names.add(name)
        return InputDecl(name)

    def _parse_output_decl(self) -> OutputDecl:
        self.expect(TT.OUTPUT)
        self.expect(TT.STREAM)
        name = self.expect(TT.IDENT).value
        if self.at(TT.LBRACKET):
            self.advance()
            size = self._parse_int_expr()
            self.expect(TT.RBRACKET)
            self.array_names.add(name)
        return OutputDecl(name)

    def _parse_param_decl(self) -> ParamDecl:
        self.expect(TT.PARAM)
        self.expect(TT.INT)
        name = self.expect(TT.IDENT).value
        self.int_names.add(name)
        return ParamDecl(name)

    def _parse_local_or_array_decl(self) -> LocalDecl | ArrayDecl:
        self.expect(TT.STREAM)
        name = self.expect(TT.IDENT).value

        # Array declaration: stream foo[N]
        if self.at(TT.LBRACKET):
            self.advance()
            size = self._parse_int_expr()
            self.expect(TT.RBRACKET)
            self.array_names.add(name)
            return ArrayDecl(name, size)

        # Optional initializer
        init = None
        if self.at(TT.ASSIGN):
            self.advance()
            init = self._parse_stream_expr()
        return LocalDecl(name, init)

    # ── Statements ───────────────────────────────────────────────

    def _parse_int_local_decl(self) -> IntLocalDecl:
        """Parse: int x = int_expr"""
        self.expect(TT.INT)
        name = self.expect(TT.IDENT).value
        self.int_names.add(name)
        self.expect(TT.ASSIGN)
        init = self._parse_int_expr()
        return IntLocalDecl(name, init)

    def _parse_assignment(self) -> Assign | ArrayAssign | IntAssign:
        name = self.expect(TT.IDENT).value

        # Array element assignment: foo[i] = expr
        if self.at(TT.LBRACKET):
            self.advance()
            idx = self._parse_int_expr()
            self.expect(TT.RBRACKET)
            self.expect(TT.ASSIGN)
            expr = self._parse_stream_expr()
            return ArrayAssign(name, idx, expr)

        self.expect(TT.ASSIGN)
        # If the target is a known int variable, parse RHS as int expression
        if name in self.int_names:
            expr = self._parse_int_expr()
            return IntAssign(name, expr)
        expr = self._parse_stream_expr()
        return Assign(name, expr)

    def _parse_control_stmt(self) -> IfStmt | WhileStmt | ForStmt:
        tok = self.peek()
        if tok.type == TT.IF:
            return self._parse_if()
        elif tok.type == TT.WHILE:
            return self._parse_while()
        else:
            return self._parse_for()

    def _parse_if(self) -> IfStmt:
        self.expect(TT.IF)
        self.expect(TT.LPAREN)
        cond = self._parse_stream_expr()
        self.expect(TT.RPAREN)
        body = self._parse_block()
        return IfStmt(cond, body)

    def _parse_while(self) -> WhileStmt:
        self.expect(TT.WHILE)
        self.expect(TT.LPAREN)
        cond = self._parse_stream_expr()
        self.expect(TT.RPAREN)
        body = self._parse_block()
        return WhileStmt(cond, body)

    def _parse_for(self) -> ForStmt:
        self.expect(TT.FOR)
        var = self.expect(TT.IDENT).value
        self.int_names.add(var)
        self.expect(TT.IN)
        lo = self._parse_int_expr()
        self.expect(TT.DOTDOT)
        hi = self._parse_int_expr()
        body = self._parse_block()
        return ForStmt(var, lo, hi, body)

    def _parse_block(self) -> list[Stmt]:
        self.expect(TT.LBRACE)
        stmts = []
        while not self.at(TT.RBRACE):
            if self.at(TT.STREAM):
                d = self._parse_local_or_array_decl()
                stmts.append(d)
            elif self.at(TT.INT):
                stmts.append(self._parse_int_local_decl())
            elif self.at(TT.IF, TT.WHILE, TT.FOR):
                stmts.append(self._parse_control_stmt())
            elif self.at(TT.IDENT):
                stmts.append(self._parse_assignment())
            else:
                tok = self.peek()
                raise ParseError(f"line {tok.line}: unexpected token in block: {tok.type.name} ({tok.value!r})")
        self.expect(TT.RBRACE)
        return stmts

    # ── Stream expressions (precedence climbing) ─────────────────
    # Precedence (high→low): ~ (unary), << >>, &, +, ^, |

    def _parse_stream_expr(self) -> Expr:
        return self._parse_or()

    def _parse_or(self) -> Expr:
        left = self._parse_xor()
        while self.at(TT.PIPE):
            self.advance()
            right = self._parse_xor()
            left = BinExpr("|", left, right)
        return left

    def _parse_xor(self) -> Expr:
        left = self._parse_add()
        while self.at(TT.CARET):
            self.advance()
            right = self._parse_add()
            left = BinExpr("^", left, right)
        return left

    def _parse_add(self) -> Expr:
        left = self._parse_and()
        while self.at(TT.PLUS):
            self.advance()
            right = self._parse_and()
            left = BinExpr("+", left, right)
        return left

    def _parse_and(self) -> Expr:
        left = self._parse_shift()
        while self.at(TT.AMP):
            self.advance()
            right = self._parse_shift()
            left = BinExpr("&", left, right)
        return left

    def _parse_shift(self) -> Expr:
        left = self._parse_unary()
        while self.at(TT.LSHIFT, TT.RSHIFT):
            op = self.advance().value
            amount = self._parse_int_expr()
            left = ShiftExpr(op, left, amount)
        return left

    def _parse_unary(self) -> Expr:
        if self.at(TT.TILDE):
            self.advance()
            operand = self._parse_unary()
            return UnaryExpr("~", operand)
        return self._parse_primary()

    def _parse_primary(self) -> Expr:
        if self.at(TT.ZERO):
            self.advance()
            return ConstExpr("ZERO")
        if self.at(TT.ONES):
            self.advance()
            return ConstExpr("ONES")
        if self.at(TT.LPAREN):
            self.advance()
            expr = self._parse_stream_expr()
            self.expect(TT.RPAREN)
            return expr
        if self.at(TT.IDENT):
            name = self.advance().value
            # Array access
            if self.at(TT.LBRACKET):
                self.advance()
                idx = self._parse_int_expr()
                self.expect(TT.RBRACKET)
                return ArrayAccessExpr(name, idx)
            return VarExpr(name)
        tok = self.peek()
        raise ParseError(f"line {tok.line}: expected stream expression, got {tok.type.name} ({tok.value!r})")

    # ── Integer expressions ──────────────────────────────────────
    # Precedence: * before +/-

    def _parse_int_expr(self) -> IntExpr:
        return self._parse_int_add()

    def _parse_int_add(self) -> IntExpr:
        left = self._parse_int_mul()
        while self.at(TT.PLUS, TT.MINUS):
            op = self.advance().value
            right = self._parse_int_mul()
            left = IntBinExpr(op, left, right)
        return left

    def _parse_int_mul(self) -> IntExpr:
        left = self._parse_int_primary()
        while self.at(TT.STAR):
            self.advance()
            right = self._parse_int_primary()
            left = IntBinExpr("*", left, right)
        return left

    def _parse_int_primary(self) -> IntExpr:
        if self.at(TT.INT_LIT):
            tok = self.advance()
            return IntLitExpr(int(tok.value))
        if self.at(TT.LPAREN):
            self.advance()
            expr = self._parse_int_expr()
            self.expect(TT.RPAREN)
            return expr
        if self.at(TT.POPCOUNT):
            self.advance()
            self.expect(TT.LPAREN)
            arg = self._parse_stream_expr()
            self.expect(TT.RPAREN)
            return PopcountExpr(arg)
        if self.at(TT.IDENT):
            name = self.advance().value
            return IntVarExpr(name)
        tok = self.peek()
        raise ParseError(f"line {tok.line}: expected integer expression, got {tok.type.name} ({tok.value!r})")


class ThreeAddrError(Exception):
    pass


def _is_primary(expr) -> bool:
    """Check if an expression is a primary (variable, constant, or array access)."""
    return isinstance(expr, (VarExpr, ConstExpr, ArrayAccessExpr))


def _validate_stream_rhs(expr, context: str):
    """Validate that a stream RHS is 3-address: at most one operator on primaries."""
    if _is_primary(expr):
        return
    if isinstance(expr, UnaryExpr):
        if not _is_primary(expr.operand):
            raise ThreeAddrError(
                f"{context}: unary operand must be a primary, "
                f"got {type(expr.operand).__name__}")
        return
    if isinstance(expr, BinExpr):
        if not _is_primary(expr.left):
            raise ThreeAddrError(
                f"{context}: binary left operand must be a primary, "
                f"got {type(expr.left).__name__}")
        if not _is_primary(expr.right):
            raise ThreeAddrError(
                f"{context}: binary right operand must be a primary, "
                f"got {type(expr.right).__name__}")
        return
    if isinstance(expr, ShiftExpr):
        if not _is_primary(expr.stream):
            raise ThreeAddrError(
                f"{context}: shift operand must be a primary, "
                f"got {type(expr.stream).__name__}")
        return
    raise ThreeAddrError(f"{context}: unknown expression type {type(expr).__name__}")


def _validate_int_rhs(expr, context: str):
    """Validate that an int RHS has only primary args for popcount."""
    if isinstance(expr, PopcountExpr):
        if not isinstance(expr.arg, VarExpr):
            raise ThreeAddrError(
                f"{context}: popcount arg must be a variable, "
                f"got {type(expr.arg).__name__}")


def _validate_stmt(stmt, context: str):
    """Validate a single statement for 3-address form."""
    if isinstance(stmt, LocalDecl):
        if stmt.init_expr is not None:
            _validate_stream_rhs(stmt.init_expr, f"{context}: stream {stmt.name}")
    elif isinstance(stmt, Assign):
        _validate_stream_rhs(stmt.expr, f"{context}: {stmt.target}")
    elif isinstance(stmt, ArrayAssign):
        _validate_stream_rhs(stmt.expr, f"{context}: {stmt.target}[...]")
    elif isinstance(stmt, IntLocalDecl):
        _validate_int_rhs(stmt.init_expr, f"{context}: int {stmt.name}")
    elif isinstance(stmt, IntAssign):
        _validate_int_rhs(stmt.expr, f"{context}: {stmt.target}")
    elif isinstance(stmt, IfStmt):
        if not isinstance(stmt.cond, VarExpr):
            raise ThreeAddrError(
                f"{context}: if condition must be a variable, "
                f"got {type(stmt.cond).__name__}")
        for s in stmt.body:
            _validate_stmt(s, context)
    elif isinstance(stmt, WhileStmt):
        if not isinstance(stmt.cond, VarExpr):
            raise ThreeAddrError(
                f"{context}: while condition must be a variable, "
                f"got {type(stmt.cond).__name__}")
        for s in stmt.body:
            _validate_stmt(s, context)
    elif isinstance(stmt, ForStmt):
        for s in stmt.body:
            _validate_stmt(s, context)
    elif isinstance(stmt, ArrayDecl):
        pass  # array declarations have no RHS


def validate_3addr(program: Program):
    """Validate that a program is in 3-address form.

    Rules:
    - Assign/LocalDecl RHS: at most one operator, operands must be primaries
    - if/while condition: must be a VarExpr
    - popcount(arg): arg must be a VarExpr
    - ~operand: operand must be a primary
    - a op b: both operands must be primaries
    - a << n: stream operand must be a primary
    """
    for stmt in program.stmts:
        _validate_stmt(stmt, program.inputs[0] if program.inputs else "<program>")


def count_stmts(program: Program) -> int:
    """Count total statements in a program (recursive into control flow bodies)."""
    def _count(stmts: list) -> int:
        total = 0
        for stmt in stmts:
            total += 1
            if isinstance(stmt, (IfStmt, WhileStmt)):
                total += _count(stmt.body)
            elif isinstance(stmt, ForStmt):
                total += _count(stmt.body)
        return total
    return _count(program.stmts)


def parse(source: str) -> Program:
    """Parse a .bs source string into a Program AST.

    Validates that the program is in 3-address form (one op per assignment).
    """
    tokens = tokenize(source)
    parser = Parser(tokens)
    program = parser.parse_program()
    validate_3addr(program)
    return program
