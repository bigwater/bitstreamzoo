"""Interpreter for the Bitstream DSL — executes AST statements directly.

Pipeline:  parse(source) → interpret(AST statements)

Each statement is one atomic operation (3-address form).
The interpreter steps through them sequentially, counting ops and
optionally tracing.
"""

from __future__ import annotations

from .parser import (
    Program, Assign, ArrayAssign, IfStmt, WhileStmt, ForStmt,
    LocalDecl, ArrayDecl, IntLocalDecl, IntAssign,
    VarExpr, ArrayAccessExpr, ConstExpr, UnaryExpr, BinExpr, ShiftExpr,
    IntLitExpr, IntVarExpr, IntBinExpr, PopcountExpr,
    parse,
)

MAX_WHILE_ITERS = 10_000

# Ops that count as bitwise operations (the ones that matter for GPU cost)
BITWISE_OPS = {"~", "&", "|", "^", "<<", ">>", "+"}


class RuntimeError_(Exception):
    pass


class Interpreter:
    def __init__(self, trace: bool = False, bitlength: int = 0):
        self.env: dict[str, int] = {}               # stream vars
        self.arrays: dict[str, dict[int, int]] = {}  # stream arrays
        self.int_env: dict[str, int] = {}            # int vars (params + loop vars)
        self.trace = trace
        self.bitlength = bitlength                   # 0 = no masking
        self._mask = (1 << bitlength) - 1 if bitlength > 0 else 0
        self.op_count = 0                            # total bitwise ops executed
        self.op_counts: dict[str, int] = {}          # per-op-type counts

    def _count(self, op: str):
        self.op_count += 1
        self.op_counts[op] = self.op_counts.get(op, 0) + 1

    def _eval_int(self, expr) -> int:
        if isinstance(expr, IntLitExpr):
            return expr.value
        if isinstance(expr, IntVarExpr):
            if expr.name not in self.int_env:
                raise RuntimeError_(f"undefined int variable: {expr.name}")
            return self.int_env[expr.name]
        if isinstance(expr, IntBinExpr):
            l = self._eval_int(expr.left)
            r = self._eval_int(expr.right)
            if expr.op == "+": return l + r
            if expr.op == "-": return l - r
            if expr.op == "*": return l * r
        if isinstance(expr, PopcountExpr):
            # In 3-address form, popcount arg is always a VarExpr
            val = self.env[expr.arg.name]
            if val < 0:
                raise RuntimeError_(f"popcount on negative value (infinite bits set)")
            count = bin(val).count('1')
            self._count("popcount")
            if self.trace:
                print(f"  [{self.op_count:3d}] popcount({expr.arg.name}) = {count}")
            return count
        raise RuntimeError_(f"unknown int expr: {type(expr)}")

    def _eval_stream(self, expr) -> int:
        """Evaluate a stream primary expression (VarExpr, ConstExpr, ArrayAccessExpr)."""
        if isinstance(expr, VarExpr):
            return self.env[expr.name]
        if isinstance(expr, ConstExpr):
            if expr.name == "ZERO":
                return 0
            return self._mask if self._mask else -1
        if isinstance(expr, ArrayAccessExpr):
            idx = self._eval_int(expr.index)
            arr = self.arrays.get(expr.name)
            if arr is None:
                raise RuntimeError_(f"undefined array: {expr.name}")
            return arr.get(idx, 0)
        raise RuntimeError_(f"unknown stream primary: {type(expr)}")

    def _exec_stmt(self, stmt):
        if isinstance(stmt, LocalDecl):
            if stmt.init_expr is None:
                self.env[stmt.name] = 0
            else:
                self.env[stmt.name] = self._exec_stream_rhs(
                    stmt.init_expr, stmt.name)
            return

        if isinstance(stmt, ArrayDecl):
            size = self._eval_int(stmt.size_expr)
            self.arrays[stmt.name] = {i: 0 for i in range(size)}
            return

        if isinstance(stmt, Assign):
            self.env[stmt.target] = self._exec_stream_rhs(
                stmt.expr, stmt.target)
            return

        if isinstance(stmt, ArrayAssign):
            idx = self._eval_int(stmt.index)
            if stmt.target not in self.arrays:
                self.arrays[stmt.target] = {}
            self.arrays[stmt.target][idx] = self._exec_stream_rhs(
                stmt.expr, f"{stmt.target}[{idx}]")
            return

        if isinstance(stmt, IfStmt):
            cond = self.env[stmt.cond.name]
            if cond != 0:
                for s in stmt.body:
                    self._exec_stmt(s)
            return

        if isinstance(stmt, WhileStmt):
            for _ in range(MAX_WHILE_ITERS):
                cond = self.env[stmt.cond.name]
                if cond == 0:
                    break
                for s in stmt.body:
                    self._exec_stmt(s)
            else:
                raise RuntimeError_(f"while loop exceeded {MAX_WHILE_ITERS} iterations")
            return

        if isinstance(stmt, ForStmt):
            lo = self._eval_int(stmt.lo)
            hi = self._eval_int(stmt.hi)
            for i in range(lo, hi):
                self.int_env[stmt.var] = i
                for s in stmt.body:
                    self._exec_stmt(s)
            return

        if isinstance(stmt, IntLocalDecl):
            self.int_env[stmt.name] = self._eval_int(stmt.init_expr)
            return

        if isinstance(stmt, IntAssign):
            self.int_env[stmt.target] = self._eval_int(stmt.expr)
            return

        raise RuntimeError_(f"unknown statement: {type(stmt)}")

    def _exec_stream_rhs(self, expr, target_name: str) -> int:
        """Execute a stream RHS expression (3-address: at most one operator)."""
        # Primaries (no op counted)
        if isinstance(expr, VarExpr):
            val = self.env[expr.name]
            if self.trace:
                print(f"       {target_name} = {expr.name}")
            return val

        if isinstance(expr, ConstExpr):
            val = 0 if expr.name == "ZERO" else -1
            if self.trace:
                print(f"       {target_name} = {expr.name}")
            return val

        if isinstance(expr, ArrayAccessExpr):
            idx = self._eval_int(expr.index)
            arr = self.arrays.get(expr.name)
            if arr is None:
                raise RuntimeError_(f"undefined array: {expr.name}")
            val = arr.get(idx, 0)
            if self.trace:
                print(f"       {target_name} = {expr.name}[{idx}]")
            return val

        # Unary (1 op)
        if isinstance(expr, UnaryExpr):
            operand = self._eval_stream(expr.operand)
            result = ~operand
            if self._mask:
                result &= self._mask  # truncate NOT to bitlength
            self._count(expr.op)
            if self.trace:
                print(f"  [{self.op_count:3d}] {target_name} = ~...")
            return result

        # Binary (1 op)
        if isinstance(expr, BinExpr):
            l = self._eval_stream(expr.left)
            r = self._eval_stream(expr.right)
            if expr.op == "&": result = l & r
            elif expr.op == "|": result = l | r
            elif expr.op == "^": result = l ^ r
            elif expr.op == "+":
                result = l + r
                if self._mask:
                    result &= self._mask  # truncate carry-out to bitlength
            else: raise RuntimeError_(f"unknown binop: {expr.op}")
            self._count(expr.op)
            if self.trace:
                print(f"  [{self.op_count:3d}] {target_name} = ... {expr.op} ...")
            return result

        # Shift (1 op)
        if isinstance(expr, ShiftExpr):
            val = self._eval_stream(expr.stream)
            amt = self._eval_int(expr.amount)
            if expr.op == "<<":
                result = val << amt
                if self._mask:
                    result &= self._mask  # bits shifted past end-of-stream vanish
            else:
                result = val >> amt
            self._count(expr.op)
            if self.trace:
                print(f"  [{self.op_count:3d}] {target_name} = ... {expr.op} {amt}")
            return result

        raise RuntimeError_(f"unknown stream expression: {type(expr)}")

    def run(self, program: Program,
            inputs: dict[str, int],
            params: dict[str, int] | None = None,
            input_arrays: dict[str, dict[int, int]] | None = None) -> dict[str, int]:
        """Execute program statements and return output streams."""
        # Load params
        if params:
            self.int_env.update(params)

        # Load inputs
        for name in program.inputs:
            if name in inputs:
                self.env[name] = inputs[name]
            elif input_arrays and name in input_arrays:
                self.arrays[name] = dict(input_arrays[name])
            else:
                self.env[name] = 0

        # Execute each statement
        for stmt in program.stmts:
            self._exec_stmt(stmt)

        # Collect outputs
        result = {}
        for name in program.outputs:
            if name in self.env:
                result[name] = self.env[name]
            elif name in self.arrays:
                result[name] = dict(self.arrays[name])
        # Collect int outputs
        for name in program.output_int_names:
            if name in self.int_env:
                result[name] = self.int_env[name]
        return result


def interpret(source: str, inputs: dict[str, int],
              params: dict[str, int] | None = None,
              input_arrays: dict[str, dict[int, int]] | None = None,
              trace: bool = False) -> dict[str, int]:
    """Parse and execute a .bs source string. Returns output streams."""
    program = parse(source)
    interp = Interpreter(trace=trace)
    return interp.run(program, inputs, params, input_arrays)
