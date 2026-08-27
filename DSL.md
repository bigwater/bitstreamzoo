# Bitstream Program DSL Specification

## 1. Overview

This document specifies the **Bitstream Program DSL**, a domain-specific language for expressing data-parallel bitwise computations. Every variable of type `stream` represents an **unbounded bitset**, a sequence of bits of arbitrary length. 

The DSL originates from the [Parabix](https://github.com/parabix) framework for text processing and extends the original grammar to the application classes the benchmark suite actually exercises: bitsliced cryptography, gate-level circuit simulation, multi-pattern regex and string matching, JSON structural indexing, binary neural networks, polar error-correcting codes, bit-sliced database scans, and bioinformatics epistasis screens.

### Design Principles

1. **Streams are first-class.** Every interesting value is an unbounded bitset.
2. **Integers are second-class.** Integers exist only to parameterize shift amounts, loop bounds, and array indices, never as the primary data.
3. **No data-dependent addressing.** Array indices and shift amounts are always computable from parameters and loop variables, never from stream content.
4. **Explicit I/O.** Inputs and outputs are declared, making dataflow visible and enabling compiler analysis.

---

## 2. Types

### `stream` --Unbounded Bitset

The core type. A `stream` is a sequence of bits indexed from 0 (LSB) upward with no fixed upper bound. 

- Default value: `ZERO` (all bits 0).
- Bitwise operations apply pointwise to every bit position.

### `int` --Bounded Integer

Integers are used for loop bounds, shift amounts, array indices, and popcount results. The Python interpreter uses unbounded Python integers; the C++ interpreter uses `int64_t`.

Used for:
- **Shift amounts**: `x << k` where `k` is an `int`
- **Loop bounds**: `for i in 0..N`
- **Array indices**: `bits[i]`
- **Reduction results**: `popcount(s)` produces an `int`
- **Output values**: `output int n` declares an integer output

Integer values come from: compile-time/runtime constants (declared via `param`), loop induction variables, `popcount()` reductions, and mutable integer locals (`int x = expr`). Integer locals can be assigned with integer expressions including `popcount()`.

---

## 3. Declarations

```
decl ::= "input" "stream" id
       | "input" "stream" id "[" int_expr "]"
       | "output" "stream" id
       | "output" "stream" id "[" int_expr "]"
       | "output" "int" id
       | "param" "int" id
       | "stream" id ("=" expr)?
       | "stream" id "[" int_expr "]"
       | "int" id "=" int_expr
```

### Input / Output

```
input stream grid            // externally provided stream
input stream bits[8]         // externally provided stream array
output stream next           // result stream, written by the program
output stream out_bits[8]    // result stream array
output int count             // result integer (e.g., from popcount)
```

Inputs are read-only. Stream outputs must be assigned exactly once on every control-flow path. Integer outputs are initialized to 0 and assigned via integer expressions. These are program contracts: the reference parsers enforce the 3-address expression form but do not statically check read-only inputs or exactly-once output assignment.

### Parameters

```
param int W             // runtime integer constant (e.g., grid width)
```

Parameters are immutable integers supplied by the caller before execution. They may appear in shift amounts, loop bounds, array indices, and integer arithmetic expressions.

### Local Variables

```
stream temp             // uninitialized (defaults to ZERO)
stream mask = ONES      // initialized
int count = 0           // integer local variable
```

Local stream variables are mutable and scoped to the enclosing block. Integer locals (`int x = expr`) can store results of integer expressions including `popcount()`.

### Stream Arrays

```
stream bits[32]         // array of 32 streams
```

Stream arrays are arrays of independent streams: `bits[i]` returns the
**entire i-th stream**, not the i-th bit of a stream.
There is no operator to access individual bit positions within a stream.
The array size must be an `int_expr` computable from parameters.
Elements are accessed by `int_expr` index: `bits[i]`, `bits[W - 1]`.

---

## 4. Expressions

### Stream Expressions (3-Address)

In 3-address form, each assignment RHS contains **at most one operator** applied to
**primaries** (variables, constants, or array accesses). The full expression grammar
is parsed for compatibility, but the 3-address validator rejects nested expressions.

```
// Allowed forms (one operator, primary operands):
simple_expr ::= primary                    // copy / load
              | "~" primary               // unary NOT
              | primary "&" primary       // bitwise AND
              | primary "|" primary       // bitwise OR
              | primary "^" primary       // bitwise XOR
              | primary "+" primary       // stream addition
              | primary "<<" int_expr     // left shift
              | primary ">>" int_expr     // right shift

primary     ::= id                        // stream variable
              | id "[" int_expr "]"       // stream array element
              | "ZERO"                    // all-zeros stream
              | "ONES"                    // all-ones stream
              | "(" stream_expr ")"       // parenthesized expression
```

**Operator precedence** (for reference only -- 3-address form means each statement has at most one operator, so precedence never applies in practice):

| Precedence | Operators     | Associativity |
|------------|---------------|---------------|
| 1          | `~` (unary)   | right         |
| 2          | `<<`, `>>`    | left          |
| 3          | `&`           | left          |
| 4          | `+`           | left          |
| 5          | `^`           | left          |
| 6          | `|`           | left          |

Stream `+` is between `&` and `^` in precedence. It treats the two operand streams as binary integers (bit 0 = LSB) and produces their binary sum with carry propagation across bit positions.

### Integer Expressions

```
int_expr ::= int_literal
           | id                             // param, loop variable, or int local
           | int_expr "+" int_expr
           | int_expr "-" int_expr
           | int_expr "*" int_expr
           | "popcount" "(" expr ")"        // count set bits (stream → int)
           | "(" int_expr ")"
```

Integer arithmetic follows standard precedence (`*` before `+`/`-`). Integer expressions are evaluated to produce shift amounts, loop bounds, array indices, or output values.

`popcount(expr)` counts the number of set bits in a stream expression, producing an integer. This is a stream-to-int reduction --the only operation that crosses the stream/int boundary.

### Built-in Constants

| Name   | Type     | Value                              |
|--------|----------|------------------------------------|
| `ZERO` | `stream` | All bits 0                         |
| `ONES` | `stream` | All bits 1                         |

---

## 5. Statements (3-Address Form)

All statements must be in **3-address form**: each assignment has at most one operator,
and all operands must be primaries (variables, constants, or array accesses). This
restriction makes `.bs` files directly executable as IR --no lowering pass needed.

```
stmt ::= id "=" simple_expr                        // stream assignment
       | id "=" int_expr                           // int assignment (if id is int-typed)
       | id "[" int_expr "]" "=" simple_expr       // array element assignment
       | "if" "(" id ")" "{" stmt* "}"             // conditional (var condition)
       | "while" "(" id ")" "{" stmt* "}"          // loop (var condition)
       | "for" id "in" int_expr ".." int_expr "{" stmt* "}"  // counted loop

simple_expr ::= primary                            // copy / load
              | "~" primary                        // unary NOT
              | primary binop primary              // one binary op
              | primary shift_op int_expr          // one shift

primary ::= id | id "[" int_expr "]" | "ZERO" | "ONES" | "(" stream_expr ")"
```

### 3-Address Rules

1. **One operator per assignment.** `x = a ^ b` is valid; `x = a ^ b ^ c` is not.
   Break multi-op expressions into temporaries: `stream t = a ^ b; x = t ^ c`.
2. **Operands must be primaries.** `x = ~a` is valid; `x = ~(a & b)` is not.
3. **`if`/`while` conditions must be variables.** `if (flag) { ... }` is valid;
   `if (a & b) { ... }` is not. The parser accepts full expressions syntactically,
   but the validator rejects non-variable conditions.
4. **`popcount` argument must be a variable.** `n = popcount(x)` is valid;
   `n = popcount(a & b)` is not. The parser accepts full expressions syntactically,
   but the validator rejects non-variable arguments.

### Assignment

```
x = a ^ b           // assign XOR of a and b to x
bits[i] = x >> i     // assign to array element
```

The left-hand side is a stream variable or an array element. The right-hand side is a stream expression.

### Conditional (`if`)

```
if (mask) {
    x = x ^ key
}
```

The condition is a stream. The body executes if any bit in `mask` is `1` --i.e., `mask != ZERO` (OR-reduction). The body is **not masked** --all bit positions execute. This is the same semantics as `while`.

### While Loop

```
while (changed) {
    stream sl = label << 1
    stream sr = label >> 1
    label = label | sl
    label = label | sr
    changed = label ^ prev
    prev = label
}
```

The condition is a stream. The loop repeats while any bit in `changed` is `1` --i.e., `changed != ZERO` (OR-reduction, same nonzero test as `if`).

The `while` body is **not masked** --all bit positions execute every iteration (same as `if`). The loop terminates when every position has converged (condition stream is all zeros). This is useful for iterative convergence algorithms (e.g., connected component labeling).

### For Loop (Counted)

```
for i in 0..10 {
    stream shifted = a << i
    result = result ^ shifted
}
```

The loop variable `i` is an `int`, ranging from the first bound (inclusive) to the second bound (exclusive). The loop variable should not be reassigned within the body (not enforced by the parser).

---

## 6. Semantics

### Streams as Unbounded Bitsets

A stream `s` maps bit indices {0, 1, 2, ...} to {0, 1}. We write `s_i` for the i-th bit of stream `s`. This subscript notation is purely mathematical --the DSL has no operator to access individual bit positions. All operations apply pointwise:

- `(~s)_i = 1 - s_i`
- `(a & b)_i = a_i * b_i`
- `(a | b)_i = a_i + b_i - a_i * b_i`
- `(a ^ b)_i = (a_i + b_i) mod 2`

### Shift Semantics

- **Left shift** `s << k`: `r_i = s_(i-k)` if `i >= k`, else `0`. Bits shift toward higher indices; lower `k` bits become 0.
- **Right shift** `s >> k`: `r_i = s_(i+k)`. Bits shift toward lower indices; bits shifted past index 0 are discarded.

Shift amounts must be non-negative integer expressions within the stream length; behavior outside that range is unspecified, and the parser does not enforce the upper bound.

### Addition Semantics

`a + b` treats streams `a` and `b` as binary integers (bit 0 = LSB) and produces their binary sum with carry propagation across bit positions. This is the only operator that breaks purely pointwise semantics: carry at position `i` depends on positions `0..i-1`.

### Popcount Semantics

`popcount(s)` counts the number of set bits across all positions, producing an integer. This is the only operation that crosses the stream-to-int boundary.

### Execution over Finite Inputs

The unbounded bitset is the mathematical abstraction; a concrete execution supplies inputs of a finite length `N` (the test's `bitlength`), with every bit at index `>= N` equal to 0. Execution is bounded at `N`: `~s` produces ones only up to index `N-1`, and bits moved past index `N-1` by `<<` or by the carry of `+` are discarded, exactly as bits moved past index 0 by `>>` are. The C++ and CUDA backends are bounded by construction (streams are `ceil(N/64)` words); the Python interpreter applies the same truncation when given the bitlength. Boundedness is what makes a `while` fixed point terminate: a cursor advanced past end-of-input vanishes, so the condition stream reaches all-zero.

### Operation Counting

All three backends (Python, C++, CUDA) count the same set of operations:
- **Bitwise ops**: `&`, `|`, `^`, `~`, `<<`, `>>`, `+` (each counts as 1 op)
- **`popcount`**: counts as 1 op (stream-to-int reduction)
- **Control flow**: `if`/`while` condition checks (is-nonzero tests) are **not** counted as ops
- **Assignments, declarations, loop overhead**: not counted

The `op_count` reported by each backend should be identical for the same program and input.

### Conditional Semantics

For `if (cond) { body }`:
- **Branch decision**: OR-reduction of `cond` --the body executes if `cond != ZERO`.
- **No masking**: all bit positions execute the body. Same as `while`.

For `while (cond) { body }`:
- **Loop decision**: OR-reduction of `cond` --the loop repeats while `cond != ZERO`.
- **No masking**: all bit positions execute the body every iteration. The loop terminates when the entire stream is zero (all positions have converged).

### Array Semantics

`stream arr[N]` allocates `N` independent streams. `arr[i]` accesses the `i`-th stream (0-indexed). The index is an `int_expr`, so there are no gather/scatter operations.

---

## 7. Complete Grammar (3-Address Form)

All assignments are restricted to **3-address form**: at most one operator per
statement, with all operands being primaries. This makes `.bs` files directly
executable as IR --no lowering pass needed. The Python, C++ SIMD, and CUDA
backends all parse `.bs` files directly.

```
program     ::= decl* stmt*

decl        ::= "input" "stream" id
              | "input" "stream" id "[" int_expr "]"
              | "output" "stream" id
              | "output" "stream" id "[" int_expr "]"
              | "output" "int" id
              | "param" "int" id
              | "stream" id ("=" simple_expr)?
              | "stream" id "[" int_expr "]"
              | "int" id "=" int_expr

stmt        ::= id "=" simple_expr
              | id "=" int_expr                         // when id is int-typed
              | id "[" int_expr "]" "=" simple_expr
              | "if" "(" id ")" "{" stmt* "}"           // condition must be a variable
              | "while" "(" id ")" "{" stmt* "}"        // condition must be a variable
              | "for" id "in" int_expr ".." int_expr "{" stmt* "}"

simple_expr ::= primary                                 // copy / load
              | "~" primary                             // unary NOT
              | primary binop primary                   // one binary op
              | primary shift_op int_expr               // one shift

primary     ::= id
              | id "[" int_expr "]"
              | "ZERO" | "ONES"

binop       ::= "&" | "|" | "^" | "+"
shift_op    ::= "<<" | ">>"

int_expr    ::= int_literal
              | id
              | int_expr arith_op int_expr
              | "popcount" "(" id ")"                   // arg must be a variable
              | "(" int_expr ")"

arith_op    ::= "+" | "-" | "*"
```

### Lexical Conventions

- **Identifiers**: `[a-zA-Z_][a-zA-Z0-9_]*`
- **Integer literals**: `[0-9]+`
- **Comments**: `//` to end of line
- **Whitespace**: insignificant (except for separating tokens)
- **Reserved words**: `input`, `output`, `param`, `stream`, `int`, `if`, `while`, `for`, `in`, `ZERO`, `ONES`, `popcount`

---

## 8. Implementation Notes

### Variable Scoping

All variables (stream and integer) have function-level scope in both interpreters.
Loop variables persist after the loop ends.
If two nested loops use the same variable name, the inner loop overwrites the outer loop's value (no shadowing).

### Array Bounds

Array access is unchecked. Reading an unwritten index returns ZERO.
Writing to any index silently succeeds (arrays use sparse storage internally).

### Shift Amounts

A shift amount must be a non-negative integer within the stream
length (Section 6, Shift Semantics). This is part of the IR contract:
the Python backend is normative inside the range, and every benchmark
program stays inside it. A shift outside the range is outside the IR
contract, analogous to signed overflow in C. Backends do not check it
and may differ (Python uses its integer semantics; C++ casts to
`size_t`), so portable external programs must respect the range.
