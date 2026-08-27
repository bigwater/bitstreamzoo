#pragma once

#include "ast.h"
#include "liveness_info.h"
#include <string>
#include <unordered_map>

// Analyze liveness of stream variables directly from the parsed AST.
// O(statements) — no loop unrolling.
//
// Array elements (tracked via arrays_) are excluded;
// only scalar stream variables go through var_to_reg.
LivenessInfo analyze_liveness_ast(
    const bs::Program& prog,
    const std::unordered_map<std::string, int64_t>& params);

// Count dynamic ops by walking AST.
// Multiplies loop body ops by bounds. Returns int64_t for large programs.
int64_t count_ops_ast(const bs::Program& prog,
                      const std::unordered_map<std::string, int64_t>& params);
