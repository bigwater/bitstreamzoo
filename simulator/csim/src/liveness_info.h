#pragma once

#include <string>
#include <unordered_map>

// Liveness analysis result (AST-direct path).
struct LivenessInfo {
    int n_variables = 0;   // total unique stream variable names
    int max_live = 0;      // max simultaneously live at any program point
    int n_registers = 0;   // registers needed with reuse (== max_live)

    // variable name -> register ID (for buffer reuse)
    std::unordered_map<std::string, int> var_to_reg;

    // variable name -> last instruction index where the variable is used
    std::unordered_map<std::string, int> last_use;
};
