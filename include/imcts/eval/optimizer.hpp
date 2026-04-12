// include/imcts/eval/optimizer.hpp
#pragma once
#include "imcts/core/tree.hpp"
#include "imcts/core/dataset.hpp"
#include "imcts/core/types.hpp"
#include "imcts/eval/interpreter.hpp"

namespace imcts {

class CoefficientOptimizer {
public:
    // Optimize constants in tree using Levenberg-Marquardt.
    // Returns a new tree with optimized coefficient values.
    static Tree optimize(const Tree& tree, const Dataset& ds, Range range,
                         int max_iter = 10);
    static Tree optimize(const Tree& tree, const Dataset& ds, Range range,
                         InterpreterWorkspace& workspace, int max_iter);
};

} // namespace imcts
