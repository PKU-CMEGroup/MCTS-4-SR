// include/imcts/evaluator/evaluator.hpp
#pragma once
#include <vector>
#include <span>
#include "imcts/core/types.hpp"
#include "imcts/core/symbol.hpp"
#include "imcts/core/bridge.hpp"
#include "imcts/core/dataset.hpp"
#include "imcts/eval/interpreter.hpp"

namespace imcts {

struct EvaluatorConfig {
    std::vector<std::vector<float>> x_cols;   // shape: [n_vars][n_samples]
    std::vector<float>              y;         // shape: [n_samples]
    int lm_iterations = 100;
};

class Evaluator {
public:
    Evaluator(const PrimitiveSet& pset, const EvaluatorConfig& cfg);

    float evaluate(std::span<uint8_t const> prefix, RandomGenerator& rng);
    Tree  build_optimized_tree(std::span<uint8_t const> prefix);

    int  cache_hits() const { return cache_hits_; }
    void cache_clear() { bridge_.cache_clear(); cache_hits_ = 0; }

private:
    float compute_reward(const Eigen::VectorXd& y_pred) const;

    Dataset dataset_;
    Range   train_range_;
    Bridge  bridge_;
    Tree    tree_workspace_;
    InterpreterWorkspace workspace_;
    InterpreterWorkspace optimizer_workspace_;
    Eigen::VectorXd target_values_;
    double  sigma_;
    int     lm_iterations_;
    int     cache_hits_ = 0;
};

} // namespace imcts
