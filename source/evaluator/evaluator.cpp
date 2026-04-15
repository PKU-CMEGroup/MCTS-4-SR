// source/evaluator/evaluator.cpp
#include "imcts/evaluator/evaluator.hpp"
#include "imcts/eval/interpreter.hpp"
#include "imcts/eval/optimizer.hpp"
#include <cmath>
#include <numeric>

namespace imcts {

Evaluator::Evaluator(const PrimitiveSet& pset, const EvaluatorConfig& cfg)
    : dataset_(cfg.x_cols, cfg.y)
    , train_range_{0, cfg.y.size()}
    , bridge_(pset)
    , lm_iterations_(cfg.lm_iterations)
{
    target_values_ = dataset_.y;

    double mean = target_values_.mean();
    double var = (target_values_.array() - mean).square().mean();
    sigma_ = std::sqrt(var);
    if (sigma_ < 1e-10) sigma_ = 1.0;
}

float Evaluator::evaluate(std::span<uint8_t const> prefix, RandomGenerator& /*rng*/)
{
    bridge_.to_tree(prefix, tree_workspace_);
    auto hash = tree_workspace_.structure_hash();

    if (auto cached = bridge_.cache_get(hash); cached.has_value()) {
        ++cache_hits_;
        return *cached;
    }

    if (tree_workspace_.num_coefficients() > 0) {
        auto tree = CoefficientOptimizer::optimize(
            tree_workspace_, dataset_, train_range_, optimizer_workspace_, lm_iterations_);
        Interpreter::evaluate(tree, dataset_, train_range_, workspace_);
    } else {
        Interpreter::evaluate(tree_workspace_, dataset_, train_range_, workspace_);
    }

    float reward = compute_reward(workspace_.result());
    bridge_.cache_put(hash, reward);
    return reward;
}

Tree Evaluator::build_optimized_tree(std::span<uint8_t const> prefix)
{
    bridge_.to_tree(prefix, tree_workspace_);
    if (tree_workspace_.num_coefficients() > 0) {
        tree_workspace_ = CoefficientOptimizer::optimize(
            tree_workspace_, dataset_, train_range_, optimizer_workspace_, lm_iterations_);
    }
    return tree_workspace_;
}

float Evaluator::compute_reward(const Eigen::VectorXd& y_pred) const
{
    if (y_pred.size() != target_values_.size()) return 0.0f;
    if (!y_pred.allFinite()) return 0.0f;

    double mse = (y_pred - target_values_).squaredNorm()
                 / static_cast<double>(y_pred.size());
    double nrmse = std::sqrt(mse) / sigma_;
    if (!std::isfinite(nrmse)) return 0.0f;
    return static_cast<float>(1.0 / (1.0 + nrmse));
}

} // namespace imcts
