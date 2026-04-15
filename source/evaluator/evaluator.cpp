// source/evaluator/evaluator.cpp
#include "imcts/evaluator/evaluator.hpp"
#include "imcts/core/profiling.hpp"
#include "imcts/eval/interpreter.hpp"
#include "imcts/eval/optimizer.hpp"
#include <chrono>
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
#ifdef IMCTS_ENABLE_PROFILING
    const auto t_total0 = std::chrono::steady_clock::now();
    std::uint64_t to_tree_ns = 0;
    std::uint64_t hash_lookup_ns = 0;
    std::uint64_t optimize_ns = 0;
    std::uint64_t forward_eval_ns = 0;
    std::uint64_t reward_ns = 0;
    std::uint64_t cache_store_ns = 0;
#endif

#ifdef IMCTS_ENABLE_PROFILING
    const auto t_to_tree0 = std::chrono::steady_clock::now();
#endif
    bridge_.to_tree(prefix, tree_workspace_);
#ifdef IMCTS_ENABLE_PROFILING
    to_tree_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t_to_tree0).count());
    const auto t_lookup0 = std::chrono::steady_clock::now();
#endif
    auto hash = tree_workspace_.structure_hash();
    const bool has_constants = tree_workspace_.num_coefficients() > 0;

    if (auto cached = bridge_.cache_get(hash); cached.has_value()) {
        ++cache_hits_;
#ifdef IMCTS_ENABLE_PROFILING
        hash_lookup_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t_lookup0).count());
        profiling::record_evaluator_call(
            static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - t_total0).count()),
            to_tree_ns,
            hash_lookup_ns,
            optimize_ns,
            forward_eval_ns,
            reward_ns,
            cache_store_ns,
            true,
            has_constants,
            static_cast<std::uint64_t>(prefix.size()));
#endif
        return *cached;
    }
#ifdef IMCTS_ENABLE_PROFILING
    hash_lookup_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t_lookup0).count());
#endif

#ifdef IMCTS_ENABLE_PROFILING
    const auto t_eval0 = std::chrono::steady_clock::now();
#endif
    if (has_constants) {
#ifdef IMCTS_ENABLE_PROFILING
        const auto t_opt0 = std::chrono::steady_clock::now();
#endif
        auto tree = CoefficientOptimizer::optimize(
            tree_workspace_, dataset_, train_range_, optimizer_workspace_, lm_iterations_);
#ifdef IMCTS_ENABLE_PROFILING
        optimize_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t_opt0).count());
#endif
        Interpreter::evaluate(tree, dataset_, train_range_, workspace_);
    } else {
        Interpreter::evaluate(tree_workspace_, dataset_, train_range_, workspace_);
    }
#ifdef IMCTS_ENABLE_PROFILING
    forward_eval_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t_eval0).count());
    const auto t_reward0 = std::chrono::steady_clock::now();
#endif

    float reward = compute_reward(workspace_.result());
#ifdef IMCTS_ENABLE_PROFILING
    reward_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t_reward0).count());
    const auto t_store0 = std::chrono::steady_clock::now();
#endif
    bridge_.cache_put(hash, reward);
#ifdef IMCTS_ENABLE_PROFILING
    cache_store_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t_store0).count());
    profiling::record_evaluator_call(
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t_total0).count()),
        to_tree_ns,
        hash_lookup_ns,
        optimize_ns,
        forward_eval_ns,
        reward_ns,
        cache_store_ns,
        false,
        has_constants,
        static_cast<std::uint64_t>(prefix.size()));
#endif
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
