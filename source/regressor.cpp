// source/regressor.cpp
#include "imcts/regressor.hpp"
#include <random>

namespace imcts {

Regressor::Regressor(std::vector<std::vector<float>> x_cols,
                     std::vector<float> y,
                     RegressorConfig cfg)
    : cfg_(cfg)
    , pset_(std::make_unique<PrimitiveSet>(make_primitive_set(cfg.ops, static_cast<int>(x_cols.size()))))
    , evaluator_(*pset_, EvaluatorConfig{x_cols, y, cfg.lm_iterations})
    , gp_manager_(*pset_)
{}

FitResult Regressor::fit(std::optional<uint64_t> seed) {
    uint64_t s = seed.value_or(std::random_device{}());
    RandomGenerator rng{s};

    MCTSConfig mcts_cfg{
        .K                = cfg_.K,
        .c                = cfg_.c,
        .gamma            = cfg_.gamma,
        .gp_rate          = cfg_.gp_rate,
        .mutation_rate    = cfg_.mutation_rate,
        .exploration_rate = cfg_.exploration_rate,
        .succ_error_tol   = cfg_.succ_error_tol,
    };
    MCTS mcts(*pset_, evaluator_, gp_manager_, mcts_cfg);
    ExpTree tree(*pset_, cfg_.max_depth, cfg_.max_unary, cfg_.max_constants);

    while (mcts.count() < cfg_.max_evals) {
        float best = mcts.search(tree, rng);
        if (1.0f - best < cfg_.succ_error_tol) break;
    }

    auto best_path = mcts.best_path();
    std::vector<Scalar> best_coefficients;
    std::string expression;
    if (!best_path.empty()) {
        auto best_tree = evaluator_.build_optimized_tree(best_path);
        best_coefficients = best_tree.get_coefficients();
        expression = best_tree.to_string();
    }

    return FitResult{
        std::move(best_path),
        std::move(best_coefficients),
        std::move(expression),
        mcts.best_reward(),
        mcts.count()
    };
}

} // namespace imcts
