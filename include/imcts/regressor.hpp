// include/imcts/regressor.hpp
#pragma once
#include <vector>
#include <string>
#include <memory>
#include <optional>
#include "imcts/core/types.hpp"
#include "imcts/core/symbol.hpp"
#include "imcts/evaluator/evaluator.hpp"
#include "imcts/gp/gp_manager.hpp"
#include "imcts/mcts/mcts.hpp"

namespace imcts {

struct RegressorConfig {
    std::vector<std::string> ops = {"+", "-", "*", "/", "sin", "cos", "exp", "log"};
    int   max_depth        = 6;
    int   K                = 500;
    float c                = 4.0f;
    float gamma            = 0.5f;
    float gp_rate          = 0.2f;
    float mutation_rate    = 0.1f;
    float exploration_rate = 0.2f;
    int   max_unary        = 999;
    int   max_constants    = 999;
    int   lm_iterations    = 100;
    int   max_evals        = 2000000;
    float succ_error_tol   = 1e-6f;
};

struct FitResult {
    std::vector<uint8_t> best_path;
    std::vector<Scalar>  best_coefficients;
    std::string          expression;
    float                best_reward;
    int                  n_evals;
};

class Regressor {
public:
    Regressor(std::vector<std::vector<float>> x_cols,
              std::vector<float> y,
              RegressorConfig cfg = {});

    FitResult fit(std::optional<uint64_t> seed = std::nullopt);

    const PrimitiveSet& pset() const { return *pset_; }

private:
    RegressorConfig                cfg_;
    std::unique_ptr<PrimitiveSet>  pset_;
    Evaluator       evaluator_;
    GPManager       gp_manager_;
};

} // namespace imcts
