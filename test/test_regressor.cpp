// test/test_regressor.cpp
#include <catch2/catch_test_macros.hpp>
#include "imcts/regressor.hpp"
#include <cmath>

TEST_CASE("Regressor finds x0^2 in reasonable evaluations") {
    int n = 20;
    std::vector<float> x0(n), y(n);
    for (int i = 0; i < n; i++) {
        x0[i] = static_cast<float>(i) / 10.0f;
        y[i]  = x0[i] * x0[i];
    }

    imcts::RegressorConfig cfg;
    cfg.ops         = {"+", "-", "*", "/", "sin"};
    cfg.max_depth   = 5;
    cfg.K           = 100;
    cfg.max_evals   = 2000;
    cfg.lm_iterations = 30;

    imcts::Regressor reg({{x0}}, y, cfg);
    auto result = reg.fit(/*seed=*/42);

    REQUIRE(result.best_reward > 0.90f);
    REQUIRE(result.n_evals > 0);
    REQUIRE_FALSE(result.best_path.empty());
    REQUIRE_FALSE(result.expression.empty());
}

TEST_CASE("Regressor returns optimized coefficients and formatted expression") {
    int n = 30;
    std::vector<float> x0(n), y(n);
    for (int i = 0; i < n; i++) {
        x0[i] = static_cast<float>(i) / 10.0f;
        y[i]  = 2.5f * x0[i] + 1.0f;
    }

    imcts::RegressorConfig cfg;
    cfg.ops = {"+", "-", "*", "/", "R"};
    cfg.max_depth = 5;
    cfg.K = 100;
    cfg.max_constants = 3;
    cfg.max_evals = 5000;
    cfg.lm_iterations = 30;

    imcts::Regressor reg({{x0}}, y, cfg);
    auto result = reg.fit(/*seed=*/123);

    REQUIRE_FALSE(result.best_path.empty());
    REQUIRE_FALSE(result.expression.empty());
    REQUIRE(result.expression.find("x0") != std::string::npos);
    REQUIRE(result.best_coefficients.size() <= 3);
}
