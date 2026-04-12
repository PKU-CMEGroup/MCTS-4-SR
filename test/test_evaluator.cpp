// test/test_evaluator.cpp
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "imcts/core/symbol.hpp"
#include "imcts/evaluator/evaluator.hpp"
#include <cmath>

static imcts::EvaluatorConfig make_linear_config() {
    int n = 20;
    std::vector<std::vector<float>> x_cols = {
        std::vector<float>(n),
        std::vector<float>(n)
    };
    std::vector<float> y(n);
    for (int i = 0; i < n; i++) {
        x_cols[0][i] = static_cast<float>(i);
        x_cols[1][i] = 1.0f;
        y[i]         = x_cols[0][i] + x_cols[1][i];
    }
    return {x_cols, y, /*lm_iterations=*/50};
}

TEST_CASE("Evaluator: x0+x1 expression gets reward close to 1") {
    auto pset = imcts::make_primitive_set({"+", "-", "*"}, 2);
    auto cfg  = make_linear_config();
    imcts::Evaluator eval(pset, cfg);

    std::vector<uint8_t> prefix = {
        pset.op_index("+"),
        pset.op_index("x0"),
        pset.op_index("x1")
    };
    imcts::RandomGenerator rng{42};
    float reward = eval.evaluate(prefix, rng);
    REQUIRE(reward > 0.99f);
}

TEST_CASE("Evaluator: cache deduplication") {
    auto pset = imcts::make_primitive_set({"+", "-"}, 1);
    std::vector<std::vector<float>> x = {std::vector<float>(5, 1.0f)};
    std::vector<float> y(5, 1.0f);
    imcts::EvaluatorConfig cfg{x, y, 10};
    imcts::Evaluator eval(pset, cfg);

    std::vector<uint8_t> prefix = {pset.op_index("x0")};
    imcts::RandomGenerator rng{42};
    float r1 = eval.evaluate(prefix, rng);
    float r2 = eval.evaluate(prefix, rng);
    REQUIRE(r1 == Catch::Approx(r2));
    REQUIRE(eval.cache_hits() == 1);
}

TEST_CASE("Evaluator: invalid numeric expressions return zero reward") {
    auto pset = imcts::make_primitive_set({"/", "log"}, 1);
    std::vector<std::vector<float>> x = {std::vector<float>(5, 0.0f)};
    std::vector<float> y(5, 1.0f);
    imcts::EvaluatorConfig cfg{x, y, 10};
    imcts::Evaluator eval(pset, cfg);
    imcts::RandomGenerator rng{42};

    SECTION("division by zero") {
        std::vector<uint8_t> prefix = {
            pset.op_index("/"),
            pset.op_index("x0"),
            pset.op_index("x0")
        };
        REQUIRE(eval.evaluate(prefix, rng) == Catch::Approx(0.0f));
    }

    SECTION("log of negative input") {
        x[0] = std::vector<float>(5, -1.0f);
        imcts::Evaluator neg_eval(pset, {x, y, 10});
        std::vector<uint8_t> prefix = {
            pset.op_index("log"),
            pset.op_index("x0")
        };
        REQUIRE(neg_eval.evaluate(prefix, rng) == Catch::Approx(0.0f));
    }
}
