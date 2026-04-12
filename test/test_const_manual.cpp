#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <vector>

#include "imcts/core/bridge.hpp"
#include "imcts/core/dataset.hpp"
#include "imcts/eval/interpreter.hpp"
#include "imcts/eval/optimizer.hpp"
#include "imcts/regressor.hpp"

TEST_CASE("Coefficient optimizer fits affine coefficients") {
    std::vector<imcts::Node> nodes;

    imcts::Node x0(imcts::NodeType::Variable);
    x0.var_index = 0;
    x0.optimize = false;
    nodes.push_back(x0);

    imcts::Node c1(imcts::NodeType::Constant);
    c1.value = 1.0;
    c1.optimize = true;
    nodes.push_back(c1);

    imcts::Node mul(imcts::NodeType::Mul);
    nodes.push_back(mul);

    imcts::Node c2(imcts::NodeType::Constant);
    c2.value = 1.0;
    c2.optimize = true;
    nodes.push_back(c2);

    imcts::Node add(imcts::NodeType::Add);
    nodes.push_back(add);

    imcts::Tree tree(std::move(nodes));
    tree.update_lengths();

    int n = 50;
    std::vector<std::vector<float>> x_cols(1, std::vector<float>(n));
    std::vector<float> y(n);
    for (int i = 0; i < n; i++) {
        x_cols[0][i] = static_cast<float>(i) / 10.0f;
        y[i] = 2.5f * x_cols[0][i] + 3.0f;
    }

    imcts::Dataset ds(x_cols, y);
    imcts::Range range{0, static_cast<std::size_t>(n)};

    auto opt_tree = imcts::CoefficientOptimizer::optimize(tree, ds, range, 50);
    auto coeffs = opt_tree.get_coefficients();
    REQUIRE(coeffs.size() == 2);

    auto pred = imcts::Interpreter::evaluate(opt_tree, ds, range);
    double mse = 0.0;
    for (int i = 0; i < n; i++) {
        const double diff = pred(i) - ds.y(i);
        mse += diff * diff;
    }
    mse /= n;

    REQUIRE(coeffs[0] == Catch::Approx(2.5).margin(1e-3));
    REQUIRE(coeffs[1] == Catch::Approx(3.0).margin(1e-3));
    REQUIRE(mse < 1e-10);
}

TEST_CASE("Tree structure hash ignores coefficient values") {
    auto pset = imcts::make_primitive_set({"+", "*", "R"}, 1);
    imcts::Bridge bridge(pset);

    std::vector<uint8_t> prefix = {
        pset.op_index("+"),
        pset.op_index("R"),
        pset.op_index("*"),
        pset.op_index("R"),
        pset.op_index("x0")
    };

    auto tree = bridge.to_tree(prefix);
    const auto hash_before = tree.structure_hash();
    tree.set_coefficients({99.0, 42.0});
    const auto hash_after = tree.structure_hash();

    REQUIRE(hash_before == hash_after);
}

TEST_CASE("Regressor can recover a simple affine expression with constants") {
    int n = 50;
    std::vector<float> x0(n);
    std::vector<float> y(n);
    for (int i = 0; i < n; i++) {
        x0[i] = static_cast<float>(i) / 10.0f;
        y[i] = 3.14f * x0[i] + 1.0f;
    }

    imcts::RegressorConfig cfg;
    cfg.ops = {"+", "-", "*", "/", "R"};
    cfg.max_depth = 5;
    cfg.K = 100;
    cfg.max_constants = 3;
    cfg.max_evals = 10000;
    cfg.lm_iterations = 30;

    imcts::Regressor reg({{x0}}, y, cfg);
    const auto result = reg.fit(123);

    REQUIRE(result.best_reward > 0.99f);
    REQUIRE(result.n_evals > 0);
    REQUIRE(result.expression.find("x0") != std::string::npos);
}
