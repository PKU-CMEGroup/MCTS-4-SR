#include <iostream>
#include <vector>
#include <cmath>
#include "imcts/regressor.hpp"
#include "imcts/core/bridge.hpp"
#include "imcts/eval/interpreter.hpp"
#include "imcts/eval/optimizer.hpp"
#include "imcts/core/dataset.hpp"

void test_optimizer_directly() {
    std::cout << "=== Test 1: Direct CoefficientOptimizer test ===\n";
    // Build tree for: R * x0 + R  (postfix: [x0, R, *, R, +])
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

    // Dataset: y = 2.5*x + 3.0
    int n = 50;
    std::vector<std::vector<float>> x_cols(1, std::vector<float>(n));
    std::vector<float> y(n);
    for (int i = 0; i < n; i++) {
        x_cols[0][i] = static_cast<float>(i) / 10.0f;
        y[i] = 2.5f * x_cols[0][i] + 3.0f;
    }
    imcts::Dataset ds(x_cols, y);
    imcts::Range range{0, static_cast<size_t>(n)};

    std::cout << "  Before opt: coeffs =";
    for (auto c : tree.get_coefficients()) std::cout << " " << c;
    std::cout << "\n";

    auto opt_tree = imcts::CoefficientOptimizer::optimize(tree, ds, range, 50);
    auto coeffs = opt_tree.get_coefficients();
    std::cout << "  After opt:  coeffs =";
    for (auto c : coeffs) std::cout << " " << c;
    std::cout << "\n";

    auto pred = imcts::Interpreter::evaluate(opt_tree, ds, range);
    double mse = 0;
    for (int i = 0; i < n; i++) {
        double d = pred(i) - ds.y(i);
        mse += d * d;
    }
    mse /= n;
    std::cout << "  MSE = " << mse << "\n";
    std::cout << "  Expected c1~2.5, c2~3.0, MSE~0\n";
    if (mse < 1e-10) std::cout << "  PASS\n"; else std::cout << "  FAIL\n";
}

void test_hash_consistency() {
    std::cout << "\n=== Test 2: Hash consistency after optimization ===\n";
    auto pset = imcts::make_primitive_set({"+", "*", "R"}, 1);
    imcts::Bridge bridge(pset);

    // prefix: + R * R x0  => R + R*x0
    std::vector<uint8_t> prefix = {
        pset.op_index("+"),
        pset.op_index("R"),
        pset.op_index("*"),
        pset.op_index("R"),
        pset.op_index("x0")
    };
    auto tree1 = bridge.to_tree(prefix);
    auto hash_before = tree1.structure_hash();

    // Modify constants
    tree1.set_coefficients({99.0, 42.0});
    auto hash_after = tree1.structure_hash();

    std::cout << "  hash before: " << hash_before << "\n";
    std::cout << "  hash after changing constants: " << hash_after << "\n";
    if (hash_before == hash_after) std::cout << "  PASS: hash stable\n";
    else std::cout << "  FAIL: hash changed!\n";
}

void test_full_regressor_with_constants() {
    std::cout << "\n=== Test 3: Full Regressor with constants ===\n";
    int n = 50;
    std::vector<float> x0(n), y(n);
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
    auto result = reg.fit(123);
    std::cout << "  best_reward: " << result.best_reward << "\n";
    std::cout << "  n_evals: " << result.n_evals << "\n";
    if (result.best_reward > 0.99f) std::cout << "  PASS\n";
    else std::cout << "  FAIL: reward too low\n";
}

int main() {
    test_optimizer_directly();
    test_hash_consistency();
    test_full_regressor_with_constants();
    std::cout << "\nAll tests completed.\n";
    return 0;
}
