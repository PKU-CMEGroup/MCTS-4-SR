// test/test_mcts.cpp
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "imcts/core/symbol.hpp"
#include "imcts/core/exp_queue.hpp"
#include "imcts/mcts/node.hpp"

TEST_CASE("MCTSNode UCB-extreme formula") {
    imcts::MCTSNode parent(nullptr, 255, 5);
    parent.visits = 10;

    imcts::MCTSNode child(&parent, 0, 5);
    child.visits = 2;
    child.path_queue.append({0}, 0.5f);

    float ucb = child.ucb(4.0f, 0.5f);
    // expected: 0.5 + sqrt(4 * log(10) / 2) ≈ 0.5 + 2.146
    REQUIRE(ucb > 2.0f);
    REQUIRE(ucb < 3.5f);
}

TEST_CASE("MCTSNode is_leaf: no children = leaf") {
    imcts::MCTSNode root(nullptr, 255, 5);
    REQUIRE(root.is_leaf());
}

TEST_CASE("MCTSNode backpropagate updates path_queue upward") {
    imcts::MCTSNode root(nullptr, 255, 3);
    imcts::MCTSNode child(&root, 0, 3);

    child.backpropagate({1, 2}, 0.8f);
    REQUIRE(child.path_queue.best().reward == Catch::Approx(0.8f));
    REQUIRE(root.path_queue.best().reward == Catch::Approx(0.8f));
    REQUIRE(root.path_queue.best().path.size() == 3);
    REQUIRE(root.path_queue.best().path[0] == 0);
}

// Task 9 integration test
#include "imcts/evaluator/evaluator.hpp"
#include "imcts/gp/gp_manager.hpp"
#include "imcts/mcts/mcts.hpp"

TEST_CASE("MCTS finds x0 expression in <500 evals") {
    int n = 20;
    std::vector<float> x0(n), y(n);
    for (int i = 0; i < n; i++) { x0[i] = static_cast<float>(i); y[i] = x0[i]; }
    imcts::EvaluatorConfig cfg{{x0}, y, 30};

    auto pset = imcts::make_primitive_set({"+", "-", "*", "sin"}, 1);
    imcts::Evaluator evaluator(pset, cfg);
    imcts::GPManager gp_mgr(pset);
    imcts::MCTSConfig mcts_cfg{
        .K = 50, .c = 4.0f, .gamma = 0.5f,
        .gp_rate = 0.2f, .mutation_rate = 0.2f, .exploration_rate = 0.2f
    };

    imcts::MCTS mcts(pset, evaluator, gp_mgr, mcts_cfg);
    imcts::RandomGenerator rng{42};

    imcts::ExpTree tree(pset, 4, 2, 0);
    float best = -1.0f;
    for (int i = 0; i < 500; i++) {
        best = mcts.search(tree, rng);
        if (best > 0.999f) break;
    }
    REQUIRE(best > 0.95f);
}

TEST_CASE("MCTS total nodes respect configured cap") {
    int n = 20;
    std::vector<float> x0(n), y(n);
    for (int i = 0; i < n; i++) { x0[i] = static_cast<float>(i); y[i] = x0[i]; }
    imcts::EvaluatorConfig cfg{{x0}, y, 30};

    auto pset = imcts::make_primitive_set({"+", "-", "*", "sin"}, 1);
    imcts::Evaluator evaluator(pset, cfg);
    imcts::GPManager gp_mgr(pset);
    imcts::MCTSConfig mcts_cfg{
        .K = 20,
        .max_tree_nodes = 25,
        .c = 4.0f,
        .gamma = 0.5f,
        .gp_rate = 0.2f,
        .mutation_rate = 0.2f,
        .exploration_rate = 0.2f
    };

    imcts::MCTS mcts(pset, evaluator, gp_mgr, mcts_cfg);
    imcts::RandomGenerator rng{7};
    imcts::ExpTree tree(pset, 4, 2, 0);

    for (int i = 0; i < 200; ++i) {
        (void)mcts.search(tree, rng);
    }

    REQUIRE(mcts.total_nodes() <= mcts_cfg.max_tree_nodes);
}
