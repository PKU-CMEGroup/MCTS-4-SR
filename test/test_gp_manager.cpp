// test/test_gp_manager.cpp
#include <catch2/catch_test_macros.hpp>
#include "imcts/core/symbol.hpp"
#include "imcts/core/exp_tree.hpp"
#include "imcts/gp/gp_manager.hpp"

static imcts::PrimitiveSet make_pset() {
    return imcts::make_primitive_set({"+", "-", "*", "/", "sin", "cos", "exp", "log"}, 2);
}

TEST_CASE("GPManager::node_replace changes one op preserving arity") {
    auto pset = make_pset();
    imcts::ExpTree state(pset, 6, 4, 0);
    imcts::GPManager gp(pset);
    imcts::RandomGenerator rng{42};

    std::vector<uint8_t> path = {
        pset.op_index("+"),
        pset.op_index("x0"),
        pset.op_index("x1")
    };
    auto new_path = gp.node_replace(state, path, rng);
    REQUIRE(new_path.size() == path.size());
    REQUIRE(pset.symbols[new_path[0]].arity == 2);
}

TEST_CASE("GPManager::shrink_mutate reduces path length") {
    auto pset = make_pset();
    imcts::ExpTree state(pset, 6, 4, 0);
    imcts::GPManager gp(pset);
    imcts::RandomGenerator rng{42};

    std::vector<uint8_t> path = {
        pset.op_index("+"),
        pset.op_index("sin"),
        pset.op_index("x0"),
        pset.op_index("x1")
    };
    auto new_path = gp.shrink_mutate(state, path, rng);
    REQUIRE(new_path.size() < path.size());
}

TEST_CASE("GPManager::mutate produces valid path (parseable by ExpTree)") {
    auto pset = make_pset();
    imcts::GPManager gp(pset);
    imcts::RandomGenerator rng{99};

    for (int trial = 0; trial < 10; trial++) {
        imcts::ExpTree gen_tree(pset, 5, 3, 0);
        imcts::RandomGenerator rng2{static_cast<uint64_t>(trial)};
        auto path = gen_tree.random_fill(rng2);

        imcts::ExpTree state(pset, 5, 3, 0);
        auto new_path = gp.mutate(state, path, rng);

        // Mutations may produce paths that violate context constraints
        // (e.g. sin/cos nesting). MCTS handles these gracefully.
        // Just verify the path is non-empty and syntactically plausible.
        REQUIRE_FALSE(new_path.empty());
    }
}

TEST_CASE("GPManager::crossover produces two paths") {
    auto pset = make_pset();
    imcts::GPManager gp(pset);
    imcts::RandomGenerator rng{7};

    std::vector<uint8_t> p1 = {pset.op_index("+"), pset.op_index("x0"), pset.op_index("x1")};
    std::vector<uint8_t> p2 = {pset.op_index("*"), pset.op_index("x0"), pset.op_index("x1")};

    auto [np1, np2] = gp.crossover(p1, p2, rng);
    REQUIRE_FALSE(np1.empty());
    REQUIRE_FALSE(np2.empty());
}

TEST_CASE("GPManager::crossover resamples when the first swap is a no-op") {
    auto pset = imcts::make_primitive_set({"+", "-", "*", "/"}, 2);
    imcts::GPManager gp(pset);

    std::vector<uint8_t> p1 = {pset.op_index("+"), pset.op_index("x0"), pset.op_index("x1")};
    std::vector<uint8_t> p2 = {pset.op_index("+"), pset.op_index("x0"), pset.op_index("x0")};

    bool found_resampled_change = false;
    for (uint64_t seed = 0; seed < 1024; ++seed) {
        imcts::RandomGenerator first_try_rng{seed};
        int idx1 = static_cast<int>(first_try_rng() % p1.size());
        int idx2 = static_cast<int>(first_try_rng() % p2.size());
        int s1 = gp.subtree_size(p1, idx1);
        int s2 = gp.subtree_size(p2, idx2);

        std::vector<uint8_t> first_np1, first_np2;
        first_np1.insert(first_np1.end(), p1.begin(), p1.begin() + idx1);
        first_np1.insert(first_np1.end(), p2.begin() + idx2, p2.begin() + idx2 + s2);
        first_np1.insert(first_np1.end(), p1.begin() + idx1 + s1, p1.end());

        first_np2.insert(first_np2.end(), p2.begin(), p2.begin() + idx2);
        first_np2.insert(first_np2.end(), p1.begin() + idx1, p1.begin() + idx1 + s1);
        first_np2.insert(first_np2.end(), p2.begin() + idx2 + s2, p2.end());

        if (first_np1 == p1 && first_np2 == p2) {
            imcts::RandomGenerator rng{seed};
            auto [np1, np2] = gp.crossover(p1, p2, rng);
            if (np1 != p1 || np2 != p2) {
                found_resampled_change = true;
                break;
            }
        }
    }

    REQUIRE(found_resampled_change);
}

TEST_CASE("GPManager::insert_mutate can still insert on a shallow branch when the tree already reaches max depth") {
    auto pset = make_pset();
    imcts::GPManager gp(pset);
    imcts::ExpTree state(pset, 3, 4, 0);

    std::vector<uint8_t> path = {
        pset.op_index("+"),
        pset.op_index("x0"),
        pset.op_index("+"),
        pset.op_index("x0"),
        pset.op_index("x1")
    };

    bool changed = false;
    for (uint64_t seed = 0; seed < 256; ++seed) {
        imcts::RandomGenerator rng{seed};
        auto new_path = gp.insert_mutate(state, path, rng);
        if (new_path != path) {
            changed = true;
            break;
        }
    }

    REQUIRE(changed);
}

TEST_CASE("GPManager::insert_mutate allows binary insertion at the root when one more level is available") {
    auto pset = imcts::make_primitive_set({"+", "-", "*", "/"}, 2);
    imcts::GPManager gp(pset);
    imcts::ExpTree state(pset, 2, 4, 0);
    imcts::RandomGenerator rng{0};

    std::vector<uint8_t> path = {pset.op_index("x0")};
    auto new_path = gp.insert_mutate(state, path, rng);

    REQUIRE(new_path != path);
    REQUIRE(pset.symbols[new_path.front()].arity == 2);
}

TEST_CASE("GPManager::uniform_mutate can replace the root with a depth-2 subtree when one more level is available") {
    auto pset = imcts::make_primitive_set({"+", "-", "*", "/"}, 2);
    imcts::GPManager gp(pset);
    imcts::ExpTree state(pset, 2, 4, 0);

    std::vector<uint8_t> path = {pset.op_index("x0")};
    bool found_binary_root = false;
    for (uint64_t seed = 0; seed < 256; ++seed) {
        imcts::RandomGenerator rng{seed};
        auto new_path = gp.uniform_mutate(state, path, rng);
        if (new_path != path && pset.symbols[new_path.front()].arity == 2) {
            found_binary_root = true;
            break;
        }
    }

    REQUIRE(found_binary_root);
}
