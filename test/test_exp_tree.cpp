// test/test_exp_tree.cpp
#include <catch2/catch_test_macros.hpp>
#include "imcts/core/symbol.hpp"
#include "imcts/core/exp_tree.hpp"

TEST_CASE("ExpTree basic add x0 x1") {
    auto pset = imcts::make_primitive_set({"+", "sin"}, 2);
    imcts::ExpTree tree(pset, /*max_depth=*/4, /*max_unary=*/2, /*max_constants=*/0);

    REQUIRE(tree.is_empty());
    REQUIRE_FALSE(tree.is_terminal());

    uint8_t plus_idx = pset.op_index("+");
    tree.add_op(plus_idx);
    REQUIRE_FALSE(tree.is_terminal());

    uint8_t x0_idx = pset.op_index("x0");
    tree.add_op(x0_idx);
    REQUIRE_FALSE(tree.is_terminal());

    uint8_t x1_idx = pset.op_index("x1");
    tree.add_op(x1_idx);
    REQUIRE(tree.is_terminal());
    REQUIRE(tree.get_op_list().size() == 3);
}

// Match the Python implementation:
// once stack_len == max_depth - 1, only leaves are allowed.
TEST_CASE("ExpTree depth constraint: at max_depth only leaves allowed") {
    auto pset = imcts::make_primitive_set({"+", "sin"}, 1);
    imcts::ExpTree tree(pset, /*max_depth=*/2, 999, 0);

    // After adding the root, depth-1 children must be leaves.
    uint8_t plus_idx = pset.op_index("+");
    tree.add_op(plus_idx);

    const auto& avail = tree.available_ops();
    bool found_non_leaf = false;
    for (uint8_t idx : avail) {
        if (pset.symbols[idx].arity > 0) { found_non_leaf = true; break; }
    }
    REQUIRE_FALSE(found_non_leaf);
}

TEST_CASE("ExpTree with max_depth=1 only allows leaves at the root") {
    auto pset = imcts::make_primitive_set({"+", "sin"}, 1);
    uint8_t plus_idx = pset.op_index("+");
    uint8_t x0_idx   = pset.op_index("x0");

    imcts::ExpTree t(pset, 1, 0, 0);
    REQUIRE_THROWS(t.add_op(plus_idx));

    t.add_op(x0_idx);
    REQUIRE(t.is_terminal());
}

TEST_CASE("ExpTree random_fill produces terminal tree") {
    auto pset = imcts::make_primitive_set({"+", "-", "sin"}, 2);
    imcts::ExpTree tree(pset, 5, 3, 0);
    imcts::RandomGenerator rng{42};
    auto path = tree.random_fill(rng);
    REQUIRE(tree.is_terminal());
    REQUIRE_FALSE(path.empty());
}
