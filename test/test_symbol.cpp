// test/test_symbol.cpp
#include <catch2/catch_test_macros.hpp>
#include "imcts/core/symbol.hpp"

TEST_CASE("make_primitive_set basic ops") {
    auto pset = imcts::make_primitive_set(
        {"+", "-", "*", "/", "sin", "cos", "exp", "log"},
        /*n_vars=*/2
    );

    // 2 vars + 8 ops = 10 symbols
    REQUIRE(pset.symbols.size() == 10);

    // + is binary
    REQUIRE(pset.symbols[pset.op_index("+")].arity == 2);
    // sin is unary
    REQUIRE(pset.symbols[pset.op_index("sin")].arity == 1);
    // x0 is leaf
    auto x0_idx = pset.op_index("x0");
    REQUIRE(pset.symbols[x0_idx].arity == 0);
    REQUIRE(pset.symbols[x0_idx].var_index == 0);
    // leaf_indices contains x0, x1
    REQUIRE(pset.leaf_indices.size() == 2);
}

TEST_CASE("PrimitiveSet inner indices") {
    auto pset = imcts::make_primitive_set({"+", "sin"}, /*n_vars=*/1);
    // inner: + (arity=2), sin (arity=1)
    REQUIRE(pset.inner_indices.size() == 2);
}
