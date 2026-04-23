// test/test_bridge.cpp
#include <catch2/catch_test_macros.hpp>
#include "imcts/core/symbol.hpp"
#include "imcts/core/bridge.hpp"

TEST_CASE("Bridge converts [+, x0, x1] to Tree with 3 nodes") {
    auto pset = imcts::make_primitive_set({"+", "sin"}, 2);
    imcts::Bridge bridge(pset);

    std::vector<uint8_t> prefix = {
        pset.op_index("+"),
        pset.op_index("x0"),
        pset.op_index("x1")
    };
    auto tree = bridge.to_tree(prefix);
    REQUIRE(tree.length() == 3);
    REQUIRE(tree.nodes().back().type == imcts::NodeType::Add);
}

TEST_CASE("Bridge converts [sin, x0] to Tree with 2 nodes") {
    auto pset = imcts::make_primitive_set({"+", "sin"}, 1);
    imcts::Bridge bridge(pset);

    std::vector<uint8_t> prefix = {
        pset.op_index("sin"),
        pset.op_index("x0")
    };
    auto tree = bridge.to_tree(prefix);
    REQUIRE(tree.length() == 2);
    REQUIRE(tree.nodes().back().type == imcts::NodeType::Sin);
    REQUIRE(tree.nodes()[0].type == imcts::NodeType::Variable);
}

TEST_CASE("Bridge in-place conversion reuses tree storage") {
    auto pset = imcts::make_primitive_set({"+", "sin"}, 2);
    imcts::Bridge bridge(pset);
    imcts::Tree tree;

    std::vector<uint8_t> prefix1 = {
        pset.op_index("+"),
        pset.op_index("x0"),
        pset.op_index("x1")
    };
    bridge.to_tree(prefix1, tree);
    auto capacity1 = tree.nodes().capacity();

    std::vector<uint8_t> prefix2 = {
        pset.op_index("sin"),
        pset.op_index("x0")
    };
    bridge.to_tree(prefix2, tree);

    REQUIRE(tree.length() == 2);
    REQUIRE(tree.nodes().back().type == imcts::NodeType::Sin);
    REQUIRE(tree.nodes().capacity() == capacity1);
}
