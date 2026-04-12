#include <catch2/catch_test_macros.hpp>

#include "imcts/core/tree.hpp"

namespace {

imcts::Node constant(double value)
{
    imcts::Node n(imcts::NodeType::Constant);
    n.value = static_cast<imcts::Scalar>(value);
    n.optimize = false;
    return n;
}

imcts::Node variable(std::uint8_t index)
{
    imcts::Node n(imcts::NodeType::Variable);
    n.var_index = index;
    return n;
}

imcts::Node op(imcts::NodeType type)
{
    return imcts::Node(type);
}

} // namespace

TEST_CASE("Tree::to_string removes redundant neutral elements")
{
    imcts::Tree tree({
        variable(0),
        constant(0.0),
        op(imcts::NodeType::Add),
        constant(1.0),
        op(imcts::NodeType::Mul),
    });

    tree.update_lengths();
    REQUIRE(tree.to_string() == "x0");
}

TEST_CASE("Tree::to_string folds constants and preserves needed parentheses")
{
    imcts::Tree tree({
        variable(0),
        constant(2.0),
        op(imcts::NodeType::Add),
        variable(1),
        op(imcts::NodeType::Mul),
    });

    tree.update_lengths();
    REQUIRE(tree.to_string() == "(x0 + 2) * x1");
}

TEST_CASE("Tree::to_string formats unary and safe binary expressions cleanly")
{
    imcts::Tree tree({
        constant(-1.0),
        variable(0),
        op(imcts::NodeType::Mul),
        constant(1.0),
        op(imcts::NodeType::Add),
        op(imcts::NodeType::Sin),
    });

    tree.update_lengths();
    REQUIRE(tree.to_string() == "sin(-x0 + 1)");
}
