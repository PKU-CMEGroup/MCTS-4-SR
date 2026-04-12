// include/imcts/core/node.hpp
#pragma once
#include <cstdint>
#include <string>
#include "types.hpp"

namespace imcts {

enum class NodeType : uint8_t {
    // binary (arity 2)
    Add, Sub, Mul, Div,
    // unary (arity 1)
    Sin, Cos, Exp, Log, Tanh, Sqrt, Abs,
    // leaf (arity 0)
    Constant, Variable
};

inline uint8_t arity_of(NodeType t) {
    switch (t) {
        case NodeType::Add: case NodeType::Sub:
        case NodeType::Mul: case NodeType::Div:
            return 2;
        case NodeType::Sin: case NodeType::Cos:
        case NodeType::Exp: case NodeType::Log:
        case NodeType::Tanh: case NodeType::Sqrt:
        case NodeType::Abs:
            return 1;
        case NodeType::Constant: case NodeType::Variable:
            return 0;
    }
    return 0;
}

inline bool is_leaf(NodeType t) {
    return t == NodeType::Constant || t == NodeType::Variable;
}

struct Node {
    NodeType type;
    uint8_t  arity{0};
    uint8_t  var_index{0};   // only meaningful for Variable
    Scalar   value{1.0};     // coefficient / constant value
    uint16_t length{1};      // subtree size including self
    bool     optimize{false}; // participate in constant optimization

    Node() = default;

    explicit Node(NodeType t)
        : type(t)
        , arity(arity_of(t))
        , value(1.0)
        , length(1)
        , optimize(t == NodeType::Constant)
    {}

    [[nodiscard]] bool is_constant() const { return type == NodeType::Constant; }
    [[nodiscard]] bool is_variable() const { return type == NodeType::Variable; }
    [[nodiscard]] bool is_leaf()     const { return imcts::is_leaf(type); }
};

inline std::string node_type_name(NodeType t) {
    switch (t) {
        case NodeType::Add:      return "+";
        case NodeType::Sub:      return "-";
        case NodeType::Mul:      return "*";
        case NodeType::Div:      return "/";
        case NodeType::Sin:      return "sin";
        case NodeType::Cos:      return "cos";
        case NodeType::Exp:      return "exp";
        case NodeType::Log:      return "log";
        case NodeType::Tanh:     return "tanh";
        case NodeType::Sqrt:     return "sqrt";
        case NodeType::Abs:      return "abs";
        case NodeType::Constant: return "R";
        case NodeType::Variable: return "x";
    }
    return "?";
}

} // namespace imcts
