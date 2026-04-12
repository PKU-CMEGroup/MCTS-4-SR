// source/core/tree.cpp
#include "imcts/core/tree.hpp"
#include <iomanip>
#include <sstream>
#include <stack>
#include <functional>

namespace imcts {

namespace {

enum class Precedence : int {
    AddSub = 1,
    MulDiv = 2,
    Unary = 3,
    Atom = 4,
};

struct Expr {
    std::string text;
    Precedence precedence{Precedence::Atom};
    bool is_constant{false};
    Scalar constant_value{0.0};
};

std::string format_scalar(Scalar value)
{
    std::ostringstream oss;
    oss << std::setprecision(12) << value;
    auto s = oss.str();
    if (s.find('.') != std::string::npos && s.find('e') == std::string::npos && s.find('E') == std::string::npos) {
        while (!s.empty() && s.back() == '0') {
            s.pop_back();
        }
        if (!s.empty() && s.back() == '.') {
            s.pop_back();
        }
    }
    if (s == "-0") {
        s = "0";
    }
    return s;
}

bool nearly_equal(Scalar a, Scalar b)
{
    return std::abs(a - b) < 1e-12;
}

Expr make_constant(Scalar value)
{
    return Expr{format_scalar(value), Precedence::Atom, true, value};
}

Expr make_variable(uint8_t var_index)
{
    return Expr{"x" + std::to_string(var_index), Precedence::Atom, false, 0.0};
}

std::string maybe_paren(const Expr& expr, Precedence parent, bool strict = false)
{
    const auto expr_prec = static_cast<int>(expr.precedence);
    const auto parent_prec = static_cast<int>(parent);
    if (expr_prec < parent_prec || (strict && expr_prec == parent_prec)) {
        return "(" + expr.text + ")";
    }
    return expr.text;
}

Expr simplify_unary(NodeType type, Expr child)
{
    if (child.is_constant) {
        switch (type) {
            case NodeType::Sin:  return make_constant(std::sin(child.constant_value));
            case NodeType::Cos:  return make_constant(std::cos(child.constant_value));
            case NodeType::Exp:  return make_constant(std::exp(std::min(child.constant_value, 500.0)));
            case NodeType::Log:  return make_constant(std::log(std::abs(child.constant_value) + 1e-10));
            case NodeType::Tanh: return make_constant(std::tanh(child.constant_value));
            case NodeType::Sqrt: return make_constant(std::sqrt(std::abs(child.constant_value)));
            case NodeType::Abs:  return make_constant(std::abs(child.constant_value));
            default: break;
        }
    }

    return Expr{
        node_type_name(type) + "(" + child.text + ")",
        Precedence::Unary,
        false,
        0.0
    };
}

Expr simplify_binary(NodeType type, Expr left, Expr right)
{
    if (left.is_constant && right.is_constant) {
        switch (type) {
            case NodeType::Add: return make_constant(left.constant_value + right.constant_value);
            case NodeType::Sub: return make_constant(left.constant_value - right.constant_value);
            case NodeType::Mul: return make_constant(left.constant_value * right.constant_value);
            case NodeType::Div:
                if (std::abs(right.constant_value) < 1e-10) {
                    return make_constant(1.0);
                }
                return make_constant(left.constant_value / right.constant_value);
            default: break;
        }
    }

    switch (type) {
        case NodeType::Add:
            if (left.is_constant && nearly_equal(left.constant_value, 0.0)) return right;
            if (right.is_constant && nearly_equal(right.constant_value, 0.0)) return left;
            return Expr{
                maybe_paren(left, Precedence::AddSub) + " + " + maybe_paren(right, Precedence::AddSub),
                Precedence::AddSub,
                false,
                0.0
            };
        case NodeType::Sub:
            if (right.is_constant && nearly_equal(right.constant_value, 0.0)) return left;
            return Expr{
                maybe_paren(left, Precedence::AddSub) + " - " + maybe_paren(right, Precedence::AddSub, true),
                Precedence::AddSub,
                false,
                0.0
            };
        case NodeType::Mul:
            if (left.is_constant && nearly_equal(left.constant_value, 0.0)) return make_constant(0.0);
            if (right.is_constant && nearly_equal(right.constant_value, 0.0)) return make_constant(0.0);
            if (left.is_constant && nearly_equal(left.constant_value, 1.0)) return right;
            if (right.is_constant && nearly_equal(right.constant_value, 1.0)) return left;
            if (left.is_constant && nearly_equal(left.constant_value, -1.0)) {
                return Expr{"-" + maybe_paren(right, Precedence::Unary), Precedence::Unary, false, 0.0};
            }
            if (right.is_constant && nearly_equal(right.constant_value, -1.0)) {
                return Expr{"-" + maybe_paren(left, Precedence::Unary), Precedence::Unary, false, 0.0};
            }
            return Expr{
                maybe_paren(left, Precedence::MulDiv) + " * " + maybe_paren(right, Precedence::MulDiv),
                Precedence::MulDiv,
                false,
                0.0
            };
        case NodeType::Div:
            if (right.is_constant && nearly_equal(right.constant_value, 1.0)) return left;
            return Expr{
                maybe_paren(left, Precedence::MulDiv) + " / " + maybe_paren(right, Precedence::MulDiv, true),
                Precedence::MulDiv,
                false,
                0.0
            };
        default:
            return Expr{"?", Precedence::Atom, false, 0.0};
    }
}

Expr to_expr_impl(const std::vector<Node>& nodes, int idx)
{
    const auto& node = nodes[static_cast<std::size_t>(idx)];
    switch (node.arity) {
        case 0:
            if (node.type == NodeType::Variable) {
                return make_variable(node.var_index);
            }
            return make_constant(node.value);
        case 1: {
            auto child = to_expr_impl(nodes, idx - 1);
            return simplify_unary(node.type, std::move(child));
        }
        case 2: {
            const int right_idx = idx - 1;
            const int left_idx = right_idx - nodes[static_cast<std::size_t>(right_idx)].length;
            auto left = to_expr_impl(nodes, left_idx);
            auto right = to_expr_impl(nodes, right_idx);
            return simplify_binary(node.type, std::move(left), std::move(right));
        }
        default:
            return Expr{"?", Precedence::Atom, false, 0.0};
    }
}

} // namespace

Tree& Tree::update_lengths() {
    // In postfix order, children come before parent.
    // For each node, length = 1 + sum of children lengths.
    std::vector<uint16_t> stack;
    for (auto& node : nodes_) {
        uint16_t len = 1;
        for (int c = 0; c < node.arity; ++c) {
            if (!stack.empty()) {
                len += stack.back();
                stack.pop_back();
            }
        }
        node.length = len;
        stack.push_back(len);
    }
    return *this;
}

std::size_t Tree::num_coefficients() const {
    std::size_t count = 0;
    for (const auto& n : nodes_) {
        if (n.optimize) ++count;
    }
    return count;
}

std::vector<Scalar> Tree::get_coefficients() const {
    std::vector<Scalar> coeffs;
    coeffs.reserve(num_coefficients());
    for (const auto& n : nodes_) {
        if (n.optimize) coeffs.push_back(n.value);
    }
    return coeffs;
}

void Tree::set_coefficients(const std::vector<Scalar>& coeffs) {
    set_coefficients(std::span<const Scalar>(coeffs));
}

void Tree::set_coefficients(std::span<const Scalar> coeffs) {
    std::size_t j = 0;
    for (auto& n : nodes_) {
        if (n.optimize && j < coeffs.size()) {
            n.value = coeffs[j++];
        }
    }
}

Hash Tree::structure_hash() const {
    // FNV-1a style hash combining type, arity, and var_index per node
    Hash h = 14695981039346656037ULL; // FNV offset basis
    for (const auto& n : nodes_) {
        auto mix = [&](uint8_t byte) {
            h ^= static_cast<Hash>(byte);
            h *= 1099511628211ULL; // FNV prime
        };
        mix(static_cast<uint8_t>(n.type));
        mix(n.arity);
        if (n.type == NodeType::Variable) {
            mix(n.var_index);
        }
    }
    return h;
}

std::string Tree::to_string() const
{
    if (nodes_.empty()) {
        return "";
    }
    return to_expr_impl(nodes_, static_cast<int>(nodes_.size() - 1)).text;
}

} // namespace imcts
