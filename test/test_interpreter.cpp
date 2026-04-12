#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <Eigen/Core>
#include <cmath>
#include <stdexcept>
#include <vector>
#include "imcts/core/dataset.hpp"
#include "imcts/core/node.hpp"
#include "imcts/core/tree.hpp"
#include "imcts/eval/interpreter.hpp"
#include "imcts/eval/optimizer.hpp"

namespace {

using Catch::Approx;

Eigen::VectorXd evaluate_reference(const imcts::Tree& tree, const imcts::Dataset& ds, imcts::Range range)
{
    using imcts::NodeType;

    const auto& nodes = tree.nodes();
    int n = static_cast<int>(range.size);
    std::vector<Eigen::ArrayXd> stack;
    stack.reserve(nodes.size());

    for (const auto& node : nodes) {
        switch (node.arity) {
            case 0:
                if (node.type == NodeType::Variable) {
                    stack.push_back(ds.X.col(node.var_index).segment(range.start, range.size).array());
                } else {
                    stack.push_back(Eigen::ArrayXd::Constant(n, node.value));
                }
                break;
            case 1: {
                Eigen::ArrayXd child = std::move(stack.back());
                stack.pop_back();
                switch (node.type) {
                    case NodeType::Sin:  stack.push_back(child.sin()); break;
                    case NodeType::Cos:  stack.push_back(child.cos()); break;
                    case NodeType::Exp:  stack.push_back(child.exp()); break;
                    case NodeType::Log:  stack.push_back(child.log()); break;
                    case NodeType::Tanh: stack.push_back(child.tanh()); break;
                    case NodeType::Sqrt: stack.push_back(child.sqrt()); break;
                    case NodeType::Abs:  stack.push_back(child.abs()); break;
                    default: throw std::runtime_error("invalid unary op");
                }
                break;
            }
            case 2: {
                Eigen::ArrayXd right = std::move(stack.back());
                stack.pop_back();
                Eigen::ArrayXd left = std::move(stack.back());
                stack.pop_back();
                switch (node.type) {
                    case NodeType::Add: stack.push_back(left + right); break;
                    case NodeType::Sub: stack.push_back(left - right); break;
                    case NodeType::Mul: stack.push_back(left * right); break;
                    case NodeType::Div: stack.push_back(left / right); break;
                    default: throw std::runtime_error("invalid binary op");
                }
                break;
            }
        }
    }
    return stack[0].matrix();
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd>
evaluate_with_jacobian_reference(const imcts::Tree& tree, const imcts::Dataset& ds, imcts::Range range)
{
    using imcts::NodeType;

    const auto& nodes = tree.nodes();
    int n = static_cast<int>(range.size);
    int nn = static_cast<int>(nodes.size());

    std::vector<Eigen::ArrayXd> values(nn);
    std::vector<std::vector<int>> child_map(nn);
    std::vector<int> idx_stack;
    idx_stack.reserve(nn);

    for (int idx = 0; idx < nn; ++idx) {
        const auto& node = nodes[idx];
        switch (node.arity) {
            case 0:
                if (node.type == NodeType::Variable) {
                    values[idx] = ds.X.col(node.var_index).segment(range.start, range.size).array();
                } else {
                    values[idx] = Eigen::ArrayXd::Constant(n, node.value);
                }
                idx_stack.push_back(idx);
                break;
            case 1: {
                int child_idx = idx_stack.back();
                idx_stack.pop_back();
                child_map[idx] = {child_idx};
                switch (node.type) {
                    case NodeType::Sin:  values[idx] = values[child_idx].sin(); break;
                    case NodeType::Cos:  values[idx] = values[child_idx].cos(); break;
                    case NodeType::Exp:  values[idx] = values[child_idx].exp(); break;
                    case NodeType::Log:  values[idx] = values[child_idx].log(); break;
                    case NodeType::Tanh: values[idx] = values[child_idx].tanh(); break;
                    case NodeType::Sqrt: values[idx] = values[child_idx].sqrt(); break;
                    case NodeType::Abs:  values[idx] = values[child_idx].abs(); break;
                    default: throw std::runtime_error("invalid unary op");
                }
                idx_stack.push_back(idx);
                break;
            }
            case 2: {
                int right_idx = idx_stack.back();
                idx_stack.pop_back();
                int left_idx = idx_stack.back();
                idx_stack.pop_back();
                child_map[idx] = {left_idx, right_idx};
                switch (node.type) {
                    case NodeType::Add: values[idx] = values[left_idx] + values[right_idx]; break;
                    case NodeType::Sub: values[idx] = values[left_idx] - values[right_idx]; break;
                    case NodeType::Mul: values[idx] = values[left_idx] * values[right_idx]; break;
                    case NodeType::Div: values[idx] = values[left_idx] / values[right_idx]; break;
                    default: throw std::runtime_error("invalid binary op");
                }
                idx_stack.push_back(idx);
                break;
            }
        }
    }

    int num_coeffs = 0;
    for (const auto& node : nodes) {
        if (node.optimize) {
            ++num_coeffs;
        }
    }

    std::vector<Eigen::ArrayXd> adjoint(nn, Eigen::ArrayXd::Zero(n));
    adjoint[nn - 1] = Eigen::ArrayXd::Ones(n);

    for (int i = nn - 1; i >= 0; --i) {
        const auto& node = nodes[i];
        if (node.arity == 1) {
            int child_idx = child_map[i][0];
            Eigen::ArrayXd local_deriv;
            switch (node.type) {
                case NodeType::Sin:  local_deriv = values[child_idx].cos(); break;
                case NodeType::Cos:  local_deriv = -values[child_idx].sin(); break;
                case NodeType::Exp:  local_deriv = values[i]; break;
                case NodeType::Log:  local_deriv = 1.0 / values[child_idx]; break;
                case NodeType::Tanh: local_deriv = 1.0 - values[i].square(); break;
                case NodeType::Sqrt: local_deriv = 1.0 / (2.0 * values[i]); break;
                case NodeType::Abs:  local_deriv = values[child_idx].sign(); break;
                default: local_deriv = Eigen::ArrayXd::Ones(n); break;
            }
            adjoint[child_idx] += adjoint[i] * local_deriv;
        } else if (node.arity == 2) {
            int left_idx = child_map[i][0];
            int right_idx = child_map[i][1];
            switch (node.type) {
                case NodeType::Add:
                    adjoint[left_idx] += adjoint[i];
                    adjoint[right_idx] += adjoint[i];
                    break;
                case NodeType::Sub:
                    adjoint[left_idx] += adjoint[i];
                    adjoint[right_idx] -= adjoint[i];
                    break;
                case NodeType::Mul:
                    adjoint[left_idx] += adjoint[i] * values[right_idx];
                    adjoint[right_idx] += adjoint[i] * values[left_idx];
                    break;
                case NodeType::Div: {
                    Eigen::ArrayXd denom = values[right_idx];
                    adjoint[left_idx] += adjoint[i] / denom;
                    adjoint[right_idx] -= adjoint[i] * values[left_idx] / denom.square();
                    break;
                }
                default:
                    break;
            }
        }
    }

    Eigen::MatrixXd jacobian(n, num_coeffs);
    int col = 0;
    for (int i = 0; i < nn; ++i) {
        if (nodes[i].optimize) {
            jacobian.col(col++) = adjoint[i].matrix();
        }
    }
    return {values[nn - 1].matrix(), jacobian};
}

imcts::Dataset make_dataset()
{
    constexpr int n = 513;
    std::vector<std::vector<float>> x_cols(2, std::vector<float>(n));
    std::vector<float> y(n);
    for (int i = 0; i < n; ++i) {
        float x0 = static_cast<float>(i) / static_cast<float>(n);
        float x1 = std::sin(0.03f * static_cast<float>(i));
        x_cols[0][i] = x0;
        x_cols[1][i] = x1;
        y[i] = x0 + x1;
    }
    return imcts::Dataset(x_cols, y);
}

imcts::Tree make_variable_tree()
{
    using imcts::Node;
    using imcts::NodeType;
    std::vector<Node> nodes;
    Node x0(NodeType::Variable); x0.var_index = 0; x0.optimize = false; nodes.push_back(x0);
    Node x1(NodeType::Variable); x1.var_index = 1; x1.optimize = false; nodes.push_back(x1);
    nodes.emplace_back(NodeType::Add);
    imcts::Tree tree(std::move(nodes));
    tree.update_lengths();
    return tree;
}

imcts::Tree make_constant_tree()
{
    using imcts::Node;
    using imcts::NodeType;
    std::vector<Node> nodes;
    Node x0(NodeType::Variable); x0.var_index = 0; x0.optimize = false; nodes.push_back(x0);
    Node c1(NodeType::Constant); c1.value = 1.5; c1.optimize = true; nodes.push_back(c1);
    nodes.emplace_back(NodeType::Mul);
    Node c2(NodeType::Constant); c2.value = 0.25; c2.optimize = true; nodes.push_back(c2);
    nodes.emplace_back(NodeType::Add);
    imcts::Tree tree(std::move(nodes));
    tree.update_lengths();
    return tree;
}

imcts::Tree make_function_tree()
{
    using imcts::Node;
    using imcts::NodeType;
    std::vector<Node> nodes;
    Node x0(NodeType::Variable); x0.var_index = 0; x0.optimize = false; nodes.push_back(x0);
    Node x1(NodeType::Variable); x1.var_index = 1; x1.optimize = false; nodes.push_back(x1);
    nodes.emplace_back(NodeType::Add);
    nodes.emplace_back(NodeType::Sin);
    Node c1(NodeType::Constant); c1.value = 0.75; c1.optimize = true; nodes.push_back(c1);
    nodes.emplace_back(NodeType::Mul);
    Node x0b(NodeType::Variable); x0b.var_index = 0; x0b.optimize = false; nodes.push_back(x0b);
    Node c2(NodeType::Constant); c2.value = 0.2; c2.optimize = true; nodes.push_back(c2);
    nodes.emplace_back(NodeType::Sub);
    nodes.emplace_back(NodeType::Abs);
    nodes.emplace_back(NodeType::Log);
    nodes.emplace_back(NodeType::Add);
    imcts::Tree tree(std::move(nodes));
    tree.update_lengths();
    return tree;
}

void require_close(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs, double tol)
{
    REQUIRE(lhs.size() == rhs.size());
    for (Eigen::Index i = 0; i < lhs.size(); ++i) {
        REQUIRE(lhs(i) == Approx(rhs(i)).margin(tol));
    }
}

void require_close(const Eigen::MatrixXd& lhs, const Eigen::MatrixXd& rhs, double tol)
{
    REQUIRE(lhs.rows() == rhs.rows());
    REQUIRE(lhs.cols() == rhs.cols());
    for (Eigen::Index i = 0; i < lhs.rows(); ++i) {
        for (Eigen::Index j = 0; j < lhs.cols(); ++j) {
            REQUIRE(lhs(i, j) == Approx(rhs(i, j)).margin(tol));
        }
    }
}

} // namespace

TEST_CASE("Interpreter batch evaluate matches reference for representative trees")
{
    auto ds = make_dataset();
    imcts::Range range{0, static_cast<std::size_t>(ds.n_samples())};

    for (const auto& tree : {make_variable_tree(), make_constant_tree(), make_function_tree()}) {
        auto ref = evaluate_reference(tree, ds, range);
        auto got = imcts::Interpreter::evaluate(tree, ds, range);
        require_close(got, ref, 1e-10);
    }
}

TEST_CASE("Interpreter batch jacobian matches reference for constant and function trees")
{
    auto ds = make_dataset();
    imcts::Range range{0, static_cast<std::size_t>(ds.n_samples())};

    for (const auto& tree : {make_constant_tree(), make_function_tree()}) {
        auto [ref_pred, ref_jac] = evaluate_with_jacobian_reference(tree, ds, range);
        auto [got_pred, got_jac] = imcts::Interpreter::evaluate_with_jacobian(tree, ds, range);
        require_close(got_pred, ref_pred, 1e-10);
        require_close(got_jac, ref_jac, 1e-10);
    }
}

TEST_CASE("CoefficientOptimizer reuses workspace across trees with different coefficient counts")
{
    auto ds = make_dataset();
    imcts::Range range{0, static_cast<std::size_t>(ds.n_samples())};
    imcts::InterpreterWorkspace workspace;

    auto tree2 = make_constant_tree();
    auto opt2 = imcts::CoefficientOptimizer::optimize(tree2, ds, range, workspace, 20);
    imcts::Interpreter::evaluate_with_jacobian(opt2, ds, range, workspace);
    REQUIRE(workspace.jacobian().cols() == static_cast<Eigen::Index>(opt2.num_coefficients()));
    REQUIRE(workspace.result().size() == ds.n_samples());

    using imcts::Node;
    using imcts::NodeType;
    std::vector<Node> nodes;
    Node x0(NodeType::Variable); x0.var_index = 0; x0.optimize = false; nodes.push_back(x0);
    Node c1(NodeType::Constant); c1.value = 1.0; c1.optimize = true; nodes.push_back(c1);
    nodes.emplace_back(NodeType::Mul);
    imcts::Tree tree1(std::move(nodes));
    tree1.update_lengths();

    auto opt1 = imcts::CoefficientOptimizer::optimize(tree1, ds, range, workspace, 20);
    imcts::Interpreter::evaluate_with_jacobian(opt1, ds, range, workspace);
    REQUIRE(workspace.jacobian().cols() == static_cast<Eigen::Index>(opt1.num_coefficients()));
    REQUIRE(workspace.result().size() == ds.n_samples());
}
