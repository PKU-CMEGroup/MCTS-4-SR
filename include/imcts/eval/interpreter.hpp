// include/imcts/eval/interpreter.hpp
#pragma once
#include <array>
#include <utility>
#include <vector>
#include <Eigen/Core>
#include "imcts/core/tree.hpp"
#include "imcts/core/dataset.hpp"
#include "imcts/core/types.hpp"

namespace imcts {

class InterpreterWorkspace {
public:
    static constexpr Eigen::Index kBatchSize = 256;

    void prepare_evaluate(std::size_t num_samples, std::size_t num_nodes);
    void prepare_jacobian(std::size_t num_samples, std::size_t num_nodes, std::size_t num_coeffs);

    [[nodiscard]] Eigen::VectorXd& result() { return result_; }
    [[nodiscard]] const Eigen::VectorXd& result() const { return result_; }

    [[nodiscard]] Eigen::MatrixXd& jacobian() { return jacobian_; }
    [[nodiscard]] const Eigen::MatrixXd& jacobian() const { return jacobian_; }

private:
    friend class Interpreter;

    Eigen::ArrayXXd stack_;
    Eigen::ArrayXXd values_;
    Eigen::ArrayXXd adjoint_;
    std::vector<std::array<int, 2>> child_map_;
    Eigen::VectorXd result_;
    Eigen::MatrixXd jacobian_;
};

class Interpreter {
public:
    // Evaluate expression tree on dataset over the given range.
    // Returns predicted values vector of size range.size.
    static Eigen::VectorXd evaluate(const Tree& tree, const Dataset& ds, Range range);
    static void evaluate(const Tree& tree, const Dataset& ds, Range range,
                         InterpreterWorkspace& workspace);

    // Evaluate and compute Jacobian w.r.t. optimizable coefficients.
    // Returns (predictions [n], jacobian [n x num_coefficients]).
    static std::pair<Eigen::VectorXd, Eigen::MatrixXd>
    evaluate_with_jacobian(const Tree& tree, const Dataset& ds, Range range);
    static void evaluate_with_jacobian(const Tree& tree, const Dataset& ds, Range range,
                                       InterpreterWorkspace& workspace);
};

} // namespace imcts
