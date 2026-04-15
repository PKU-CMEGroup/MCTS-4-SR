// source/eval/optimizer.cpp
#include "imcts/eval/optimizer.hpp"
#include "imcts/eval/interpreter.hpp"
#include <Eigen/Core>
#include <span>
#include <unsupported/Eigen/NumericalDiff>
#include <unsupported/Eigen/LevenbergMarquardt>

namespace imcts {

namespace {

// Functor for Eigen's LevenbergMarquardt solver
struct LMFunctor {
    using Scalar = double;
    enum {
        InputsAtCompileTime = Eigen::Dynamic,
        ValuesAtCompileTime = Eigen::Dynamic
    };
    using InputType    = Eigen::VectorXd;
    using ValueType    = Eigen::VectorXd;
    using JacobianType = Eigen::MatrixXd;
    using QRSolver     = Eigen::ColPivHouseholderQR<JacobianType>;

    Tree tree_template;
    const Dataset& dataset;
    Range range;
    Eigen::VectorXd target;
    int n_coeffs;
    int n_samples;
    mutable Tree working_tree;
    mutable InterpreterWorkspace* workspace;

    LMFunctor(const Tree& t, const Dataset& ds, Range r, InterpreterWorkspace& ws)
        : tree_template(t), dataset(ds), range(r), working_tree(t), workspace(&ws)
    {
        n_coeffs = static_cast<int>(t.num_coefficients());
        n_samples = static_cast<int>(r.size);
        target = ds.y.segment(r.start, r.size);
    }

    int inputs() const { return n_coeffs; }
    int values() const { return n_samples; }

    // Compute residuals = predictions - target
    int operator()(const InputType& coeffs, ValueType& residuals) const {
        working_tree = tree_template;
        working_tree.set_coefficients(std::span<const Scalar>(coeffs.data(), static_cast<size_t>(coeffs.size())));

        Interpreter::evaluate(working_tree, dataset, range, *workspace);
        residuals = workspace->result() - target;

        // Replace NaN/Inf residuals with large values
        for (int i = 0; i < residuals.size(); ++i) {
            if (!std::isfinite(residuals(i))) {
                residuals(i) = 1e6;
            }
        }
        return 0;
    }

    // Compute Jacobian of residuals w.r.t. coefficients
    int df(const InputType& coeffs, JacobianType& jacobian) const {
        working_tree = tree_template;
        working_tree.set_coefficients(std::span<const Scalar>(coeffs.data(), static_cast<size_t>(coeffs.size())));

        Interpreter::evaluate_with_jacobian(working_tree, dataset, range, *workspace);
        jacobian = workspace->jacobian();

        // Replace NaN/Inf in Jacobian
        for (int i = 0; i < jacobian.rows(); ++i) {
            for (int j = 0; j < jacobian.cols(); ++j) {
                if (!std::isfinite(jacobian(i, j))) {
                    jacobian(i, j) = 0.0;
                }
            }
        }
        return 0;
    }
};

} // anonymous namespace

Tree CoefficientOptimizer::optimize(const Tree& tree, const Dataset& ds, Range range,
                                     int max_iter) {
    InterpreterWorkspace workspace;
    return optimize(tree, ds, range, workspace, max_iter);
}

Tree CoefficientOptimizer::optimize(const Tree& tree, const Dataset& ds, Range range,
                                     InterpreterWorkspace& workspace, int max_iter) {
    auto coeffs = tree.get_coefficients();
    if (coeffs.empty()) {
        return tree; // nothing to optimize
    }

    Tree result = tree;
    LMFunctor functor(tree, ds, range, workspace);

    Eigen::LevenbergMarquardt<LMFunctor> lm(functor);
    lm.setMaxfev(max_iter + 2);

    Eigen::VectorXd x = Eigen::Map<Eigen::VectorXd>(coeffs.data(), coeffs.size());
    lm.minimize(x);

    std::vector<Scalar> opt_coeffs(x.data(), x.data() + x.size());
    result.set_coefficients(opt_coeffs);

    return result;
}

} // namespace imcts
