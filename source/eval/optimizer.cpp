// source/eval/optimizer.cpp
#include "imcts/eval/optimizer.hpp"
#include "imcts/eval/interpreter.hpp"
#include "imcts/eval/timing.hpp"
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace imcts {

namespace {

struct NormalEquationState {
    Eigen::MatrixXd jtj;
    Eigen::VectorXd jtr;
    double cost = std::numeric_limits<double>::infinity();
};

void accumulate_state(Tree& working_tree,
                      const Dataset& ds,
                      Range range,
                      InterpreterWorkspace& workspace,
                      const Eigen::VectorXd& coeffs,
                      NormalEquationState& state)
{
    working_tree.set_coefficients(
        std::span<const Scalar>(coeffs.data(), static_cast<std::size_t>(coeffs.size())));
    Interpreter::accumulate_normal_equations(
        working_tree, ds, range,
        ds.y.segment(static_cast<Eigen::Index>(range.start),
                     static_cast<Eigen::Index>(range.size)),
        workspace, state.jtj, state.jtr, state.cost);
}

bool is_finite_vector(const Eigen::VectorXd& values)
{
    return values.array().isFinite().all();
}

} // anonymous namespace

Tree CoefficientOptimizer::optimize(const Tree& tree, const Dataset& ds, Range range,
                                     int max_iter) {
    InterpreterWorkspace workspace;
    return optimize(tree, ds, range, workspace, max_iter);
}

Tree CoefficientOptimizer::optimize(const Tree& tree, const Dataset& ds, Range range,
                                     InterpreterWorkspace& workspace, int max_iter) {
    ScopedTimer timer(TimingSection::CoefficientOptimize);
    auto coeffs = tree.get_coefficients();
    if (coeffs.empty()) {
        return tree; // nothing to optimize
    }

    Tree result = tree;
    Eigen::VectorXd x = Eigen::Map<Eigen::VectorXd>(coeffs.data(), coeffs.size());
    Tree working_tree = tree;
    NormalEquationState current;
    {
        ScopedTimer solve_timer(TimingSection::OptimizerLMMinimize);
        accumulate_state(working_tree, ds, range, workspace, x, current);
        if (std::isfinite(current.cost)) {
            double damping = 1e-6;
            constexpr double kMinDiagonal = 1e-12;
            constexpr double kMinDamping = 1e-12;
            constexpr double kMaxDamping = 1e12;
            constexpr int kMaxDampingAttempts = 6;

            for (int iter = 0; iter < max_iter; ++iter) {
                if (!std::isfinite(current.cost)) {
                    break;
                }
                if (current.jtr.size() == 0 || current.jtr.array().abs().maxCoeff() < 1e-10) {
                    break;
                }

                bool accepted = false;
                bool converged = false;
                for (int attempt = 0; attempt < kMaxDampingAttempts; ++attempt) {
                    Eigen::MatrixXd lhs = current.jtj;
                    const Eigen::VectorXd diag =
                        current.jtj.diagonal().cwiseAbs().cwiseMax(kMinDiagonal);
                    lhs.diagonal().array() += damping * diag.array();

                    Eigen::LDLT<Eigen::MatrixXd> solver(lhs);
                    if (solver.info() != Eigen::Success) {
                        damping = std::min(damping * 10.0, kMaxDamping);
                        continue;
                    }

                    Eigen::VectorXd step = solver.solve(-current.jtr);
                    if (solver.info() != Eigen::Success || !is_finite_vector(step)) {
                        damping = std::min(damping * 10.0, kMaxDamping);
                        continue;
                    }
                    if (step.norm() <= 1e-12 * (x.norm() + 1e-12)) {
                        accepted = true;
                        converged = true;
                        break;
                    }

                    Eigen::VectorXd trial_x = x + step;
                    NormalEquationState trial;
                    accumulate_state(working_tree, ds, range, workspace, trial_x, trial);
                    if (std::isfinite(trial.cost) && trial.cost < current.cost) {
                        x = std::move(trial_x);
                        current = std::move(trial);
                        damping = std::max(damping * 0.3, kMinDamping);
                        accepted = true;
                        break;
                    }

                    damping = std::min(damping * 10.0, kMaxDamping);
                }

                if (!accepted) {
                    break;
                }
                if (converged) {
                    break;
                }
            }
        }
    }

    std::vector<Scalar> opt_coeffs(x.data(), x.data() + x.size());
    result.set_coefficients(opt_coeffs);

    return result;
}

} // namespace imcts
