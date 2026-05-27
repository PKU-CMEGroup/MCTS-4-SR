// source/eval/interpreter.cpp
#include "imcts/eval/interpreter.hpp"
#include "imcts/eval/timing.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace imcts {

namespace {

constexpr int kMinBatchesForParallelJacobian = 32;
constexpr int kMinBatchesForParallelEvaluate = 32;
constexpr int kBatchesPerThreadTarget = 8;
constexpr double kInvalidResidual = 1e6;

inline int count_coefficients(const std::vector<Node>& nodes)
{
    int num_coeffs = 0;
    for (const auto& node : nodes) {
        if (node.optimize) {
            ++num_coeffs;
        }
    }
    return num_coeffs;
}

inline void fill_child_map(const std::vector<Node>& nodes, std::span<std::array<int, 2>> child_map)
{
    std::fill(child_map.begin(), child_map.begin() + static_cast<std::ptrdiff_t>(nodes.size()),
              std::array<int, 2>{-1, -1});
    std::vector<int> idx_stack;
    idx_stack.reserve(nodes.size());

    for (int idx = 0; idx < static_cast<int>(nodes.size()); ++idx) {
        const auto& node = nodes[idx];
        switch (node.arity) {
            case 0:
                idx_stack.push_back(idx);
                break;
            case 1: {
                int child_idx = idx_stack.back();
                idx_stack.pop_back();
                child_map[static_cast<std::size_t>(idx)] = {child_idx, -1};
                idx_stack.push_back(idx);
                break;
            }
            case 2: {
                int right_idx = idx_stack.back();
                idx_stack.pop_back();
                int left_idx = idx_stack.back();
                idx_stack.pop_back();
                child_map[static_cast<std::size_t>(idx)] = {left_idx, right_idx};
                idx_stack.push_back(idx);
                break;
            }
        }
    }
}

inline void ensure_array_buffers(std::vector<Eigen::ArrayXXd>& buffers,
                                 int num_threads,
                                 Eigen::Index rows,
                                 Eigen::Index cols)
{
    if (num_threads <= 1) {
        return;
    }
    if (buffers.size() != static_cast<std::size_t>(num_threads)) {
        buffers.resize(static_cast<std::size_t>(num_threads));
    }
    for (auto& buffer : buffers) {
        if (buffer.rows() != rows || buffer.cols() < cols) {
            buffer.resize(rows, cols);
        }
    }
}

} // anonymous namespace

void InterpreterWorkspace::prepare_evaluate(std::size_t num_samples, std::size_t num_nodes)
{
    const auto result_rows = static_cast<Eigen::Index>(num_samples);
    const auto stack_rows = std::min(InterpreterWorkspace::kBatchSize, result_rows);
    const auto cols = static_cast<Eigen::Index>(num_nodes);
    if (stack_.rows() != stack_rows || stack_.cols() < cols) {
        stack_.resize(stack_rows, cols);
    }
    if (result_.size() != result_rows) {
        result_.resize(result_rows);
    }
}

void InterpreterWorkspace::prepare_adjoint(std::size_t num_nodes)
{
    const auto cols = static_cast<Eigen::Index>(num_nodes);
    if (values_.rows() != kBatchSize || values_.cols() < cols) {
        values_.resize(kBatchSize, cols);
    }
    if (adjoint_.rows() != kBatchSize || adjoint_.cols() < cols) {
        adjoint_.resize(kBatchSize, cols);
    }
    if (child_map_.size() < num_nodes) {
        child_map_.resize(num_nodes, {-1, -1});
    }
}

void InterpreterWorkspace::prepare_jacobian(std::size_t num_samples, std::size_t num_nodes,
                                            std::size_t num_coeffs)
{
    const auto rows = static_cast<Eigen::Index>(num_samples);
    const auto cols = static_cast<Eigen::Index>(num_nodes);
    if (values_.rows() != kBatchSize || values_.cols() < cols) {
        values_.resize(kBatchSize, cols);
    }
    if (adjoint_.rows() != kBatchSize || adjoint_.cols() < cols) {
        adjoint_.resize(kBatchSize, cols);
    }
    if (child_map_.size() < num_nodes) {
        child_map_.resize(num_nodes, {-1, -1});
    }
    if (result_.size() != rows) {
        result_.resize(rows);
    }
    if (jacobian_.rows() != rows || jacobian_.cols() != static_cast<Eigen::Index>(num_coeffs)) {
        jacobian_.resize(rows, static_cast<Eigen::Index>(num_coeffs));
    }
}

Eigen::VectorXd Interpreter::evaluate(const Tree& tree, const Dataset& ds, Range range) {
    InterpreterWorkspace workspace;
    evaluate(tree, ds, range, workspace);
    return workspace.result();
}

void Interpreter::evaluate(const Tree& tree, const Dataset& ds, Range range,
                           InterpreterWorkspace& workspace) {
    ScopedTimer timer(TimingSection::InterpreterEvaluate);
    const auto& nodes = tree.nodes();
    workspace.prepare_evaluate(range.size, nodes.size());
    const auto total = static_cast<Eigen::Index>(range.size);
    const auto num_batches = static_cast<int>((total + InterpreterWorkspace::kBatchSize - 1)
                                              / InterpreterWorkspace::kBatchSize);

    int num_threads = 1;
#ifdef _OPENMP
    if (num_batches >= kMinBatchesForParallelEvaluate) {
        const int max_threads = omp_get_max_threads();
        const int target_threads = std::max(2, num_batches / kBatchesPerThreadTarget);
        num_threads = std::min(max_threads, target_threads);
    }
#endif

    ensure_array_buffers(workspace.stack_buffers_, num_threads,
                         InterpreterWorkspace::kBatchSize,
                         static_cast<Eigen::Index>(nodes.size()));

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (num_threads > 1) num_threads(num_threads)
#endif
    for (int batch = 0; batch < num_batches; ++batch) {
        int thread_id = 0;
#ifdef _OPENMP
        thread_id = omp_get_thread_num();
#endif
        auto& stack = num_threads == 1
            ? workspace.stack_
            : workspace.stack_buffers_[static_cast<std::size_t>(thread_id)];
        const Eigen::Index row0 = static_cast<Eigen::Index>(batch) * InterpreterWorkspace::kBatchSize;
        const Eigen::Index rem = std::min(InterpreterWorkspace::kBatchSize, total - row0);
        Eigen::Index sp = 0;

        for (const auto& node : nodes) {
            switch (node.arity) {
                case 0: {
                    auto dst = stack.col(sp).head(rem);
                    if (node.type == NodeType::Variable) {
                        dst = ds.X.col(node.var_index)
                                 .segment(static_cast<Eigen::Index>(range.start) + row0, rem)
                                 .array();
                    } else {
                        dst.setConstant(node.value);
                    }
                    ++sp;
                    break;
                }
                case 1: {
                    auto top = stack.col(sp - 1).head(rem);
                    switch (node.type) {
                        case NodeType::Sin:  top = top.sin(); break;
                        case NodeType::Cos:  top = top.cos(); break;
                        case NodeType::Exp:  top = top.exp(); break;
                        case NodeType::Log:  top = top.log(); break;
                        case NodeType::Tanh: top = top.tanh(); break;
                        case NodeType::Sqrt: top = top.sqrt(); break;
                        case NodeType::Abs:  top = top.abs(); break;
                        default: throw std::runtime_error("Interpreter: invalid unary op");
                    }
                    break;
                }
                case 2: {
                    auto right = stack.col(sp - 1).head(rem);
                    auto left  = stack.col(sp - 2).head(rem);
                    switch (node.type) {
                        case NodeType::Add: left += right; break;
                        case NodeType::Sub: left -= right; break;
                        case NodeType::Mul: left *= right; break;
                        case NodeType::Div: left /= right; break;
                        default: throw std::runtime_error("Interpreter: invalid binary op");
                    }
                    --sp;
                    break;
                }
            }
        }

        if (sp != 1) {
            throw std::runtime_error("Interpreter: invalid expression tree");
        }

        workspace.result_.segment(row0, rem) = stack.col(0).head(rem).matrix();
    }
}

void Interpreter::evaluate_residual(const Tree& tree, const Dataset& ds, Range range,
                                    const Eigen::Ref<const Eigen::VectorXd>& target,
                                    Eigen::VectorXd& residuals,
                                    InterpreterWorkspace& workspace) {
    ScopedTimer timer(TimingSection::InterpreterEvaluateResidual);
    if (target.size() != static_cast<Eigen::Index>(range.size)) {
        throw std::runtime_error("Interpreter: residual target size mismatch");
    }

    const auto& nodes = tree.nodes();
    workspace.prepare_evaluate(range.size, nodes.size());
    const auto total = static_cast<Eigen::Index>(range.size);
    residuals.resize(total);
    const auto num_batches = static_cast<int>((total + InterpreterWorkspace::kBatchSize - 1)
                                              / InterpreterWorkspace::kBatchSize);

    int num_threads = 1;
#ifdef _OPENMP
    if (num_batches >= kMinBatchesForParallelEvaluate) {
        const int max_threads = omp_get_max_threads();
        const int target_threads = std::max(2, num_batches / kBatchesPerThreadTarget);
        num_threads = std::min(max_threads, target_threads);
    }
#endif

    ensure_array_buffers(workspace.stack_buffers_, num_threads,
                         InterpreterWorkspace::kBatchSize,
                         static_cast<Eigen::Index>(nodes.size()));

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (num_threads > 1) num_threads(num_threads)
#endif
    for (int batch = 0; batch < num_batches; ++batch) {
        int thread_id = 0;
#ifdef _OPENMP
        thread_id = omp_get_thread_num();
#endif
        auto& stack = num_threads == 1
            ? workspace.stack_
            : workspace.stack_buffers_[static_cast<std::size_t>(thread_id)];
        const Eigen::Index row0 = static_cast<Eigen::Index>(batch) * InterpreterWorkspace::kBatchSize;
        const Eigen::Index rem = std::min(InterpreterWorkspace::kBatchSize, total - row0);
        Eigen::Index sp = 0;

        for (const auto& node : nodes) {
            switch (node.arity) {
                case 0: {
                    auto dst = stack.col(sp).head(rem);
                    if (node.type == NodeType::Variable) {
                        dst = ds.X.col(node.var_index)
                                 .segment(static_cast<Eigen::Index>(range.start) + row0, rem)
                                 .array();
                    } else {
                        dst.setConstant(node.value);
                    }
                    ++sp;
                    break;
                }
                case 1: {
                    auto top = stack.col(sp - 1).head(rem);
                    switch (node.type) {
                        case NodeType::Sin:  top = top.sin(); break;
                        case NodeType::Cos:  top = top.cos(); break;
                        case NodeType::Exp:  top = top.exp(); break;
                        case NodeType::Log:  top = top.log(); break;
                        case NodeType::Tanh: top = top.tanh(); break;
                        case NodeType::Sqrt: top = top.sqrt(); break;
                        case NodeType::Abs:  top = top.abs(); break;
                        default: throw std::runtime_error("Interpreter: invalid unary op");
                    }
                    break;
                }
                case 2: {
                    auto right = stack.col(sp - 1).head(rem);
                    auto left  = stack.col(sp - 2).head(rem);
                    switch (node.type) {
                        case NodeType::Add: left += right; break;
                        case NodeType::Sub: left -= right; break;
                        case NodeType::Mul: left *= right; break;
                        case NodeType::Div: left /= right; break;
                        default: throw std::runtime_error("Interpreter: invalid binary op");
                    }
                    --sp;
                    break;
                }
            }
        }

        if (sp != 1) {
            throw std::runtime_error("Interpreter: invalid expression tree");
        }

        auto residual = stack.col(0).head(rem) - target.segment(row0, rem).array();
        residuals.segment(row0, rem) =
            residual.isFinite().select(residual, kInvalidResidual).matrix();
    }
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd>
Interpreter::evaluate_with_jacobian(const Tree& tree, const Dataset& ds, Range range) {
    InterpreterWorkspace workspace;
    evaluate_with_jacobian(tree, ds, range, workspace);
    return {workspace.result(), workspace.jacobian()};
}

void Interpreter::evaluate_with_jacobian(const Tree& tree, const Dataset& ds, Range range,
                                         InterpreterWorkspace& workspace) {
    evaluate_with_jacobian(tree, ds, range, workspace, workspace.jacobian_);
}

void Interpreter::evaluate_with_jacobian(const Tree& tree, const Dataset& ds, Range range,
                                         InterpreterWorkspace& workspace,
                                         Eigen::MatrixXd& jacobian) {
    ScopedTimer timer(TimingSection::InterpreterEvaluateWithJacobian);
    const auto& nodes = tree.nodes();
    const int nn = static_cast<int>(nodes.size());
    int num_coeffs = count_coefficients(nodes);

    workspace.prepare_jacobian(range.size, nodes.size(), static_cast<std::size_t>(num_coeffs));
    const auto total = static_cast<Eigen::Index>(range.size);
    jacobian.resize(total, static_cast<Eigen::Index>(num_coeffs));
    fill_child_map(nodes, std::span<std::array<int, 2>>(workspace.child_map_.data(), nodes.size()));
    const auto child_map = std::span<const std::array<int, 2>>(workspace.child_map_.data(), nodes.size());
    const auto num_batches = static_cast<int>((total + InterpreterWorkspace::kBatchSize - 1)
                                              / InterpreterWorkspace::kBatchSize);

    int num_threads = 1;
#ifdef _OPENMP
    if (num_batches >= kMinBatchesForParallelJacobian) {
        const int max_threads = omp_get_max_threads();
        const int target_threads = std::max(2, num_batches / kBatchesPerThreadTarget);
        num_threads = std::min(max_threads, target_threads);
    }
#endif

    ensure_array_buffers(workspace.value_buffers_, num_threads,
                         InterpreterWorkspace::kBatchSize,
                         static_cast<Eigen::Index>(nodes.size()));
    ensure_array_buffers(workspace.adjoint_buffers_, num_threads,
                         InterpreterWorkspace::kBatchSize,
                         static_cast<Eigen::Index>(nodes.size()));

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (num_threads > 1) num_threads(num_threads)
#endif
    for (int batch = 0; batch < num_batches; ++batch) {
        int thread_id = 0;
#ifdef _OPENMP
        thread_id = omp_get_thread_num();
#endif
        auto& values = num_threads == 1
            ? workspace.values_
            : workspace.value_buffers_[static_cast<std::size_t>(thread_id)];
        auto& adjoint = num_threads == 1
            ? workspace.adjoint_
            : workspace.adjoint_buffers_[static_cast<std::size_t>(thread_id)];
        const Eigen::Index row0 = static_cast<Eigen::Index>(batch) * InterpreterWorkspace::kBatchSize;
        const Eigen::Index rem = std::min(InterpreterWorkspace::kBatchSize, total - row0);
        for (int idx = 0; idx < nn; ++idx) {
            const auto& node = nodes[idx];
            switch (node.arity) {
                case 0: {
                    auto dst = values.col(idx).head(rem);
                    if (node.type == NodeType::Variable) {
                        dst = ds.X.col(node.var_index)
                                 .segment(static_cast<Eigen::Index>(range.start) + row0, rem)
                                 .array();
                    } else {
                        dst.setConstant(node.value);
                    }
                    break;
                }
                case 1: {
                    auto dst = values.col(idx).head(rem);
                    auto src = values.col(child_map[idx][0]).head(rem);
                    switch (node.type) {
                        case NodeType::Sin:  dst = src.sin(); break;
                        case NodeType::Cos:  dst = src.cos(); break;
                        case NodeType::Exp:  dst = src.exp(); break;
                        case NodeType::Log:  dst = src.log(); break;
                        case NodeType::Tanh: dst = src.tanh(); break;
                        case NodeType::Sqrt: dst = src.sqrt(); break;
                        case NodeType::Abs:  dst = src.abs(); break;
                        default: throw std::runtime_error("Interpreter: invalid unary op");
                    }
                    break;
                }
                case 2: {
                    auto dst = values.col(idx).head(rem);
                    auto left = values.col(child_map[idx][0]).head(rem);
                    auto right = values.col(child_map[idx][1]).head(rem);
                    switch (node.type) {
                        case NodeType::Add: dst = left + right; break;
                        case NodeType::Sub: dst = left - right; break;
                        case NodeType::Mul: dst = left * right; break;
                        case NodeType::Div: dst = left / right; break;
                        default: throw std::runtime_error("Interpreter: invalid binary op");
                    }
                    break;
                }
            }
        }

        adjoint.topRows(rem).setZero();
        adjoint.col(nn - 1).head(rem).setOnes();

        for (int i = nn - 1; i >= 0; --i) {
            const auto& node = nodes[i];
            if (node.arity == 1) {
                const int child_idx = child_map[static_cast<std::size_t>(i)][0];
                switch (node.type) {
                    case NodeType::Sin:
                        adjoint.col(child_idx).head(rem) += adjoint.col(i).head(rem)
                            * values.col(child_idx).head(rem).cos();
                        break;
                    case NodeType::Cos:
                        adjoint.col(child_idx).head(rem) -= adjoint.col(i).head(rem)
                            * values.col(child_idx).head(rem).sin();
                        break;
                    case NodeType::Exp:
                        adjoint.col(child_idx).head(rem) += adjoint.col(i).head(rem)
                            * values.col(i).head(rem);
                        break;
                    case NodeType::Log:
                        adjoint.col(child_idx).head(rem) += adjoint.col(i).head(rem)
                            / values.col(child_idx).head(rem);
                        break;
                    case NodeType::Tanh:
                        adjoint.col(child_idx).head(rem) += adjoint.col(i).head(rem)
                            * (1.0 - values.col(i).head(rem).square());
                        break;
                    case NodeType::Sqrt:
                        adjoint.col(child_idx).head(rem) += adjoint.col(i).head(rem)
                            / (2.0 * values.col(i).head(rem));
                        break;
                    case NodeType::Abs:
                        adjoint.col(child_idx).head(rem) += adjoint.col(i).head(rem)
                            * values.col(child_idx).head(rem).sign();
                        break;
                    default:
                        adjoint.col(child_idx).head(rem) += adjoint.col(i).head(rem);
                        break;
                }
            } else if (node.arity == 2) {
                const int left_idx  = child_map[static_cast<std::size_t>(i)][0];
                const int right_idx = child_map[static_cast<std::size_t>(i)][1];

                switch (node.type) {
                    case NodeType::Add:
                        adjoint.col(left_idx).head(rem)  += adjoint.col(i).head(rem);
                        adjoint.col(right_idx).head(rem) += adjoint.col(i).head(rem);
                        break;
                    case NodeType::Sub:
                        adjoint.col(left_idx).head(rem)  += adjoint.col(i).head(rem);
                        adjoint.col(right_idx).head(rem) -= adjoint.col(i).head(rem);
                        break;
                    case NodeType::Mul:
                        adjoint.col(left_idx).head(rem)  += adjoint.col(i).head(rem) * values.col(right_idx).head(rem);
                        adjoint.col(right_idx).head(rem) += adjoint.col(i).head(rem) * values.col(left_idx).head(rem);
                        break;
                    case NodeType::Div: {
                        Eigen::ArrayXd denom = values.col(right_idx).head(rem);
                        adjoint.col(left_idx).head(rem)  += adjoint.col(i).head(rem) / denom;
                        adjoint.col(right_idx).head(rem) -= adjoint.col(i).head(rem)
                                                          * values.col(left_idx).head(rem)
                                                          / denom.square();
                        break;
                    }
                    default:
                        break;
                }
            }
        }

        int col = 0;
        for (int i = 0; i < nn; ++i) {
            if (nodes[i].optimize) {
                auto src = adjoint.col(i).head(rem);
                jacobian.block(row0, col++, rem, 1) =
                    src.isFinite().select(src, 0.0).matrix();
            }
        }
        workspace.result_.segment(row0, rem) = values.col(nn - 1).head(rem).matrix();
    }
}

void Interpreter::accumulate_normal_equations(const Tree& tree, const Dataset& ds, Range range,
                                              const Eigen::Ref<const Eigen::VectorXd>& target,
                                              InterpreterWorkspace& workspace,
                                              Eigen::MatrixXd& jtj,
                                              Eigen::VectorXd& jtr,
                                              double& cost) {
    ScopedTimer timer(TimingSection::NormalEquationAccumulate);
    if (target.size() != static_cast<Eigen::Index>(range.size)) {
        throw std::runtime_error("Interpreter: normal equation target size mismatch");
    }

    const auto& nodes = tree.nodes();
    const int nn = static_cast<int>(nodes.size());
    const int num_coeffs = count_coefficients(nodes);

    workspace.prepare_adjoint(nodes.size());
    const auto total = static_cast<Eigen::Index>(range.size);
    fill_child_map(nodes, std::span<std::array<int, 2>>(workspace.child_map_.data(), nodes.size()));
    const auto child_map = std::span<const std::array<int, 2>>(workspace.child_map_.data(), nodes.size());
    const auto num_batches = static_cast<int>((total + InterpreterWorkspace::kBatchSize - 1)
                                              / InterpreterWorkspace::kBatchSize);

    int num_threads = 1;
#ifdef _OPENMP
    if (num_batches >= kMinBatchesForParallelJacobian) {
        const int max_threads = omp_get_max_threads();
        const int target_threads = std::max(2, num_batches / kBatchesPerThreadTarget);
        num_threads = std::min(max_threads, target_threads);
    }
#endif

    ensure_array_buffers(workspace.value_buffers_, num_threads,
                         InterpreterWorkspace::kBatchSize,
                         static_cast<Eigen::Index>(nodes.size()));
    ensure_array_buffers(workspace.adjoint_buffers_, num_threads,
                         InterpreterWorkspace::kBatchSize,
                         static_cast<Eigen::Index>(nodes.size()));

    std::vector<Eigen::MatrixXd> jtj_buffers(
        static_cast<std::size_t>(num_threads),
        Eigen::MatrixXd::Zero(num_coeffs, num_coeffs));
    std::vector<Eigen::VectorXd> jtr_buffers(
        static_cast<std::size_t>(num_threads),
        Eigen::VectorXd::Zero(num_coeffs));
    std::vector<double> cost_buffers(static_cast<std::size_t>(num_threads), 0.0);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (num_threads > 1) num_threads(num_threads)
#endif
    for (int batch = 0; batch < num_batches; ++batch) {
        int thread_id = 0;
#ifdef _OPENMP
        thread_id = omp_get_thread_num();
#endif
        auto& values = num_threads == 1
            ? workspace.values_
            : workspace.value_buffers_[static_cast<std::size_t>(thread_id)];
        auto& adjoint = num_threads == 1
            ? workspace.adjoint_
            : workspace.adjoint_buffers_[static_cast<std::size_t>(thread_id)];
        auto& local_jtj = jtj_buffers[static_cast<std::size_t>(thread_id)];
        auto& local_jtr = jtr_buffers[static_cast<std::size_t>(thread_id)];
        auto& local_cost = cost_buffers[static_cast<std::size_t>(thread_id)];

        const Eigen::Index row0 = static_cast<Eigen::Index>(batch) * InterpreterWorkspace::kBatchSize;
        const Eigen::Index rem = std::min(InterpreterWorkspace::kBatchSize, total - row0);

        for (int idx = 0; idx < nn; ++idx) {
            const auto& node = nodes[idx];
            switch (node.arity) {
                case 0: {
                    auto dst = values.col(idx).head(rem);
                    if (node.type == NodeType::Variable) {
                        dst = ds.X.col(node.var_index)
                                 .segment(static_cast<Eigen::Index>(range.start) + row0, rem)
                                 .array();
                    } else {
                        dst.setConstant(node.value);
                    }
                    break;
                }
                case 1: {
                    auto dst = values.col(idx).head(rem);
                    auto src = values.col(child_map[idx][0]).head(rem);
                    switch (node.type) {
                        case NodeType::Sin:  dst = src.sin(); break;
                        case NodeType::Cos:  dst = src.cos(); break;
                        case NodeType::Exp:  dst = src.exp(); break;
                        case NodeType::Log:  dst = src.log(); break;
                        case NodeType::Tanh: dst = src.tanh(); break;
                        case NodeType::Sqrt: dst = src.sqrt(); break;
                        case NodeType::Abs:  dst = src.abs(); break;
                        default: throw std::runtime_error("Interpreter: invalid unary op");
                    }
                    break;
                }
                case 2: {
                    auto dst = values.col(idx).head(rem);
                    auto left = values.col(child_map[idx][0]).head(rem);
                    auto right = values.col(child_map[idx][1]).head(rem);
                    switch (node.type) {
                        case NodeType::Add: dst = left + right; break;
                        case NodeType::Sub: dst = left - right; break;
                        case NodeType::Mul: dst = left * right; break;
                        case NodeType::Div: dst = left / right; break;
                        default: throw std::runtime_error("Interpreter: invalid binary op");
                    }
                    break;
                }
            }
        }

        adjoint.topRows(rem).setZero();
        adjoint.col(nn - 1).head(rem).setOnes();

        for (int i = nn - 1; i >= 0; --i) {
            const auto& node = nodes[i];
            if (node.arity == 1) {
                const int child_idx = child_map[static_cast<std::size_t>(i)][0];
                switch (node.type) {
                    case NodeType::Sin:
                        adjoint.col(child_idx).head(rem) += adjoint.col(i).head(rem)
                            * values.col(child_idx).head(rem).cos();
                        break;
                    case NodeType::Cos:
                        adjoint.col(child_idx).head(rem) -= adjoint.col(i).head(rem)
                            * values.col(child_idx).head(rem).sin();
                        break;
                    case NodeType::Exp:
                        adjoint.col(child_idx).head(rem) += adjoint.col(i).head(rem)
                            * values.col(i).head(rem);
                        break;
                    case NodeType::Log:
                        adjoint.col(child_idx).head(rem) += adjoint.col(i).head(rem)
                            / values.col(child_idx).head(rem);
                        break;
                    case NodeType::Tanh:
                        adjoint.col(child_idx).head(rem) += adjoint.col(i).head(rem)
                            * (1.0 - values.col(i).head(rem).square());
                        break;
                    case NodeType::Sqrt:
                        adjoint.col(child_idx).head(rem) += adjoint.col(i).head(rem)
                            / (2.0 * values.col(i).head(rem));
                        break;
                    case NodeType::Abs:
                        adjoint.col(child_idx).head(rem) += adjoint.col(i).head(rem)
                            * values.col(child_idx).head(rem).sign();
                        break;
                    default:
                        adjoint.col(child_idx).head(rem) += adjoint.col(i).head(rem);
                        break;
                }
            } else if (node.arity == 2) {
                const int left_idx  = child_map[static_cast<std::size_t>(i)][0];
                const int right_idx = child_map[static_cast<std::size_t>(i)][1];

                switch (node.type) {
                    case NodeType::Add:
                        adjoint.col(left_idx).head(rem)  += adjoint.col(i).head(rem);
                        adjoint.col(right_idx).head(rem) += adjoint.col(i).head(rem);
                        break;
                    case NodeType::Sub:
                        adjoint.col(left_idx).head(rem)  += adjoint.col(i).head(rem);
                        adjoint.col(right_idx).head(rem) -= adjoint.col(i).head(rem);
                        break;
                    case NodeType::Mul:
                        adjoint.col(left_idx).head(rem)  += adjoint.col(i).head(rem) * values.col(right_idx).head(rem);
                        adjoint.col(right_idx).head(rem) += adjoint.col(i).head(rem) * values.col(left_idx).head(rem);
                        break;
                    case NodeType::Div: {
                        Eigen::ArrayXd denom = values.col(right_idx).head(rem);
                        adjoint.col(left_idx).head(rem)  += adjoint.col(i).head(rem) / denom;
                        adjoint.col(right_idx).head(rem) -= adjoint.col(i).head(rem)
                                                          * values.col(left_idx).head(rem)
                                                          / denom.square();
                        break;
                    }
                    default:
                        break;
                }
            }
        }

        Eigen::VectorXd residual = (
            values.col(nn - 1).head(rem) - target.segment(row0, rem).array())
            .isFinite()
            .select(values.col(nn - 1).head(rem) - target.segment(row0, rem).array(),
                    kInvalidResidual)
            .matrix();
        local_cost += 0.5 * residual.squaredNorm();

        Eigen::MatrixXd batch_j(rem, num_coeffs);
        int col = 0;
        for (int i = 0; i < nn; ++i) {
            if (nodes[i].optimize) {
                auto src = adjoint.col(i).head(rem);
                batch_j.col(col++) = src.isFinite().select(src, 0.0).matrix();
            }
        }
        local_jtj.noalias() += batch_j.transpose() * batch_j;
        local_jtr.noalias() += batch_j.transpose() * residual;
    }

    jtj = Eigen::MatrixXd::Zero(num_coeffs, num_coeffs);
    jtr = Eigen::VectorXd::Zero(num_coeffs);
    cost = 0.0;
    for (int i = 0; i < num_threads; ++i) {
        jtj += jtj_buffers[static_cast<std::size_t>(i)];
        jtr += jtr_buffers[static_cast<std::size_t>(i)];
        cost += cost_buffers[static_cast<std::size_t>(i)];
    }
}

} // namespace imcts
