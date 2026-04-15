// source/eval/interpreter.cpp
#include "imcts/eval/interpreter.hpp"
#include "imcts/core/profiling.hpp"
#include <array>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace imcts {

namespace {

constexpr int kMinBatchesForParallelJacobian = 32;
constexpr int kBatchesPerThreadTarget = 8;

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

} // anonymous namespace

void InterpreterWorkspace::prepare_evaluate(std::size_t num_samples, std::size_t num_nodes)
{
    const auto rows = static_cast<Eigen::Index>(num_samples);
    const auto cols = static_cast<Eigen::Index>(num_nodes);
    if (stack_.rows() != kBatchSize || stack_.cols() < cols) {
        stack_.resize(kBatchSize, cols);
    }
    if (result_.size() != rows) {
        result_.resize(rows);
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
    const auto& nodes = tree.nodes();
    workspace.prepare_evaluate(range.size, nodes.size());
    auto& stack = workspace.stack_;
    const auto total = static_cast<Eigen::Index>(range.size);

    for (Eigen::Index row0 = 0; row0 < total; row0 += InterpreterWorkspace::kBatchSize) {
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

std::pair<Eigen::VectorXd, Eigen::MatrixXd>
Interpreter::evaluate_with_jacobian(const Tree& tree, const Dataset& ds, Range range) {
    InterpreterWorkspace workspace;
    evaluate_with_jacobian(tree, ds, range, workspace);
    return {workspace.result(), workspace.jacobian()};
}

void Interpreter::evaluate_with_jacobian(const Tree& tree, const Dataset& ds, Range range,
                                         InterpreterWorkspace& workspace) {
#ifdef IMCTS_ENABLE_PROFILING
    const auto t_total0 = std::chrono::steady_clock::now();
    std::uint64_t prepare_ns = 0;
    std::uint64_t child_map_ns = 0;
    std::uint64_t buffer_setup_ns = 0;
    std::uint64_t forward_ns = 0;
    std::uint64_t reverse_ns = 0;
    std::uint64_t writeback_ns = 0;
#endif
    const auto& nodes = tree.nodes();
    const int n = static_cast<int>(range.size);
    const int nn = static_cast<int>(nodes.size());
    int num_coeffs = count_coefficients(nodes);

#ifdef IMCTS_ENABLE_PROFILING
    const auto t_prepare0 = std::chrono::steady_clock::now();
#endif
    workspace.prepare_jacobian(range.size, nodes.size(), static_cast<std::size_t>(num_coeffs));
#ifdef IMCTS_ENABLE_PROFILING
    prepare_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t_prepare0).count());
    const auto t_child_map0 = std::chrono::steady_clock::now();
#endif
    const auto total = static_cast<Eigen::Index>(range.size);
    fill_child_map(nodes, std::span<std::array<int, 2>>(workspace.child_map_.data(), nodes.size()));
    const auto child_map = std::span<const std::array<int, 2>>(workspace.child_map_.data(), nodes.size());
    const auto num_batches = static_cast<int>((total + InterpreterWorkspace::kBatchSize - 1)
                                              / InterpreterWorkspace::kBatchSize);
#ifdef IMCTS_ENABLE_PROFILING
    child_map_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t_child_map0).count());
    const auto t_buffer0 = std::chrono::steady_clock::now();
#endif

    int num_threads = 1;
#ifdef _OPENMP
    if (num_batches >= kMinBatchesForParallelJacobian) {
        const int max_threads = omp_get_max_threads();
        const int target_threads = std::max(2, num_batches / kBatchesPerThreadTarget);
        num_threads = std::min(max_threads, target_threads);
    }
#endif

    std::vector<Eigen::ArrayXXd> value_buffers;
    std::vector<Eigen::ArrayXXd> adjoint_buffers;
    if (num_threads > 1) {
        value_buffers.reserve(static_cast<std::size_t>(num_threads));
        adjoint_buffers.reserve(static_cast<std::size_t>(num_threads));
        for (int i = 0; i < num_threads; ++i) {
            value_buffers.emplace_back(InterpreterWorkspace::kBatchSize,
                                       static_cast<Eigen::Index>(nodes.size()));
            adjoint_buffers.emplace_back(InterpreterWorkspace::kBatchSize,
                                         static_cast<Eigen::Index>(nodes.size()));
        }
    }
#ifdef IMCTS_ENABLE_PROFILING
    buffer_setup_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t_buffer0).count());
#endif

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (num_batches > 1) num_threads(num_threads)
#endif
    for (int batch = 0; batch < num_batches; ++batch) {
        int thread_id = 0;
#ifdef _OPENMP
        thread_id = omp_get_thread_num();
#endif
        auto& values = num_threads == 1
            ? workspace.values_
            : value_buffers[static_cast<std::size_t>(thread_id)];
        auto& adjoint = num_threads == 1
            ? workspace.adjoint_
            : adjoint_buffers[static_cast<std::size_t>(thread_id)];
        const Eigen::Index row0 = static_cast<Eigen::Index>(batch) * InterpreterWorkspace::kBatchSize;
        const Eigen::Index rem = std::min(InterpreterWorkspace::kBatchSize, total - row0);

#ifdef IMCTS_ENABLE_PROFILING
        const auto t_forward0 = std::chrono::steady_clock::now();
#endif
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
#ifdef IMCTS_ENABLE_PROFILING
        const auto forward_elapsed = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t_forward0).count());
#ifdef _OPENMP
#pragma omp atomic
#endif
        forward_ns += forward_elapsed;
        const auto t_reverse0 = std::chrono::steady_clock::now();
#endif

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
#ifdef IMCTS_ENABLE_PROFILING
        const auto reverse_elapsed = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t_reverse0).count());
#ifdef _OPENMP
#pragma omp atomic
#endif
        reverse_ns += reverse_elapsed;
        const auto t_writeback0 = std::chrono::steady_clock::now();
#endif

        int col = 0;
        for (int i = 0; i < nn; ++i) {
            if (nodes[i].optimize) {
                workspace.jacobian_.block(row0, col++, rem, 1) = adjoint.col(i).head(rem).matrix();
            }
        }
        workspace.result_.segment(row0, rem) = values.col(nn - 1).head(rem).matrix();
#ifdef IMCTS_ENABLE_PROFILING
        const auto writeback_elapsed = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t_writeback0).count());
#ifdef _OPENMP
#pragma omp atomic
#endif
        writeback_ns += writeback_elapsed;
#endif
    }

#ifdef IMCTS_ENABLE_PROFILING
    profiling::record_jacobian_call(
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t_total0).count()),
        prepare_ns,
        child_map_ns,
        buffer_setup_ns,
        forward_ns,
        reverse_ns,
        writeback_ns,
        static_cast<std::uint64_t>(num_batches),
        static_cast<std::uint64_t>(nn),
        static_cast<std::uint64_t>(num_coeffs),
        static_cast<std::uint64_t>(range.size));
#endif
}

} // namespace imcts
