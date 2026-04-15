#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "imcts/core/profiling.hpp"
#include "imcts/core/symbol.hpp"
#include "imcts/evaluator/evaluator.hpp"

namespace {

using Clock = std::chrono::steady_clock;

double ns_to_ms(std::uint64_t ns)
{
    return static_cast<double>(ns) / 1'000'000.0;
}

double ns_to_us(std::uint64_t ns)
{
    return static_cast<double>(ns) / 1'000.0;
}

imcts::EvaluatorConfig make_config(int n_samples, int lm_iterations)
{
    std::vector<std::vector<float>> x_cols(2, std::vector<float>(n_samples));
    std::vector<float> y(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        const float x0 = static_cast<float>(i) / static_cast<float>(std::max(1, n_samples - 1));
        const float x1 = std::sin(0.013f * static_cast<float>(i));
        x_cols[0][i] = x0;
        x_cols[1][i] = x1;
        y[i] = 0.75f * std::sin(x0 + x1) + std::log(std::abs(x0 - 0.2f) + 1e-3f);
    }
    return imcts::EvaluatorConfig{std::move(x_cols), std::move(y), lm_iterations};
}

std::vector<uint8_t> make_prefix(const imcts::PrimitiveSet& pset, std::initializer_list<const char*> ops)
{
    std::vector<uint8_t> prefix;
    prefix.reserve(ops.size());
    for (const char* op : ops) {
        prefix.push_back(pset.op_index(op));
    }
    return prefix;
}

void print_evaluator_stats(const std::string& name, const imcts::profiling::EvaluatorStats& stats, std::uint64_t wall_ns)
{
    std::cout << "\n[Evaluator] " << name << '\n';
    std::cout << "  calls                : " << stats.calls << '\n';
    std::cout << "  cache_hits           : " << stats.cache_hits << '\n';
    std::cout << "  cache_misses         : " << stats.cache_misses << '\n';
    std::cout << "  calls_with_constants : " << stats.calls_with_constants << '\n';
    std::cout << "  calls_without_consts : " << stats.calls_without_constants << '\n';
    std::cout << "  wall_total_ms        : " << ns_to_ms(wall_ns) << '\n';
    if (stats.calls > 0) {
        const double calls = static_cast<double>(stats.calls);
        std::cout << "  avg_total_us         : " << ns_to_us(stats.total_ns) / calls << '\n';
        std::cout << "  avg_to_tree_us       : " << ns_to_us(stats.to_tree_ns) / calls << '\n';
        std::cout << "  avg_hash_lookup_us   : " << ns_to_us(stats.hash_lookup_ns) / calls << '\n';
        std::cout << "  avg_optimize_us      : " << ns_to_us(stats.optimize_ns) / calls << '\n';
        std::cout << "  avg_forward_eval_us  : " << ns_to_us(stats.forward_eval_ns) / calls << '\n';
        std::cout << "  avg_reward_us        : " << ns_to_us(stats.reward_ns) / calls << '\n';
        std::cout << "  avg_cache_store_us   : " << ns_to_us(stats.cache_store_ns) / calls << '\n';
        std::cout << "  avg_prefix_len       : " << static_cast<double>(stats.total_prefix_length) / calls << '\n';
    }
}

void run_evaluator_profile(const std::string& name,
                           const imcts::PrimitiveSet& pset,
                           const imcts::EvaluatorConfig& cfg,
                           const std::vector<std::vector<uint8_t>>& workload,
                           int outer_loops)
{
    imcts::profiling::reset();
    imcts::Evaluator evaluator(pset, cfg);
    imcts::RandomGenerator rng{42};

    const auto wall0 = Clock::now();
    float reward_sum = 0.0f;
    for (int loop = 0; loop < outer_loops; ++loop) {
        for (const auto& prefix : workload) {
            reward_sum += evaluator.evaluate(prefix, rng);
        }
    }
    const auto wall_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - wall0).count());

    auto stats = imcts::profiling::evaluator_stats();
    print_evaluator_stats(name, stats, wall_ns);
    std::cout << "  reward_checksum      : " << reward_sum << '\n';
}

} // namespace

int main()
{
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "IMCTS evaluator profiling\n";

    const auto cfg_small = make_config(512, 30);
    const auto cfg_large = make_config(4096, 50);

    const auto pset_full = imcts::make_primitive_set({"+", "-", "*", "/", "sin", "cos", "exp", "log", "abs", "R"}, 2);
    const auto pset_no_r = imcts::make_primitive_set({"+", "-", "*", "/", "sin", "cos", "exp", "log", "abs"}, 2);

    const auto no_const_simple = make_prefix(pset_no_r, {"+", "x0", "x1"});
    const auto no_const_function = make_prefix(pset_no_r, {"+", "sin", "+", "x0", "x1", "log", "abs", "-", "x0", "x1"});
    const auto no_const_simple_full = make_prefix(pset_full, {"+", "x0", "x1"});
    const auto no_const_function_full = make_prefix(pset_full, {"+", "sin", "+", "x0", "x1", "log", "abs", "-", "x0", "x1"});

    const auto with_const_simple = make_prefix(pset_full, {"+", "*", "R", "x0", "R"});
    const auto with_const_function = make_prefix(pset_full, {"+", "*", "R", "sin", "+", "x0", "x1", "log", "abs", "-", "x0", "R"});

    run_evaluator_profile(
        "small / no constants / repeated same expression",
        pset_no_r,
        cfg_small,
        {no_const_function},
        500);

    run_evaluator_profile(
        "small / with constants / repeated same expression",
        pset_full,
        cfg_small,
        {with_const_function},
        100);

    run_evaluator_profile(
        "large / mixed / mostly cache misses",
        pset_full,
        cfg_large,
        {
            no_const_simple_full,
            no_const_function_full,
            with_const_simple,
            with_const_function,
        },
        40);

    run_evaluator_profile(
        "large / with constants / hot-cache repeated",
        pset_full,
        cfg_large,
        {with_const_simple, with_const_function},
        100);

    return 0;
}
