#include "imcts/eval/timing.hpp"

#include <atomic>

namespace imcts {

namespace {

constexpr std::array<std::string_view, static_cast<std::size_t>(TimingSection::Count)> kSectionNames = {
    "coefficient_optimize",
    "bridge_to_tree",
    "lm_residual",
    "lm_jacobian",
    "mcts_backpropagate",
    "mcts_crossover",
    "mcts_mutation",
    "mcts_rollout",
    "mcts_search",
    "normal_equation_accumulate",
    "optimizer_lm_minimize",
    "interpreter_evaluate",
    "interpreter_evaluate_residual",
    "interpreter_evaluate_with_jacobian",
};

struct TimingCounter {
    std::atomic<std::uint64_t> calls{0};
    std::atomic<std::uint64_t> nanoseconds{0};
};

std::array<TimingCounter, static_cast<std::size_t>(TimingSection::Count)> g_timing_counters;

std::size_t section_index(TimingSection section)
{
    return static_cast<std::size_t>(section);
}

} // namespace

void reset_timing_stats()
{
    for (auto& counter : g_timing_counters) {
        counter.calls.store(0, std::memory_order_relaxed);
        counter.nanoseconds.store(0, std::memory_order_relaxed);
    }
}

void record_timing(TimingSection section, std::chrono::nanoseconds elapsed)
{
    const auto idx = section_index(section);
    if (idx >= g_timing_counters.size()) {
        return;
    }

    g_timing_counters[idx].calls.fetch_add(1, std::memory_order_relaxed);
    g_timing_counters[idx].nanoseconds.fetch_add(
        static_cast<std::uint64_t>(elapsed.count()), std::memory_order_relaxed);
}

TimingSnapshot timing_stats()
{
    TimingSnapshot snapshot{};
    for (std::size_t i = 0; i < g_timing_counters.size(); ++i) {
        const auto calls = g_timing_counters[i].calls.load(std::memory_order_relaxed);
        const auto nanoseconds = g_timing_counters[i].nanoseconds.load(std::memory_order_relaxed);
        snapshot[i] = TimingEntry{
            kSectionNames[i],
            calls,
            static_cast<double>(nanoseconds) / 1e9,
        };
    }
    return snapshot;
}

} // namespace imcts
