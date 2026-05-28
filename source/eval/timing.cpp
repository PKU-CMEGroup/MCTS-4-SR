#include "imcts/eval/timing.hpp"

#ifdef IMCTS_ENABLE_TIMING
#include <atomic>
#endif

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

#ifdef IMCTS_ENABLE_TIMING
struct TimingCounter {
    std::atomic<std::uint64_t> calls{0};
    std::atomic<std::uint64_t> nanoseconds{0};
};

std::array<TimingCounter, static_cast<std::size_t>(TimingSection::Count)> g_timing_counters;
#endif

std::size_t section_index(TimingSection section)
{
    return static_cast<std::size_t>(section);
}

} // namespace

void reset_timing_stats()
{
#ifdef IMCTS_ENABLE_TIMING
    for (auto& counter : g_timing_counters) {
        counter.calls.store(0, std::memory_order_relaxed);
        counter.nanoseconds.store(0, std::memory_order_relaxed);
    }
#endif
}

void record_timing(TimingSection section, std::chrono::nanoseconds elapsed)
{
#ifdef IMCTS_ENABLE_TIMING
    const auto idx = section_index(section);
    if (idx >= g_timing_counters.size()) {
        return;
    }

    g_timing_counters[idx].calls.fetch_add(1, std::memory_order_relaxed);
    g_timing_counters[idx].nanoseconds.fetch_add(
        static_cast<std::uint64_t>(elapsed.count()), std::memory_order_relaxed);
#else
    (void)section;
    (void)elapsed;
#endif
}

TimingSnapshot timing_stats()
{
    TimingSnapshot snapshot{};
    for (std::size_t i = 0; i < kSectionNames.size(); ++i) {
#ifdef IMCTS_ENABLE_TIMING
        const auto calls = g_timing_counters[i].calls.load(std::memory_order_relaxed);
        const auto nanoseconds = g_timing_counters[i].nanoseconds.load(std::memory_order_relaxed);
        snapshot[i] = TimingEntry{
            kSectionNames[i],
            calls,
            static_cast<double>(nanoseconds) / 1e9,
        };
#else
        snapshot[i] = TimingEntry{
            kSectionNames[i],
            0,
            0.0,
        };
#endif
    }
    return snapshot;
}

} // namespace imcts
