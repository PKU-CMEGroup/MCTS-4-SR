#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <string_view>

namespace imcts {

enum class TimingSection : std::size_t {
    CoefficientOptimize = 0,
    BridgeToTree,
    LMResidual,
    LMJacobian,
    MCTSBackpropagate,
    MCTSCrossover,
    MCTSMutation,
    MCTSRollout,
    MCTSSearch,
    NormalEquationAccumulate,
    OptimizerLMMinimize,
    InterpreterEvaluate,
    InterpreterEvaluateResidual,
    InterpreterEvaluateWithJacobian,
    Count,
};

struct TimingEntry {
    std::string_view name;
    std::uint64_t calls;
    double total_seconds;
};

using TimingSnapshot = std::array<TimingEntry, static_cast<std::size_t>(TimingSection::Count)>;

void reset_timing_stats();
void record_timing(TimingSection section, std::chrono::nanoseconds elapsed);
TimingSnapshot timing_stats();

class ScopedTimer {
public:
    explicit ScopedTimer(TimingSection section)
        : section_(section)
        , start_(std::chrono::steady_clock::now())
    {}

    ~ScopedTimer()
    {
        record_timing(section_, std::chrono::steady_clock::now() - start_);
    }

    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    TimingSection section_;
    std::chrono::steady_clock::time_point start_;
};

} // namespace imcts
