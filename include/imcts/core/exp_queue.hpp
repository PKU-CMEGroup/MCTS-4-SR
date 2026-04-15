// include/imcts/core/exp_queue.hpp
#pragma once
#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include "profiling.hpp"
#include "types.hpp"

namespace imcts {

struct QueueEntry {
    std::vector<uint8_t> path;
    float                reward = 0.0f;
};

class ExpQueue {
public:
    explicit ExpQueue(int max_size) : max_size_(max_size) {
        entries_.reserve(static_cast<std::size_t>(std::max(0, max_size_)));
    }

    // Returns true if inserted; false if rejected (near-duplicate or too low)
    bool append(std::vector<uint8_t> path, float reward) {
        return append(std::span<const uint8_t>(path), reward);
    }

    // Returns true if inserted; false if rejected (near-duplicate or too low)
    bool append(std::span<const uint8_t> path, float reward) {
#ifdef IMCTS_ENABLE_PROFILING
        const auto t_total0 = std::chrono::steady_clock::now();
        std::uint64_t duplicate_scan_ns = 0;
        std::uint64_t lower_bound_ns = 0;
        std::uint64_t insert_ns = 0;
        const auto queue_size_before = static_cast<std::uint64_t>(entries_.size());
        const auto path_size = static_cast<std::uint64_t>(path.size());
#endif
        if (!std::isfinite(reward)) return false;

        // near-duplicate rejection
        constexpr float kDupThresh = 1e-5f;
        if (max_size_ <= 0) return false;

#ifdef IMCTS_ENABLE_PROFILING
        const auto t_lb0 = std::chrono::steady_clock::now();
#endif
        const auto pos = std::lower_bound(
            entries_.begin(), entries_.end(), reward,
            [](const QueueEntry& entry, float candidate_reward) {
                return entry.reward > candidate_reward;
            });
#ifdef IMCTS_ENABLE_PROFILING
        lower_bound_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t_lb0).count());
        const auto t_dup0 = std::chrono::steady_clock::now();
#endif

        auto is_duplicate = [&](const auto& it) {
            return it != entries_.end() && std::abs(it->reward - reward) < kDupThresh;
        };
        if (is_duplicate(pos)
            || (pos != entries_.begin() && is_duplicate(std::prev(pos)))) {
#ifdef IMCTS_ENABLE_PROFILING
            duplicate_scan_ns = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - t_dup0).count());
            profiling::record_exp_queue_append(
                static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - t_total0).count()),
                duplicate_scan_ns,
                lower_bound_ns,
                insert_ns,
                false,
                true,
                false,
                path_size,
                queue_size_before);
#endif
            return false;
        }
#ifdef IMCTS_ENABLE_PROFILING
        duplicate_scan_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t_dup0).count());
#endif

        if (static_cast<int>(entries_.size()) == max_size_ && reward <= entries_.back().reward) {
#ifdef IMCTS_ENABLE_PROFILING
            profiling::record_exp_queue_append(
                static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - t_total0).count()),
                duplicate_scan_ns,
                lower_bound_ns,
                insert_ns,
                false,
                false,
                true,
                path_size,
                queue_size_before);
#endif
            return false;
        }

#ifdef IMCTS_ENABLE_PROFILING
        const auto t_insert0 = std::chrono::steady_clock::now();
#endif

        entries_.insert(
            pos,
            QueueEntry{std::vector<uint8_t>(path.begin(), path.end()), reward});

        if (static_cast<int>(entries_.size()) > max_size_) {
            entries_.pop_back();
        }
#ifdef IMCTS_ENABLE_PROFILING
        insert_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t_insert0).count());
        profiling::record_exp_queue_append(
            static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - t_total0).count()),
            duplicate_scan_ns,
            lower_bound_ns,
            insert_ns,
            true,
            false,
            false,
            path_size,
            queue_size_before);
#endif
        return true;
    }

    const QueueEntry& best() const {
        static const QueueEntry kEmpty{{}, 0.0f};
        return entries_.empty() ? kEmpty : entries_.front();
    }

    const QueueEntry& random_sample(RandomGenerator& rng) const {
        return entries_[rng() % entries_.size()];
    }

    bool is_empty() const { return entries_.empty(); }
    int  size()     const { return static_cast<int>(entries_.size()); }

    const std::vector<QueueEntry>& entries() const { return entries_; }

private:
    int                    max_size_;
    std::vector<QueueEntry> entries_;
};

} // namespace imcts
