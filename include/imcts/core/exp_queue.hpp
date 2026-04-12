// include/imcts/core/exp_queue.hpp
#pragma once
#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#include "types.hpp"

namespace imcts {

struct QueueEntry {
    std::vector<uint8_t> path;
    float                reward = 0.0f;
};

class ExpQueue {
public:
    explicit ExpQueue(int max_size) : max_size_(max_size) {}

    // Returns true if inserted; false if rejected (near-duplicate or too low)
    bool append(std::vector<uint8_t> path, float reward) {
        if (!std::isfinite(reward)) return false;

        // near-duplicate rejection
        constexpr float kDupThresh = 1e-5f;
        for (const auto& e : entries_) {
            if (std::abs(e.reward - reward) < kDupThresh) return false;
        }

        if (static_cast<int>(entries_.size()) < max_size_) {
            entries_.push_back({std::move(path), reward});
            sort_desc();
            return true;
        }
        // full: only insert if better than worst
        if (reward <= entries_.back().reward) return false;
        entries_.back() = {std::move(path), reward};
        sort_desc();
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
    void sort_desc() {
        std::sort(entries_.begin(), entries_.end(),
                  [](const auto& a, const auto& b) { return a.reward > b.reward; });
    }

    int                    max_size_;
    std::vector<QueueEntry> entries_;
};

} // namespace imcts
