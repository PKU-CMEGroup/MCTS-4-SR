// include/imcts/core/exp_queue.hpp
#pragma once
#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstddef>
#include <memory>
#include <span>
#include "types.hpp"

namespace imcts {

class SharedPath {
public:
    SharedPath() = default;

    explicit SharedPath(std::span<const uint8_t> path)
        : storage_(std::make_shared<std::vector<uint8_t>>(path.begin(), path.end()))
        , offset_(0) {}

    SharedPath(std::shared_ptr<const std::vector<uint8_t>> storage, std::size_t offset)
        : storage_(std::move(storage))
        , offset_(offset) {}

    [[nodiscard]] std::span<const uint8_t> span() const {
        if (!storage_ || offset_ >= storage_->size()) {
            return {};
        }
        return std::span<const uint8_t>(storage_->data() + static_cast<std::ptrdiff_t>(offset_),
                                        storage_->size() - offset_);
    }

    [[nodiscard]] bool empty() const { return size() == 0; }
    [[nodiscard]] std::size_t size() const { return span().size(); }

    [[nodiscard]] uint8_t operator[](std::size_t index) const {
        return span()[index];
    }

    [[nodiscard]] auto begin() const { return span().begin(); }
    [[nodiscard]] auto end() const { return span().end(); }

    [[nodiscard]] SharedPath suffix(std::size_t count) const {
        const auto next_offset = std::min(offset_ + count, storage_ ? storage_->size() : offset_ + count);
        return SharedPath(storage_, next_offset);
    }

    [[nodiscard]] std::vector<uint8_t> to_vector() const {
        const auto view = span();
        return {view.begin(), view.end()};
    }

private:
    std::shared_ptr<const std::vector<uint8_t>> storage_;
    std::size_t offset_ = 0;
};

struct QueueEntry {
    SharedPath path;
    float      reward = 0.0f;
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
        return append_impl(SharedPath(path), reward);
    }

    bool append(SharedPath path, float reward) {
        return append_impl(std::move(path), reward);
    }

    bool append_shared(const std::shared_ptr<const std::vector<uint8_t>>& storage,
                       std::size_t offset,
                       float reward) {
        SharedPath path(storage, offset);
        return append_impl(std::move(path), reward);
    }

    const QueueEntry& best() const {
        static const QueueEntry kEmpty{SharedPath{}, 0.0f};
        return entries_.empty() ? kEmpty : entries_.front();
    }

    const QueueEntry& random_sample(RandomGenerator& rng) const {
        return entries_[rng() % entries_.size()];
    }

    bool is_empty() const { return entries_.empty(); }
    int  size()     const { return static_cast<int>(entries_.size()); }

    const std::vector<QueueEntry>& entries() const { return entries_; }

private:
    bool append_impl(SharedPath path, float reward) {
        if (!std::isfinite(reward)) return false;

        // near-duplicate rejection
        constexpr float kDupThresh = 1e-5f;
        if (max_size_ <= 0) return false;

        const auto pos = std::lower_bound(
            entries_.begin(), entries_.end(), reward,
            [](const QueueEntry& entry, float candidate_reward) {
                return entry.reward > candidate_reward;
            });

        auto is_duplicate = [&](const auto& it) {
            return it != entries_.end() && std::abs(it->reward - reward) < kDupThresh;
        };
        if (is_duplicate(pos)
            || (pos != entries_.begin() && is_duplicate(std::prev(pos)))) {
            return false;
        }

        if (static_cast<int>(entries_.size()) == max_size_ && reward <= entries_.back().reward) {
            return false;
        }

        entries_.insert(pos, QueueEntry{std::move(path), reward});

        if (static_cast<int>(entries_.size()) > max_size_) {
            entries_.pop_back();
        }
        return true;
    }

    int                     max_size_;
    std::vector<QueueEntry> entries_;
};

} // namespace imcts
