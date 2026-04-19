// include/imcts/mcts/node.hpp
#pragma once
#include <cstdint>
#include <vector>
#include <memory>
#include <cmath>
#include <span>
#include "imcts/core/types.hpp"
#include "imcts/core/exp_queue.hpp"
#include "imcts/core/exp_tree.hpp"

namespace imcts {

struct MCTSNode {
    MCTSNode*  parent;
    uint8_t    move;           // operator index that led to this node (255 for root)
    int        visits = 0;
    bool       is_terminal_flag = false;
    ExpQueue   path_queue;
    std::vector<uint8_t>                    unexpanded_moves;
    std::vector<std::unique_ptr<MCTSNode>>  children;

    MCTSNode(MCTSNode* parent, uint8_t move, int K)
        : parent(parent), move(move), path_queue(K) {}

    float ucb(float c, float gamma) const {
        float q = path_queue.best().reward;
        if (parent == nullptr || visits == 0) return q;
        float explore = std::pow(
            c * std::log(static_cast<float>(parent->visits))
              / static_cast<float>(visits),
            gamma);
        return q + explore;
    }

    bool is_leaf() const {
        return children.empty() || !unexpanded_moves.empty();
    }

    // Select child with highest UCB (skip terminal children)
    MCTSNode* choose(float c, float gamma) {
        MCTSNode* best = nullptr;
        float best_ucb = -1e30f;
        for (auto& ch : children) {
            if (ch->is_terminal_flag && ch->visits > 0) continue;
            float u = ch->ucb(c, gamma);
            if (u > best_ucb) { best_ucb = u; best = ch.get(); }
        }
        if (best == nullptr && !children.empty())
            best = children[0].get();
        return best;
    }

    // Random non-terminal child
    MCTSNode* random_child(RandomGenerator& rng) {
        int valid_count = 0;
        for (const auto& ch : children) {
            if (!ch->is_terminal_flag) {
                ++valid_count;
            }
        }
        if (valid_count == 0) return children.empty() ? nullptr : children[0].get();

        int selected = static_cast<int>(rng() % static_cast<std::size_t>(valid_count));
        for (auto& ch : children) {
            if (ch->is_terminal_flag) {
                continue;
            }
            if (selected-- == 0) {
                return ch.get();
            }
        }
        return children.empty() ? nullptr : children[0].get();
    }

    // Propagate (reward, path) upward through the tree
    void backpropagate(std::span<const uint8_t> path, float reward) {
        std::size_t prefix_len = 0;
        for (MCTSNode* cur = this; cur != nullptr && cur->parent != nullptr; cur = cur->parent) {
            ++prefix_len;
        }

        auto full_path = std::make_shared<std::vector<uint8_t>>(prefix_len + path.size());
        std::copy(path.begin(), path.end(), full_path->begin() + static_cast<std::ptrdiff_t>(prefix_len));

        MCTSNode* cur = this;
        std::size_t index = prefix_len;
        while (cur != nullptr && cur->parent != nullptr) {
            (*full_path)[--index] = cur->move;
            cur = cur->parent;
        }

        cur = this;
        std::size_t start = prefix_len;
        while (cur != nullptr) {
            if (!cur->path_queue.append_shared(full_path, start, reward)) {
                break;
            }
            if (cur->parent != nullptr) {
                --start;
            }
            cur = cur->parent;
        }
    }

    // Propagate downward: update children along a known good path
    void propagate(SharedPath path, float reward) {
        MCTSNode* cur = this;
        auto remaining = std::move(path);
        while (!remaining.empty()) {
            MCTSNode* found = nullptr;
            for (auto& ch : cur->children) {
                if (ch->move == remaining[0]) { found = ch.get(); break; }
            }
            if (!found) break;
            remaining = remaining.suffix(1);
            found->path_queue.append(remaining, reward);
            cur = found;
        }
    }

    void propagate(std::span<const uint8_t> path, float reward) {
        propagate(SharedPath(path), reward);
    }
};

} // namespace imcts
