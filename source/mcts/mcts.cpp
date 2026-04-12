// source/mcts/mcts.cpp
#include "imcts/mcts/mcts.hpp"
#include <algorithm>
#include <stdexcept>

namespace imcts {

namespace {

bool is_valid_rollout_path(const ExpTree& state, std::span<uint8_t const> path) {
    try {
        ExpTree cloned = state;
        for (uint8_t op : path) {
            cloned.add_op(op);
        }
        return true;
    } catch (...) {
        return false;
    }
}

} // namespace

MCTS::MCTS(const PrimitiveSet& pset, Evaluator& evaluator,
           GPManager& gp_manager, MCTSConfig cfg)
    : pset_(&pset), evaluator_(&evaluator), gp_manager_(&gp_manager), cfg_(cfg)
    , root_(std::make_unique<MCTSNode>(nullptr, 255, cfg.K))
{}

MCTSNode* MCTS::expand_node(MCTSNode* node, ExpTree& state, RandomGenerator& rng) {
    if (node->unexpanded_moves.empty())
        node->unexpanded_moves = state.available_ops();
    int idx = static_cast<int>(rng() % node->unexpanded_moves.size());
    uint8_t mv = node->unexpanded_moves[idx];
    node->unexpanded_moves.erase(node->unexpanded_moves.begin() + idx);
    node->children.push_back(std::make_unique<MCTSNode>(node, mv, cfg_.K));
    ++total_nodes_;
    return node->children.back().get();
}

float MCTS::search(ExpTree& tree, RandomGenerator& rng) {
    MCTSNode* node = root_.get();
    ++count_num_;
    ++node->visits;

    ExpTree state = tree; // copy fresh state for this iteration

    // --- Selection ---
    while (!node->is_leaf()) {
        float u1 = static_cast<float>(rng() % 10000) / 10000.0f;
        if (u1 < cfg_.gp_rate) {
            float u2 = static_cast<float>(rng() % 10000) / 10000.0f;
            float gp_reward;
            if (u2 < cfg_.mutation_rate)
                gp_reward = perform_mutation(node, state, rng);
            else
                gp_reward = perform_crossover(node, state, rng);
            best_reward_ = std::max(best_reward_, gp_reward);
        }

        float u3 = static_cast<float>(rng() % 10000) / 10000.0f;
        MCTSNode* next;
        if (u3 < cfg_.exploration_rate)
            next = node->random_child(rng);
        else
            next = node->choose(cfg_.c, cfg_.gamma);

        if (!next) break;
        node = next;
        ++node->visits;
        state.add_op(node->move);
    }

    // --- Expansion ---
    if (!state.is_terminal()) {
        node = expand_node(node, state, rng);
        ++node->visits;
        state.add_op(node->move);
    }

    // --- Simulation & Backpropagation ---
    std::vector<uint8_t> path;
    float sim_reward = rollout_once(state, nullptr, rng, path);
    best_reward_ = std::max(best_reward_, sim_reward);

    if (path.empty())
        node->is_terminal_flag = true;
    node->backpropagate(path, sim_reward);

    update_terminal_status(node);

    if (node->parent) {
        for (const auto& entry : node->parent->path_queue.entries())
            node->parent->propagate(entry.path, entry.reward);
    }

    return best_reward_;
}

float MCTS::rollout_once(
    ExpTree& state,
    const std::vector<uint8_t>* given_path,
    RandomGenerator& rng,
    std::vector<uint8_t>& out_path)
{
    if (given_path) {
        ExpTree cloned = state;
        for (uint8_t op : *given_path) cloned.add_op(op);
        out_path = *given_path;
        return evaluator_->evaluate(cloned.get_op_list(), rng);
    } else {
        ExpTree cloned = state;
        out_path = cloned.random_fill(rng);
        return evaluator_->evaluate(cloned.get_op_list(), rng);
    }
}

float MCTS::perform_mutation(MCTSNode* node, ExpTree& state, RandomGenerator& rng) {
    try {
        if (node->path_queue.is_empty()) return 0.0f;
        const auto& old_entry = node->path_queue.random_sample(rng);
        auto new_path = gp_manager_->mutate(state, old_entry.path, rng);
        std::vector<uint8_t> out_path;
        float reward = rollout_once(state, &new_path, rng, out_path);
        ++count_num_;
        node->backpropagate(out_path, reward);
        node->propagate(out_path, reward);
        return reward;
    } catch (...) { return 0.0f; }
}

float MCTS::perform_crossover(MCTSNode* node, ExpTree& state, RandomGenerator& rng) {
    if (node->path_queue.is_empty()) return 0.0f;
    const auto& e1 = node->path_queue.random_sample(rng);
    const auto& e2 = node->path_queue.random_sample(rng);

    constexpr int kMaxValidityResamples = 4;
    std::vector<uint8_t> np1, np2;
    bool valid1 = false;
    bool valid2 = false;

    for (int attempt = 0; attempt < kMaxValidityResamples; ++attempt) {
        std::tie(np1, np2) = gp_manager_->crossover(e1.path, e2.path, rng);
        valid1 = is_valid_rollout_path(state, np1);
        valid2 = is_valid_rollout_path(state, np2);
        if (valid1 || valid2) {
            break;
        }
    }

    float best = 0.0f;
    for (const auto& [path, is_valid] : {
             std::pair<std::vector<uint8_t> const&, bool>{np1, valid1},
             std::pair<std::vector<uint8_t> const&, bool>{np2, valid2},
         }) {
        if (!is_valid) {
            continue;
        }
        try {
            std::vector<uint8_t> out_path;
            float reward = rollout_once(state, &path, rng, out_path);
            ++count_num_;
            node->backpropagate(out_path, reward);
            node->propagate(out_path, reward);
            best = std::max(best, reward);
        } catch (...) {}
    }
    return best;
}

void MCTS::update_terminal_status(MCTSNode* node) {
    MCTSNode* cur = node;
    while (cur && cur->parent) {
        MCTSNode* par = cur->parent;
        bool all_terminal = !par->children.empty() && par->unexpanded_moves.empty();
        if (all_terminal) {
            for (auto& ch : par->children)
                if (!ch->is_terminal_flag) { all_terminal = false; break; }
        }
        if (all_terminal && !par->is_terminal_flag) {
            par->is_terminal_flag = true;
            cur = par;
        } else { break; }
    }
}

} // namespace imcts
