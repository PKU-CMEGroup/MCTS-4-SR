// source/gp/gp_manager.cpp
#include "imcts/gp/gp_manager.hpp"
#include <algorithm>
#include <stdexcept>

namespace imcts {

namespace {

bool is_constant_op(const PrimitiveSet& pset, uint8_t op) {
    return pset.symbols[op].node_type == NodeType::Constant;
}

int count_constants(std::span<uint8_t const> path, const PrimitiveSet& pset) {
    int count = 0;
    for (uint8_t op : path) {
        if (is_constant_op(pset, op)) {
            ++count;
        }
    }
    return count;
}

int count_unary(std::span<uint8_t const> path, const PrimitiveSet& pset) {
    int count = 0;
    for (uint8_t op : path) {
        if (pset.symbols[op].arity == 1) {
            ++count;
        }
    }
    return count;
}

int expression_depth(std::span<uint8_t const> path, const PrimitiveSet& pset) {
    if (path.empty()) {
        return 0;
    }

    std::vector<int> stack = {1};
    int max_depth = 0;
    for (uint8_t op : path) {
        if (stack.empty()) {
            break;
        }

        const int current_depth = stack.back();
        stack.pop_back();
        max_depth = std::max(max_depth, current_depth);

        const int arity = pset.symbols[op].arity;
        for (int i = 0; i < arity; ++i) {
            stack.push_back(current_depth + 1);
        }
    }

    return max_depth;
}

bool same_path(std::span<uint8_t const> lhs, std::span<uint8_t const> rhs) {
    return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

} // namespace

GPManager::GPManager(const PrimitiveSet& pset) : pset_(&pset) {}

int GPManager::subtree_size(std::span<uint8_t const> path, int index) const {
    if (index >= static_cast<int>(path.size())) return 0;
    struct Frame { int pos; int remaining; };
    std::vector<Frame> stk;
    stk.push_back({index, pset_->symbols[path[index]].arity});
    int size = 1;
    int cur = index + 1;
    while (!stk.empty()) {
        auto& top = stk.back();
        if (top.remaining == 0) {
            stk.pop_back();
        } else {
            if (cur >= static_cast<int>(path.size())) break;
            top.remaining--;
            int arity = pset_->symbols[path[cur]].arity;
            stk.push_back({cur, arity});
            size++;
            cur++;
        }
    }
    return size;
}

int GPManager::depth_at(std::span<uint8_t const> path, int target) const {
    if (target < 0 || target >= static_cast<int>(path.size())) return 0;
    std::vector<int> stack = {1};
    for (int i = 0; i < static_cast<int>(path.size()); i++) {
        if (stack.empty()) break;
        int cur_depth = stack.back(); stack.pop_back();
        if (i == target) return cur_depth;
        int arity = pset_->symbols[path[i]].arity;
        for (int c = 0; c < arity; c++) stack.push_back(cur_depth + 1);
    }
    return 0;
}

std::vector<uint8_t> GPManager::node_replace(
    const ExpTree& state,
    std::span<uint8_t const> path,
    RandomGenerator& rng,
    int num_replacements) const
{
    std::vector<uint8_t> new_path(path.begin(), path.end());
    int constant_count = state.const_count() + count_constants(path, *pset_);

    for (int r = 0; r < num_replacements; r++) {
        if (new_path.empty()) break;
        int idx = static_cast<int>(rng() % new_path.size());
        uint8_t replace_op = new_path[idx];
        uint8_t old_arity = pset_->symbols[replace_op].arity;
        std::vector<uint8_t> candidates;
        for (uint8_t s = 0; s < static_cast<uint8_t>(pset_->symbols.size()); s++) {
            if (pset_->symbols[s].arity != old_arity || s == replace_op) {
                continue;
            }
            if (constant_count == state.max_constants() && is_constant_op(*pset_, s)) {
                continue;
            }
            candidates.push_back(s);
        }
        if (!candidates.empty()) {
            uint8_t new_op = candidates[rng() % candidates.size()];
            new_path[idx] = new_op;
            constant_count += static_cast<int>(is_constant_op(*pset_, new_op))
                           - static_cast<int>(is_constant_op(*pset_, replace_op));
        }
    }
    return new_path;
}

std::vector<uint8_t> GPManager::shrink_mutate(
    const ExpTree& /*state*/,
    std::span<uint8_t const> path,
    RandomGenerator& rng) const
{
    if (path.empty()) return {path.begin(), path.end()};
    std::vector<int> valid;
    for (int i = 0; i < static_cast<int>(path.size()); i++) {
        if (pset_->symbols[path[i]].arity > 0) valid.push_back(i);
    }
    if (valid.empty()) return {path.begin(), path.end()};
    int idx = valid[rng() % valid.size()];
    int sub_size = subtree_size(path, idx);
    uint8_t arity = pset_->symbols[path[idx]].arity;
    std::vector<uint8_t> result;
    if (arity == 1) {
        result.insert(result.end(), path.begin(), path.begin() + idx);
        result.insert(result.end(), path.begin() + idx + 1, path.end());
    } else {
        int left_start = idx + 1;
        int left_size  = subtree_size(path, left_start);
        if (rng() % 2 == 0) {
            result.insert(result.end(), path.begin(), path.begin() + idx);
            result.insert(result.end(), path.begin() + left_start,
                          path.begin() + left_start + left_size);
            result.insert(result.end(), path.begin() + idx + sub_size, path.end());
        } else {
            int right_start = left_start + left_size;
            int right_size  = sub_size - 1 - left_size;
            result.insert(result.end(), path.begin(), path.begin() + idx);
            result.insert(result.end(), path.begin() + right_start,
                          path.begin() + right_start + right_size);
            result.insert(result.end(), path.begin() + idx + sub_size, path.end());
        }
    }
    return result;
}

std::vector<uint8_t> GPManager::uniform_mutate(
    const ExpTree& state,
    std::span<uint8_t const> path,
    RandomGenerator& rng) const
{
    if (path.empty()) return {path.begin(), path.end()};
    int idx      = static_cast<int>(rng() % path.size());
    int sub_size = subtree_size(path, idx);
    int cur_depth = depth_at(path, idx);
    int replacement_max_depth = std::max(1, state.max_depth() - cur_depth + 1);

    std::vector<uint8_t> outside_subtree_path;
    outside_subtree_path.reserve(path.size() - sub_size);
    outside_subtree_path.insert(outside_subtree_path.end(), path.begin(), path.begin() + idx);
    outside_subtree_path.insert(
        outside_subtree_path.end(),
        path.begin() + idx + sub_size,
        path.end());

    int remaining_unary_budget = state.max_unary() - state.unary_count()
                               - count_unary(outside_subtree_path, *pset_);
    int remaining_constant_budget = state.max_constants() - state.const_count()
                                  - count_constants(outside_subtree_path, *pset_);

    ExpTree subtree(*pset_, replacement_max_depth, remaining_unary_budget,
                    remaining_constant_budget);
    auto new_sub = subtree.random_fill(rng);

    std::vector<uint8_t> result;
    result.insert(result.end(), path.begin(), path.begin() + idx);
    result.insert(result.end(), new_sub.begin(), new_sub.end());
    result.insert(result.end(), path.begin() + idx + sub_size, path.end());
    return result;
}

std::vector<uint8_t> GPManager::insert_mutate(
    const ExpTree& state,
    std::span<uint8_t const> path,
    RandomGenerator& rng) const
{
    if (path.empty())
        return {path.begin(), path.end()};

    int idx      = static_cast<int>(rng() % path.size());
    int sub_size = subtree_size(path, idx);
    int cur_depth = depth_at(path, idx);
    int selected_subtree_depth = expression_depth(path.subspan(idx, sub_size), *pset_);
    if (cur_depth + selected_subtree_depth > state.max_depth())
        return {path.begin(), path.end()};

    int sibling_max_depth = std::max(1, state.max_depth() - cur_depth);
    int remaining_unary_budget = state.max_unary() - state.unary_count()
                               - count_unary(path, *pset_);
    int remaining_constant_budget = state.max_constants() - state.const_count()
                                  - count_constants(path, *pset_);

    std::vector<uint8_t> insert_ops;
    for (uint8_t s = 0; s < static_cast<uint8_t>(pset_->symbols.size()); s++) {
        int a = pset_->symbols[s].arity;
        if (a == 0) continue;
        if (a == 1 && remaining_unary_budget == 0) continue;
        insert_ops.push_back(s);
    }
    if (insert_ops.empty()) return {path.begin(), path.end()};

    uint8_t insert_op    = insert_ops[rng() % insert_ops.size()];
    uint8_t insert_arity = pset_->symbols[insert_op].arity;

    std::vector<uint8_t> result;
    if (insert_arity == 1) {
        result.insert(result.end(), path.begin(), path.begin() + idx);
        result.push_back(insert_op);
        result.insert(result.end(), path.begin() + idx, path.end());
    } else {
        ExpTree sibling(*pset_, sibling_max_depth, remaining_unary_budget,
                        remaining_constant_budget);
        auto sib_path = sibling.random_fill(rng);

        result.insert(result.end(), path.begin(), path.begin() + idx);
        result.push_back(insert_op);
        if (rng() % 2 == 0) {
            result.insert(result.end(), sib_path.begin(), sib_path.end());
            result.insert(result.end(), path.begin() + idx, path.end());
        } else {
            result.insert(result.end(), path.begin() + idx,
                          path.begin() + idx + sub_size);
            result.insert(result.end(), sib_path.begin(), sib_path.end());
            result.insert(result.end(), path.begin() + idx + sub_size, path.end());
        }
    }
    return result;
}

std::vector<uint8_t> GPManager::mutate(
    const ExpTree& state,
    std::span<uint8_t const> path,
    RandomGenerator& rng) const
{
    int choice = static_cast<int>(rng() % 4);
    std::vector<uint8_t> new_path;
    switch (choice) {
        case 0: new_path = node_replace(state, path, rng); break;
        case 1: new_path = shrink_mutate(state, path, rng); break;
        case 2: new_path = uniform_mutate(state, path, rng); break;
        case 3: new_path = insert_mutate(state, path, rng); break;
        default: new_path = {path.begin(), path.end()}; break;
    }
    if (new_path == std::vector<uint8_t>(path.begin(), path.end()))
        new_path = uniform_mutate(state, path, rng);
    return new_path;
}

std::pair<std::vector<uint8_t>, std::vector<uint8_t>>
GPManager::crossover(
    std::span<uint8_t const> p1,
    std::span<uint8_t const> p2,
    RandomGenerator& rng) const
{
    if (p1.empty() || p2.empty())
        return {{p1.begin(),p1.end()},{p2.begin(),p2.end()}};

    constexpr int kMaxNoOpResamples = 8;
    std::vector<uint8_t> np1, np2;

    for (int attempt = 0; attempt < kMaxNoOpResamples; ++attempt) {
        int idx1 = static_cast<int>(rng() % p1.size());
        int idx2 = static_cast<int>(rng() % p2.size());
        int s1   = subtree_size(p1, idx1);
        int s2   = subtree_size(p2, idx2);

        np1.clear();
        np2.clear();
        np1.reserve(p1.size() - s1 + s2);
        np2.reserve(p2.size() - s2 + s1);
        np1.insert(np1.end(), p1.begin(), p1.begin() + idx1);
        np1.insert(np1.end(), p2.begin() + idx2, p2.begin() + idx2 + s2);
        np1.insert(np1.end(), p1.begin() + idx1 + s1, p1.end());

        np2.insert(np2.end(), p2.begin(), p2.begin() + idx2);
        np2.insert(np2.end(), p1.begin() + idx1, p1.begin() + idx1 + s1);
        np2.insert(np2.end(), p2.begin() + idx2 + s2, p2.end());

        if (!same_path(np1, p1) || !same_path(np2, p2)) {
            break;
        }
    }

    return {np1, np2};
}

} // namespace imcts
