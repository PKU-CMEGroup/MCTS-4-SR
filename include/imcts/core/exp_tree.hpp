// include/imcts/core/exp_tree.hpp
#pragma once
#include <cstdint>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include "symbol.hpp"
#include "types.hpp"

namespace imcts {

struct StackEntry {
    uint8_t op_idx;
    uint8_t arity;
    uint8_t children_added;
    int     depth;  // tree depth where this operator sits
};

class ExpTree {
public:
    ExpTree(const PrimitiveSet& pset, int max_depth, int max_unary, int max_constants)
        : pset_(&pset), max_depth_(max_depth)
        , max_unary_(max_unary), max_constants_(max_constants)
    {
        update_available_ops();
    }

    void add_op(uint8_t op_idx) {
        if (std::find(available_.begin(), available_.end(), op_idx) == available_.end())
            throw std::runtime_error("add_op: op not in available_ops");

        const Symbol& sym = pset_->symbols[op_idx];
        op_list_.push_back(op_idx);

        if (sym.arity == 1) unary_count_++;
        if (sym.node_type == NodeType::Constant) const_count_++;

        if (!root_set_) root_set_ = true;

        int cur_depth = static_cast<int>(stack_.size());
        StackEntry entry{op_idx, sym.arity, 0, cur_depth};
        stack_.push_back(entry);

        update_stack();
        update_available_ops();
    }

    bool is_empty()    const { return !root_set_; }
    bool is_terminal() const { return root_set_ && stack_.empty(); }

    const std::vector<uint8_t>& available_ops() const { return available_; }
    const std::vector<uint8_t>& get_op_list()   const { return op_list_; }

    int  stack_depth()   const { return static_cast<int>(stack_.size()); }
    int  unary_count()   const { return unary_count_; }
    int  const_count()   const { return const_count_; }
    int  max_depth()     const { return max_depth_; }
    int  max_unary()     const { return max_unary_; }
    int  max_constants() const { return max_constants_; }
    const PrimitiveSet* pset() const { return pset_; }

    // Randomly fill until is_terminal(); returns the ops added
    std::vector<uint8_t> random_fill(RandomGenerator& rng) {
        std::vector<uint8_t> path;
        while (!is_terminal()) {
            const auto& avail = available_ops();
            uint8_t idx = avail[rng() % avail.size()];
            path.push_back(idx);
            add_op(idx);
        }
        return path;
    }

    void reset() {
        op_list_.clear();
        stack_.clear();
        root_set_    = false;
        unary_count_ = 0;
        const_count_ = 0;
        update_available_ops();
    }

private:
    void update_stack() {
        while (!stack_.empty()) {
            auto& top = stack_.back();
            if (top.children_added == top.arity) {
                stack_.pop_back();
                if (!stack_.empty()) {
                    stack_.back().children_added++;
                }
            } else {
                break;
            }
        }
    }

    void update_available_ops() {
        available_.clear();
        // Match the original Python implementation:
        // once the current stack depth reaches max_depth - 1, only leaves can be placed.
        bool at_max_depth = static_cast<int>(stack_.size()) == max_depth_ - 1;

        for (uint8_t i = 0; i < static_cast<uint8_t>(pset_->symbols.size()); i++) {
            const Symbol& sym = pset_->symbols[i];

            // depth constraint: at max_depth only leaves allowed
            if (at_max_depth && sym.arity > 0) continue;

            // unary limit
            if (sym.arity == 1 && unary_count_ >= max_unary_) continue;

            // constant limit
            if (sym.node_type == NodeType::Constant
                && const_count_ >= max_constants_) continue;

            // log/exp mutual exclusion
            if (!stack_.empty()) {
                const Symbol& top_sym = pset_->symbols[stack_.back().op_idx];
                if (top_sym.node_type == NodeType::Log
                    && sym.node_type == NodeType::Exp) continue;
                if (top_sym.node_type == NodeType::Exp
                    && sym.node_type == NodeType::Log) continue;
            }

            // no nested sin/cos inside a sin/cos path
            bool in_sincos = false;
            for (const auto& entry : stack_) {
                const Symbol& s = pset_->symbols[entry.op_idx];
                if (s.node_type == NodeType::Sin
                    || s.node_type == NodeType::Cos) {
                    in_sincos = true; break;
                }
            }
            if (in_sincos && (sym.node_type == NodeType::Sin
                              || sym.node_type == NodeType::Cos)) continue;

            available_.push_back(i);
        }
    }

    const PrimitiveSet*   pset_;
    int                   max_depth_, max_unary_, max_constants_;
    std::vector<uint8_t>  op_list_;
    std::vector<StackEntry> stack_;
    std::vector<uint8_t>  available_;
    bool                  root_set_    = false;
    int                   unary_count_ = 0;
    int                   const_count_ = 0;
};

} // namespace imcts
