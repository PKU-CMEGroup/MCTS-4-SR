// source/core/bridge.cpp
#include "imcts/core/bridge.hpp"
#include <stdexcept>

namespace imcts {

static int build_postfix(
    std::span<uint8_t const> prefix,
    int pos,
    const PrimitiveSet& pset,
    std::vector<Node>& nodes)
{
    if (pos >= static_cast<int>(prefix.size()))
        throw std::runtime_error("Bridge: prefix sequence too short");

    uint8_t op_idx = prefix[pos++];
    const Symbol& sym = pset.symbols[op_idx];

    for (int c = 0; c < sym.arity; c++) {
        pos = build_postfix(prefix, pos, pset, nodes);
    }

    Node node(sym.node_type);
    node.arity = sym.arity;

    if (sym.node_type == NodeType::Variable) {
        node.var_index = sym.var_index;
        node.optimize = false;
    } else if (sym.node_type == NodeType::Constant) {
        node.value = 1.0;
        node.optimize = true;
    }

    nodes.push_back(node);
    return pos;
}

Tree Bridge::to_tree(std::span<uint8_t const> prefix) const
{
    Tree tree;
    to_tree(prefix, tree);
    return tree;
}

void Bridge::to_tree(std::span<uint8_t const> prefix, Tree& out) const
{
    auto& nodes = out.nodes();
    nodes.clear();
    nodes.reserve(prefix.size());
    build_postfix(prefix, 0, *pset_, nodes);
    out.update_lengths();
}

std::optional<float> Bridge::cache_get(Hash hash) const {
    auto it = cache_.find(hash);
    if (it == cache_.end()) return std::nullopt;
    return it->second;
}

void Bridge::cache_put(Hash hash, float reward) {
    cache_[hash] = reward;
}

void Bridge::cache_clear() {
    cache_.clear();
}

} // namespace imcts
