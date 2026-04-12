// include/imcts/core/bridge.hpp
#pragma once
#include <cstdint>
#include <vector>
#include <span>
#include <optional>
#include <unordered_map>
#include "tree.hpp"
#include "symbol.hpp"
#include "types.hpp"

namespace imcts {

class Bridge {
public:
    explicit Bridge(const PrimitiveSet& pset) : pset_(&pset) {}

    // Convert prefix sequence to postfix Tree
    Tree to_tree(std::span<uint8_t const> prefix) const;
    void to_tree(std::span<uint8_t const> prefix, Tree& out) const;

    // Cache interface (keyed by structure hash)
    std::optional<float> cache_get(Hash hash) const;
    void cache_put(Hash hash, float reward);
    void cache_clear();

private:
    const PrimitiveSet* pset_;
    std::unordered_map<Hash, float> cache_;
};

} // namespace imcts
