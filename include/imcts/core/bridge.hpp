// include/imcts/core/bridge.hpp
#pragma once
#include <cstdint>
#include <vector>
#include <span>
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

private:
    const PrimitiveSet* pset_;
};

} // namespace imcts
