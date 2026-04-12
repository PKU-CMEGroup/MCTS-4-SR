#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "node.hpp"
#include "types.hpp"

namespace imcts {

class Tree {
public:
    Tree() = default;
    explicit Tree(std::vector<Node> nodes) : nodes_(std::move(nodes)) {}

    // Recompute length fields for all nodes (postfix order).
    Tree& update_lengths();

    // Number of nodes with optimize == true.
    [[nodiscard]] std::size_t num_coefficients() const;

    // Get optimizable coefficient values (in postfix order).
    [[nodiscard]] std::vector<Scalar> get_coefficients() const;

    // Set optimizable coefficient values (in postfix order).
    void set_coefficients(const std::vector<Scalar>& coeffs);
    void set_coefficients(std::span<const Scalar> coeffs);

    // Structure-only hash: based on type, arity, and var_index only.
    // Ignores constant values to avoid cache inconsistency.
    [[nodiscard]] Hash structure_hash() const;
    [[nodiscard]] std::string to_string() const;

    [[nodiscard]] std::size_t length() const { return nodes_.size(); }
    [[nodiscard]] bool empty() const { return nodes_.empty(); }

    std::vector<Node>& nodes() { return nodes_; }
    [[nodiscard]] const std::vector<Node>& nodes() const { return nodes_; }

    Node& operator[](std::size_t i) { return nodes_[i]; }
    const Node& operator[](std::size_t i) const { return nodes_[i]; }

private:
    std::vector<Node> nodes_;  // postfix order (children before parent).
};

} // namespace imcts
