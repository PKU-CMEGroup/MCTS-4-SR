// include/imcts/core/symbol.hpp
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <stdexcept>
#include <unordered_map>
#include "node.hpp"

namespace imcts {

struct Symbol {
    NodeType node_type;
    uint8_t  arity;
    uint8_t  var_index; // only valid for Variable nodes
};

struct PrimitiveSet {
    std::vector<Symbol>       symbols;
    std::vector<uint8_t>      leaf_indices;    // arity == 0
    std::vector<uint8_t>      inner_indices;   // arity > 0
    std::vector<std::string>  names;

    uint8_t op_index(const std::string& name) const {
        for (uint8_t i = 0; i < static_cast<uint8_t>(names.size()); i++) {
            if (names[i] == name) return i;
        }
        throw std::runtime_error("Unknown op: " + name);
    }
};

inline PrimitiveSet make_primitive_set(
    const std::vector<std::string>& ops,
    int n_vars)
{
    static const std::unordered_map<std::string, std::pair<NodeType, uint8_t>> kOpMap = {
        {"+",    {NodeType::Add,      2}},
        {"-",    {NodeType::Sub,      2}},
        {"*",    {NodeType::Mul,      2}},
        {"/",    {NodeType::Div,      2}},
        {"sin",  {NodeType::Sin,      1}},
        {"cos",  {NodeType::Cos,      1}},
        {"exp",  {NodeType::Exp,      1}},
        {"log",  {NodeType::Log,      1}},
        {"tanh", {NodeType::Tanh,     1}},
        {"sqrt", {NodeType::Sqrt,     1}},
        {"abs",  {NodeType::Abs,      1}},
        {"R",    {NodeType::Constant, 0}},
    };

    PrimitiveSet pset;
    uint8_t idx = 0;

    for (const auto& op : ops) {
        auto it = kOpMap.find(op);
        if (it == kOpMap.end())
            throw std::runtime_error("Unsupported op: " + op);
        const auto [node_type, arity] = it->second;
        pset.symbols.push_back({node_type, arity, 0});
        pset.names.push_back(op);
        if (arity == 0) pset.leaf_indices.push_back(idx);
        else            pset.inner_indices.push_back(idx);
        ++idx;
    }

    for (int i = 0; i < n_vars; i++) {
        pset.symbols.push_back({NodeType::Variable, 0, static_cast<uint8_t>(i)});
        pset.names.push_back("x" + std::to_string(i));
        pset.leaf_indices.push_back(idx++);
    }

    return pset;
}

} // namespace imcts
