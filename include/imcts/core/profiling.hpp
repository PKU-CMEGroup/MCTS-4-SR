#pragma once

#include <cstdint>

namespace imcts::profiling {

struct ExpQueueStats {
    std::uint64_t append_calls = 0;
    std::uint64_t inserted = 0;
    std::uint64_t rejected_duplicate = 0;
    std::uint64_t rejected_low = 0;
    std::uint64_t total_ns = 0;
    std::uint64_t duplicate_scan_ns = 0;
    std::uint64_t lower_bound_ns = 0;
    std::uint64_t insert_ns = 0;
    std::uint64_t total_path_bytes = 0;
    std::uint64_t total_queue_size_before = 0;
};

struct JacobianStats {
    std::uint64_t calls = 0;
    std::uint64_t total_ns = 0;
    std::uint64_t prepare_ns = 0;
    std::uint64_t child_map_ns = 0;
    std::uint64_t buffer_setup_ns = 0;
    std::uint64_t forward_ns = 0;
    std::uint64_t reverse_ns = 0;
    std::uint64_t writeback_ns = 0;
    std::uint64_t total_batches = 0;
    std::uint64_t total_nodes = 0;
    std::uint64_t total_coefficients = 0;
    std::uint64_t total_samples = 0;
};

struct EvaluatorStats {
    std::uint64_t calls = 0;
    std::uint64_t cache_hits = 0;
    std::uint64_t cache_misses = 0;
    std::uint64_t calls_with_constants = 0;
    std::uint64_t calls_without_constants = 0;
    std::uint64_t total_ns = 0;
    std::uint64_t to_tree_ns = 0;
    std::uint64_t hash_lookup_ns = 0;
    std::uint64_t optimize_ns = 0;
    std::uint64_t forward_eval_ns = 0;
    std::uint64_t reward_ns = 0;
    std::uint64_t cache_store_ns = 0;
    std::uint64_t total_prefix_length = 0;
};

void reset();

ExpQueueStats exp_queue_stats();
JacobianStats jacobian_stats();
EvaluatorStats evaluator_stats();

void record_exp_queue_append(
    std::uint64_t total_ns,
    std::uint64_t duplicate_scan_ns,
    std::uint64_t lower_bound_ns,
    std::uint64_t insert_ns,
    bool inserted,
    bool rejected_duplicate,
    bool rejected_low,
    std::uint64_t path_size,
    std::uint64_t queue_size_before);

void record_jacobian_call(
    std::uint64_t total_ns,
    std::uint64_t prepare_ns,
    std::uint64_t child_map_ns,
    std::uint64_t buffer_setup_ns,
    std::uint64_t forward_ns,
    std::uint64_t reverse_ns,
    std::uint64_t writeback_ns,
    std::uint64_t batches,
    std::uint64_t nodes,
    std::uint64_t coefficients,
    std::uint64_t samples);

void record_evaluator_call(
    std::uint64_t total_ns,
    std::uint64_t to_tree_ns,
    std::uint64_t hash_lookup_ns,
    std::uint64_t optimize_ns,
    std::uint64_t forward_eval_ns,
    std::uint64_t reward_ns,
    std::uint64_t cache_store_ns,
    bool cache_hit,
    bool has_constants,
    std::uint64_t prefix_length);

} // namespace imcts::profiling
