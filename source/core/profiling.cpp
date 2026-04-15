#include "imcts/core/profiling.hpp"

#include <atomic>

namespace imcts::profiling {

namespace {

struct ExpQueueAtomicStats {
    std::atomic<std::uint64_t> append_calls{0};
    std::atomic<std::uint64_t> inserted{0};
    std::atomic<std::uint64_t> rejected_duplicate{0};
    std::atomic<std::uint64_t> rejected_low{0};
    std::atomic<std::uint64_t> total_ns{0};
    std::atomic<std::uint64_t> duplicate_scan_ns{0};
    std::atomic<std::uint64_t> lower_bound_ns{0};
    std::atomic<std::uint64_t> insert_ns{0};
    std::atomic<std::uint64_t> total_path_bytes{0};
    std::atomic<std::uint64_t> total_queue_size_before{0};
};

struct JacobianAtomicStats {
    std::atomic<std::uint64_t> calls{0};
    std::atomic<std::uint64_t> total_ns{0};
    std::atomic<std::uint64_t> prepare_ns{0};
    std::atomic<std::uint64_t> child_map_ns{0};
    std::atomic<std::uint64_t> buffer_setup_ns{0};
    std::atomic<std::uint64_t> forward_ns{0};
    std::atomic<std::uint64_t> reverse_ns{0};
    std::atomic<std::uint64_t> writeback_ns{0};
    std::atomic<std::uint64_t> total_batches{0};
    std::atomic<std::uint64_t> total_nodes{0};
    std::atomic<std::uint64_t> total_coefficients{0};
    std::atomic<std::uint64_t> total_samples{0};
};

struct EvaluatorAtomicStats {
    std::atomic<std::uint64_t> calls{0};
    std::atomic<std::uint64_t> cache_hits{0};
    std::atomic<std::uint64_t> cache_misses{0};
    std::atomic<std::uint64_t> calls_with_constants{0};
    std::atomic<std::uint64_t> calls_without_constants{0};
    std::atomic<std::uint64_t> total_ns{0};
    std::atomic<std::uint64_t> to_tree_ns{0};
    std::atomic<std::uint64_t> hash_lookup_ns{0};
    std::atomic<std::uint64_t> optimize_ns{0};
    std::atomic<std::uint64_t> forward_eval_ns{0};
    std::atomic<std::uint64_t> reward_ns{0};
    std::atomic<std::uint64_t> cache_store_ns{0};
    std::atomic<std::uint64_t> total_prefix_length{0};
};

ExpQueueAtomicStats& exp_queue_atomic_stats()
{
    static ExpQueueAtomicStats stats;
    return stats;
}

JacobianAtomicStats& jacobian_atomic_stats()
{
    static JacobianAtomicStats stats;
    return stats;
}

EvaluatorAtomicStats& evaluator_atomic_stats()
{
    static EvaluatorAtomicStats stats;
    return stats;
}

template <typename T>
void store_zero(T& value)
{
    value.store(0, std::memory_order_relaxed);
}

template <typename T>
std::uint64_t load_relaxed(const T& value)
{
    return value.load(std::memory_order_relaxed);
}

} // namespace

void reset()
{
    auto& eq = exp_queue_atomic_stats();
    store_zero(eq.append_calls);
    store_zero(eq.inserted);
    store_zero(eq.rejected_duplicate);
    store_zero(eq.rejected_low);
    store_zero(eq.total_ns);
    store_zero(eq.duplicate_scan_ns);
    store_zero(eq.lower_bound_ns);
    store_zero(eq.insert_ns);
    store_zero(eq.total_path_bytes);
    store_zero(eq.total_queue_size_before);

    auto& jac = jacobian_atomic_stats();
    store_zero(jac.calls);
    store_zero(jac.total_ns);
    store_zero(jac.prepare_ns);
    store_zero(jac.child_map_ns);
    store_zero(jac.buffer_setup_ns);
    store_zero(jac.forward_ns);
    store_zero(jac.reverse_ns);
    store_zero(jac.writeback_ns);
    store_zero(jac.total_batches);
    store_zero(jac.total_nodes);
    store_zero(jac.total_coefficients);
    store_zero(jac.total_samples);

    auto& eval = evaluator_atomic_stats();
    store_zero(eval.calls);
    store_zero(eval.cache_hits);
    store_zero(eval.cache_misses);
    store_zero(eval.calls_with_constants);
    store_zero(eval.calls_without_constants);
    store_zero(eval.total_ns);
    store_zero(eval.to_tree_ns);
    store_zero(eval.hash_lookup_ns);
    store_zero(eval.optimize_ns);
    store_zero(eval.forward_eval_ns);
    store_zero(eval.reward_ns);
    store_zero(eval.cache_store_ns);
    store_zero(eval.total_prefix_length);
}

ExpQueueStats exp_queue_stats()
{
    const auto& src = exp_queue_atomic_stats();
    return ExpQueueStats{
        .append_calls = load_relaxed(src.append_calls),
        .inserted = load_relaxed(src.inserted),
        .rejected_duplicate = load_relaxed(src.rejected_duplicate),
        .rejected_low = load_relaxed(src.rejected_low),
        .total_ns = load_relaxed(src.total_ns),
        .duplicate_scan_ns = load_relaxed(src.duplicate_scan_ns),
        .lower_bound_ns = load_relaxed(src.lower_bound_ns),
        .insert_ns = load_relaxed(src.insert_ns),
        .total_path_bytes = load_relaxed(src.total_path_bytes),
        .total_queue_size_before = load_relaxed(src.total_queue_size_before),
    };
}

JacobianStats jacobian_stats()
{
    const auto& src = jacobian_atomic_stats();
    return JacobianStats{
        .calls = load_relaxed(src.calls),
        .total_ns = load_relaxed(src.total_ns),
        .prepare_ns = load_relaxed(src.prepare_ns),
        .child_map_ns = load_relaxed(src.child_map_ns),
        .buffer_setup_ns = load_relaxed(src.buffer_setup_ns),
        .forward_ns = load_relaxed(src.forward_ns),
        .reverse_ns = load_relaxed(src.reverse_ns),
        .writeback_ns = load_relaxed(src.writeback_ns),
        .total_batches = load_relaxed(src.total_batches),
        .total_nodes = load_relaxed(src.total_nodes),
        .total_coefficients = load_relaxed(src.total_coefficients),
        .total_samples = load_relaxed(src.total_samples),
    };
}

EvaluatorStats evaluator_stats()
{
    const auto& src = evaluator_atomic_stats();
    return EvaluatorStats{
        .calls = load_relaxed(src.calls),
        .cache_hits = load_relaxed(src.cache_hits),
        .cache_misses = load_relaxed(src.cache_misses),
        .calls_with_constants = load_relaxed(src.calls_with_constants),
        .calls_without_constants = load_relaxed(src.calls_without_constants),
        .total_ns = load_relaxed(src.total_ns),
        .to_tree_ns = load_relaxed(src.to_tree_ns),
        .hash_lookup_ns = load_relaxed(src.hash_lookup_ns),
        .optimize_ns = load_relaxed(src.optimize_ns),
        .forward_eval_ns = load_relaxed(src.forward_eval_ns),
        .reward_ns = load_relaxed(src.reward_ns),
        .cache_store_ns = load_relaxed(src.cache_store_ns),
        .total_prefix_length = load_relaxed(src.total_prefix_length),
    };
}

void record_exp_queue_append(
    std::uint64_t total_ns,
    std::uint64_t duplicate_scan_ns,
    std::uint64_t lower_bound_ns,
    std::uint64_t insert_ns,
    bool inserted,
    bool rejected_duplicate,
    bool rejected_low,
    std::uint64_t path_size,
    std::uint64_t queue_size_before)
{
#ifdef IMCTS_ENABLE_PROFILING
    auto& stats = exp_queue_atomic_stats();
    stats.append_calls.fetch_add(1, std::memory_order_relaxed);
    if (inserted) {
        stats.inserted.fetch_add(1, std::memory_order_relaxed);
    }
    if (rejected_duplicate) {
        stats.rejected_duplicate.fetch_add(1, std::memory_order_relaxed);
    }
    if (rejected_low) {
        stats.rejected_low.fetch_add(1, std::memory_order_relaxed);
    }
    stats.total_ns.fetch_add(total_ns, std::memory_order_relaxed);
    stats.duplicate_scan_ns.fetch_add(duplicate_scan_ns, std::memory_order_relaxed);
    stats.lower_bound_ns.fetch_add(lower_bound_ns, std::memory_order_relaxed);
    stats.insert_ns.fetch_add(insert_ns, std::memory_order_relaxed);
    stats.total_path_bytes.fetch_add(path_size, std::memory_order_relaxed);
    stats.total_queue_size_before.fetch_add(queue_size_before, std::memory_order_relaxed);
#else
    (void)total_ns;
    (void)duplicate_scan_ns;
    (void)lower_bound_ns;
    (void)insert_ns;
    (void)inserted;
    (void)rejected_duplicate;
    (void)rejected_low;
    (void)path_size;
    (void)queue_size_before;
#endif
}

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
    std::uint64_t samples)
{
#ifdef IMCTS_ENABLE_PROFILING
    auto& stats = jacobian_atomic_stats();
    stats.calls.fetch_add(1, std::memory_order_relaxed);
    stats.total_ns.fetch_add(total_ns, std::memory_order_relaxed);
    stats.prepare_ns.fetch_add(prepare_ns, std::memory_order_relaxed);
    stats.child_map_ns.fetch_add(child_map_ns, std::memory_order_relaxed);
    stats.buffer_setup_ns.fetch_add(buffer_setup_ns, std::memory_order_relaxed);
    stats.forward_ns.fetch_add(forward_ns, std::memory_order_relaxed);
    stats.reverse_ns.fetch_add(reverse_ns, std::memory_order_relaxed);
    stats.writeback_ns.fetch_add(writeback_ns, std::memory_order_relaxed);
    stats.total_batches.fetch_add(batches, std::memory_order_relaxed);
    stats.total_nodes.fetch_add(nodes, std::memory_order_relaxed);
    stats.total_coefficients.fetch_add(coefficients, std::memory_order_relaxed);
    stats.total_samples.fetch_add(samples, std::memory_order_relaxed);
#else
    (void)total_ns;
    (void)prepare_ns;
    (void)child_map_ns;
    (void)buffer_setup_ns;
    (void)forward_ns;
    (void)reverse_ns;
    (void)writeback_ns;
    (void)batches;
    (void)nodes;
    (void)coefficients;
    (void)samples;
#endif
}

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
    std::uint64_t prefix_length)
{
#ifdef IMCTS_ENABLE_PROFILING
    auto& stats = evaluator_atomic_stats();
    stats.calls.fetch_add(1, std::memory_order_relaxed);
    if (cache_hit) {
        stats.cache_hits.fetch_add(1, std::memory_order_relaxed);
    } else {
        stats.cache_misses.fetch_add(1, std::memory_order_relaxed);
    }
    if (has_constants) {
        stats.calls_with_constants.fetch_add(1, std::memory_order_relaxed);
    } else {
        stats.calls_without_constants.fetch_add(1, std::memory_order_relaxed);
    }
    stats.total_ns.fetch_add(total_ns, std::memory_order_relaxed);
    stats.to_tree_ns.fetch_add(to_tree_ns, std::memory_order_relaxed);
    stats.hash_lookup_ns.fetch_add(hash_lookup_ns, std::memory_order_relaxed);
    stats.optimize_ns.fetch_add(optimize_ns, std::memory_order_relaxed);
    stats.forward_eval_ns.fetch_add(forward_eval_ns, std::memory_order_relaxed);
    stats.reward_ns.fetch_add(reward_ns, std::memory_order_relaxed);
    stats.cache_store_ns.fetch_add(cache_store_ns, std::memory_order_relaxed);
    stats.total_prefix_length.fetch_add(prefix_length, std::memory_order_relaxed);
#else
    (void)total_ns;
    (void)to_tree_ns;
    (void)hash_lookup_ns;
    (void)optimize_ns;
    (void)forward_eval_ns;
    (void)reward_ns;
    (void)cache_store_ns;
    (void)cache_hit;
    (void)has_constants;
    (void)prefix_length;
#endif
}

} // namespace imcts::profiling
