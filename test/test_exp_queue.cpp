// test/test_exp_queue.cpp
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "imcts/core/exp_queue.hpp"

TEST_CASE("ExpQueue keeps K best, evicts worst") {
    imcts::ExpQueue q(3);
    REQUIRE(q.is_empty());

    q.append({0, 1, 2}, 0.5f);
    q.append({3, 4}, 0.8f);
    q.append({5}, 0.3f);
    REQUIRE(q.size() == 3);

    // Insert better than the lowest entry.
    q.append({6, 7}, 0.9f);
    REQUIRE(q.size() == 3);
    REQUIRE(q.best().reward == Catch::Approx(0.9f));

    // Insert worse than the lowest entry; it should not enter the queue.
    q.append({8}, 0.1f);
    REQUIRE(q.size() == 3);
}

TEST_CASE("ExpQueue near-duplicate rejection") {
    imcts::ExpQueue q(5);
    q.append({0}, 0.5f);

    // diff < 1e-5 should be rejected
    const bool inserted = q.append({1}, 0.5f + 1e-6f);
    REQUIRE_FALSE(inserted);
    REQUIRE(q.size() == 1);
}

TEST_CASE("ExpQueue best on empty returns reward 0") {
    imcts::ExpQueue q(5);
    REQUIRE(q.best().reward == Catch::Approx(0.0f));
}

TEST_CASE("ExpQueue random_sample returns valid entry") {
    imcts::ExpQueue q(3);
    q.append({0}, 0.5f);
    q.append({1}, 0.7f);
    imcts::RandomGenerator rng{42};
    const auto& entry = q.random_sample(rng);
    REQUIRE(entry.reward > 0.0f);
}
