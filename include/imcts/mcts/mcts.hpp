// include/imcts/mcts/mcts.hpp
#pragma once
#include <memory>
#include <vector>
#include <cstdint>
#include <limits>
#include <span>
#include "imcts/core/types.hpp"
#include "imcts/mcts/node.hpp"
#include "imcts/evaluator/evaluator.hpp"
#include "imcts/gp/gp_manager.hpp"

namespace imcts {

struct MCTSConfig {
    int   K                = 500;
    float c                = 4.0f;
    float gamma            = 0.5f;
    float gp_rate          = 0.2f;
    float mutation_rate    = 0.2f;
    float exploration_rate = 0.2f;
    float succ_error_tol   = 1e-6f;
};

class MCTS {
public:
    MCTS(const PrimitiveSet& pset,
         Evaluator& evaluator,
         GPManager& gp_manager,
         MCTSConfig cfg = {});

    float search(ExpTree& tree, RandomGenerator& rng);

    float best_reward()  const { return best_reward_; }
    int   count()        const { return count_num_; }
    int   total_nodes()  const { return total_nodes_; }

    std::vector<uint8_t> best_path() const {
        return root_->path_queue.best().path.to_vector();
    }

private:
    float rollout_once(ExpTree& state,
                       RandomGenerator& rng,
                       std::vector<uint8_t>& out_path);
    float rollout_once(ExpTree& state,
                       std::span<uint8_t const> given_path,
                       RandomGenerator& rng,
                       std::vector<uint8_t>& out_path);

    float perform_mutation(MCTSNode* node, ExpTree& state, RandomGenerator& rng);
    float perform_crossover(MCTSNode* node, ExpTree& state, RandomGenerator& rng);
    void  update_terminal_status(MCTSNode* node);

    MCTSNode* expand_node(MCTSNode* node, ExpTree& state, RandomGenerator& rng);

    const PrimitiveSet* pset_;
    Evaluator*          evaluator_;
    GPManager*          gp_manager_;
    MCTSConfig          cfg_;

    std::unique_ptr<MCTSNode> root_;
    int   count_num_   = 0;
    int   total_nodes_ = 1;
    float best_reward_ = -std::numeric_limits<float>::infinity();
};

} // namespace imcts
