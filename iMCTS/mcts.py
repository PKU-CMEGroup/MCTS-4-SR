from __future__ import annotations
import random
import copy
from typing import List, Tuple, Optional
import numpy as np

from iMCTS.src import ExpTree, Exp_Queue, Optimizer
from iMCTS.gp import GPManager


class MCTS_Node:
    """Monte Carlo Tree Search node implementation."""
    
    def __init__(
        self,
        mcts: MCTS,
        parent: Optional[MCTS_Node] = None,
        move: str = ""
    ) -> None:
        self.mcts = mcts  # Reference to the MCTS instance for parameters
        self.parent = parent
        self.move = move
        self.children: List[MCTS_Node] = []
        self.visits: int = 0
        self.path_queue = Exp_Queue(max_size=mcts.K)
        self.value: float = 0
        self.unexpanded_moves: List[str] = []
        self.is_terminal: bool = False

    def expand(self, state: ExpTree) -> MCTS_Node:
        """Expand the node by creating a new child node."""
        if not self.unexpanded_moves:
            self.unexpanded_moves = list(state.available_ops)
        
        child = None
        if self.unexpanded_moves:
            selected_index = random.randrange(len(self.unexpanded_moves))
            move = self.unexpanded_moves.pop(selected_index)
            child = MCTS_Node(mcts=self.mcts, parent=self, move=move)
            self.children.append(child)
        
        return child

    @property
    def ucb(self) -> float:
        """Calculate Upper Confidence Bound (UCB) value."""
        q_value = self.path_queue.best()[1]
        exploration_weight = (
            self.mcts.c * np.log(self.parent.visits) / self.visits
        ) ** self.mcts.gamma
        return q_value + exploration_weight
    
    def choose(self) -> MCTS_Node:
        """Select child node with highest UCB value."""
        selected_node = max(self.children, key=lambda c: c.ucb)
        
        if selected_node.is_terminal and selected_node.visits > 0:
            selected_node = max(
                (c for c in self.children if not c.is_terminal),
                default=selected_node,
                key=lambda c: c.ucb
            )
        return selected_node
    
    def random_child(self) -> MCTS_Node:
        """Select a random child node."""
        valid_children = [child for child in self.children if not child.is_terminal]
        if valid_children:
            return random.choice(valid_children)
        return random.choice(self.children)
    
    def backpropagate(self, path: List[Tuple], value: float) -> None:
        """Backpropagate the simulation results through the tree."""
        current_node = self
        while current_node:
            if not current_node.path_queue.append(path, value):
                break
            path = [current_node.move] + path
            current_node = current_node.parent

    def propagate(self, path: List[Tuple], value: float) -> None:
        """Propagate results down the tree hierarchy."""
        current_node = self
        for idx, action in enumerate(path):
            current_node = next(
                (child for child in current_node.children if child.move == action),
                None
            )
            if not current_node:
                break
            current_node.path_queue.append(path[idx + 1:], value)
            # current_node.visits += 1

    def is_leaf(self) -> bool:
        """Check if the node is a leaf node."""
        return not self.children or self.unexpanded_moves


class MCTS:
    """Monte Carlo Tree Search implementation for symbolic regression."""
    
    def __init__(
        self,
        optimizer: Optimizer,
        gp_manager: GPManager,
        gp_rate: float = 0.2,
        mutation_rate: float = 0.2,
        exploration_rate: float = 0.2,
        K: int = 500,
        c: float = 4,
        gamma: float = 0.5,
        verbose: bool = False,
        succ_error_tol: float = 1e-6
    ) -> None:
        self.optimizer = optimizer
        self.gp_manager = gp_manager
        self.gp_rate = gp_rate
        self.mutation_rate = mutation_rate
        self.exploration_rate = exploration_rate
        self.K = K
        self.c = c
        self.gamma = gamma
        self.verbose = verbose
        self.succ_error_tol = succ_error_tol
        
        # Initialize root node with reference to this MCTS instance
        self.root = MCTS_Node(mcts=self)
        self.exp_queue = Exp_Queue(max_size=10)
        self.count_num: int = 0
        self.best_reward: float = -np.inf
        self.total_nodes: int = 1  # Root node

    def search(self, exp_tree: ExpTree) -> None:
        """Execute the MCTS search algorithm."""
        node = self.root
        self.count_num += 1
        node.visits += 1
        state = copy.deepcopy(exp_tree)
        
        # Selection phase
        while not node.is_leaf():
            if random.random() < self.gp_rate:
                if random.random() < self.mutation_rate:
                    gp_reward = self._perform_mutation(node, state)
                else:
                    gp_reward = self._perform_crossover(node, state)
                self.best_reward = max(self.best_reward, gp_reward)

            if random.random() < self.exploration_rate:
                node = node.random_child()
            else:
                node = node.choose()
            node.visits += 1
            state.add_op(node.move)
        
        # Expansion phase
        if not state.is_terminal():
            node = node.expand(state)
            self.total_nodes += 1
            node.visits += 1
            state.add_op(node.move)
        
        # Simulation and backpropagation
        simulation_reward, path = self.rollout_once(state)
        self.best_reward = max(self.best_reward, simulation_reward)
        
        if not path:
            node.is_terminal = True
        node.backpropagate(path, simulation_reward)

        self._update_terminal_status(node)
        
        # Update parent nodes
        for parent_path, parent_reward in node.parent.path_queue.list:
            node.parent.propagate(parent_path, parent_reward)
            
        return self.best_reward
    
    def _update_terminal_status(self, node: MCTS_Node) -> None:
        """Recursively update terminal status of nodes."""
        current = node
        while current and current.parent:
            parent = current.parent
            # Check if all children of parent are terminal
            if (parent.children and 
                all(child.is_terminal for child in parent.children) and
                not parent.unexpanded_moves):
                if not parent.is_terminal:
                    parent.is_terminal = True
                    current = parent  # Continue checking upward
                else:
                    break  # Already terminal, no need to continue
            else:
                break  # Not all children are terminal, stop here

    def _perform_mutation(self, node: MCTS_Node, state: ExpTree) -> float:
        """Execute mutation operation."""
        try:
            old_path = node.path_queue.random_sample()[0]
            new_path = self.gp_manager.mutate(state, old_path)
            reward, path = self.rollout_once(state, new_path)
            self.count_num += 1
            node.backpropagate(path, reward)
            node.propagate(path, reward)
        except:
            reward = 0
        return reward

    def _perform_crossover(self, node: MCTS_Node, state: ExpTree) -> float:
        """Execute crossover operation."""
        path1 = node.path_queue.random_sample()[0]
        path2 = node.path_queue.random_sample()[0]
        new_path1, new_path2 = self.gp_manager.crossover(state, path1, path2)
        
        rewards = []
        for path in [new_path1, new_path2]:
            try:
                reward, path = self.rollout_once(state, path)
                self.count_num += 1
                node.backpropagate(path, reward)
                node.propagate(path, reward)
                rewards.append(reward)
            except ValueError:
                rewards.append(0)
        return max(rewards)

    def reward(self, state: ExpTree) -> Tuple[float, bool]:
        """Calculate reward for the given state."""
        expression, reward = self.optimizer.optimize_constants(state)
        self.exp_queue.append(expression, reward)
        return reward

    def rollout_once(self, state: ExpTree, path: Optional[List[str]] = None) -> Tuple[float, List[str]]:
        """Perform a single simulation rollout."""
        if path:
            cloned_state = copy.deepcopy(state)
            for op in path:
                cloned_state.add_op(op)
            return self.reward(cloned_state), path
        else:
            filled_state, path = state.random_fill()
            return self.reward(filled_state), path