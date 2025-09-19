from typing import List, Dict, Tuple
import numpy as np
import random
import copy
from iMCTS.src import ExpTree

class GPManager:
    def __init__(self, ops: List[str], arity_dict: Dict[str, int], verbose: bool = True):
        """Initialize the GPManager class."""
        self.ops = ops
        self.arity_dict = arity_dict
        self.verbose = verbose

        # Calculate the occurrence count of each arity
        arity_count = {}
        for op in self.ops:
            arity = self.arity_dict[op]
            arity_count[arity] = arity_count.get(arity, 0) + 1
        
        # Record arities that appear only once to determine if they can be effectively replaced
        self.unique_arities = [arity for arity, count in arity_count.items() if count == 1]

        self.mutations = [
            self.node_replace,
            self.shrink_mutate,
            self.uniform_mutate,
            self.insert_mutate,
        ]   

    def node_replace(self, old_state: 'ExpTree', path: List[str], num_replacements: int = 1) -> List[str]:
        """Replace a node in the path with a new operator."""
        path = copy.copy(path)
        constant_count = old_state.constant_count + sum(op in ['C', 'R'] for op in path)

        for _ in range(num_replacements):
            if path:
                # Cannot replace with unique arities
                valid_indices = [i for i in range(len(path)) if self.arity_dict[path[i]] not in self.unique_arities]
                if not valid_indices:
                    return path
                index = np.random.choice(valid_indices) 
                replace_op = path[index]
                # Arity must match and cannot be the same operator
                available_ops = [op for op in self.ops if self.arity_dict[op] == self.arity_dict[replace_op] and op != replace_op]

                if constant_count == old_state.max_constants:
                    available_ops = [op for op in available_ops if op not in ['C', 'R']]

                if available_ops:
                    new_op = np.random.choice(available_ops)
                    path[index] = new_op
                    constant_count += (new_op in ['C', 'R']) - (replace_op in ['C', 'R'])

        return path

    def shrink_mutate(self, old_state: 'ExpTree', path: List[str]) -> List[str]:
        """Shrink a path by deleting a subtree."""
        if not path:
            return path

        valid_indices = [i for i in range(len(path)) if self.arity_dict[path[i]] != 0]
        if not valid_indices:
            return path

        index = np.random.choice(valid_indices)
        if self.arity_dict[path[index]] == 1:  # If it's a unary operator, directly delete it
            return path[:index] + path[index+1:]

        # For binary operators, replace with either left or right child
        current_index = index + 1
        left_subtree_size = self.cal_subtree_size_at_index(path, current_index)
        subtree_size = self.cal_subtree_size_at_index(path, index)
        
        if random.randint(0, 1):  # 50% chance to keep the left subtree
            return path[:index] + path[current_index: current_index + left_subtree_size] + path[index + subtree_size:]
        else:  # 50% chance to keep the right subtree
            right_start = current_index + left_subtree_size
            return path[:index] + path[right_start: index + subtree_size] + path[index + subtree_size:]

    def uniform_mutate(self, old_state: 'ExpTree', path: List[str]) -> List[str]:
        """Mutate a path by replacing a random subtree."""
        if not path:
            return path

        index = random.randint(0, len(path)-1)
        subtree_size = self.cal_subtree_size_at_index(path, index)
        start, end = index, index + subtree_size
        
        relevant_ops = (op for op in path[:start] + path[end:])
        single_arity_count = sum(1 for op in relevant_ops if self.arity_dict[op] == 1) + old_state.single_arity_op_count
        constants_count = sum(1 for op in path[:start] + path[end:] if op in {'C', 'R'}) + old_state.constant_count
        
        # Calculate depth at this index - better estimation based on path structure
        # Count the depth by analyzing operators before the current index
        current_depth = self.cal_depth_at_index(path, index)
        available_depth = max(1, old_state.max_depth - current_depth)
        
        # Calculate available resources by using the difference
        available_arity = old_state.max_single_arity_ops - single_arity_count
        available_constants = old_state.max_constants - constants_count
        
        # Create a new subtree
        subtree = ExpTree(
            max_depth=available_depth,
            max_single_arity_ops=available_arity,
            max_constants=available_constants,
            arity_dict=self.arity_dict,
            ops=self.ops
        )
        if self.arity_dict[path[index]] in self.unique_arities:
            available_ops = [op for op in subtree.available_ops if self.arity_dict[op] not in self.unique_arities]
            if available_ops:
                subtree.available_ops = available_ops
        _, new_path = subtree.random_fill()
        
        return path[:start] + new_path + path[end:]
    
    def insert_mutate(self, old_state: 'ExpTree', path: List[str]) -> List[str]:
        """Mutate a path by inserting a subtree."""
        if not path:
            return path
        
        index = random.randint(0, len(path)-1)  # Insert before the index
        
        # Calculate subtree size at this index
        subtree_size = self.cal_subtree_size_at_index(path, index)
        
        # Calculate available resources
        available_arity = old_state.max_single_arity_ops - old_state.single_arity_op_count
        available_constants = old_state.max_constants - old_state.constant_count
        
        # Calculate accurate depth at this index
        current_depth = self.cal_depth_at_index(path, index)
        available_depth = max(1, old_state.max_depth - current_depth)
        
        insert_ops = [op for op in self.ops if self.arity_dict[op] != 0]  # Operators that can be used as the root node
        if not available_arity:  # If the unary operator limit is reached, do not insert unary operators
            insert_ops = [op for op in insert_ops if self.arity_dict[op] != 1]

        if not insert_ops:
            return path
            
        insert_op = random.choice(insert_ops)

        if self.arity_dict[insert_op] == 1:  # If inserting a unary operator, just insert it
            return path[:index] + [insert_op] + path[index:]
        
        # If inserting a binary operator, need to insert a subtree
        available_depth -= 1  # Binary operator requires one depth level
        if available_depth <= 0:
            return path
            
        subtree = ExpTree(
            max_depth=available_depth,
            max_single_arity_ops=available_arity,
            max_constants=available_constants,
            arity_dict=self.arity_dict,
            ops=self.ops
        )
        _, new_path = subtree.random_fill()
        if random.randint(0, 1):  # 50% chance to insert into the left subtree
            return path[:index] + [insert_op] + new_path + path[index:]
        else:  # 50% chance to insert into the right subtree
            return path[:index] + [insert_op] + path[index:index+subtree_size] + new_path + path[index+subtree_size:]
        
    def crossover(self, old_state: 'ExpTree', path1: List[str], path2: List[str]) -> Tuple[List[str], List[str]]:
        """Perform crossover between two paths."""
        # Return original paths if either is empty
        if not path1 or not path2:
            return path1, path2

        # Find valid crossover points in path1 (nodes with non-zero arity)
        valid_indices_path1 = [i for i, op in enumerate(path1) if self.arity_dict[op] not in self.unique_arities]
        
        # Randomly select crossover points
        crossover_index_path1 = (
            np.random.choice(valid_indices_path1) 
            if valid_indices_path1 
            else np.random.choice(len(path1))
        )
        crossover_index_path2 = np.random.choice(len(path2))

        # Calculate subtree sizes for crossover replacement
        subtree_size_path1 = self.cal_subtree_size_at_index(path1, crossover_index_path1)
        subtree_size_path2 = self.cal_subtree_size_at_index(path2, crossover_index_path2)

        # Construct new paths by swapping subtrees
        new_path1 = (
            path1[:crossover_index_path1] +
            path2[crossover_index_path2:crossover_index_path2+subtree_size_path2] +
            path1[crossover_index_path1+subtree_size_path1:]
        )
        
        new_path2 = (
            path2[:crossover_index_path2] +
            path1[crossover_index_path1:crossover_index_path1+subtree_size_path1] +
            path2[crossover_index_path2+subtree_size_path2:]
        )

        return new_path1, new_path2

    def cal_subtree_size_at_index(self, path: List[str], index: int) -> int:
        """Calculate the size of subtree starting at given index in the path."""
        if index >= len(path):
            return 0
        
        stack = [index]  # Stack to store indices to process
        total_size = 0
        
        while stack:
            current_index = stack.pop()
            if current_index >= len(path):
                continue
                
            op = path[current_index]
            arity = self.arity_dict[op]
            total_size += 1  # Count current node
            
            # Add children indices to stack (in reverse order for correct processing)
            child_index = current_index + 1
            for _ in range(arity):
                if child_index < len(path):
                    # Calculate this child's subtree size to get next sibling's position
                    child_size = 1  # At least the child node itself
                    temp_stack = [child_index]
                    temp_processed = 0
                    
                    while temp_stack:
                        temp_idx = temp_stack.pop()
                        if temp_idx >= len(path):
                            continue
                        temp_op = path[temp_idx]
                        temp_arity = self.arity_dict[temp_op]
                        temp_processed += 1
                        
                        temp_child_idx = temp_idx + 1
                        for _ in range(temp_arity):
                            if temp_child_idx < len(path):
                                temp_stack.append(temp_child_idx)
                                temp_child_idx += 1
                            else:
                                break
                    
                    child_size = temp_processed
                    stack.append(child_index)
                    child_index += child_size
                else:
                    break
        
        return total_size

    def cal_subtree_depth_at_index(self, path: List[str], index: int) -> int:
        """Calculate the depth of subtree starting at given index in the path."""
        if index >= len(path):
            return 0
        
        # Stack stores tuples of (index, current_depth)
        stack = [(index, 1)]
        max_depth = 0
        
        while stack:
            current_index, current_depth = stack.pop()
            if current_index >= len(path):
                continue
                
            op = path[current_index]
            arity = self.arity_dict[op]
            max_depth = max(max_depth, current_depth)
            
            if arity == 0:
                continue
            
            # Add children to stack with increased depth
            child_index = current_index + 1
            for _ in range(arity):
                if child_index < len(path):
                    # Calculate child subtree size to position next sibling
                    child_size = self._get_subtree_size_iterative(path, child_index)
                    stack.append((child_index, current_depth + 1))
                    child_index += child_size
                else:
                    break
        
        return max_depth

    def _get_subtree_size_iterative(self, path: List[str], start_index: int) -> int:
        """Helper function to calculate subtree size iteratively."""
        if start_index >= len(path):
            return 0
        
        stack = [start_index]
        size = 0
        
        while stack:
            current_index = stack.pop()
            if current_index >= len(path):
                continue
                
            op = path[current_index]
            arity = self.arity_dict[op]
            size += 1
            
            # Add children indices to stack
            child_index = current_index + 1
            children_to_add = []
            
            for _ in range(arity):
                if child_index < len(path):
                    children_to_add.append(child_index)
                    # Move to next sibling by skipping current child's subtree
                    child_subtree_size = 1  # Will be calculated properly
                    temp_stack = [child_index]
                    temp_size = 0
                    
                    while temp_stack:
                        temp_idx = temp_stack.pop()
                        if temp_idx >= len(path):
                            continue
                        temp_op = path[temp_idx]
                        temp_arity = self.arity_dict[temp_op]
                        temp_size += 1
                        
                        temp_child_idx = temp_idx + 1
                        for _ in range(temp_arity):
                            if temp_child_idx < len(path):
                                temp_stack.append(temp_child_idx)
                                temp_child_idx += 1
                    
                    child_index += temp_size
                else:
                    break
            
            # Add children in reverse order for correct processing
            stack.extend(reversed(children_to_add))
        
        return size
    
    def cal_depth_at_index(self, path: List[str], target_index: int) -> int:
        """Calculate the depth of the node at target_index in the path."""
        if target_index >= len(path) or target_index < 0:
            return 0
        
        # Use a stack to track depth as we traverse the path
        depth_stack = []
        current_depth = 1  # Root starts at depth 1
        
        for i, op in enumerate(path):
            if i == target_index:
                return current_depth
            
            arity = self.arity_dict[op]
            
            if arity == 0:
                # Leaf node - pop from stack if we've completed a parent's children
                while depth_stack and depth_stack[-1][1] == 0:
                    depth_stack.pop()
                    if depth_stack:
                        depth_stack[-1] = (depth_stack[-1][0], depth_stack[-1][1] - 1)
            else:
                # Internal node - push to stack with remaining children count
                depth_stack.append((current_depth, arity))
                current_depth += 1
                
            # Decrease children count for parent
            if depth_stack and depth_stack[-1][1] > 0:
                depth_stack[-1] = (depth_stack[-1][0], depth_stack[-1][1] - 1)
                
            # Update current depth for next node
            if depth_stack:
                current_depth = depth_stack[-1][0] + 1
            else:
                current_depth = 1
        
        return current_depth
    
    def mutate(self, state: 'ExpTree', path: List[str]) -> List[str]:
        """Mutate a path using a random mutation strategy."""
        mutation = np.random.choice(self.mutations)
        new_path = mutation(state, path)
        if new_path == path:  # If the mutation does not change the path, try again with uniform mutation
            new_path = self.uniform_mutate(state, path)
        return new_path

    def generate(self, state: 'ExpTree') -> List[str]:
        """Generate a random path."""
        state = copy.deepcopy(state)
        path = []
        while not state.is_terminal():
            op = np.random.choice(state.available_ops)
            path.append(op)
            state.add_op(op)
        return path