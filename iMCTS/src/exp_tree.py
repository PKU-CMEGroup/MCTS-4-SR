from typing import List, Dict, Tuple
import random
from abc import abstractmethod
import copy

class ExpTreeBase:
    def __init__(self, max_depth: int, max_single_arity_ops: int, max_constants: int, arity_dict: Dict[str, int], ops: List[str]):
        self.root_op = None
        self.max_depth = max_depth
        self.depth = 0
        self.length = 0
        self.arity_dict = arity_dict
        self.ops = ops
        # Stack entries: (op, arity, children_added, depth)
        self.stack: List[Tuple[str, int, int, int]] = []
        self.max_single_arity_ops = max_single_arity_ops
        self.single_arity_op_count = 0
        self.max_constants = max_constants
        self.constant_count = 0
        self.real_constant_count = 0
        self.available_ops = ops
        self.available_ops_by_depth = {True: [op for op in self.ops if self.arity_dict[op] == 0],
                                        False: self.ops}
        self.op_list = []
        self.update_available_ops()

        self.prior_p = self._calculate_prior_probabilities()

    def _calculate_prior_probabilities(self) -> Dict[str, float]:
        """
        Calculate prior probabilities for each operator based on arity and type.
        Default prior probabilities are uniform for each operator type.

        Returns:
            prior_p: dict
                Dictionary of prior probabilities for each operator
        """
        var_ops = [op for op in self.ops if op.startswith('x')]
        constant_ops = [op for op in self.ops if op.startswith(('C', 'R'))]
        non_leaf_ops = [op for op in self.ops if self.arity_dict[op] != 0]

        if constant_ops:
            constant_prob = 0.5 / len(constant_ops)
            var_prob = 0.5 / len(var_ops)
        else:
            constant_prob = 0
            var_prob = 1 / len(var_ops)

        non_leaf_prob = 1 / len(non_leaf_ops)

        prior_p = {op: var_prob for op in var_ops}
        prior_p.update({op: constant_prob for op in constant_ops})
        prior_p.update({op: non_leaf_prob for op in non_leaf_ops})

        return prior_p

    def is_empty(self) -> bool:
        return self.root_op is None
    
    def is_full(self) -> bool:
        return not self.stack

    def _is_stack_entry_complete(self, stack_entry: Tuple[str, int, int, int]) -> bool:
        """Check if all children have been added to this stack entry."""
        op, arity, children_added, depth = stack_entry
        return children_added == arity

    @abstractmethod
    def is_terminal(self) -> bool:
        pass

    @abstractmethod
    def add_op(self, op: str) -> Tuple[str, int, int, int]:
        pass

    def add_op_common(self, op: str):
        self.op_list.append(op)

        # Update counts for single arity operators and constants
        if self.arity_dict[op] == 1:
            if self.single_arity_op_count == self.max_single_arity_ops:
                raise ValueError('Max single arity operators limit reached')
            self.single_arity_op_count += 1

        if op in ['R', 'C']:
            if self.constant_count == self.max_constants:
                raise ValueError('Max constants limit reached')
            if op == 'R':
                self.real_constant_count += 1
            self.constant_count += 1
    
    def add_op_post_process(self, stack_entry: Tuple[str, int, int, int]):
        self.stack.append(stack_entry)
        self.depth = max(self.depth, len(self.stack))
        self.update_stack()
        self.update_available_ops()
        self.length += 1

    def update_stack(self):
        # Update stack by removing completed entries and updating parent entries
        while self.stack and self._is_stack_entry_complete(self.stack[-1]):
            self.stack.pop()
            # Update the parent entry to increment children_added
            if self.stack:
                op, arity, children_added, depth = self.stack[-1]
                self.stack[-1] = (op, arity, children_added + 1, depth)

    def get_expression(self) -> str:
        if not self.root_op or self.stack:
            raise ValueError('Expression tree is incomplete or empty')

        op_list_copy = self.op_list[:]
        real_constant_count = 0
        complex_constant_count = 0

        def build_expr():
            nonlocal real_constant_count, complex_constant_count
            if not op_list_copy:
                raise ValueError("Invalid op_list: not enough operators.")
            
            op = op_list_copy.pop(0)
            arity = self.arity_dict.get(op, 0)

            if arity == 0:
                if op == 'R':
                    op_str = f'R[{real_constant_count}]'
                    real_constant_count += 1
                    return op_str
                elif op == 'C':
                    op_str = f'C[{complex_constant_count}]'
                    complex_constant_count += 1
                    return op_str
                elif op.startswith('x'):
                    return f'x[{op[1:]}]'
                else:
                    return op

            children = [build_expr() for _ in range(arity)]
            
            if arity == 1:
                return f'{op}({children[0]})'
            
            return f'({children[0]}{op}{children[1]})'

        return build_expr()

    @abstractmethod
    def update_available_ops(self):
        pass

    def random_fill(self) -> List[str]:
        # s = copy.copy(self)  # Use shallow copy
        op_list = []
        while not self.is_terminal():
            # weights = [self.prior_p[op] for op in self.available_ops]
            # op = random.choices(self.available_ops, weights=weights, k=1)[0]
            op = random.choice(self.available_ops)
            op_list.append(op)
            self.add_op(op)
        return self, op_list
    
    def clear(self):
        self.root_op = None
        self.stack = []
        self.op_list = []
        self.depth = 0
        self.length = 0
        self.single_arity_op_count = 0
        self.constant_count = 0
        self.real_constant_count = 0
        self.available_ops = self.ops
        self.update_available_ops()

class ExpTree(ExpTreeBase):
    def is_terminal(self) -> bool:
        return not self.stack and self.root_op is not None

    def add_op(self, op: str) -> Tuple[str, int, int, int]:
        if op not in self.available_ops:
            raise ValueError(f'Invalid op, not in available ops, try to add {op} to {self.available_ops}')
        
        self.add_op_common(op)
        arity = self.arity_dict.get(op, 0)
        current_depth = len(self.stack)

        if self.is_empty():
            self.root_op = op
            stack_entry = (op, arity, 0, 0)
        else:
            stack_entry = (op, arity, 0, current_depth)

        self.add_op_post_process(stack_entry)
        return stack_entry

    def update_available_ops(self):
        stack_len = len(self.stack)
        at_max_depth = stack_len == self.max_depth - 1

        available_ops = self.available_ops_by_depth[at_max_depth]

        # Check the stack for specific conditions, trigonometric functions and log/exp
        for stack_entry in self.stack:
            op, _, _, _ = stack_entry
            if op == 'sin' or op == 'cos':
                available_ops = [op for op in available_ops if op != 'sin' and op != 'cos']
                break
        
        if self.stack:
            op, _, _, _ = self.stack[-1]
            if op == 'log':
                available_ops = [op for op in available_ops if op != 'exp']
            if op == 'exp':
                available_ops = [op for op in available_ops if op != 'log']
        
        # Precompute filter conditions
        filter_single_arity = self.single_arity_op_count == self.max_single_arity_ops
        filter_constants = self.constant_count == self.max_constants
        
        # Single pass filtering with combined conditions
        self.available_ops = [
            op for op in available_ops
            if (not filter_single_arity or self.arity_dict[op] != 1)  # Arity filter
            and (not filter_constants or (not op.startswith(('C', 'R')))) # Constant filter
        ]