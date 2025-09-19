from sortedcontainers import SortedList
from typing import Any, Tuple, Iterable
from abc import ABC, abstractmethod
import math
import random

class Queue_Base(ABC):
    """Abstract base for prioritized experience / expression queues.

    Stored elements are tuples of (state, reward). The internal SortedList is
    maintained in descending order of reward (achieved by key = -reward).
    """
    def __init__(self, max_size: int):
        self.max_size = max_size
        # Main container: sorted descending by reward (via negative key)
        self.list: SortedList[Tuple[Any, float]] = SortedList(key=lambda x: -x[1])
        # Maintain a secondary sorted structure of raw reward values (ascending) for O(log n) near-duplicate checks
        self._reward_values: SortedList[float] = SortedList()
        self.min_reward: float = float('-inf')  # Cached minimum reward in queue (last element)

    @abstractmethod
    def append(self, state: Any, reward: float) -> bool:
        pass

    def __len__(self) -> int:  # Convenience
        return len(self.list)

    def __iter__(self) -> Iterable[Tuple[Any, float]]:
        return iter(self.list)

    # Best reward value in the queue
    def best_reward(self) -> float:
        if not self.list:
            return 0.0
        return self.list[0][1]
    
    def best(self) -> Any:
        if not self.list:
            return None, None
        return self.list[0]

    def random_sample(self) -> Any:
        if not self.list:
            return None, None
        return random.choice(self.list)

    def is_empty(self) -> bool:
        return not self.list

class Exp_Queue(Queue_Base):
    """Experience / expression priority queue with fast approximate duplicate suppression.

    Performance improvements over previous version:
    - Near-duplicate reward detection reduced from O(n) scan to O(log n) by using a
      secondary SortedList of reward values and only inspecting neighbors.
    - Fewer attribute lookups inside hot append path via local bindings.
    - Early exits for invalid rewards (inf / nan).
    """

    def append(self, state: Any, reward: float, threshold: float = 1e-5) -> bool:
        """Attempt to insert (state, reward).

        Duplicate / near-duplicate rewards (|Δ| < threshold) are rejected to avoid
        bloating the queue with numerically indistinguishable entries.
        Returns True if inserted, False otherwise.
        """
        # Reject invalid numeric cases early (math.isinf faster; also guard nan)
        if math.isinf(reward) or math.isnan(reward):
            return False

        reward_values = self._reward_values  # local binding
        # Binary search position (ascending order). Only immediate neighbors can be within threshold.
        pos = reward_values.bisect_left(reward)
        if pos > 0 and abs(reward_values[pos - 1] - reward) < threshold:
            return False
        if pos < len(reward_values) and abs(reward_values[pos] - reward) < threshold:
            return False

        lst = self.list
        if len(lst) < self.max_size:
            lst.add((state, reward))
            reward_values.add(reward)
        else:
            # Fast check: if not better than current minimum, discard.
            if reward <= self.min_reward:
                return False
            # Remove worst (last element due to descending order via key)
            removed_state, removed_reward = lst.pop(-1)
            # Remove from structures
            reward_values.remove(removed_reward)
            # Insert new element
            lst.add((state, reward))
            reward_values.add(reward)

        # Update cached minimum reward (worst at index -1)
        self.min_reward = lst[-1][1] if lst else float('-inf')
        return True
