from typing import Any, NamedTuple, Generic, List, Optional, Sequence, Tuple, TypeVar, Mapping, Text
import numpy as np
import snappy

CompressedArray = Tuple[bytes, Tuple, np.dtype]

# Generic replay structure: Any flat named tuple.
ReplayStructure = TypeVar('ReplayStructure', bound=Tuple[Any, ...])


class Transition(NamedTuple):
    state: Optional[np.ndarray]  # stacked history of observation and actions
    action: Optional[int]
    pi_prob: Optional[np.ndarray]
    value: Optional[float]  # n-step value for atari, (or game final score for board games)
    reward: Optional[float]

TransitionStructure = Transition(state=None, action=None, pi_prob=None, value=None, reward=None)



class PrioritizedReplay(Generic[ReplayStructure]):
    """Prioritized replay, with circular buffer storage for flat named tuples."""

    def __init__(
        self,
        capacity: int,
        priority_exponent: float,
        importance_sampling_exponent: float,
        random_state: np.random.RandomState,
    ):
        """
        Args:
            capacity: maximum number of samples in the replay.
            priority_exponent: priority exponent, if 0 becomes uniform random replay.
            importance_sampling_exponent: importance sampling exponent.
            random_state: np.random.RandomState for sampling transitions.
        """
        if capacity <= 0:
            raise ValueError(f'Expect capacity to be a positive integer, got {capacity}')
        self.structure = TransitionStructure
        self._capacity = capacity
        self._storage = [None] * int(capacity)
        self._random_state = random_state
        self._num_added = 0

        # Prioritized related.
        self._priority_exponent = priority_exponent
        self._importance_sampling_exponent = importance_sampling_exponent
        self._priorities = np.zeros((int(capacity),), dtype=np.float32)

    def add(self, item: Transition, priority: float) -> None:
        """Adds single item to replay."""
        if not np.isfinite(priority) or priority < 0.0:
            raise ValueError('Priority must be finite and positive.')

        index = self._num_added % self._capacity
        self._storage[int(index)] = encoder(item)
        self._priorities[int(index)] = priority
        self._num_added += 1

    def get(self, indices: Sequence[int]) -> List[Transition]:
        """Retrieves items by indices."""
        return [decoder(self._storage[i]) for i in indices]

    def sample(self, batch_size: int) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        """Samples batch of items from replay uniformly, with replacement."""
        if self.size < batch_size:
            raise RuntimeError(f'Replay only have {self.size} samples, got sample batch size {batch_size}')

        if self._priority_exponent == 0:
            indices = self._random_state.uniform(0, self.size, size=batch_size).astype(np.int64)
            weights = np.ones_like(indices, dtype=np.float32)
        else:
            priorities = self._priorities[: self.size] ** self._priority_exponent
            probs = priorities / np.sum(priorities)

            indices = np.random.choice(np.arange(probs.shape[0]), size=batch_size, replace=True, p=probs)

            # Importance weights.
            weights = ((1.0 / self.size) / probs[indices]) ** self._importance_sampling_exponent
            weights /= np.max(weights)  # Normalize.

        samples = self.get(indices)
        transposed = zip(*samples)
        stacked = [np.stack(xs, axis=0) for xs in transposed]  # Stack on batch dimension (0)
        batched_sample = type(self.structure)(*stacked)
        return batched_sample, indices, weights

    def update_priorities(self, indices: Sequence[int], priorities: Sequence[float]) -> None:
        """Updates indices with given priorities."""
        priorities = np.asarray(priorities)
        if not np.isfinite(priorities).all() or (priorities < 0.0).any():
            raise ValueError('Priorities must be finite and positive.')

        for i, p in zip(indices, priorities):
            self._priorities[i] = p

    @property
    def num_added(self) -> int:
        """Number of items added into replay."""
        return self._num_added

    @property
    def size(self) -> int:
        """Number of items currently contained in replay."""
        return min(self._num_added, self._capacity)

    @property
    def capacity(self) -> int:
        """Total capacity of replay (max number of items stored at any one time)."""
        return self._capacity

    def reset(self) -> None:
        """Reset the state of replay, should be called at the begining of every episode"""
        self._num_added = 0

    def get_state(self) -> Mapping[Text, Any]:
        """Retrieves replay state as a dictionary (e.g. for serialization)."""
        return {'num_added': self._num_added, 'storage': self._storage, 'priorities': self._priorities}

    def set_state(self, state: Mapping[Text, Any]) -> None:
        """Sets replay state from a (potentially de-serialized) dictionary."""
        self._num_added = state['num_added']
        self._storage = state['storage']
        self._priorities = state['priorities']


def compress_array(array: np.ndarray) -> CompressedArray:
    """Compresses a numpy array with snappy."""
    return snappy.compress(array), array.shape, array.dtype


def uncompress_array(compressed: CompressedArray) -> np.ndarray:
    """Uncompresses a numpy array with snappy given its shape and dtype."""
    compressed_array, shape, dtype = compressed
    byte_string = snappy.uncompress(compressed_array)
    return np.frombuffer(byte_string, dtype=dtype).reshape(shape)


def encoder(transition: Transition) -> Transition:
    return transition._replace(
        state=compress_array(transition.state),
    )


def decoder(transition: Transition) -> Transition:
    return transition._replace(
        state=uncompress_array(transition.state),
    )