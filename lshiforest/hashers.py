from typing import Optional, Tuple, Any, Set, Dict
import numpy as np
from sklearn.random_projection import SparseRandomProjection

# A large prime number for the hash function
HASH_PRIME = 4294967311
DEFAULT_NUM_HASHES = 3

class LSHHash:
    """A class for Locality Sensitive Hashing (LSH)."""
    def __init__(self, input_dim: int, hash_dim: int, seed: Optional[int] = None, center: Optional[np.ndarray] = None):
        """
        Args:
            input_dim (int): The dimension of the input vectors.
            hash_dim (int): The dimension of the hash.
            seed (Optional[int]): The random seed.
            center (Optional[np.ndarray]): The center of the projections.
        """
        self.input_dim = input_dim
        self.hash_dim = hash_dim
        if seed is not None:
            np.random.seed(seed)
        self.projections = SparseRandomProjection(
            n_components=hash_dim, dense_output=True, random_state=seed
        ).fit(np.eye(input_dim)).components_
        self.center = center if center is not None else np.zeros(input_dim)

    def hash(self, vec: np.ndarray) -> Tuple[bool, ...]:
        """Hashes the input vector.

        Args:
            vec (np.ndarray): The input vector.

        Returns:
            Tuple[bool, ...]: The hash of the input vector.
        """
        vec_centered = vec - self.center
        return tuple((self.projections @ vec_centered) > 0)

class MinHash:
    """
    A MinHash implementation using the Linear Congruential Hash method.
    This is used to create signatures for sets of features.
    """
    def __init__(self, num_hashes: int = DEFAULT_NUM_HASHES, seed: Optional[int] = None):
        """
        Args:
            num_hashes (int): The number of hash functions to use.
            seed (Optional[int]): The random seed for generating hash function parameters.
        """
        if num_hashes <= 0:
            raise ValueError("Number of hashes must be greater than 0.")
        self.num_hashes = num_hashes
        rng = np.random.RandomState(seed)
        # Generate parameters (a, b) for the linear congruential hash functions
        self.params = [(rng.randint(1, 1 << 30), rng.randint(0, 1 << 30)) for _ in range(num_hashes)]
        self.prime = HASH_PRIME
        self.index: Dict[Any, int] = {}

    def fit(self, universe: Set[Any]):
        """
        Creates an index mapping from each item in the universe to an integer.

        Args:
            universe (Set[Any]): The set of all possible items.
        """
        if not universe:
            raise ValueError("Universe cannot be empty.")
        self.index = {item: idx for idx, item in enumerate(universe)}

    def transform(self, items: Set[Any]) -> Tuple[int, ...]:
        """
        Transforms a set of items into a MinHash signature.

        Args:
            items (Set[Any]): The set of items to hash.

        Returns:
            Tuple[int, ...]: The MinHash signature.
        """
        if not items:
            return tuple([np.inf] * self.num_hashes)

        indices = [self.index[item] for item in items if item in self.index]
        if not indices:
            return tuple([np.inf] * self.num_hashes)
            
        signature = []
        for a, b in self.params:
            min_val = min(((a * i + b) % self.prime) for i in indices)
            signature.append(min_val)
        return tuple(signature)
