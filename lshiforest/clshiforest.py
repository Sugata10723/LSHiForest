from collections import defaultdict
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from .base import BaseLSHiForest, BaseLSHiTreeNode
from .hashers import LSHHash

DEFAULT_HASH_DIM = 3

class CLSHiTreeNode(BaseLSHiTreeNode):
    """A node in the LSH-based isolation tree for categorical data."""
    def __init__(self, input_dim: int, hash_dim: int, random_state: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.hash_dim = hash_dim
        self.random_state = random_state

    def fit(self, X: np.ndarray):
        """Fits the node to the input data."""
        self.n_samples = X.shape[0]
        seed = (self.random_state or 0) + self.depth + np.random.randint(10000)
        center = X[np.random.randint(len(X))]
        self.hasher = LSHHash(
            input_dim=self.input_dim,
            hash_dim=self.hash_dim,
            seed=seed,
            center=center
        )

        if self.depth >= self.max_depth or self.n_samples <= self.min_samples:
            self.is_leaf = True
            return

        buckets: Dict[Tuple[bool, ...], List[np.ndarray]] = defaultdict(list)
        for x in X:
            h = self.hasher.hash(x)
            buckets[h].append(x)

        for hval, group in buckets.items():
            child = CLSHiTreeNode(
                input_dim=self.input_dim,
                hash_dim=self.hash_dim,
                random_state=self.random_state,
                depth=self.depth + 1,
                max_depth=self.max_depth,
                min_samples=self.min_samples
            )
            child.fit(np.array(group))
            self.children[hval] = child

    def path_length(self, x: np.ndarray) -> float:
        """Calculates the path length of the input vector."""
        if self.is_leaf or not self.children or not self.hasher:
            return self.depth + BaseLSHiForest._c_factor(self.n_samples)

        hval = self.hasher.hash(x)
        child = self.children.get(hval)
        if child:
            return child.path_length(x)
        else:
            return self.depth + BaseLSHiForest._c_factor(self.n_samples)

class CLSHiForest(BaseLSHiForest):
    """Categorical LSH-based Isolation Forest."""
    def __init__(self, hash_dim: int = DEFAULT_HASH_DIM, subsample_size: int = 256, **kwargs):
        super().__init__(**kwargs)
        self.hash_dim = hash_dim
        self.subsample_size = subsample_size
        self.encoder: Optional[OneHotEncoder] = None

    def _preprocess(self, X_raw: np.ndarray) -> np.ndarray:
        """Preprocesses the raw input data using one-hot encoding."""
        if self.encoder is None:
            raise ValueError("The model has not been fitted yet.")
        return self.encoder.transform(X_raw)

    def fit(self, X_raw: np.ndarray, y: Any = None):
        """Fits the forest to the input data."""
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        X = self.encoder.fit_transform(X_raw)
        print(f'Number of features after one-hot encoding: {X.shape[1]}')
        input_dim = X.shape[1]

        self.trees = []
        print("Building CLSHiForest...")
        for i in range(self.n_trees):
            print(f"  Tree {i+1}/{self.n_trees}", end='\r')
            sample_size = min(self.subsample_size, len(X))
            idx = np.random.choice(len(X), size=sample_size, replace=False)
            X_sample = X[idx]
            tree = CLSHiTreeNode(
                input_dim=input_dim,
                hash_dim=self.hash_dim,
                random_state=(self.random_state or 0) + i,
                depth=0,
                max_depth=self.max_depth,
                min_samples=self.min_samples
            )
            tree.fit(X_sample)
            self.trees.append(tree)
        return self
