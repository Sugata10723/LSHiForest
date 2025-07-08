from collections import defaultdict
from typing import Optional, List, Tuple, Dict, Any, Set
import numpy as np
import pandas as pd
from .base import BaseLSHiForest, BaseLSHiTreeNode
from .hashers import MinHash

DEFAULT_NUM_HASHES = 3

class JLSHiTreeNode(BaseLSHiTreeNode):
    """A node in the Jaccard LSH-based isolation tree."""
    def __init__(self, num_hashes: int, random_state: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.num_hashes = num_hashes
        self.random_state = random_state

    def fit(self, X: List[Set[Any]], universe: Set[Any]):
        """Builds the isolation tree node by recursively partitioning the data."""
        self.n_samples = len(X)
        seed = (self.random_state or 0) + self.depth + np.random.randint(10000)
        self.hasher = MinHash(num_hashes=self.num_hashes, seed=seed)
        self.hasher.fit(universe)

        if self.depth >= self.max_depth or self.n_samples <= self.min_samples:
            self.is_leaf = True
            return

        buckets: Dict[Tuple[int, ...], List[Set[Any]]] = defaultdict(list)
        for x in X:
            if self.hasher:
                h = self.hasher.transform(x)
                buckets[h].append(x)

        for hval, group in buckets.items():
            child = JLSHiTreeNode(
                num_hashes=self.num_hashes,
                random_state=self.random_state,
                depth=self.depth + 1,
                max_depth=self.max_depth,
                min_samples=self.min_samples
            )
            child.fit(group, universe)
            self.children[hval] = child

    def path_length(self, x: Set[Any]) -> float:
        """Calculates the path length for a given sample."""
        if self.is_leaf or not self.children or not self.hasher:
            return self.depth + BaseLSHiForest._c_factor(self.n_samples)

        hval = self.hasher.transform(x)
        child = self.children.get(hval)
        if child:
            return child.path_length(x)
        else:
            return self.depth + BaseLSHiForest._c_factor(self.n_samples)

class JLSHiForest(BaseLSHiForest):
    """Jaccard LSH-based Isolation Forest for anomaly detection on categorical data."""
    def __init__(self, num_hashes: int = DEFAULT_NUM_HASHES, subsample_size: int = 256, **kwargs):
        super().__init__(**kwargs)
        self.num_hashes = num_hashes
        self.subsample_size = subsample_size
        self.universe: Set[Any] = set()

    def _row_to_set(self, row: pd.Series) -> Set[str]:
        """Converts a DataFrame row to a set of "column=value" strings."""
        return {f"{col}={row[col]}" for col in row.index}

    def _preprocess(self, X_raw: pd.DataFrame) -> List[Set[str]]:
        """Transforms the raw DataFrame into a list of sets."""
        return X_raw.apply(self._row_to_set, axis=1).tolist()

    def fit(self, X_raw: pd.DataFrame, y: Any = None):
        """Fits the forest to the input data."""
        X_processed = self._preprocess(X_raw)
        
        self.universe = set()
        for items in X_processed:
            self.universe.update(items)
        
        if not self.universe:
            raise ValueError("Could not create a feature universe from the data. Is the input data empty?")
        
        print(f'Universe size: {len(self.universe)}')

        self.trees = []
        print("Building JLSHiForest...")
        for i in range(self.n_trees):
            print(f"  Tree {i+1}/{self.n_trees}", end='\r')
            sample_size = min(self.subsample_size, len(X_processed))
            idx = np.random.choice(len(X_processed), size=sample_size, replace=False)
            X_sample = [X_processed[j] for j in idx]
            
            tree = JLSHiTreeNode(
                num_hashes=self.num_hashes,
                random_state=(self.random_state or 0) + i,
                depth=0,
                max_depth=self.max_depth,
                min_samples=self.min_samples
            )
            tree.fit(X_sample, self.universe)
            self.trees.append(tree)
        return self
