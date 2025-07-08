from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Optional, List, Tuple, Dict, Any, Set
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score, roc_curve

# Define constants for magic numbers
DEFAULT_MAX_DEPTH = 10
DEFAULT_MIN_SAMPLES = 5
DEFAULT_N_TREES = 100
ANOMALY_THRESHOLD_PERCENTILE = 25
NODE_SIZE_MIN = 300
NODE_SIZE_MAX = 2000
FIG_SIZE_DISTRIBUTION = (8, 5)
FIG_SIZE_ROC = (6, 6)
FIG_SIZE_TREE = (14, 10)

class BaseLSHiTreeNode(ABC):
    """Abstract base class for a node in the LSH-based isolation tree."""
    def __init__(self, depth: int = 0, max_depth: int = DEFAULT_MAX_DEPTH, min_samples: int = DEFAULT_MIN_SAMPLES):
        self.depth = depth
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.children: Dict[Any, "BaseLSHiTreeNode"] = {}
        self.is_leaf = False
        self.n_samples = 0
        self.hasher: Optional[Any] = None

    @abstractmethod
    def fit(self, X: Any, **kwargs):
        """Fits the node to the input data."""
        pass

    @abstractmethod
    def path_length(self, x: Any) -> float:
        """Calculates the path length of the input vector."""
        pass

class BaseLSHiForest(BaseEstimator, ABC):
    """Abstract base class for LSH-based Isolation Forest."""
    def __init__(self, n_trees: int = DEFAULT_N_TREES, max_depth: int = DEFAULT_MAX_DEPTH, min_samples: int = DEFAULT_MIN_SAMPLES, random_state: Optional[int] = None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.random_state = random_state
        self.trees: List[BaseLSHiTreeNode] = []

    @staticmethod
    def _c_factor(n: int) -> float:
        """Correction factor for path length calculation, same as in standard Isolation Forest."""
        if n <= 1:
            return 0
        euler_mascheroni = 0.5772156649
        return 2 * (np.log(n - 1) + euler_mascheroni) - (2 * (n - 1) / n)

    @abstractmethod
    def fit(self, X_raw: Any, y: Any = None):
        """Fits the forest to the input data."""
        pass

    def decision_function(self, X_raw: Any) -> np.ndarray:
        """Calculates the anomaly score for each sample."""
        if not self.trees:
            raise ValueError("The model has not been fitted yet. Call fit() first.")
        
        X_processed = self._preprocess(X_raw)
        scores = np.zeros(len(X_processed))
        for i, x in enumerate(X_processed):
            lengths = [tree.path_length(x) for tree in self.trees]
            scores[i] = np.mean(lengths) if lengths else 0
        return scores

    def predict(self, X_raw: Any, threshold: Optional[float] = None) -> np.ndarray:
        """Predicts whether each sample is an anomaly or not."""
        scores = self.decision_function(X_raw)
        if threshold is None:
            threshold = np.percentile(scores, ANOMALY_THRESHOLD_PERCENTILE)
        return (scores < threshold).astype(int)

    @abstractmethod
    def _preprocess(self, X_raw: Any) -> Any:
        """Preprocesses the raw input data."""
        pass



    def visualize_tree(self, tree_index: int = 0):
        """Visualizes the structure of a single tree in the forest."""
        if not self.trees:
            raise ValueError("No trees found. Please fit the model first.")
        if not 0 <= tree_index < len(self.trees):
            raise ValueError(f"tree_index {tree_index} is out of range (total {len(self.trees)} trees).")

        tree = self.trees[tree_index]
        G = nx.DiGraph()
        node_samples: Dict[int, int] = {}

        def add_nodes_edges(node: BaseLSHiTreeNode, parent_id: Optional[int] = None, node_id: int = 0, counter: List[int] = [0]):
            label = f"Depth {node.depth}\nSamples {node.n_samples}"
            G.add_node(node_id, label=label)
            node_samples[node_id] = node.n_samples

            if parent_id is not None:
                G.add_edge(parent_id, node_id)

            this_id = node_id
            for child in node.children.values():
                counter[0] += 1
                add_nodes_edges(child, parent_id=this_id, node_id=counter[0])

        add_nodes_edges(tree)
        
        if not G.nodes:
            print("Graph is empty, cannot visualize.")
            return

        pos = nx.nx_pydot.graphviz_layout(G, prog="dot")
        labels = nx.get_node_attributes(G, 'label')
        
        sizes = np.array(list(node_samples.values()))
        if sizes.size > 0 and sizes.max() > 0:
            sizes_scaled = NODE_SIZE_MIN + (NODE_SIZE_MAX - NODE_SIZE_MIN) * (sizes / sizes.max())
            colors_scaled = sizes / sizes.max()
        else:
            sizes_scaled = NODE_SIZE_MIN
            colors_scaled = 0.5

        plt.figure(figsize=FIG_SIZE_TREE)
        nx.draw_networkx_nodes(G, pos, node_size=sizes_scaled, node_color=colors_scaled, cmap='YlOrRd')
        nx.draw_networkx_edges(G, pos, arrows=True)
        nx.draw_networkx_labels(G, pos, labels, font_size=10)

        plt.title(f"Visualization of Tree {tree_index} (node size & color by n_samples)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
