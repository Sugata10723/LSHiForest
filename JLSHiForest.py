import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score, roc_curve
from collections import defaultdict

class MinHash:
    '''
    ハッシュ関数として線形合同法(Linear Congruential Hash)を使用 
    '''
    def __init__(self, num_hashes=3, seed=None):
        self.num_hashes = num_hashes
        rng = np.random.RandomState(seed)  
        self.params = [(rng.randint(1, 1<<30), rng.randint(0, 1<<30)) for _ in range(num_hashes)]  # a, bを作る
        self.prime = 4294967311  # 32bitよりちょっと大きい素数（適当でOK）

    def fit(self, universe):
        self.index = {item: idx for idx, item in enumerate(universe)}

    def transform(self, items):
        if not items:
            return tuple([np.inf] * self.num_hashes)
        
        indices = [self.index[item] for item in items if item in self.index]
        signature = []
        for a, b in self.params:
            min_val = min(((a * i + b) % self.prime) for i in indices)
            signature.append(min_val)
        return tuple(signature)

class MinHashTreeNode:
    def __init__(self, num_hashes, random_state=None, depth=0, max_depth=10, min_samples=5):
        self.num_hashes = num_hashes
        self.random_state = random_state
        self.depth = depth
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.children = {}
        self.is_leaf = False
        self.n_samples = 0

    def build(self, X, universe):
        self.n_samples = len(X)
        seed = (self.random_state or 0) + self.depth + np.random.randint(10000)
        self.hasher = MinHash(num_hashes=self.num_hashes, seed=seed)
        self.hasher.fit(universe)

        if self.depth >= self.max_depth or self.n_samples <= self.min_samples:
            self.is_leaf = True
            return

        buckets = defaultdict(list)
        for x in X:
            h = self.hasher.transform(x)
            buckets[h].append(x)

        for hval, group in buckets.items():
            child = MinHashTreeNode(
                num_hashes=self.num_hashes,
                random_state=self.random_state,
                depth=self.depth + 1,
                max_depth=self.max_depth,
                min_samples=self.min_samples
            )
            child.build(group, universe)
            self.children[hval] = child

    def path_length(self, x):
        if self.is_leaf or not self.children:
            return self.depth + c_factor(self.n_samples)
        hval = self.hasher.transform(x)
        child = self.children.get(hval)
        if child:
            return child.path_length(x)
        else:
            return self.depth + c_factor(self.n_samples)

def c_factor(n):
    if n <= 1:
        return 0
    return 2 * np.log(n - 1) + 0.5772156649 - (2 * (n - 1) / n)

class JLSHiForest():
    '''
    '''
    def __init__(self, n_trees=100, max_depth=10, num_hashes=3, min_samples=5, random_state=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.num_hashes = num_hashes
        self.min_samples = min_samples
        self.random_state = random_state
        self.trees = []
        self.universe = set() # 全空間の初期化

    def row_to_set(self, row): # よりメモリ効率いいデータ構造にできないか？
        return set(f"{col}={row[col]}" for col in row.index)

    def preprocess(self, X_raw):
        return X_raw.apply(self.row_to_set, axis=1).tolist()

    def fit(self, X_raw): # fitって何？
        # 下準備
        X_processed = self.preprocess(X_raw)
        self.universe = set()
        for items in X_processed: # 一つのデータ行に対して
            self.universe.update(items) # 要素の全空間を生成
        print(f'universe size : {len(self.universe)}') ##### debug用コード

        self.trees = []
        for i in range(self.n_trees):
            idx = np.random.choice(len(X_processed), size=min(256, len(X_processed)), replace=False) # randomサンプリング, データ構造はlist？
            X_sample = [X_processed[j] for j in idx]
            tree = MinHashTreeNode(
                num_hashes=self.num_hashes,
                random_state=self.random_state,
                depth=0,
                max_depth=self.max_depth,
                min_samples=self.min_samples # 最小サンプル消す
            )
            tree.build(X_sample, self.universe) # fitって何？→buildに修正
            self.trees.append(tree)
        return self

    def decision_function(self, X_raw):
        X_processed = self.preprocess(X_raw)
        scores = np.zeros(len(X_processed))
        for i, x in enumerate(X_processed):
            lengths = [tree.path_length(x) for tree in self.trees]
            scores[i] = np.mean(lengths)
        return scores

    def predict(self, scores, threshold=None): # 意味をなしてない
        if threshold is None:
            threshold = np.percentile(scores, 25)
        return (scores < threshold).astype(int) 

    def plot_score_distribution(self, scores, y_true=None, bins=50):
        plt.figure(figsize=(8, 5))
        if y_true is not None:
            normal_scores = scores[y_true == 0]
            anomaly_scores = scores[y_true == 1]
            plt.hist(normal_scores, bins=bins, alpha=0.6, label='Normal', color='blue')
            plt.hist(anomaly_scores, bins=bins, alpha=0.6, label='Anomaly', color='red')
            plt.legend()
        else:
            plt.hist(scores, bins=bins, color='gray', alpha=0.7)

        plt.xlabel("Anomaly Score")
        plt.ylabel("Frequency")
        plt.title("Distribution of Anomaly Scores")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_roc_curve(self, scores, y_true):
        fpr, tpr, _ = roc_curve(y_true, -scores)
        auc_score = roc_auc_score(y_true, -scores)

        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return auc_score
    
    def visualize_tree(self, tree_index=0):
        """
        指定したインデックスのツリーを可視化する（ノードのサイズ・色をn_samplesに応じて変化させる）
        """

        if not self.trees:
            raise ValueError("No trees found. Please fit the model first.")

        if tree_index >= len(self.trees):
            raise ValueError(f"tree_index {tree_index} is out of range (total {len(self.trees)} trees)")

        tree = self.trees[tree_index]
        G = nx.DiGraph()

        node_samples = {}  # node_id -> n_samples
        def add_nodes_edges(node, parent_id=None, node_id=0, counter=[0]):
            label = f"Depth {node.depth}\nSamples {node.n_samples}"
            G.add_node(node_id, label=label)
            node_samples[node_id] = node.n_samples

            if parent_id is not None:
                G.add_edge(parent_id, node_id)

            this_id = node_id
            for child_hval, child in node.children.items():
                counter[0] += 1
                add_nodes_edges(child, parent_id=this_id, node_id=counter[0])

        add_nodes_edges(tree)

        pos = nx.nx_pydot.graphviz_layout(G, prog="dot")
        labels = nx.get_node_attributes(G, 'label')

        # ノードのサイズ・色をデータ数ベースで決める
        sizes = np.array(list(node_samples.values()))
        sizes_scaled = 300 + 1700 * (sizes / sizes.max())  # ノードサイズ (300〜2000くらいにスケーリング)
        colors_scaled = sizes / sizes.max()  # 色（0〜1正規化）

        plt.figure(figsize=(14, 10))
        nodes = nx.draw_networkx_nodes(G, pos, node_size=sizes_scaled, node_color=colors_scaled, cmap='YlOrRd')
        nx.draw_networkx_edges(G, pos, arrows=True)
        nx.draw_networkx_labels(G, pos, labels, font_size=10)

        plt.title(f"Visualization of Tree {tree_index} (node size & color by n_samples)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

