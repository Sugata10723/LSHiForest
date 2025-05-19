import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator
from collections import defaultdict
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.random_projection import SparseRandomProjection

class LSHHash:
    def __init__(self, input_dim, hash_dim, seed=None, center=None):
        self.input_dim = input_dim
        self.hash_dim = hash_dim
        if seed is not None:
            np.random.seed(seed)
        self.projections = SparseRandomProjection(
            n_components=hash_dim, dense_output=True, random_state=seed
        ).fit(np.eye(input_dim)).components_
        self.center = center if center is not None else np.zeros(input_dim)

    def hash(self, vec):
        vec_centered = vec - self.center
        return tuple((self.projections @ vec_centered) > 0)

class LSHiTreeNode:
    def __init__(self, input_dim, hash_dim, random_state=None, depth=0, max_depth=10, min_samples=5):
        self.input_dim = input_dim
        self.hash_dim = hash_dim
        self.random_state = random_state
        self.depth = depth
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.children = {}
        self.is_leaf = False
        self.n_samples = 0

    def fit(self, X):
        self.n_samples = X.shape[0] # サブサンプルサイズの確認
        seed = (self.random_state or 0) + self.depth + np.random.randint(10000) # seed値の更新
        center = X[np.random.randint(len(X))]  # 1点だけランダムに選ぶ
        hasher = LSHHash(
            input_dim=self.input_dim,
            hash_dim=self.hash_dim,
            seed=seed,
            center=center 
        )

        if self.depth >= self.max_depth or self.n_samples <= self.min_samples: # 停止条件
            self.is_leaf = True # 葉ノードとする
            return # 終了

        buckets = defaultdict(list) # バケツの作成、valueはリスト
        for x in X: # 全てのデータに対して
            h = hasher.hash(x) # hash化
            buckets[h].append(x) # key:h, value:xをbucketsに追加
        

        for hval, group in buckets.items(): # 全てのkey:hに対して
            child = LSHiTreeNode(
                input_dim=self.input_dim,
                hash_dim=self.hash_dim,
                random_state=self.random_state,
                depth=self.depth + 1,
                max_depth=self.max_depth,
                min_samples=self.min_samples
            ) # 子ノードのインスタンス化
            child.fit(np.array(group)) # childインスタンスの変数を分割により更新
            self.children[hval] = child # hvalごとに子ノードを保存

        self.hasher = hasher  # 自分のノードが使ったハッシャーも保持

    def path_length(self, x):
        if self.is_leaf or not self.children: # 葉ノードの場合
            return self.depth + c_factor(self.n_samples) # 深さを計算
        hval = self.hasher.hash(x) # 各ノードのhaserを用いてhash化
        child = self.children.get(hval) # hvalのchildインスタンスを取得
        if child: # childインスタンスが存在=葉ノードじゃない
            return child.path_length(x) # 再帰的にchildに対して葉ノードを計算
        else: # 一致する子ノードがない場合=hvalが葉ノード
            return self.depth + c_factor(self.n_samples) 
    
def c_factor(n):
    if n <= 1:
        return 0
    return 2 * np.log(n - 1) + 0.5772156649 - (2 * (n - 1) / n)


class CLSHiForest(BaseEstimator):
    def __init__(self, n_trees=100, max_depth=10, hash_dim=3, min_samples=5, random_state=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.hash_dim = hash_dim
        self.min_samples = min_samples
        self.random_state = random_state
        self.trees = []
        self.encoder = None

    def fit(self, X_raw):
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        X = self.encoder.fit_transform(X_raw)
        print(f'One-hotEncode後の特徴量数:{X.shape}')
        input_dim = X.shape[1]

        self.trees = []

        for i in range(self.n_trees):
            idx = np.random.choice(len(X), size=min(256, len(X)), replace=False) # サンプリング用のindex
            X_sample = X[idx]
            tree = LSHiTreeNode(
                input_dim=input_dim,
                hash_dim=self.hash_dim,
                random_state=self.random_state,
                depth=0,
                max_depth=self.max_depth,
                min_samples=self.min_samples
            )
            tree.fit(X_sample)
            self.trees.append(tree)

        return self

    def decision_function(self, X_raw):
        X = self.encoder.transform(X_raw)
        scores = np.zeros(X.shape[0])
        for i, x in enumerate(X): # 要素とindexをとり出せる（各データに対して）
            lengths = [tree.path_length(x) for tree in self.trees]
            scores[i] = np.mean(lengths)
        return scores  # Lower path length = more anomalous

    def predict(self, scores, threshold=None):
        if threshold is None:
            threshold = np.percentile(scores, 25)  # lower 25% treated as anomaly
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
        """
        ROC曲線をプロットし、AUCスコアを表示する
        """
        fpr, tpr, _ = roc_curve(y_true, -scores)  # 負スコアに変換（異常ほど大きく）
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
