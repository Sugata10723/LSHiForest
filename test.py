import argparse
from lshiforest import DataLoader, CLSHiForest, JLSHiForest, MinLSHiForest, AttackAnalyzer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt
from PIL import Image

def plot_results(scores, y_true, model_name, dataset_name):
    roc_fig_path = f"results/{dataset_name}_{model_name}_roc_curve.png"
    dist_fig_path = f"results/{dataset_name}_{model_name}_score_distribution.png"

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_true, -scores)
    auc_score = roc_auc_score(y_true, -scores)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name} on {dataset_name}")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(roc_fig_path)
    plt.close()

    # Plot Score Distribution
    plt.figure(figsize=(8, 5))
    normal_scores = scores[y_true == 0]
    anomaly_scores = scores[y_true == 1]
    plt.hist(normal_scores, bins=50, alpha=0.6, label='Normal', color='blue')
    plt.hist(anomaly_scores, bins=50, alpha=0.6, label='Anomaly', color='red')
    plt.legend()
    plt.xlabel("Anomaly Score")
    plt.ylabel("Frequency")
    plt.title(f"Score Distribution - {model_name} on {dataset_name}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(dist_fig_path)
    plt.close()

    return roc_fig_path, dist_fig_path, auc_score

def combine_images(image_paths, output_path):
    images = [Image.open(p) for p in image_paths]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    new_im.save(output_path)
    new_im.show()

def main():
    parser = argparse.ArgumentParser(
        description="Run anomaly detection models on specified datasets.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--dataset', type=str, required=True, choices=['nsl_kdd', 'unsw_nb15'], help='Dataset to use.')
    parser.add_argument('--model', type=str, required=True, choices=['iso', 'clsh', 'jlsh', 'minhash'], help='Model to use.')
    
    hyperparameters = parser.add_argument_group('Hyperparameters')
    hyperparameters.add_argument('--n_trees', type=int, default=100, help='Number of trees in the forest. Default: 100')
    hyperparameters.add_argument('--max_depth', type=int, default=10, help='Maximum depth of the trees. Default: 10')
    hyperparameters.add_argument('--min_samples', type=int, default=10, help='Minimum number of samples required to split a node. Default: 10')
    hyperparameters.add_argument('--hash_dim', type=int, default=3, help='Dimension of the hash for CLSH. Default: 3')
    hyperparameters.add_argument('--num_hashes', type=int, default=1, help='Number of hashes for JLSH and MinHash. Default: 1')
    hyperparameters.add_argument('--subsample_size', type=int, default=256, help='Subsample size for each tree. Default: 256')
    
    args = parser.parse_args()

    if args.dataset == 'nsl_kdd':
        X_train, y_train, X_test, y_test, attack_cats = DataLoader.load_nsl_kdd()
    elif args.dataset == 'unsw_nb15':
        X_train, y_train, X_test, y_test, attack_cats = DataLoader.load_unsw_nb15()

    if args.model == 'iso':
        model = IsolationForest(n_estimators=args.n_trees, max_samples=args.min_samples, random_state=42)
        model.fit(X_train)
        scores = model.decision_function(X_test)
    elif args.model == 'clsh':
        model = CLSHiForest(n_trees=args.n_trees, max_depth=args.max_depth, hash_dim=args.hash_dim, min_samples=args.min_samples, subsample_size=args.subsample_size, random_state=42)
        model.fit(X_train.to_numpy())
        scores = model.decision_function(X_test.to_numpy())
    elif args.model == 'jlsh':
        model = JLSHiForest(n_trees=args.n_trees, max_depth=args.max_depth, num_hashes=args.num_hashes, min_samples=args.min_samples, subsample_size=args.subsample_size, random_state=42)
        model.fit(X_train)
        scores = model.decision_function(X_test)
    elif args.model == 'minhash':
        model = MinLSHiForest(n_trees=args.n_trees, max_depth=args.max_depth, num_hashes=args.num_hashes, min_samples=args.min_samples, subsample_size=args.subsample_size, random_state=42)
        model.fit(X_train)
        scores = model.decision_function(X_test)

    roc_fig_path, dist_fig_path, auc_score = plot_results(scores, y_test, args.model, args.dataset)
    
    print(f"\n--- Experimental Results ---")
    print(f"  Model: {args.model}")
    print(f"  Dataset: {args.dataset}")
    print(f"  AUC Score: {auc_score:.4f}")
    
    y_pred = (scores < 0).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  Accuracy: {accuracy:.4f}")
    
    attack_analysis = AttackAnalyzer.analyze_by_attack(y_test, y_pred, attack_cats)
    AttackAnalyzer.print_analysis(attack_analysis)
    
    combine_images([roc_fig_path, dist_fig_path], f"results/{args.dataset}_{args.model}_composite.png")

    if hasattr(model, 'visualize_tree'):
        tree_fig_path = f"results/{args.dataset}_{args.model}_tree.png"
        model.visualize_tree(tree_index=0)
        plt.savefig(tree_fig_path)
        plt.close()

if __name__ == "__main__":
    main()
