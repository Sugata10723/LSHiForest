# LSH-based Isolation Forest

This project implements and evaluates LSH-based Isolation Forest algorithms for anomaly detection.

## Installation

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

The main script for running experiments is `test.py`. It allows you to select the dataset, model, and hyperparameters from the command line.

### Command-Line Options

*   `--dataset`: The dataset to use. Choices are `nsl_kdd` and `unsw_nb15`. (Required)
*   `--model`: The model to use. Choices are `iso` (Isolation Forest), `clsh` (Categorical LSH), `jlsh` (Jaccard LSH), and `minhash` (MinHash LSH). (Required)
*   `--n_trees`: The number of trees in the forest. (Default: 100)
*   `--max_depth`: The maximum depth of the trees. (Default: 10)
*   `--min_samples`: The minimum number of samples required to split a node. (Default: 10)
*   `--hash_dim`: The dimension of the hash for CLSH. (Default: 3)
*   `--num_hashes`: The number of hashes for JLSH and MinHash. (Default: 1)
*   `--subsample_size`: The subsample size for each tree. (Default: 256)

### Example

To run the CLSH model on the NSL-KDD dataset, use the following command:

```bash
python3 test.py --dataset nsl_kdd --model clsh
```

To run the MinHash model on the UNSW-NB15 dataset with 200 trees and a max depth of 15:

```bash
python3 test.py --dataset unsw_nb15 --model minhash --n_trees 200 --max_depth 15
```
To see all available options and their descriptions, run:
```bash
python3 test.py --help
```
