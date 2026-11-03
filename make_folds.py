"""Make the fold assignment that every model shares.

Input:  the competition train.csv.
Does:   StratifiedKFold(5, shuffle=True, random_state=42) on the target.
Output: fold_ids.npy, one integer per train.csv row, in train.csv row order.

Run:    python make_folds.py --train-csv train.csv --target-col is_fraud --out fold_ids.npy
"""
import argparse

import numpy as np
from sklearn.model_selection import StratifiedKFold

from common import print_summary, save_npy, sha256_file
from tabtext import label_text, read_csv_as_strings


def make_fold_ids(labels, n_splits=5, seed=42):
    labels = np.asarray(labels)
    fold_ids = np.full(len(labels), -1, dtype=np.int8)
    splitter = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    for fold, (_, held_out) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
        fold_ids[held_out] = fold
    return fold_ids


def read_labels(csv_path, target_col):
    """The target as 0 and 1, in CSV row order."""
    values = read_csv_as_strings(csv_path)[target_col]
    return (values.map(label_text) == "True").to_numpy(dtype=np.int8)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--target-col", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    labels = read_labels(args.train_csv, args.target_col)
    fold_ids = make_fold_ids(labels, args.n_splits, args.seed)
    save_npy(args.out, fold_ids)
    positives = [int(labels[fold_ids != k].sum()) for k in range(args.n_splits)]
    print_summary("make_folds", rows=len(labels), positives=int(labels.sum()), positives_in_fit_rows=positives,
                  fit_rows=[int((fold_ids != k).sum()) for k in range(args.n_splits)], sha256=sha256_file(args.out))


if __name__ == "__main__":
    main()
