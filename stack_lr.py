"""Stack several models with an L2 logistic regression on their log-odds, on the shared folds.

Follows the stacker the post describes: logits clipped at +/-30, L2, C=1, no standardization, four folds to fit and
the fifth to predict, test as the mean of the fold models' probabilities. scikit-learn replaces cuML.

Input:  folders holding oof.npy and test.npy as log-odds, train.csv, fold_ids.npy.
Output: OUT/oof.npy in train.csv row order, OUT/test.npy in test.csv row order, both log-odds, OUT/manifest.json with
        settings, per-fold AUC, weights, seconds and sha256.

Run:    python stack_lr.py --input GPT_DIR --input XGB_DIR --train-csv train.csv --fold-ids fold_ids.npy
            --target-col is_fraud --out OUT
"""
import argparse
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import fit_and_held_out, named_hash, print_summary, save_npy, sha256_file, write_json
from make_folds import read_labels

LOGIT_CLIP = 30.0
PROBABILITY_CLIP = 1e-15


def load_inputs(directories, n_train):
    """Stacked inputs, rows x models. Raises on a wrong shape or a missing value."""
    oof = np.column_stack([np.load(Path(d) / "oof.npy") for d in directories])
    test = np.column_stack([np.load(Path(d) / "test.npy") for d in directories])
    if oof.shape[0] != n_train:
        raise ValueError("an oof.npy does not match train.csv")
    if not (np.isfinite(oof).all() and np.isfinite(test).all()):
        raise ValueError("an input has a missing or non-finite value")
    return np.clip(oof, -LOGIT_CLIP, LOGIT_CLIP), np.clip(test, -LOGIT_CLIP, LOGIT_CLIP)


def stack(oof_inputs, test_inputs, labels, fold_ids, C=1.0):
    """OOF log-odds, test log-odds, the per-fold AUC and the fold models' weights."""
    oof = np.full(len(labels), np.nan)
    test_probability = np.zeros(test_inputs.shape[0])
    per_fold = []
    n_folds = int(fold_ids.max()) + 1
    for fold in range(n_folds):
        fit_rows, held_out = fit_and_held_out(fold_ids, fold)
        model = LogisticRegression(C=C, fit_intercept=True, solver="lbfgs", max_iter=2000, tol=1e-7)
        model.fit(oof_inputs[fit_rows], labels[fit_rows])
        oof[held_out] = model.decision_function(oof_inputs[held_out])
        test_probability += model.predict_proba(test_inputs)[:, 1]
        per_fold.append({"fold": fold, "auc": float(roc_auc_score(labels[held_out], oof[held_out])),
                         "weights": model.coef_[0].tolist(), "intercept": float(model.intercept_[0])})
    probability = np.clip(test_probability / n_folds, PROBABILITY_CLIP, 1 - PROBABILITY_CLIP)
    return oof, np.log(probability / (1 - probability)), per_fold


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", action="append", required=True, help="a folder with oof.npy and test.npy; repeat it")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--fold-ids", required=True)
    parser.add_argument("--target-col", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--C", type=float, default=1.0, help="the post uses C=1")
    args = parser.parse_args()

    started = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    labels = read_labels(args.train_csv, args.target_col)
    fold_ids = np.load(args.fold_ids)
    if len(fold_ids) != len(labels):
        raise ValueError("fold_ids does not match train.csv")
    oof_inputs, test_inputs = load_inputs(args.input, len(labels))
    oof, test, per_fold = stack(oof_inputs, test_inputs, labels, fold_ids, args.C)
    save_npy(out / "oof.npy", oof)
    save_npy(out / "test.npy", test)
    names = [Path(d).name for d in args.input]
    manifest = {
        "model": "logistic_regression_stack",
        "output": "log-odds; oof.npy in train.csv row order, test.npy in test.csv row order",
        "members": names,
        "member_auc": {name: float(roc_auc_score(labels, oof_inputs[:, j])) for j, name in enumerate(names)},
        "settings": {"C": args.C, "logit_clip": LOGIT_CLIP, "solver": "lbfgs", "max_iter": 2000, "tol": 1e-7,
                     "standardize": False, "test": "mean of the fold models' probabilities"},
        "auc": {"pooled": float(roc_auc_score(labels, oof)), "mean_of_folds": float(np.mean([f["auc"] for f in per_fold]))},
        "per_fold": per_fold,
        "seconds": time.time() - started,
        "sha256": {"oof.npy": sha256_file(out / "oof.npy"), "test.npy": sha256_file(out / "test.npy")},
        "inputs": {"fold_ids": named_hash(args.fold_ids),
                   "member_oof_sha256": {n: sha256_file(Path(d) / "oof.npy") for n, d in zip(names, args.input)}},
    }
    write_json(out / "manifest.json", manifest)
    print_summary("stack_lr", status="done", auc=manifest["auc"], member_auc=manifest["member_auc"])


if __name__ == "__main__":
    main()
