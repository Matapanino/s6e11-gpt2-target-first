"""Baseline for comparison: XGBoost on the raw columns, same folds, no feature engineering and no tuning.

Input:  train.csv, test.csv, fold_ids.npy.
Does:   numeric columns stay numeric, every other column becomes a category shared between train and test. Early
        stopping reads a split taken from the fit rows only, because stopping on the scored fold would make CV
        optimistic (the post warns about this in public OOFs).
Output: OUT/oof.npy in train.csv row order, OUT/test.npy in test.csv row order, both log-odds (test is the mean over
        the fold models), OUT/manifest.json with settings, per-fold AUC, seconds and sha256.

Run:    python baseline_xgb.py --train-csv train.csv --test-csv test.csv --fold-ids fold_ids.npy --id-col id
            --target-col is_fraud --out OUT
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from common import fit_and_held_out, named_hash, print_summary, save_npy, sha256_file, write_json
from tabtext import label_text


def raw_features(train_csv, test_csv, id_col, target_col):
    """Feature tables for train and test, and the labels. Categories come from the feature columns of both files."""
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)
    labels = (train[target_col].astype(str).map(label_text) == "True").to_numpy(dtype=np.int8)
    columns = [c for c in train.columns if c not in (id_col, target_col)]
    train, test = train[columns].copy(), test[columns].copy()
    for column in columns:
        if train[column].dtype.kind not in "if":
            categories = sorted(set(train[column].astype(str)) | set(test[column].astype(str)))
            train[column] = pd.Categorical(train[column].astype(str), categories=categories)
            test[column] = pd.Categorical(test[column].astype(str), categories=categories)
    return train, test, labels


def fit_one_fold(train, test, labels, fold_ids, fold, params, rounds, stopping_rounds, inner_fraction, seed):
    """One fold model. Returns the log-odds for the held-out rows and for test, and the rounds used."""
    fit_rows, held_out = fit_and_held_out(fold_ids, fold)
    fit_part, stop_part = train_test_split(fit_rows, test_size=inner_fraction, random_state=seed + fold,
                                           stratify=labels[fit_rows])
    fit_matrix = xgb.DMatrix(train.iloc[fit_part], label=labels[fit_part], enable_categorical=True)
    stop_matrix = xgb.DMatrix(train.iloc[stop_part], label=labels[stop_part], enable_categorical=True)
    booster = xgb.train({**params, "seed": seed + fold}, fit_matrix, num_boost_round=rounds,
                        evals=[(stop_matrix, "stop")], early_stopping_rounds=stopping_rounds, verbose_eval=False)
    best = booster.best_iteration + 1
    held_matrix = xgb.DMatrix(train.iloc[held_out], enable_categorical=True)
    test_matrix = xgb.DMatrix(test, enable_categorical=True)
    return (held_out, booster.predict(held_matrix, iteration_range=(0, best), output_margin=True),
            booster.predict(test_matrix, iteration_range=(0, best), output_margin=True), best)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--fold-ids", required=True)
    parser.add_argument("--id-col", required=True)
    parser.add_argument("--target-col", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rounds", type=int, default=5000)
    parser.add_argument("--stopping-rounds", type=int, default=200)
    parser.add_argument("--inner-fraction", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train, test, labels = raw_features(args.train_csv, args.test_csv, args.id_col, args.target_col)
    fold_ids = np.load(args.fold_ids)
    if len(fold_ids) != len(labels):
        raise ValueError("fold_ids does not match train.csv")
    params = {"objective": "binary:logistic", "eval_metric": "auc", "tree_method": "hist", "device": args.device,
              "learning_rate": args.learning_rate, "max_depth": args.max_depth, "subsample": args.subsample,
              "colsample_bytree": args.colsample}

    oof = np.full(len(labels), np.nan)
    test_sum = np.zeros(len(test))
    per_fold = []
    started = time.time()
    n_folds = int(fold_ids.max()) + 1
    for fold in range(n_folds):
        fold_started = time.time()
        held_out, oof_part, test_part, best = fit_one_fold(train, test, labels, fold_ids, fold, params, args.rounds,
                                                           args.stopping_rounds, args.inner_fraction, args.seed)
        oof[held_out] = oof_part
        test_sum += test_part
        per_fold.append({"fold": fold, "auc": float(roc_auc_score(labels[held_out], oof_part)), "rounds": best,
                         "seconds": time.time() - fold_started, "rows": len(held_out)})
        print_summary("baseline_xgb", **per_fold[-1])
    test_log_odds = test_sum / n_folds
    if not (np.isfinite(oof).all() and np.isfinite(test_log_odds).all()):
        raise ValueError("non-finite prediction")
    save_npy(out / "oof.npy", oof)
    save_npy(out / "test.npy", test_log_odds)
    settings = {k: v for k, v in vars(args).items() if k not in ("train_csv", "test_csv", "fold_ids", "out", "device")}
    manifest = {
        "model": "xgboost_baseline_raw_columns",
        "output": "log-odds; oof.npy in train.csv row order, test.npy in test.csv row order",
        "settings": settings,
        "auc": {"pooled": float(roc_auc_score(labels, oof)), "mean_of_folds": float(np.mean([f["auc"] for f in per_fold]))},
        "per_fold": per_fold,
        "seconds": time.time() - started,
        "sha256": {"oof.npy": sha256_file(out / "oof.npy"), "test.npy": sha256_file(out / "test.npy")},
        "inputs": {"train_csv": named_hash(args.train_csv), "test_csv": named_hash(args.test_csv),
                   "fold_ids": named_hash(args.fold_ids)},
    }
    write_json(out / "manifest.json", manifest)
    print_summary("baseline_xgb", status="done", auc=manifest["auc"], seconds=manifest["seconds"])


if __name__ == "__main__":
    main()
