"""Step 4: turn scored chunks into oof.npy and test.npy, as log-odds, and write manifest.json.

The post's formula for one row:
    log-odds = logsumexp over orders of log P(x | True, o)
             - logsumexp over orders of log P(x | False, o)
             + log(prior / (1 - prior))
The prior is the training rate of the fold's fit rows, so it uses no held-out label.

Input:  SCORES/f{k}_oof/ and SCORES/f{k}_test/ from score_rows.py, train.csv, test.csv, fold_ids.npy, and optionally
        the folders holding train_summary.json for the seconds and settings. With --fold-summaries, scores made by any
        model other than their fold's are refused.
Does:   OOF row k gets the score from the one fold model that never saw it. Test is the mean of the five fold models'
        log-odds (the post does not say how test was built; see the README).
Output: OUT/oof.npy in train.csv row order, OUT/test.npy in test.csv row order, OUT/manifest.json.
        The manifest also reads AUC at fewer orders, mean against log-sum-exp, and the per-label fit, because
        P(features | True) trains on far fewer rows.

Run:    python make_predictions.py --scores SCORES --train-csv train.csv --test-csv test.csv --fold-ids fold_ids.npy
            --target-col is_fraud --out OUT
Early:  add --allow-partial and a separate OUT to read the AUC from whatever chunks exist while scoring runs. Add
        --compare-oof BASELINE/oof.npy to read another model on the same rows: a first chunk has few positives and its
        AUC differs from the whole fold's, so compare it only with compare_auc_same_rows.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score

from common import fold_prior, named_hash, print_summary, save_npy, sha256_file, write_json
from make_folds import read_labels
from tabtext import read_csv_as_strings


def log_odds_from_orders(log_p_true, log_p_false, prior, combine="lse", layout="first"):
    """The post's formula. Both inputs are rows x orders; the same number of orders cancels any 1/K."""
    prior_term = np.log(prior / (1 - prior)) if layout == "first" else 0
    if combine == "mean":
        return (log_p_true - log_p_false).mean(axis=1) + prior_term
    if combine != "lse":
        raise ValueError(combine)
    return logsumexp(log_p_true, axis=1) - logsumexp(log_p_false, axis=1) + prior_term


def same_settings(name, settings_by_fold, differ_by_design):
    """One settings entry for all folds, for the manifest. Folds that used different settings are refused."""
    kept = [{k: v for k, v in settings.items() if k not in differ_by_design} for settings in settings_by_fold]
    if any(settings != kept[0] for settings in kept):
        raise ValueError(f"{name} settings differ between folds: {kept}")
    return kept[0] if kept else None


def check_scored_by_fold_model(score_directory, fit_directory):
    """A run with the wrong --model-dir would pass the row check and read as a great CV, so compare the model hashes."""
    recorded = json.loads((Path(score_directory) / "settings.json").read_text())["model_sha256"]
    if recorded != sha256_file(Path(fit_directory) / "final" / "model.safetensors"):
        raise ValueError(f"{Path(score_directory).name} was scored by another model than {Path(fit_directory).name}")


def has_scores(directory):
    return any(Path(directory).glob("chunk_?????.npz"))


def load_scores(directory):
    """All chunks of one score folder: rows, log_p_true, log_p_false, seconds. Unfinished temp files are ignored."""
    chunks = [np.load(path) for path in sorted(Path(directory).glob("chunk_?????.npz"))]
    if not chunks:
        raise FileNotFoundError(f"no chunks in {Path(directory).name}")
    return (np.concatenate([c["rows"] for c in chunks]),
            np.concatenate([c["log_p_true"] for c in chunks]),
            np.concatenate([c["log_p_false"] for c in chunks]),
            float(sum(c["seconds"] for c in chunks)))


def safe_auc(labels, scores):
    return float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None


def label_fit(labels, log_p_true, log_p_false):
    """Mean summed log-probability under each label, on rows that are really positive and really negative.

    If the True model were starved of rows, True would read far below False on every row, positive rows included.
    """
    fit = {}
    for name, mask in (("positive_rows", labels == 1), ("negative_rows", labels == 0)):
        if mask.any():
            fit[name] = {"rows": int(mask.sum()),
                         "mean_log_p_true": float(log_p_true[mask].mean()),
                         "mean_log_p_false": float(log_p_false[mask].mean()),
                         "mean_true_minus_false": float((log_p_true[mask] - log_p_false[mask]).mean())}
    return fit


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--fold-ids", required=True)
    parser.add_argument("--target-col", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--replica-summary", default="", help="train_summary.json of the replica, for the seconds")
    parser.add_argument("--fold-summaries", default="", help="folder with f{k}/train_summary.json, for the seconds")
    parser.add_argument("--compare-oof", default="", help="oof.npy of another model: each fold's entry also reads it on the same rows")
    parser.add_argument("--model-name", default="gpt2_target_first")
    parser.add_argument("--use-orders", type=int, default=None,
                        help="assemble from the first N orders of every OOF and test chunk")
    parser.add_argument("--combine", choices=["lse", "mean"], default="lse")
    parser.add_argument("--allow-partial", action="store_true",
                        help="for smoke runs and early reads: rows or folds not scored yet stay NaN")
    args = parser.parse_args()
    if args.use_orders is not None and args.use_orders < 1:
        parser.error("--use-orders must be positive")

    started = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scores = Path(args.scores)
    available_orders = {}
    layouts = set()
    for path in sorted(scores.glob('f*_*/settings.json')):
        setting = json.loads(path.read_text())
        count = int(setting['orders'])
        layouts.add(setting.get('layout', 'first'))
        available_orders[path.parent.name] = count
        if args.use_orders is not None and count < args.use_orders:
            raise ValueError(f'{path.parent.name} has {count} orders; --use-orders {args.use_orders} requested')
    if len(layouts) > 1:
        raise ValueError(f'score folders have mixed layouts: {sorted(layouts)}')
    layout = next(iter(layouts), 'first')
    labels = read_labels(args.train_csv, args.target_col)
    fold_ids = np.load(args.fold_ids)
    if len(fold_ids) != len(labels):
        raise ValueError("fold_ids does not match train.csv")
    n_test = len(read_csv_as_strings(args.test_csv))
    n_folds = int(fold_ids.max()) + 1
    other = np.load(args.compare_oof) if args.compare_oof else None
    if other is not None and len(other) != len(labels):
        raise ValueError("--compare-oof does not match train.csv")

    oof = np.full(len(labels), np.nan)
    test_by_fold = np.full((n_folds, n_test), np.nan)
    per_fold, score_seconds, n_orders = [], 0.0, None
    all_true, all_false, all_rows = [], [], []
    for fold in range(n_folds):
        prior = fold_prior(labels, fold_ids, fold)
        if args.allow_partial and not has_scores(scores / f"f{fold}_oof"):
            continue
        rows, log_p_true, log_p_false, seconds = load_scores(scores / f"f{fold}_oof")
        if args.use_orders is not None:
            if log_p_true.shape[1] < args.use_orders:
                raise ValueError(f'f{fold}_oof chunks have fewer than {args.use_orders} orders')
            log_p_true = log_p_true[:, :args.use_orders]
            log_p_false = log_p_false[:, :args.use_orders]
        score_seconds += seconds
        if not np.all(fold_ids[rows] == fold):
            raise ValueError(f"fold {fold} has a score for a row its model trained on")
        if len(np.unique(rows)) != len(rows):
            raise ValueError(f"fold {fold} scored a row twice")
        if n_orders is not None and n_orders != log_p_true.shape[1]:
            raise ValueError('OOF folds have different orders; pass --use-orders N')
        n_orders = log_p_true.shape[1]
        oof[rows] = log_odds_from_orders(log_p_true, log_p_false, prior, args.combine, layout)
        all_true.append(log_p_true)
        all_false.append(log_p_false)
        all_rows.append(rows)
        if not args.allow_partial or has_scores(scores / f"f{fold}_test"):
            rows_test, test_true, test_false, seconds = load_scores(scores / f"f{fold}_test")
            if args.use_orders is not None:
                if test_true.shape[1] < args.use_orders:
                    raise ValueError(f'f{fold}_test chunks have fewer than {args.use_orders} orders')
                test_true = test_true[:, :args.use_orders]
                test_false = test_false[:, :args.use_orders]
            score_seconds += seconds
            test_by_fold[fold, rows_test] = log_odds_from_orders(test_true, test_false, prior, args.combine, layout)
        per_fold.append({"fold": fold, "auc": safe_auc(labels[rows], oof[rows]), "prior": prior, "rows": len(rows),
                         "positives": int(labels[rows].sum()),
                         "label_fit": label_fit(labels[rows], log_p_true, log_p_false)})
        if other is not None:
            per_fold[-1]["compare_auc_same_rows"] = safe_auc(labels[rows], other[rows])
    models_per_test_row = np.isfinite(test_by_fold).sum(axis=0)
    test = np.full(n_test, np.nan)
    test[models_per_test_row > 0] = np.nansum(test_by_fold, axis=0)[models_per_test_row > 0] \
        / models_per_test_row[models_per_test_row > 0]

    scored = np.isfinite(oof)
    if not args.allow_partial and not (scored.all() and (models_per_test_row == n_folds).all()):
        raise ValueError("some rows have no score: run score_rows.py to the end, or use --allow-partial for an early read")
    if not np.isfinite(oof[scored]).all():
        raise ValueError("non-finite OOF score")
    save_npy(out / "oof.npy", oof)
    save_npy(out / "test.npy", test)

    rows = np.concatenate(all_rows)
    log_p_true = np.concatenate(all_true)
    log_p_false = np.concatenate(all_false)
    priors = np.array([fold_prior(labels, fold_ids, k) for k in range(n_folds)])[fold_ids[rows]]
    by_orders = {}
    for count in sorted({1, 6, 10, 16, n_orders} & set(range(1, n_orders + 1))):
        reading = log_odds_from_orders(log_p_true[:, :count], log_p_false[:, :count], 0.5, args.combine, layout)
        if layout == 'first':
            reading += np.log(priors / (1 - priors))
        by_orders[str(count)] = safe_auc(labels[rows], reading)
    mean_of_per_order = (log_p_true - log_p_false).mean(axis=1)

    seconds = {"scoring": score_seconds, "predictions": None}
    step_settings = {}
    if args.replica_summary:
        replica_summary = json.loads(Path(args.replica_summary).read_text())
        seconds["replica"] = replica_summary["seconds"]
        step_settings["replica"] = replica_summary["settings"]
    if args.fold_summaries:
        summaries = {k: Path(args.fold_summaries) / f"f{k}" / "train_summary.json" for k in range(n_folds)}
        summaries = {k: json.loads(path.read_text()) for k, path in summaries.items() if path.exists()}
        seconds["fold_fits"] = {str(k): summary["seconds"] for k, summary in summaries.items()}
        step_settings["fit"] = same_settings("fit", [s["settings"] for s in summaries.values()], {"fold"})
        for fold in range(n_folds):
            for split in ("oof", "test"):
                if has_scores(scores / f"f{fold}_{split}"):
                    check_scored_by_fold_model(scores / f"f{fold}_{split}", Path(args.fold_summaries) / f"f{fold}")
    step_settings["scoring"] = {}
    for split in ("oof", "test"):
        folders = [scores / f"f{fold}_{split}" / "settings.json" for fold in range(n_folds)]
        step_settings["scoring"][split] = same_settings(
            f"{split} scoring", [json.loads(path.read_text()) for path in folders if path.exists()],
            {"split", "fold", "model_sha256", *(('orders',) if args.use_orders is not None else ())})
    seconds["predictions"] = time.time() - started

    manifest = {
        "model": args.model_name,
        "credit": "Target-first GPT-2 classifier from the S6E10 1st-place solution by Chris Deotte: "
                  "https://www.kaggle.com/competitions/playground-series-s6e10/discussion/754683",
        "output": "log-odds; oof.npy in train.csv row order, test.npy in test.csv row order",
        "settings": {"orders": n_orders, "combine": (("mean of per-order log-odds" if args.combine == "mean"
                     else "log-sum-exp over orders per label, then the difference") +
                     (", plus the prior" if layout == "first" else "")),
                     "prior": "training rate of the fold's fit rows" if layout == "first" else "included in scores",
                     "test": "mean over fold models of the per-fold log-odds",
                     **({"available_orders_by_folder": available_orders} if args.use_orders is not None else {}),
                     **step_settings},
        "rows": {"train": len(labels), "test": n_test, "oof_scored": int(scored.sum()),
                 "test_scored": int(np.isfinite(test).sum())},
        "auc": {"pooled": safe_auc(labels[scored], oof[scored]),
                "mean_of_folds": float(np.mean([f["auc"] for f in per_fold if f["auc"] is not None]))
                if any(f["auc"] is not None for f in per_fold) else None},
        "per_fold": per_fold,
        "diagnostics": {"auc_by_number_of_orders": by_orders,
                        "auc_mean_of_per_order_log_odds": safe_auc(labels[rows], mean_of_per_order)},
        "seconds": seconds,
        "sha256": {"oof.npy": sha256_file(out / "oof.npy"), "test.npy": sha256_file(out / "test.npy")},
        "inputs": {"train_csv": named_hash(args.train_csv), "test_csv": named_hash(args.test_csv),
                   "fold_ids": named_hash(args.fold_ids)},
    }
    write_json(out / "manifest.json", manifest)
    print_summary("make_predictions", status="done", auc=manifest["auc"], rows=manifest["rows"])


if __name__ == "__main__":
    main()
