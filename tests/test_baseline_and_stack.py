"""The baseline XGBoost and the stack, on tiny data: row order, folds, log-odds, and manifests."""
import hashlib
import json

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from conftest import ID_COL, TARGET_COL, run_script, tiny_table
from make_folds import make_fold_ids
from stack_lr import load_inputs, stack


@pytest.fixture
def signal_data(tmp_path):
    """A table where the label depends on a feature, so that a model has something to learn."""
    table = tiny_table(600, seed=12)
    rng = np.random.default_rng(12)
    age = table["customer_age"].astype(int)
    table[TARGET_COL] = np.where(age + rng.normal(0, 8, 600) > 60, "1.0", "0.0")
    test = tiny_table(100, seed=13).drop(columns=[TARGET_COL])
    train_csv, test_csv, fold_path = tmp_path / "train.csv", tmp_path / "test.csv", tmp_path / "fold_ids.npy"
    table.to_csv(train_csv, index=False)
    test.to_csv(test_csv, index=False)
    labels = (table[TARGET_COL] == "1.0").astype(int).to_numpy()
    fold_ids = make_fold_ids(labels, 3)
    np.save(fold_path, fold_ids)
    return {"train_csv": train_csv, "test_csv": test_csv, "fold_path": fold_path, "table": table, "labels": labels,
            "fold_ids": fold_ids}


def baseline(data, out, train_csv=None):
    return run_script("baseline_xgb.py", "--train-csv", train_csv or data["train_csv"], "--test-csv", data["test_csv"],
                      "--fold-ids", data["fold_path"], "--id-col", ID_COL, "--target-col", TARGET_COL, "--out", out,
                      "--rounds", 30, "--stopping-rounds", 10, "--device", "cpu")


def test_baseline_saves_log_odds_in_row_order_with_a_manifest(signal_data, tmp_path):
    baseline(signal_data, tmp_path / "xgb")
    oof, test = np.load(tmp_path / "xgb" / "oof.npy"), np.load(tmp_path / "xgb" / "test.npy")
    assert oof.shape == (600,) and test.shape == (100,)
    assert np.isfinite(oof).all() and np.isfinite(test).all()
    assert (oof < 0).any()   # log-odds: a probability could not be negative
    assert roc_auc_score(signal_data["labels"], oof) > 0.8   # row order is right, or this would read near 0.5
    manifest = json.loads((tmp_path / "xgb" / "manifest.json").read_text())
    assert manifest["auc"]["pooled"] == pytest.approx(roc_auc_score(signal_data["labels"], oof), abs=1e-12)
    assert manifest["sha256"]["oof.npy"] == hashlib.sha256((tmp_path / "xgb" / "oof.npy").read_bytes()).hexdigest()
    assert len(manifest["per_fold"]) == 3 and manifest["seconds"] > 0
    assert str(tmp_path) not in (tmp_path / "xgb" / "manifest.json").read_text()


def test_baseline_held_out_labels_do_not_reach_its_fold_model(signal_data, tmp_path):
    baseline(signal_data, tmp_path / "plain")
    held_out = signal_data["fold_ids"] == 0
    changed = signal_data["table"].copy()
    changed.loc[held_out, TARGET_COL] = np.where(changed.loc[held_out, TARGET_COL] == "1.0", "0.0", "1.0")
    changed_csv = tmp_path / "changed.csv"
    changed.to_csv(changed_csv, index=False)
    baseline(signal_data, tmp_path / "changed", changed_csv)
    assert (np.load(tmp_path / "plain" / "oof.npy")[held_out] == np.load(tmp_path / "changed" / "oof.npy")[held_out]).all()


def test_stack_equals_fitting_each_fold_by_hand():
    rng = np.random.default_rng(14)
    labels = rng.integers(0, 2, 300)
    oof_inputs = np.column_stack([labels * 2.0 + rng.normal(0, 1, 300), labels * 1.0 + rng.normal(0, 1, 300)])
    test_inputs = rng.normal(0, 1, (50, 2))
    fold_ids = make_fold_ids(labels, 3)
    oof, test, per_fold = stack(oof_inputs, test_inputs, labels, fold_ids)
    probabilities = []
    for fold in range(3):
        fit_rows, held = fold_ids != fold, fold_ids == fold
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, tol=1e-7).fit(oof_inputs[fit_rows], labels[fit_rows])
        assert np.allclose(oof[held], model.decision_function(oof_inputs[held]))
        probabilities.append(model.predict_proba(test_inputs)[:, 1])
        assert per_fold[fold]["auc"] == pytest.approx(roc_auc_score(labels[held], oof[held]))
    mean_probability = np.mean(probabilities, axis=0)
    assert np.allclose(test, np.log(mean_probability / (1 - mean_probability)))


def test_stack_refuses_a_missing_value_and_clips_extremes(tmp_path):
    good, bad = tmp_path / "good", tmp_path / "bad"
    good.mkdir()
    bad.mkdir()
    np.save(good / "oof.npy", np.array([1.0, 50.0, -50.0]))
    np.save(good / "test.npy", np.array([0.0]))
    np.save(bad / "oof.npy", np.array([1.0, np.nan, 0.0]))
    np.save(bad / "test.npy", np.array([0.0]))
    oof, _ = load_inputs([good], 3)
    assert oof.max() == 30 and oof.min() == -30
    with pytest.raises(ValueError):
        load_inputs([bad], 3)
    with pytest.raises(ValueError):
        load_inputs([good], 4)


def test_stack_script_end_to_end_on_baseline_outputs(signal_data, tmp_path):
    baseline(signal_data, tmp_path / "one")
    second = tmp_path / "two"
    second.mkdir()
    rng = np.random.default_rng(15)
    np.save(second / "oof.npy", signal_data["labels"] * 1.0 + rng.normal(0, 1, 600))
    np.save(second / "test.npy", rng.normal(0, 1, 100))
    run_script("stack_lr.py", "--input", tmp_path / "one", "--input", second, "--train-csv", signal_data["train_csv"],
               "--fold-ids", signal_data["fold_path"], "--target-col", TARGET_COL, "--out", tmp_path / "stack")
    oof = np.load(tmp_path / "stack" / "oof.npy")
    manifest = json.loads((tmp_path / "stack" / "manifest.json").read_text())
    assert oof.shape == (600,) and np.load(tmp_path / "stack" / "test.npy").shape == (100,)
    assert manifest["auc"]["pooled"] == pytest.approx(roc_auc_score(signal_data["labels"], oof), abs=1e-12)
    assert manifest["members"] == ["one", "two"]
    assert str(tmp_path) not in (tmp_path / "stack" / "manifest.json").read_text()
