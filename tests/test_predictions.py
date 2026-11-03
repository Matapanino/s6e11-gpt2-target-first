"""Log-sum-exp and prior arithmetic by hand, row order of the saved arrays, and the manifest."""
import hashlib
import json

import numpy as np
import pytest
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from conftest import TARGET_COL, run_script
from common import fold_prior
from make_folds import read_labels
from make_predictions import load_scores, log_odds_from_orders


def test_log_sum_exp_and_prior_by_hand():
    log_p_true = np.array([[-10.0, -12.0]])
    log_p_false = np.array([[-11.0, -11.5]])
    prior = 0.25
    true_part = -10.0 + np.log(1 + np.exp(-2.0))
    false_part = -11.0 + np.log(1 + np.exp(-0.5))
    by_hand = true_part - false_part + np.log(0.25 / 0.75)
    assert log_odds_from_orders(log_p_true, log_p_false, prior)[0] == pytest.approx(by_hand, abs=1e-12)


def test_one_order_is_the_plain_difference_plus_the_prior():
    """The post's worked form: one order gives lt - lf + log(prior / (1 - prior))."""
    result = log_odds_from_orders(np.array([[-100.0]]), np.array([[-101.0]]), 0.4436)
    assert result[0] == pytest.approx(np.log(0.4436 / 0.5564) + 1.0, abs=1e-12)


def test_log_sum_exp_is_the_same_as_averaging_probabilities():
    rng = np.random.default_rng(0)
    log_p_true, log_p_false = -rng.random((50, 7)) * 40 - 150, -rng.random((50, 7)) * 40 - 150
    averaged = np.log(np.exp(log_p_true + 170).mean(1)) - np.log(np.exp(log_p_false + 170).mean(1))
    assert np.allclose(log_odds_from_orders(log_p_true, log_p_false, 0.5), averaged, atol=1e-10)


def test_log_sum_exp_stays_finite_on_very_small_probabilities():
    assert np.isfinite(log_odds_from_orders(np.full((3, 30), -2000.0), np.full((3, 30), -2500.0), 0.0167)).all()


def test_the_prior_shifts_every_row_equally_and_keeps_the_order():
    rng = np.random.default_rng(1)
    log_p_true, log_p_false = -rng.random((200, 5)) * 40, -rng.random((200, 5)) * 40
    labels = rng.integers(0, 2, 200)
    low = log_odds_from_orders(log_p_true, log_p_false, 0.0167)
    high = log_odds_from_orders(log_p_true, log_p_false, 0.5)
    assert np.allclose(low - high, np.log(0.0167 / 0.9833))
    assert roc_auc_score(labels, low) == roc_auc_score(labels, high)


def fake_model_hash(fold):
    return hashlib.sha256(f"model {fold}".encode()).hexdigest()


def write_chunks(directory, rows, n_orders, seed, chunk_rows=7):
    """Fake score chunks, written so that a row's true score encodes its own row number."""
    directory.mkdir(parents=True)
    (directory / "settings.json").write_text(json.dumps({
        "split": directory.name.split("_")[1], "fold": seed, "orders": n_orders, "chunk_rows": chunk_rows,
        "precision": "fp32", "seed": 42, "max_rows": 0, "model_sha256": fake_model_hash(seed)}))
    for index in range(0, len(rows), chunk_rows):
        part = rows[index:index + chunk_rows]
        true_scores = np.tile((-part / 1000.0)[:, None], (1, n_orders)) + seed
        false_scores = np.full((len(part), n_orders), float(seed))
        np.savez(directory / f"chunk_{index // chunk_rows:05d}.npz", rows=part, log_p_true=true_scores,
                 log_p_false=false_scores, seconds=1.5)


@pytest.fixture
def scored(tiny_data, tmp_path):
    scores = tmp_path / "scores"
    fold_ids = tiny_data["fold_id_array"]
    for fold in range(3):
        held_out = np.flatnonzero(fold_ids == fold)
        write_chunks(scores / f"f{fold}_oof", held_out, 3, fold)
        write_chunks(scores / f"f{fold}_test", np.arange(len(tiny_data["test"])), 3, fold)
    return scores


def predict(tiny_data, scores, out, *extra, check=True):
    return run_script("make_predictions.py", "--scores", scores, "--train-csv", tiny_data["train_csv"], "--test-csv",
                      tiny_data["test_csv"], "--fold-ids", tiny_data["fold_ids"], "--target-col", TARGET_COL, "--out",
                      out, *extra, check=check)


def test_use_orders_slices_mixed_fold_scores_and_checks_settings(tiny_data, scored, tmp_path):
    # Give the unused third order a large effect in both splits.
    for split in ('oof', 'test'):
        path = scored / f'f0_{split}/chunk_00000.npz'
        with np.load(path) as chunk:
            rows = chunk['rows'].copy()
            true = chunk['log_p_true'].copy()
            false = chunk['log_p_false'].copy()
            seconds = float(chunk['seconds'])
        true[:, 2] += 20
        np.savez(path, rows=rows, log_p_true=true, log_p_false=false, seconds=seconds)
    path = scored / 'f1_oof/settings.json'
    settings = json.loads(path.read_text())
    settings['orders'] = 2
    path.write_text(json.dumps(settings))
    for chunk_path in (scored / 'f1_oof').glob('chunk_*.npz'):
        with np.load(chunk_path) as chunk:
            np.savez(chunk_path, rows=chunk['rows'], log_p_true=chunk['log_p_true'][:, :2],
                     log_p_false=chunk['log_p_false'][:, :2], seconds=float(chunk['seconds']))
    assert predict(tiny_data, scored, tmp_path / 'default', check=False).returncode != 0
    out = tmp_path / 'selected'
    predict(tiny_data, scored, out, '--use-orders', 2)
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['settings']['orders'] == 2
    assert manifest['settings']['available_orders_by_folder']['f1_oof'] == 2
    assert manifest['settings']['available_orders_by_folder']['f0_test'] == 3
    labels = read_labels(tiny_data['train_csv'], TARGET_COL)
    folds = tiny_data['fold_id_array']
    oof = np.load(out / 'oof.npy')
    rows, true, false, _ = load_scores(scored / 'f0_oof')
    prior = fold_prior(labels, folds, 0)
    expected = log_odds_from_orders(true[:, :2], false[:, :2], prior)
    assert np.allclose(oof[rows], expected)
    assert not np.allclose(oof[rows], log_odds_from_orders(true, false, prior))
    test_parts = []
    for fold in range(3):
        test_rows, true, false, _ = load_scores(scored / f'f{fold}_test')
        test_parts.append((test_rows, log_odds_from_orders(true[:, :2], false[:, :2],
                                                           fold_prior(labels, folds, fold))))
    expected_test = np.zeros(len(tiny_data['test']))
    for test_rows, part in test_parts:
        expected_test[test_rows] += part / 3
    assert np.allclose(np.load(out / 'test.npy'), expected_test)
    assert 'f1_oof has 2 orders' in predict(tiny_data, scored, tmp_path / 'too_many',
                                          '--use-orders', 3, check=False).stderr


def test_oof_and_test_keep_the_csv_row_order_and_the_manifest_is_complete(tiny_data, scored, tmp_path):
    out = tmp_path / "out"
    predict(tiny_data, scored, out)
    oof, test = np.load(out / "oof.npy"), np.load(out / "test.npy")
    assert oof.shape == (60,) and test.shape == (20,)
    fold_ids = tiny_data["fold_id_array"]
    labels = (tiny_data["train"][TARGET_COL] == "1.0").to_numpy().astype(int)
    for row in range(60):
        prior = labels[fold_ids != fold_ids[row]].mean()
        assert oof[row] == pytest.approx(-row / 1000.0 + np.log(prior / (1 - prior)), abs=1e-9)
    priors = [labels[fold_ids != k].mean() for k in range(3)]
    log_prior = np.mean([np.log(p / (1 - p)) for p in priors])
    for row in range(20):
        assert test[row] == pytest.approx(-row / 1000.0 + log_prior, abs=1e-9)

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["sha256"]["oof.npy"] == hashlib.sha256((out / "oof.npy").read_bytes()).hexdigest()
    assert manifest["sha256"]["test.npy"] == hashlib.sha256((out / "test.npy").read_bytes()).hexdigest()
    assert manifest["auc"]["pooled"] == pytest.approx(roc_auc_score(labels, oof), abs=1e-12)
    assert len(manifest["per_fold"]) == 3
    for entry in manifest["per_fold"]:
        rows = fold_ids == entry["fold"]
        assert entry["auc"] == pytest.approx(roc_auc_score(labels[rows], oof[rows]), abs=1e-12)
    assert manifest["seconds"]["scoring"] > 0
    assert "Chris Deotte" in manifest["credit"]
    assert str(tmp_path) not in (out / "manifest.json").read_text()


@pytest.fixture
def run_records(tmp_path):
    """A replica summary and fits/f{k}/ folders, as the training scripts leave them."""
    fit_settings = {"epochs": 8, "batch_size": 32, "lr": 5e-5, "precision": "bf16", "seed": 42}
    replica_summary = tmp_path / "replica_summary.json"
    replica_summary.write_text(json.dumps({"seconds": 9.0, "settings": {"epochs": 60, "batch_size": 16, "model": "gpt2"}}))
    for fold in range(3):
        final = tmp_path / "fits" / f"f{fold}" / "final"
        final.mkdir(parents=True)
        (final / "model.safetensors").write_text(f"model {fold}")
        (final.parent / "train_summary.json").write_text(json.dumps(
            {"seconds": 5.0, "settings": {**fit_settings, "fold": fold}}))
    return ["--replica-summary", replica_summary, "--fold-summaries", tmp_path / "fits"]


def test_the_manifest_holds_the_settings_of_every_step(tiny_data, scored, run_records, tmp_path):
    predict(tiny_data, scored, tmp_path / "out", *run_records)
    settings = json.loads((tmp_path / "out" / "manifest.json").read_text())["settings"]
    assert settings["replica"] == {"epochs": 60, "batch_size": 16, "model": "gpt2"}
    assert settings["fit"] == {"epochs": 8, "batch_size": 32, "lr": 5e-5, "precision": "bf16", "seed": 42}
    assert settings["scoring"]["oof"] == {"orders": 3, "chunk_rows": 7, "precision": "fp32", "seed": 42, "max_rows": 0}
    assert settings["scoring"]["test"] == settings["scoring"]["oof"]


def test_folds_trained_with_different_settings_are_refused(tiny_data, scored, run_records, tmp_path):
    path = tmp_path / "fits" / "f2" / "train_summary.json"
    path.write_text(json.dumps({"seconds": 5.0, "settings": {"epochs": 7, "fold": 2}}))
    result = predict(tiny_data, scored, tmp_path / "out", *run_records, check=False)
    assert result.returncode != 0 and "settings differ" in result.stderr


def test_scores_made_by_another_fold_model_are_refused(tiny_data, scored, run_records, tmp_path):
    path = scored / "f1_test" / "settings.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "model_sha256": fake_model_hash(2)}))
    result = predict(tiny_data, scored, tmp_path / "out", *run_records, check=False)
    assert result.returncode != 0 and "another model" in result.stderr


def test_compare_oof_reads_the_other_model_on_exactly_the_scored_rows(tiny_data, scored, tmp_path):
    fold_ids = tiny_data["fold_id_array"]
    labels = (tiny_data["train"][TARGET_COL] == "1.0").to_numpy().astype(int)
    other = np.random.default_rng(3).normal(size=60)
    np.save(tmp_path / "other.npy", other)
    (scored / "f0_oof" / "chunk_00000.npz").unlink()   # fold 0 is only partly scored, as in an early read
    predict(tiny_data, scored, tmp_path / "out", "--allow-partial", "--compare-oof", tmp_path / "other.npy")
    oof = np.load(tmp_path / "out" / "oof.npy")
    per_fold = json.loads((tmp_path / "out" / "manifest.json").read_text())["per_fold"]
    assert per_fold[0]["rows"] == 13
    for entry in per_fold:
        rows = np.flatnonzero((fold_ids == entry["fold"]) & np.isfinite(oof))
        assert entry["compare_auc_same_rows"] == pytest.approx(roc_auc_score(labels[rows], other[rows]), abs=1e-12)


def test_compare_oof_of_the_wrong_length_is_refused(tiny_data, scored, tmp_path):
    np.save(tmp_path / "other.npy", np.zeros(59))
    result = predict(tiny_data, scored, tmp_path / "out", "--compare-oof", tmp_path / "other.npy", check=False)
    assert result.returncode != 0 and "does not match" in result.stderr


def test_a_score_from_a_model_that_trained_on_the_row_is_refused(tiny_data, scored, tmp_path):
    fold_ids = tiny_data["fold_id_array"]
    path = scored / "f0_oof" / "chunk_00000.npz"
    chunk = dict(np.load(path))
    chunk["rows"][0] = np.flatnonzero(fold_ids == 1)[0]
    np.savez(path, **chunk)
    result = predict(tiny_data, scored, tmp_path / "out", check=False)
    assert result.returncode != 0 and "trained on" in result.stderr


def test_missing_rows_are_refused_unless_it_is_a_smoke(tiny_data, scored, tmp_path):
    (scored / "f1_oof" / "chunk_00000.npz").unlink()
    result = predict(tiny_data, scored, tmp_path / "out", check=False)
    assert result.returncode != 0 and "no score" in result.stderr
    predict(tiny_data, scored, tmp_path / "out", "--allow-partial")
    oof = np.load(tmp_path / "out" / "oof.npy")
    assert np.isnan(oof).sum() == 7
    assert expit(oof[np.isfinite(oof)]).max() < 1


def test_an_early_read_works_from_one_fold_with_nothing_else_scored(tiny_data, scored, tmp_path):
    import shutil

    for name in ("f1_oof", "f2_oof", "f0_test", "f1_test", "f2_test"):
        shutil.rmtree(scored / name)
    assert predict(tiny_data, scored, tmp_path / "strict", check=False).returncode != 0
    predict(tiny_data, scored, tmp_path / "early", "--allow-partial")
    oof = np.load(tmp_path / "early" / "oof.npy")
    assert np.isfinite(oof).sum() == (tiny_data["fold_id_array"] == 0).sum()
    assert np.isfinite(oof[tiny_data["fold_id_array"] == 0]).all()
    assert np.isnan(np.load(tmp_path / "early" / "test.npy")).all()
    manifest = json.loads((tmp_path / "early" / "manifest.json").read_text())
    assert [f["fold"] for f in manifest["per_fold"]] == [0]
