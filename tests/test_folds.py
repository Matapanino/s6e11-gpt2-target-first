"""Folds: same as scikit-learn's, and a fold's fit rows exclude that fold."""
import numpy as np
from sklearn.model_selection import StratifiedKFold

from common import fit_and_held_out, fold_prior
from conftest import ID_COL, TARGET_COL
from make_folds import make_fold_ids
from train_classifier import batch_rows, fit_table


def test_fold_ids_match_stratified_kfold():
    labels = np.random.default_rng(0).integers(0, 2, 500)
    fold_ids = make_fold_ids(labels, 5, 42)
    for fold, (_, held_out) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(500), labels)):
        assert (np.flatnonzero(fold_ids == fold) == np.sort(held_out)).all()


def test_fit_and_held_out_split_the_rows_without_overlap():
    fold_ids = make_fold_ids(np.random.default_rng(1).integers(0, 2, 300), 5, 42)
    for fold in range(5):
        fit, held = fit_and_held_out(fold_ids, fold)
        assert not set(fit) & set(held)
        assert len(fit) + len(held) == 300
        assert (fold_ids[fit] != fold).all() and (fold_ids[held] == fold).all()


def test_fit_table_holds_only_fit_rows(tiny_data):
    for fold in range(3):
        table, labels = fit_table(tiny_data["train_csv"], tiny_data["fold_id_array"], fold, TARGET_COL)
        expected = np.flatnonzero(tiny_data["fold_id_array"] != fold)
        assert table[ID_COL].astype(int).tolist() == expected.tolist()
        assert len(labels) == len(expected)


def test_every_training_batch_stays_inside_the_fit_table_and_covers_it_once():
    n_rows, batch_size = 103, 8
    steps_per_epoch = int(np.ceil(n_rows / batch_size))
    for epoch in range(3):
        seen = np.concatenate([batch_rows(epoch * steps_per_epoch + s, n_rows, batch_size, 42, 0)
                               for s in range(steps_per_epoch)])
        assert sorted(seen.tolist()) == list(range(n_rows))
    first = batch_rows(0, n_rows, batch_size, 42, 0)
    next_epoch_first = batch_rows(steps_per_epoch, n_rows, batch_size, 42, 0)
    assert first.tolist() != next_epoch_first.tolist()


def test_fold_prior_reads_only_fit_rows():
    labels = np.array([1, 0, 0, 0, 1, 1, 0, 1])
    fold_ids = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    assert fold_prior(labels, fold_ids, 0) == 0.75
    assert fold_prior(labels, fold_ids, 1) == 0.25
    flipped = labels.copy()
    flipped[fold_ids == 0] = 1 - flipped[fold_ids == 0]
    assert fold_prior(flipped, fold_ids, 0) == 0.75
