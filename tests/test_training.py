"""Training details: target first, fresh orders, loss on all tokens, schedule, and no held-out data in the model."""
import numpy as np
import torch
from safetensors.torch import load_file

from conftest import ID_COL, TARGET_COL, run_script, tiny_table
from tabtext import RowTokens, feature_columns, label_prefix_ids
from train_classifier import make_batch, schedule


def test_batch_is_target_first_with_fresh_orders_and_loss_on_all_tokens(tokenizer):
    table = tiny_table(8, seed=0)
    columns = feature_columns(table, ID_COL, TARGET_COL)
    tokens = RowTokens(table, columns, tokenizer)
    prefix_true, prefix_false = label_prefix_ids(tokenizer, TARGET_COL)
    labels = np.array([1, 0, 1, 0, 0, 0, 1, 0], dtype=np.int8)
    rows = np.arange(8)
    ids, mask, targets = make_batch(tokens, rows, labels, prefix_true, prefix_false, 3, 42, 0, "cpu")
    for i, row in enumerate(rows):
        length = int(mask[i].sum())
        assert ids[i, :len(prefix_true)].tolist() == (prefix_true if labels[row] else prefix_false)
        assert ids[i, length - 1].item() == tokenizer.eos_token_id
        assert targets[i, :length].tolist() == ids[i, :length].tolist()   # every real token is in the loss
        assert (targets[i, length:] == -100).all() and (mask[i, length:] == 0).all()   # padding on the right, no loss
        assert mask[i, :length].all()
    other_step, _, _ = make_batch(tokens, rows, labels, prefix_true, prefix_false, 4, 42, 0, "cpu")
    assert not torch.equal(ids, other_step)   # a fresh order every step


def test_schedule_equals_the_hugging_face_linear_schedule():
    from transformers import get_linear_schedule_with_warmup

    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=5e-5)
    hugging_face = get_linear_schedule_with_warmup(optimizer, 0, 20)
    for step in range(20):
        assert abs(optimizer.param_groups[0]["lr"] - schedule(step, 20, 5e-5)) < 1e-15
        optimizer.step()
        hugging_face.step()
    assert schedule(20, 20, 5e-5) == 0.0


def fit(train_csv, fold_ids, replica, out, *extra):
    return run_script("train_classifier.py", "--train-csv", train_csv, "--fold-ids", fold_ids, "--fold", 0, "--id-col", ID_COL,
                      "--target-col", TARGET_COL, "--replica", replica, "--out", out, "--epochs", 2, "--batch-size", 4,
                      "--log-every-steps", 5, "--save-every-steps", 3, "--precision", "fp32", *extra)


def weights(directory):
    return load_file(directory / "final" / "model.safetensors")


def test_no_held_out_row_or_label_reaches_the_model(tiny_data, tiny_model_dir, tmp_path):
    """Scramble only the held-out rows, labels (even to garbage) and features: the weights must not move at all."""
    fit(tiny_data["train_csv"], tiny_data["fold_ids"], tiny_model_dir, tmp_path / "plain")

    changed = tiny_data["train"].copy()
    held_out = tiny_data["fold_id_array"] == 0
    changed.loc[held_out, TARGET_COL] = "not a label"
    changed.loc[held_out, "amount_usd"] = "999.99"
    changed.loc[held_out, "merchant_category"] = "Never Seen"
    changed_csv = tmp_path / "changed.csv"
    changed.to_csv(changed_csv, index=False)
    fit(changed_csv, tiny_data["fold_ids"], tiny_model_dir, tmp_path / "changed")

    first, second = weights(tmp_path / "plain"), weights(tmp_path / "changed")
    assert first.keys() == second.keys()
    assert all(torch.equal(first[k], second[k]) for k in first)


def test_fit_rows_do_move_the_model(tiny_data, tiny_model_dir, tmp_path):
    """The check above would pass on a model that learns nothing, so a changed fit row must change the weights."""
    fit(tiny_data["train_csv"], tiny_data["fold_ids"], tiny_model_dir, tmp_path / "plain")
    changed = tiny_data["train"].copy()
    first_fit_row = np.flatnonzero(tiny_data["fold_id_array"] != 0)[0]
    changed.loc[first_fit_row, "amount_usd"] = "999.99"
    changed_csv = tmp_path / "changed.csv"
    changed.to_csv(changed_csv, index=False)
    fit(changed_csv, tiny_data["fold_ids"], tiny_model_dir, tmp_path / "changed")
    first, second = weights(tmp_path / "plain"), weights(tmp_path / "changed")
    assert not all(torch.equal(first[k], second[k]) for k in first)


def test_a_saved_model_loads_with_tied_weights_and_no_local_path(tiny_data, tiny_model_dir, tmp_path):
    from transformers import AutoModelForCausalLM

    fit(tiny_data["train_csv"], tiny_data["fold_ids"], tiny_model_dir, tmp_path / "run")
    model = AutoModelForCausalLM.from_pretrained(tmp_path / "run" / "final")
    assert model.lm_head.weight is model.transformer.wte.weight
    for path in (tmp_path / "run" / "final").glob("*.json"):
        assert str(tmp_path) not in path.read_text()
    assert str(tmp_path) not in (tmp_path / "run" / "train_summary.json").read_text()
    assert str(tmp_path) not in (tmp_path / "run" / "progress.jsonl").read_text()
    assert not (tmp_path / "run" / "resume.pt").exists()
