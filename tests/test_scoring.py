"""Scoring: a model whose probabilities are known by hand, the post's own code, batches, and no label on reading."""
import numpy as np
import torch
from scipy.special import logsumexp
from transformers import AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel

from conftest import ID_COL, TARGET_COL, run_script, tiny_table
from make_predictions import load_scores
from score_rows import score_chunk, sequence_log_probs
from tabtext import RowTokens, feature_columns, features_text, label_prefix_ids, random_orders


def post_log_p(model, tok, text, n_prefix):
    """The post's log_p(), copied as written."""
    ids = tok(text, return_tensors="pt").input_ids
    lp = model(ids).logits[0, :-1].log_softmax(-1)
    tok_lp = lp.gather(-1, ids[0, 1:, None])[:, 0]            # log-prob of every next token
    return tok_lp[n_prefix - 1:].sum().item()                  # sum over the feature tokens only


def constant_distribution_model(seed=0):
    """A GPT-2 that gives every position the same known next-token distribution, whatever it reads.

    With all weights zero the blocks output zero, and the final layer norm then returns its own bias. So the logits
    are the head times that bias, for every position, and any token's log-probability can be read off by hand.
    """
    torch.manual_seed(seed)
    config = GPT2Config(n_layer=1, n_head=1, n_embd=4, n_positions=256, tie_word_embeddings=False)
    model = GPT2LMHeadModel(config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.transformer.ln_f.bias.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        logits = torch.randn(config.vocab_size) * 3
        model.lm_head.weight[:, 0] = logits
    return model, logits.numpy().astype(np.float64)


def test_known_model_scored_by_hand_equals_the_batched_path_and_the_posts_code(tokenizer):
    model, logits = constant_distribution_model()
    by_token = logits - logsumexp(logits)
    table = tiny_table(12, seed=8)
    columns = feature_columns(table, ID_COL, TARGET_COL)
    tokens = RowTokens(table, columns, tokenizer)
    prefix_true, prefix_false = label_prefix_ids(tokenizer, TARGET_COL)
    n_prefix = len(prefix_true)
    rows = np.arange(len(table))
    orders = random_orders([0], len(rows) * 3, len(columns)).reshape(len(rows), 3, len(columns))
    log_p_true, log_p_false = score_chunk(model, tokens, rows, orders, prefix_true, prefix_false, 8, False, "cpu")

    values = table[columns].to_numpy()
    for r in rows:
        for o in range(3):
            body = tokens.body_ids(int(r), orders[r, o])
            by_hand = float(sum(by_token[t] for t in body))  # every token after the prefix, the end token included
            assert body[-1] == tokenizer.eos_token_id
            assert abs(log_p_true[r, o] - by_hand) < 1e-2
            assert abs(log_p_false[r, o] - by_hand) < 1e-2  # this model ignores the label, so the two must agree
            text = " " + features_text(columns, values[r], orders[r, o]) + tokenizer.eos_token
            with torch.no_grad():
                posts = post_log_p(model, tokenizer, f"{TARGET_COL} is True," + text, n_prefix)
            assert abs(posts - by_hand) < 1e-2


def test_the_prefix_tokens_are_out_of_the_sum(tokenizer):
    """Off-by-one guard: the sum must start at the first token after the prefix and no earlier or later."""
    model, logits = constant_distribution_model(seed=3)
    by_token = logits - logsumexp(logits)
    sequence = [10, 20, 30, 40, 50, 60]
    input_ids = torch.tensor([sequence])
    for n_prefix in (1, 2, 3, 5):
        score = sequence_log_probs(model, input_ids, n_prefix, False).item()
        assert abs(score - by_token[sequence[n_prefix:]].sum()) < 1e-3


def test_batched_path_equals_the_posts_code_on_a_real_tiny_gpt2(tokenizer, tiny_model_dir):
    model = AutoModelForCausalLM.from_pretrained(tiny_model_dir).eval()
    table = tiny_table(15, seed=9)
    table.loc[0, "merchant_category"] = "A Much Longer Merchant Category Name Here"  # so lengths differ
    columns = feature_columns(table, ID_COL, TARGET_COL)
    tokens = RowTokens(table, columns, tokenizer)
    prefix_true, prefix_false = label_prefix_ids(tokenizer, TARGET_COL)
    rows = np.arange(len(table))
    assert len(np.unique(tokens.lengths)) > 1
    orders = random_orders([1], len(rows) * 4, len(columns)).reshape(len(rows), 4, len(columns))
    log_p_true, log_p_false = score_chunk(model, tokens, rows, orders, prefix_true, prefix_false, 6, False, "cpu")
    values = table[columns].to_numpy()
    with torch.no_grad():
        for r in rows:
            for o in range(4):
                text = " " + features_text(columns, values[r], orders[r, o]) + tokenizer.eos_token
                n_true = len(tokenizer(f"{TARGET_COL} is True,").input_ids)
                n_false = len(tokenizer(f"{TARGET_COL} is False,").input_ids)
                lt = post_log_p(model, tokenizer, f"{TARGET_COL} is True," + text, n_true)
                lf = post_log_p(model, tokenizer, f"{TARGET_COL} is False," + text, n_false)
                assert abs(log_p_true[r, o] - lt) < 1e-3
                assert abs(log_p_false[r, o] - lf) < 1e-3
    assert not np.allclose(log_p_true, log_p_false)   # a random GPT-2 does read the label


def test_batch_size_does_not_change_the_scores(tokenizer, tiny_model_dir):
    model = AutoModelForCausalLM.from_pretrained(tiny_model_dir).eval()
    table = tiny_table(10, seed=10)
    columns = feature_columns(table, ID_COL, TARGET_COL)
    tokens = RowTokens(table, columns, tokenizer)
    prefix_true, prefix_false = label_prefix_ids(tokenizer, TARGET_COL)
    rows = np.arange(10)
    orders = random_orders([2], 30, len(columns)).reshape(10, 3, len(columns))
    small = score_chunk(model, tokens, rows, orders, prefix_true, prefix_false, 2, False, "cpu")
    large = score_chunk(model, tokens, rows, orders, prefix_true, prefix_false, 64, False, "cpu")
    assert np.allclose(small[0], large[0], atol=1e-3) and np.allclose(small[1], large[1], atol=1e-3)


def score(data, model_dir, out, csv=None, split="oof", *extra):
    return run_script("score_rows.py", "--model-dir", model_dir, "--csv", csv or data["train_csv"], "--split", split,
                      "--fold-ids", data["fold_ids"], "--fold", 0, "--id-col", ID_COL, "--target-col", TARGET_COL,
                      "--out", out, "--orders", 3, "--batch-sequences", 8, "--chunk-rows", 5, "--precision", "fp32", *extra)


def test_oof_scores_only_the_folds_rows_and_cannot_see_a_label(tiny_data, tiny_model_dir, tmp_path):
    """Flipping or removing every label in the CSV must leave the scores untouched."""
    score(tiny_data, tiny_model_dir, tmp_path / "plain")
    rows, true_plain, false_plain, _ = load_scores(tmp_path / "plain")
    assert sorted(rows.tolist()) == np.flatnonzero(tiny_data["fold_id_array"] == 0).tolist()

    flipped = tiny_data["train"].copy()
    flipped[TARGET_COL] = "garbage"
    flipped_csv = tmp_path / "flipped.csv"
    flipped.to_csv(flipped_csv, index=False)
    score(tiny_data, tiny_model_dir, tmp_path / "flipped", flipped_csv)
    removed_csv = tmp_path / "removed.csv"
    tiny_data["train"].drop(columns=[TARGET_COL]).to_csv(removed_csv, index=False)
    score(tiny_data, tiny_model_dir, tmp_path / "removed", removed_csv)
    for name in ("flipped", "removed"):
        other_rows, true_other, false_other, _ = load_scores(tmp_path / name)
        assert (other_rows == rows).all()
        assert (true_other == true_plain).all() and (false_other == false_plain).all()


def test_test_split_scores_every_row_in_file_order(tiny_data, tiny_model_dir, tmp_path):
    score(tiny_data, tiny_model_dir, tmp_path / "test", tiny_data["test_csv"], "test")
    rows, log_p_true, _, _ = load_scores(tmp_path / "test")
    assert rows.tolist() == list(range(len(tiny_data["test"])))
    assert log_p_true.shape == (len(tiny_data["test"]), 3)


def test_tf32_setting_changes_nothing_on_cpu_and_is_kept_apart_in_settings(tiny_data, tiny_model_dir, tmp_path):
    score(tiny_data, tiny_model_dir, tmp_path / "plain")
    run_script("score_rows.py", "--model-dir", tiny_model_dir, "--csv", tiny_data["train_csv"], "--split", "oof",
               "--fold-ids", tiny_data["fold_ids"], "--fold", 0, "--id-col", ID_COL, "--target-col", TARGET_COL,
               "--out", tmp_path / "tf32", "--orders", 3, "--batch-sequences", 8, "--chunk-rows", 5, "--precision", "tf32")
    assert (load_scores(tmp_path / "plain")[1] == load_scores(tmp_path / "tf32")[1]).all()
