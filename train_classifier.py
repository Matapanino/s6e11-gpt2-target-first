"""Step 2, once per fold: continue training the replica on the fold's fit rows, target first.

Settings from the post: 8 epochs, AdamW, lr 5e-5 with linear decay to 0, batch 32, bf16, loss on all tokens, the
target first and then the features in a fresh random order every step.
Settings the post leaves open, chosen here (README lists them all): weight decay 0, gradient clipping 1.0 and dropout
0.1 (the same as the replica's Hugging Face and GPT-2 defaults), padding on the right with pad positions out of the
loss, so that every row starts at position 0, as it does when scored.

Input:  train.csv, fold_ids.npy, the replica directory.
Does:   keeps only the rows whose fold is not the held-out fold, before anything is tokenized or any label is read.
        Held-out features and labels never reach the model.
Output: OUT/final/ (the model), OUT/progress.jsonl, OUT/train_summary.json. progress.jsonl shows the loss per label,
        because P(features | True) trains on far fewer rows than P(features | False).

Run:    python train_classifier.py --train-csv train.csv --fold-ids fold_ids.npy --fold 0 --id-col id
            --target-col is_fraud --replica REPLICA/final --out OUT
Resume: run the same command again. The batches, the column orders and the dropout depend only on the step number,
        so a resumed run gives the same weights as one that never stopped.
"""
import argparse
import functools
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

from common import STOPPED_EXIT_CODE, StopFlag, fit_and_held_out, log_progress, named_hash, print_summary, write_json
from tabtext import (LABEL_SPELLINGS, RowTokens, anywhere_orders, check_anywhere_matches_tokenizer, check_matches_tokenizer,
                     feature_columns, label_field_ids, label_prefix_ids, label_text, random_orders, read_csv_as_strings)
from train_replica import base_model_name, save_clean


def fit_table(train_csv, fold_ids, fold, target_col, max_rows=0, label_spelling="truefalse"):
    """The fit rows of one fold and their labels. Held-out rows are dropped before any label is read."""
    df = read_csv_as_strings(train_csv)
    if len(df) != len(fold_ids):
        raise ValueError("fold_ids does not match train.csv")
    fit_rows, _ = fit_and_held_out(fold_ids, fold)
    if max_rows:
        fit_rows = fit_rows[:max_rows]
    table = df.iloc[fit_rows].reset_index(drop=True)
    labels = (table[target_col].map(lambda value: label_text(value, label_spelling)) ==
              LABEL_SPELLINGS[label_spelling][0]).to_numpy(dtype=np.int8)
    return table, labels


def schedule(step, total_steps, base_lr):
    """Linear decay to 0 with no warmup, the same formula as the Hugging Face linear schedule."""
    return base_lr * max(0.0, (total_steps - step) / max(1, total_steps))


@functools.lru_cache(maxsize=1)
def epoch_shuffle(n_rows, seed, fold, epoch):
    return np.random.default_rng([seed, fold, epoch]).permutation(n_rows)


def batch_rows(step, n_rows, batch_size, seed, fold):
    """Row positions of a training step. Each epoch has its own shuffle, fixed by (seed, fold, epoch)."""
    epoch, position = divmod(step, math.ceil(n_rows / batch_size))
    return epoch_shuffle(n_rows, seed, fold, epoch)[position * batch_size:(position + 1) * batch_size]


def make_batch(tokens, rows, labels, prefix_true, prefix_false, step, seed, fold, device,
               layout="first", field_ids=None, return_spans=False, label_spelling="truefalse"):
    """Target first, then a fresh random column order per row. Padding goes on the right, out of the loss."""
    if layout == "first":
        orders = random_orders([seed, fold, step, 1], len(rows), len(tokens.columns))
        sequences = [(prefix_true if labels[row] else prefix_false) + tokens.body_ids(int(row), orders[i])
                     for i, row in enumerate(rows)]
    else:
        orders = anywhere_orders(seed, fold, step, rows, len(tokens.columns))
        built = [tokens.with_label(int(row), orders[i], LABEL_SPELLINGS[label_spelling][not bool(labels[row])], field_ids,
                                   include_field_span=True) for i, row in enumerate(rows)]
        sequences = [item[0] for item in built]
        field_spans = [item[2] for item in built]
    longest = max(len(s) for s in sequences)
    input_ids = torch.full((len(sequences), longest), tokens.eos_id, dtype=torch.long)
    attention = torch.zeros((len(sequences), longest), dtype=torch.long)
    targets = torch.full((len(sequences), longest), -100, dtype=torch.long)
    for i, sequence in enumerate(sequences):
        values = torch.tensor(sequence, dtype=torch.long)
        input_ids[i, :len(sequence)] = values
        attention[i, :len(sequence)] = 1
        targets[i, :len(sequence)] = values
    result = input_ids.to(device), attention.to(device), targets.to(device)
    return (*result, field_spans) if return_spans else result


def training_step(model, optimizer, input_ids, attention, targets, n_prefix, use_bf16, max_grad_norm,
                  label_field_spans=None):
    """One step. Returns the loss and the per-row summed feature-token loss, the quantity scoring sums."""
    with torch.autocast(device_type=input_ids.device.type, dtype=torch.bfloat16, enabled=use_bf16):
        logits = model(input_ids=input_ids, attention_mask=attention, use_cache=False).logits
    per_token = torch.nn.functional.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]), targets[:, 1:].reshape(-1),
        ignore_index=-100, reduction="none").view(input_ids.shape[0], -1)
    loss = per_token.sum() / (targets[:, 1:] != -100).sum()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    feature_nats = per_token[:, n_prefix - 1:].sum(1).detach()
    if label_field_spans is not None:
        feature_nats = feature_nats.clone()
        for i, (start, end) in enumerate(label_field_spans):
            feature_nats[i] -= per_token[i, max(1, start)-1:end-1].sum().detach()
    return float(loss.detach()), feature_nats.cpu().numpy()


def save_checkpoint(path, model, optimizer, step, seconds, epochs=8, layout="first", label_spelling="truefalse",
                    total_steps=None):
    temporary = Path(str(path) + ".tmp")
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "seconds": seconds}
    if (epochs, layout, label_spelling) != (8, "first", "truefalse"):
        state.update(epochs=epochs, layout=layout, total_steps=total_steps)
        if label_spelling != "truefalse":
            state["label_text"] = label_spelling
    torch.save(state, temporary)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--fold-ids", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--id-col", required=True)
    parser.add_argument("--target-col", required=True)
    parser.add_argument("--replica", required=True, help="directory from train_replica.py, or a model name")
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, help="the post uses 8")
    parser.add_argument("--layout", choices=["first", "anywhere"], default="first")
    parser.add_argument("--label-text", choices=["truefalse", "01"], default="truefalse")
    parser.add_argument("--batch-size", type=int, default=32, help="the post uses 32")
    parser.add_argument("--lr", type=float, default=5e-5, help="the post uses 5e-5, linear decay to 0")
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16", help="the post uses bf16")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="chosen: the Hugging Face default")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="chosen: the Hugging Face default")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every-steps", type=int, default=5000)
    parser.add_argument("--log-every-steps", type=int, default=200)
    parser.add_argument("--max-rows", type=int, default=0, help="for smoke runs only")
    parser.add_argument("--stop-after-steps", type=int, default=0, help="for tests: stop hard after this step")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.layout == "anywhere" and args.epochs is None:
        parser.error("--epochs is required for --layout anywhere")
    if args.epochs is None:
        args.epochs = 8
    if args.epochs < 1:
        parser.error("--epochs must be positive")

    out = Path(args.out)
    if (out / "train_summary.json").exists():
        print_summary("train_classifier", status="already done", fold=args.fold)
        return
    out.mkdir(parents=True, exist_ok=True)
    progress_path = out / "progress.jsonl"
    checkpoint_path = out / "resume.pt"
    stop = StopFlag()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.replica)
    table, labels = fit_table(args.train_csv, np.load(args.fold_ids), args.fold, args.target_col,
                              args.max_rows, args.label_text)
    columns = feature_columns(table, args.id_col, args.target_col)
    tokens = RowTokens(table, columns, tokenizer)
    check_matches_tokenizer(table, tokens, tokenizer, args.target_col, spelling=args.label_text)
    if args.layout == "anywhere":
        check_anywhere_matches_tokenizer(table, tokens, tokenizer, args.target_col, spelling=args.label_text)
    del table
    prefix_true, prefix_false = label_prefix_ids(tokenizer, args.target_col, args.label_text)
    n_prefix = len(prefix_true)
    field_ids = label_field_ids(tokenizer, args.target_col, args.label_text) if args.layout == "anywhere" else None
    if args.layout == "anywhere":
        n_prefix = 1

    model = AutoModelForCausalLM.from_pretrained(args.replica).to(args.device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
                                  weight_decay=args.weight_decay)
    step_start, seconds_before = 0, 0.0
    steps_per_epoch = math.ceil(tokens.n_rows / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=args.device, weights_only=True)
        expected = {"epochs": args.epochs, "layout": args.layout, "label_text": args.label_text,
                    "total_steps": total_steps}
        actual = {"epochs": state.get("epochs", 8), "layout": state.get("layout", "first"),
                  "label_text": state.get("label_text", "truefalse"),
                  "total_steps": state.get("total_steps", total_steps)}
        if actual != expected:
            raise ValueError(f"checkpoint settings differ: {actual} != {expected}")
        if state["step"] > total_steps:
            raise ValueError("checkpoint step exceeds total_steps")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step_start, seconds_before = state["step"], state["seconds"]
        del state

    use_bf16 = args.precision == "bf16"
    log_progress(progress_path, event="start", fold=args.fold, fit_rows=tokens.n_rows, positives=int(labels.sum()),
                 total_steps=total_steps, resumed_at_step=step_start, median_tokens=float(np.median(tokens.lengths)))
    started = time.time()
    window_start, window_rows = time.time(), 0
    window = {"loss": [], "nats_true": [], "nats_false": []}
    for step in range(step_start, total_steps):
        rows = batch_rows(step, tokens.n_rows, args.batch_size, args.seed, args.fold)
        batch = make_batch(tokens, rows, labels, prefix_true, prefix_false, step, args.seed,
                           args.fold, args.device, args.layout, field_ids, args.layout == "anywhere", args.label_text)
        input_ids, attention, targets = batch[:3]
        for group in optimizer.param_groups:
            group["lr"] = schedule(step, total_steps, args.lr)
        torch.manual_seed(int(np.random.default_rng([args.seed, args.fold, step, 2]).integers(2**62)))
        loss, feature_nats = training_step(model, optimizer, input_ids, attention, targets, n_prefix, use_bf16,
                                           args.max_grad_norm, batch[3] if args.layout == "anywhere" else None)
        done = step + 1
        window["loss"].append(loss)
        window["nats_true"].extend(feature_nats[labels[rows] == 1].tolist())
        window["nats_false"].extend(feature_nats[labels[rows] == 0].tolist())
        window_rows += len(rows)
        if done % args.log_every_steps == 0 or done == total_steps:
            rows_per_second = window_rows / (time.time() - window_start)
            log_progress(progress_path, step=done, total_steps=total_steps, epoch=done / steps_per_epoch,
                         loss=float(np.mean(window["loss"])), lr=schedule(step, total_steps, args.lr),
                         feature_nats_true_rows=float(np.mean(window["nats_true"])) if window["nats_true"] else None,
                         feature_nats_false_rows=float(np.mean(window["nats_false"])) if window["nats_false"] else None,
                         true_rows_in_window=len(window["nats_true"]), rows_per_second=rows_per_second,
                         eta_hours=(total_steps - done) * args.batch_size / rows_per_second / 3600,
                         peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0)
            window_start, window_rows = time.time(), 0
            window = {"loss": [], "nats_true": [], "nats_false": []}
        if done % args.save_every_steps == 0:
            save_checkpoint(checkpoint_path, model, optimizer, done, seconds_before + time.time() - started,
                            args.epochs, args.layout, args.label_text, total_steps)
        if args.stop_after_steps and done >= args.stop_after_steps:
            os._exit(3)  # a hard stop with no saving, the same as a crash
        if stop.requested:
            save_checkpoint(checkpoint_path, model, optimizer, done, seconds_before + time.time() - started,
                            args.epochs, args.layout, args.label_text, total_steps)
            log_progress(progress_path, event="stopped", step=done)
            sys.exit(STOPPED_EXIT_CODE)

    save_clean(model, tokenizer, out / "final", base_model_name(args.replica), args.label_text)
    seconds = seconds_before + time.time() - started
    settings = {k: v for k, v in vars(args).items()
                if k not in ("train_csv", "fold_ids", "replica", "out", "device", "stop_after_steps")}
    if args.layout == "first":
        settings.pop("layout")
    if args.label_text == "truefalse":
        settings.pop("label_text")
    write_json(out / "train_summary.json", {
        "fold": args.fold, "fit_rows": tokens.n_rows, "positives": int(labels.sum()), "total_steps": total_steps,
        "seconds": seconds, "settings": settings, "train_csv": named_hash(args.train_csv),
        "fold_ids": named_hash(args.fold_ids)})
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    print_summary("train_classifier", status="done", fold=args.fold, seconds=seconds, steps=total_steps)


if __name__ == "__main__":
    main()
