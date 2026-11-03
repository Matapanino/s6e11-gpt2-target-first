"""Step 3: score rows twice, log P(features | True) and log P(features | False), per feature order.

Input:  one fold model; train.csv with --split oof (the fold's held-out rows) or test.csv with --split test.
Does:   drops the target column on reading, so a label cannot reach the model. Each row gets its own random feature
        orders, and the same order is used under both labels. The sum covers every token after the label prefix,
        the end token included, as in the post's code. Rows are batched by token length, so no padding is involved
        and a batched score equals scoring the row alone.
Output: OUT/chunk_00000.npz and so on, each with rows (positions in the CSV), log_p_true and log_p_false (rows x
        orders), seconds. OUT/settings.json, with the hash of the model that made the scores. make_predictions.py turns
        these into oof.npy and test.npy.

Run:    python score_rows.py --model-dir FOLD/final --csv train.csv --split oof --fold-ids fold_ids.npy --fold 0
            --id-col id --target-col is_fraud --out OUT
Resume: run the same command again. Finished chunks are kept. Each chunk's orders depend only on the seed, the fold,
        the split and the chunk number, so a resumed run gives the same scores.
"""
import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

from common import STOPPED_EXIT_CODE, StopFlag, log_progress, named_hash, print_summary, sha256_file, write_json
from tabtext import LABEL_SPELLINGS, RowTokens, check_anywhere_matches_tokenizer, check_matches_tokenizer, feature_columns, \
    label_field_ids, label_prefix_ids, random_orders, read_csv_as_strings

SPLIT_CODE = {"oof": 0, "test": 1}


@torch.inference_mode()
def sequence_log_probs(model, input_ids, n_prefix, use_bf16, lm_head_tokens=4096):
    """Summed log-probability of the tokens after the prefix, for same-length sequences with no padding.

    This is the post's log_p() in batches: position i predicts token i + 1, and the sum starts at the first token
    after the prefix. The last-layer projection and the softmax always run in fp32, in pieces, because full-vocabulary
    logits for a whole batch would not fit on a GPU.
    """
    with torch.autocast(device_type=input_ids.device.type, dtype=torch.bfloat16, enabled=use_bf16):
        hidden = model.transformer(input_ids=input_ids, use_cache=False).last_hidden_state
    hidden = hidden[:, n_prefix - 1:-1].float()
    targets = input_ids[:, n_prefix:]
    flat_hidden = hidden.reshape(-1, hidden.shape[-1])
    flat_targets = targets.reshape(-1)
    token_log_probs = torch.empty(flat_targets.shape, dtype=torch.float64, device=input_ids.device)
    for start in range(0, flat_targets.shape[0], lm_head_tokens):
        stop = start + lm_head_tokens
        log_probs = model.lm_head(flat_hidden[start:stop]).log_softmax(-1)
        token_log_probs[start:stop] = log_probs.gather(1, flat_targets[start:stop, None])[:, 0].double()
    return token_log_probs.view(targets.shape).sum(1)


@torch.inference_mode()
def sequence_span_log_probs(model, input_ids, spans, use_bf16, lm_head_tokens=4096):
    """Log probability of each sequence's selected token span, excluding token zero."""
    with torch.autocast(device_type=input_ids.device.type, dtype=torch.bfloat16, enabled=use_bf16):
        hidden = model.transformer(input_ids=input_ids, use_cache=False).last_hidden_state
    hidden = hidden[:, :-1].float()
    targets = input_ids[:, 1:]
    flat_hidden = hidden.reshape(-1, hidden.shape[-1])
    flat_targets = targets.reshape(-1)
    token_log_probs = torch.empty(flat_targets.shape, dtype=torch.float64, device=input_ids.device)
    for start in range(0, flat_targets.shape[0], lm_head_tokens):
        stop = start + lm_head_tokens
        log_probs = model.lm_head(flat_hidden[start:stop]).log_softmax(-1)
        token_log_probs[start:stop] = log_probs.gather(1, flat_targets[start:stop, None])[:, 0].double()
    values = token_log_probs.view(targets.shape)
    return torch.stack([values[i, max(1, a)-1:b-1].sum() for i, (a, b) in enumerate(spans)])


def score_chunk(model, tokens, rows, orders, prefix_true, prefix_false, batch_sequences, use_bf16, device,
                should_stop=lambda: False, layout="first", field_ids=None, positions=None,
                label_spelling="truefalse"):
    """Scores for one chunk: (log_p_true, log_p_false), each rows x orders. Returns None if asked to stop."""
    n_rows, n_orders = orders.shape[:2]
    log_p = np.zeros((n_rows, n_orders, 2), dtype=np.float64)
    if layout != "first":
        buckets = {}
        for i, row in enumerate(rows):
            for j in range(n_orders):
                for k, label in enumerate(LABEL_SPELLINGS[label_spelling]):
                    order = orders[i, j]
                    if layout == "anywhere":
                        order = np.insert(order, int(positions[i, j]), len(tokens.columns))
                    else:
                        order = np.append(order, len(tokens.columns))
                    ids, span = tokens.with_label(int(row), order, label, field_ids)
                    buckets.setdefault(len(ids), []).append((i, j, k, ids, span))
        for length, entries in buckets.items():
            for start in range(0, len(entries), batch_sequences):
                if should_stop():
                    return None
                batch = entries[start:start + batch_sequences]
                ids = torch.tensor([e[3] for e in batch], dtype=torch.long, device=device)
                spans = [(1, length) if layout == "anywhere" else e[4] for e in batch]
                scores = sequence_span_log_probs(model, ids, spans, use_bf16).cpu().numpy()
                for entry, value in zip(batch, scores):
                    log_p[entry[0], entry[1], entry[2]] = value
        return log_p[:, :, 0], log_p[:, :, 1]
    lengths = tokens.lengths[rows]
    bodies_per_batch = max(1, batch_sequences // 2)  # each body runs once under each label
    n_prefix = len(prefix_true)
    for length in np.unique(lengths):
        same_length = np.flatnonzero(lengths == length)
        row_index = np.repeat(same_length, n_orders)
        order_index = np.tile(np.arange(n_orders), len(same_length))
        for start in range(0, len(row_index), bodies_per_batch):
            if should_stop():
                return None
            r, o = row_index[start:start + bodies_per_batch], order_index[start:start + bodies_per_batch]
            bodies = np.array([tokens.body_ids(int(rows[i]), orders[i, j]) for i, j in zip(r, o)], dtype=np.int64)
            ids_true = np.concatenate([np.tile(prefix_true, (len(r), 1)), bodies], axis=1)
            ids_false = np.concatenate([np.tile(prefix_false, (len(r), 1)), bodies], axis=1)
            input_ids = torch.from_numpy(np.concatenate([ids_true, ids_false])).to(device)
            scores = sequence_log_probs(model, input_ids, n_prefix, use_bf16).cpu().numpy()
            log_p[r, o, 0] = scores[:len(r)]
            log_p[r, o, 1] = scores[len(r):]
    return log_p[:, :, 0], log_p[:, :, 1]


def chunk_orders(seed, fold, split, chunk_index, n_rows, n_orders, n_columns):
    """Random feature orders for a chunk, shape rows x orders x columns. A fresh set per row."""
    flat = random_orders([seed, fold, SPLIT_CODE[split], chunk_index], n_rows * n_orders, n_columns)
    return flat.reshape(n_rows, n_orders, n_columns)


def chunk_positions(seed, fold, split, chunk_index, n_rows, n_orders, n_columns):
    rng = np.random.default_rng([seed, fold, SPLIT_CODE[split], chunk_index, 1])
    return rng.integers(0, n_columns + 1, size=(n_rows, n_orders))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--split", choices=["oof", "test"], required=True)
    parser.add_argument("--fold-ids", required=True, help="needed for --split oof; recorded for --split test")
    parser.add_argument("--fold", type=int, required=True, help="the fold whose model this is")
    parser.add_argument("--id-col", required=True)
    parser.add_argument("--target-col", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--orders", type=int, default=30, help="the post uses 30")
    parser.add_argument("--layout", choices=["first", "anywhere", "last"], default="first")
    parser.add_argument("--label-text", choices=["truefalse", "01"], default="truefalse")
    parser.add_argument("--batch-sequences", type=int, default=256, help="sets memory use, not speed; scores do not depend on it")
    parser.add_argument("--chunk-rows", type=int, default=2000,
                        help="the most work a stop or a crash can lose; 2000 rows at 30 orders is about 8 minutes on a fast GPU")
    parser.add_argument("--chunk-start", type=int, default=0, help="first chunk index for a distributed scoring worker")
    parser.add_argument("--chunk-stop", type=int, default=-1, help="exclusive last chunk index; -1 means all chunks")
    parser.add_argument("--precision", choices=["fp32", "tf32", "bf16"], default="fp32",
                        help="the post's code uses fp32; the other two are faster but add rounding noise to the scores")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=0, help="for smoke runs only")
    parser.add_argument("--stop-after-chunks", type=int, default=0, help="for tests: stop hard after this many chunks")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model_label_path = Path(args.model_dir) / "label_text.json"
    model_label = json.loads(model_label_path.read_text())["label_text"] if model_label_path.exists() else "truefalse"
    if model_label != args.label_text:
        raise ValueError(f"model label spelling {model_label} differs from --label-text {args.label_text}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    settings = {"split": args.split, "fold": args.fold, "orders": args.orders, "chunk_rows": args.chunk_rows,
                "precision": args.precision, "seed": args.seed, "max_rows": args.max_rows,
                "model_sha256": sha256_file(Path(args.model_dir) / "model.safetensors")}
    if args.label_text != "truefalse":
        settings["label_text"] = args.label_text
    if args.layout != "first":
        settings["layout"] = args.layout
    settings_path = out / "settings.json"
    if settings_path.exists():
        existing = json.loads(settings_path.read_text())
        if existing.get("label_text", "truefalse") == "truefalse":
            existing.pop("label_text", None)
        if existing != settings:
            raise ValueError("this output folder holds chunks made with other settings")
    write_json(settings_path, settings)

    stop = StopFlag()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    torch.backends.cuda.matmul.allow_tf32 = args.precision == "tf32"
    from transformers import AutoModelForCausalLM, AutoTokenizer

    table = read_csv_as_strings(args.csv)
    if args.target_col in table.columns:
        table = table.drop(columns=[args.target_col])  # no label from here on
    if args.split == "oof":
        fold_ids = np.load(args.fold_ids)
        if len(fold_ids) != len(table):
            raise ValueError("fold_ids does not match the CSV")
        selected = np.flatnonzero(fold_ids == args.fold)
    else:
        selected = np.arange(len(table))
    if args.max_rows:
        selected = selected[:args.max_rows]

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    columns = feature_columns(table, args.id_col, args.target_col)
    tokens = RowTokens(table, columns, tokenizer)
    check_matches_tokenizer(table, tokens, tokenizer, args.target_col, spelling=args.label_text)
    if args.layout != "first":
        check_anywhere_matches_tokenizer(table, tokens, tokenizer, args.target_col, spelling=args.label_text)
    del table
    prefix_true, prefix_false = label_prefix_ids(tokenizer, args.target_col, args.label_text)
    field_ids = label_field_ids(tokenizer, args.target_col, args.label_text) if args.layout != "first" else None
    model = AutoModelForCausalLM.from_pretrained(args.model_dir).to(args.device).eval()

    n_chunks = int(np.ceil(len(selected) / args.chunk_rows))
    if not (0 <= args.chunk_start <= n_chunks) or (args.chunk_stop != -1 and not args.chunk_start <= args.chunk_stop <= n_chunks):
        raise ValueError("invalid chunk range")
    progress_path = out / "progress.jsonl"
    log_progress(progress_path, event="start", split=args.split, fold=args.fold, rows=len(selected), chunks=n_chunks,
                 orders=args.orders, median_tokens=float(np.median(tokens.lengths[selected])))
    heartbeat_stop = threading.Event()

    def heartbeat():
        while not heartbeat_stop.wait(60):
            log_progress(progress_path, event="heartbeat", split=args.split, fold=args.fold)

    threading.Thread(target=heartbeat, name="score-progress-heartbeat", daemon=True).start()
    started = time.time()
    scored_sequences = 0
    for chunk_index in range(args.chunk_start, n_chunks if args.chunk_stop == -1 else args.chunk_stop):
        path = out / f"chunk_{chunk_index:05d}.npz"
        if path.exists():
            continue
        rows = selected[chunk_index * args.chunk_rows:(chunk_index + 1) * args.chunk_rows]
        orders = chunk_orders(args.seed, args.fold, args.split, chunk_index, len(rows), args.orders, len(columns))
        positions = (chunk_positions(args.seed, args.fold, args.split, chunk_index, len(rows), args.orders,
                                     len(columns)) if args.layout == "anywhere" else None)
        chunk_started = time.time()
        result = score_chunk(model, tokens, rows, orders, prefix_true, prefix_false, args.batch_sequences,
                             args.precision == "bf16", args.device, lambda: stop.requested,
                             args.layout, field_ids, positions, args.label_text)
        if result is None:
            log_progress(progress_path, event="stopped", chunk=chunk_index)
            sys.exit(STOPPED_EXIT_CODE)
        seconds = time.time() - chunk_started
        temporary = path.with_name(path.name + ".tmp.npz")
        np.savez(temporary, rows=rows, log_p_true=result[0], log_p_false=result[1], seconds=seconds)
        os.replace(temporary, path)
        n_sequences = 2 * len(rows) * args.orders
        scored_sequences += n_sequences
        rate = scored_sequences / (time.time() - started)
        remaining = sum(1 for k in range(chunk_index + 1, n_chunks) if not (out / f"chunk_{k:05d}.npz").exists())
        log_progress(progress_path, chunk=chunk_index + 1, chunks=n_chunks, sequences_per_second=rate,
                     chunk_sequences_per_second=n_sequences / seconds,
                     eta_hours=remaining * 2 * args.chunk_rows * args.orders / rate / 3600,
                     peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0)
        if args.stop_after_chunks and chunk_index + 1 >= args.stop_after_chunks:
            os._exit(3)  # a hard stop, the same as a crash

    finished = sorted(out.glob("chunk_?????.npz"))
    if len(finished) == n_chunks:
        write_json(out / "done.json", {"chunks": n_chunks, "csv": named_hash(args.csv), "settings": settings})
    heartbeat_stop.set()
    print_summary("score_rows", status="done", split=args.split, fold=args.fold, chunks=len(finished),
                  sequences_this_run=scored_sequences, seconds=time.time() - started)


if __name__ == "__main__":
    main()
