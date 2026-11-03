"""Step 1, once: fine-tune GPT-2 on the original dataset with be_great, as the post does.

The post's call:  GReaT(llm="gpt2", batch_size=16, epochs=60).fit(original_df)

Input:  the original dataset CSV.
Does:   writes every value as its exact CSV string and the target as True or False, drops the id column, then calls
        be_great. be_great puts all columns in a fresh random order on every fetch, the target included, adds no end
        token, and uses the Hugging Face Trainer defaults (AdamW, lr 5e-5, linear decay, no warmup, no weight decay).
        This script adds logging, rolling checkpoints, the final save, and one small change: the row shuffle of an
        epoch and each column shuffle are seeded by (seed, epoch) and (seed, epoch, row) instead of drawn from a
        running stream, so a resumed run matches an unbroken one. They are just as random.
Output: OUT/final/ (the model and tokenizer for step 2), OUT/progress.jsonl, OUT/train_summary.json.

Run:    python train_replica.py --original-csv original.csv --id-col transaction_id --target-col is_fraud --out OUT
Resume: run the same command again. It continues from the newest checkpoint in OUT/trainer/ and gives the same
        weights as a run that never stopped.
"""
import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

from common import STOPPED_EXIT_CODE, StopFlag, log_progress, named_hash, print_summary, write_json
from tabtext import LABEL_SPELLINGS, label_text, read_csv_as_strings, value_text


def original_frame(csv_path, id_col, target_col, max_rows=0, label_spelling="truefalse"):
    """The table be_great receives. All values are exact strings, so its str(value).strip() changes nothing."""
    df = read_csv_as_strings(csv_path)
    if id_col in df.columns:
        df = df.drop(columns=[id_col])
    if max_rows:
        df = df.iloc[:max_rows].reset_index(drop=True)
    for column in df.columns:
        df[column] = df[column].map((lambda value: label_text(value, label_spelling))
                                    if column == target_col else value_text)
    return df


class EpochSeededSampler:
    """A random row order for each epoch, fixed by (seed, epoch).

    be_great's data loader draws the row order from the running torch stream, so after a restart the rest of that
    epoch would use a different order.
    """

    def __init__(self, n_rows, seed, dataset_class):
        self.n_rows, self.seed, self.dataset_class = n_rows, seed, dataset_class

    def __len__(self):
        return self.n_rows

    def __iter__(self):
        return iter(np.random.default_rng([self.seed, self.dataset_class.epoch]).permutation(self.n_rows).tolist())


def repeatable_order_dataset(base_class, seed):
    """be_great's dataset, with each column shuffle seeded by (seed, epoch, row).

    The Trainer restores its random state only after the data loader has fetched the next batches, so with be_great's
    running stream a resumed run gets different column orders from there on.
    """

    class RepeatableOrderDataset(base_class):
        epoch = 0

        def _getitem(self, key, *more, **kwargs):
            random.seed(f"{seed}/{RepeatableOrderDataset.epoch}/{key}")
            return super()._getitem(key, *more, **kwargs)

    return RepeatableOrderDataset


def allow_trainer_rng_file():
    """Let torch below 2.6 read the Trainer's own RNG file on resume.

    Transformers sets this same allowlist itself, but only from torch 2.6. Without it a resume stops with
    "Weights only load failed". The file is one this script wrote.
    """
    import numpy as np
    import torch

    core = np._core if hasattr(np, "_core") else np.core
    torch.serialization.add_safe_globals([core.multiarray._reconstruct, np.ndarray, np.dtype, type(np.dtype(np.uint32))])


def newest_checkpoint(trainer_dir):
    """The newest complete checkpoint. The Trainer writes trainer_state.json last, so a kill during a save leaves none."""
    found = sorted((p for p in Path(trainer_dir).glob("checkpoint-*") if (p / "trainer_state.json").exists()),
                   key=lambda p: int(p.name.split("-")[1]))
    return str(found[-1]) if found else False


def base_model_name(model_dir_or_name):
    """The Hugging Face name a saved model started from, or the name itself when it is not a directory."""
    config = Path(model_dir_or_name) / "config.json"
    return json.loads(config.read_text()).get("_name_or_path", "unknown") if config.exists() else model_dir_or_name


def save_clean(model, tokenizer, directory, source_name, label_spelling="truefalse"):
    """Save a model for later steps, with the base model's name in place of the local path Hugging Face records."""
    directory = Path(directory)
    temporary = directory.with_name(directory.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    model.save_pretrained(temporary)
    tokenizer.save_pretrained(temporary)
    config = temporary / "config.json"
    settings = json.loads(config.read_text())
    settings["_name_or_path"] = source_name
    config.write_text(json.dumps(settings, indent=2) + "\n")
    if label_spelling != "truefalse":
        (temporary / "label_text.json").write_text(json.dumps({"label_text": label_spelling}) + "\n")
    if directory.exists():  # a re-run after a crash just after the last save
        shutil.rmtree(directory)
    os.replace(temporary, directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--original-csv", required=True)
    parser.add_argument("--id-col", required=True)
    parser.add_argument("--target-col", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="gpt2", help="the post uses gpt2")
    parser.add_argument("--label-text", choices=["truefalse", "01"], default="truefalse")
    parser.add_argument("--epochs", type=float, default=60, help="the post uses 60")
    parser.add_argument("--batch-size", type=int, default=16, help="the post uses 16")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32", help="the post's call leaves it at fp32")
    parser.add_argument("--workers", type=int, default=0, help="be_great's default is 0")
    parser.add_argument("--save-steps", type=int, default=5000)
    parser.add_argument("--log-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42, help="the Hugging Face default")
    parser.add_argument("--max-rows", type=int, default=0, help="for smoke runs only")
    parser.add_argument("--stop-after-steps", type=int, default=0, help="for tests: stop hard after this step")
    args = parser.parse_args()

    out = Path(args.out)
    if (out / "train_summary.json").exists():
        print_summary("train_replica", status="already done")
        return
    out.mkdir(parents=True, exist_ok=True)
    progress_path = out / "progress.jsonl"

    import be_great.great as great_module
    import torch
    from be_great import GReaT
    from transformers import TrainerCallback

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    started = time.time()
    stop = StopFlag()
    seconds_path = out / "seconds_at_last_save.json"
    seconds_before = json.loads(seconds_path.read_text()) if seconds_path.exists() else 0.0

    class ProgressAndStop(TrainerCallback):
        def __init__(self):
            self.last_time, self.last_step = time.time(), None

        def on_log(self, train_args, state, control, logs=None, **_):
            if not logs or "loss" not in logs:
                return
            now = time.time()
            steps = state.global_step - (self.last_step if self.last_step is not None else state.global_step)
            rows_per_second = steps * args.batch_size / (now - self.last_time) if steps > 0 else None
            self.last_time, self.last_step = now, state.global_step
            eta_hours = (state.max_steps - state.global_step) * args.batch_size / rows_per_second / 3600 \
                if rows_per_second else None
            log_progress(progress_path, step=state.global_step, total_steps=state.max_steps, epoch=state.epoch,
                         loss=logs["loss"], lr=logs.get("learning_rate"), rows_per_second=rows_per_second,
                         eta_hours=eta_hours)

        def on_save(self, train_args, state, control, **_):
            write_json(seconds_path, seconds_before + time.time() - started)

        def on_epoch_begin(self, train_args, state, control, **_):
            dataset_class.epoch = state.global_step // math.ceil(len(frame) / args.batch_size)

        def on_step_end(self, train_args, state, control, **_):
            if args.stop_after_steps and state.global_step >= args.stop_after_steps:
                os._exit(3)  # a hard stop with no saving, the same as a crash
            if stop.requested:
                control.should_save = True
                control.should_training_stop = True
            return control

    class QuietTrainer(great_module.GReaTTrainer):
        """The be_great trainer with its progress bar replaced by plain JSON lines in a log file."""

        def __init__(self, *trainer_args, **trainer_kwargs):
            super().__init__(*trainer_args, **trainer_kwargs)
            self.remove_callback(great_module.ChugProgressCallback)
            self.add_callback(ProgressAndStop())

        def _get_train_sampler(self):
            return EpochSeededSampler(len(self.train_dataset), args.seed, dataset_class)

    great_module.GReaTTrainer = QuietTrainer
    dataset_class = repeatable_order_dataset(great_module.GReaTDataset, args.seed)
    great_module.GReaTDataset = dataset_class
    allow_trainer_rng_file()

    frame = original_frame(args.original_csv, args.id_col, args.target_col, args.max_rows, args.label_text)
    trainer_dir = out / "trainer"
    resume_from = newest_checkpoint(trainer_dir)
    great = GReaT(llm=args.model, experiment_dir=str(trainer_dir), epochs=args.epochs, batch_size=args.batch_size,
                  seed=args.seed, save_strategy="steps", save_steps=args.save_steps, save_total_limit=1,
                  logging_steps=args.log_steps, logging_first_step=True, dataloader_num_workers=args.workers,
                  bf16=args.precision == "bf16", disable_tqdm=True)
    log_progress(progress_path, event="start", rows=len(frame), columns=len(frame.columns), resumed=bool(resume_from),
                 positives=int((frame[args.target_col] == LABEL_SPELLINGS[args.label_text][0]).sum()))
    trainer = great.fit(frame, conditional_col=args.target_col, resume_from_checkpoint=resume_from)
    if stop.requested:
        log_progress(progress_path, event="stopped", step=trainer.state.global_step)
        sys.exit(STOPPED_EXIT_CODE)

    save_clean(great.model, great.tokenizer, out / "final", base_model_name(args.model), args.label_text)
    seconds = seconds_before + time.time() - started
    summary = {
        "rows": len(frame),
        "positives": int((frame[args.target_col] == LABEL_SPELLINGS[args.label_text][0]).sum()),
        "columns": list(frame.columns),
        "seconds": seconds,
        "global_step": trainer.state.global_step,
        "last_loss": next((r["loss"] for r in reversed(trainer.state.log_history) if "loss" in r), None),
        "settings": {k: v for k, v in vars(args).items() if k not in ("original_csv", "out", "stop_after_steps")},
        "original_csv": named_hash(args.original_csv),
    }
    summary["settings"]["model"] = base_model_name(args.model)
    if args.label_text == "truefalse":
        summary["settings"].pop("label_text")
    write_json(out / "train_summary.json", summary)
    shutil.rmtree(trainer_dir, ignore_errors=True)  # the rolling checkpoints are no longer needed
    seconds_path.unlink(missing_ok=True)
    print_summary("train_replica", status="done", seconds=seconds, global_step=summary["global_step"],
                  last_loss=summary["last_loss"])


if __name__ == "__main__":
    main()
