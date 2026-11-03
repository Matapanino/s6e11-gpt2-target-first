"""Shared fixtures: a tiny CSV, a tiny GPT-2 with GPT-2's real tokenizer, and a helper to run a script as a job.

All tests run on CPU in seconds, with no GPU. The tokenizer comes from public Hugging Face with no token.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE))

ID_COL, TARGET_COL = "id", "is_fraud"


def tiny_table(n_rows=60, seed=0, target_true="1.0", target_false="0.0"):
    """A table of strings like the competition's: exact number formats, words with spaces, flags, some positives."""
    rng = np.random.default_rng(seed)
    amounts = ["61.16", "3233.0", "6.36", "644.9", "20.53", "1046.88"]
    categories = ["Online Retail", "Groceries", "3D Secure", "In-App"]
    flags = ["True", "False"]
    table = pd.DataFrame({
        ID_COL: [str(i) for i in range(n_rows)],
        "amount_usd": rng.choice(amounts, n_rows),
        "customer_age": rng.integers(18, 80, n_rows).astype(str),
        "merchant_category": rng.choice(categories, n_rows),
        "used_vpn": rng.choice(flags, n_rows),
        TARGET_COL: np.where(rng.random(n_rows) < 0.25, target_true, target_false),
    })
    return table


@pytest.fixture(scope="session")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory, tokenizer):
    """A two-layer GPT-2 with random weights and the real vocabulary, saved like a replica."""
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=2, n_embd=16, n_positions=256))
    directory = tmp_path_factory.mktemp("tiny_gpt2")
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    return directory


@pytest.fixture
def tiny_data(tmp_path):
    """train.csv, test.csv and fold_ids.npy on disk."""
    from make_folds import make_fold_ids
    from tabtext import label_text

    train = tiny_table(60, seed=1)
    test = tiny_table(20, seed=2).drop(columns=[TARGET_COL])
    train_csv, test_csv, fold_path = tmp_path / "train.csv", tmp_path / "test.csv", tmp_path / "fold_ids.npy"
    train.to_csv(train_csv, index=False)
    test.to_csv(test_csv, index=False)
    fold_ids = make_fold_ids((train[TARGET_COL].map(label_text) == "True").astype(int).to_numpy(), n_splits=3)
    np.save(fold_path, fold_ids)
    return {"train_csv": train_csv, "test_csv": test_csv, "fold_ids": fold_path, "train": train, "test": test,
            "fold_id_array": fold_ids}


def run_script(script, *arguments, check=True):
    """Run a script the way the queue does: a separate process, no GPU, two threads."""
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false")
    result = subprocess.run([sys.executable, str(CODE / script), *map(str, arguments)], capture_output=True, text=True,
                            env=environment)
    if check and result.returncode != 0:
        raise AssertionError(f"{script} exited {result.returncode}\n{result.stdout[-2000:]}\n{result.stderr[-3000:]}")
    return result
