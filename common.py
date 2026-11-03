"""Small helpers shared by the scripts: atomic writes, progress logs, hashes, the stop signal, folds.

Anything written for keeps must be free of machine-specific text, so manifests store file names and hashes,
never full paths.
"""
import hashlib
import json
import os
import signal
import time
from pathlib import Path

import numpy as np

STOPPED_EXIT_CODE = 75  # a job that saved its state and stopped on request; the queue restarts it later


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def named_hash(path):
    """File name and hash, for manifests. No directory, so nothing machine-specific is kept."""
    return {"file": Path(path).name, "sha256": sha256_file(path)}


def write_json(path, payload):
    """Write JSON through a temporary file so that a crash never leaves half a file."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    os.replace(temporary, path)


def save_npy(path, array):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp.npy")
    np.save(temporary, array)
    os.replace(temporary, path)


def log_progress(path, **fields):
    """Append one JSON line to a progress file and print it. The time is UTC."""
    line = json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **fields}, sort_keys=True)
    with open(path, "a") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def print_summary(script, **fields):
    """Every script ends by printing one JSON line, so a reader or an agent can pick up the result."""
    print(json.dumps({"script": script, **fields}, sort_keys=True), flush=True)


class StopFlag:
    """Set by SIGTERM. Long jobs check it between steps, save their state and exit with STOPPED_EXIT_CODE."""

    def __init__(self):
        self.requested = False
        signal.signal(signal.SIGTERM, self._request)

    def _request(self, *_):
        self.requested = True


def fit_and_held_out(fold_ids, fold):
    """Positions of the rows a fold model trains on, and of the rows it must never see."""
    fold_ids = np.asarray(fold_ids)
    return np.flatnonzero(fold_ids != fold), np.flatnonzero(fold_ids == fold)


def fold_prior(labels, fold_ids, fold):
    """The training rate of the fit rows of a fold. Uses no held-out label."""
    fit_rows, _ = fit_and_held_out(fold_ids, fold)
    return float(np.mean(np.asarray(labels)[fit_rows]))
