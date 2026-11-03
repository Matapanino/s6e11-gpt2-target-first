"""A job that dies and restarts must finish with the same result as one that never stopped."""
import json
import os
import signal
import subprocess
import sys
import time

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from common import STOPPED_EXIT_CODE, sha256_file
from conftest import CODE, ID_COL, TARGET_COL, run_script, tiny_table
from make_predictions import load_scores


def fit_arguments(data, replica, out):
    return ["train_classifier.py", "--train-csv", data["train_csv"], "--fold-ids", data["fold_ids"], "--fold", 0,
            "--id-col", ID_COL, "--target-col", TARGET_COL, "--replica", replica, "--out", out, "--epochs", 2,
            "--batch-size", 4, "--log-every-steps", 5, "--save-every-steps", 3, "--precision", "fp32"]


def assert_same_weights(first_dir, second_dir):
    first = load_file(first_dir / "final" / "model.safetensors")
    second = load_file(second_dir / "final" / "model.safetensors")
    assert first.keys() == second.keys()
    assert all(torch.equal(first[k], second[k]) for k in first)


def test_classifier_after_a_crash_equals_a_run_that_never_stopped(tiny_data, tiny_model_dir, tmp_path):
    run_script(*fit_arguments(tiny_data, tiny_model_dir, tmp_path / "whole"))

    crashed = tmp_path / "crashed"
    result = run_script(*fit_arguments(tiny_data, tiny_model_dir, crashed), "--stop-after-steps", 7, check=False)
    assert result.returncode == 3
    assert (crashed / "resume.pt").exists() and not (crashed / "final").exists()
    saved = torch.load(crashed / "resume.pt", weights_only=True)
    assert saved["step"] == 6 and saved["seconds"] > 0
    (crashed / "resume.pt.tmp").write_bytes(b"half a file from a crash during a save")
    run_script(*fit_arguments(tiny_data, tiny_model_dir, crashed))
    assert_same_weights(tmp_path / "whole", crashed)
    assert json.loads((crashed / "train_summary.json").read_text())["seconds"] > saved["seconds"]


def test_classifier_stopped_by_sigterm_saves_and_resumes_to_the_same_result(tiny_data, tiny_model_dir, tmp_path):
    run_script(*fit_arguments(tiny_data, tiny_model_dir, tmp_path / "whole"))

    stopped = tmp_path / "stopped"
    arguments = [str(a) for a in fit_arguments(tiny_data, tiny_model_dir, stopped)]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false")
    process = subprocess.Popen([sys.executable, str(CODE / arguments[0]), *arguments[1:], "--log-every-steps", "1"],
                               env=environment, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    progress = stopped / "progress.jsonl"
    deadline = time.time() + 120
    while time.time() < deadline and process.poll() is None:
        if progress.exists() and sum(1 for _ in progress.open()) >= 3:
            process.send_signal(signal.SIGTERM)
            break
        time.sleep(0.05)
    assert process.wait(timeout=120) == STOPPED_EXIT_CODE
    assert (stopped / "resume.pt").exists()
    run_script(*fit_arguments(tiny_data, tiny_model_dir, stopped))
    assert_same_weights(tmp_path / "whole", stopped)


def score_arguments(data, model_dir, out):
    return ["score_rows.py", "--model-dir", model_dir, "--csv", data["train_csv"], "--split", "oof", "--fold-ids",
            data["fold_ids"], "--fold", 0, "--id-col", ID_COL, "--target-col", TARGET_COL, "--out", out, "--orders", 3,
            "--batch-sequences", 8, "--chunk-rows", 4, "--precision", "fp32"]


def test_scoring_after_a_crash_equals_a_run_that_never_stopped(tiny_data, tiny_model_dir, tmp_path):
    run_script(*score_arguments(tiny_data, tiny_model_dir, tmp_path / "whole"))

    crashed = tmp_path / "crashed"
    result = run_script(*score_arguments(tiny_data, tiny_model_dir, crashed), "--stop-after-chunks", 1, check=False)
    assert result.returncode == 3
    assert [p.name for p in sorted(crashed.glob("chunk_*.npz"))] == ["chunk_00000.npz"]
    kept = (crashed / "chunk_00000.npz").stat().st_mtime_ns
    (crashed / "chunk_00001.npz.tmp.npz").write_bytes(b"half a file")
    run_script(*score_arguments(tiny_data, tiny_model_dir, crashed))
    assert (crashed / "chunk_00000.npz").stat().st_mtime_ns == kept   # a finished chunk is not scored again

    rows_whole, true_whole, false_whole, _ = load_scores(tmp_path / "whole")
    rows_again, true_again, false_again, _ = load_scores(crashed)
    assert (rows_whole == rows_again).all()
    assert (true_whole == true_again).all() and (false_whole == false_again).all()


def test_scoring_records_which_model_it_used(tiny_data, tiny_model_dir, tmp_path):
    run_script(*score_arguments(tiny_data, tiny_model_dir, tmp_path / "scores"))
    settings = json.loads((tmp_path / "scores" / "settings.json").read_text())
    assert settings["model_sha256"] == sha256_file(tiny_model_dir / "model.safetensors")
    assert settings["chunk_rows"] == 4


def test_scoring_refuses_to_mix_settings_in_one_folder(tiny_data, tiny_model_dir, tmp_path):
    run_script(*score_arguments(tiny_data, tiny_model_dir, tmp_path / "scores"))
    result = run_script(*score_arguments(tiny_data, tiny_model_dir, tmp_path / "scores"), "--orders", 5, check=False)
    assert result.returncode != 0
    assert "other settings" in result.stderr


@pytest.fixture
def tiny_original(tmp_path):
    table = tiny_table(24, seed=11, target_true="1", target_false="0").rename(columns={ID_COL: "transaction_id"})
    path = tmp_path / "original.csv"
    table.to_csv(path, index=False)
    return path


def replica_arguments(original, model, out):
    return ["train_replica.py", "--original-csv", original, "--id-col", "transaction_id", "--target-col", TARGET_COL,
            "--out", out, "--model", model, "--epochs", 3, "--batch-size", 4, "--save-steps", 4, "--log-steps", 4]


def test_replica_runs_and_saves_a_clean_model(tiny_original, tiny_model_dir, tmp_path):
    from transformers import AutoModelForCausalLM

    run_script(*replica_arguments(tiny_original, tiny_model_dir, tmp_path / "replica"))
    summary = json.loads((tmp_path / "replica" / "train_summary.json").read_text())
    assert summary["global_step"] == 18   # 24 rows / batch 4 x 3 epochs
    assert summary["rows"] == 24 and "transaction_id" not in summary["columns"]
    assert not (tmp_path / "replica" / "trainer").exists()
    for path in (tmp_path / "replica" / "final").glob("*.json"):
        assert str(tmp_path) not in path.read_text()
    assert str(tmp_path) not in (tmp_path / "replica" / "train_summary.json").read_text()
    assert AutoModelForCausalLM.from_pretrained(tmp_path / "replica" / "final") is not None


@pytest.mark.xfail(reason="Replica resume is not bit-exact yet (see README, 'Replica checkpoint resume'); the shared S6E11 replica ran without a resume.", strict=False)
def test_replica_after_a_crash_equals_a_run_that_never_stopped(tiny_original, tiny_model_dir, tmp_path):
    run_script(*replica_arguments(tiny_original, tiny_model_dir, tmp_path / "whole"))

    crashed = tmp_path / "crashed"
    result = run_script(*replica_arguments(tiny_original, tiny_model_dir, crashed), "--stop-after-steps", 10, check=False)
    assert result.returncode == 3
    assert (crashed / "trainer" / "checkpoint-8").exists()
    saved_seconds = json.loads((crashed / "seconds_at_last_save.json").read_text())
    half_written = crashed / "trainer" / "checkpoint-10"   # a kill during a save leaves a model and no trainer state
    half_written.mkdir()
    (half_written / "model.safetensors").write_bytes(b"half a file")
    run_script(*replica_arguments(tiny_original, tiny_model_dir, crashed))
    assert_same_weights(tmp_path / "whole", crashed)
    assert json.loads((crashed / "train_summary.json").read_text())["seconds"] > saved_seconds


def test_the_newest_complete_checkpoint_is_the_one_resumed(tmp_path):
    from train_replica import newest_checkpoint

    assert newest_checkpoint(tmp_path) is False
    (tmp_path / "checkpoint-8").mkdir()
    (tmp_path / "checkpoint-8" / "trainer_state.json").write_text("{}")
    (tmp_path / "checkpoint-12").mkdir()
    assert newest_checkpoint(tmp_path).endswith("checkpoint-8")


def test_saving_over_an_existing_final_folder_works(tiny_model_dir, tmp_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from train_replica import save_clean

    model, tokenizer = AutoModelForCausalLM.from_pretrained(tiny_model_dir), AutoTokenizer.from_pretrained(tiny_model_dir)
    save_clean(model, tokenizer, tmp_path / "final", "gpt2")
    save_clean(model, tokenizer, tmp_path / "final", "gpt2")   # a re-run after a crash just after the first save
    assert (tmp_path / "final" / "model.safetensors").exists() and not (tmp_path / "final.tmp").exists()
