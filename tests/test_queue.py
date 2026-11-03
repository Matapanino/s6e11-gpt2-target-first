"""The queue runner, with small real processes in place of training jobs and fake GPU and memory readings."""
import argparse
import json
import shutil
import subprocess
import sys
import time

import pytest

from conftest import run_script
from run_queue import Queue, command_line, foreign_processes
import make_queue

FINISH_ON_SIGTERM = ("import signal,sys,time\n"
                     "signal.signal(signal.SIGTERM, lambda *_: sys.exit(75))\n"
                     "time.sleep(60)\n")


def job(name, code="pass", after=(), cpu=False):
    return {"job": name, "argv": [sys.executable, "-c", code], "after": list(after), "cpu": cpu}


def make_runner(tmp_path, jobs, foreign=None, memory=100.0, disk=1.0, gpus="0,1", per_gpu=1):
    queue_path = tmp_path / "queue.json"
    queue_path.write_text(json.dumps(jobs))
    args = argparse.Namespace(queue=str(queue_path), state=str(tmp_path / "state"), gpus=gpus, per_gpu=per_gpu,
                              own_marker="ours", disk_folder=str(tmp_path), max_disk_gib=25, min_memory_gib=30,
                              floor_memory_gib=12, poll_seconds=0.01)
    foreign = foreign if foreign is not None else {}
    return Queue(args, list_foreign=lambda gpu: foreign.get(gpu, []), read_memory=lambda: memory,
                 read_disk=lambda folder: disk)


def drain(runner, seconds=20):
    deadline = time.time() + seconds
    while runner.tick():
        assert time.time() < deadline, "the queue did not finish"
        time.sleep(0.02)


def test_dependencies_run_in_order_and_all_finish(tmp_path):
    marker = tmp_path / "order.txt"
    append = lambda text: f"open({str(marker)!r},'a').write({text!r})"
    runner = make_runner(tmp_path, [job("b", append("b"), after=["a"]), job("a", append("a"))])
    drain(runner)
    assert marker.read_text() == "ab"
    assert runner.is_done("a") and runner.is_done("b")
    assert (tmp_path / "state" / "pids.txt").read_text().count("\n") == 2


def test_a_restart_skips_jobs_that_are_done(tmp_path):
    marker = tmp_path / "runs.txt"
    code = f"open({str(marker)!r},'a').write('x')"
    drain(make_runner(tmp_path, [job("a", code)]))
    drain(make_runner(tmp_path, [job("a", code)]))
    assert marker.read_text() == "x"


def test_a_failing_job_is_retried_then_blocks_only_its_dependents(tmp_path):
    marker = tmp_path / "runs.txt"
    fail = f"open({str(marker)!r},'a').write('x'); raise SystemExit(1)"
    runner = make_runner(tmp_path, [job("bad", fail), job("after_bad", after=["bad"]), job("other")])
    drain(runner)
    assert marker.read_text() == "xx"
    assert runner.failed == {"bad"}
    assert runner.is_done("other") and not runner.is_done("after_bad")


def test_a_chain_of_jobs_behind_a_failure_ends_the_run(tmp_path):
    runner = make_runner(tmp_path, [job("bad", "raise SystemExit(1)"), job("next", after=["bad"]), job("last", after=["next"])])
    drain(runner)
    assert runner.failed == {"bad"} and not runner.is_done("last")


def test_nothing_new_starts_on_a_gpu_another_user_is_on(tmp_path):
    runner = make_runner(tmp_path, [job("a", FINISH_ON_SIGTERM)], foreign={0: [999], 1: [998]})
    for _ in range(3):
        runner.tick()
    assert not runner.running
    runner.foreign_checks = {0: 0, 1: 0}
    runner.list_foreign = lambda gpu: [] if gpu == 1 else [999]
    runner.tick()
    assert runner.running["a"]["gpu"] == 1
    runner.stop_job("a", "end of test")
    runner.running["a"]["process"].wait()


def test_another_user_arriving_makes_our_job_yield_and_return_to_the_queue(tmp_path):
    foreign = {}
    runner = make_runner(tmp_path, [job("a", FINISH_ON_SIGTERM)], foreign=foreign, gpus="0")
    runner.tick()
    time.sleep(0.5)  # let the job install its signal handler
    assert runner.running["a"]["gpu"] == 0
    foreign[0] = [999]
    runner.tick()
    assert "stopping" not in runner.running["a"]   # one sighting is not enough: it may be a passing process
    runner.tick()
    assert runner.running["a"]["stopping"]
    runner.running["a"]["process"].wait(timeout=10)
    foreign.clear()
    runner.tick()
    assert runner.attempts["a"] == 1 and "a" not in runner.failed   # a yield does not count as a failed attempt
    assert runner.running["a"]["gpu"] == 0   # started again, now that the GPU is free
    runner.stop_job("a", "end of test")
    runner.running["a"]["process"].wait(timeout=10)


def stop_all(runner):
    for entry in runner.running.values():
        entry["process"].terminate()
        entry["process"].wait(timeout=10)


def test_a_gpu_can_be_added_to_a_running_queue_with_the_gpu_list_file(tmp_path):
    runner = make_runner(tmp_path, [job(n, FINISH_ON_SIGTERM) for n in "ab"], gpus="0")
    runner.tick()
    assert set(runner.running) == {"a"}
    (tmp_path / "state" / "gpus.txt").write_text("0,1\n")
    runner.tick()
    assert runner.running["b"]["gpu"] == 1
    stop_all(runner)


def test_an_unreadable_gpu_list_file_changes_nothing(tmp_path):
    runner = make_runner(tmp_path, [job("a", FINISH_ON_SIGTERM)], gpus="0")
    (tmp_path / "state" / "gpus.txt").write_text("0,\n")
    runner.tick()
    assert runner.gpus == [0] and runner.running["a"]["gpu"] == 0
    stop_all(runner)


def test_a_job_still_yields_after_its_gpu_left_the_list(tmp_path):
    foreign = {}
    runner = make_runner(tmp_path, [job("a", FINISH_ON_SIGTERM)], foreign=foreign, gpus="0")
    runner.tick()
    time.sleep(0.5)
    (tmp_path / "state" / "gpus.txt").write_text("1\n")
    foreign[0] = [999]
    runner.tick()
    runner.tick()
    assert runner.running["a"]["stopping"]
    runner.running["a"]["process"].wait(timeout=10)


def test_an_error_from_nvidia_smi_starts_nothing_on_that_gpu_and_the_runner_survives(tmp_path):
    def list_foreign(gpu):
        if gpu == 0:
            raise ValueError("invalid literal for int(): 'Unable'")
        return []

    runner = make_runner(tmp_path, [job(n, FINISH_ON_SIGTERM) for n in "ab"])
    runner.list_foreign = list_foreign
    runner.tick()
    assert set(runner.running) == {"a"} and runner.running["a"]["gpu"] == 1
    stop_all(runner)


def test_a_second_runner_refuses_while_a_process_of_the_first_is_alive(tmp_path):
    from run_queue import live_own_processes

    state = tmp_path / "state"
    state.mkdir()
    queue_path = tmp_path / "queue.json"
    queue_path.write_text("[]")
    alive = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "ours"])
    (state / "pids.txt").write_text(f"{alive.pid} a gpu0\n")
    result = run_script("run_queue.py", "--queue", queue_path, "--state", state, "--gpus", "0", "--own-marker", "ours",
                        "--disk-folder", tmp_path, check=False)
    assert result.returncode != 0 and "still alive" in result.stdout
    assert (state / "pids.txt").read_text().count("\n") == 1
    assert live_own_processes(state, "ours") == [alive.pid]
    assert live_own_processes(state, "somebody_else") == []
    alive.terminate()
    alive.wait(timeout=10)
    assert live_own_processes(state, "ours") == []


def test_a_job_that_dies_after_a_long_run_does_not_use_up_its_attempts(tmp_path):
    runner = make_runner(tmp_path, [job("a", "raise SystemExit(1)")], gpus="0")
    runner.tick()
    runner.running["a"]["process"].wait(timeout=10)
    runner.running["a"]["started"] -= 2 * 3600   # as if it had run for two hours, saving checkpoints
    runner.tick()
    assert runner.attempts["a"] == 1 and "a" in runner.running and not runner.failed
    runner.running["a"]["process"].wait(timeout=10)
    runner.tick()   # a quick death counts
    assert runner.attempts["a"] == 2 and not runner.failed
    runner.running["a"]["process"].wait(timeout=10)
    runner.tick()   # and a second quick death ends it
    assert runner.failed == {"a"}


def test_the_stop_now_file_asks_running_jobs_to_save_and_ends_the_run(tmp_path):
    runner = make_runner(tmp_path, [job("a", FINISH_ON_SIGTERM), job("b", after=["a"])], gpus="0")
    runner.tick()
    time.sleep(0.5)
    (tmp_path / "state" / "STOP_NOW").write_text("")
    assert runner.tick() is True
    assert runner.running["a"]["stopping"]
    runner.running["a"]["process"].wait(timeout=10)
    assert runner.tick() is False
    assert not runner.running and not runner.failed and not runner.is_done("a")


def test_jobs_see_gpus_in_bus_order(tmp_path):
    marker = tmp_path / "seen.txt"
    code = (f"import os; open({str(marker)!r},'w').write(os.environ['CUDA_DEVICE_ORDER'] + ' ' + "
            "os.environ['CUDA_VISIBLE_DEVICES'])")
    drain(make_runner(tmp_path, [job("a", code)], gpus="1"))
    assert marker.read_text() == "PCI_BUS_ID 1"


@pytest.mark.skipif(shutil.which("nvidia-smi") is None, reason="the real runner reads nvidia-smi")
def test_the_runner_ends_by_printing_one_json_line(tmp_path):
    queue_path = tmp_path / "queue.json"
    queue_path.write_text("[]")
    result = run_script("run_queue.py", "--queue", queue_path, "--state", tmp_path / "state", "--gpus", "0",
                        "--own-marker", "ours", "--disk-folder", tmp_path)
    assert json.loads(result.stdout.strip().splitlines()[-1])["script"] == "run_queue"


def test_a_long_command_line_is_read_whole():
    """The marker of a real job sits far along its command line."""
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", "x" * 400, "--out", "ours"])
    try:
        assert command_line(process.pid).endswith("--out ours")
    finally:
        process.terminate()
        process.wait(timeout=10)


def test_our_own_processes_and_exited_ones_are_not_foreign():
    commands = {1: "/some/python /work/ours/train.py", 2: "", 3: "[python] <defunct>", 4: "/usr/bin/other_user_job"}
    found = foreign_processes(0, "ours", list_pids=lambda gpu: list(commands), read_command=commands.get)
    assert found == [4]


def test_low_memory_or_a_full_disk_starts_nothing(tmp_path):
    runner = make_runner(tmp_path, [job("a")], memory=5.0)
    runner.tick()
    assert not runner.running and not runner.is_done("a")
    runner = make_runner(tmp_path, [job("a")], disk=26.0)
    runner.tick()
    assert not runner.running and not runner.is_done("a")


def test_very_low_memory_returns_the_newest_job(tmp_path):
    runner = make_runner(tmp_path, [job("old", FINISH_ON_SIGTERM), job("new", FINISH_ON_SIGTERM)])
    runner.tick()
    time.sleep(0.5)
    assert set(runner.running) == {"old", "new"}
    runner.read_memory = lambda: 5.0
    runner.tick()
    assert runner.running["new"].get("stopping") and not runner.running["old"].get("stopping")
    for name in list(runner.running):
        runner.running[name]["process"].terminate()
        runner.running[name]["process"].wait(timeout=10)


def test_the_stop_file_starts_nothing_new_and_ends_the_run(tmp_path):
    runner = make_runner(tmp_path, [job("a")])
    (tmp_path / "state" / "STOP").write_text("")
    assert runner.tick() is False
    assert not runner.is_done("a")


def test_per_gpu_limit(tmp_path):
    jobs = [job(n, FINISH_ON_SIGTERM) for n in "abc"]
    runner = make_runner(tmp_path, jobs, gpus="0", per_gpu=2)
    runner.tick()
    assert len(runner.running) == 2
    for name in list(runner.running):
        runner.running[name]["process"].terminate()
        runner.running[name]["process"].wait(timeout=10)


def queue_arguments(smoke, stage="all"):
    return argparse.Namespace(python="/py", code="/code", train_csv="/d/train.csv", test_csv="/d/test.csv",
                              original_csv="/d/orig.csv", id_col="id", original_id_col="transaction_id",
                              target_col="is_fraud", run="/run", n_folds=5, orders=30, batch_sequences=256,
                              score_precision="fp32", smoke=smoke, smoke_rows=400, stage=stage)


def test_full_queue_uses_the_recipe_and_keeps_each_fold_on_its_own_model():
    jobs = {j["job"]: j for j in make_queue.build_jobs(queue_arguments(False))}
    assert len(jobs) == 1 + 1 + 5 * 3 + 3
    assert "60" in jobs["replica"]["argv"] and "--max-rows" not in jobs["replica"]["argv"]
    for fold in range(5):
        fit = jobs[f"fit_f{fold}"]
        assert fit["after"] == ["replica"] and "8" in fit["argv"]
        assert fit["argv"][fit["argv"].index("--fold") + 1] == str(fold)
        for split in ("oof", "test"):
            score = jobs[f"score_{split}_f{fold}"]
            assert score["after"] == [f"fit_f{fold}"]
            assert score["argv"][score["argv"].index("--fold") + 1] == str(fold)
            assert score["argv"][score["argv"].index("--model-dir") + 1] == f"/run/fits/f{fold}/final"
            assert score["argv"][score["argv"].index("--orders") + 1] == "30"
    assert set(jobs["predictions"]["after"]) == {f"score_{s}_f{k}" for s in ("oof", "test") for k in range(5)}
    order = list(jobs)
    assert order.index("fit_f0") + 1 == order.index("score_oof_f0")   # the first reading comes before the other fits
    assert max(order.index(f"fit_f{k}") for k in range(5)) < min(order.index(f"score_oof_f{k}") for k in range(1, 5))
    assert max(order.index(f"score_oof_f{k}") for k in range(5)) < min(order.index(f"score_test_f{k}") for k in range(5))
    assert jobs["stack"]["after"] == ["predictions", "xgb"]
    assert all(j["argv"][0].startswith("/") and j["argv"][1].startswith("/") for j in jobs.values())


def test_stage_oof_holds_test_scoring_and_stage_all_adds_it_back_unchanged():
    held = {j["job"]: j for j in make_queue.build_jobs(queue_arguments(False, stage="oof"))}
    everything = {j["job"]: j for j in make_queue.build_jobs(queue_arguments(False, stage="all"))}
    assert not [name for name in held if name.startswith("score_test")]
    assert {"predictions", "stack"}.isdisjoint(held)
    assert {f"score_oof_f{k}" for k in range(5)} <= set(held) and "xgb" in held
    assert all(everything[name] == held[name] for name in held)   # same names, so done jobs stay done after the switch


def test_make_queue_creates_the_folder_for_its_output(tmp_path):
    out = tmp_path / "new_folder" / "queue.json"
    run_script("make_queue.py", "--python", "/py", "--code", "/code", "--train-csv", "t", "--test-csv", "t", "--original-csv",
               "o", "--id-col", "id", "--original-id-col", "tid", "--target-col", "y", "--run", tmp_path / "run", "--out", out)
    assert json.loads(out.read_text())[0]["job"] == "folds"


def test_smoke_queue_limits_rows_and_epochs():
    jobs = {j["job"]: j for j in make_queue.build_jobs(queue_arguments(True))}
    assert "--max-rows" in jobs["replica"]["argv"] and "--max-rows" in jobs["fit_f0"]["argv"]
    assert "--allow-partial" in jobs["predictions"]["argv"]
