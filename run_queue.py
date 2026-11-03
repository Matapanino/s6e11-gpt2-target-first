"""Run a queue of jobs on a shared multi-GPU machine, restartable and polite to other users.

Input:  a queue JSON from make_queue.py: a list of {"job", "argv", "after", "cpu"}. argv must start with absolute paths.
Does:   starts a job when its "after" jobs are done, on a GPU with a free slot. Rules:
          * Another user's process on a GPU: start nothing new there. If it stays for two checks in a row, stop our
            jobs on that GPU (they save and exit) and return them to the queue, so we yield the GPU.
          * Memory low: start nothing new, and return the newest job to the queue if it gets very low.
          * Folder over the disk limit: start nothing new.
          * STATE/STOP exists: start nothing new, and finish once our jobs end.
          * STATE/STOP_NOW exists: the same, and also ask running jobs to save and exit.
          * STATE/gpus.txt exists (for example 0,3): it replaces --gpus, so a GPU can be added without a restart.
          * The queue file is read again on every pass, so it can grow while the runner is alive.
        A process counts as ours when its command line contains --own-marker. Every job has --out under the run
        folder, so "--out RUN" marks exactly our jobs and no other project's.
Output: STATE/done/<job>.json, STATE/logs/<job>.log, STATE/pids.txt (this runner's PID and every job's), STATE/status.json.

Run:    python run_queue.py --queue queue.json --state STATE --gpus 0 --per-gpu 1 --own-marker="--out RUN"
            --disk-folder RUN
Resume: run the same command again. Jobs with a done file are skipped, and long jobs continue from their checkpoints.
        It refuses to start while a process from STATE/pids.txt is alive, because two copies of a job would write the
        same files. Jobs outlive the runner, so wait for them, or stop them by the PIDs in that file.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from common import STOPPED_EXIT_CODE, print_summary, write_json

MAX_ATTEMPTS = 2
PROGRESS_SECONDS = 30 * 60  # every long job saves more often than this, so a job that lived this long lost little
YIELD_AFTER_CHECKS = 2
SILENT_WARNING_SECONDS = 90 * 60


def log(message):
    print(f"{time.strftime('%F %T')} {message}", flush=True)


def gpu_process_ids(gpu):
    """PIDs of the compute processes on one GPU."""
    result = subprocess.run(["nvidia-smi", f"--id={gpu}", "--query-compute-apps=pid", "--format=csv,noheader"],
                            capture_output=True, text=True)
    return [int(pid) for pid in result.stdout.split()]


def command_line(pid):
    """The whole command line. Without -ww, ps may cut it at 80 columns, and our own job would then look foreign."""
    return subprocess.run(["ps", "-ww", "-o", "args=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()


def foreign_processes(gpu, own_marker, list_pids=gpu_process_ids, read_command=command_line):
    """PIDs on a GPU that are not ours. An exited process shows no command line, or <defunct>: not another user."""
    foreign = []
    for pid in list_pids(gpu):
        command = read_command(pid)
        if command and "<defunct>" not in command and own_marker not in command:
            foreign.append(pid)
    return foreign


def live_own_processes(state, own_marker, read_command=command_line):
    """PIDs from STATE/pids.txt that are alive and ours: an earlier runner, or a job that outlived it."""
    path = Path(state) / "pids.txt"
    if not path.exists():
        return []
    pids = sorted({int(line.split()[0]) for line in path.read_text().splitlines() if line.strip()})
    return [pid for pid in pids if own_marker in read_command(pid) and "<defunct>" not in read_command(pid)]


def available_memory_gib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 2**20
    raise RuntimeError("no MemAvailable in /proc/meminfo")


def folder_size_gib(folder):
    output = subprocess.run(["du", "-sk", str(folder)], capture_output=True, text=True).stdout.split()
    return int(output[0]) / 2**20 if output else 0.0


class Queue:
    def __init__(self, args, list_foreign=None, read_memory=available_memory_gib, read_disk=folder_size_gib):
        self.args = args
        self.state = Path(args.state)
        (self.state / "done").mkdir(parents=True, exist_ok=True)
        (self.state / "logs").mkdir(exist_ok=True)
        self.gpus = [int(g) for g in args.gpus.split(",")]
        self.running = {}   # job name -> {"process", "gpu", "handle", "started", "log"}
        self.attempts = {}
        self.failed = set()
        self.foreign_checks = {gpu: 0 for gpu in self.gpus}
        self.warned_silent = set()
        self.warned_disk = False
        self.list_foreign = list_foreign or (lambda gpu: foreign_processes(gpu, args.own_marker))
        self.read_memory = read_memory
        self.read_disk = read_disk

    def is_done(self, job):
        return (self.state / "done" / f"{job}.json").exists()

    def collect_finished(self):
        for job, entry in list(self.running.items()):
            code = entry["process"].poll()
            if code is None:
                continue
            entry["handle"].close()
            del self.running[job]
            seconds = time.time() - entry["started"]
            if code == 0:
                write_json(self.state / "done" / f"{job}.json", {"job": job, "seconds": seconds, "gpu": entry["gpu"]})
                log(f"{job} finished on GPU {entry['gpu']} in {seconds:.0f}s")
            elif code == STOPPED_EXIT_CODE or entry.get("stopping"):
                self.attempts[job] -= 1
                log(f"{job} stopped on GPU {entry['gpu']}; back in the queue")
            else:
                log(f"{job} FAILED attempt {self.attempts[job]} on GPU {entry['gpu']} exit {code}")
                if seconds > PROGRESS_SECONDS:
                    self.attempts[job] = 0  # it saved progress, so only jobs that die early use up their attempts
                if self.attempts[job] >= MAX_ATTEMPTS:
                    self.failed.add(job)

    def warn_silent(self):
        for job, entry in self.running.items():
            silent = time.time() - entry["log"].stat().st_mtime
            if silent > SILENT_WARNING_SECONDS and job not in self.warned_silent:
                self.warned_silent.add(job)
                log(f"{job} has written nothing for {silent / 60:.0f} minutes")

    def stop_job(self, job, reason):
        """Ask one of our jobs to save and exit. Its PID is held by this process, so nothing is matched by name."""
        entry = self.running[job]
        if not entry.get("stopping"):
            entry["stopping"] = True
            entry["process"].terminate()
            log(f"stopping {job} on GPU {entry['gpu']}: {reason}")

    def enforce_limits(self):
        if (self.state / "STOP_NOW").exists():
            for job in list(self.running):
                self.stop_job(job, "the STOP_NOW file")
        if self.read_memory() < self.args.floor_memory_gib and len(self.running) > 1:
            self.stop_job(list(self.running)[-1], "memory very low")

    def stop_file_exists(self):
        return (self.state / "STOP").exists() or (self.state / "STOP_NOW").exists()

    def read_gpu_list(self):
        """STATE/gpus.txt, when present, replaces the GPU list. A file that cannot be read changes nothing."""
        path = self.state / "gpus.txt"
        if not path.exists():
            return
        try:
            gpus = [int(gpu) for gpu in path.read_text().split(",")]
        except ValueError:
            log(f"cannot read {path.name}; keeping GPUs {self.gpus}")
            return
        if gpus != self.gpus:
            log(f"GPU list is now {gpus}")
            self.gpus = gpus

    def yield_gpus(self):
        """Returns the GPUs that may start new jobs now. A GPU that left the list still yields while our job is on it."""
        usable = []
        for gpu in sorted(set(self.gpus) | {entry["gpu"] for entry in self.running.values()}):
            try:
                foreign = self.list_foreign(gpu)
            except ValueError as error:  # nvidia-smi printed an error message in place of PIDs
                log(f"cannot read GPU {gpu}, starting nothing there this pass: {error}")
                continue
            if not foreign:
                self.foreign_checks[gpu] = 0
                if gpu in self.gpus:
                    usable.append(gpu)
                continue
            self.foreign_checks[gpu] = self.foreign_checks.get(gpu, 0) + 1
            if self.foreign_checks[gpu] == 1:
                log(f"GPU {gpu} is in use by another user {foreign}; starting nothing there")
            if self.foreign_checks[gpu] >= YIELD_AFTER_CHECKS:
                for job, entry in list(self.running.items()):
                    if entry["gpu"] == gpu:
                        self.stop_job(job, f"another user on GPU {gpu}")
        return usable

    def start_jobs(self, queue, usable_gpus):
        if self.stop_file_exists():
            return
        if self.read_disk(self.args.disk_folder) >= self.args.max_disk_gib:
            if not self.warned_disk:
                log(f"disk limit reached: over {self.args.max_disk_gib} GiB; starting nothing")
            self.warned_disk = True
            return
        self.warned_disk = False
        pending = [item for item in queue if not self.is_done(item["job"]) and item["job"] not in self.running
                   and item["job"] not in self.failed and all(self.is_done(name) for name in item.get("after", []))]
        for gpu in usable_gpus:
            while pending and sum(e["gpu"] == gpu for e in self.running.values()) < self.args.per_gpu \
                    and self.read_memory() >= self.args.min_memory_gib:
                self.launch(pending.pop(0), gpu)

    def launch(self, item, gpu):
        job = item["job"]
        self.attempts[job] = self.attempts.get(job, 0) + 1
        log_path = self.state / "logs" / f"{job}.log"
        handle = log_path.open("a")
        visible = "" if item.get("cpu") else str(gpu)
        environment = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=visible, PYTHONUNBUFFERED="1")
        process = subprocess.Popen(item["argv"], stdout=handle, stderr=subprocess.STDOUT, env=environment)
        with (self.state / "pids.txt").open("a") as pids:
            pids.write(f"{process.pid} {job} gpu{gpu}\n")
        self.running[job] = {"process": process, "gpu": gpu, "handle": handle, "started": time.time(), "log": log_path}
        log(f"{job} started on GPU {gpu} attempt {self.attempts[job]} pid {process.pid}")

    def write_status(self, queue):
        finished = [item["job"] for item in queue if self.is_done(item["job"])]
        write_json(self.state / "status.json", {
            "utc": time.time(), "total": len(queue), "finished": len(finished), "running": sorted(self.running),
            "failed": sorted(self.failed), "waiting": len(queue) - len(finished) - len(self.running) - len(self.failed)})
        return finished

    def tick(self):
        """One pass of the loop. Returns False when there is nothing more to do."""
        queue = json.loads(Path(self.args.queue).read_text())
        self.read_gpu_list()
        self.collect_finished()
        self.warn_silent()
        self.enforce_limits()
        usable = self.yield_gpus()
        self.start_jobs(queue, usable)
        finished = self.write_status(queue)
        if len(finished) == len(queue):
            log("queue complete")
            return False
        if not self.running:
            if self.stop_file_exists():
                log("stopped by the stop file")
                return False
            blocked = self.blocked_jobs(queue)
            if self.failed and all(self.is_done(item["job"]) or item["job"] in blocked for item in queue):
                log(f"stopped: failed jobs {sorted(self.failed)}")
                return False
        return True

    def blocked_jobs(self, queue):
        """Failed jobs, and every job waiting on one of them, directly or through another job."""
        blocked = set(self.failed)
        changed = True
        while changed:
            changed = False
            for item in queue:
                if item["job"] not in blocked and any(name in blocked for name in item.get("after", [])):
                    blocked.add(item["job"])
                    changed = True
        return blocked


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--gpus", required=True, help="for example 0,1,2,3")
    parser.add_argument("--per-gpu", type=int, default=1, help="jobs at once on one GPU")
    parser.add_argument("--own-marker", required=True, help="text found in the command line of our own processes")
    parser.add_argument("--disk-folder", required=True, help="the folder held under --max-disk-gib")
    parser.add_argument("--max-disk-gib", type=float, default=25)
    parser.add_argument("--min-memory-gib", type=float, default=30, help="start nothing below this much free memory")
    parser.add_argument("--floor-memory-gib", type=float, default=12, help="return the newest job below this")
    parser.add_argument("--poll-seconds", type=float, default=20)
    args = parser.parse_args()

    alive = live_own_processes(args.state, args.own_marker)
    if alive:
        print_summary("run_queue", status="refused", reason="processes from pids.txt are still alive", pids=alive)
        sys.exit(1)
    queue = Queue(args)
    with (queue.state / "pids.txt").open("a") as pids:
        pids.write(f"{os.getpid()} runner\n")
    while queue.tick():
        time.sleep(args.poll_seconds)
    print_summary("run_queue", status="ended", finished=len(list((queue.state / "done").glob("*.json"))),
                  failed=sorted(queue.failed))


if __name__ == "__main__":
    main()
