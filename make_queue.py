"""Write the job list for run_queue.py: the whole pipeline, or a tiny version of it with --smoke.

Pipeline: folds -> replica -> fit fold 0 -> score fold 0 -> fit the other folds -> score their OOF rows -> test scoring
          -> predictions -> baseline XGBoost -> stack of the two. Jobs start in this order when a GPU is free.
          Fold 0 is scored right after its fit, so the first complete reading comes after one fit, on any number of GPUs.
          OOF comes before test because test is over twice the work, and OOF alone gives the CV to decide on.

Stages: --stage oof stops the list at the OOF rows and the baseline, so the CV can be read before paying for test.
        --stage all is the whole list. Job names do not change between stages, so writing the queue again with
        --stage all adds the rest and keeps what is done. The runner reads the file on every pass.

Input:  absolute paths for the Python, the code folder, the data files and the run folder.
Output: the queue JSON at --out. All outputs of the jobs go under --run.

Run:    python make_queue.py --python PYTHON --code CODE --train-csv train.csv --test-csv test.csv
            --original-csv original.csv --id-col id --original-id-col transaction_id --target-col is_fraud
            --run RUN --out RUN/queue.json --stage oof
"""
import argparse
import json
from pathlib import Path

from common import write_json


def build_jobs(a):
    code = Path(a.code)
    run = Path(a.run)
    fold_ids = str(run / "fold_ids.npy")
    replica, fits, scores = run / "replica", run / "fits", run / "scores"
    predictions, baseline, stacked = run / "gpt2_target_first", run / "xgb_baseline", run / "stack"
    limit = lambda n: ["--max-rows", str(n)] if a.smoke else []

    def job(name, script, *arguments, after=(), cpu=False):
        return {"job": name, "argv": [a.python, str(code / script), *map(str, arguments)], "after": list(after), "cpu": cpu}

    def fit(fold):
        return job(f"fit_f{fold}", "train_classifier.py", "--train-csv", a.train_csv, "--fold-ids", fold_ids, "--fold", fold,
                   "--id-col", a.id_col, "--target-col", a.target_col, "--replica", replica / "final",
                   "--out", fits / f"f{fold}", "--epochs", 1 if a.smoke else 8, *limit(a.smoke_rows), after=["replica"])

    def score(split, fold):
        csv = a.train_csv if split == "oof" else a.test_csv
        return job(f"score_{split}_f{fold}", "score_rows.py", "--model-dir", fits / f"f{fold}" / "final", "--csv", csv,
                   "--split", split, "--fold-ids", fold_ids, "--fold", fold, "--id-col", a.id_col, "--target-col",
                   a.target_col, "--out", scores / f"f{fold}_{split}", "--batch-sequences", a.batch_sequences, "--orders",
                   a.orders, "--precision", a.score_precision, *limit(a.smoke_rows // 2), after=[f"fit_f{fold}"])

    folds = range(a.n_folds)
    jobs = [job("folds", "make_folds.py", "--train-csv", a.train_csv, "--target-col", a.target_col, "--out", fold_ids,
                cpu=True)]
    jobs.append(job("replica", "train_replica.py", "--original-csv", a.original_csv, "--id-col", a.original_id_col,
                    "--target-col", a.target_col, "--out", replica, "--epochs", 1 if a.smoke else 60,
                    *limit(a.smoke_rows), after=["folds"]))
    jobs += [fit(0), score("oof", 0)]
    jobs += [fit(fold) for fold in folds if fold > 0]
    jobs += [score("oof", fold) for fold in folds if fold > 0]
    if a.stage == "all":
        jobs += [score("test", fold) for fold in folds]
        jobs.append(job("predictions", "make_predictions.py", "--scores", scores, "--train-csv", a.train_csv, "--test-csv",
                        a.test_csv, "--fold-ids", fold_ids, "--target-col", a.target_col, "--out", predictions,
                        "--replica-summary", replica / "train_summary.json", "--fold-summaries", fits,
                        *(["--allow-partial"] if a.smoke else []),
                        after=[f"score_{split}_f{fold}" for split in ("oof", "test") for fold in folds], cpu=True))
    if not a.smoke:  # the partial OOF of a smoke run cannot be stacked; the tests cover these two on tiny data
        jobs.append(job("xgb", "baseline_xgb.py", "--train-csv", a.train_csv, "--test-csv", a.test_csv, "--fold-ids",
                        fold_ids, "--id-col", a.id_col, "--target-col", a.target_col, "--out", baseline, after=["folds"]))
    if not a.smoke and a.stage == "all":
        jobs.append(job("stack", "stack_lr.py", "--input", predictions, "--input", baseline, "--train-csv", a.train_csv,
                        "--fold-ids", fold_ids, "--target-col", a.target_col, "--out", stacked,
                        after=["predictions", "xgb"], cpu=True))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--python", required=True, help="absolute path of the Python to run the jobs with")
    parser.add_argument("--code", required=True, help="absolute path of this folder")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--original-csv", required=True)
    parser.add_argument("--id-col", required=True)
    parser.add_argument("--original-id-col", required=True)
    parser.add_argument("--target-col", required=True)
    parser.add_argument("--run", required=True, help="absolute path of the folder for all outputs")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--orders", type=int, default=30, help="the post uses 30")
    parser.add_argument("--batch-sequences", type=int, default=256)
    parser.add_argument("--score-precision", choices=["fp32", "tf32", "bf16"], default="fp32")
    parser.add_argument("--stage", choices=["oof", "all"], default="all", help="oof holds test scoring; see the top of this file")
    parser.add_argument("--smoke", action="store_true", help="1 epoch and --smoke-rows rows, to test the pipeline")
    parser.add_argument("--smoke-rows", type=int, default=400)
    args = parser.parse_args()
    jobs = build_jobs(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, jobs)  # through a temporary file, so a runner reading it never sees half a file
    print(json.dumps({"script": "make_queue", "jobs": len(jobs), "stage": args.stage, "smoke": args.smoke}))


if __name__ == "__main__":
    main()
