# Target-first GPT-2 classifier for Playground S6E11

The method is [Chris Deotte](https://www.kaggle.com/cdeotte)'s: [1st Place - TFMs and LLMs with tabular data!](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/754683) (S6E10). This applies that recipe to S6E11; any mistakes are mine.

The post also links [Stacking TFMs with Public OOF](https://www.kaggle.com/code/cdeotte/stacking-tfms-with-public-oof). The replica step uses GReaT: Borisov et al., [Language Models are Realistic Tabular Data Generators](https://arxiv.org/abs/2210.06280) (ICLR 2023), and the [be_great library](https://github.com/tabularis-ai/be_great).

The replica was trained on [Credit Card Fraud Detection 2026](https://www.kaggle.com/datasets/uditjain13/credit-card-fraud-detection-2026) by [uditjain13](https://www.kaggle.com/uditjain13).

A GPT-2 with no classifier head reads each table row as text with the label written first. Each row is scored twice, once as "True" and once as "False", and Bayes' rule turns the two scores into a prediction.

## For people and AI agents: how to run it

Run from any folder. All paths and settings are arguments; nothing is read from the environment except the GPU choice.

| # | Script | Input | Does | Output |
|---|---|---|---|---|
| 1 | `make_folds.py` | `train.csv` | `StratifiedKFold(5, shuffle=True, random_state=42)` | `fold_ids.npy` |
| 2 | `train_replica.py` | original dataset CSV | Fine-tunes GPT-2 with be_great, 60 epochs | `replica/final/` |
| 3 | `train_classifier.py` | `train.csv`, `fold_ids.npy`, replica | Continues training on one fold's fit rows, target first, 8 epochs. Run once per fold | `fits/f{k}/final/` |
| 4 | `score_rows.py` | a fold model, `train.csv` or `test.csv` | Scores each row under both labels at 30 feature orders. Run with `--split oof` and `--split test` for every fold | `scores/f{k}_oof/`, `scores/f{k}_test/` |
| 5 | `make_predictions.py` | `scores/`, both CSVs, `fold_ids.npy` | Combines orders by log-sum-exp, adds the prior, and fills the rows in CSV order | `oof.npy`, `test.npy`, `manifest.json` |
| 6 | `baseline_xgb.py` | both CSVs, `fold_ids.npy` | XGBoost on the raw columns, same folds, for comparison | `oof.npy`, `test.npy`, `manifest.json` |
| 7 | `stack_lr.py` | folders with `oof.npy`, `test.npy` | L2 logistic regression on log-odds, same folds | `oof.npy`, `test.npy`, `manifest.json` |

`make_queue.py` writes all of this as a job list, and `run_queue.py` runs that list on several GPUs.

```sh
python make_queue.py --python "$PWD/.venv/bin/python" --code "$PWD" \
    --train-csv "$PWD/data/train.csv" --test-csv "$PWD/data/test.csv" --original-csv "$PWD/data/original.csv" \
    --id-col id --original-id-col transaction_id --target-col is_fraud \
    --run "$PWD/run" --out run/queue.json --stage oof
python run_queue.py --queue run/queue.json --state run/state --gpus 0 --per-gpu 1 \
    --own-marker="--out $PWD/run" --disk-folder "$PWD"
```

Every long job writes `progress.jsonl` and picks up where it stopped if run again. Every script ends by printing one JSON line. If the runner dies, its jobs keep running: run the same command once they have ended, and it refuses to start until then.

`copy_finished_stages.sh` copies each finished stage to another folder as soon as it ends, and checks the copy.

## Reading early and holding the run

The full recipe costs far more GPU time than a pilot, and a pilot reads well below the full recipe, so the queue is built to read a real result as early as possible.

**The order.** Fold 0 is scored right after its fit, before the other folds are trained. `--stage oof` stops the list after the OOF rows and the baseline, so test scoring waits until you have read the CV.

**Read while scoring runs.** Run `baseline_xgb.py` first if you have no baseline yet; its scoring rows match the GPT read. A separate `--out` keeps the read apart from the final predictions.

```sh
python make_predictions.py --scores run/scores --train-csv data/train.csv --test-csv data/test.csv \
    --fold-ids run/fold_ids.npy --target-col is_fraud --out run/early_read \
    --allow-partial --compare-oof BASELINE/oof.npy
```

**How to read it.** In `manifest.json`, under `per_fold`, compare `auc` with `compare_auc_same_rows` only: both read the same rows. A first chunk of 2,000 rows holds about 33 positives, so treat it as a look. Decide when the fold's scoring has ended. `label_fit` shows whether the model for the rarer label fits at all.

**Go on to test.** Write the queue again with `--stage all`. The runner reads the file on every pass, and finished jobs stay finished. If the runner has already ended, start it again with the same command.

**Control a live run.** These all act on the next pass of the runner, within about 20 seconds.

| To do this | Do this |
|---|---|
| See where it is | Read `STATE/status.json`, `STATE/logs/<job>.log` and each job's `progress.jsonl` |
| Use more or fewer GPUs | Write the list to `STATE/gpus.txt`, for example `0,3` |
| Start nothing new, let running jobs finish | Create `STATE/STOP` |
| Stop now: running jobs save and exit | Create `STATE/STOP_NOW` |
| Start again after a stop | Remove the stop file and run the same command |

Long jobs lose at most their last save when stopped hard. Save intervals are command-line arguments.

**Saved predictions.** `oof.npy` is in `train.csv` row order and `test.npy` in `test.csv` row order, both as log-odds. `manifest.json` holds the settings, per-fold AUC, seconds and sha256.

**Do not change** these without re-reading the post: the text rules in `tabtext.py`, the sum in `sequence_log_probs`, or the row selection in `fit_table`. The tests pin all three.

**Tests.** `pytest tests` runs on CPU in a few minutes with tiny data, and needs only public Hugging Face access for the tokenizer.

## The recipe, and where each setting comes from

| Setting | Value | Source |
|---|---|---|
| Base model, tokenizer | GPT-2 small, its own tokenizer | Post |
| Replica | `GReaT(llm="gpt2", batch_size=16, epochs=60).fit(original_df)` | Post |
| Classifier | Target first, fresh feature order every step, loss on all tokens | Post |
| Classifier training | 8 epochs, AdamW, lr 5e-5 with linear decay to 0, batch 32, bf16 | Post |
| Text | `column is value` joined by `, `; values as written; missing `None`; target `True` or `False` | Post |
| Score | Sum of log-probabilities after the label prefix, end token included | Post's code |
| Combine | Log-sum-exp over 30 orders per label, then the difference, plus the prior | Post |
| Stack | Logits clipped at +/-30, L2, C=1, no standardization | Post and its stacking notebook |

## Choices the post leaves open

| Choice | Picked | Why |
|---|---|---|
| Original data target | Written `True`/`False` for the replica too | S6E10's original was boolean, which be_great writes that way. This one is `0`/`1` |
| Original data id | Dropped | Not a feature in the competition data |
| be_great version | 0.0.14, the latest on PyPI when this was written | The post names none |
| Replica shuffles | Rows seeded by (seed, epoch), columns by (seed, epoch, row) | Makes row and column orders repeatable; see the replica-resume limitation below |
| Classifier weight decay, gradient clipping | 0 and 1.0 | The Hugging Face defaults the replica trains with |
| Classifier dropout | 0.1 | GPT-2's default, also what the replica trains with |
| Classifier padding | Right, with padding out of the loss | Every row then starts at position 0, as it does when scored. be_great pads left for the replica |
| Classifier training loop | Written directly, not the Hugging Face Trainer | Gives an exact resume and a loss per label |
| Scoring precision | fp32 | The post's code uses the default. `--precision tf32` introduces rounding noise; `bf16` also introduces rounding noise |
| Feature orders at scoring | A fresh set per row; the same order under both labels of a row | The formula implies one set per row |
| Prior | Training rate of the fold's fit rows | The post uses the training rate; this keeps it fold-safe. No effect on AUC within a fold |
| Test predictions | Each of the 5 fold models scores test, mean of log-odds | The post does not say |
| Stacker | scikit-learn `lbfgs` in place of cuML `qn` | No GPU needed; the same L2 objective |
| Baseline early stopping | On a 10% split of the fit rows | Stopping on the scored fold would flatter CV |

## Limits

**Replica checkpoint resume is not currently certified.** The CPU crash/resume test in `tests/test_resume.py` fails exact equality between resumed and uninterrupted replica weights on macOS. The saved S6E11 replica completed without a resume, so this failure does not explain or change the shared predictions. The classifier and scoring resume checks pass. Keep replica training uninterrupted until this is resolved; do not interpret the deterministic shuffles as a guarantee of identical resumed weights.


The post reports these results on S6E10 only. A cheap pilot reads well below the full recipe there, so a short replica or a few orders does not predict the final score. S6E11 has a much smaller original dataset (20,000 rows) and far fewer positive rows, so the S6E10 scores are not a forecast for this competition.

## Results on S6E11

5-fold stratified CV, `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, in CSV row order. The shared predictions use **10 feature orders**, selected with `make_predictions.py --use-orders 10`. The post's recipe table above describes its larger scoring budget; this release shares the smaller measured budget.

| Fold | AUC |
|---|---:|
| 0 | 0.948882301690 |
| 1 | 0.950317217156 |
| 2 | 0.955311983854 |
| 3 | 0.949874220040 |
| 4 | 0.945853958754 |
| Pooled OOF | 0.950027203793 |

[Member notebook](https://www.kaggle.com/code/masayakawamata/s6e11-gpt2-target-first-classifier-cv-0-9500) · [OOF and test dataset](https://www.kaggle.com/datasets/masayakawamata/s6e11-gpt2-target-first-oof)

Recorded training job time: 60.29 GPU hours (5.07 replica + 55.22 fold classifiers). Recorded scoring chunk time: 61.62 GPU hours. These are summed job times, not elapsed time with jobs running concurrently. Scoring includes all saved orders in the run, including extra OOF orders for the first two folds; it is not an estimate for an all-ten-order rerun. Interrupted work and setup are not necessarily included.

## Code versions and reproduction

The replica and all five classifiers used the same target-first, True/False training code. The archived training script has SHA-256 `d0c060fe9e5e8b7b9fe57fa1db85580a99517255d69fee187f745fd36dd85045`; the replica script has `04198bbbd3aa23b31781507bcde71f288f0dbd30eca928653b4449fadd38c2b4`. All fold summaries agree on the training settings. The archived scoring code and its later heartbeat-only update have the same scoring arithmetic.

This release includes the latest scripts. Later changes add optional label positions and spellings, checkpoint-settings validation, and scoring heartbeats. The saved run uses the unchanged defaults: `--layout first --label-text truefalse`, eight classifier epochs, and log-sum-exp aggregation. For those defaults, the text, row and feature permutations, loss, optimizer, learning-rate schedule and scoring arithmetic are unchanged. CPU comparisons against the archived scripts check identical batches, loss and parameter updates, scores and aggregation. The new settings are separate experiments; changing them will not reproduce these arrays.

The result is reproducible as a recipe, not a promise of identical floating-point weights across GPU types. To reuse the exact predictions, load the linked dataset. To train, install the GPU-appropriate PyTorch build, then `pip install -r requirements.txt`. Run the pipeline in the order above. For scoring, pass `--orders 10` to every OOF and test scoring job and `--use-orders 10` when assembling predictions. To duplicate the historical extra OOF scoring budget, use the counts listed in the dataset manifest.

Code is MIT licensed; shared predictions are CC BY 4.0.
