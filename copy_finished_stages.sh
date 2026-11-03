#!/bin/bash
# Copy each finished stage of a run from the training machine to a local folder, soon after the stage ends.
#
# Input:  HOST (ssh target), RUN (the run folder on that machine, with its queue state in RUN/state), LOCAL (a folder
#         here), and how often to look, in seconds. 0 means one pass and exit.
# Does:   lists RUN/state/done. For each finished job not copied yet, copies what that job leaves for later: final models,
#         summaries, progress logs, score chunks, predictions. It never copies resumable checkpoints: they change while
#         a job runs and are deleted when it ends. A second rsync pass compares checksums, and a job counts as copied
#         only if nothing differs. A failed ssh or rsync is logged and tried again on the next pass.
# Output: LOCAL/<same relative paths as under RUN>, LOCAL/.copied/<job> markers, and one log line per event.
# Ends:   by itself once the last job ("stack") is copied, or when it is stopped. Running it again is safe.
#
# Run:    copy_finished_stages.sh HOST RUN LOCAL 600
set -u

if [ $# -ne 4 ]; then
    sed -n '2,11p' "$0"
    exit 2
fi
HOST=$1 RUN=$2 LOCAL=$3 SECONDS_BETWEEN=$4
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15"

log() { echo "$(date '+%F %T') $*"; }

paths_of_job() {
    case "$1" in
        folds) echo fold_ids.npy ;;
        replica) echo replica/final replica/train_summary.json replica/progress.jsonl ;;
        fit_f*) fold=${1#fit_}; echo "fits/$fold/final fits/$fold/train_summary.json fits/$fold/progress.jsonl" ;;
        score_oof_f*) echo "scores/${1#score_oof_}_oof" ;;
        score_test_f*) echo "scores/${1#score_test_}_test" ;;
        predictions) echo gpt2_target_first ;;
        xgb) echo xgb_baseline ;;
        stack) echo stack ;;
        *) return 1 ;;
    esac
}

copy_job() {
    job=$1
    paths=$(paths_of_job "$job") || { log "$job: no known files; skipping"; return 1; }
    for path in $paths; do
        parent=$(dirname "$path")
        mkdir -p "$LOCAL/$parent" || return 1
        rsync -rtc -e "$SSH" "$HOST:$RUN/$path" "$LOCAL/$parent/" || { log "$job: copy of $path failed"; return 1; }
        differing=$(rsync -rtcn --itemize-changes -e "$SSH" "$HOST:$RUN/$path" "$LOCAL/$parent/") \
            || { log "$job: check of $path failed"; return 1; }
        if echo "$differing" | grep -q '^>f'; then
            log "$job: $path differs after copying"
            return 1
        fi
    done
    touch "$LOCAL/.copied/$job"
    log "$job: copied and checked: $paths"
}

mkdir -p "$LOCAL/.copied" || exit 1
while :; do
    if finished=$($SSH "$HOST" "ls '$RUN/state/done'" 2>/dev/null); then
        for name in $finished; do
            job=${name%.json}
            [ -e "$LOCAL/.copied/$job" ] || copy_job "$job"
        done
    else
        log "cannot read the run's state; trying again on the next pass"
    fi
    if [ -e "$LOCAL/.copied/stack" ]; then
        log "the last job is copied; done"
        exit 0
    fi
    [ "$SECONDS_BETWEEN" -eq 0 ] && exit 0
    sleep "$SECONDS_BETWEEN"
done
