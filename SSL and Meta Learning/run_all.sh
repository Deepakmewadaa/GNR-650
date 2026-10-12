#!/usr/bin/env bash
# One-command training pipeline (run on the GPU machine after `bash setup.bash`).
#
#   bash run_all.sh [--config base|small|dino] [--budget HOURS] [--run-name main] [--data-root PATH]
#                   [--skip s1,s2] [--only s1,s2] [--force] [--device auto|cuda|cpu] [--set key=value ...]
#
# Stages (in order): inspect doctor ssl probe baseline ood export eval report
# Every stage is resumable/skippable: finished stages are skipped (use --force to redo them) and
# train_ssl.py resumes automatically from runs/<name>/ssl/last.pt after an interruption.
# Logs: logs/<stage>.log. On failure the script stops and tells you which log to send back.
set -u
set -o pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT" || exit 1

SETS=(); CONFIG=base; BUDGET=6; RUN=main; DATA_ROOT=""; SKIP=""; ONLY=""; FORCE=0; DEVICE=auto
while [ $# -gt 0 ]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --budget|--time-budget-hours) BUDGET="$2"; shift 2 ;;
    --run-name) RUN="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --skip) SKIP="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --device) DEVICE="$2"; shift 2 ;;
    --set) SETS+=("$2"); shift 2 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
done

if [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"; else PY="$(command -v python3 || command -v python)"; fi
[ -d "$ROOT/.pydeps" ] && export PYTHONPATH="$ROOT/.pydeps${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs
COMMON=(--config "$CONFIG" --run-name "$RUN" --time-budget-hours "$BUDGET" --device "$DEVICE")
for kv in ${SETS[@]+"${SETS[@]}"}; do COMMON+=(--set "$kv"); done
DR=()
[ -n "$DATA_ROOT" ] && DR=(--data-root "$DATA_ROOT") && COMMON+=("${DR[@]}")
RES="results/$RUN"

want() {  # want <stage>
  if [ -n "$ONLY" ]; then [[ ",$ONLY," == *",$1,"* ]]; return; fi
  [[ ",$SKIP," != *",$1,"* ]]
}
done_marker() {  # done_marker <file> : true if the stage already finished and --force not given
  [ $FORCE = 0 ] && [ -e "$1" ]
}
run_stage() {  # run_stage <name> <cmd...>
  local name="$1"; shift
  echo ""
  echo "=================================================================== [$name] $(date '+%F %T')"
  echo "+ $*"
  if ! "$@" 2>&1 | tee -a "logs/$name.log"; then
    echo ""
    echo "!!! STAGE '$name' FAILED. Send back: logs/$name.log (and runs/$RUN/*/ *.log if present)."
    echo "!!! Fix the problem and re-run the same command; finished stages are skipped automatically."
    exit 1
  fi
}

echo "[run_all] python=$PY config=$CONFIG budget=${BUDGET}h run=$RUN device=$DEVICE"
T0=$(date +%s)

want inspect  && run_stage inspect  "$PY" scripts/inspect_data.py ${DR[@]+"${DR[@]}"} --run-name "$RUN"
if want doctor; then
  if done_marker "$RES/doctor.json"; then echo "[doctor] already done (use --force to redo)";
  else run_stage doctor "$PY" scripts/doctor.py "${COMMON[@]}"; fi
fi
if want ssl; then
  if done_marker "runs/$RUN/ssl/summary.json"; then echo "[ssl] already finished (use --force to retrain from scratch)";
  elif [ $FORCE = 1 ]; then run_stage ssl "$PY" train_ssl.py "${COMMON[@]}" --fresh;
  else run_stage ssl "$PY" train_ssl.py "${COMMON[@]}"; fi
fi
if want probe; then
  if done_marker "$RES/probe_results.json"; then echo "[probe] already done";
  else run_stage probe "$PY" train_probe.py "${COMMON[@]}"; fi
fi
if want baseline; then
  if done_marker "$RES/baseline_results.json"; then echo "[baseline] already done";
  else run_stage baseline "$PY" train_supervised_baseline.py "${COMMON[@]}"; fi
fi
if want ood; then
  if done_marker "$RES/ood_results.json"; then echo "[ood] already done";
  else run_stage ood "$PY" fit_ood.py "${COMMON[@]}"; fi
fi
if want export; then
  if done_marker "checkpoints/final.pt" && [ "checkpoints/final.pt" -nt "runs/$RUN/ood/ood_final.pt" ]; then echo "[export] already done";
  else run_stage export "$PY" export_final.py "${COMMON[@]}"; fi
fi
if want eval; then
  # Format / end-to-end check of the grader path on labelled images (optimistic: the final model was trained
  # on them). Honest numbers come from the leave-class-out results of fit_ood.py.
  run_stage eval "$PY" scripts/local_eval.py make-test --out-dir local_test --max-per-class 4 ${DR[@]+"${DR[@]}"}
  run_stage eval "$PY" test.py --test-csv "$ROOT/local_test/test.csv" --output-path "$ROOT/local_test/out"
  run_stage eval "$PY" scripts/local_eval.py score --pred local_test/out/predictions.csv --gt local_test/gt.csv \
    --json-out "$RES/local_eval.json"
fi
want report && run_stage report "$PY" scripts/make_report.py --run-name "$RUN"

echo ""
echo "[run_all] finished in $(( ($(date +%s) - T0) / 60 )) min."
echo "[run_all] Now commit the results (run these yourself):"
echo "    git add checkpoints/final.pt results/ docs/"
echo "    git commit -m \"Trained model + results ($RUN, $CONFIG, ${BUDGET}h)\""
echo "    git push"
