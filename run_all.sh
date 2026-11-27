#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
ALGOS="${ALGOS:-ddpg td3 sac}"
SEEDS="${SEEDS:-0 1 2}"
MODES="${MODES:-nominal dr}"
CORES="$(sysctl -n hw.ncpu 2>/dev/null || nproc)"
JOBS="${JOBS:-$(( CORES > 2 ? CORES - 2 : 1 ))}"

mkdir -p logs

run_one() {
  local algo="$1" seed="$2" mode="$3" tag extra=""
  if [ "$mode" = "dr" ]; then
    tag="${algo}_dr_seed${seed}"
    extra="--dr"
  else
    tag="${algo}_seed${seed}"
  fi
  if "$PYTHON" train.py --algo "$algo" --seed "$seed" $extra > "logs/${tag}.log" 2>&1; then
    echo "done    $tag"
  else
    echo "FAILED  $tag (see logs/${tag}.log)"
  fi
}
export -f run_one
export PYTHON

runs=()
for seed in $SEEDS; do
  for mode in $MODES; do
    for algo in $ALGOS; do
      runs+=("$algo" "$seed" "$mode")
    done
  done
done

echo "Launching $(( ${#runs[@]} / 3 )) runs with up to $JOBS in parallel using $PYTHON"
printf '%s\n' "${runs[@]}" | xargs -n 3 -P "$JOBS" bash -c 'run_one "$@"' _
echo "All runs finished. Logs are in logs/, weights in models/."
