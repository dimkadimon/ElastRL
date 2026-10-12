#!/usr/bin/env bash
# chain_runs.sh - train the harder levels sequentially (2 vCPUs => one run at a time).
#
# Reward/exploration settings differ from level 0 on purpose: on the harder levels the
# "brake and stand still" policy never crashes, and with a large crash penalty that is
# the safest way to accumulate return, so the agent locks into a do-nothing optimum.
# Lowering the crash penalty, adding a cost for giving up (stuck/timeout) and keeping a
# floor under the entropy coefficient make trying strictly better than not trying.
set -u
cd /home/user/elma-rl

run () {
  local level="$1"; local steps="$2"; local minutes="$3"; local seed="$4"
  echo "[chain] === level $level: $steps steps, ${minutes} min budget, seed $seed ($(date))"
  python3 train.py \
    --level "$level" --envs 64 --steps "$steps" --upd-per-step 2 \
    --batch-size 128 --eval-every 500 --ckpt-every 500 --log-every 100 \
    --video-every 1000 --video-size 960x540 \
    --crash-penalty 2.0 --stuck-penalty 1.0 --timeout-penalty 1.0 \
    --alpha-min 0.005 \
    --outdir "runs/level$level" --max-minutes "$minutes" --seed "$seed" \
    >> "runs/level$level/train.log" 2>&1
  echo "[chain] === level $level done ($(date))"
}

run 1 20000 40 2
run 2 20000 40 3

echo "[chain] ALL RUNS DONE at $(date)"
