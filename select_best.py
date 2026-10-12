"""
select_best.py — choose the genuinely best checkpoint of a run by replaying *all*
saved checkpoints and ranking them on the metric that actually matters:

    1. did the greedy policy finish the level, and
    2. how fast (mean completion time over several evaluation seeds).

Background: the training loop's `best.pt` uses "progress along the reference line" as a
tie-breaker, but that metric is not monotone in skill - a slow, wobbly ride that hugs
the reference line can accumulate *more* path progress than a fast, clean finish that
cuts corners (and cutting corners is what going fast means). Completion time is the
honest criterion, so this script re-ranks the checkpoints with it and overwrites
`best.pt`.

Usage:
    python3 select_best.py --run runs/level0 [--seeds 1234,999,7,42,2024] [--apply]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from elma_env import ElmaEnv, EnvConfig, N_ACTIONS, OBS_DIM, CTRL_DT  # noqa: E402
from elma_levels import make_levels  # noqa: E402
from sac import SAC  # noqa: E402


def level_of(run):
    base = os.path.basename(run.rstrip("/"))
    for i in range(len(make_levels())):
        if base == f"level{i}":
            return i
    return 0


def evaluate_checkpoint(ckpt, level, seeds, hidden=192, cfg=None):
    agent = SAC(obs_dim=OBS_DIM, n_actions=N_ACTIONS, hidden=(hidden, hidden),
                batch_size=128, pi_loss_mode="exact")
    extra = agent.load(ckpt)
    times, progresses, finishes, deads, stucks = [], [], [], [], []
    for s in seeds:
        env = ElmaEnv(level, B=1, seed=s, cfg=cfg or EnvConfig())
        env.reset()
        for t in range(env.max_steps):
            act = agent.act(env.observe(), deterministic=True)
            obs, r, d, info = env.step(act)
            if d[0]:
                break
        times.append(float(info["ep_time"][0]))
        progresses.append(float(info["progress"][0]))
        finishes.append(bool(info["finish"][0]))
        deads.append(bool(info["dead"][0]))
        stucks.append(bool(info["stuck"][0]))
    return dict(
        ckpt=ckpt, step=int(extra.get("step", -1)),
        finish_rate=float(np.mean(finishes)),
        mean_time=float(np.mean(times)) if any(finishes) else None,
        mean_progress=float(np.mean(progresses)),
        mean_time_all=float(np.mean(times)),
        crash_rate=float(np.mean(deads)),
        stuck_rate=float(np.mean(stucks)),
    )


def rank(rows):
    """Best = highest finish rate, then fastest completion among finishers."""
    return sorted(rows, key=lambda r: (-r["finish_rate"],
                                       r["mean_time"] if r["mean_time"] is not None else 1e9,
                                       -r["mean_progress"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--seeds", default="1234,999,7,42,2024")
    ap.add_argument("--apply", action="store_true", help="copy the winner to best.pt")
    ap.add_argument("--json", default="", help="also write the table here")
    ap.add_argument("--goal-radius", type=float, default=None,
                    help="override EnvConfig.goal_radius (default: the calibrated 1.5 m)")
    a = ap.parse_args()

    level = make_levels()[level_of(a.run)]
    cfg = EnvConfig()
    if a.goal_radius is not None:
        cfg.goal_radius = a.goal_radius
    seeds = [int(s) for s in a.seeds.split(",")]
    ckpts = sorted(glob.glob(os.path.join(a.run, "checkpoints", "ckpt_*.pt")))
    final = os.path.join(a.run, "checkpoints", "final.pt")
    if os.path.exists(final):
        ckpts.append(final)
    if not ckpts:
        raise SystemExit(f"no checkpoints in {a.run}/checkpoints")

    rows = []
    for c in ckpts:
        r = evaluate_checkpoint(c, level, seeds, cfg=cfg)
        rows.append(r)
        print(f"  step {r['step']:7d}  finish {r['finish_rate']:.2f}  "
              f"time {('%.2f' % r['mean_time']) if r['mean_time'] is not None else '   - '}  "
              f"prog {r['mean_progress']:6.2f}  crash {r['crash_rate']:.2f}  stuck {r['stuck_rate']:.2f}")

    best = rank(rows)[0]
    print(f"\nWINNER: {os.path.basename(best['ckpt'])} (step {best['step']}) "
          f"finish {best['finish_rate']:.2f}, time "
          f"{('%.2f s' % best['mean_time']) if best['mean_time'] is not None else 'n/a'}")
    if a.apply:
        shutil.copy(best["ckpt"], os.path.join(a.run, "checkpoints", "best.pt"))
        print("wrote", os.path.join(a.run, "checkpoints", "best.pt"))
    out = a.json or os.path.join(a.run, "checkpoint_ranking.json")
    with open(out, "w") as f:
        json.dump(rank(rows), f, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
