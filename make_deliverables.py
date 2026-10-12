"""
make_deliverables.py — build the user-facing artefacts for a trained run:

  runs/<run>/best_episode.mp4      the best policy riding the level (greedy, with HUD)
  runs/<run>/training_progress.mp4 grid of synchronised episodes from successive
                                   checkpoints -> the agent visibly getting better
  runs/<run>/training_curves.png   static training-curve summary
  runs/<run>/training_curves.mp4   animated curves with a playhead over training time
  runs/<run>/comparison.mp4        side-by-side: untrained vs best policy

Usage:
    python3 make_deliverables.py --run runs/level0
    python3 make_deliverables.py --all
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from elma_env import (ElmaEnv, EnvConfig, N_ACTIONS, OBS_DIM, VideoWriter,  # noqa: E402
                      CTRL_DT)
from elma_levels import make_levels  # noqa: E402
from sac import SAC  # noqa: E402
from train import render_episode  # noqa: E402


def load_agent(ckpt, hidden=192, batch=128):
    agent = SAC(obs_dim=OBS_DIM, n_actions=N_ACTIONS, hidden=(hidden, hidden),
                batch_size=batch, pi_loss_mode="exact")
    extra = agent.load(ckpt)
    return agent, extra


def want(run):
    """Map a run directory name to a level index."""
    base = os.path.basename(run.rstrip("/"))
    for i in range(len(make_levels())):
        if base == f"level{i}":
            return i
    return 0


def read_clip_frames(path):
    import imageio_ffmpeg
    gen = imageio_ffmpeg.read_frames(path, pix_fmt="rgb24")
    meta = next(gen)
    frames = []
    for buf in gen:
        frames.append(Image.frombytes("RGB", meta["size"], buf))
    return frames, meta


def build_comparison(run, level_idx, ckpt_best, out, size=(480, 270), fps=20):
    """Left = untrained (step 0) clip, right = best checkpoint. Loops to the longer one."""
    vdir = os.path.join(run, "videos")
    early = None
    for fn in sorted(os.listdir(vdir)):
        if fn.startswith("step_0000000"):
            early = os.path.join(vdir, fn)
    if early is None:
        print("  no untrained clip; skipping comparison")
        return
    level = make_levels()[level_idx]
    a, _ = read_clip_frames(early)
    best_path = os.path.join(run, "videos", "_best_tmp.mp4")
    agent, _extra = load_agent(ckpt_best)
    render_episode(agent, level, EnvConfig(), best_path, caption="trained", size=size)
    b, _ = read_clip_frames(best_path)
    n = max(len(a), len(b))
    W = size[0] * 2 + 8
    head = 40
    vw = VideoWriter(out, size=(W, head + size[1]), fps=fps, crf=22)
    print(f"  comparison: {len(a)} vs {len(b)} frames -> {out}")
    try:
        for i in range(n):
            img = Image.new("RGB", (W, head + size[1]), (18, 18, 20))
            dr = ImageDraw.Draw(img)
            dr.text((10, 6), f"Elasto Mania RL — untrained (random policy) vs trained policy "
                             f"({os.path.basename(level.name)})", fill=(240, 240, 240))
            for k, (clip, label) in enumerate(((a, "UNTRAINED (step 0)"), (b, "TRAINED (best checkpoint)"))):
                cx = k * (size[0] + 8)
                if i < len(clip):
                    img.paste(clip[i].resize(size, Image.LANCZOS), (cx, head))
                elif clip:
                    img.paste(clip[-1].resize(size, Image.LANCZOS), (cx, head))
                dr.text((cx + 8, head + 6), label, fill=(255, 220, 120))
                dr.rectangle([cx, head, cx + size[0] - 1, head + size[1] - 1], outline=(70, 70, 75))
            vw.write(np.asarray(img, dtype=np.uint8))
    finally:
        vw.close()
    os.remove(best_path)
    print(f"  wrote {out}")


def build_run(run, with_curves=True, with_grid=True, with_comparison=True, grid="3x2"):
    level_idx = want(run)
    level = make_levels()[level_idx]
    ck = os.path.join(run, "checkpoints")
    best = os.path.join(ck, "best.pt")
    final = os.path.join(ck, "final.pt")
    ckpt = best if os.path.exists(best) else (final if os.path.exists(final) else None)
    if ckpt is None:
        print(f"{run}: no checkpoint yet")
        return
    print(f"=== {run}  (level {level_idx}: {level.name}), using {os.path.basename(ckpt)}")

    agent, extra = load_agent(ckpt)
    print(f"  checkpoint step: {extra.get('step')}")

    # 1. best-policy episode
    out_best = os.path.join(run, "best_episode.mp4")
    st = render_episode(agent, level, EnvConfig(), out_best,
                        caption=f"trained SAC agent - {level.name}",
                        extra_lines=[f"checkpoint step {extra.get('step')}",
                                     "greedy (deterministic) policy"])
    print(f"  best_episode.mp4  finish={st['finish']} prog={st['progress']:.2f} "
          f"t={st['time']:.2f}s")

    # 2. progress grid
    if with_grid:
        os.system(f"python3 make_progress_video.py --run {run} --grid {grid} "
                  f"--out {run}/training_progress.mp4 >/dev/null 2>&1")

    # 3. curves
    if with_curves:
        os.system(f"python3 plot_curves.py --run {run} >/dev/null 2>&1")

    # 4. comparison
    if with_comparison:
        build_comparison(run, level_idx, ckpt, os.path.join(run, "comparison.mp4"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--no-grid", action="store_true")
    ap.add_argument("--no-compare", action="store_true")
    a = ap.parse_args()
    if a.all:
        for i in range(len(make_levels())):
            build_run(f"runs/level{i}", with_grid=not a.no_grid,
                      with_comparison=not a.no_compare)
    else:
        build_run(a.run, with_grid=not a.no_grid, with_comparison=not a.no_compare)
