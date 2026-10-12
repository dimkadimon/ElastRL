"""
plot_curves.py — turn a run's stats.jsonl / eval.jsonl into training-curve figures:

  * training_curves.png  : static multi-panel summary used in the README
  * training_curves.mp4  : animated version ("the agent getting better with time"),
                           with a playhead sweeping across training steps

Usage:
    python3 plot_curves.py --run runs/level0
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

import imageio_ffmpeg

plt.rcParams.update({"figure.facecolor": "#111114", "axes.facecolor": "#111114",
                     "axes.edgecolor": "#55555c", "axes.labelcolor": "#e8e8ec",
                     "text.color": "#e8e8ec", "xtick.color": "#b8b8c0",
                     "ytick.color": "#b8b8c0", "grid.color": "#2a2a30",
                     "font.size": 9})


def load(run):
    stats, evals = [], []
    with open(os.path.join(run, "stats.jsonl")) as f:
        for line in f:
            try:
                stats.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    p = os.path.join(run, "eval.jsonl")
    if os.path.exists(p):
        with open(p) as f:
            for line in f:
                try:
                    evals.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return stats, evals


def rolling(x, w):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x
    c = np.convolve(x, np.ones(w) / w, mode="valid")
    return np.concatenate([np.full(w - 1, np.nan), c])


PANELS = [
    ("mean_progress", "progress per episode (m, rolling mean)", "#7ec452"),
    ("finish_rate", "episode finish rate (rolling mean)", "#64b5f6"),
    ("crash_rate", "crash rate (rolling mean)", "#ef5350"),
    ("entropy", "policy entropy (nats)", "#ffca28"),
]


def build(run, out_png, out_mp4, window=20):
    stats, evals = load(run)
    if not stats:
        raise SystemExit(f"no stats in {run}")
    es = np.array([s["env_steps"] for s in stats], dtype=float)
    ev = [e for e in evals if "step" in e]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.2), dpi=120)
    fig.suptitle(f"Elasto Mania RL — SAC training curves  ({os.path.basename(run)})",
                 fontsize=13, color="#f0f0f4")
    curves = {}
    for ax, (key, label, color) in zip(axes.ravel(), PANELS):
        vals = [s.get(key) or 0.0 for s in stats]
        r = rolling(vals, window)
        curves[key] = (ax, es, r, color, label)
        if key == "mean_progress" and ev:
            ee = np.array([e["env_steps"] for e in ev], dtype=float)
            ep = np.array([e["progress"] for e in ev], dtype=float)
            ax.scatter(ee, ep, s=9, color="#ffffff", zorder=5, label="greedy eval")
            done = np.array([bool(e["finish"]) for e in ev])
            if done.any():
                ax.scatter(ee[done], ep[done], s=34, marker="*", color="#ffd54f", zorder=6,
                           label="eval finish")
            ax.legend(loc="lower right", fontsize=8, facecolor="#1a1a1f", edgecolor="#55555c")

    def draw(fig_idx_frame):
        for key, (ax, es, r, color, label) in curves.items():
            ax.clear()
            ax.set_facecolor("#111114")
            ax.grid(True, lw=0.5, alpha=0.6)
            ax.plot(es, r, color=color, lw=1.6)
            ax.set_xlabel("environment steps")
            ax.set_ylabel(label)
            if key == "mean_progress":
                ev_e = [e for e in ev if e["env_steps"] <= fig_idx_frame]
                if ev_e:
                    ee = [e["env_steps"] for e in ev_e]
                    ep = [e["progress"] for e in ev_e]
                    ax.scatter(ee, ep, s=9, color="#ffffff", zorder=5)
                    fin = [(e["env_steps"], e["progress"]) for e in ev_e if e["finish"]]
                    if fin:
                        ax.scatter([f[0] for f in fin], [f[1] for f in fin], s=34, marker="*",
                                   color="#ffd54f", zorder=6)
            ax.axvline(fig_idx_frame, color="#ffffff", lw=1.0, alpha=0.55)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        return img

    # static PNG (full data)
    draw(int(es[-1]) if len(es) else 0)
    fig.savefig(out_png, facecolor=fig.get_facecolor())
    print("wrote", out_png)

    # animation
    if out_mp4:
        # x264 needs even pixel dimensions for yuv420p; pad if the figure lands odd.
        writer = FFMpegWriter(fps=15, codec="libx264", bitrate=2600,
                              extra_args=["-pix_fmt", "yuv420p",
                                          "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"],
                              metadata={"title": "Elasto Mania RL training"})
        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        n_frames = 90
        grid = np.linspace(es[0] if len(es) else 0, es[-1] if len(es) else 1, n_frames)
        with writer.saving(fig, out_mp4, dpi=110):
            for g in grid:
                draw(int(g))
                writer.grab_frame(facecolor=fig.get_facecolor())
        print("wrote", out_mp4)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/level0")
    ap.add_argument("--png", default=None)
    ap.add_argument("--mp4", default=None, help="pass '' to skip the animation")
    ap.add_argument("--window", type=int, default=20)
    a = ap.parse_args()
    build(a.run,
          a.png or os.path.join(a.run, "training_curves.png"),
          a.mp4 if a.mp4 is not None else os.path.join(a.run, "training_curves.mp4"),
          a.window)
