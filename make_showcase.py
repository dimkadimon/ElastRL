"""
make_showcase.py — the headline artefact: one video showing the agent getting better
with training time, across all three levels.

Layout: one column per level, one row per training progress snapshot
(early / mid / late / final checkpoints of that run). Every cell plays that
checkpoint's greedy episode synchronised in episode time, so the columns visibly
converge from "falls over immediately" to "rides the level".

Usage:
    python3 make_showcase.py --runs runs/level0 runs/level1 runs/level2 \
                             --out showcase.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
from PIL import Image, ImageDraw

import imageio_ffmpeg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from elma_env import VideoWriter  # noqa: E402
from elma_levels import make_levels  # noqa: E402


def clips_of(run):
    vdir = os.path.join(run, "videos")
    out = []
    for fn in sorted(os.listdir(vdir)):
        m = re.match(r"step_(\d+)\.mp4$", fn)
        if m:
            out.append((int(m.group(1)), os.path.join(vdir, fn)))
    out.sort()
    return out


def run_totals(run):
    """step of the final eval, so 'late' means late in *this* run rather than a fixed step."""
    p = os.path.join(run, "eval.jsonl")
    mx = 0
    if os.path.exists(p):
        with open(p) as f:
            for line in f:
                try:
                    mx = max(mx, json.loads(line).get("step", 0))
                except json.JSONDecodeError:
                    pass
    return mx


def pick_rows(runs, rows=3):
    """For each run pick `rows` clips spread over its training (always including the last)."""
    sel = {}
    for run in runs:
        cl = clips_of(run)
        if not cl:
            sel[run] = []
            continue
        if len(cl) <= rows:
            sel[run] = cl
            continue
        idx = np.linspace(0, len(cl) - 1, rows).round().astype(int)
        sel[run] = [cl[i] for i in sorted(set(idx.tolist()))]
    # pad short columns with their own last clip so the grid stays rectangular
    nrow = max(len(v) for v in sel.values())
    for run in runs:
        while len(sel[run]) < nrow:
            sel[run].append(sel[run][-1])
    return sel, nrow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["runs/level0", "runs/level1", "runs/level2"])
    ap.add_argument("--out", default="showcase.mp4")
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--cell", type=int, default=520)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--max-sec", type=float, default=16.0)
    a = ap.parse_args()

    levels = make_levels()
    sel, nrow = pick_rows(a.runs, a.rows)
    ncol = len(a.runs)

    readers = {}
    for ci, run in enumerate(a.runs):
        for ri, (step, path) in enumerate(sel[run]):
            try:
                gen = imageio_ffmpeg.read_frames(path, pix_fmt="rgb24")
                meta = next(gen)
            except Exception as e:      # a truncated clip should not kill the build
                print(f"  skip {path}: {e}")
                continue
            readers[(ci, ri)] = dict(gen=gen, size=meta["size"], last=None, done=False,
                                     step=step, run=run)

    cw, ch = a.cell, int(round(a.cell * 9 / 16))
    head = 58
    W, H = ncol * cw, head + nrow * ch
    vw = VideoWriter(a.out, size=(W, H), fps=a.fps, crf=22)
    print(f"{ncol}x{nrow} cells -> {W}x{H} {a.out}")

    names = [os.path.basename(r.rstrip("/")) for r in a.runs]
    titles = [levels[i].name if i < len(levels) else names[i] for i in range(ncol)]

    k = 0
    max_frames = int(a.max_sec * a.fps)
    frozen_for = 0
    try:
        while k < max_frames:
            if all(r["done"] for r in readers.values()):
                frozen_for += 1
                if frozen_for > a.fps * 2:      # 2 s of frozen tail, then stop
                    break
            img = Image.new("RGB", (W, H), (16, 16, 19))
            dr = ImageDraw.Draw(img)
            dr.text((14, 8), "Elasto Mania RL — SAC learning three levels, in training order",
                    fill=(245, 245, 248))
            dr.text((14, 26), "each cell: the deterministic policy of a checkpoint riding the level "
                              "(same start, synchronised in time); later rows = more training",
                    fill=(150, 150, 158))
            dr.text((W - 120, 8), f"t = {k / a.fps:4.1f}s", fill=(245, 245, 248))
            for ci, title in enumerate(titles):
                dr.text((ci * cw + 10, 40), f"{title}", fill=(255, 214, 90))
            for (ci, ri), r in readers.items():
                if not r["done"]:
                    try:
                        buf = next(r["gen"])
                        r["last"] = Image.frombytes("RGB", r["size"], buf).resize(
                            (cw, ch), Image.LANCZOS)
                    except StopIteration:
                        r["done"] = True
                cx, cy = ci * cw, head + ri * ch
                img.paste(r["last"] or Image.new("RGB", (cw, ch), (0, 0, 0)), (cx, cy))
                dr.rectangle([cx, cy, cx + cw - 1, cy + ch - 1], outline=(66, 66, 72))
                dr.rectangle([cx, cy + ch - 18, cx + 165, cy + ch - 1], fill=(0, 0, 0))
                dr.text((cx + 5, cy + ch - 15), f"step {r['step']:,}", fill=(240, 240, 240))
            vw.write(np.asarray(img, dtype=np.uint8))
            k += 1
    finally:
        vw.close()
    print(f"done: {a.out} ({k} frames, {k / a.fps:.1f}s)")


if __name__ == "__main__":
    main()
