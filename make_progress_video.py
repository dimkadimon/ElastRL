"""
make_progress_video.py — stitch the per-checkpoint episodes recorded during training
into a single synchronised "the agent getting better over time" grid video.

Each cell plays one checkpoint's greedy episode (the same level, same start), all
synchronised on episode time, so the difference between an early random policy and a
late trained policy is directly visible. Cells that crashed/stalled simply freeze on
their last frame and get a red tag.

Usage:
    python3 make_progress_video.py --run runs/level0 --out runs/level0/training_progress.mp4
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
from PIL import Image, ImageDraw

import imageio_ffmpeg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from elma_env import VideoWriter  # noqa: E402


def discover_clips(run_dir: str):
    vdir = os.path.join(run_dir, "videos")
    clips = []
    for fn in sorted(os.listdir(vdir)):
        m = re.match(r"step_(\d+)(?:_(\w+))?\.mp4$", fn)
        if m:
            clips.append(dict(step=int(m.group(1)), tag=m.group(2) or "", path=os.path.join(vdir, fn)))
    clips.sort(key=lambda c: (c["step"], c["tag"]))
    return clips


def pick_clips(clips, n):
    """Evenly spaced selection over training steps, always including the very first clip."""
    if len(clips) <= n:
        return clips
    idx = np.linspace(0, len(clips) - 1, n).round().astype(int)
    idx = sorted(set(idx.tolist()))
    return [clips[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/level0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--grid", default="3x2", help="cols x rows")
    ap.add_argument("--cell", type=int, default=640, help="cell width in px (16:9)")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--max-sec", type=float, default=22.0)
    ap.add_argument("--title", default="Elasto Mania RL - SAC learning to ride \"Warm Up\"")
    ap.add_argument("--n-cells", type=int, default=0, help="how many clips to show (0 = cols*rows)")
    args = ap.parse_args()

    cols, rows = (int(v) for v in args.grid.lower().split("x"))
    n = args.n_cells or cols * rows
    out = args.out or os.path.join(args.run, "training_progress.mp4")

    clips = discover_clips(args.run)
    if not clips:
        raise SystemExit(f"no clips found under {args.run}/videos")
    sel = pick_clips(clips, n)
    print("clips:", [(c["step"], c["tag"]) for c in sel])

    # open one streaming decoder per clip
    readers = []
    for c in sel:
        gen = imageio_ffmpeg.read_frames(c["path"], pix_fmt="rgb24")
        meta = next(gen)
        readers.append(dict(gen=gen, size=meta["size"], done=False, last=None, step=c["step"],
                            tag=c["tag"], n_frames=int(meta["duration"] * meta["fps"])))

    cw, ch = args.cell, int(round(args.cell * 9 / 16))
    head = 46
    W, H = cols * cw, head + rows * ch
    vw = VideoWriter(out, size=(W, H), fps=args.fps, crf=22)
    print(f"writing {W}x{H} -> {out}")

    k = 0
    max_frames = int(args.max_sec * args.fps)
    try:
        while k < max_frames:
            if all(r["done"] for r in readers):
                # keep playing a couple of seconds of frozen frames, then stop
                if k > 0 and k % (args.fps * 2) == 0:
                    break
            img = Image.new("RGB", (W, H), (18, 18, 20))
            dr = ImageDraw.Draw(img)
            dr.text((14, 8), args.title, fill=(240, 240, 240))
            dr.text((14, 26), f"cells: greedy episode at successive checkpoints, synchronised in time",
                    fill=(160, 160, 165))
            dr.text((W - 150, 8), f"t = {k / args.fps:4.1f}s", fill=(240, 240, 240))
            for i, r in enumerate(readers):
                if not r["done"]:
                    try:
                        buf = next(r["gen"])
                        frame = Image.frombytes("RGB", r["size"], buf).resize((cw, ch), Image.LANCZOS)
                        r["last"] = frame
                    except StopIteration:
                        r["done"] = True
                cx = (i % cols) * cw
                cy = head + (i // cols) * ch
                cell = r["last"] or Image.new("RGB", (cw, ch), (0, 0, 0))
                img.paste(cell, (cx, cy))
                # cell frame + label bar
                dr.rectangle([cx, cy, cx + cw - 1, cy + ch - 1], outline=(70, 70, 75))
                label = f"training step {r['step']:,}" + (" (untrained)" if r["tag"] == "untrained" else "")
                dr.rectangle([cx, cy + ch - 20, cx + 250, cy + ch - 1], fill=(0, 0, 0))
                dr.text((cx + 6, cy + ch - 16), label, fill=(255, 255, 255))
                if r["done"]:
                    # the clip ran out; whether that was a finish, a crash or a stall is
                    # recorded in the run's eval.jsonl, so just report the fact here.
                    done_txt = "episode over"
                    tw = dr.textlength(done_txt)
                    dr.rectangle([cx + cw / 2 - tw / 2 - 6, cy + ch / 2 - 12, cx + cw / 2 + tw / 2 + 6, cy + ch / 2 + 12],
                                 fill=(0, 0, 0))
                    dr.text((cx + cw / 2 - tw / 2, cy + ch / 2 - 8), done_txt, fill=(255, 200, 90))
            vw.write(np.asarray(img, dtype=np.uint8))
            k += 1
    finally:
        vw.close()
    print(f"done: {out} ({k} frames, {k / args.fps:.1f}s)")


if __name__ == "__main__":
    main()
