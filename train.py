"""
train.py — train a Soft Actor-Critic agent (TMRL's Spinup SAC recipe) to ride
Elasto Mania levels through the batched physics environment.

Run:  python3 train.py --level 0 --envs 64 --steps 40000 --outdir runs/run1
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from collections import deque

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from elma_env import (ElmaEnv, EnvConfig, N_ACTIONS, OBS_DIM, CTRL_DT,  # noqa: E402
                      VideoWriter)
from elma_levels import make_levels  # noqa: E402
from sac import SAC, ReplayBuffer  # noqa: E402


def evaluate(agent: SAC, level, cfg: EnvConfig, seeds=(1234,), max_steps=None, render=False):
    """Greedy rollout of full episodes; returns per-episode stats."""
    out = []
    for s in seeds:
        env = ElmaEnv(level, B=1, seed=s, cfg=cfg)
        env.reset()
        ep_rew, ep_frames = 0.0, []
        for t in range(max_steps or env.max_steps):
            act = agent.act(env.observe(), deterministic=True)
            obs, r, d, info = env.step(act)
            ep_rew += float(r[0])
            if d[0]:
                break
        out.append(dict(
            finish=bool(info["finish"][0]), dead=bool(info["dead"][0]),
            stuck=bool(info["stuck"][0]), timeout=bool(info["timeout"][0]),
            progress=float(info["progress"][0]), time=float(info["ep_time"][0]),
            reward=ep_rew, steps=t + 1,
        ))
    return out


def render_episode(agent: SAC, level, cfg: EnvConfig, path: str, seed: int = 1234,
                   caption: str = "", extra_lines=None, size=(960, 540), fps: int = 20,
                   max_steps=None):
    """Greedy rollout of one episode, rendered frame by frame into an mp4.

    This is the "training progress video" hook: call it at regular intervals and the
    resulting clips show the same level ridden by gradually better policies.
    """
    env = ElmaEnv(level, B=1, seed=seed, cfg=cfg)
    env.reset()
    vw = VideoWriter(path, size=size, fps=fps)
    ep_rew = 0.0
    info = None
    try:
        for t in range(max_steps or env.max_steps):
            act = agent.act(env.observe(), deterministic=True)
            obs, r, d, info = env.step(act)
            ep_rew += float(r[0])
            vw.write(env.render(0, size=size, caption=caption, extra_lines=extra_lines))
            if d[0]:
                break
    finally:
        vw.close()
    return dict(finish=bool(info["finish"][0]), dead=bool(info["dead"][0]),
                stuck=bool(info["stuck"][0]), timeout=bool(info["timeout"][0]),
                progress=float(info["progress"][0]), time=float(info["ep_time"][0]),
                reward=ep_rew, steps=t + 1, path=path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=40000, help="control steps (each = %d physics ticks)" % 9)
    ap.add_argument("--upd-per-step", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="runs/run")
    ap.add_argument("--ckpt-every", type=int, default=1500, help="control steps between checkpoints")
    ap.add_argument("--eval-every", type=int, default=600, help="control steps between evaluations")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--reward-scale", type=float, default=0.2)
    ap.add_argument("--crash-penalty", type=float, default=None)
    ap.add_argument("--stuck-penalty", type=float, default=None)
    ap.add_argument("--timeout-penalty", type=float, default=None)
    ap.add_argument("--alpha-min", type=float, default=1e-4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--max-minutes", type=float, default=0.0, help="stop after this many minutes (0 = no limit)")
    ap.add_argument("--video-every", type=int, default=0,
                    help="render a greedy episode video every N control steps (0 = off)")
    ap.add_argument("--video-size", type=str, default="960x540")
    ap.add_argument("--video-fps", type=int, default=20)
    ap.add_argument("--video-seed", type=int, default=1234)
    args = ap.parse_args()

    os.makedirs(os.path.join(args.outdir, "checkpoints"), exist_ok=True)
    if args.video_every:
        os.makedirs(os.path.join(args.outdir, "videos"), exist_ok=True)
    torch.set_num_threads(args.threads)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    levels = make_levels()
    level = levels[args.level]
    cfg = EnvConfig()
    cfg.reward_scale = args.reward_scale
    for name in ("crash_penalty", "stuck_penalty", "timeout_penalty"):
        v = getattr(args, name)
        if v is not None:
            setattr(cfg, name, v)

    env = ElmaEnv(level, B=args.envs, seed=args.seed, cfg=cfg)
    agent = SAC(obs_dim=OBS_DIM, n_actions=N_ACTIONS, hidden=(args.hidden, args.hidden),
                batch_size=args.batch_size, lr=args.lr, pi_loss_mode="exact",
                alpha_min=args.alpha_min)
    buf = ReplayBuffer(OBS_DIM, size=600_000, seed=args.seed)
    start_step = 0
    if args.resume and os.path.exists(args.resume):
        extra = agent.load(args.resume)
        start_step = extra.get("step", 0)
        print(f"resumed from {args.resume} at step {start_step}")

    best_progress = -1.0
    best_eval = None
    best_score = (-1, 0.0)
    stats_path = os.path.join(args.outdir, "stats.jsonl")
    eval_path = os.path.join(args.outdir, "eval.jsonl")
    cfg_path = os.path.join(args.outdir, "config.json")
    with open(cfg_path, "w") as f:
        json.dump(vars(args) | dict(level=level.name, obs_dim=OBS_DIM, n_actions=N_ACTIONS), f, indent=2)

    # ---- bookkeeping ----
    ep_finish = deque(maxlen=100)
    ep_dead = deque(maxlen=100)
    ep_stuck = deque(maxlen=100)
    ep_timeout = deque(maxlen=100)
    ep_prog = deque(maxlen=100)
    ep_rew = deque(maxlen=100)
    ep_len = deque(maxlen=100)
    recent = dict(q_loss=0.0, pi_loss=0.0, entropy=0.0)

    obs = env.observe()
    t0 = time.time()
    last_log = t0
    n_env_steps = start_step * args.envs
    stop = {"flag": False}

    def _sigterm(*_a):
        stop["flag"] = True
        print("\nSIGTERM: finishing up and saving...")

    signal.signal(signal.SIGTERM, _sigterm)

    vsize = tuple(int(v) for v in args.video_size.lower().split("x"))

    def record_video(step_no: int, tag: str = ""):
        vpath = os.path.join(args.outdir, "videos", f"step_{step_no:07d}{tag}.mp4")
        lines = [f"training step {step_no:,}",
                 f"env steps {step_no * args.envs:,}",
                 f"updates {agent.updates:,}"]
        st = render_episode(agent, level, cfg, vpath, seed=args.video_seed,
                            caption=f"step {step_no:,}", extra_lines=lines,
                            size=vsize, fps=args.video_fps)
        print(f"    video @{step_no}: {tag or 'eval'} prog={st['progress']:.1f} "
              f"t={st['time']:.1f} {'FIN' if st['finish'] else ''}"
              f"{'CRASH' if st['dead'] else ''} -> {vpath}")
        return st

    if args.video_every and start_step == 0:
        record_video(0, tag="_untrained")

    for step in range(start_step, args.steps):
        act = agent.act(obs)
        obs2, rew, done, info = env.step(act)
        buf.add_batch(obs, act, rew, obs2, done.astype(np.float32))
        obs = obs2

        for i in np.where(done)[0]:
            ep_finish.append(bool(info["finish"][i]))
            ep_dead.append(bool(info["dead"][i]))
            ep_stuck.append(bool(info["stuck"][i]))
            ep_timeout.append(bool(info["timeout"][i]))
            ep_prog.append(float(info["progress"][i]))
            ep_rew.append(float(info["ep_reward"][i]))
            ep_len.append(float(info["ep_time"][i] / CTRL_DT))

        if buf.n > 20000:
            for _ in range(args.upd_per_step):
                recent = agent.update(buf)

        if (step + 1) % args.log_every == 0:
            now = time.time()
            el = now - t0
            rec = dict(
                step=step + 1, env_steps=(step + 1) * args.envs, updates=agent.updates,
                wall_s=round(el, 1),
                env_steps_per_s=round((step + 1 - start_step) * args.envs / el, 1),
                upd_per_s=round(agent.updates / el, 1),
                episodes=len(ep_finish),
                finish_rate=round(float(np.mean(ep_finish)), 3) if ep_finish else None,
                crash_rate=round(float(np.mean(ep_dead)), 3) if ep_dead else None,
                stuck_rate=round(float(np.mean(ep_stuck)), 3) if ep_stuck else None,
                timeout_rate=round(float(np.mean(ep_timeout)), 3) if ep_timeout else None,
                mean_progress=round(float(np.mean(ep_prog)), 2) if ep_prog else None,
                mean_reward=round(float(np.mean(ep_rew)), 2) if ep_rew else None,
                mean_ep_len=round(float(np.mean(ep_len)), 1) if ep_len else None,
                alpha=round(agent.alpha, 4),
                q_loss=round(recent.get("q_loss", 0.0), 3),
                pi_loss=round(recent.get("pi_loss", 0.0), 3),
                entropy=round(recent.get("entropy", 0.0), 3),
            )
            with open(stats_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[{rec['wall_s']:7.1f}s] step {rec['step']:6d} env {rec['env_steps']:9,d} "
                  f"upd {rec['updates']:7,d} | finish {rec['finish_rate']} crash {rec['crash_rate']} "
                  f"stuck {rec['stuck_rate']} | prog {rec['mean_progress']} rew {rec['mean_reward']} "
                  f"| alpha {rec['alpha']} H {rec['entropy']} | {rec['env_steps_per_s']:.0f} env/s {rec['upd_per_s']:.0f} upd/s")
            last_log = now

        if (step + 1) % args.eval_every == 0:
            ev = evaluate(agent, level, cfg, seeds=(1234, 999))
            with open(eval_path, "a") as f:
                for e in ev:
                    e["step"] = step + 1
                    e["env_steps"] = (step + 1) * args.envs
                    f.write(json.dumps(e) + "\n")
            print(f"    eval @{step+1}: " + " | ".join(
                f"{'FIN' if e['finish'] else 'run'} prog={e['progress']:.1f} t={e['time']:.1f} "
                f"{'CRASH' if e['dead'] else ''}{'STUCK' if e['stuck'] else ''}" for e in ev))
            # "best" = the most reliable, fastest finisher. Progress-along-the-line is
            # NOT a good criterion on its own: a slow, wobbly ride that hugs the
            # reference line accumulates more path progress than a fast clean finish
            # that cuts corners. So rank on (finish count, then completion time).
            n_fin = sum(1 for e in ev if e["finish"])
            fin_times = [e["time"] for e in ev if e["finish"]]
            mean_t = float(np.mean(fin_times)) if fin_times else float("inf")
            mean_prog = float(np.mean([e["progress"] for e in ev]))
            score = (n_fin, -mean_t)
            if score > best_score:
                best_score = score
                best_progress = mean_prog
                best_eval = n_fin
                agent.save(os.path.join(args.outdir, "checkpoints", "best.pt"),
                           extra=dict(step=step + 1, env_steps=(step + 1) * args.envs,
                                      level=level.name, eval=ev))
                print(f"    new best: {n_fin}/{len(ev)} finishes, mean time "
                      f"{mean_t if np.isfinite(mean_t) else float('nan'):.2f}s")
            if args.video_every and (step + 1) % args.video_every == 0:
                record_video(step + 1)

        if (step + 1) % args.ckpt_every == 0:
            p = os.path.join(args.outdir, "checkpoints", f"ckpt_{step+1:07d}.pt")
            agent.save(p, extra=dict(step=step + 1, env_steps=(step + 1) * args.envs,
                                     level=level.name))
            # keep the newest 12 numbered checkpoints (never delete best.pt/final.pt)
            cdir = os.path.join(args.outdir, "checkpoints")
            cks = sorted(f for f in os.listdir(cdir) if f.startswith("ckpt_"))
            for stale in cks[:-12]:
                os.remove(os.path.join(cdir, stale))

        if args.max_minutes and (time.time() - t0) / 60.0 > args.max_minutes:
            print(f"time budget reached after {(time.time()-t0)/60.0:.1f} min")
            break
        if stop["flag"]:
            break

    p = os.path.join(args.outdir, "checkpoints", "final.pt")
    agent.save(p, extra=dict(step=step + 1, env_steps=(step + 1) * args.envs, level=level.name))
    print(f"saved final checkpoint to {p}")


if __name__ == "__main__":
    main()
