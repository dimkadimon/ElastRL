"""
scripted_baseline.py — a hand-written feedback controller used as a *solvability
reference* for each hand-made level.

It is deliberately simple and uses only quantities the RL agent can also observe
(forward speed, pitch, wheel spin, wheel contact):

  * throttle duty-cycle dithering: full throttle burns out / flips the bike over
    backwards, so gas is applied on a fraction of the control steps to hold a target
    speed (this is exactly what a human does with the keyboard).
  * air control: while neither wheel touches the ground, rotate the bike toward the
    attitude of the terrain ahead so the landing is flat.
  * ground leveling: on the ground, gently counter-rotate to keep the chassis level.

Run:
    python3 scripted_baseline.py            # all levels, 3 seeds each
"""

from __future__ import annotations

import sys

import numpy as np

from elma_env import ElmaEnv, CTRL_DT
from elma_levels import make_levels


def rollout(level, seed=0, v_target=7.5, verbose=False):
    env = ElmaEnv(level, B=1, seed=seed)
    env.reset()
    m = env.motor
    obs = env.observe()
    hist = []
    for t in range(env.max_steps):
        # ---- proprioception (all of this is inside the agent's observation too) ----
        i1 = np.array([np.cos(m.alfa[0]), np.sin(m.alfa[0])])
        fwd = -i1
        v_fwd = float(np.dot(m.v[0], fwd))                 # forward speed (m/s)
        vy = float(np.dot(m.v[0], np.array([-fwd[1], fwd[0]])))   # speed in the bike frame
        pitch = float(m.alfa[0])
        # wheel spin is *always* non-zero while rolling, so contact must come from the
        # wheel-contact booleans of the observation (indices 26/27), not from the spin.
        contact2, contact4 = obs[0, 26] > 0.5, obs[0, 27] > 0.5
        airborne = not (contact2 or contact4)

        if airborne:
            # level the bike for the landing: nose up -> rotate forward, nose down -> back
            a = 4 if pitch < -0.28 else (3 if pitch > 0.28 else 1)
        else:
            if v_fwd > v_target * 1.75:
                a = 2                                     # way too fast -> brake
            elif v_fwd < v_target and (t % 3) != 0:
                a = 1                                     # throttle dither (gas 2 of 3)
            else:
                a = 0
        obs, r, d, info = env.step(np.array([a]))
        hist.append((float(info["progress"][0]), float(info["ep_time"][0])))
        if verbose and t % 20 == 0:
            print(f"   t={info['ep_time'][0]:5.2f} prog={info['progress'][0]:6.2f} "
                  f"v={v_fwd:5.2f} pitch={pitch:5.2f} air={airborne}")
        if d[0]:
            break
    return dict(finish=bool(info["finish"][0]), dead=bool(info["dead"][0]),
                stuck=bool(info["stuck"][0]), timeout=bool(info["timeout"][0]),
                progress=float(info["progress"][0]), time=float(info["ep_time"][0]),
                actions=t + 1)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=-1)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    levels = make_levels()
    idxs = range(len(levels)) if args.level < 0 else [args.level]
    for i in idxs:
        L = levels[i]
        print(f"=== {L.name}  (path {len(L.reference_path()) * 0.25:.1f} m)")
        for seed in (0, 1, 2):
            st = rollout(L, seed=seed, verbose=args.verbose)
            print(f"   seed {seed}: finish={st['finish']} dead={st['dead']} stuck={st['stuck']} "
                  f"prog={st['progress']:6.2f} t={st['time']:5.2f} actions={st['actions']}")
