"""
elma_env.py — batched RL environment around the Elasto Mania physics port,
plus an Elma-style software renderer used for the training videos.

Design notes (mirroring TMRL, github.com/trackmania-rl/tmrl):
  * Observation is a LIDAR-style vector: 19 range measurements around the bike
    (as in TMRL's LIDAR interface: speed + lidar), plus a small proprioceptive
    state block (speeds, wheel spin, body attitude, contacts, progress).
  * Actions are discrete key combinations, like TMRL's discrete ACT_BUF action set.
  * The reward is TMRL's "progress along a reference trajectory" reward: the agent
    is rewarded for advancing along a reference riding line, cannot rewind it for
    free, is terminated when it stalls (failure counter), and cannot cheat by
    cutting far away from the line (max_dist_from_traj).
  * Episodes end on: crash (head touches terrain), reaching the flower (finish),
    timeout, or stalling.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw

import elma_physics as ph
from elma_levels import Level, make_levels

# ---------------------------------------------------------------------------- actions
# The real game has 4 keys: accelerate (gaz), brake (fek), rotate-1 (ugrik1),
# rotate-2 (ugrik2). We use the 9 sensible combinations as the action set.
ACTIONS = [
    (0, 0, 0, 0),  # 0 coast
    (1, 0, 0, 0),  # 1 accelerate
    (0, 1, 0, 0),  # 2 brake
    (0, 0, 1, 0),  # 3 rotate 1
    (0, 0, 0, 1),  # 4 rotate 2
    (1, 0, 1, 0),  # 5 accelerate + rotate 1
    (1, 0, 0, 1),  # 6 accelerate + rotate 2
    (0, 1, 1, 0),  # 7 brake + rotate 1
    (0, 1, 0, 1),  # 8 brake + rotate 2
]
N_ACTIONS = len(ACTIONS)
ACTION_TABLE = np.array(ACTIONS, dtype=np.int64)  # (A,4)

# observation layout
N_RAYS = 19
LIDAR_RANGE = 8.0
OBS_DIM = N_RAYS + 11

SUBSTEPS = 9  # 9 * 0.0055 s = 0.0495 s per control step (~20 Hz, like TMRL's 20 Hz)
CTRL_DT = SUBSTEPS * ph.DT


@dataclass
class EnvConfig:
    max_dist_from_traj: float = 6.0   # TMRL max_dist_from_traj (in metres)
    nb_obs_forward: int = 60          # how far ahead (in path samples) a cut may jump
    nb_obs_backward: int = 24         # rewind window (in path samples)
    min_steps_before_failure: int = 30
    nb_zero_rew_before_failure: int = 45
    finish_bonus: float = 30.0
    crash_penalty: float = 5.0
    # A policy that simply stands still never crashes and never stalls into a wall, so
    # without a cost for "gave up" the optimum on a hard level is to brake and wait.
    # These make every form of not-trying strictly worse than trying.
    stuck_penalty: float = 0.0
    timeout_penalty: float = 0.0
    apple_bonus: float = 1.0
    step_cost: float = 0.005
    apple_radius: float = 0.55
    # Finish-test radius around the goal flower. The check is done against the bike's
    # *body centre*, so the radius must cover the flower plus half the bike: the drawn
    # flower is ~0.85 m across at the level scale and the bike is ~1 m long. 1.15 m was
    # the first guess and turned out to be razor-thin - on the hardest level the agent
    # rode the entire route but flew a few centimetres over the flower and was scored
    # as a stall. 1.5 m is the calibrated value; see README section 5.
    goal_radius: float = 1.5
    reward_scale: float = 0.2


class ElmaEnv:
    """A vectorised Elasto Mania environment (B parallel worlds sharing one level)."""

    def __init__(self, level: Level, B: int = 64, seed: int = 0,
                 cfg: EnvConfig | None = None):
        self.level = level
        self.ter = level.terrain
        self.cfg = cfg or EnvConfig()
        self.B = B
        self.rng = np.random.default_rng(seed)
        self.path = level.reference_path(ds=0.25)      # (P,2) physics coords
        self.path_ds = 0.25
        self.n_path = len(self.path)
        self.goal = level.goal_pos()
        self.apples = level.apple_positions()
        self.n_apples = len(self.apples)
        self.max_steps = int(level.max_time / CTRL_DT)

        self.motor = ph.Motor.zeros(B)
        self.obs = np.zeros((B, OBS_DIM), dtype=np.float32)
        self.cur_idx = np.zeros(B, dtype=np.int64)
        self.step_counter = np.zeros(B, dtype=np.int64)
        self.failure_counter = np.zeros(B, dtype=np.int64)
        self.ep_time = np.zeros(B)
        self.ep_progress = np.zeros(B)      # distance progressed (m)
        self.ep_reward = np.zeros(B)
        self.alive = np.zeros(B, dtype=bool)
        self.apple_mask = np.ones((B, max(self.n_apples, 1)), dtype=bool)
        self.last_dead = np.zeros(B, dtype=bool)
        self.last_finish = np.zeros(B, dtype=bool)
        self.last_stuck = np.zeros(B, dtype=bool)
        self.last_timeout = np.zeros(B, dtype=bool)
        self.reset()

    # ------------------------------------------------------------------ reset
    def reset(self, idx=None):
        m = self.motor
        if idx is None:
            idx = np.arange(self.B)
            self.cur_idx[:] = 0
            self.ep_time[:] = 0.0
            self.ep_progress[:] = 0.0
            self.ep_reward[:] = 0.0
            self.apple_mask[:] = True
        idx = np.atleast_1d(idx)
        n = len(idx)
        pose = self.level.start_pose()
        jitter_x = self.rng.uniform(-0.25, 0.25, n)
        jitter_a = self.rng.uniform(-0.02, 0.02, n)

        m.alfa[idx] = jitter_a
        m.omega[idx] = 0.0
        m.v[idx] = 0.0
        m.a2[idx] = jitter_a
        m.o2[idx] = 0.0
        m.v2[idx] = 0.0
        m.a4[idx] = jitter_a
        m.o4[idx] = 0.0
        m.v4[idx] = 0.0
        dx = jitter_x  # shift the whole pose along x
        m.r[idx] = pose["r"][None, :] + np.stack([dx, np.zeros(n)], axis=-1)
        m.r2[idx] = pose["r2"][None, :] + np.stack([dx, np.zeros(n)], axis=-1)
        m.r4[idx] = pose["r4"][None, :] + np.stack([dx, np.zeros(n)], axis=-1)
        m.vr[idx] = pose["vr"][None, :] + np.stack([dx, np.zeros(n)], axis=-1)
        m.vv[idx] = 0.0
        m.t[idx] = 0.0
        m.voltfek[idx] = 0
        m.dfek2[idx] = 0.0
        m.dfek4[idx] = 0.0
        m.ugrasban1[idx] = 0
        m.ugrasban2[idx] = 0
        m.ugras1kezd[idx] = -1.0
        m.ugras2kezd[idx] = -1.0
        m.kezdoomega1[idx] = -1.0
        m.kezdoomega2[idx] = -1.0
        ph.szamitfejr_idx(m, idx)

        self.cur_idx[idx] = 0
        self.step_counter[idx] = 0
        self.failure_counter[idx] = 0
        self.ep_time[idx] = 0.0
        self.ep_progress[idx] = 0.0
        self.ep_reward[idx] = 0.0
        self.alive[idx] = True
        self.apple_mask[idx] = True

    # -------------------------------------------------------------- observation
    def observe(self) -> np.ndarray:
        """LIDAR + proprioception for all worlds, shape (B, OBS_DIM)."""
        m = self.motor
        i1 = np.stack([np.cos(m.alfa), np.sin(m.alfa)], axis=-1)
        j1 = ph.rot90(i1)
        fwd = -i1  # direction of travel

        lidar = ph.observe_lidar(m, self.ter, N_RAYS, 2.1, LIDAR_RANGE)
        _, _, c2 = ph.contact_points(m.r2, ph.R_WHL, self.ter)
        _, _, c4 = ph.contact_points(m.r4, ph.R_WHL, self.ter)

        extras = np.stack([
            ph.dot(m.v, fwd) / 20.0,      # forward speed
            ph.dot(m.v, j1) / 10.0,       # vertical speed (bike frame)
            m.omega / 15.0,               # body spin
            m.o2 / ph.TULPORGES,          # front wheel spin
            m.o4 / ph.TULPORGES,          # rear wheel spin
            np.sin(m.alfa), np.cos(m.alfa),
            (c2 > 0).astype(np.float32), (c4 > 0).astype(np.float32),
            self.cur_idx / max(self.n_path - 1, 1),
            self.ep_time / self.level.max_time,
        ], axis=-1)
        return np.concatenate([lidar, extras], axis=-1).astype(np.float32)

    # --------------------------------------------------------------- stepping
    def step(self, actions: np.ndarray):
        """actions: (B,) ints. Returns (obs, rew, done, info)."""
        cfg = self.cfg
        m, ter = self.motor, self.ter
        acts = ACTION_TABLE[actions]
        gaz = acts[:, 0].astype(np.int64)
        fek = acts[:, 1].astype(np.int64)
        u1 = acts[:, 2].astype(np.int64)
        u2 = acts[:, 3].astype(np.int64)

        rew = np.zeros(self.B)
        done = np.zeros(self.B, dtype=bool)
        self.last_dead[:] = False
        self.last_finish[:] = False
        self.last_stuck[:] = False
        self.last_timeout[:] = False

        for k in range(SUBSTEPS):
            live = self.alive
            if not np.any(live):
                break
            zero = np.zeros(self.B, dtype=np.int64)
            # a world in the middle of a blow-up can emit overflow warnings before the
            # guard below kills it; those are expected and not worth spamming the log.
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                ph.leptet(m,
                          np.where(live, gaz, 0), np.where(live, fek, 0),
                          np.where(live, u1, 0), np.where(live, u2, 0), ph.DT, ter)

            dead_now = ph.vizsgalat_dead(m, ter) & live
            if np.any(dead_now):
                self.alive[dead_now] = False
                self.last_dead[dead_now] = True
                rew[dead_now] -= cfg.crash_penalty

        # Safety guard (cheap, once per control step): any world that leaves the sane
        # state space - a state the real game never reaches, it would have crashed long
        # before - is killed here, so no non-finite value can ever enter replay.
        insane = self.alive & ~(
            np.isfinite(m.r).all(axis=1) & np.isfinite(m.v).all(axis=1)
            & np.isfinite(m.omega) & np.isfinite(m.o2) & np.isfinite(m.o4)
            & (np.abs(m.omega) < 300.0)
            & (np.abs(m.v[:, 0]) < 90.0) & (np.abs(m.v[:, 1]) < 90.0)
            & (np.abs(m.o2) < 500.0) & (np.abs(m.o4) < 500.0))
        if np.any(insane):
            for name in ("r", "r2", "r4", "v", "v2", "v4", "vr", "vv", "fejr"):
                getattr(m, name)[insane] = 0.0
            for name in ("omega", "o2", "o4", "alfa", "a2", "a4", "dfek2", "dfek4"):
                getattr(m, name)[insane] = 0.0
            self.alive[insane] = False
            self.last_dead[insane] = True
            rew[insane] -= cfg.crash_penalty

        self.ep_time[self.alive] += CTRL_DT  # motor.t advances inside leptet()

        # ---- reward: TMRL-style progress along the reference path ----
        pos = m.r
        adv_rew, progressed = self._path_progress(pos)

        # apples
        if self.n_apples:
            d_body = np.linalg.norm(pos[:, None, :] - self.apples[None, :, :], axis=-1)  # (B,M)
            grabbed = (d_body < cfg.apple_radius) & self.apple_mask & self.alive[:, None]
            n_grab = grabbed.sum(axis=1)
            self.apple_mask &= ~grabbed
            rew += n_grab * cfg.apple_bonus
            adv_rew = adv_rew + n_grab * cfg.apple_bonus

        # finish
        d_goal = np.linalg.norm(pos - self.goal[None, :], axis=-1)
        finished = (d_goal < cfg.goal_radius) & self.alive
        if np.any(finished):
            rew[finished] += cfg.finish_bonus
            adv_rew[finished] += cfg.finish_bonus
            self.alive[finished] = False
            self.last_finish[finished] = True

        # failure counter / stall termination (TMRL logic)
        stalled = ~progressed
        self.failure_counter[stalled] += 1
        self.failure_counter[progressed] = 0
        stuck = ((self.failure_counter > cfg.nb_zero_rew_before_failure)
                 & (self.step_counter > cfg.min_steps_before_failure) & self.alive)
        if np.any(stuck):
            self.alive[stuck] = False
            self.last_stuck[stuck] = True
            rew[stuck] -= cfg.stuck_penalty
            adv_rew[stuck] -= cfg.stuck_penalty

        # timeout
        self.step_counter += 1
        timeout = (self.step_counter >= self.max_steps) & self.alive
        if np.any(timeout):
            self.alive[timeout] = False
            self.last_timeout[timeout] = True
            rew[timeout] -= cfg.timeout_penalty
            adv_rew[timeout] -= cfg.timeout_penalty

        # living cost + global reward scaling
        rew = (rew - cfg.step_cost) * cfg.reward_scale
        adv_rew = adv_rew * cfg.reward_scale

        self.ep_reward += adv_rew
        self.ep_progress += np.where(progressed, self.path_ds, 0.0)

        done = ~self.alive
        # capture episode stats *before* the auto-reset of finished/dead worlds
        info = dict(
            dead=self.last_dead.copy(), finish=self.last_finish.copy(),
            stuck=self.last_stuck.copy(), timeout=self.last_timeout.copy(),
            progress=self.ep_progress.copy(), ep_reward=self.ep_reward.copy(),
            ep_time=self.ep_time.copy(), cur_idx=self.cur_idx.copy(),
        )
        if np.any(done):
            self.reset(np.where(done)[0])
        obs = self.observe()
        return obs, rew.astype(np.float32), done, info

    def _path_progress(self, pos: np.ndarray):
        """TMRL RewardFunction.compute_reward port (batched).

        Returns (reward_in_metres, progressed_bool). cur_idx only moves forward
        except for explicit rewinds within nb_obs_backward.
        """
        cfg = self.cfg
        B = pos.shape[0]
        P = self.n_path
        win = cfg.nb_obs_forward
        idx = self.cur_idx[:, None] + np.arange(win)[None, :]           # (B,win)
        idxc = np.clip(idx, 0, P - 1)
        d = np.linalg.norm(pos[:, None, :] - self.path[idxc], axis=-1)
        d = np.where(idx < P, d, np.inf)
        bi = np.argmin(d, axis=1)
        bd = d[np.arange(B), bi]
        new_idx = self.cur_idx + bi
        too_far = bd > cfg.max_dist_from_traj
        new_idx = np.where(too_far, self.cur_idx, new_idx)
        progressed = new_idx > self.cur_idx

        # rewind (find the best index behind) for stalled worlds only
        stall = ~progressed
        if np.any(stall):
            ss = np.where(stall)[0]
            win2 = cfg.nb_obs_backward
            idx2 = self.cur_idx[ss, None] - np.arange(win2)[None, :]
            idx2c = np.clip(idx2, 0, P - 1)
            d2 = np.linalg.norm(pos[ss, None, :] - self.path[idx2c], axis=-1)
            d2 = np.where(idx2 >= 0, d2, np.inf)
            bi2 = np.argmin(d2, axis=1)
            bd2 = d2[np.arange(len(ss)), bi2]
            new2 = self.cur_idx[ss] - bi2
            new2 = np.where(bd2 > cfg.max_dist_from_traj, self.cur_idx[ss], new2)
            new_idx[ss] = new2

        rew = (new_idx - self.cur_idx).astype(np.float64) * self.path_ds
        self.cur_idx = new_idx
        return rew, progressed

    # ------------------------------------------------------------------ render
    def render(self, idx: int = 0, size=(960, 540), caption: str = "",
               trail: bool = True, extra_lines=None, scale: float = 26.0):
        """Render one world in the style of Elasto Mania (PIL, software)."""
        W, H = size
        img = Image.new("RGB", (W, H), (0, 0, 0))
        dr = ImageDraw.Draw(img)
        m = self.motor.index(np.array([idx]))
        cam_x = float(m.r[0, 0])
        cam_y = float(m.r[0, 1])

        def s2p(x, y):
            # physics coords (the bike drives toward -x) -> screen (bike drives right)
            px = W * 0.42 - (x - cam_x) * scale
            py = H * 0.62 - (y - cam_y) * scale
            return px, py

        # ---- terrain: fill the polygon, then draw the grass band + top line ----
        poly = self.level._polygon()
        pts = [s2p(x, y) for x, y in poly]
        dr.polygon(pts, fill=(74, 62, 40))
        # grass strip along the top surface
        prof = poly[:len(self.level.profile)]
        band = 0.45 * scale
        top_pts = [s2p(x, y) for x, y in prof]
        bot_pts = [s2p(x, y - 0.45) for x, y in prof]
        dr.polygon(top_pts + bot_pts[::-1], fill=(58, 112, 46))
        dr.line(top_pts, fill=(126, 196, 82), width=3)

        # ---- start platform edge marker + goal flower ----
        gx, gy = self.goal
        px, py = s2p(gx, gy)
        # flower
        dr.line([(px, py + 18), (px, py)], fill=(60, 150, 60), width=3)
        for k in range(6):
            ang = k * np.pi / 3.0
            fx = px + np.cos(ang) * 11
            fy = py + np.sin(ang) * 11
            dr.ellipse([fx - 7, fy - 7, fx + 7, fy + 7], fill=(250, 214, 60))
        dr.ellipse([px - 6, py - 6, px + 6, py + 6], fill=(240, 120, 40))

        # ---- apples ----
        for j in range(self.n_apples):
            if not self.apple_mask[idx, j]:
                continue
            ax, ay = self.apples[j]
            ax, ay = s2p(ax, ay)
            dr.ellipse([ax - 8, ay - 8, ax + 8, ay + 8], fill=(214, 40, 32))
            dr.line([(ax, ay - 8), (ax + 4, ay - 14)], fill=(90, 60, 20), width=2)

        # ---- bike ----
        def circle(r, color, outline=None, w=2):
            x, y = s2p(r[0], r[1])
            rad = ph.R_WHL * scale
            dr.ellipse([x - rad, y - rad, x + rad, y + rad], fill=color,
                       outline=outline, width=w)
            return x, y, rad

        for wheel, ang in ((m.r2[0], m.a2[0]), (m.r4[0], m.a4[0])):
            x, y, rad = circle(wheel, (28, 28, 32), (150, 150, 160), 2)
            for k in range(4):
                a = ang + k * np.pi / 2
                dr.line([(x, y), (x + np.cos(a) * rad, y - np.sin(a) * rad)],
                        fill=(150, 150, 160), width=2)
        bx, by = s2p(m.r[0, 0], m.r[0, 1])
        w2x, w2y = s2p(m.r2[0, 0], m.r2[0, 1])
        w4x, w4y = s2p(m.r4[0, 0], m.r4[0, 1])
        dr.line([(w2x, w2y), (bx, by), (w4x, w4y)], fill=(210, 210, 215), width=7)
        # rider
        rx, ry = s2p(m.vr[0, 0], m.vr[0, 1])
        hx, hy = s2p(m.fejr[0, 0], m.fejr[0, 1])
        dr.line([(bx, by), (rx, ry)], fill=(60, 60, 70), width=7)
        dr.ellipse([rx - 6, ry - 6, rx + 6, ry + 6], fill=(190, 40, 40))
        dr.ellipse([hx - 8, hy - 8, hx + 8, hy + 8], fill=(190, 40, 40))
        dr.ellipse([hx - 8, hy - 8, hx + 8, hy + 8], outline=(240, 240, 240), width=2)
        dr.line([(rx, ry), (hx, hy)], fill=(150, 30, 30), width=6)

        # ---- HUD ----
        v = m.v[0]
        speed = float(np.hypot(*v))
        t = float(self.ep_time[idx])
        prog_pct = 100.0 * self.cur_idx[idx] / max(self.n_path - 1, 1)
        apples_got = int((~self.apple_mask[idx]).sum()) if self.n_apples else 0
        hud = [
            f"time {t:5.2f}s",
            f"speed {speed * 3.6:5.1f} km/h",
            f"track {prog_pct:5.1f}%",
            f"apples {apples_got}/{self.n_apples}",
        ]
        for i, line in enumerate(hud):
            dr.text((12, 10 + i * 16), line, fill=(235, 235, 235))
        if extra_lines:
            for i, line in enumerate(extra_lines):
                dr.text((12, H - 24 - (len(extra_lines) - 1 - i) * 16), line, fill=(255, 235, 120))
        # progress bar
        dr.rectangle([W - 220, 14, W - 14, 26], outline=(200, 200, 200))
        dr.rectangle([W - 218, 16, W - 218 + int(200 * prog_pct / 100.0), 24], fill=(126, 196, 82))
        if caption:
            tw = dr.textlength(caption)
            dr.rectangle([W / 2 - tw / 2 - 10, H - 44, W / 2 + tw / 2 + 10, H - 14],
                         fill=(0, 0, 0))
            dr.text((W / 2 - tw / 2, H - 36), caption, fill=(255, 255, 255))
        return np.asarray(img, dtype=np.uint8)


# --------------------------------------------------------------------------- video
class VideoWriter:
    """Streams RGB frames to an H.264 mp4 via the ffmpeg bundled with imageio-ffmpeg."""

    def __init__(self, path: str, size=(960, 540), fps: int = 20, crf: int = 20):
        import imageio_ffmpeg
        self.ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        self.size = size
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self.proc = subprocess.Popen([
            self.ffmpeg, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-",
            "-an", "-vcodec", "libx264", "-preset", "medium", "-crf", str(crf),
            "-pix_fmt", "yuv420p", path,
        ], stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray):
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self):
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait()
            self.proc = None


def demo_gif_frames():
    """Quick visual check: roll out a heuristic controller and render a few frames."""
    lv = make_levels()[0]
    env = ElmaEnv(lv, B=1, seed=1)
    frames = []
    for i in range(120):
        obs, r, d, info = env.step(np.array([1]))  # just accelerate
        if i % 20 == 0:
            frames.append(env.render(0, caption=f"t={i*CTRL_DT:.1f}s"))
    return frames


if __name__ == "__main__":
    frames = demo_gif_frames()
    for i, f in enumerate(frames):
        Image.fromarray(f).save(f"/tmp/elma_demo_{i}.png")
    print("saved", len(frames), "frames")
