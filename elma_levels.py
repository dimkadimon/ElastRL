"""
elma_levels.py — level definitions for the RL environment.

IMPORTANT: the original Elasto Mania level assets (.lev files shipped in elma.res)
are NOT part of the open-source release, so the real 54 internal levels cannot be
loaded here. These levels are *hand-authored recreations in the style of the
internals* (flat opener, rolling hills, steep descents), sharing their dimensions,
obstacle vocabulary (apples) and finish-by-flower goal.

Levels are authored in "natural" coordinates (x grows to the right, ground around
y = 0, 1 unit = 1 metre). The game's physics drives the bike toward -x, so the
environment mirrors x internally; the renderer mirrors it back for display.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import elma_physics as ph


@dataclass
class Level:
    name: str
    profile: np.ndarray      # (N,2) ground profile, natural coords
    start_x: float           # spawn position (natural x)
    goal_x: float
    apples: np.ndarray       # (M,2) collectible positions (natural coords)
    max_time: float = 25.0
    par_time: float = 12.0

    # ---- derived (physics coordinates) ----
    @property
    def terrain(self) -> ph.Terrain:
        if not hasattr(self, "_terrain"):
            self._terrain = ph.Terrain.from_polylines([self._polygon()])
        return self._terrain

    def _polygon(self) -> np.ndarray:
        """Closed polygon = profile + a deep skirt. Physics coords (x mirrored)."""
        p = self.profile.copy()
        p[:, 0] = -p[:, 0]  # mirror: natural +x  ->  physics -x (the driving direction)
        bottom = np.array([[p[-1, 0], -8.0], [p[0, 0], -8.0]])
        return np.vstack([p, bottom])

    # ---- geometry helpers ----
    def ground_y(self, x_nat: float) -> float:
        """Terrain height at a natural x."""
        return float(np.interp(x_nat, self.profile[:, 0], self.profile[:, 1]))

    def start_pose(self) -> dict:
        """Bike spawn in physics coordinates, resting on the ground."""
        x_nat = self.start_x
        y = self.ground_y(x_nat) + ph.R_WHL
        x = -x_nat
        return dict(
            r=np.array([x, y + 0.6]),
            r2=np.array([x + ph.OFF_FRONT[0], y]),
            r4=np.array([x + ph.OFF_REAR[0], y]),
            vr=np.array([x, y + 0.6 + 0.44]),
        )

    def goal_pos(self) -> np.ndarray:
        return np.array([-self.goal_x, self.ground_y(self.goal_x) + 0.55])

    def apple_positions(self) -> np.ndarray:
        if len(self.apples) == 0:
            return np.zeros((0, 2))
        a = self.apples.copy()
        a[:, 0] = -a[:, 0]
        return a

    def reference_path(self, ds: float = 0.25, height: float = 0.45) -> np.ndarray:
        """Reference riding line above the ground (TMRL-style reward trajectory).

        Sampled in physics coordinates, ordered in the direction of travel.
        """
        xs = np.arange(self.profile[0, 0], self.profile[-1, 0] + 1e-9, ds)
        ys = np.interp(xs, self.profile[:, 0], self.profile[:, 1]) + height
        # Physics x = -natural x, and the bike drives toward -x, so points ordered by
        # increasing natural x are already ordered in the direction of travel.
        pts = np.stack([-xs, ys], axis=-1)
        return np.ascontiguousarray(pts)


def _smooth(points, n=6):
    """Insert smooth interpolation between control points (dense profile)."""
    pts = np.asarray(points, dtype=np.float64)
    xs = np.linspace(pts[0, 0], pts[-1, 0], n * (len(pts) - 1) + 1)
    ys = np.interp(xs, pts[:, 0], pts[:, 1])
    # light smoothing to avoid hard kinks that the stiff wheels hate
    k = np.ones(3) / 3.0
    ys = np.convolve(ys, k, mode="same")
    ys[0], ys[-1] = pts[0, 1], pts[-1, 1]
    return np.stack([xs, ys], axis=-1)


def make_levels() -> list[Level]:
    # ---------------------------------------------------------------- level 1
    # "Warm Up" — a flat opening run with two shallow dips (internal #1 vibes).
    l1_profile = _smooth([(0.0, 0.0), (10.0, 0.0), (14.0, -0.9), (19.0, 0.3),
                          (26.0, 0.3), (31.0, -1.2), (36.0, 0.0), (46.0, 0.0)], n=7)
    l1 = Level(
        name="1. Warm Up (flat sprint)",
        profile=l1_profile,
        start_x=2.0,
        goal_x=45.0,
        apples=np.array([[16.0, 1.0], [33.0, 0.8]]),
        max_time=22.0,
        par_time=4.5,
    )

    # ---------------------------------------------------------------- level 2
    # "Hi Flyer" — rolling hills, a jump over a gap-ish crest, uphill finish.
    l2_profile = _smooth([(0.0, 0.0), (8.0, 0.0), (13.0, 1.6), (18.0, 0.2),
                          (24.0, 2.6), (29.0, 0.4), (34.0, 3.2), (38.0, 0.8),
                          (43.0, -1.6), (47.0, -1.6), (52.0, 1.2), (58.0, 2.8),
                          (64.0, 4.2), (70.0, 4.2)], n=6)
    l2 = Level(
        name="2. Hi Flyer (hills)",
        profile=l2_profile,
        start_x=2.0,
        goal_x=69.0,
        apples=np.array([[20.0, 1.5], [41.0, 2.0], [60.0, 5.5]]),
        max_time=30.0,
        par_time=9.0,
    )

    # ---------------------------------------------------------------- level 3
    # "Hang Tight" — steep walls and deep basins; needs momentum management.
    # NOTE ON LEVEL DESIGN: measured on this simulator, the bike can climb up to
    # roughly 25-30 degrees with throttle dithering and stalls beyond that (wheels
    # break traction / the bike flips). This level therefore keeps every *local*
    # slope comfortably under that limit - it is the hardest of the three hand-made
    # levels, but it is genuinely ridable.
    l3_profile = _smooth([(0.0, 0.0), (10.0, 0.0), (18.0, 2.4), (26.0, 2.4),
                          (34.0, -0.8), (42.0, -0.8), (50.0, 2.4), (58.0, 2.4),
                          (64.0, 1.2)], n=6)
    l3 = Level(
        name="3. Hang Tight (hilly)",
        profile=l3_profile,
        start_x=2.0,
        goal_x=62.0,
        apples=np.array([[19.0, 3.0], [35.0, 0.2], [51.0, 3.2]]),
        max_time=32.0,
        par_time=11.0,
    )
    return [l1, l2, l3]


if __name__ == "__main__":
    for lv in make_levels():
        pose = lv.start_pose()
        print(f"{lv.name}: {len(lv.profile)} profile pts, {lv.terrain.a.shape[0]} segments, "
              f"start={pose['r']}, goal={lv.goal_pos()}, path={lv.reference_path().shape}")
