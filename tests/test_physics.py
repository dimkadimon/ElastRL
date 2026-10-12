"""Sanity tests for the Elasto Mania physics port.

NOTE ON DIRECTION: in the original game's coordinates, holding the accelerate key
drives the bike toward -x (engine torque is positive on the rear wheel, and the
"front" wheel kor2 sits at x-0.85). Levels are authored in natural left-to-right
coordinates and mirrored internally, so tests here drive toward -x directly.
"""
import sys
import numpy as np

sys.path.insert(0, "/home/user/elma-rl")
import elma_physics as ph

GROUND_Y = 0.0


def long_flat(B=1, xmin=-400.0, xmax=20.0):
    poly = [(xmax, -10.0), (xmax, 0.0), (xmin, 0.0), (xmin, -10.0)]
    return ph.Terrain.from_polylines([poly])


def spawn(m, x=0.0, ground_y=0.0):
    """Place the bike resting on the ground (wheels touching), ready to drive -x."""
    wy = ground_y + ph.R_WHL
    m.r[:] = (x, wy + 0.6)
    m.r2[:] = (x + ph.OFF_FRONT[0], wy)
    m.r4[:] = (x + ph.OFF_REAR[0], wy)
    m.vr[:] = (x, wy + 0.6 + 0.44)
    ph.szamitfejr(m)


def make(B=1, x=0.0):
    m = ph.Motor.zeros(B)
    ph.init_motor(m, B)
    spawn(m, x)
    return m


Z = lambda B=1: np.zeros(B, int)              # noqa: E731
O = lambda B=1: np.ones(B, int)               # noqa: E731


def test_rest():
    ter = long_flat()
    m = make()
    for _ in range(1000):
        ph.leptet(m, Z(), Z(), Z(), Z(), ph.DT, ter)
    wy = m.r2[0, 1] - ph.R_WHL
    print(f"  rest: wheel contact y={wy:+.4f} (expect ~0), body x={m.r[0, 0]:+.4f}, |v|={np.hypot(*m.v[0]):.2e}")
    assert abs(wy) < 0.02
    assert abs(m.v[0, 1]) < 0.05
    assert not ph.vizsgalat_dead(m, ter)[0]


def test_drop():
    ter = long_flat()
    m = make()
    for k in ("r", "r2", "r4", "vr"):
        getattr(m, k)[:, 1] += 3.0
    ph.szamitfejr(m)
    miny = 1e9
    for _ in range(1500):
        ph.leptet(m, Z(), Z(), Z(), Z(), ph.DT, ter)
        miny = min(miny, m.r2[0, 1], m.r4[0, 1])
    wy = m.r2[0, 1] - ph.R_WHL
    print(f"  drop: settled wheel contact y={wy:+.4f}, min wheel centre y={miny:+.4f}")
    assert miny > -0.5, "no tunnelling"
    assert abs(wy) < 0.05, "should rest after landing"


def test_gas_and_top_speed():
    """Holding gas: bike accelerates toward -x and reaches the game's plausible top speed."""
    ter = long_flat()
    m = make()
    x0 = m.r[0, 0]
    vmax = 0.0
    v90 = None
    for i in range(3600):  # ~19.8 s
        ph.leptet(m, O(), Z(), Z(), Z(), ph.DT, ter)
        v = -m.v[0, 0]
        vmax = max(vmax, v)
        if v90 is None and v > 0.9 * vmax and i > 10:
            v90 = i
    dx = x0 - m.r[0, 0]
    v_end = -m.v[0, 0]
    print(f"  gas: travelled {dx:+.1f} m in 19.8 s, peak v={vmax:.2f} m/s, end v={v_end:.2f} m/s, dead={ph.vizsgalat_dead(m, ter)[0]}")
    assert dx > 50.0, "bike should accelerate forward (-x)"
    assert 20.0 < vmax < 50.0, f"top speed {vmax:.1f} m/s should be near the engine's omega cap (44 m/s)"


def test_brake():
    ter = long_flat()
    m = make()
    for _ in range(900):
        ph.leptet(m, O(), Z(), Z(), Z(), ph.DT, ter)
    v_before = -m.v[0, 0]
    for _ in range(400):
        ph.leptet(m, Z(), O(), Z(), Z(), ph.DT, ter)
    v_after = -m.v[0, 0]
    print(f"  brake: v {v_before:+.2f} -> {v_after:+.2f} m/s (toward -x)")
    assert v_after < v_before * 0.8, "braking should slow the bike"


def test_rotate():
    ter = long_flat()
    m = make()
    for _ in range(200):
        ph.leptet(m, O(), Z(), Z(), Z(), ph.DT, ter)
    o0 = m.omega[0]
    for _ in range(4):
        ph.leptet(m, Z(), Z(), O(), Z(), ph.DT, ter)
    d1 = m.omega[0] - o0
    o1 = m.omega[0]
    for _ in range(4):
        ph.leptet(m, Z(), Z(), Z(), O(), ph.DT, ter)
    d2 = m.omega[0] - o1
    print(f"  rotate: key1 dω={d1:+.2f}, key2 dω={d2:+.2f} rad/s")
    assert d1 * d2 < 0, "the two rotate keys must spin the bike in opposite directions"
    assert abs(d1) > 1.0 and abs(d2) > 1.0


def test_crash_on_wall():
    """Drive fast into a wall placed in the driving direction: head contact -> death."""
    poly = [(20.0, -10.0), (20.0, 0.0), (-10.0, 0.0), (-10.0, 6.0), (-10.6, 6.0), (-10.6, 0.0), (-60.0, 0.0), (-60.0, -10.0)]
    ter = ph.Terrain.from_polylines([poly])
    m = make()
    died, t_die = False, None
    for i in range(3000):
        ph.leptet(m, O(), Z(), Z(), Z(), ph.DT, ter)
        if ph.vizsgalat_dead(m, ter)[0]:
            died, t_die = True, i * ph.DT
            break
    print(f"  crash: dead={died} at t={t_die}")
    assert died, "riding into a wall should end in a crash"


def test_slope_climb():
    """A gentle up-slope should be climbable; a very steep one should not be."""
    def slope(deg):
        import math
        a = math.radians(deg)
        # NOTE: the flat approach must be long enough for the bike to build up speed
        # before the ramp; the bike drives toward -x and spawns around x = +2.75.
        pts = [(20.0, -20.0), (20.0, 0.0), (-8.0, 0.0),
               (-8.0 - 40.0 * math.cos(a), 40.0 * math.sin(a)),
               (-8.0 - 40.0 * math.cos(a), 40.0 * math.sin(a) - 20.0),
               (-60.0, -20.0)]
        return ph.Terrain.from_polylines([pts])

    # Throttle must be dithered: holding full gas on a slope makes the bike wheelie and
    # flip over backwards (this is authentic - it is why Elasto Mania requires the rider
    # to feather the accelerator and counter-rotate). Full-throttle climbs are therefore
    # NOT a valid test of the physics.
    def climb(deg, gas_pattern=(1, 1, 0), steps=2400):
        m = make()
        ter = slope(deg)
        climbed = 0.0
        for i in range(steps):
            gaz = np.array([gas_pattern[i % len(gas_pattern)]], dtype=int)
            ph.leptet(m, gaz, Z(), Z(), Z(), ph.DT, ter)
            if ph.vizsgalat_dead(m, ter)[0]:
                break
            climbed = max(climbed, m.r[0, 1])
        return climbed

    res = {deg: climb(deg) for deg in (15.0, 25.0, 40.0)}
    print(f"  slopes: climbed to y={res[15.0]:+.2f} at 15°, y={res[25.0]:+.2f} at 25°, "
          f"y={res[40.0]:+.2f} at 40°")
    assert res[15.0] > 3.0, "15° slope should be climbable with dithered throttle"
    assert res[25.0] > 3.0, "25° slope should be climbable with dithered throttle"
    assert res[40.0] < res[15.0], "40° should be harder than 15°"


def test_batch_consistency():
    ter = long_flat()
    B = 4
    mb = ph.Motor.zeros(B)
    ph.init_motor(mb, B)
    for i in range(B):
        spawn_single(mb, i, x=i * 1.0)
    ph.szamitfejr(mb)

    singles = []
    for i in range(B):
        m1 = ph.Motor.zeros(1)
        for name in ("alfa", "omega", "a2", "o2", "a4", "o4", "t"):
            getattr(m1, name)[:] = getattr(mb, name)[i]
        for name in ("r", "r2", "r4", "v", "v2", "v4", "vr", "vv", "fejr"):
            getattr(m1, name)[:] = getattr(mb, name)[i]
        for name in ("voltfek", "ugrasban1", "ugrasban2"):
            getattr(m1, name)[:] = getattr(mb, name)[i]
        for name in ("ugras1kezd", "ugras2kezd", "kezdoomega1", "kezdoomega2", "dfek2", "dfek4"):
            getattr(m1, name)[:] = getattr(mb, name)[i]
        singles.append(m1)

    g = np.array([1, 1, 0, 0]); f = np.zeros(B, int)
    u1 = np.array([0, 0, 1, 1]); u2 = np.array([0, 1, 0, 0])
    for _ in range(400):
        ph.leptet(mb, g, f, u1, u2, ph.DT, ter)
        for i in range(B):
            ph.leptet(singles[i], np.array([g[i]]), np.array([f[i]]), np.array([u1[i]]), np.array([u2[i]]), ph.DT, ter)
    err = 0.0
    for i in range(B):
        err = max(err, float(np.abs(mb.r[i] - singles[i].r[0]).max()),
                  float(abs(mb.alfa[i] - singles[i].alfa[0])),
                  float(np.abs(mb.r2[i] - singles[i].r2[0]).max()),
                  float(abs(mb.o4[i] - singles[i].o4[0])))
    print(f"  batch: max deviation vs single-world sim = {err:.2e}")
    assert err < 1e-9, "batched and single-world physics must agree exactly"


def spawn_single(m, i, x=0.0, ground_y=0.0):
    wy = ground_y + ph.R_WHL
    m.r[i] = (x, wy + 0.6)
    m.r2[i] = (x + ph.OFF_FRONT[0], wy)
    m.r4[i] = (x + ph.OFF_REAR[0], wy)
    m.vr[i] = (x, wy + 0.6 + 0.44)


if __name__ == "__main__":
    ok = True
    for fn in [test_rest, test_drop, test_gas_and_top_speed, test_brake, test_rotate,
               test_crash_on_wall, test_slope_climb, test_batch_consistency]:
        print(f"[{fn.__name__}]")
        try:
            fn()
            print("  OK")
        except AssertionError as e:
            ok = False
            print(f"  FAILED: {e}")
    print("\nALL PASSED" if ok else "\nSOME TESTS FAILED")
    sys.exit(0 if ok else 1)
