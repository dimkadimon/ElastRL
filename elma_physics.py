"""
elma_physics.py — A batched, faithful re-implementation of Elasto Mania's bike physics.

Ported line-by-line from the original C++ sources (Elasto Mania Team, CC BY-NC-SA 4.0):
  - BEALLIT.CPP   : `beallit()`         -> circle-vs-terrain integration / contact resolution
  - LEPTET.CPP    : `leptet()`          -> the per-tick physics step (engine, brake, jump/rotate, gravity)
  - ADATOK.CPP    : `initadatok()`      -> masses, radii, inertias, stiffnesses
  - UTKOZES.CPP   : `talppontkereses()` -> contact-point search
  - VEKT2.CPP     : vector algebra conventions
  - LEJATSZO.CPP  : `double dt = 0.0055;` -> the game's own integration timestep

The model (exactly as in the original):
  * The bike is THREE circles: body (kor1), front wheel (kor2), rear wheel (kor4).
  * Only the wheels and the rider's head collide with terrain; the body does not.
  * Each wheel is pulled toward its rigid seat in the body frame by a stiff "rubber rod"
    spring, with viscous damping (Drsugar / Sr). Wheel torque is the engine/brake input.
  * Wheel<->ground contact resolution: 0 contact points = free fall/spin; 1 = the wheel
    pivots/rolls around the contact point (this is where grip comes from); 2 = locked.
  * The rider ("vezeto") is a spring-constrained point on the seat; the head hangs off it.
  * Touching terrain with the head = death.
  * Controls are the real 4 keys: accelerate (gaz), brake (fek), rotate-one-way (ugrik1),
    rotate-the-other-way (ugrik2).

All state arrays are batched with leading dimension B (number of parallel worlds).
Angles are radians; lengths in metres; the world y-axis points UP (internal convention).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# float32 keeps the batched simulator memory-bandwidth light; the game's own
# arithmetic is double precision, and the port reproduces the same equations.
DTYPE = np.float32

# ----------------------------------------------------------------------------------
# Constants — exact values from ADATOK.CPP / initadatok()
# ----------------------------------------------------------------------------------

PI = 3.141592  # the game uses its own truncated pi (VEKT2.CPP)
DT = 0.0055  # LEJATSZO.CPP: "double dt = 0.0055;" (Elasto Mania 1.0 timestep)

G = 10.0  # gravity [m/s^2]

R_BODY, M_BODY, TH_BODY = 0.3, 200.0, 200.0 * 0.55 * 0.55  # kor1 "Test elso"
R_WHL, M_WHL, TH_WHL = 0.4, 10.0, 0.32  # kor2 "Kerek elso", kor4 "Kerek hatso"
OFF_FRONT = (-0.85, -0.6)  # front wheel offset in the body frame (kor2)
OFF_REAR = (0.85, -0.6)  # rear wheel offset in the body frame (kor4)

DRSUGAR = 10000.0  # Drsugar: rod stiffness along the rod axis (F/l) [N/m]
DRTANG = 10000.0  # Drtang: rod stiffness perpendicular to the rod axis (F/l) [N/m]
SR = 1000.0  # rod damping (F/v) [N*s/m]

ELSZAKADASI = 0.01  # Elszakadasisebhat: slip/contact separation threshold
BELSOSAV = 0.005  # Belsosav: allowed penetration
TALPPONT_EGYBE = 0.1  # Talppontegybeolvadasitav: merge 2 contact points closer than this

GAZ_NYOMATEK = 600.0  # engine torque on the rear wheel [N*m]
TULPORGES = 110.0  # wheel angular-velocity cap
FEKERO = 1000.0  # brake torque gain
FEKSURLODAS = 100.0  # brake damping
LOKET = 12.0  # rotate-key angular kick [rad/s]
OMEGAVALT = 3.0  # residual spin when a rotate key is released
UGROTURELEM = 0.4  # rotate-key max hold time [s]

FEJSUGAR = 0.238  # head radius (death check)
SEAT_SPRING = DRSUGAR * 5.0  # beallitvezeto: rider restraint stiffness
SEAT_DAMP = SR * 3.0  # beallitvezeto: rider damping


# ----------------------------------------------------------------------------------
# Small vector helpers (VEKT2.CPP semantics; x is the last axis)
# ----------------------------------------------------------------------------------


def rot90(v: np.ndarray) -> np.ndarray:
    """forgatas90fokkal(): (-y, x)"""
    return np.stack([-v[..., 1], v[..., 0]], axis=-1)


def dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]


def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """vekt2 % vekt2"""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def norm(a: np.ndarray) -> np.ndarray:
    return np.sqrt(a[..., 0] ** 2 + a[..., 1] ** 2)


def unit(a: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """v/|v| with a safe fallback direction."""
    n = norm(a)
    safe = np.where(n < eps, 1.0, n)
    out = a / safe[..., None]
    # where degenerate, return (1, 0)
    bad = n < eps
    if np.any(bad):
        out[bad] = np.array([1.0, 0.0])
    return out


def rot_vec(v: np.ndarray, ang: np.ndarray) -> np.ndarray:
    """Rotate vectors v (B,2) by angles ang (B,)."""
    c, s = np.cos(ang)[..., None], np.sin(ang)[..., None]
    return np.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], axis=-1)


def szogigazit(a: np.ndarray) -> np.ndarray:
    """szogigazit(): wrap angles into [-pi, pi]."""
    out = a.copy()
    out = np.where(out < -PI, out + 2 * PI, out)
    out = np.where(out > PI, out - 2 * PI, out)
    return out


# ----------------------------------------------------------------------------------
# State
# ----------------------------------------------------------------------------------


@dataclass
class Motor:
    """motorst — the full bike + rider state, batched over B worlds."""

    alfa: np.ndarray  # (B,) body angle
    omega: np.ndarray  # (B,) body angular velocity
    r: np.ndarray  # (B,2) body position
    v: np.ndarray  # (B,2) body velocity

    a2: np.ndarray  # front wheel angle        (kor2)
    o2: np.ndarray  # front wheel ang. vel.
    r2: np.ndarray  # front wheel position
    v2: np.ndarray  # front wheel velocity

    a4: np.ndarray  # rear wheel angle         (kor4)
    o4: np.ndarray
    r4: np.ndarray
    v4: np.ndarray

    vr: np.ndarray  # (B,2) rider (vezeto) position
    vv: np.ndarray  # (B,2) rider velocity

    fejr: np.ndarray  # (B,2) head position (derived)

    voltfek: np.ndarray  # (B,) previous brake state
    dfek2: np.ndarray  # (B,) wheel angles at brake start
    dfek4: np.ndarray

    ugrasban1: np.ndarray  # (B,) rotate-key 1 active
    ugrasban2: np.ndarray  # (B,) rotate-key 2 active
    ugras1kezd: np.ndarray  # (B,) rotate-key 1 start time
    ugras2kezd: np.ndarray
    kezdoomega1: np.ndarray  # (B,) body omega when the key was pressed
    kezdoomega2: np.ndarray

    t: np.ndarray  # (B,) simulation time of this world [s]

    @staticmethod
    def zeros(B: int) -> "Motor":
        z = lambda n=1: np.zeros((B, n), dtype=DTYPE)
        zs = lambda: np.zeros((B,), dtype=DTYPE)
        return Motor(
            alfa=zs(), omega=zs(), r=z(2), v=z(2),
            a2=zs(), o2=zs(), r2=z(2), v2=z(2),
            a4=zs(), o4=zs(), r4=z(2), v4=z(2),
            vr=z(2), vv=z(2), fejr=z(2),
            voltfek=np.zeros(B, dtype=np.int64),
            dfek2=zs(), dfek4=zs(),
            ugrasban1=np.zeros(B, dtype=np.int64),
            ugrasban2=np.zeros(B, dtype=np.int64),
            ugras1kezd=-np.ones(B, dtype=DTYPE), ugras2kezd=-np.ones(B, dtype=DTYPE),
            kezdoomega1=-np.ones(B, dtype=DTYPE), kezdoomega2=-np.ones(B, dtype=DTYPE),
            t=zs(),
        )

    def copy(self) -> "Motor":
        kw = {}
        for k, val in self.__dict__.items():
            kw[k] = val.copy()
        return Motor(**kw)

    def index(self, idx) -> "Motor":
        """Gather a subset of worlds (used by the renderer / evaluation)."""
        kw = {}
        for k, val in self.__dict__.items():
            kw[k] = val[idx]
        return Motor(**kw)

    def scatter_from(self, other: "Motor", idx) -> None:
        for k, val in other.__dict__.items():
            getattr(self, k)[idx] = val


def init_motor(m: Motor, B: int) -> None:
    """initmotor() — rest pose from ADATOK.CPP, but placed on the ground by the level."""
    m.alfa[:] = 0.0
    m.omega[:] = 0.0
    m.a2[:] = 0.0
    m.o2[:] = 0.0
    m.a4[:] = 0.0
    m.o4[:] = 0.0
    # relative geometry is fixed; r is set by the level's spawn
    m.r[:] = (2.75, 3.6)
    m.r2[:] = (1.9, 3.0)
    m.r4[:] = (3.6, 3.0)
    m.v[:] = 0.0
    m.v2[:] = 0.0
    m.v4[:] = 0.0
    m.vr[:] = (2.75, 4.04)
    m.vv[:] = 0.0
    m.fejr[:] = m.vr + np.array([-0.09, 0.63], dtype=DTYPE)
    m.voltfek[:] = 0
    m.dfek2[:] = 0.0
    m.dfek4[:] = 0.0
    m.ugrasban1[:] = 0
    m.ugrasban2[:] = 0
    m.ugras1kezd[:] = -1.0
    m.ugras2kezd[:] = -1.0
    m.kezdoomega1[:] = -1.0
    m.kezdoomega2[:] = -1.0
    m.t[:] = 0.0


# ----------------------------------------------------------------------------------
# Terrain — segments ("vonal": r, egyseg, hossz)
# ----------------------------------------------------------------------------------


@dataclass
class Terrain:
    """A set of collision segments (B worlds all share one level)."""

    a: np.ndarray  # (S,2) segment start
    b: np.ndarray  # (S,2) segment end
    e: np.ndarray  # (S,2) unit direction
    L: np.ndarray  # (S,) length
    n: np.ndarray  # (S,2) left normal (rot90 of e)

    @staticmethod
    def from_polylines(polys) -> "Terrain":
        segs = []
        for poly in polys:
            pts = np.asarray(poly, dtype=DTYPE)
            for i in range(len(pts)):
                a = pts[i]
                b = pts[(i + 1) % len(pts)]
                if np.hypot(*(b - a)) > 1e-9:
                    segs.append((a, b))
        a = np.array([s[0] for s in segs])
        b = np.array([s[1] for s in segs])
        d = b - a
        L = np.hypot(d[:, 0], d[:, 1])
        e = d / L[:, None]
        return Terrain(a=a, b=b, e=e, L=L, n=rot90(e))


def contact_points(r: np.ndarray, radius: float, ter: Terrain, max_pts: int = 2):
    """talppontkereses() / gombszakasz() — circle vs all segments.

    r: (B,2) circle centres.
    Returns (t1, t2, n_contacts) where t1/t2 are (B,2) and n_contacts is (B,) in {0,1,2}.
    Equivalent to the original: perpendicular distance test along the segment,
    true distance at the endpoints; then the two closest contacts are kept, and
    contacts closer together than Talppontegybeolvadasitav are merged.
    """
    B = r.shape[0]
    ax, ay = ter.a[:, 0], ter.a[:, 1]
    ex, ey = ter.e[:, 0], ter.e[:, 1]
    rx, ry = r[:, 0][:, None], r[:, 1][:, None]           # (B,1)

    relx = rx - ax[None, :]
    rely = ry - ay[None, :]
    proj = relx * ex[None, :] + rely * ey[None, :]        # (B,S) distance along the segment
    tc = np.minimum(np.maximum(proj, 0.0), ter.L[None, :])
    # squared distance to the closest point, computed without materialising it
    dx = relx - ex[None, :] * tc
    dy = rely - ey[None, :] * tc
    d2 = dx * dx + dy * dy
    d2 = np.where(d2 < radius * radius, d2, np.inf)

    flat = np.arange(B)
    i1 = np.argmin(d2, axis=1)
    d1 = d2[flat, i1]
    has1 = np.isfinite(d1)

    # second-closest contact: mask out the first and re-argmin
    d2_2 = d2.copy()
    d2_2[flat, i1] = np.inf
    i2 = np.argmin(d2_2, axis=1)
    has2 = np.isfinite(d2_2[flat, i2])

    t1 = np.stack([ax[i1] + ex[i1] * tc[flat, i1], ay[i1] + ey[i1] * tc[flat, i1]], axis=-1)
    t2 = np.stack([ax[i2] + ex[i2] * tc[flat, i2], ay[i2] + ey[i2] * tc[flat, i2]], axis=-1)

    # merge contacts that are too close (Talppontegyolvadasitav)
    ddx = t1[:, 0] - t2[:, 0]
    ddy = t1[:, 1] - t2[:, 1]
    merge = has2 & (ddx * ddx + ddy * ddy < TALPPONT_EGYBE * TALPPONT_EGYBE)
    t1 = np.where(merge[:, None], 0.5 * (t1 + t2), t1)
    has2 = has2 & ~merge

    n_contacts = has1.astype(np.int64) + has2.astype(np.int64)
    return t1, t2, n_contacts


# ----------------------------------------------------------------------------------
# BEALLIT.CPP
# ----------------------------------------------------------------------------------


def _helyigazitas(r, pt, radius, n_contacts, mask):
    """Push the circle out of penetration (allowed penetration = Belsosav)."""
    d = r - pt
    hossz = norm(d)
    n = d / np.maximum(hossz, 1e-12)[:, None]
    need = mask & (hossz < radius - BELSOSAV)
    push = np.where(need, (radius - BELSOSAV - hossz), 0.0)
    return r + n * push[:, None]


def _talppontigazitas(r, v, pt, F, mask):
    """talppontigazitas(): decide whether a contact is real; kill normal velocity.

    A contact is not real if the circle is (nearly) separating AND the net force
    pushes it away from the contact point. When the contact *is* real, the normal
    component of the velocity is removed (perfectly inelastic).

    Returns (still_contact_mask, v_after) where v_after == v except on real contacts.
    """
    d = r - pt
    hossz = norm(d)
    n = d / np.maximum(hossz, 1e-12)[:, None]
    nv = dot(n, v)
    nf = dot(n, F)
    real = ~((nv > -ELSZAKADASI) & (nf > 0.0))
    keep = mask & real
    new_v = v - n * nv[:, None]
    return keep, np.where(keep[:, None], new_v, v)


def _biztostalppont_uj(t1, t2, r, v, omega, mask):
    """biztostalppont_uj(): can the wheel really rest on both points?"""
    d = r - t2
    hossz = norm(d)
    n = d / np.maximum(hossz, 1e-12)[:, None]
    n90 = rot90(n)
    mtm = omega + hossz * dot(n90, v)
    val = cross(t1 - t2, n90) * mtm
    return mask & (val >= 0.0)


def _biztostalppont_regi(t1, t2, r, F, M, mask):
    """biztostalppont_regi(): force-based version, used at low speed."""
    d = r - t2
    hossz = norm(d)
    n = d / np.maximum(hossz, 1e-12)[:, None]
    n90 = rot90(n)
    mtm = M + hossz * dot(n90, F)
    val = cross(t1 - t2, n90) * mtm
    return mask & (val >= 0.0)


def beallit(r, v, alfa, omega, m, radius, theta, F, M, dt, ter: Terrain, kelltalppont: bool):
    """beallit(): collide-aware integration of one circle (BEALLIT.CPP).

    Returns (r, v, alfa, omega).
    """
    B = r.shape[0]
    if not kelltalppont:
        # the body does not collide with terrain in Elasto Mania
        beta = M / theta
        omega = omega + beta * dt
        alfa = alfa + omega * dt
        v = v + (F / m) * dt
        r = r + v * dt
        return r, v, alfa, omega

    t1, t2, np_ = contact_points(r, radius, ter)

    # geometry fix-up (penetration)
    m1 = np_ >= 1
    m2 = np_ >= 2
    r = _helyigazitas(r, t1, radius, m1, m1)
    r = _helyigazitas(r, t2, radius, m2, m2)

    speed = norm(v)

    # ---- 2 contact points: try to reduce to 1 by checking the pivot direction ----
    # High speed uses the velocity-based test (biztostalppont_uj)...
    hi = m2 & (speed > 1.0)
    ok12 = _biztostalppont_uj(t1, t2, r, v, omega, hi)
    ok21 = _biztostalppont_uj(t2, t1, r, v, omega, hi & ok12)
    fails12 = hi & ~ok12
    drop_hi = fails12 | (hi & ok12 & ~ok21)
    t1 = np.where(fails12[:, None], t2, t1)
    np_ = np.where(drop_hi, 1, np_)

    # ... and low speed uses the force-based test (biztostalppont_regi).
    lo = (np_ == 2) & (speed < 1.0)
    ok12 = _biztostalppont_regi(t1, t2, r, F, M, lo)
    ok21 = _biztostalppont_regi(t2, t1, r, F, M, lo & ok12)
    fails12 = lo & ~ok12
    drop_lo = fails12 | (lo & ok12 & ~ok21)
    t1 = np.where(fails12[:, None], t2, t1)
    np_ = np.where(drop_lo, 1, np_)

    # ---- validate contacts against actual motion/force (accumulating in v) ----
    both = np_ == 2
    keep2, v = _talppontigazitas(r, v, t2, F, both)
    np_ = np.where(both & ~keep2, 1, np_)

    one = np_ >= 1
    keep1, v = _talppontigazitas(r, v, t1, F, one)
    fail1_was2 = (np_ == 2) & ~keep1  # original: count=1 and t1 := t2
    np_ = np.where((np_ == 1) & ~keep1, 0, np_)
    np_ = np.where(fail1_was2, 1, np_)
    t1 = np.where(fail1_was2[:, None], t2, t1)

    # ---- integrate, split by number of contact points ----
    free = np_ == 0
    locked = np_ == 2
    roll = np_ == 1

    new_r, new_v, new_a, new_o = r.copy(), v.copy(), alfa.copy(), omega.copy()

    if np.any(free):
        beta = M[free] / theta
        o = omega[free] + beta * dt
        a = alfa[free] + o * dt
        vv = v[free] + (F[free] / m) * dt
        rr = r[free] + vv * dt
        new_o[free], new_a[free], new_v[free], new_r[free] = o, a, vv, rr

    if np.any(locked):
        new_v[locked] = 0.0
        new_o[locked] = 0.0
        # position/angle unchanged (matches the original)

    if np.any(roll):
        sub = np.where(roll)[0]
        d = r[sub] - t1[sub]
        hossz = norm(d)
        n = d / np.maximum(hossz, 1e-12)[:, None]
        n90 = rot90(n)
        # rolling without slipping around the contact point
        o = dot(v[sub], n90) / radius
        # torque: external M plus the moment of F about the contact point
        Mloc = M[sub] + dot(F[sub], n90) * radius
        thetaszelso = theta + m * hossz * hossz
        beta = Mloc / thetaszelso
        o = o + beta * dt
        a = alfa[sub] + o * dt
        vv = (o * radius)[:, None] * n90
        rr = r[sub] + vv * dt
        new_o[sub], new_a[sub], new_v[sub], new_r[sub] = o, a, vv, rr

    return new_r, new_v, new_a, new_o


# ----------------------------------------------------------------------------------
# LEPTET.CPP
# ----------------------------------------------------------------------------------


def _erokszamitasa(m: Motor, pr, pv, po, kordx, kordy, Mkerek, out):
    """erokszamitasa(): wheel<->body rubber-rod forces. Writes (Fkerek, Ftest, Mtest)."""
    gumis = m.r * 0.0
    i1 = np.stack([np.cos(m.alfa), np.sin(m.alfa)], axis=-1)
    j1 = rot90(i1)
    gumis = i1 * kordx + j1 * kordy  # seat offset in world frame
    gumisabsz = gumis + m.r  # target wheel position
    gumi = gumisabsz - pr  # spring displacement

    # The wheel is held by a 2D "rubber rod" whose rest direction is the (constant)
    # body-frame offset gumis. The displacement `gumi` is resolved in the frame of
    # `rudegys` = unit(gumis): the component along the rod uses Drsugar, the component
    # perpendicular to it uses Drtang. (Drsugar == Drtang == 10000, so the net force is
    # DRSUGAR*gumi, but the *tangential* part also produces a torque on the body below.)
    rudhossz = norm(gumis)                    # |gumis|, constant per wheel (1.0404 here)
    rudegys = gumis / np.maximum(rudhossz, 1e-12)[:, None]
    rudegysmer = rot90(rudegys)
    Fsugar = dot(gumi, rudegys) * DRSUGAR
    Ftang = dot(gumi, rudegysmer) * DRTANG
    Fkerek = rudegys * Fsugar[:, None] + rudegysmer * Ftang[:, None]

    # damping: relative velocity of the body point at the wheel location vs the wheel
    koto = pr - m.r
    kotol = np.maximum(norm(koto), 1e-12)
    kotoe = koto / kotol[:, None]
    kotomer = rot90(koto)
    kotoemer = rot90(kotoe)
    korongrelv = (rot90(koto) * m.omega[:, None] + m.v) - pv
    vlong = dot(korongrelv, kotoe)
    vtang = dot(korongrelv, kotoemer)
    Fklong = kotoe * (vlong * SR)[:, None]
    Fktang = kotoemer * (vtang * SR)[:, None]

    # torque reaction from the wheel's engine torque
    Ftestnyom = ((Mkerek / kotol)[:, None]) * kotoemer

    Fkerek = Fkerek + Fklong + Fktang - Ftestnyom
    # Torque on the body: the rod's tangential force acting at the rod's rest length
    # (LEPTET.CPP: `*pMtest = -Ftang*rudhossz`), plus the damping force's contribution
    # (`*pMtest += -(Fktang*kotomer)`, where Fktang*kotomer == (vtang*Sr)*|koto|).
    # Dropping the first term removes the spring's ability to keep the body aligned
    # with its own wheels, which is what made the bike pitch and wobble uncontrollably.
    Mtest = -Ftang * rudhossz - (vtang * SR) * kotol
    Ftest = -Fkerek

    out[0][:] = Fkerek
    out[1][:] = Ftest
    out[2][:] = Mtest
    return Ftest, Fkerek, Mtest


def leptet(m: Motor, gaz, fek, ugrik1, ugrik2, dt: float, ter: Terrain):
    """leptet(): one physics tick. Mirrors LEPTET.CPP exactly.

    gaz/fek/ugrik1/ugrik2 are int arrays (B,), as in the original game's key states.
    """
    B = m.r.shape[0]
    i1 = np.stack([np.cos(m.alfa), np.sin(m.alfa)], axis=-1)
    j1 = rot90(i1)

    gaz = np.asarray(gaz).astype(bool)
    fek = np.asarray(fek).astype(bool)
    ugrik1 = np.asarray(ugrik1).astype(bool)
    ugrik2 = np.asarray(ugrik2).astype(bool)

    # ---- brake bookkeeping (wheel angles captured when the brake is pressed) ----
    start_brake = (m.voltfek == 0) & fek
    m.dfek2 = np.where(start_brake, m.a2 - m.alfa, m.dfek2)
    m.dfek4 = np.where(start_brake, m.a4 - m.alfa, m.dfek4)
    m.voltfek = fek.astype(np.int64)

    # ---- engine / brake torques ----
    Mk2 = np.zeros(B, dtype=DTYPE)
    Mk4 = np.zeros(B, dtype=DTYPE)

    # hatra_f == 0 (bike faces its forward axis): engine torque goes to the rear wheel
    gas_f = np.where(gaz & (m.o4 < TULPORGES), np.float32(GAZ_NYOMATEK), np.float32(0.0))
    Mk4 = np.where(gaz, gas_f, Mk4)

    brake_on = fek
    dalfa2 = m.a2 - (m.alfa + m.dfek2)
    domega2 = m.o2 - m.omega
    dalfa4 = m.a4 - (m.alfa + m.dfek4)
    domega4 = m.o4 - m.omega
    Mk2 = np.where(brake_on, -FEKERO * dalfa2 - FEKSURLODAS * domega2, Mk2)
    Mk4 = np.where(brake_on, -FEKERO * dalfa4 - FEKSURLODAS * domega4, Mk4)
    # Safety clamp: the brake is a position servo with no torque limit in the original
    # code; in pathological states (wheel locked while the brake is held) the angle
    # error can grow without bound. Real gameplay never reaches these magnitudes
    # (|Mk| <= ~1 kN*m in normal riding), so clamping only tames blow-ups.
    Mk2 = np.clip(Mk2, -20000.0, 20000.0)
    Mk4 = np.clip(Mk4, -20000.0, 20000.0)

    if np.any(~brake_on):
        nb = ~brake_on
        m.a2 = np.where(nb, szogigazit(m.a2), m.a2)
        m.a4 = np.where(nb, szogigazit(m.a4), m.a4)

    # ---- wheel<->body forces ----
    out = (np.zeros((B, 2), dtype=DTYPE), np.zeros((B, 2), dtype=DTYPE), np.zeros(B, dtype=DTYPE))
    Ftest2, Fkerek2, Mtest2 = _erokszamitasa(
        m, m.r2, m.v2, m.o2, OFF_FRONT[0], OFF_FRONT[1], Mk2, out)
    Ftest4, Fkerek4, Mtest4 = _erokszamitasa(
        m, m.r4, m.v4, m.o4, OFF_REAR[0], OFF_REAR[1], Mk4, out)

    # ---- jump / rotate keys ----
    oldomega = np.where(ugrik1 | ugrik2, m.omega, 0.0)

    # finish rotate-1 if it was active
    act1 = (m.ugrasban1 == 1) & (ugrik1 | ugrik2 | (m.t > m.ugras1kezd + UGROTURELEM * 0.25))
    o = m.omega + LOKET
    o = np.where(o > m.kezdoomega1, m.kezdoomega1, o)
    o = np.where(o > 0.0, np.maximum(o - OMEGAVALT, 0.0), o)
    m.omega = np.where(act1, o, m.omega)
    m.ugrasban1 = np.where(act1, 0, m.ugrasban1)
    m.kezdoomega1 = np.where(act1, -1.0, m.kezdoomega1)
    m.ugras1kezd = np.where(act1, -1.0, m.ugras1kezd)

    act2 = (m.ugrasban2 == 1) & (ugrik1 | ugrik2 | (m.t > m.ugras2kezd + UGROTURELEM * 0.25))
    o = m.omega - LOKET
    o = np.where(o < m.kezdoomega2, m.kezdoomega2, o)
    o = np.where(o < 0.0, np.minimum(o + OMEGAVALT, 0.0), o)
    m.omega = np.where(act2, o, m.omega)
    m.ugrasban2 = np.where(act2, 0, m.ugrasban2)
    m.kezdoomega2 = np.where(act2, -1.0, m.kezdoomega2)
    m.ugras2kezd = np.where(act2, -1.0, m.ugras2kezd)

    # start rotate-1
    m.ugrasban1 = np.where(ugrik1, 1, m.ugrasban1)
    m.kezdoomega1 = np.where(ugrik1, m.omega, m.kezdoomega1)
    m.ugras1kezd = np.where(ugrik1, m.t, m.ugras1kezd)
    m.omega = np.where(ugrik1, m.omega - LOKET, m.omega)

    # start rotate-2
    m.ugrasban2 = np.where(ugrik2, 1, m.ugrasban2)
    m.kezdoomega2 = np.where(ugrik2, m.omega, m.kezdoomega2)
    m.ugras2kezd = np.where(ugrik2, m.t, m.ugras2kezd)
    m.omega = np.where(ugrik2, m.omega + LOKET, m.omega)

    # the rider is rotated along with the body when a key is pressed
    kick = ugrik1 | ugrik2
    domega = np.where(kick, m.omega - oldomega, 0.0)
    tangens = rot90(m.vr - m.r)
    m.vv = m.vv + tangens * domega[:, None]

    # ---- gravity + integration ----
    grav = np.array([0.0, -1.0], dtype=DTYPE) * (M_BODY * G)
    F_body = Ftest2 + Ftest4 + grav[None, :]
    M_body = Mtest2 + Mtest4

    m.r, m.v, m.alfa, m.omega = beallit(
        m.r, m.v, m.alfa, m.omega, M_BODY, R_BODY, TH_BODY,
        F_body, M_body, dt, ter, kelltalppont=False)

    Fw2 = Fkerek2 + np.array([0.0, -M_WHL * G], dtype=DTYPE)[None, :]
    m.r2, m.v2, m.a2, m.o2 = beallit(
        m.r2, m.v2, m.a2, m.o2, M_WHL, R_WHL, TH_WHL,
        Fw2, Mk2, dt, ter, kelltalppont=True)

    Fw4 = Fkerek4 + np.array([0.0, -M_WHL * G], dtype=DTYPE)[None, :]
    m.r4, m.v4, m.a4, m.o4 = beallit(
        m.r4, m.v4, m.a4, m.o4, M_WHL, R_WHL, TH_WHL,
        Fw4, Mk4, dt, ter, kelltalppont=True)

    # ---- rider (beallitvezeto) ----
    _beallitvezeto(m, dt)

    # ---- head from body ----
    szamitfejr(m)

    m.t = m.t + dt
    return m


def _vezeto_hatarolas(m: Motor):
    """vezeto_hatarolas(): keep the rider inside the seat region (body frame)."""
    i1 = np.stack([np.cos(m.alfa), np.sin(m.alfa)], axis=-1)
    j1 = rot90(i1)
    rel = m.vr - m.r
    x = dot(i1, rel)
    y = dot(j1, rel)
    r = np.stack([x, y], axis=-1)

    # seat constraint
    ar = np.array([-0.35, 0.13], dtype=DTYPE)
    av = np.array([0.14 - (-0.35), 0.36 - 0.13], dtype=DTYPE)
    avnorm = np.array([-av[1], av[0]], dtype=DTYPE)
    avnormegys = avnorm / np.hypot(*avnorm)
    lelogas = dot(r - ar, np.broadcast_to(avnormegys, r.shape))
    under = lelogas < 0.0
    r = np.where(under[:, None], r - avnormegys[None, :] * lelogas[:, None], r)

    r[:, 1] = np.minimum(r[:, 1], 0.48)  # felhat
    r[:, 0] = np.maximum(r[:, 0], -0.5)
    r[:, 0] = np.minimum(r[:, 0], 0.26)  # jobbhat

    # circular clamp in the front-upper quadrant
    felhat, jobbhat = 0.48, 0.26
    tavmax = float(np.sqrt(felhat * felhat))
    rxszorzo = (felhat / jobbhat) ** 2
    quad = (r[:, 0] > 0) & (r[:, 1] > 0)
    tv = r[:, 0] ** 2 * rxszorzo + r[:, 1] ** 2
    need = quad & (tv > felhat * felhat)
    szorzo = np.where(need, tavmax / np.sqrt(np.maximum(tv, 1e-12)), 1.0)
    r = np.where(need[:, None], r * szorzo[:, None], r)

    m.vr = i1 * r[:, :1] + j1 * r[:, 1:2] + m.r


def _beallitvezeto(m: Motor, dt: float):
    """beallitvezeto(): rider spring + damping + gravity."""
    _vezeto_hatarolas(m)

    i1 = np.stack([np.cos(m.alfa), np.sin(m.alfa)], axis=-1)
    j1 = rot90(i1)
    rugor = m.r + j1 * 0.44  # Kord5y = vezetor.y - kor1.r.y = 4.04 - 3.6 (ADATOK.CPP)
    d = rugor - m.vr
    hl = np.maximum(norm(d), 1e-7)
    e = d / hl[:, None]
    Frugo = e * (hl * SEAT_SPRING)[:, None]

    tangens = rot90(m.vr - m.r)
    motseb = tangens * m.omega[:, None] + m.v
    relv = m.vv - motseb
    Fsurl = relv * SEAT_DAMP

    F = Frugo - Fsurl + np.array([0.0, -M_BODY * G], dtype=DTYPE)[None, :]
    a = F / M_BODY
    m.vv = m.vv + a * dt
    m.vr = m.vr + m.vv * dt


def szamitfejr(m: Motor):
    """szamitfejr(): head position from the rider point (hatra_f = 0 here)."""
    i1 = np.stack([np.cos(m.alfa), np.sin(m.alfa)], axis=-1)
    j1 = rot90(i1)
    m.fejr = m.vr - i1 * 0.09 + j1 * 0.63


def szamitfejr_idx(m: Motor, idx) -> None:
    """szamitfejr() for a subset of worlds."""
    i1 = np.stack([np.cos(m.alfa[idx]), np.sin(m.alfa[idx])], axis=-1)
    j1 = rot90(i1)
    m.fejr[idx] = m.vr[idx] - i1 * 0.09 + j1 * 0.63


def vizsgalat_dead(m: Motor, ter: Terrain) -> np.ndarray:
    """vizsgalat() == 0  ->  the head touches terrain: crash."""
    _, _, npts = contact_points(m.fejr, FEJSUGAR, ter)
    return npts > 0


def observe_lidar(m: Motor, ter: Terrain, n_rays: int = 19, span: float = 2.1,
                  max_range: float = 8.0) -> np.ndarray:
    """LIDAR-style range measurements around the bike (body frame), TMRL-style.

    Returns (B, n_rays) normalised distances in [0,1] (1 = nothing within range).
    The middle ray points along the bike's direction of travel (in the original
    game's coordinates, engine torque drives the bike toward -i1, i.e. -x at rest).
    """
    B = m.r.shape[0]
    S = ter.a.shape[0]
    angs = PI + np.linspace(-span, span, n_rays)  # centred on the travel direction
    base = m.alfa[:, None] + angs[None, :]  # (B,R)
    d = np.stack([np.cos(base), np.sin(base)], axis=-1)  # (B,R,2)

    # solve |o + t*d - a - u*e| = 0 for segment hits, t in [0, max_range], u in [0,L]
    o = m.r + np.array([0.0, 0.3])[None, :]  # cast from a little above the body centre
    denom = (d[:, :, None, 0] * ter.e[None, None, :, 1]
             - d[:, :, None, 1] * ter.e[None, None, :, 0])  # (B,R,S)
    q = ter.a[None, None, :, :] - o[:, None, None, :]  # (B,R,S,2)
    safe = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
    t = (q[..., 0] * ter.e[None, None, :, 1] - q[..., 1] * ter.e[None, None, :, 0]) / safe
    u = (q[..., 0] * d[:, :, None, 1] - q[..., 1] * d[:, :, None, 0]) / safe
    valid = (t > 0.0) & (t < max_range) & (u >= 0.0) & (u <= ter.L[None, None, :])
    t = np.where(valid, t, np.inf)
    dist = np.min(t, axis=2)  # (B,R)
    return np.where(np.isfinite(dist), dist / max_range, 1.0)
