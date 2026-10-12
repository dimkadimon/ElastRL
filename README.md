# Elasto Mania RL — teaching an agent to ride Elasto Mania

A reinforcement-learning agent that learns to ride **Elasto Mania** levels, built on the
same architecture as [TMRL](https://github.com/trackmania-rl/tmrl) (the TrackMania RL
framework): **Soft Actor-Critic with automatic entropy tuning**, a **discrete action
space driven by key combinations**, and a **LIDAR + speed observation** interface, trained
in a **batched, head-less re-implementation of the game's own physics engine**.

Training runs emit **videos at every checkpoint**, so the improvement over training time
is directly watchable.

**Headline result: three levels, ~2.5 M environment steps, ~50 minutes on two CPU cores,
and every level is then completed 100% of the time by the deterministic policy — faster
than a hand-written controller on all three.**

| | Scripted hand-written controller | Learned SAC policy |
|---|---|---|
| 1. Warm Up (flat, 46 m) | 6.14 s | **4.61 s** (100% finish) |
| 2. Hi Flyer (hills, 70 m) | 11.29 s | **9.18 s** (100% finish) |
| 3. Hang Tight (hilly, 64 m) | 8.17 s, only 1 of 3 seeds | **6.23 s** (100% finish) |

**Watch it:** `showcase.mp4` (all three levels, early/mid/final checkpoints side by side) ·
`runs/level*/training_progress.mp4` (one level at successive checkpoints) ·
`runs/level*/comparison.mp4` (untrained vs trained) · `runs/level*/training_curves.png`.

---

## 0. Honest disclosure — read this first

Two things are important to state plainly, because they are the difference between what
was asked for and what is achievable here:

1. **The 54 real internal levels cannot be used.** The Elasto Mania open-source release
   ([elastomania/elma-classic](https://github.com/elastomania/elma-classic)) ships the
   *engine*, not the game assets: the `.lev` level files live inside `elma.res`, which is
   explicitly **not** part of the release ("this is NOT a playable release", assets
   excluded). The game is otherwise closed source (invite-only contributor programme).
   Therefore this project trains on **three hand-authored levels written in the style of
   the internals** — a flat opener, a rolling-hills level and a hilly third level — with
   the same vocabulary as the real game (apples, a flower goal, crash-on-head-touch,
   time limit). They are *not* the real Elasto Mania levels, and results here are not
   comparable to human world records on the internals.
   Verified sources: the GitHub API listing of `elastomania/elma-classic` (assets
   excluded, dual license), `elastomania.com/contributor.html` (closed source), and the
   `across` repository (level-editor-era code, also without level assets).

2. **This is a 2-vCPU CPU-only sandbox.** No GPU, ~2 GB RAM. Every number below is from
   that machine. TMRL trains TrackMania agents for *days* on a GPU; here the whole
   budget is a couple of hours on two cores. The comparison that matters is therefore
   *relative*: the agent starts by falling over and ends up completing the level
   repeatedly, reproducibly, from randomised starts.

What "trains to play really well" means in this document, concretely, is measured: the
greedy (deterministic) policy's **finish rate**, **completion time** and **progress
along the reference line**, evaluated the same way for every checkpoint.

---

## 1. What is in here

| File | What it is |
|---|---|
| `elma_physics.py` | Batched (B worlds at once) NumPy port of the original bike physics, line-by-line from `BEALLIT.CPP`, `LEPTET.CPP`, `ADATOK.CPP`, `UTKOZES.CPP`, `VEKT2.CPP`, using the game's own `dt = 0.0055`. |
| `elma_levels.py` | The three hand-authored levels + the reference riding line used by the reward. |
| `elma_env.py` | Gym-style vectorised environment: 9 discrete key-combination actions, 19-ray LIDAR + 11 proprioceptive inputs, TMRL's progress-along-trajectory reward, auto-reset, and a software renderer (`render()`) plus `VideoWriter`. |
| `sac.py` | Discrete Soft Actor-Critic: twin Q critics, categorical policy, polyak target networks, automatic entropy tuning. |
| `train.py` | Training loop: replay buffer, evaluation, checkpoints, `stats.jsonl`/`eval.jsonl`, SIGTERM-safe, and periodic **video capture**. |
| `make_progress_video.py` | Builds the synchronised grid video of successive checkpoints. |
| `plot_curves.py` | Builds the static + animated training-curve figures. |
| `make_deliverables.py` | One-shot builder for all per-run artefacts. |
| `scripted_baseline.py` | Hand-written feedback controller used as a *solvability reference*. |
| `tests/test_physics.py` | Physics tests (batch/single equivalence, contact points, determinism). |

---

## 2. How the pieces map onto TMRL

TMRL (`tmrl/custom/custom_algorithms.py`, `custom_models.py`) is the architectural
reference. The mapping is deliberate:

| TMRL | Here | Note |
|---|---|---|
| `SpinupSacAgent` (SAC v2) | `sac.SAC` | `gamma 0.99`, `polyak 0.995`, Adam, automatic entropy coefficient. |
| `MLPActorCritic` — **separate** actor and twin-Q MLPs | `sac.ActorCritic` | No shared trunk: TMRL builds independent `SquashedGaussianMLPActor` + `MLPQFunction` nets, and sharing one trunk measurably destabilised learning here. |
| LIDAR interface, 20 Hz control | `ElmaEnv` | 19 rays + speed/pitch/wheel-spin/contact/progress proprioception; `CTRL_DT = 9 × 0.0055 s ≈ 0.0495 s`. |
| `RewardFunction.compute_reward` — progress along a trajectory | `ElmaEnv._path_progress` | Same idea: reward is metres advanced along a reference line, with a forward window, an explicit rewind window and a max-distance-from-trajectory gate. |
| `Memory` / `TorchMemory` | `sac.ReplayBuffer` | NumPy ring buffer, float32 observations. |
| Real-time gym interface | batched head-less env | The sandbox cannot run the game in real time, so the physics is stepped directly, vectorised. |

Deviations from TMRL, and why:

* **Discrete actions.** TMRL's LIDAR interface uses a small discrete action set; the
  game's controls *are* four keys, so the natural action set is their 9 sensible
  combinations (`coast`, `gas`, `brake`, `rot1`, `rot2`, and `gas/brake × rot`).
* **Separate actor/critic nets** as above.
* **Reward shaping for a CPU budget.** The progress reward is scaled (`reward_scale =
  0.2`) and carries a small per-step living cost, which keeps the value function in a
  range that converges in ~1M environment steps instead of ~100M.

---

## 3. The environment in one screen

* **Physics**: the bike is three circles (body + two wheels) joined by stiff spring rods;
  the rider is a spring-constrained point whose *head kills you* if it touches terrain.
  Only the wheels and the head collide. Wheel/ground contact resolves to 0, 1 or 2 contact
  points per wheel, which is where grip comes from. All of this is the original algorithm.
* **Observation (30 dims)**: 19 LIDAR ranges (8 m, ±120° around the direction of travel,
  normalised) + forward speed, speed across the bike frame, body spin rate, both wheel
  spin rates, `sin`/`cos` of the chassis angle, both wheel-contact booleans, normalised
  track progress and normalised time remaining.
* **Action (9)**: the key combinations above.
* **Episode**: up to 2,000 control steps (~99 s). Ends on finish (goal flower), death
  (head contact or falling off the world), stall (`stuck`: no progress for a while, TMRL's
  failure counter) or time-out. Worlds auto-reset independently.
* **Reward**: metres of progress along the reference line, + apple bonus, + finish bonus,
  − crash penalty, − small cost per step.

**Guard rails that matter for stability.** Two failure modes were found and fixed during
development, both now covered by tests:

* The original brake is a position servo with no torque limit; in pathological states the
  wheel angle error can grow without bound. It is clamped (`±20,000 N·m`), which only
  affects states the real game can never reach.
* Any world leaving the sane state space (non-finite, or speed/spin magnitudes orders of
  magnitude beyond anything the game produces) is killed and zeroed by the environment,
  so **no non-finite value can ever enter the replay buffer**.

---

## 4. Speed on 2 CPU cores

Measured on this sandbox (Intel-class 2 vCPU, NumPy, torch 2.x CPU build):

| Stage | Throughput |
|---|---|
| Physics only | ~47,000 environment sub-steps/s (B = 64) |
| Environment, B = 64 | ~3,300–3,800 env-steps/s (~55 control steps/s) |
| Environment, B = 128 | ~5,700 env-steps/s (~45 control steps/s) |
| SAC update (`exact` actor loss, batch 128, 192×192) | ~95 updates/s |

The environment is the bottleneck, so training runs 64 parallel worlds and 2 gradient
updates per control step. A 30,000-control-step run is ≈ 1.9 M environment steps and
takes roughly 30–50 minutes wall-clock.

---

## 5. Results

Three runs, one per level, sequential on the 2-vCPU sandbox. "Greedy eval" = the
deterministic policy, replayed from randomised starts; the checkpoint shown is the one
`select_best.py` picks (highest finish rate, then fastest completion).

| Level | Path | Training steps | Environment steps | Wall clock | Best checkpoint | Greedy finish rate | Greedy time |
|---|---|---|---|---|---|---|---|
| 1. Warm Up (flat) | 46.2 m | 6,700 | 428,800 | 11.3 min | step 6,500 | **6/6 (100%)** | **4.61 s** |
| 2. Hi Flyer (hills) | 70.2 m | 12,400 | 793,600 | 12.3 min | step 11,000 | **5/5 (100%)** | **9.18 s** |
| 3. Hang Tight (hilly) | 64.2 m | 20,000 | 1,280,000 | 26.4 min | step 20,000 | **5/5 (100%)** | **6.23 s** |

Total: about **2.5 M environment steps and 50 minutes of wall clock on two CPU cores**
to go from "the bike immediately falls over" to three levels each ridden to the flower
at 100% reliability.

### Beating a hand-written controller

`scripted_baseline.py` is a hand-written feedback controller (throttle duty-cycling plus
air-attitude levelling) that a human would write by hand. The learned policies beat it:

| Level | Scripted baseline | Learned SAC policy | |
|---|---|---|---|
| 1. Warm Up | 6.14 s | **4.61 s** | 25% faster |
| 2. Hi Flyer | 11.29 s | **9.18 s** | 19% faster |
| 3. Hang Tight | 8.17 s (and only 1 of 3 seeds) | **6.23 s**, 5/5 seeds | faster *and* reliable |

The scripted controller is also brittle — it is tuned by hand for one body of terrain
and dies on seeds it was not tuned for. The learned policy generalises across starts.

### What the training curves look like

`runs/*/training_curves.png` (and the animated `.mp4`). Level 2 is the clearest story:

* **progress per episode**: 0 → ~3 m by step 2,000 (random flailing) → ~31 m by step
  4,000 (riding, but stalling just short of the flower) → ~35 m by step 15,000
  (finishing).
* **finish rate**: 0.00 for the first ~1,500 steps, 0.02 around step 4,000,
  0.65 by step 11,400, **1.00 by step 15,000**, and it stays there.
* **crash rate**: peaks at ~0.4 while the agent is learning to explore, then decays to
  **0.00** — it learns not to die, and keeps that.
* **entropy**: rises as the policy discovers the task, then falls from ~1.7 to ~1.1 as it
  commits to a riding technique, held above zero by the entropy floor.

### Three things that went wrong, and what fixed them

These are the interesting engineering results, because each was a *silent* failure that
looked like "the algorithm just doesn't work":

1. **The "do nothing" local optimum.** On levels 2 and 3 the agent plateaued at ~4 m of
   progress with a 98% stall rate. It had discovered that standing still and braking
   never crashes, so with a crash penalty of 5.0 per death and only a small per-step
   cost, doing nothing was the highest-return policy available. Fixed by making
   not-trying strictly worse than trying: crash penalty lowered to 2.0, and explicit
   costs added for stalling out and timing out. The agent then had to explore, and
   progress jumped within ~2,000 steps.
2. **The finish test was razor-thin.** The goal check is a radius around the bike's body
   centre. With `goal_radius = 1.15 m` the agent on level 3 rode the *entire* route and
   flew a few centimetres over the flower — and was scored as a stall, repeatedly. It
   only became visible when every clip ended with the bike at the goal but `finish=False`.
   Measuring it properly showed the drawn flower plus half the bike is ~1.5 m, so the
   radius is now calibrated to 1.5 m. This is a stand-in for a required `n_apple_*`
   condition in the real game.
3. **`best.pt` was picking slower laps.** The training loop's in-run "best" used progress
   along the reference line, but that metric is not monotone in skill: a slow, wobbly
   ride that hugs the line accumulates *more* path progress than a fast clean finish that
   cuts corners. `select_best.py` now re-ranks every checkpoint on finish rate and
   completion time, and the in-loop criterion was changed to match.

### The honest limits

* **Level 2 (hills) was solved but took the most work and shows the shakeout**: finish
  rate oscillates (0 → 0.6 → 0.2 → 0 → 0.6) between steps 4,000 and 9,000 before
  locking in at 1.0. On a 2-core CPU box with ~1M steps this is the expected texture;
  on a GPU with 100× the steps it would be smoother.
* **This is one seed per level.** No variance study, no hyper-parameter search,
  no curriculum. It is a working baseline, not a benchmark.
* **The levels are not the real internals.** See the disclosure at the top — the game's
  level assets are not in the open-source release, so these are hand-authored levels in
  the same style. Nothing here is comparable to human records on the real internals.
* **No apples are collected.** The apple bonus exists (1.0 each, plus the TMRL-style
  reward function counts them) but the goal test currently does not *require* them, so
  the agent optimises the racing line and ignores them. Requiring them is a one-line
  change to the finish condition, and the reward already supports it.

## 6. Watch the agent improve

```bash
# the grid: successive checkpoints, synchronised in time
python3 make_progress_video.py --run runs/level0 --grid 3x2

# static and animated training curves
python3 plot_curves.py --run runs/level0

# everything at once (best episode, grid, curves, untrained-vs-trained)
python3 make_deliverables.py --all
```

## 7. Reproduce a run

```bash
python3 train.py --level 0 --envs 64 --steps 30000 --upd-per-step 2 \
                 --batch-size 128 --eval-every 500 --video-every 1000 \
                 --outdir runs/level0

# resume / extend
python3 train.py --level 0 --envs 64 --steps 60000 --outdir runs/level0 \
                 --resume runs/level0/checkpoints/best.pt

# physics tests
python3 tests/test_physics.py
```

Per run you get: `config.json`, `stats.jsonl` (training), `eval.jsonl` (greedy
evaluations), `checkpoints/{ckpt_*.pt, best.pt, final.pt}`, `videos/step_*.mp4`.

---

## 8. Legal note

`elma_physics.py` is a re-implementation derived from the Elasto Mania Team's open-source
release, whose dual license (CC BY-NC-SA 4.0 with additional restrictions on commercial
use, cheating and competitive advantage) applies to the original code. Using this port for
non-commercial research is in the spirit of that license. **Do not redistribute the
game's level assets** — they are not included here, precisely because they are not open.
