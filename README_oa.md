# Physical AI with MuJoCo

> **Changing the code?** Add an entry to [`MODIFICHE.md`](MODIFICHE.md) in the
> same session — it records what changed, when, by whom and, above all, *why*.
> The rules are at the top of that file; `CLAUDE.md` carries the same
> instruction for AI tooling.

This repository contains the modular MuJoCo testbed developed for the internship project.

Phase 0A generates a scene from three inputs:

- an object dataset;
- a ground dataset;
- scene rules.

The generated scene is converted to MJCF, simulated until the objects settle, and saved as a reproducible JSON description.

Phase 0B wraps that scene in a Gymnasium environment for the target-extraction task.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project and test dependencies:

```bash
python -m pip install -e ".[test]"
```

## Run it

One entry point, for everything:

```bash
python main.py
```

Or open `main.py` in VS Code and press Run (F5 works too — `.vscode/launch.json`
points at it). A menu asks what to run; every question has a default you accept
by pressing Enter. The scripts below are the non-interactive equivalents, for
use from other scripts or CI.

## Run Phase 0A

Run without rendering:

```bash
python -m scripts.run_phase_0a --seed 42
```

The resolved scene description is saved in `outputs/scenes/`.

## Run Phase 0B

Run with no arguments and it asks two questions — how many objects, and whether
to open the window — then starts:

```bash
python -m scripts.run_phase_0b
```

Every choice is also a flag, to skip the questions:

```bash
python -m scripts.run_phase_0b --objects 5 --render     # window stays open
python -m scripts.run_phase_0b --objects 3 --headless   # text only
python -m scripts.run_phase_0b --obs-mode both --save-stereo outputs/stereo
python -m scripts.run_phase_0b --no-catalogue           # actions only
```

Each episode prints, in this order: the **object catalogue** and the **support
graph** (see the section below) before the policy picks anything, then one line
per action with the object name, reward and disturbance, and finally the
**removal order** actually carried out:

```
    ordine di rimozione: object_002 → object_000 → object_001 (TARGET)
```

That sequence, not the return, is what you compare between policies: it is the
answer the policy gave for this scene. `--no-catalogue` drops the two tables
when running many episodes.

### The Gymnasium viewer

**The window stays open across episodes.** That is possible because `--render`
switches the environment to a fixed compiled model (`resample_shapes=False`):
between episodes the objects get new starting positions, but not new shapes. A
viewer is bound to a compiled model, so recompiling would invalidate it. During
real training use the default instead, which resamples shapes too — more
variety, no window.

**Object count.** `--objects N` cycles through the types listed in
`configs/phase_0b/scene_rules.json` and sizes the spawn area from their largest
bounding radius, so the sampler always finds room. Difficulty rises with N: a
masked random policy succeeds roughly 10/12 at N=2, 8/12 at N=3, 3/12 at N=5,
and essentially never from N=8 up. For large scenes raise
`disturbance_threshold` in `configs/phase_0b/env.json`, or the task is
unwinnable by construction.

On a headless machine set `MUJOCO_GL=egl` (or `osmesa`) for offscreen rendering.

## Record a clean clip (what the docs' images are)

```bash
python -m scripts.record_phase_0b                        # GIF, 3 objects
python -m scripts.record_phase_0b --objects 5 --format mp4
python -m scripts.record_phase_0b --camera cam_left      # the student's view
```

Offscreen `rgb_array` rendering saved with `imageio`: no OS window, no title
bar, no debug overlay — just the 3D canvas. This is how the images on the
Gymnasium site are produced, which is why they look different from a `human`
window even though the scene and the renderer are the same.

Not `gymnasium.wrappers.RecordVideo`, which captures one frame per `env.step()`.
Here a step is a whole removal-and-settle, so that would be a slideshow of
static poses. This script uses the frames the environment captures *during*
settling (`info["frames"]`), dozens per step.

## Check that the physics is sound

```bash
python scripts/verifica_fisica.py                 # eight analytic checks
python scripts/verifica_fisica.py --scene 20      # and the piles themselves
```

Every check has an answer that can be worked out by hand, so a wrong simulator
fails it: free fall against sqrt(2h/g) from three heights, the sliding angle
against atan(mu), the tipping angle against atan(base/height), and the inertia
of each primitive against the compiler's. The static checks build their own
minimal MJCF — they have to measure the *physics*, not our scene generator —
but read the contact parameters from `configs/phase_0b/simulation.json`, so
they measure what the project actually runs. With `--scene N` it also reports,
over real scenes: settling times, objects resting on nothing, contact
interpenetration, and the drift remaining after quiescence is declared.

Run it after touching anything in the `contact` block. It is deliberately small;
the randomised bench with error distributions is described in
`upgrade/banco di prova della fisica.md` and comes later.

## Inspect an episode: catalogue, supports, actions

```bash
python -m scripts.inspect_episode                        # you choose the removals
python -m scripts.inspect_episode --objects 5 --policy top
python -m scripts.inspect_episode --policy random --episodes 50 --quiet
```

The other scripts *run* episodes; this one *explains* them. It prints three
things:

- the **object catalogue** after settling — id, type, mass, friction, size,
  position, and which one is the target. The row index is the action index;
- the **support graph** — who rests on whom, read from MuJoCo's live contacts
  (`data.contact`). This is what determines which removal order is safe, and it
  is exactly the information the agent has to infer on its own. It also reports
  how many objects sit above the target, and warns when the target is already
  on top — in which case the episode does not pose the ordering problem at all;
- the **actions**, step by step, with reward, cumulative disturbance and the
  per-step delta, which is what tells you *which* removal did the damage.

Four policies. `manual` asks you for the index at every step. `random` samples
among the valid actions. `top` always removes the highest object — the obvious
heuristic, and the reference a learned policy has to beat. `target` grabs the
target immediately, which is the lower bound.

Over 20 episodes with 6 objects (seed 0), they separate cleanly:

| policy | success | return | worst single move |
|---|---|---|---|
| `top` | 20/20 | +0.684 ± 0.026 | 0.0030 m |
| `random` | 16/20 | +0.313 ± 0.551 | 0.0349 m |
| `target` | 13/20 | +0.081 ± 0.640 | 0.0498 m |

The target was buried in 18 of the 20 scenes — a property of the *generator*,
not of the policy, and the number to watch when changing how scenes are built.

That separation is the point: an environment where ordering does not change the
outcome cannot teach ordering, and these three numbers are how you check it
still does.

With `--episodes N --quiet` it prints only the summary: success rate, mean
reward with standard deviation, and the fraction of scenes where the target was
actually buried. That last number is a property of the *generator*, not of the
policy.

Also available as entry 6 of `main.py`.

## Watch several episodes at once

```bash
python -m scripts.watch_phase_0b                    # 4 environments, forever
python -m scripts.watch_phase_0b --parallel 6
python -m scripts.watch_phase_0b --episodes 5       # then stop
python -m scripts.watch_phase_0b --save monitor.mp4 # record, no window
```

One window, a grid of N environments running independently, each auto-resetting
into a new scene when its episode ends. `q` closes it.

Why this exists rather than the interactive viewer: Gymnasium's viewer is bound
to a *compiled* model, and this project recompiles the
model on every reset (object shapes and sizes are sampled per scene — which is
what you want for RL). An interactive window would therefore have to close and
reopen every episode. The monitor renders offscreen into an OpenCV window that
does not belong to MuJoCo, so it survives recompilation — and, since it composes
frames itself, it can show several episodes side by side.

Frame capture is what makes the motion visible rather than a series of jumps:
with `capture_camera` set, the environment renders a frame every
`capture_stride` physics steps during settling and returns them in
`info["frames"]`. It costs rendering time and is off by default, so training is
unaffected.

## Run the tests

```bash
pytest
```

## Phase 0B: the environment

`TargetExtraction-v0` is a standard `gymnasium.Env`, registered on import of
`physical_ai_mujoco.envs`.

**Task.** The scene is generated and left to settle. One object is designated as
the target. At every step the agent removes one object; the scene is then left
to settle again. The goal is to remove the target while disturbing the other
objects as little as possible.

**Two ways an episode can end**, selected in `configs/phase_0b/env.json`:

- `terminate_on_target` and `terminate_on_collapse` **false** (the default):
  the episode runs until the scene is empty. This is the **measurement** mode —
  every removal yields a data point (how far the pile moved, with how many
  objects still in it) instead of ending the run on the first move. You cannot
  choose a sensible `disturbance_threshold` without first knowing the
  distribution of disturbances, and you cannot see that distribution if the
  episode stops at step one.
- both **true**: the episode is a task again, ending on target removal
  (success) or when the disturbance exceeds the threshold (collapse).

The disturbance is computed and reported identically either way; only
termination changes. No module knows which mode it is running in.

**Disturbance is per move.** `info["disturbance_step"]` is how far the pile
moved *because of this removal* — the reference poses are re-recorded after
every settle. `info["disturbance_total"]` is the running sum. A single
cumulative figure grows and saturates, and stops telling you which move did the
damage.

**Action.** `Discrete(n_objects)` — which object to remove. Already-removed
objects are invalid actions: they are penalised rather than raising, and a
boolean mask is available both as `env.action_masks()` (the convention
`sb3-contrib`'s `MaskablePPO` looks for) and as `info["action_mask"]`.

**Observation** — selected by `obs_mode`:

| `obs_mode` | Content | Role |
|---|---|---|
| `"state"` | pose, velocity, mass, friction, present flag, target flag per object | **teacher** (trains with RL) |
| `"stereo"` | `rgb_left`, `rgb_right` from the stereo rig | **student** (distilled) |
| `"both"` | all of the above in one dict | **distillation** |

`"both"` is what makes teacher-student possible: a single rollout yields both
the privileged state the teacher acts on and the images the student must learn
to map to that same action. Collecting the two streams from separate rollouts
would not work — contact physics is chaotic, and the trajectories would diverge.

**Stereo rig.** Two parallel (rectified) cameras separated by `baseline`
(default 0.15 m), declared in `configs/phase_0b/env.json` and emitted into the
MJCF as `cam_left` / `cam_right`. Because the cameras are parallel, disparity is
purely horizontal and depth follows `z = f * B / d`, with `f` given by
`StereoCameraDescription.focal_length_px()`. `tests/test_phase_0b.py` checks the
measured pixel disparity against that prediction, so a misbuilt rig fails loudly
instead of silently teaching the student a wrong depth scale.

**Objects are not uniform solids.** `randomisation.center_of_mass_jitter` in
the scene rules offsets each object's centre of mass from its geometric centre
by up to that fraction of the half-extent on each axis (Phase 0B: 0.25). MuJoCo
derives the inertia tensor from geometry and mass on its own, and it is correct
— checked against the analytic formula to 6e-11 — but it puts the centre of mass
at the geometric centre, which is a solid of perfectly uniform density.

For a task decided by what topples when its neighbour is removed, *where* the
mass sits matters more than how much of it there is. It is also a parameter
nobody measures on a real object, which makes randomising it defensible domain
randomisation — unlike frictions and masses, which can be identified from
recorded data instead of guessed. Measured over 15 scenes of 6 objects: the
worst single move rises from 0.0348 m to 0.0443 m at ±25%. Harder scenes, on
purpose.

When the centre of mass is offset the MJCF declares `<inertial>` explicitly,
because the compiler would otherwise derive "uniform" from the geometry. That
makes our own formulas the source of truth, so a test checks them against what
the compiler derives for all three shapes.

**The scene-construction log.** `info["release_log"]` records one row per
drop: mass, release height, how far the centre of mass actually fell, settling
time in simulated seconds and in wall-clock milliseconds, whether quiescence
was reached, what the object ended up resting on, and **how far that drop moved
the objects already in the scene**. That last column is the pile's stability: if
every addition reshuffles everything, the scene is not a pile, it is a dice
roll. Both `inspect_episode` and `run_phase_0b` print it as a table.

**Stackable objects.** The Phase 0B dataset adds `slab`, `plank` and `disc`:
wide flat faces, low centre of gravity. Spheres roll and tall cylinders topple —
asking a pile to form out of those is asking the physics to do something those
shapes do not do. Flat debris is also closer to the real thing: a demining
scene is planks, tiles and sheet metal, not marbles.

**Settling is by quiescence, not by the clock.** After every release and every
removal the simulation runs until nothing has moved for `stable_duration`, or
until the timeout. Running for a fixed duration is wrong: the fall time depends
on the release height, and reading poses while objects are still moving records
motion nobody caused as "disturbance". `info["settled"]` is false when the
timeout ran out with something still moving — the poses behind that reading are
not a stable configuration, and whoever is measuring needs to know.

**"Has not moved" is a displacement, not a velocity.** The tolerances in
`configs/phase_0b/simulation.json` are `linear_tolerance` (1 mm) and
`angular_tolerance` (1°), applied over the `stable_duration` window: an object
counts as at rest while it stays within them of where it was. The earlier
velocity thresholds looked more direct and were wrong. Contacts hand a resting
object angular-velocity spikes of 0.1–0.4 rad/s lasting a few steps, which do
not move it; each spike reset the counter, so a scene that had been still for
seconds was reported as never settled. Measured over 120 settles: on the ones
declared failed, objects were moving **0.14 mm per second**. No velocity
threshold can separate those spikes from a real rotation — 0.4 rad/s is 23 °/s,
and a threshold that admits them admits an object that is genuinely turning.
Averaging over time is the only separation, and averaging over time means
looking at displacement.

**Contact parameters.** The `contact` block of the same file sets an `elliptic`
friction cone with `impratio` 10, `noslip_iterations` 10, and a contact time
constant of 0.01 s. With MuJoCo's defaults, objects that look at rest creep: the
residual drift *after* quiescence was declared had a median of **1.5 mm/s and a
worst case of 38 mm/s**, which over a twenty-second settle is the 24 cm an
object was once seen to travel "while stationary". With these parameters the
median drift is **0.018 mm/s**, contact interpenetration is 0.21 mm, and the
median settle is 1.0 s instead of 1.5 s. `noslip_iterations` is MuJoCo's
documented remedy for the slip inherent to soft constraints; its documentation
warns it can destabilise multi-contact models, which is exactly this case, so it
was measured rather than trusted. Raising the solver's iteration count changes
nothing — the solver already converges.

**Objects are released one at a time.** Each falls from a few centimetres above
the current top of the pile, aiming at it; the objects released after the target
aim at the *target*, so burying it is a property of the construction rather than
a lucky outcome. Releasing everything at once forces the initial poses apart —
otherwise they interpenetrate — which means either spreading the scene, and then
no pile forms, or raising the column, and then they fall from metres and the
impact scatters them. Neither happens one at a time.

**The ground is a floor, not a table.** `floor_infinite` is a MuJoCo `plane`:
infinite for collisions, so nothing can slide off an edge and fall into the
void, which would otherwise register as a huge displacement and pollute the
measurement. It also decouples the spawn area from the ground size — where an
object may be released is now a scene rule and nothing else.

**Phase 0B has its own datasets and simulation settings** (`datasets` in
`env.json`). The object dataset raises rolling and torsional friction from
0.005 to 0.05/0.025: at Phase 0A's value a cylinder landing on an edge rolls
away for tens of centimetres and no pile ever forms. The settling timeout is
raised from 5 s to 20 s, because each drop is now awaited separately. Phase 0A
keeps its own files and stays byte-identical for a given seed.

**Object removal.** MuJoCo cannot delete a body from a compiled model. Rather
than recompiling on every removal — which would also change the size of the
observation and action spaces mid-episode — a removed object is excluded from
contacts, made invisible, and held at a parking position outside every camera's
field of view. The observable result is identical to deletion, and the spaces
stay fixed.

## Configuration

| File | Controls |
|---|---|
| `configs/phase_0a/scene_rules.json` | Phase 0A scene: which objects, where they spawn |
| `configs/phase_0a/simulation.json` | gravity, timestep, integrator, settling thresholds |
| `configs/phase_0b/scene_rules.json` | Phase 0B scene: a tight column, sequential release |
| `configs/phase_0b/simulation.json` | Phase 0B settling: same thresholds as 0A, longer timeout |
| `configs/phase_0b/env.json` | stereo rig, task parameters, termination rules, dataset paths |

Cameras in the MJCF: `cam_left` / `cam_right` are the student's stereo pair,
`cam_overview` is a fixed wide shot used only for visualisation.

Phase 0B has its own scene rules on purpose: Phase 0A output is unchanged by
everything in this phase, and stays byte-identical for a given seed.

## Current scope

Phase 0A includes:

- parametric geometric objects;
- a configurable ground;
- deterministic scene generation;
- MJCF construction;
- MuJoCo physics;
- settling detection;
- perfect final state reading.

Phase 0B includes:

- the Gymnasium environment;
- the action mask;
- reward, target and episode metrics;
- the stereo rig and the three observation modes for teacher-student;
- the parallel grid monitor.

Not yet done: the teacher policy itself, the distillation loop, domain
randomisation of appearance, and deformable objects (DLO).
