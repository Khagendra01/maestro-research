# LIBERO dev smoke r2 — Setting C (oracle geometry)

Date: 2026-11-08. Result: **15/15 official successes** on the declared development
smoke (3 tasks × 5 init states). This is an oracle-geometry diagnostic, not a
perception result and not a leaderboard-comparable number.

## Provenance

- Code: MAESTRO `e60d326233a238ddc496bf7a08acb7a278e9e66d`, working branch
  `research/fast-vlm-geometric-control`.
- Simulator: LIBERO `8f1084e3132a39270c3a13ebe37270a43ece2a01`, robosuite 1.4.0,
  MuJoCo 2.3.7, OSMesa CPU rendering, Modal image `maestro-research-libero`.
- Run: Modal function `run_smoke`, volume `maestro-research-data`, remote path
  `/data/runs/dev_r2_oracle_v6-oracle` (local client label `dev_r2_oracle_v6`).
  The container was preempted once and resumed by Modal; the completed run is v6.
- Manifest `libero-dev-smoke-r2`, sha256
  `b8ca154ecbe8595f3823f04965619421ed927bbb8590bf6c10d0103739fc6839`.
- Protocol: OpenVLA-OFT horizons (spatial 220, object 280, goal 300), 20 Hz
  OSC_POSE, 7-D delta action, gripper `-1` open / `+1` close; `num_steps_wait=10`
  is applied before policy stepping and is not counted in `steps`.
- Selector: `scripted_oracle_control` (task-vocabulary parser plus simulator
  object identity; no simulator poses enter the selector).
- Success: official LIBERO success check held for ≥1 s after placement, plus the
  internal placement check. All 15 episodes report `official_success=True`.

## Setting declaration

Setting C (see `OBSERVATION_MANIFEST.md`): simulator body poses and meshes are
used for the target/receptacle, for grasp proposals, and for internal
verification. No RGB, depth, segmentation, or VLM is used in this run. These
numbers isolate geometry + planning + execution and **must not** be compared to
RGB-policy leaderboard results.

## Results

| suite | episodes | successes | rate |
|---|---:|---:|---:|
| libero_object | 5 | 5 | 1.0 |
| libero_spatial | 5 | 5 | 1.0 |
| libero_goal | 5 | 5 | 1.0 |
| **total** | **15** | **15** | **1.0** |

Per-episode outcome (steps are simulated control steps):

| episode | official | steps | grasp attempts | placement xy error (m) |
|---|---|---:|---:|---:|
| object state 0 | True | 129 | 1 | 0.0150 |
| object state 1 | True | 130 | 1 | 0.0150 |
| object state 2 | True | 129 | 1 | 0.0165 |
| object state 3 | True | 130 | 1 | 0.0170 |
| object state 4 | True | 129 | 1 | 0.0155 |
| spatial state 0 | True | 104 | 1 | 0.0057 |
| spatial state 1 | True | 106 | 1 | 0.0096 |
| spatial state 2 | True | 114 | 1 | 0.0136 |
| spatial state 3 | True | 119 | 1 | 0.0144 |
| spatial state 4 | True | 103 | 1 | 0.0072 |
| goal state 0 | True | 103 | 1 | 0.0048 |
| goal state 1 | True | 149 | 2 | 0.0050 |
| goal state 2 | True | 100 | 1 | 0.0032 |
| goal state 3 | True | 100 | 1 | 0.0021 |
| goal state 4 | True | 98 | 1 | 0.0023 |

The internal placement tolerance is 0.03 m; all errors are well inside it. Goal
state 1 needed a second attempt (drop variant `bowl_grasp_drop-0.007`) and still
finished inside the 300-step horizon.

## Changes that moved the smoke from 12/15 to 15/15

1. **Safe lift before lateral transport.** After grasp verification, the eef now
   moves vertically at the current xy to
   `cruise_z = max(grasp_z + 0.25, receptacle_top + place_clearance)` and only
   then translates over the receptacle, descending at the end (`LIFT_SAFE` +
   `TRANSPORT` + `PLACE`). In the 12/15 run, goal states 1/3/4 hit
   `transport_move_timeout` with a 0.041–0.050 m tracking error because the
   diagonal path clipped scene geometry. With the safe lift, transport succeeded
   and the residual failures moved to the descent.
2. **Canonical bowl-rim side (`grasp_side=-1`).** The probe shows the top-down
   eef's local +y finger axis is world **−y**, so `grasp_side=-1` selects the
   world **+y** rim. The previous rule chose the side from the *current* eef y
   relative to the bowl y; because the pregrasp eef y is essentially the bowl y,
   the sign was noise and flipped across init states. The −y-side placements put
   the wrist/forearm over the microwave/stove and blocked the descent
   (`place_move_timeout`, 0.117–0.147 m residual); the world +y side keeps the
   descent clear.
3. **Geometry and tracking constants.** Collision-geometry `body_extent` with
   `radius_top` (not a bounding sphere), `kp=18.0`, per-stage tolerances
   (approach ≥0.02, lift 0.02, transport 0.025, place 0.02, retreat 0.02),
   12-step settle, drop variants on retry (`base, base−0.007, base+0.003`),
   placement offset `place_xy = receptacle_xy + (grasp_xy − bowl_center_xy)`,
   and a 25-step release settle before scoring.

Development evidence (same manifest, same image):

| run | change | success |
|---|---|---:|
| `dev_r2_oracle` | pre-safe-lift | 12/15 (goal 2/5) |
| `dev_side_pos` | safe lift, `grasp_side=+1` (world −y) | 9/15 (goal 0/5) |
| `dev_r2_oracle_v3` | safe lift, clearance-based side | 10/15 (goal 0/5) |
| `dev_side_neg` | safe lift, `grasp_side=-1` (world +y) | 15/15 |
| `dev_r2_oracle_v6` | default `grasp_side=-1` | 15/15 |

The clearance-based side heuristic was tried and removed: distance to body
centres does not capture the wrist/forearm clearance that actually matters.

## Resolved policy constants (r2)

| constant | value |
|---|---:|
| `kp` | 18.0 |
| `position_tolerance` | 0.015 |
| `settle_steps` | 12 |
| `approach_height` | 0.12 |
| `bowl_grasp_drop` | 0.022 |
| `center_grasp_drop` | 0.025 |
| `bowl_rim_fraction` | (from `radius_top`, not `radius`) |
| `grasp_side` | −1 (world +y) |
| `min_finger_separation` | 0.003 |
| `place_clearance` | 0.15 |
| `cruise_z` | `max(grasp_z + 0.25, receptacle_top + place_clearance)` |
| `place_z` | `receptacle_top + target_half_height + 0.005` |
| `retreat_height` | 0.15 |
| `verify_placement_xy` | 0.03 |
| `release_settle_steps` | 25 |
| `grasp_retries` | 2 (three drop variants) |
| per-stage tolerance | approach max(1.5·tol, 0.02), lift/place/retreat 0.02, transport 0.025 |

`stage_errors` in `summary.json` holds the final tracking error per stage (the
field was renamed from the misleading `stage_timings`).

## Limitations

- Oracle geometry only (Setting C); no perception claim.
- 15 declared development episodes, 3 tasks; state 1 exercised the retry path
  once. This is a smoke, not an evaluation-suite result.
- Setting B (RGB-D) is implemented but not validated for same-colour classes and
  was not run here; no VLM/API credentials are available locally, so Setting A
  is not implemented.
- OpenVLA-OFT (7B) reference was not attempted yet.
- The libero_object scene is a floor scene (no table), unlike spatial/goal.
- `cruise_z` adds 0.25 m above the grasp; the arm's reachable workspace and the
  max horizon were not stress-tested beyond these tasks.

## Reproduce

```bash
python3 -m unittest discover -s tests -p 'test_research_*.py'
modal run maestro_research/modal_libero.py::run_smoke --limit 5 --run-name dev_r2_oracle_v6
```

Artifacts on the volume: `summary.json`, `episodes.jsonl`, and `frames/` under
`/data/runs/dev_r2_oracle_v6-oracle`. Local copy used for this report:
`/tmp/dev_v6/dev_r2_oracle_v6-oracle`.
