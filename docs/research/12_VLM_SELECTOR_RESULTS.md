# VLM selector on the LIBERO dev smoke (Setting B, RGB-D + Meta Muse Spark)

Date: 2026-11-08. Result: **15/45 official successes across three full repeats**
(object 0/15, spatial 9/15, goal 6/15). The best and worst repeats were 6/15 and
3/15. This is a real API-grounded modular-perception result, **not** a learned
visuomotor policy and **not** a Setting A / leaderboard-comparable number.

## What was run

The selector is a VLM call that returns two normalized image points (target and
receptacle) plus short labels. Depth back-projects each point to 3-D; a local
point-cloud patch supplies the object extent (center, top/bottom, radius,
`radius_top`). The existing scripted `StagedExecutor` then plans and executes the
grasp/lift/transport/place. The VLM never returns metric coordinates, never sees
simulator poses, segmentation IDs, BDDL goal facts, or the success signal.

- Model: `muse-spark-1.3-contributor` through the Meta Model API
  (`https://api.meta.ai/v1/chat/completions`).
- Credential: Modal secret `meta-model-api` (`MODEL_API_KEY`), attached only to
  the Modal `smoke` function. The key is not in the repository.
- Prompt: task instruction + agentview RGB; JSON only; origin top-left, `x`
  right, `y` down; object centers; normalized points.
- Decoding: `temperature=0.0`, `max_tokens=5000`, one retry on empty content,
  240 s timeout. The model's reasoning consumes output tokens; 3000 was raised
  to 5000 after one empty-content failure.
- Geometry: target patch radius 0.07 m, receptacle patch radius 0.10 m, z window
  ±0.16 m around the seed; median-cloud extent. Depth is sampled at the top-left
  row of the VLM point (the depth buffer is top-left aligned; no vertical flip).
- Observation setting: **B** (RGB-D modular perception). Depth is the declared
  extra modality. No task-vocabulary parser or colour prior is used in this
  setting; the VLM supplies the semantic binding. Setting C (oracle) is the
  15/15 planning-isolation result in `10_DEV_SMOKE_RESULTS_R2.md`.

## Results

Manifest `libero-dev-smoke-r2` (3 tasks × 5 init states), official LIBERO
success, same protocol/horizons as the oracle run.

| run (Modal label) | object | spatial | goal | total |
|---|---:|---:|---:|---:|
| `dev_vlm_full-vlm` (first) | 0/5 | 3/5 | 3/5 | **6/15** |
| `dev_vlm_full3-vlm` (repeat) | 0/5 | 2/5 | 1/5 | **3/15** |
| `dev_vlm_full5-vlm` (repeat) | 0/5 | 4/5 | 2/5 | **6/15** |
| **pooled** | **0/15** | **9/15** (60%) | **6/15** (40%) | **15/45** (33.3%) |

The oracle-geometry control is 15/15 on the same manifest. The gap is entirely
perception/grounding; planning and execution are unchanged between the oracle
and VLM runs.

### Final repeat (`dev_vlm_full5-vlm`), per episode

`off` is official success. `status` is the executor's internal status; in
non-oracle settings placement cannot be simulator-verified, so a completed
episode is `ok` / `placement_unverifiable_non_oracle` even when the official
check later fails. Only `off` is scored.

| episode | off | status / reason | steps | VLM s | target point | receptacle point |
|---|---|---|---|---:|---:|---|---|
| object alphabet soup 0 | False | failed / grasp_not_retained | 156 | 24.7 | (0.255, 0.345) | (0.805, 0.370) |
| object alphabet soup 1 | False | failed / place_move_timeout | 188 | 23.3 | (0.540, 0.340) | (0.760, 0.410) |
| object alphabet soup 2 | False | failed / place_move_timeout | 189 | 25.8 | (0.540, 0.340) | (0.750, 0.410) |
| object alphabet soup 3 | False | failed / grasp_not_retained | 169 | 35.7 | (0.250, 0.350) | (0.800, 0.470) |
| object alphabet soup 4 | False | failed / transport_move_timeout | 170 | 18.3 | (0.540, 0.300) | (0.790, 0.380) |
| spatial center bowl 0 | False | failed / approach_move_timeout | 103 | 22.0 | (0.360, 0.540) | (0.760, 0.270) |
| spatial center bowl 1 | True | ok / placement_unverifiable | 101 | 17.0 | (0.498, 0.422) | (0.742, 0.270) |
| spatial center bowl 2 | True | ok / placement_unverifiable | 102 | 19.1 | (0.510, 0.420) | (0.770, 0.280) |
| spatial center bowl 3 | True | ok / placement_unverifiable | 104 | 17.4 | (0.510, 0.420) | (0.760, 0.280) |
| spatial center bowl 4 | True | ok / placement_unverifiable | 100 | 19.2 | (0.496, 0.426) | (0.734, 0.254) |
| goal bowl on plate 0 | True | ok / placement_unverifiable | 106 | 12.0 | (0.490, 0.440) | (0.485, 0.270) |
| goal bowl on plate 1 | False | failed / grasp_not_retained | 111 | 20.4 | (0.520, 0.340) | (0.450, 0.180) |
| goal bowl on plate 2 | False | failed / grasp_not_retained | 115 | 10.2 | (0.500, 0.345) | (0.480, 0.180) |
| goal bowl on plate 3 | True | ok / placement_unverifiable | 98 | 14.2 | (0.498, 0.433) | (0.470, 0.295) |
| goal bowl on plate 4 | False | failed / grasp_not_retained | 103 | 13.3 | (0.515, 0.440) | (0.485, 0.295) |

Failure counts across all three repeats: 18 `grasp_not_retained`, 6
`place_move_timeout`, 4 `approach_move_timeout`, 1 `transport_move_timeout`,
plus 1 executor-completed episode that failed the official check; 15 official
successes. All three repeats use the same geometry configuration.

### Latency and cost

Per-episode VLM call latency (the logged latency is the successful call; a
retried empty call is not included):

| run | mean | median | min | max | sum over 15 |
|---|---:|---:|---:|---:|---:|
| first | 19.8 s | 19.4 s | 10.7 s | 27.4 s | 297.6 s |
| repeat 1 | 26.5 s | 23.8 s | 13.7 s | 52.7 s | 397.9 s |
| repeat 2 | 19.5 s | 19.1 s | 10.2 s | 35.7 s | 292.7 s |

In the final repeat the mean per-episode wall time (reset + VLM + execution) was
44.4 s (median 43.4 s); VLM calls were 44% of wall time. Simulator work was
98–189 control steps at 20 Hz (4.9–9.5 s), so the remote VLM call, not the
controller, dominates wall time. Latency varied even at `temperature=0`; the
repeat-1 outlier was 52.7 s.

## Failure analysis (honest)

1. **Semantic mis-grounding.** The VLM is not reliable on this scene. Spatial
   state 0 and some repeats point at the stove burner (0.36, 0.54) rather than
   the centre bowl; goal states 1/2 point at the plate/table (y ≈ 0.34) rather
   than the bowl (y ≈ 0.44); the object scene alternates between the two
   alphabet-soup cans at x ≈ 0.25 and x ≈ 0.54. The basket/plate receptacle
   points were stable and usually correct.
2. **Single-view surface depth is a weak extent estimator.** For a hollow bowl
   the VLM point samples the inner surface; the local patch median is biased
   ~1.5–2.5 cm toward the sampled side (e.g. goal state 0 patch center
   x ≈ −0.115 vs the oracle body center x ≈ −0.098). The rim grasp then misses.
   The oracle run uses collision-body extents and does not have this bias.
3. **Object scene is unsolved (0/15).** The thin cans on the floor and the large
   wire basket produce grasp misses and place/transport timeouts. Perceived
   basket/rim geometry plus the arm's reachable workspace is not sufficient here;
   no object episode succeeded in any repeat.
4. **Controller timeouts are a secondary effect.** `approach_move_timeout` occurs
   when the chosen point is unreachable or wrong (stove); `place_move_timeout` /
   `transport_move_timeout` occur after a bad grasp or with an over-estimated
   receptacle extent. Successes took 98–106 steps; failures 103–189, within the
   official horizons (spatial 220, object 280, goal 300) but often at the
   controller's own stage limits.
5. **Non-oracle placement verification.** The executor deliberately does not
   call the success oracle. In `vlm` and `rgbd` settings it returns
   `placement_unverifiable_non_oracle`; official success is evaluated afterwards
   by the runner. Executor `status=ok` must not be read as task success (one
   object episode completed the executor but failed the official check).

### Estimator iterations that did not help

Two attempts to fix the bowl-center bias were run and rejected: (a) snapping
the VLM point to the nearest colour-agnostic depth cluster, and (b) a top-rim
ring center. Both scored **0/15** (the cluster step selected the arm/cabinet or
merged objects; the top-ring center was unstable), and were reverted. The
reported configuration is the median local-patch extent. The VLM point variance
itself remains; three repeats are a smoke, not a generalization estimate.

## Contract and fixtures

The provider boundary is `maestro_research/perception/vlm.py`:
`call_vlm(instruction, rgb)` returns parsed JSON plus `latency_seconds`,
`finish_reason`, `usage`, `model`, and raw `content`; `build_scene_from_vlm`
returns the target/receptacle positions, extents, `TaskDecision`, and the full
metadata block logged under `record["vlm"]`. Invalid JSON, out-of-range points,
missing objects, missing depth, and too-few-point patches raise `VLMError`.
Empty content is retried once.

The prompt is a fixture, not a learned policy: changing the model, prompt, or
`max_tokens` requires a new declared run label. The API key was provided
out-of-band and expires; reruns need a fresh Modal secret.

## Reproduce

```bash
# oracle control (no API call)
modal run maestro_research/modal_libero.py::run_smoke --settings oracle --limit 5 --run-name dev_oracle_repeat

# VLM selector; requires the meta-model-api secret
modal run maestro_research/modal_libero.py::run_smoke --settings vlm --limit 5 --run-name dev_vlm_repeat

python3 -m unittest discover -s tests -p 'test_research_*.py'
```

Artifacts on volume `maestro-research-data`: `/data/runs/<run-name>-vlm/`
(`summary.json`, `episodes.jsonl`, `frames/`). Example local copy:
`/tmp/dev_vlm5/dev_vlm_full5-vlm`.

## What this does not establish

- It is not a learned-policy reference. The OpenVLA-OFT 7B reference required by
  the first milestone has not been run (needs the OFT repo, 7B LIBERO
  checkpoints, and a Modal A100 run). No learned-policy number exists yet.
- It is not a Setting A result (depth is used) and not a leaderboard comparison.
- n=45 over three repeats of the same 15 states; no confidence interval or
  per-initial-state generalization claim.
- The VLM key expires; the numeric result is tied to this model snapshot.
