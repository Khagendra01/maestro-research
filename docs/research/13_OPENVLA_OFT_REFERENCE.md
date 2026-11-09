# OpenVLA-OFT learned-policy reference — LIBERO dev smoke

Date: 2026-11-09. Result: **15/15 official successes** (object 5/5, spatial
5/5, goal 5/5). This is the learned-policy reference on the same 15 declared
episodes as the modular oracle and VLM runs. It is a Setting A policy (RGB +
wrist RGB + proprioception; no depth, no simulator poses). For the later
full-suite evaluation, see
[`14_OPENVLA_OFT_FULL_SUITE_RESULTS.md`](14_OPENVLA_OFT_FULL_SUITE_RESULTS.md).

## Provenance and setup

- Policy: OpenVLA-OFT 7B, **suite-specific** checkpoints
  `moojink/openvla-7b-oft-finetuned-libero-{spatial,object,goal}`. The combined
  `spatial-object-goal-10` checkpoint was not run.
- Code: OpenVLA-OFT `e4287e94541f459edc4feabc4e181f537cd569a8` (the same commit
  named as the protocol reference in the frozen dev manifest), plus its custom
  `transformers` 4.40.1 fork; LIBERO `8f1084e3132a39270c3a13ebe37270a43ece2a01`.
- Runtime: Python 3.10, PyTorch 2.2.0+cu121, TensorFlow 2.15.0, MuJoCo 2.3.7,
  **robosuite 1.4.1** (the OpenVLA-OFT requirement; the modular runs use the
  manifest's 1.4.0), EGL rendering, NVIDIA A100-SXM4-40GB.
- Evaluation: a thin runner calls the unmodified OpenVLA-OFT eval primitives
  (`initialize_model`, `prepare_observation`, `get_action`, `run_episode`) with
  `center_crop=True`, `num_images_in_input=2`, `use_proprio=True`,
  `use_l1_regression=True`, `num_open_loop_steps=8` (the action-chunk size),
  `num_steps_wait=10`, seed 7. The task selection is by exact LIBERO task name;
  initial-state indices 0..4 are the frozen manifest's indices. The official
  success is the environment's `done` flag from `env.step`.
- Checkpoints and artifacts: Modal volume `maestro-research-data`;
  `/data/checkpoints/openvla-oft-libero-*`, `/data/runs/dev_oft_r1-oft/`.
  The checkpoints are public HF repos; no token is stored in the repository.
- Runner: `maestro_research/oft_reference.py`; Modal image/function:
  `maestro_research/modal_oft.py`.

The image needed explicit pins to reproduce the OFT environment:
`protobuf==4.21.6` and `tensorflow-metadata==1.17.0` (the default resolver
picks a `tensorflow-metadata` whose generated protobuf code conflicts with
TensorFlow 2.15). `clang` is installed for the `evdev` wheel pulled by
robosuite 1.4.1.

## Results

| suite | episodes | successes | rate |
|---|---:|---:|---:|
| libero_object | 5 | 5 | 1.0 |
| libero_spatial | 5 | 5 | 1.0 |
| libero_goal | 5 | 5 | 1.0 |
| **total** | **15** | **15** | **1.0** |

Per episode. `steps` excludes the 10 stabilization steps. `policy` is total
time inside the policy calls (action chunks); `sim_other` is wall minus policy
(environment stepping, rendering, observation preprocessing and bookkeeping).

| episode | official | steps | wall (s) | policy (s) | sim/other (s) | policy calls |
|---|---:|---:|---:|---:|---:|---:|
| object alphabet soup 0 | True | 148 | 18.6 | 2.9 | 15.7 | 19 |
| object alphabet soup 1 | True | 137 | 8.9 | 2.0 | 6.9 | 18 |
| object alphabet soup 2 | True | 134 | 8.6 | 1.9 | 6.7 | 17 |
| object alphabet soup 3 | True | 144 | 9.4 | 2.1 | 7.3 | 18 |
| object alphabet soup 4 | True | 144 | 8.9 | 2.0 | 6.9 | 18 |
| spatial center bowl 0 | True | 95 | 7.9 | 1.4 | 6.5 | 12 |
| spatial center bowl 1 | True | 98 | 7.8 | 1.5 | 6.4 | 13 |
| spatial center bowl 2 | True | 91 | 7.5 | 1.3 | 6.2 | 12 |
| spatial center bowl 3 | True | 94 | 7.7 | 1.4 | 6.3 | 12 |
| spatial center bowl 4 | True | 94 | 7.7 | 1.4 | 6.3 | 12 |
| goal bowl on plate 0 | True | 75 | 6.1 | 1.1 | 5.0 | 10 |
| goal bowl on plate 1 | True | 75 | 6.1 | 1.2 | 5.0 | 10 |
| goal bowl on plate 2 | True | 76 | 6.1 | 1.1 | 5.0 | 10 |
| goal bowl on plate 3 | True | 68 | 5.6 | 1.0 | 4.6 | 9 |
| goal bowl on plate 4 | True | 78 | 6.2 | 1.1 | 5.0 | 10 |

Model load (bf16 7B + action head + proprio projector + processor) took 21.0 s
(object), 22.5 s (spatial), and 20.0 s (goal). Mean episode wall time was 8.2 s
(median 7.7 s); mean policy time 1.57 s (range 1.0–2.9 s); mean 103.4 steps.
The first object episode is the slow outlier (15.7 s sim/other) due to
first-use initialization; later object episodes are 6.7–7.3 s.

## Cross-setting comparison on the frozen 15 episodes

| setting | policy/perception | episodes | successes | steps | policy/VLM time |
|---|---|---:|---:|---|---|
| C (oracle geometry) | scripted geometry + planner | 15 | 15 | 98–149 | n/a |
| A (learned reference) | OpenVLA-OFT 7B, RGB+wrist+proprio | 15 | 15 | 68–148 | 1.0–2.9 s |
| B (VLM selector) | Meta Muse Spark + depth patch, 3 repeats | 45 | 15 | 98–189 | 10.2–52.7 s/call |

These are not the same observation access or interface, so the table is a
task-level status check, not an apples-to-apples method comparison. The oracle
and VLM use the modular executor; OpenVLA-OFT uses its own end-to-end eval
loop. The VLM selector is not a visuomotor policy.

## Reading

- The benchmark and initial states are solvable: the learned reference gets
  15/15 with a 1.6 s mean policy time and 8.2 s mean episode wall time. The
  modular oracle also gets 15/15, so the controller/planner side is not the
  bottleneck on this smoke.
- The VLM selector's 15/45 is a perception/grounding failure, not evidence that
  the modular executor is weak: when the oracle supplies geometry, the same
  executor is 15/15.
- OpenVLA-OFT is fast because it executes 8-step action chunks and does ~9–19
  policy calls per episode; the simulator/render still dominates wall time.
- The reference has no depth and no explicit target/receptacle geometry; it
  is the learned-policy baseline the modular stack is measured against.

## Limitations

- One seed (`seed=7`), 15 episodes, 3 tasks. The OFT paper averages three seeds
  over the full suites (10 tasks × 50 episodes); this smoke is not a
  leaderboard estimate.
- robosuite 1.4.1 (OFT's official requirement) vs 1.4.0 in the frozen modular
  manifest. The LIBERO commit, tasks, initial-state indices and official
  success condition are the same. Disclosed rather than silently mixed.
- The reference's suite-specific checkpoints were used, not the combined
  `spatial-object-goal-10` checkpoint.
- The OFT image was built from PyPI/git dependencies at run time; pins that
  mattered (`protobuf`, `tensorflow-metadata`, `robosuite`, MuJoCo) are listed
  above. The checkpoint files are public and were downloaded once to the
  volume.
- No training, fine-tuning, or checkpoint merging was performed.
- The "same interface" part of the first milestone is only task-level: the
  reference is evaluated by its own script, not behind the modular
  `TaskDecision`/executor API. This is a remaining integration item, not a
  hidden claim.

## Reproduce

```bash
# one-time checkpoint download (public HF repos; CPU)
modal run maestro_research/modal_oft.py::download_checkpoints

# GPU/environment probe (EGL render, task names, checkpoints)
modal run maestro_research/modal_oft.py::probe_oft --gl egl

# 15-episode learned-policy reference
modal run maestro_research/modal_oft.py::run_oft_smoke --limit 5 --run-name dev_oft_r1
```

Artifacts: `/data/runs/dev_oft_r1-oft/summary.json`, `episodes.jsonl`,
`frames/`; local copy used here: `/tmp/oft_r1/dev_oft_r1-oft`.
