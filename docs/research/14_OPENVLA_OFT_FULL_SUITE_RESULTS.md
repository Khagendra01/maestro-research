# OpenVLA-OFT full-suite reference — LIBERO Spatial, Object, and Goal

Date: 2026-11-09. Result: **1471/1500 official successes (98.07%)** in one
evaluation seed. The three suites used the suite-matched public checkpoints:

| suite | episodes | successes | rate |
|---|---:|---:|---:|
| `libero_spatial` | 500 | 491 | 0.982 |
| `libero_object` | 500 | 491 | 0.982 |
| `libero_goal` | 500 | 489 | 0.978 |
| **total** | **1500** | **1471** | **0.98067** |

This is a paper-style full-suite evaluation, but with only one evaluation seed
(`seed=7`) and without LIBERO-10/Long. It is the same Observation Setting A
policy as the 15-episode reference in
[`13_OPENVLA_OFT_REFERENCE.md`](13_OPENVLA_OFT_REFERENCE.md): RGB + wrist RGB +
proprioception, no depth, no simulator poses, no segmentation IDs, no BDDL
facts, and no success-signal input.

## Provenance and setup

- Policy checkpoints:
  - `moojink/openvla-7b-oft-finetuned-libero-spatial`
  - `moojink/openvla-7b-oft-finetuned-libero-object`
  - `moojink/openvla-7b-oft-finetuned-libero-goal`
- Code and protocol: OpenVLA-OFT `e4287e94541f459edc4feabc4e181f537cd569a8`,
  custom `transformers` 4.40.1 fork, LIBERO
  `8f1084e3132a39270c3a13ebe37270a43ece2a01`.
- Runtime: Python 3.10, PyTorch 2.2.0+cu121, TensorFlow 2.15.0, MuJoCo 2.3.7,
  **robosuite 1.4.1** (OpenVLA-OFT's stated requirement; the modular runs use
  the manifest's 1.4.0), EGL rendering, NVIDIA A100-SXM4-40GB.
- Policy inputs and execution: center crop, two images, proprioception,
  L1-regression continuous action head, 8-step action chunks,
  `num_steps_wait=10`; suite horizons 220/280/300.
- Evaluation: the same unmodified OpenVLA-OFT primitives as the smoke run
  (`initialize_model`, `prepare_observation`, `get_action`, `run_episode`).
  One model was loaded per suite and reused for all ten tasks and all 50
  official initial states. Official success is the environment `done` signal.
- Run identity: Modal function `run_oft_smoke --full-suite --limit 50
  --run-name dev_oft_full_r1`; remote output directory
  `/data/runs/dev_oft_full_r1-oft`, local copy `/tmp/oft_full_r1/dev_oft_full_r1-oft`.

The Modal image required `protobuf==4.21.6` and
`tensorflow-metadata==1.17.0`; the default newer metadata build generates a
protobuf import that conflicts with TensorFlow 2.15.

## Timing

Across all 1500 episodes:

- Mean 118.1 steps, 11.85 s wall time, 2.07 s policy time, 15.21 policy calls,
  9.79 s simulator/render/other time.
- Medians: 117.0 steps, 11.57 s wall, 2.04 s policy, 15 calls, 9.54 s
  simulator/render/other.
- Policy inference was 17.4% of episode wall time. Model loading took 17.08 s
  (spatial), 17.02 s (object), and 18.25 s (goal).
- Remote logs spanned 21:14:51 to 02:16:34 PT, 5.03 elapsed hours, including
  image startup, three model loads, 1500 episodes, and volume commits.

Mean performance by suite was 12.01 s wall / 1.80 s policy (spatial), 12.59 s
wall / 2.43 s policy (object), and 10.96 s wall / 1.97 s policy (goal).

## Tasks

Fifteen of 30 tasks were perfect. Every failure below consumed its full
suite horizon.

| suite | task | successes |
|---|---|---:|
| spatial | pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 49/50 |
| spatial | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 47/50 |
| spatial | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 47/50 |
| spatial | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 49/50 |
| spatial | pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 49/50 |
| object | pick_up_the_cream_cheese_and_place_it_in_the_basket | 49/50 |
| object | pick_up_the_salad_dressing_and_place_it_in_the_basket | 48/50 |
| object | pick_up_the_tomato_sauce_and_place_it_in_the_basket | 49/50 |
| object | pick_up_the_milk_and_place_it_in_the_basket | 49/50 |
| object | pick_up_the_chocolate_pudding_and_place_it_in_the_basket | 46/50 |
| goal | open_the_middle_drawer_of_the_cabinet | 48/50 |
| goal | put_the_bowl_on_the_stove | 47/50 |
| goal | open_the_top_drawer_and_put_the_bowl_inside | 46/50 |
| goal | put_the_bowl_on_top_of_the_cabinet | 49/50 |
| goal | put_the_wine_bottle_on_the_rack | 49/50 |

The remaining 15 tasks were 50/50.

Failure-state indices:

- Spatial:
  - Cookie-box black bowl: state 16.
  - Top-drawer black bowl: states 20, 29, 32.
  - Ramekin black bowl: states 38, 41, 49.
  - Stove black bowl: state 32.
  - Wooden-cabinet black bowl: state 22.
- Object:
  - Cream cheese: state 27.
  - Salad dressing: states 15, 48.
  - Tomato sauce: state 17.
  - Milk: state 8.
  - Chocolate pudding: states 5, 15, 24, 26.
- Goal:
  - Middle drawer: states 38, 45.
  - Bowl on stove: states 27, 30, 40.
  - Bowl inside top drawer: states 29, 44, 46, 48.
  - Bowl on cabinet: state 3.
  - Wine bottle on rack: state 27.

These failure concentrations suggest harder initial states rather than one
systematically unsolvable task: 21 tasks had no failures, while drawers,
high cabinet/stove placements, and chocolate pudding had two to four misses
each.

## Reading and limitations

- The reference is 1471/1500 after consuming full horizons on every miss, so
  the misses are not timeouts caused by an artificially short budget.
- The VLM selector's smoke-level 15/45 result is still a perception/grounding
  result, while this result is a paper-style end-to-end learned-policy result.
  The two are not the same observation access or interface and therefore not
  an apples-to-apples method comparison.
- One evaluation seed only. The released checkpoints are deterministic in this
  harness and the initial states are fixed, so repeating the same 1500
  evaluation states under additional evaluation seeds was not expected to add
  information.
- The learned-policy harness remains separate from the modular executor's
  interface. This establishes the task-level benchmark reference, not the
  requested API unification.
- LIBERO-10/Long was not run. The target suite selection was
  Spatial/Object/Goal because those three public checkpoints match the three
  modular task groups.
- Frame-directory IDs reused across suites during this long run, so the
  retained `frames/` only reliably cover the last suite; the numerical
  `episodes.jsonl` and `summary.json` records for all 1500 episodes are intact.
  The runner was subsequently updated to use globally unique frame/episode
  IDs.

## Reproduce

```bash
modal run maestro_research/modal_oft.py::download_checkpoints
modal run maestro_research/modal_oft.py::probe_oft --gl egl
modal run maestro_research/modal_oft.py::run_oft_smoke --limit 50 --full-suite --run-name dev_oft_full_r1
```

Artifacts: `/data/runs/dev_oft_full_r1-oft/summary.json`,
`episodes.jsonl`, `frames/`; local copy `/tmp/oft_full_r1/dev_oft_full_r1-oft`.
