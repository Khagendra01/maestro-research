# Observation-access manifest (frozen before comparative reporting)

Benchmark: LIBERO, official `OffScreenRenderEnv` (`Lifelong-Robot-Learning/LIBERO`,
commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`), robosuite 1.4.0, MuJoCo 2.3.7.
Robot: Panda, OSC_POSE delta controller, 7-D action `[dx, dy, dz, dax, day, daz, gripper]`,
control frequency 20 Hz. Gripper sign: `-1` open, `+1` close (robosuite). Units: metres,
radians. Evaluator horizon per suite from the public OpenVLA-OFT protocol:
Spatial 220, Object 280, Goal 300, Long/10 520.

Camera: LIBERO `agentview` (fixed third-person). Intrinsics from `model.cam_fovy` and image
size; extrinsics from `data.cam_xpos` / `data.cam_xmat`. Calibration identity is the simulator
model and camera name, recorded per episode. The wrist camera `robot0_eye_in_hand` is available
as a second declared view.

## Settings (never mixed in one reported number)

### A. Matched-observation evaluation
Only the observations a reference policy is allowed: `agentview_image` (plus optionally the
wrist image and proprioception), resized/rotated exactly as the reference's preprocessing.
No depth, no simulator poses, no segmentation IDs, no BDDL goal facts, no success flags.
Not implemented in this milestone. Nothing in this report is a Setting A result.

### B. Declared RGB-D setting (modular perception condition)
Adds `agentview` depth rendered from the same camera (`MuJoCo` depth buffer converted with the
model's near/far planes). Depth is used only to infer geometry (object point clouds, receptacle
surface). This is an explicit modality advantage over an RGB-only reference; no architectural
superiority claim may be made against Setting A.

Manual, declared rules used in this setting: a task-vocabulary parser maps the instruction noun
phrases to object/receptacle labels, and a colour/shape prior selects the target's pixel region.
These are task-family rules, not per-initial-state answers; they are frozen before evaluation
and disclosed. They do not read simulator object poses, segmentation IDs, BDDL goal facts, or
the success signal.

### C. Oracle-geometry diagnostic (planning isolation)
Uses simulator body poses/meshes for the target and receptacle, and simulator poses to verify
grasp/placement. This isolates geometry + planning + execution. It is **not** a perception
result, **not** leaderboard-comparable, and is always reported under the `oracle` label.

## Forbidden inputs to the policy in every setting
Exact object poses (except Setting C by declaration), simulator body/geom IDs used as target
identity, target-part labels from the evaluator, BDDL goal facts, hidden task predicates,
`env.check_success()`, `info["success"]`, and reward. The instruction text is allowed.

The success oracle is used only to score after the episode and to write diagnostic logs. It is
never read inside the executor for stage completion, retries, or recovery.

## Scoring
Official success is the environment's own termination/success condition, evaluated with the
official initial states. No termination changes, no task-specific success additions, no horizon
extension for only one method, no failed-episode erasure. Diagnostic metrics (grasp retention,
placement distance) are reported separately.

## Episode accounting
All declared episodes are reported, including timeouts, invalid outputs, unrecovered planning
failures and aborted runs. A 15-episode smoke is an integration check, not a generalization
estimate. Confidence intervals are not meaningful at n=15 and are not reported as such.
