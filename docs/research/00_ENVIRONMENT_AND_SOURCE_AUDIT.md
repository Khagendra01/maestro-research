# Environment, provenance and source audit

Date: 2026-11-08 (America/Los_Angeles)
Repository: https://github.com/penn-pal-lab/maestro.git
Checkout: `/Users/kgensupertramp/Desktop/newtest/maestro`
Upstream SHA at clone: `e60d326233a238ddc496bf7a08acb7a278e9e66d` ("Maestro: initial release")
Working branch: `research/fast-vlm-geometric-control` (not pushed)
Working tree at clone: clean; no user changes were present in this directory.

## Machine audit (actual, measured)

| Item | Value |
|---|---|
| OS | macOS 12.7.4 (build 21H1123), x86_64 |
| CPU | Intel Core m3-6Y30 @ 0.90 GHz, 4 logical cores |
| RAM | 8 GiB |
| GPU | Intel HD Graphics 515, 1536 MB dynamic VRAM (no CUDA) |
| Free disk | 4.3 GiB on `/` and `/System/Volumes/Data` |
| Python | 3.12.0 (`/Library/Frameworks/Python.framework/Versions/3.12`) |
| `uv` | present (`~/.local/bin/uv`) |
| Docker / Compose | not installed; Colima present but stopped |
| NVIDIA driver / CUDA | absent |
| Simulators | CoppeliaSim 4.10 installed locally; no MuJoCo/LIBERO local install |
| Robot hardware | none attached; no DROID/ZED/Franka configuration |
| Modal CLI | 1.5.5, authenticated workspace `khagendrakhatri365` |
| Local cloud VLM keys | no `GEMINI_KEY` / `OPENAI_API_KEY` in the process environment |

Consequences:

- MAESTRO's released stack (DROID/ZED client, Franka hardware, four CUDA Docker services,
  cuRobo) cannot run natively here: no NVIDIA GPU, no Docker, no cameras, no arm, 4.3 GiB
  free disk. No physical motion is authorized or possible.
- Modal is available and is the only usable GPU path. Prior work in
  `/Users/kgensupertramp/Downloads/object-dynamics-vla` already validated a pinned LIBERO
  image on Modal (LIBERO `8f1084e3132a39270c3a13ebe37270a43ece2a01`, robosuite 1.4.0,
  MuJoCo 2.3.7, OSMesa, CPU rendering). That image definition is reused, not rebuilt blindly.
- Billing report: USD 32.98 metered, USD 28.64 credited, USD 0.00 billed at audit time.
  No paid infrastructure was provisioned.

## Upstream source map (inspected, authoritative at the SHA above)

| Path | Role | Relevance to this work |
|---|---|---|
| `maestro/runner.py` | plan → execute → replan loop; writes `runs/logs_all_tasks` | Baseline orchestration; hardware-first and coupled to DROID |
| `maestro/vlm.py` | planner/pointing/part-selection VLM clients (Gemini, OpenAI) | Reference VLM contract; model IDs are upstream defaults, not verified available |
| `maestro/api.py` | `RealFrankaRobotApi` + `ImageEditApi`, 2,079 lines | Tool semantics to mirror: locate/propose/execute/move/close/open/reset, SE(3) geometry |
| `maestro/perception_utils.py` | HTTP clients for FoundationStereo, GroundedSAM, GraspGen, cuRobo | Service contract; not run here |
| `maestro/camera_geometry.py` | intrinsics validation, stereo baseline | Reusable pure geometry |
| `maestro/config.py` | YAML/.env config | Read-only reference |
| `droid/robot_env.py` | constructs `FrankaRobot`/`ServerInterface` at `__init__`, resets at import-time construction | Hard hardware coupling; must not be constructed in simulation |
| `droid/` | DROID Franka client, calibration, cameras | Hardware path; gated off |
| `servers/*` | four CUDA GPU services | Cannot build/run locally; not required for LIBERO evaluation |
| `tests/` | 3 upstream unit tests | Baseline checks; pure CPU |
| `docs/ARCHITECTURE.md`, `docs/TOOLS.md` | design and tool contracts | Used for tool naming and semantics |
| `pyproject.toml`, `uv.lock` | pinned client dependencies | Authoritative client env |

Notable audited facts:

- The public release has no VLA tool and no simulator backend. There is no existing adapter
  boundary that separates tool execution from `RobotEnv` construction.
- `RealFrankaRobotApi()` requires a live `RobotEnv`; `RobotEnv.__init__` connects to DROID and
  calls `reset()` on construction. Any simulation/dry-run path must bypass this entirely.
- `maestro/runner.py` performs a startup scan by moving the arm. It must never be launched on
  this machine.
- cuRobo is consumed only through an HTTP service (`POST /plan`, `POST /plan_grasp`,
  `POST /fuse_observation`). The exact cuRobo version is in `servers/curobo_server/requirements.txt`;
  it is not exercised in this milestone because the LIBERO adapter uses the benchmark's own
  OSC controller. A cuRobo baseline is therefore reported as blocked, not as done.
- Tool semantics use metres, robot-base `x` forward, `y` left, `z` up, and roll/pitch/yaw at
  the boundary. Internal pose math is done with matrices/quaternions.

## Component reuse decision

Reused from MAESTRO: tool vocabulary and semantics, metres/SE(3) conventions, camera-geometry
validation, staged skill conceptual model, and the documentation structure. Not reused in this
milestone: DROID client, ZED/FoundationStereo, GroundedSAM, GraspGen, cuRobo, upstream runner.

New work (all in `maestro_research/`): typed data contracts, a hardware-refusing backend
boundary, a LIBERO simulation backend using the official environment/action interface, a
recording-free staged executor, scripted instruction selection (explicitly a control, not a
VLM), metrics, a CLI, and a Modal runner.

## Ordered execution plan

1. Preserve the checkout above; run upstream CPU unit tests and record exact results.
2. Add `maestro_research/` contracts, hardware gate, dry-run backend, tests.
3. Add the LIBERO backend with the official `OffScreenRenderEnv` and 7-D OSC action interface.
4. Pin three development tasks across Object/Spatial/Goal, five official initial states each.
5. Run a 15-episode modular smoke on Modal; report official success, timings, failures.
6. Attempt the OpenVLA-OFT reference with a compatible public checkpoint; if blocked, report
   the exact dependency and keep the modular evidence separate.
7. Write results, blockers, commands, and a reviewable local diff.

## Explicit limitations at audit time

- No learned VLA reference has been reproduced yet.
- No cuRobo plan has been validated (no CUDA; service not built).
- No RGB-D perception result exists yet; the first modular smoke will declare its observation
  setting explicitly (oracle-geometry diagnostic vs RGB-D).
- The prior LIBERO image is a CPU-rendered OSMesa image. Physics and rendering run on CPU;
  it is not evidence of GPU execution or of the paper's real-robot benchmark.
