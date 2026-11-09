# Research harness

This directory is the entry point for the LIBERO research work in this fork.
It is separate from the upstream MAESTRO release; the harness lives in
`maestro_research/` and is pinned to a declared simulator/version set.

## Read in this order

1. [`00_ENVIRONMENT_AND_SOURCE_AUDIT.md`](00_ENVIRONMENT_AND_SOURCE_AUDIT.md) —
   machine limits, pinned LIBERO/robosuite/MuJoCo versions, Modal setup.
2. [`OBSERVATION_MANIFEST.md`](OBSERVATION_MANIFEST.md) — frozen observation
   settings A/B/C and the forbidden-input list.
3. [`10_DEV_SMOKE_RESULTS_R2.md`](10_DEV_SMOKE_RESULTS_R2.md) — oracle-geometry
   planning control: **15/15** on the 3-task × 5-state dev smoke.
4. [`11_SETTING_B_RGBD_PROBE.md`](11_SETTING_B_RGBD_PROBE.md) — scripted RGB-D
   perception negative result (colour/semantic failures on same-colour objects).
5. [`12_VLM_SELECTOR_RESULTS.md`](12_VLM_SELECTOR_RESULTS.md) — real Meta Muse
   Spark VLM grounding on the same smoke: **15/45** official successes over three
   repeats (object 0/15, spatial 9/15, goal 6/15), with latency and failure
   analysis.

## Status at a glance

| piece | state |
|---|---|
| Oracle geometry + planning + execution | 15/15 (Setting C; planning isolation only) |
| Scripted RGB-D perception | implemented; negative probe (Setting B) |
| VLM semantic selector (Meta Muse Spark) | implemented and run; 15/45 over 3 repeats (Setting B) |
| OpenVLA-OFT 7B learned-policy reference | **not run**; repo/checkpoints/Modal A100 pending |
| cuRobo baseline | blocked: no CUDA/Docker locally; no baseline service |
| Learned-policy + modular on the same interface | **open milestone**; the VLM selector does not satisfy it |

The VLM selector is a grounding module, not a visuomotor policy. Nothing here
should be read as a Setting A or leaderboard result; depth is used in Settings
B/C. The first-milestone learned-policy requirement remains the main open item.

## Reproduce

```bash
python3 -m unittest discover -s tests -p 'test_research_*.py'
modal run maestro_research/modal_libero.py::run_smoke --settings oracle --limit 5 --run-name dev_oracle_repeat
modal run maestro_research/modal_libero.py::run_smoke --settings vlm --limit 5 --run-name dev_vlm_repeat   # needs meta-model-api secret
```

The VLM secret is `meta-model-api` (`MODEL_API_KEY`); it is not stored in the
repository and the key expires. Manifest `libero-dev-smoke-r2` (sha256
`b8ca154ecbe8595f3823f04965619421ed927bbb8590bf6c10d0103739fc6839`) is the
declared 15-episode development smoke.
