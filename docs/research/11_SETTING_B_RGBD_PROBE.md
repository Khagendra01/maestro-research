# Setting B (scripted RGB-D) — probe and negative result

Date: 2026-11-08. Status: **not viable for the declared dev tasks with the
current hand-written colour/shape prior.** This is a negative result for the
scripted grounding baseline, not for the geometry/executor stack.

## What was run

- `modal run maestro_research/modal_libero.py::run_smoke --settings rgbd --limit 1`
  (one init state per suite; three episodes): **0/3**, all `failed` at SELECT
  with `AmbiguousReferenceError`; `perceived_objects` was empty.
- `run_rgbd_probe` on `libero_spatial` state 0 (same depth convention
  `bottom_left`, 256×256): dumped the above-table clusters.

## Probe evidence (libero_spatial state 0)

Table plane from the depth histogram: `table_z = 0.901` (oracle table origin
`0.875`; the rendered top surface is higher). 16,927 points above the plane.
The clusters, with mean RGB in 0–255:

| n points | mean RGB | centre xyz (m) | span z (m) | radius (m) | what it is |
|---:|---|---|---:|---:|---|
| 8054 | (32, 28, 26) | (0.078, −0.239, 1.094) | 0.215 | 0.224 | wooden cabinet (dark brown) |
| 4784 | (112, 112, 111) | (−0.190, −0.048, 1.243) | 0.456 | 0.289 | robot arm / wrist |
| 1502 | (82, 82, 82) | (−0.611, 0.002, 0.930) | 0.323 | 0.190 | robot base / mount |
| 1149 | (123, 114, 110) | (0.009, 0.281, 0.922) | 0.039 | 0.170 | plate + second bowl merged |
| 596 | (108, 109, 107) | (−0.102, 0.015, 0.936) | 0.039 | 0.082 | **target black bowl** |
| 348 | (124, 125, 126) | (−0.210, 0.188, 0.931) | 0.031 | 0.052 | ramekin |

Oracle positions for comparison: bowl 1 `(−0.075, 0.015, 0.898)`, bowl 2
`(−0.002, 0.307, 0.898)`, plate `(0.072, 0.200, 0.902)`, ramekin
`(−0.201, 0.187, 0.899)`, cabinet `(0.036, −0.278, 0.905)`.

## Diagnosis

The depth/geometry is sound (the table plane and cluster centres are within a
few centimetres of the oracle bodies). The failure is **semantic grounding**:

1. The "black" bowl renders as neutral gray (mean RGB ≈ 108/255, value 0.42),
   so the colour prior (`v < 0.22` → bowl) rejects it.
2. The dark-brown cabinet (RGB ≈ 32/255) passes the same black-bowl rule; the
   only detected "object" was the cabinet, with a 0.224 m radius.
3. The plate is thin (span z 0.039 m) and merges with the second bowl at this
   voxel resolution; it is either dropped by the height threshold or absorbed
   into an adjacent cluster.
4. The two black bowls and the glazed-rim ramekin are nearly the same colour;
   the instruction modifier ("from table center") is the only disambiguator,
   and the RGB-D selector was constructed with `table_center=None`.
5. In `libero_object`, the alphabet-soup can and the tomato-sauce can are both
   red; a colour prior cannot separate them at all (the declared ambiguity in
   `localize.py`).

This is exactly the grounding problem an open-vocabulary detector or VLM
selector solves. A real VLM is **not** needed for the Setting C geometry claim,
where simulator object identity is declared as an input. It is needed only if
the system must map raw language + RGB to identities in Settings A/B, or for an
end-to-end comparison against a learned VLM policy (OpenVLA-OFT), which performs
its own grounding.

## Consequence

- Keep Setting C as the reported result (`10_DEV_SMOKE_RESULTS_R2.md`).
- Do not report Setting B as a success; if a Setting B number is needed, the
  next step is an open-vocabulary detector (or a VLM selector) with the same
  frozen observation rules, not more colour thresholds.
- The executor's RGB-D wiring is now real (perceived positions/extents are
  passed to `StagedExecutor.run`, and internal placement verification is skipped
  when `oracle_geometry=False`); the remaining blocker is perception.
