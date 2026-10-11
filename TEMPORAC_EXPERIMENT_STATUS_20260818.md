# TempoRAC experiment status — 2026-08-18

## Bottom line

The attached five-page manuscript is a pre-results draft. Its Table 1 and
Table 2 contain no measured TempoRAC cells. The current handoff commit cannot
run the frozen 27-job formal campaign end to end: the natural-data entrance
gate fails before training, the native campaign launcher/pseudo-target runtime
is absent, and one ignored historical operator candidate directory is missing.

No historical model source was overwritten during this audit. The only server
additions are diagnostic scripts and previously created, explicitly
non-contract preprocessing/checkpoint artifacts.

## Runs completed in this audit

### Source and data binding

- Branch commit: `49f820ff` (`codex/aris-fullrun-icassp27`).
- v44 train cache: 18,333,016 bytes,
  SHA-256 `c96fc1dfa233ec4bf1af19e7480ee8c721f4cddee9f121185f4e494c7244e1eb`.
- v44 validation cache: 9,153,988 bytes,
  SHA-256 `4064ef24dc2970e9b07de8adddf1ecd4932aecb90453a67b4f90d7bd35d10251`.
- Population: 268 train identities and 134 validation identities.

### Code-level test run

- Container image: `pams-rac:07192de`.
- Result: 133 passed, 2 failed.
- Both failures are caused by the absent ignored directory
  `data/temporac_p00_v4/operator_candidate_20260816`; no copy exists under
  `/media/lenovo/data2/pams-rac`.
- The tested current v2 operator candidate itself passed; the failures are in
  historical-regression assertions that require the missing v1 directory.

### Frozen preprocessor entrance audit

| Split | Valid identities | Accepted | Rejected | Acceptance |
|---|---:|---:|---:|---:|
| train | 268 | 172 | 96 | 64.18% |
| val | 134 | 91 | 43 | 67.91% |

Dominant failures are invalid frames/unsupported physical edges inside the
trimmed span and duplicate clocks without a valid root. Because the frozen
population requires 268/134 identities, this is a hard pre-training failure.

### Experimental interpolating preprocessor audit

| Split | Valid identities | Accepted | Rejected | Acceptance |
|---|---:|---:|---:|---:|
| train | 268 | 268 | 0 | 100% |
| val | 134 | 134 | 0 | 100% |

This result is diagnostic only. The implementation is
`src/pams/temporac/preprocess_interpolated_v1.py`, a new file that preserves
the original preprocessor. It has not been integrated into or accepted by the
frozen contract.

### Existing X0-only natural-data diagnostic

Checkpoint:
`/media/lenovo/data2/pams-rac/runs/temporac_uniform_v1/x0_full_s20260818_steps4000.pt`

- X0 tune loss: 0.004117 (training-domain objective).
- v44 validation identities: 134.
- Raw count MAE: 21.604477611940297.
- Relative per-identity NMAE: 0.9915196743554954.
- OBO: 0.0.
- Prediction range: 0–4; target range: 4–92.

The low X0 loss does not transfer to natural counting; the response mostly
collapses to zero. This checkpoint is not eligible for the manuscript tables.

## Manuscript experiment inventory

### Table 1: main comparisons

All measured cells remain missing:

- Native MultiCounter and MultiCounter+ context rows.
- Oracle-track Track-PAMS, global-tempo control, and TempoRAC.
- Common-predicted-track Track-PAMS, global-tempo control, and TempoRAC.
- Metrics: Period-mAP, AP50, AP75, AvgMAE, and AvgOBO.
- Predicted-track diagnostics: HOTA, IDF1, and identity switches.
- At least three predeclared seeds, sample standard deviation, and paired
  video-cluster bootstrap 95% confidence intervals.

### Table 2: paired mechanism ablations

All seven non-reference contrasts remain missing:

- global ACF+FFT / uniform / NOLA;
- local ACF / soft / NOLA;
- local FFT / soft / NOLA;
- local ACF+FFT / single expert / NOLA;
- local ACF+FFT / hard route / NOLA;
- local ACF+FFT / soft / boxcar merge;
- local ACF+FFT / soft / decode-per-window merge.

The local ACF+FFT / soft / NOLA row is the reference, but it has no eligible
natural-data result yet.

### Nonstationary-tempo diagnostic

Still missing: frozen piecewise time warp, periodic and nonperiodic
missingness controls, fixed-mask tempo control, occlusion stress, identity
switch stress, and aligned reporting of local period, routing probability or
entropy, reconstructed response, and clean-to-corrupt change.

## First blockers in execution order

1. Restore or regenerate the exact historical operator-candidate v1 bytes, or
   explicitly revise the historical-regression requirement.
2. Decide and independently review the natural-data preprocessing amendment:
   either accept an interpolation policy or rebuild a cache that satisfies the
   existing frozen preprocessor without changing the 268/134 population.
3. Finish and review the native teacher/pseudo-target, checkpoint, prediction,
   receipt, and campaign-launch runtime. Current modules validate pieces but do
   not provide an end-to-end 27-job launcher.
4. Run the three teacher jobs, then the 24 response jobs, followed by the
   synthetic gates and natural clean/drift predictions.
5. Only after eligible predictions exist, run Table 1, Table 2, tracking
   diagnostics, bootstrap intervals, and manuscript result promotion.

## Metric-boundary warning

The historical MotionBERT generalization value `mae=0.087504` and paper-style
`avg_obo=0.725` are strictly replay-bound to a different supervised
MotionBERT/prototype-calibration pipeline. They are not TempoRAC results and
must not be inserted into this manuscript's TempoRAC rows.
