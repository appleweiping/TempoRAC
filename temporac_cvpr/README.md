# TempoRAC (CVPR 2027 continuation) — reproduction guide

All commands run on the GPU server from `ROOT/code` with `PYTHONPATH=.` and the
project environment `ROOT/env` (Python 3.10, torch 2.5.1+cu124). `ROOT` is the
project directory (here `/root/autodl-tmp/TempoRAC`).

## 1. Data (complete official splits, no sub-sampling)

| Dataset | Source | Script | Splits used |
|---|---|---|---|
| RepCount-A | `lmms-lab-eval/repcounta-lance` (original media bytes of `RepCountA.tar.gz`) | `prepare_repcount.py` | 758 / 131 / 152 |
| UCFRep | UCF101 (`quchenyuan/UCF101-ZIP`) + Context `.mat` + ESCounts CSV | `prepare_ucfrep.py` | 421 / 105; dev = 331 / 90 by UCF101 group |
| Countix | Kinetics-700 HF mirror, remote-zip range download + ESCounts split | `countix_index.py`, `countix_download.py`, `prepare_countix.py` | 4224 / 1374 / 2403 (of 4277 / 1406 / 2453) |
| QUVA | UvA original archive | `prepare_quva.py` | 100 (test only) |

Every manifest row carries the file SHA-256; counts and cycle bounds are
cross-checked against an independent copy of the official annotation (reports
written next to the manifests). Exclusions are listed, never silent.

## 2. Frozen multi-rate tokens

```
python -m temporac_cvpr.check_encoder pretrained/VIT_B_16x4_MAE_PT.pyth third_party/EveryShotCounts
python -m temporac_cvpr.extract_features --manifest M --out features/DS_stmae --ckpt pretrained/VIT_B_16x4_MAE_PT.pyth \
    --rates 2,4,8 --pool 7 --shard 0 --num-shards 2
python -m temporac_cvpr.pool_features --src features/DS_stmae --dst features/DS_stmae_p4 --pool 4
```

ST-MAE ViT-B/16 (16x4, K400 self-supervised; ESCounts' encoder), shorter side
224 + centre crop, ImageNet normalisation, 16-frame clips with stride `s`, hop
`4s`, tokens averaged on a `2s`-frame grid.

## 3. Training and evaluation

```
python -m temporac_cvpr.train --manifest M --features F --out runs/G/NAME --seed S --cfg '{"head":"concat","rates":[2,4,8],"pool_to":4,"tg_bias":true}'
python -m temporac_cvpr.launch specs/SPEC.json --max-per-gpu 4      # queued multi-seed runs
python -m temporac_cvpr.aggregate runs/G [--tags valid_warp]          # mean +- SD over seeds
python -m temporac_cvpr.eval_saved --run runs/G/NAME --manifest M --features F --split S
```

Model selection uses the validation (UCFRep: dev) split only; `--with-test`
scores the held-out split once with the selected checkpoint. Metrics follow the
official ESCounts code (positive-count videos, |round(p)-g|/(g+0.1), OBO);
all-video variants are logged as well (`metrics.py`).

## 4. Baselines and diagnostics

- `escounts_official.py`: released ESCounts checkpoint, exemplar-free, official two-offset decoding.
- `models.py` heads: `single` (ESCounts-style decoder), `concat` (pyramid; options `tg_bias`, `tempo_gate`,
  `period_head`, `matched_gate`, `periodic_pool`, `tc_decode`), `tsm` (TransRAC-style), `repnet` (RepNet-style).
- `make_warp.py`: count-preserving piecewise tempo warps (valid seed 20261109 for development,
  test seed 20261108 for the final stress test only).
- `analyze.py`, `route_analysis.py`, `harmonic_probe.py`, `period_probe.py`: stratified errors and mechanism probes.
- `efficiency.py`: parameters, head FLOPs, latency.
