# Corrected-protocol results collected on 2026-11-17

Copied from the two servers after every run had finished. Model checkpoints, caches and per-frame features stay on the servers.
`python -m temporac_cvpr.final_tables_ks` (repository root) reads only this directory plus
`docs/evidence/resume_20261113/external-comparison-transrac/COMPARISON_SAFE.json` and writes the paper's table fragments
(`paper/cvpr27/tab/*_rows.tex`, `main_row.tex`, `app_row.tex`) and `docs/final_numbers_ks_20261117.json`.

| Path | Source (server root `/root/autodl-tmp/TempoRAC`) | Content |
|---|---|---|
| `results/runs/full_s{0,1,2}` | first server, `runs/frozen_known_source_v1_resume20261112/` | main model, three seeds: `eval_{rc_test,ucf_test,cx_test,quva}.json`, config, result |
| `results/runs/wo_*_s{0,1,2}` | second server, `runs/frozen_known_source_v1_ablation20261112/` | 21 ablation arms; RepCount-only arms also `eval_xfer_{ucf_test,cx_test,quva}.json`. The 7 arms first finished on the first server were checked byte-identical (35/35 files) |
| `results/runs/wo_joint_ucf_s{0,1,2}` | first server, `runs/frozen_a/` | UCFRep-only decoders, reused under the ablation contract (UCFRep training data unaffected by the correction) |
| `results/countix/escounts` | `runs/baselines/escounts_official_countix` | released RepCount-A checkpoint on Countix test (`temporac_cvpr/escounts_official.py`) |
| `results/countix/repnet` | `resume20261117/repnet_countix/run` | released RepNet ckpt-70 on Countix test (`temporac_cvpr/repnet_countix.py`) |
| `results/countix/transrac` | `resume20261117/transrac_countix` | official-code TransRAC trained on RepCount-A (`temporac_cvpr/transrac_countix.py`) |
| `results/countix/detrc` | `resume20261117/detrc_countix` | official-code DeTRC trained on RepCount-A (`temporac_cvpr/detrc_countix.py`); `predictions_compact.jsonl` has the per-clip counts after and before Soft-NMS. The full per-clip JSON files with all segments (301 MB) stay on the first server under `predictions/` |
| `results/countix/logs` | `resume20261117/`, ESCounts run dir | run logs, including the two failed first attempts (RepNet without GPU libraries, TransRAC checkpoint loading) |
| `results/analysis` | both servers, `resume20261117/analysis` | rate knockout (`rate_knockout2.py`, validation; full 3×3 in the paper's `tab:knockout`), two-view tempo bins (`view_tempo.py`, validation), over-count error share, RepCount-A annotation tempo statistics (`tempo_stats.py`, annotations only) |
| `results/queue` | both servers | ablation queue events and logs (`temporac_cvpr/ks_queue.py`) |

All evaluations use the frozen scorer of the training snapshot `489dc77` (`temporac_cvpr/metrics.py`, SHA-256 `fce39c72…`):
positive-count NMAE with rounded counts and denominator gt+0.1, and OBO. Test splits were scored once per model.
