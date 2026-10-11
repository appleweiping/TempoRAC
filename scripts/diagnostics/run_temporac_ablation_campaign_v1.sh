#!/usr/bin/env bash
set -euo pipefail

repo="${1:?usage: run_temporac_ablation_campaign_v1.sh REPO CACHE_DIR OUTPUT_DIR [CUDA_INDEX]}"
cache_dir="${2:?missing CACHE_DIR}"
output_dir="${3:?missing OUTPUT_DIR}"
cuda_index="${4:-0}"
python_bin="${TEMPORAC_PYTHON:-/home/lenovo/anaconda3/envs/cja_py310/bin/python}"
trainer="/media/lenovo/data2/pams-rac/cache/train_temporac_x0_full_v1.py"

mkdir -p "${output_dir}"
cd "${repo}"
export PYTHONPATH="${repo}/src"
export CUDA_VISIBLE_DEVICES="${cuda_index}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

for seed in 20260815 20260816 20260817; do
    checkpoint="${output_dir}/x0_seed${seed}_steps4000.pt"
    train_log="${output_dir}/x0_seed${seed}_steps4000.log"
    prediction="${output_dir}/ablation_seed${seed}_predictions.json"
    eval_log="${output_dir}/ablation_seed${seed}.log"

    if [[ ! -f "${checkpoint}" ]]; then
        "${python_bin}" "${trainer}" \
            --output "${checkpoint}" \
            --steps 4000 \
            --evaluate-every 250 \
            --seed "${seed}" \
            --device cuda 2>&1 | tee "${train_log}"
    fi
    if [[ ! -f "${prediction}" ]]; then
        "${python_bin}" scripts/diagnostics/run_temporac_ablation_suite_v1.py \
            --cache-dir "${cache_dir}" \
            --checkpoint "${checkpoint}" \
            --output "${prediction}" \
            --device cuda 2>&1 | tee "${eval_log}"
    fi
done

"${python_bin}" - "${output_dir}" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for seed in (20260815, 20260816, 20260817):
    path = root / f"ablation_seed{seed}_predictions.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows.append(
        {
            "seed": seed,
            "checkpoint_sha256": payload["checkpoint_sha256"],
            "identities": payload["identities"],
            "variants_completed": payload["variants_completed"],
            "variants_skipped": payload["variants_skipped"],
            "prediction_file": path.name,
            "prediction_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    )
summary = {
    "schema": "temporac.ablation-campaign-diagnostic.v1",
    "scientific_status": "X0_TRAINED_SUPPLIED_TRACK_PARTIAL_CACHE_DIAGNOSTIC_NOT_PAPER_MAIN",
    "seeds": rows,
    "metrics": None,
    "metric_blocker": "No safe evaluator-only label JSON is available; predictions are frozen.",
}
target = root / "campaign_summary.json"
target.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, sort_keys=True))
PY
