"""Official-code DeTRC (TSN-R50 route, trained by us on RepCount-A in baselines/detrc_native_v1) on the Countix test split.

Every numerical piece is the one used for that run's RepCount/UCFRep/QUVA evaluation, imported unchanged:
  * frames: mmcv.VideoReader frames inside the inclusive manifest frame_range, driver/rawframes.resize_native (released
    build_rawframes short-256 branch), default mmcv.imwrite JPEG, img_00001.jpg ... (as driver/rawframes.build_video);
  * features: the released tools/data/activitynet/tsn_feature_extraction.py (prerequisite copy, unmodified), every frame,
    TSN-R50 Kinetics-400 backbone;
  * model and decoding: the selected checkpoint (SELECTED_CHECKPOINT.json, SHA-256 checked), driver/inference.py's
    LabelFreeDataset / collate / postprocess (Soft-NMS sigma 0.1, top-k 30, count = boxes with score > 0.2), batches of
    64 sorted by length, as driver/inference.predict;
  * scoring: count_metrics of the frozen scorer used for the other populations (SHA-256 checked).
Only the population is new (Countix cx_test, 2403 repetition-span clips). Each stage resumes per video.
Usage (server, any python): detrc_countix.py frames --workers 12 | extract --gpu 0 | predict --gpu 0 | score
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import pickle
import subprocess
import sys
import time

R = pathlib.Path("/root/autodl-tmp/TempoRAC")
D = R / "baselines/detrc_native_v1"
PREREQ = R / "external-baselines-20261112/detrc-prerequisites"
PY = PREREQ / "env/bin/python"
EXTRACTOR = PREREQ / "candidate/tools/data/activitynet/tsn_feature_extraction.py"
BACKBONE = PREREQ / "assets/tsn_r50_320p_1x1x3_100e_kinetics400_rgb_20200702-cc665e2a.pth"
BACKBONE_SHA = "cc665e2a86857b4ff4873747ae4e98f9cd6de3f72111cf3379954f551cc63c49"
SCORER = R / "source_snapshots/489dc77b91a21d21215784ff49961edb20dab4fb/temporac_cvpr/metrics.py"
SCORER_SHA = "fce39c727fc2f139254d20b7b3833e2b9e1dcfebfe23c45be0b69c1669cb83c0"
MANIFEST = R / "data/UNION/manifest_dev.json"
OUT = R / "resume20261117/detrc_countix"


def sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path: pathlib.Path, data) -> None:
    tmp = path.with_name(path.name + ".pending")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


def countix_rows() -> list[dict]:
    rows = [r for r in json.loads(MANIFEST.read_text()) if r.get("dataset") == "Countix" and r["split"] == "cx_test"]
    assert len(rows) == 2403 and len({r["video_id"] for r in rows}) == 2403
    return rows


def runtime_env(gpu) -> dict:
    """driver/common.runtime_env of the original run, with this run's TMPDIR."""
    e = os.environ.copy()
    e.update(CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu), CUDA_HOME=str(PREREQ / "cuda-12.4"),
             TMPDIR=str(OUT / "tmp"), TORCH_EXTENSIONS_DIR=str(PREREQ / "torch-extensions"), OMP_NUM_THREADS="2",
             MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1",
             PYTHONPATH=f"{D / 'candidate'}:{PREREQ / 'vendor/mmaction2-0.24.1'}:{D / 'driver'}",
             DETRC_COUNTIX_ENV="1")
    (OUT / "tmp").mkdir(parents=True, exist_ok=True)
    return e


# ---------------------------------------------------------------- frames (CPU)
def _nice() -> None:
    os.nice(10)


def build_frames(r: dict) -> tuple[str, int, str]:
    import cv2
    import mmcv
    from rawframes import resize_native  # driver/rawframes.py of the original run
    cv2.setNumThreads(1)
    d = OUT / "rawframes" / r["video_id"]
    if (d / "IDENTITY.json").exists():
        return r["video_id"], json.loads((d / "IDENTITY.json").read_text())["native_frames"], "reused"
    tmp = d.with_name(d.name + ".pending")
    tmp.mkdir(parents=True, exist_ok=True)
    for stale in tmp.glob("img_*.jpg"):  # own partial output of an interrupted attempt
        stale.unlink()
    assert sha(r["file"]) == r["sha256"], r["video_id"]
    a, b = map(int, r["frame_range"])
    k = 0
    for i, frame in enumerate(mmcv.VideoReader(r["file"])):
        assert frame is not None, (r["video_id"], i)
        if i < a:
            continue
        k += 1
        assert mmcv.imwrite(resize_native(frame), str(tmp / f"img_{k:05d}.jpg"))
        if i == b:
            break
    assert k == b - a + 1, (r["video_id"], a, b, k)  # implies b < number of decodable frames
    atomic_json(tmp / "IDENTITY.json", dict(video_id=r["video_id"], source_video_sha256=r["sha256"], frame_range=[a, b],
                                            native_frames=k, producer="driver/rawframes.resize_native + mmcv.imwrite"))
    os.replace(tmp, d)
    return r["video_id"], k, "built"


def stage_frames(workers: int) -> None:
    from multiprocessing import Pool
    rows = countix_rows()
    t0, n, built = time.time(), 0, 0
    with Pool(workers, initializer=_nice) as pool:
        for vid, k, status in pool.imap_unordered(build_frames, rows, chunksize=4):
            n += 1
            built += status == "built"
            if n % 200 == 0:
                print(f"frames {n}/{len(rows)} built={built} {(time.time() - t0) / 60:.1f}min", flush=True)
    atomic_json(OUT / "FRAMES_COMPLETE.json", dict(rows=len(rows), frames=sum(
        json.loads((OUT / "rawframes" / r["video_id"] / "IDENTITY.json").read_text())["native_frames"] for r in rows)))
    print("FRAMES_COMPLETE", flush=True)


# ---------------------------------------------------------------- TSN features (GPU)
def native_frames(vid: str) -> int:
    return json.loads((OUT / "rawframes" / vid / "IDENTITY.json").read_text())["native_frames"]


def load_feature(vid: str, k: int):
    import numpy as np
    p = OUT / "pickles" / f"{vid}.pkl"
    # pickle is the released extractor's own output format; these files are written by stage_extract in this run dir
    with p.open("rb") as f:
        a = np.asarray(pickle.load(f))
    assert a.shape == (k, 2048) and a.dtype == np.float32 and np.isfinite(a).all(), (vid, a.shape, a.dtype)
    return a


def stage_extract(gpu: int) -> None:
    assert (OUT / "FRAMES_COMPLETE.json").exists() and sha(BACKBONE) == BACKBONE_SHA
    need = []
    for r in countix_rows():
        k = native_frames(r["video_id"])
        try:
            load_feature(r["video_id"], k)
        except (OSError, EOFError, AssertionError, pickle.UnpicklingError):
            need.append(f"{OUT / 'rawframes' / r['video_id']} {k} 0\n")
    print(f"extract: {len(need)} videos without a valid feature file", flush=True)
    if need:
        lst = OUT / "extract_list.txt"
        lst.write_text("".join(need))
        argv = [str(PY), str(EXTRACTOR), "--data-list", str(lst), "--data-prefix", "", "--output-prefix",
                str(OUT / "pickles"), "--frame-interval", "1", "--modality", "RGB", "--ckpt", str(BACKBONE),
                "--part", "0", "--total", "1"]
        with (OUT / "extract.log").open("a") as log:
            subprocess.run(argv, env=runtime_env(gpu), stdout=log, stderr=subprocess.STDOUT, check=True)
    for r in countix_rows():
        load_feature(r["video_id"], native_frames(r["video_id"]))
    atomic_json(OUT / "FEATURES_COMPLETE.json", dict(rows=2403, backbone_sha256=BACKBONE_SHA, dimension=2048))
    print("FEATURES_COMPLETE", flush=True)


# ---------------------------------------------------------------- DeTRC predictions (GPU)
def stage_predict() -> None:
    assert os.environ.get("DETRC_COUNTIX_ENV") == "1"
    sys.path[:0] = [str(D / "candidate"), str(PREREQ / "vendor/mmaction2-0.24.1"), str(D / "driver")]
    import torch
    from inference import LabelFreeDataset, collate, postprocess  # driver/inference.py of the original run
    from mmaction.models import build_model
    import DeTRC.model.DetTRC  # noqa: F401  (registers the model)
    from mmcv import Config
    assert (OUT / "FEATURES_COMPLETE.json").exists()
    sel = json.loads((D / "SELECTED_CHECKPOINT.json").read_text())
    assert sel["epochs_complete"] == 80 and sha(sel["path"]) == sel["sha256"]
    # our own RepCount-trained mmcv checkpoint (SHA-256 verified above); loaded exactly as driver/inference.predict
    saved = torch.load(sel["path"], map_location="cpu", weights_only=False)
    model = build_model(Config.fromfile(str(D / "candidate/DeTRC/config/project_tsn_full80.py")).model)
    model.load_state_dict(saved["state_dict"], strict=True)
    assert torch.count_nonzero(model.pos_embed.weight).item() == 0
    model = model.cuda().eval()
    rows = [dict(key="cx_test/" + r["video_id"], video_id=r["video_id"], native_frames=native_frames(r["video_id"]))
            for r in countix_rows()]
    ds = LabelFreeDataset(rows, D)  # sorts by native_frames, as the original
    pred_dir = OUT / "predictions"
    pred_dir.mkdir(exist_ok=True)
    t0, produced = time.time(), 0
    for start in range(0, len(ds.rows), 64):
        batch = ds.rows[start:start + 64]
        keys = [r["key"] for r in batch]
        paths = [pred_dir / f"{r['video_id']}.json" for r in batch]
        if all(p.exists() for p in paths):
            continue
        samples = [ds.structure(r, torch.from_numpy(load_feature(r["video_id"], r["native_frames"]))) for r in batch]
        with torch.no_grad():
            output = model(return_loss=False, **collate(samples))
        assert len(output) == len(batch)
        for r, p, res in zip(batch, paths, output):
            assert len(res[1]["labels"]) == 0  # structural empty GT only
            atomic_json(p, dict(video_id=r["video_id"], native_frames=r["native_frames"], checkpoint_sha256=sel["sha256"],
                                batch_keys=keys, **postprocess(res[0])))
            produced += 1
        print(f"predict {start + len(batch)}/{len(ds.rows)} {(time.time() - t0) / 60:.1f}min", flush=True)
    assert all((pred_dir / f"{r['video_id']}.json").exists() for r in rows)
    atomic_json(OUT / "PREDICTIONS_COMPLETE.json", dict(rows=len(rows), checkpoint_sha256=sel["sha256"], new_rows=produced))
    print("PREDICTIONS_COMPLETE", flush=True)


# ---------------------------------------------------------------- scoring (CPU)
def stage_score() -> None:
    import importlib.util
    assert sha(SCORER) == SCORER_SHA and (OUT / "PREDICTIONS_COMPLETE.json").exists()
    spec = importlib.util.spec_from_file_location("frozen_tempo_metrics", SCORER)
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    rows = countix_rows()
    preds = [json.loads((OUT / "predictions" / f"{r['video_id']}.json").read_text()) for r in rows]
    gt = [float(r["count"]) for r in rows]
    res = dict(cx_test=metrics.count_metrics([p["count"] for p in preds], gt),
               cx_test_before_softnms=metrics.count_metrics([p["ori_count"] for p in preds], gt),
               n=len(rows), checkpoint=json.loads((D / "SELECTED_CHECKPOINT.json").read_text()))
    atomic_json(OUT / "result.json", res)
    print("RESULT", json.dumps(res["cx_test"]), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["frames", "extract", "predict", "score"])
    ap.add_argument("--gpu", type=int)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if os.environ.get("DETRC_COUNTIX_ENV") != "1":  # run every stage under the original run's interpreter and paths
        gpu = a.gpu if a.stage in ("extract", "predict") else None
        os.execve(str(PY), [str(PY), "-u", "-B", os.path.abspath(sys.argv[0]), *sys.argv[1:]], runtime_env(gpu))
    if a.stage == "frames":
        stage_frames(a.workers)
    elif a.stage == "extract":
        stage_extract(a.gpu)
    elif a.stage == "predict":
        stage_predict()
    else:
        stage_score()


if __name__ == "__main__":
    main()
