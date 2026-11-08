"""Write spatially pooled copies of extracted features (e.g. 7x7 -> 4x4) to disk.

Pooling once on disk keeps training processes memory-mapped (shared page cache)
instead of each holding a private pooled copy.  Meta files are copied unchanged.
"""
from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def pool_one(args):
    src, dst, P = args
    import torch

    torch.set_num_threads(1)
    a = np.load(src)
    p = int(round(a.shape[1] ** 0.5))
    t = torch.from_numpy(a.astype(np.float32)).view(a.shape[0], p, p, -1).permute(0, 3, 1, 2)
    t = torch.nn.functional.adaptive_avg_pool2d(t, P).permute(0, 2, 3, 1).reshape(a.shape[0], P * P, -1)
    np.save(dst, t.numpy().astype(np.float16))
    return dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--pool", type=int, default=4)
    ap.add_argument("--rates", default="1,2,4")
    ap.add_argument("--procs", type=int, default=24)
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    jobs = []
    for s in args.rates.split(","):
        (dst / f"r{s}").mkdir(parents=True, exist_ok=True)
        for f in sorted((src / f"r{s}").glob("*.npy")):
            o = dst / f"r{s}" / f.name
            if not o.exists():
                jobs.append((str(f), str(o), args.pool))
    if (src / "meta").exists() and not (dst / "meta").exists():
        shutil.copytree(src / "meta", dst / "meta")
    with ProcessPoolExecutor(args.procs) as ex:
        for i, _ in enumerate(ex.map(pool_one, jobs, chunksize=8)):
            if i % 500 == 0:
                print(i, "/", len(jobs), flush=True)
    print("POOL_DONE", len(jobs))


if __name__ == "__main__":
    main()
