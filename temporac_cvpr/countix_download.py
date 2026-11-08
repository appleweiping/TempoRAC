"""Fetch only the matched Countix clips from the remote Kinetics-700 zips (HTTP ranges).

Resumable: existing files with the expected size are skipped.  Writes a receipt
with per-file size and SHA-256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from temporac_cvpr.countix_index import BASE


def fetch_zip(args):
    zi, items, vdir = args
    from remotezip import RemoteZip

    done, failed = [], []
    todo = [it for it in items if not ((vdir / Path(it["member"]).name).exists()
                                       and (vdir / Path(it["member"]).name).stat().st_size == it["size"])]
    if not todo:
        return zi, [(it["member"], None) for it in items], []
    for attempt in range(5):
        try:
            with RemoteZip(BASE.format(zi)) as z:
                for it in list(todo):
                    p = vdir / Path(it["member"]).name
                    data = z.read(it["member"])
                    assert len(data) == it["size"], (it["member"], len(data), it["size"])
                    p.write_bytes(data)
                    done.append((it["member"], hashlib.sha256(data).hexdigest()))
                    todo.remove(it)
            break
        except Exception as e:  # retry with a fresh connection; record the last error
            failed = [(it["member"], repr(e)) for it in todo]
            time.sleep(10)
    return zi, done, failed if todo else []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="data/Countix")
    ap.add_argument("--workers", type=int, default=22)
    args = ap.parse_args()
    root = Path(args.root)
    vdir = root / "videos"
    vdir.mkdir(parents=True, exist_ok=True)
    matched = json.loads((root / "countix_matched.json").read_text())
    by_zip = defaultdict(dict)
    for m in matched:
        by_zip[m["zip"]][m["member"]] = m  # several Countix rows can share one Kinetics clip
    jobs = []
    for zi, v in sorted(by_zip.items()):  # several connections per zip: chunks of ~40 members
        items = list(v.values())
        for k in range(0, len(items), 40):
            jobs.append((zi, items[k:k + 40], vdir))
    receipt, failures = {}, []
    with ThreadPoolExecutor(args.workers) as ex:
        for zi, done, failed in ex.map(fetch_zip, jobs):
            for name, sha in done:
                receipt[name] = sha
            failures += failed
            print(time.strftime("%H:%M:%S"), "zip", zi, "files", len(done), "failed", len(failed), flush=True)
    (root / "download_receipt.json").write_text(json.dumps({"files": receipt, "failures": failures}, indent=0))
    print("DOWNLOAD_DONE", len(receipt), "failures", len(failures))


if __name__ == "__main__":
    main()
