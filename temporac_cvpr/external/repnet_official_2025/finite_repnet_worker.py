"""One finite dependency worker: predecessor exit, one witness, one full evaluation."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import sys

R = Path("/root/autodl-tmp/TempoRAC")
B = R / "external-baselines-20261111/repnet-runner"
ap = argparse.ArgumentParser()
ap.add_argument("--review", required=True, help="Completed corrected-draft review artifact")
args = ap.parse_args()
review = Path(args.review).resolve()
assert review.is_relative_to(R) and review.is_file()
cfg = json.loads((B / "evaluation_config.json").read_text())
assert cfg["runtime"]["gpu"] == 1
job = R / "external-baselines-20261111/runs/finite-worker-v2"
job.mkdir(parents=True, exist_ok=False)
def event(status, **extra):
    row = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               status=status, pid=os.getpid(), **extra)
    with (job / "events.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)
try:
    review_sha = hashlib.sha256(review.read_bytes()).hexdigest()
    event("started", review=str(review), review_sha256=review_sha)
    pid = cfg["predecessor"]["worker_pid"]
    try:
        fd = os.pidfd_open(pid)
    except ProcessLookupError:
        event("predecessor_already_exited", predecessor_pid=pid)
    else:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        assert "known_source_worker.py --gpu 1 --runs full_s1" in command, command
        event("waiting_for_predecessor_exit", predecessor_pid=pid)
        poller = select.poll()
        poller.register(fd, select.POLLIN)
        poller.poll()  # Blocks on that process exit; no timed/recurring polling.
        os.close(fd)
        event("predecessor_exited", predecessor_pid=pid)
    spec = json.loads((B / "env-spec.json").read_text())
    prefix = Path(spec["path"])
    libs = sorted(str(p) for p in (prefix / "lib/python3.10/site-packages/nvidia").glob("*/lib"))
    env = dict(os.environ, **spec["env"], CUDA_VISIBLE_DEVICES="1",
               LD_LIBRARY_PATH=":".join(libs))
    python = str(prefix / "bin/python")
    for mode, dirname in [("witness", "witness-v2"), ("evaluate", "repnet-full357-v2")]:
        command = [python, "-u", "-B", str(B / "evaluate_repnet.py"),
                   "--config", str(B / "evaluation_config.json"),
                   "--output", str(R / "external-baselines-20261111/runs" / dirname),
                   "--mode", mode]
        event("launching", mode=mode, command=command)
        with (job / f"{mode}.log").open("x") as log:
            child = subprocess.Popen(command, cwd=str(R), env=env, stdout=log, stderr=subprocess.STDOUT)
            event("child_started", mode=mode, child_pid=child.pid)
            code = child.wait()
        event("child_exited", mode=mode, returncode=code)
        assert code == 0, (mode, code)
    event("complete")
except BaseException as error:
    event("failed", error=repr(error))
    raise

