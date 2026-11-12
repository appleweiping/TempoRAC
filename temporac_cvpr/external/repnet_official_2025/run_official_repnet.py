"""One authorized official restored-weight witness followed by fixed full357."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

R = Path("/root/autodl-tmp/TempoRAC")
B = R / "external-baselines-20261112/repnet-runner"
ap = argparse.ArgumentParser()
ap.add_argument("--review", required=True)
args = ap.parse_args()
review = Path(args.review).resolve()
assert review.is_relative_to(R) and review.is_file()
cfg = json.loads((B / "evaluation_config.json").read_text())
assert cfg["runtime"]["gpu"] == 1
job = R / "external-baselines-20261112/runs/official-worker-v3"
job.mkdir(parents=True, exist_ok=False)
def event(status, **extra):
    row = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               status=status, pid=os.getpid(), **extra)
    with (job / "events.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)
try:
    event("started", review=str(review),
          review_sha256=hashlib.sha256(review.read_bytes()).hexdigest(),
          authorization_basis="user_resume_after_recorded_stop_not_training_completion")
    spec = json.loads((B / "env-spec.json").read_text())
    prefix = Path(spec["path"])
    libs = sorted(str(p) for p in (prefix / "lib/python3.10/site-packages/nvidia").glob("*/lib"))
    env = dict(os.environ, **spec["env"], CUDA_VISIBLE_DEVICES="1",
               LD_LIBRARY_PATH=":".join(libs))
    python = str(prefix / "bin/python")
    predecessor = json.loads(Path(cfg["compatible_predecessor"]["manifest"]).read_text())
    prior_witness = R / "external-baselines-20261112/runs/witness-v2/witness.json"
    prior_restore = prior_witness.with_name("restore.json")
    assert hashlib.sha256(prior_witness.read_bytes()).hexdigest() == predecessor["witness_receipt_sha256"]
    assert hashlib.sha256(prior_restore.read_bytes()).hexdigest() == predecessor["witness_restore_sha256"]
    assert json.loads(prior_witness.read_text())["status"] == "complete"
    event("inherited_witness", receipt=str(prior_witness), receipt_sha256=predecessor["witness_receipt_sha256"])
    for mode, dirname in [("evaluate", "repnet-full357-v3")]:
        command = [python, "-u", "-B", str(B / "evaluate_repnet.py"),
                   "--config", str(B / "evaluation_config.json"),
                   "--output", str(R / "external-baselines-20261112/runs" / dirname),
                   "--mode", mode, "--adopt-predecessor"]
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

