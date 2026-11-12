"""Finite CPU-only isolated environment preparation; never imports a model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import urllib.request

R = Path("/root/autodl-tmp/TempoRAC")
B = R / "external-baselines-20261111/repnet-runner"
ap = argparse.ArgumentParser()
ap.add_argument("--continue-install", action="store_true")
args = ap.parse_args()
assert sys.platform == "linux" and B.is_relative_to(R)
spec = json.loads((B / "env-spec.json").read_text())
prefix = Path(spec["path"])
assert prefix == R / "envs/repnet"
digest = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
evidence = B / "environment-evidence"
evidence.mkdir(exist_ok=True)
env = dict(os.environ, **spec["env"], CUDA_VISIBLE_DEVICES="", PIP_DISABLE_PIP_VERSION_CHECK="1")
for key in ["PIP_CACHE_DIR", "XDG_CACHE_HOME", "KERAS_HOME", "TMPDIR"]:
    Path(env[key]).mkdir(parents=True, exist_ok=True)
metadata = {}
for name, version in [("tensorflow", "2.16.1"), ("keras", "3.3.3")]:
    with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json", timeout=20) as response:
        data = json.load(response)
    metadata[name] = {k: data["info"][k] for k in ["version", "requires_python", "requires_dist"]}
(evidence / "primary_package_metadata.json").write_text(json.dumps(metadata, indent=2))
if prefix.exists():
    assert args.continue_install, "Existing isolated prefix; diagnose before explicit continuation."
    assert (evidence / "spec_sha256").read_text().strip() == digest
else:
    (evidence / "spec_sha256").write_text(digest)
    subprocess.run([spec["base_python"], "-m", "venv", "--copies", str(prefix)], check=True, env=env)
python = str(prefix / "bin/python")
with (evidence / ("pip-continue.log" if args.continue_install else "pip-install.log")).open("x") as log:
    for phase in spec["pip_phases"]:
        subprocess.run([python, "-m", "pip", "install", "--retries", "0", "--timeout", "30",
                        "--report", str(evidence / "pip-report.json"), *phase],
                       check=True, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
subprocess.run([python, "-m", "pip", "check"], check=True, env=env)
freeze = subprocess.check_output([python, "-m", "pip", "freeze"], text=True, env=env)
(evidence / "requirements.lock.txt").write_text(freeze)
libs = sorted(str(p) for p in (prefix / "lib/python3.10/site-packages/nvidia").glob("*/lib"))
exports = dict(spec["env"], KERAS_BACKEND="tensorflow",
               LD_LIBRARY_PATH=":".join(libs), CUDA_VISIBLE_DEVICES="1")
(B / "environment.sh").write_text("\n".join(
    f"export {k}={shlex.quote(v)}" for k, v in exports.items()) + "\n")
(evidence / "installed.json").write_text(json.dumps(
    dict(status="installed_not_kernel_validated", spec_sha256=digest, prefix=str(prefix)), indent=2))
print("ISOLATED_ENV_INSTALLED_NO_MODEL_RUN", flush=True)

