"""Paper tables for the corrected (known-source v1) protocol, from collected evaluation files only.

Every number is read from saved evaluation outputs under docs/evidence/resume_20261117/results (copied from the
servers; see README.md there) and from the earlier external-baseline packet; nothing is re-scored here.
  * TempoRAC: full_s0..2 (three seeds, one jointly trained model per seed, checkpoint per benchmark selected on its
    own validation split); reported count = fixed mean of the original and the shifted view (``metrics_tta``).
  * Ablations: wo_* arms, three seeds each, same evaluation; the arm without pyramid shift has only the original view.
    "one model per dataset": RepCount-A-only, Countix-only and UCFRep-only decoders, each scored on its own benchmark.
  * Cross-dataset matrix: models trained on RepCount-A only, scored on RepCount-A, UCFRep, Countix and QUVA with one
    common scorer (positive-count NMAE with denominator gt+0.1 and OBO, rounded counts).
  * Matched-seed ablation differences (ablation minus full model of the same seed) with each arm's parameters and
    epochs, and the inference-time stream-removal analysis (validation; analysis/knockout_ks_full3.json).
Writes paper/cvpr27/tab/{main_row,app_row,ablation_rows,app_ablation_rows,matrix_rows,app_paired_rows,knockout_rows}.tex
and docs/final_numbers_ks_20261117.json.
Usage (repo root): python -m temporac_cvpr.final_tables_ks
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

EV = Path("docs/evidence/resume_20261117/results")
EXT = Path("docs/evidence/resume_20261113/external-comparison-transrac/COMPARISON_SAFE.json")
TAB = Path("paper/cvpr27/tab")
SEEDS = (0, 1, 2)
BENCH = [("rc", "rc_test"), ("ucf", "ucf_test"), ("cx", "cx_test")]


def load(p: Path) -> dict:
    return json.loads(p.read_text())


def evalm(run: str, tag: str, key: str) -> dict:
    return load(EV / "runs" / run / f"eval_{tag}.json")[key]


def agg(ms: list[dict]) -> dict:
    out = {"n_seeds": len(ms)}
    for k in ("mae_p", "obo_p"):
        x = np.asarray([m[k] for m in ms], float)
        out[k] = (float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0)
    return out


def arm(prefix: str, key: str, tags=BENCH) -> dict:
    return {s: agg([evalm(f"{prefix}_s{i}", tag, key) for i in SEEDS]) for s, tag in tags}


def joint_removed() -> dict:
    return {"rc": agg([evalm(f"wo_joint_rc_s{i}", "rc_test", "metrics_tta") for i in SEEDS]),
            "cx": agg([evalm(f"wo_joint_cx_s{i}", "cx_test", "metrics_tta") for i in SEEDS]),
            "ucf": agg([load(EV / "runs" / f"wo_joint_ucf_s{i}" / "result.json")["test_tta"] for i in SEEDS])}


def cell(v, bold=False, d=3) -> str:
    if v is None:
        return "--"
    m, s = v if isinstance(v, tuple) else (v, None)
    t = f"{m:.{d}f}" + (f"{{\\scriptsize$\\pm${s:.{d}f}}}" if s is not None else "")
    return f"\\best{{{t}}}" if bold else t


def external() -> dict:
    """Executed baselines: RepCount-A / UCFRep / QUVA from the earlier packet, Countix from this collection."""
    names = {"ESCounts released RC weights": "escounts", "RepNet released updated2025 ckpt70": "repnet",
             "DeTRC TSN/R50 full80 seed0": "detrc", "TransRAC official-code full200 seed1": "transrac"}
    dmap = {"RepCount-A": "rc", "UCFRep": "ucf", "QUVA": "quva"}
    out = {v: {} for v in names.values()}
    for r in load(EXT)["common_metrics"]:
        if r["method"] in names:
            out[names[r["method"]]][dmap[r["dataset"]]] = {k: r["metrics"][k] for k in ("mae_p", "obo_p")}
    cx = EV / "countix"
    out["escounts"]["cx"] = load(cx / "escounts/result.json")["test"]
    out["repnet"]["cx"] = load(cx / "repnet/metrics.json")["Countix"]["harmonized"]
    out["transrac"]["cx"] = load(cx / "transrac/result.json")["cx_test"]
    out["detrc"]["cx"] = load(cx / "detrc/result.json")["cx_test"]
    for v in out.values():
        for s in v:
            v[s] = {k: v[s][k] for k in ("mae_p", "obo_p")}
    return out


def main() -> None:
    full = arm("full", "metrics_tta")
    full["quva"] = agg([evalm(f"full_s{i}", "quva", "metrics_tta") for i in SEEDS])
    # consistency with the closed group summary (two-view mean, three seeds)
    gs = load(Path("docs/evidence/resume_20261112/full3-results/group_summary.json"))["benchmarks"]
    for s, b in [("rc", "rc_test"), ("ucf", "ucf_test"), ("cx", "cx_test"), ("quva", "quva")]:
        ref = gs[b]["views"]["two_view_mean"]["aggregate"]["mae_p"]["mean"]
        assert abs(full[s]["mae_p"][0] - ref) < 1e-9, (s, full[s]["mae_p"][0], ref)

    rows = [
        (r"\method", full),
        (r"\quad single view at test (no $2\times$ view)", arm("full", "metrics")),
        (r"\quad $-$ multi-rate pyramid ($s{=}4$ only)", arm("wo_pyramid", "metrics_tta")),
        (r"\quad $-$ pyramid shift (training and test)", arm("wo_shift", "metrics")),
        (r"\quad $-$ second depth (block 12 only)", arm("wo_depth", "metrics_tta")),
        (r"\quad $-$ dataset token", arm("wo_token", "metrics_tta")),
        (r"\quad $-$ balanced sampling", arm("wo_balance", "metrics_tta")),
        (r"\quad $-$ joint training (one model per dataset)", joint_removed()),
    ]
    best = {(s, k): (min if k == "mae_p" else max)(r[s][k][0] for _, r in rows) for s, _ in BENCH for k in ("mae_p", "obo_p")}

    def c(r, s, k):
        return cell(r[s][k], abs(r[s][k][0] - best[(s, k)]) < 1e-12)

    abl = [f"{lab} & {c(r, 'rc', 'mae_p')} & {c(r, 'rc', 'obo_p')}" + r" \\" for lab, r in rows]
    app_abl = [f"{lab} & " + " & ".join(c(r, s, k) for s, _ in BENCH for k in ("mae_p", "obo_p")) + r" \\" for lab, r in rows]

    # cross-dataset matrix: trained on RepCount-A only
    ext = external()
    rc_only = {s: agg([evalm(f"wo_joint_rc_s{i}", tag, "metrics_tta") for i in SEEDS])
               for s, tag in [("rc", "rc_test"), ("ucf", "xfer_ucf_test"), ("cx", "xfer_cx_test"), ("quva", "xfer_quva")]}
    order = ["rc", "ucf", "cx", "quva"]
    mrows = [("RepNet~\\cite{dwibedi2020repnet}$^\\ddagger$", "Synthetic, CX", ext["repnet"]),
             ("ESCounts~\\cite{sinha2024escounts}$^\\ddagger$", "RC", ext["escounts"]),
             ("TransRAC~\\cite{hu2022transrac}$^\\ast$", "RC", ext["transrac"]),
             ("DeTRC~\\cite{ma2026detrc}$^\\ast$", "RC", ext["detrc"]),
             ("\\method", "RC", {s: {k: rc_only[s][k] for k in ("mae_p", "obo_p")} for s in order}),
             ("\\method", "RC, UCF, CX", {s: {k: full[s][k] for k in ("mae_p", "obo_p")} for s in order})]

    def val(x):
        return x[0] if isinstance(x, tuple) else x

    mbest = {(s, k): (min if k == "mae_p" else max)(val(r[s][k]) for _, _, r in mrows) for s in order for k in ("mae_p", "obo_p")}
    mlines = [f"{lab} & {data} & " + " & ".join(cell(r[s][k], abs(val(r[s][k]) - mbest[(s, k)]) < 1e-12)
                                                for s in order for k in ("mae_p", "obo_p")) + r" \\" for lab, data, r in mrows]

    # matched-seed differences (ablation minus full model, same seed), with parameters and epochs of each arm
    def ps(prefix, key):
        return {s: np.asarray([evalm(f"{prefix}_s{i}", tag, key)["mae_p"] for i in SEEDS]) for s, tag in BENCH}

    def cfg(run):
        c = load(EV / "runs" / f"{run}_s0" / "config.json")
        return c["params"], c["cfg"]["epochs"]

    ps_joint = {"rc": np.asarray([evalm(f"wo_joint_rc_s{i}", "rc_test", "metrics_tta")["mae_p"] for i in SEEDS]),
                "cx": np.asarray([evalm(f"wo_joint_cx_s{i}", "cx_test", "metrics_tta")["mae_p"] for i in SEEDS]),
                "ucf": np.asarray([load(EV / "runs" / f"wo_joint_ucf_s{i}" / "result.json")["test_tta"]["mae_p"] for i in SEEDS])}
    ref = ps("full", "metrics_tta")
    paired_arms = [(ps("full", "metrics"), "full"), (ps("wo_pyramid", "metrics_tta"), "wo_pyramid"),
                   (ps("wo_shift", "metrics"), "wo_shift"), (ps("wo_depth", "metrics_tta"), "wo_depth"),
                   (ps("wo_token", "metrics_tta"), "wo_token"), (ps("wo_balance", "metrics_tta"), "wo_balance"),
                   (ps_joint, "wo_joint_rc")]
    assert all(cfg(f"wo_joint_{d}") == cfg("wo_joint_rc") for d in ("cx", "ucf"))
    p_full, e_full = cfg("full")
    paired, paired_json = [f"{rows[0][0]} & {p_full / 1e6:.2f} & {e_full} & " + " & ".join(["--", "--"] * 3) + r" \\"], {}
    for (lab, _), (a, run) in zip(rows[1:], paired_arms):
        prm, ep = cfg(run)
        cells, paired_json[lab] = [], {}
        for s, _ in BENCH:
            d = a[s] - ref[s]
            cells.append(f"${d.mean():+.3f}$ & {int((d > 0).sum())}/3")
            paired_json[lab][s] = {"diff_per_seed": d.round(4).tolist(), "seeds_worse": int((d > 0).sum())}
        paired.append(f"{lab} & {prm / 1e6:.2f} & {ep} & " + " & ".join(cells) + r" \\")

    # inference-time stream removal (validation, original view, three seeds): change relative to the full decoder
    ko = load(EV / "analysis" / "knockout_ks_full3.json")
    kb = ["RepCountA", "Countix", "UCFRep"]
    ko_lines = ["none (MAE) & " + " & ".join(f"{ko[b]['mae_p']['all'][j]:.3f}" for b in kb for j in range(3)) + r" \\",
                r"\midrule"]
    for r, lab in [("-r2", "$s{=}2$"), ("-r4", "$s{=}4$"), ("-r8", "$s{=}8$")]:
        ko_lines.append(lab + " & " + " & ".join(f"${100 * (ko[b]['mae_p'][r][j] / ko[b]['mae_p']['all'][j] - 1):+.0f}$"
                                                 for b in kb for j in range(3)) + r" \\")

    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / "app_paired_rows.tex").write_text("\n".join(paired) + "\n")
    (TAB / "knockout_rows.tex").write_text("\n".join(ko_lines) + "\n")
    (TAB / "main_row.tex").write_text(f"\\best{{{cell(full['rc']['mae_p'])}}} & \\best{{{cell(full['rc']['obo_p'])}}}\n")
    (TAB / "app_row.tex").write_text(" & ".join([cell(full["ucf"]["mae_p"]), cell(full["ucf"]["obo_p"]), cell(full["cx"]["mae_p"]),
                                                 cell(full["cx"]["obo_p"]), cell(full["quva"]["mae_p"]), cell(full["quva"]["obo_p"])]) + "\n")
    (TAB / "ablation_rows.tex").write_text("\n".join(abl) + "\n")
    (TAB / "app_ablation_rows.tex").write_text("\n".join(app_abl) + "\n")
    (TAB / "matrix_rows.tex").write_text("\n".join(mlines[:4]) + "\n\\midrule\n" + "\n".join(mlines[4:]) + "\n")
    numbers = {"protocol": "known-source v1 (training-only exclusion), three seeds, two-view mean unless stated",
               "full": full, "ablation": {lab: r for lab, r in rows}, "ablation_matched_seeds": paired_json,
               "rc_only_transfer": rc_only, "external_executed": ext}
    Path("docs/final_numbers_ks_20261117.json").write_text(json.dumps(numbers, indent=1))
    print("\n".join(app_abl))
    print("\n".join(mlines))


if __name__ == "__main__":
    main()
