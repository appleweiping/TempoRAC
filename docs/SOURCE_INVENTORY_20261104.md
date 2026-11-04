# Source inventory — 2026-11-04

This is the label-free canonical **byte-identity** provenance audit for the
existing TempoRAC recovery, not a new research route. It keeps all 794 train,
183 validation and 180 official-candidate test identities. No sample is deleted.
The full current train/validation composite byte audit is reused rather than
rerun. Official release identity and synthesis recipes remain separate checks.

## Meaning and assumptions fixed before implementation

| ID | Chosen rule | Provenance |
|---|---|---|
| A001 | Decompose composite names on `+`; map each token to the complete set of exact, case-sensitive original-media stems. No lowercase or fuzzy matching of tokens. | User |
| A002 | Canonical IDs are original-media SHA256 values. Multiple paths with identical bytes resolve to one byte identity; multiple distinct byte values for a token remain ambiguous. SHA does not establish recording identity after transcoding/cropping. | Declared default |
| A003 | Current exposure includes full train and validation. Historical pilot/v4x and other design exposure is not fully reconciled; every B result is explicitly a candidate. | User |
| A004 | Inspect media filenames under three explicit original-media roots; read bytes only for tokens referenced by any current composite. Read no annotation, count, period, or metric. | Declared default |
| A005 | Existing composite duplicate evidence is retained, without reverse-inferring original-source equivalence from composite equality. | Declared default |

Filename decomposition is a provenance candidate edge, not a verified generation
recipe. Even a unique byte match does not verify the source was used in the
composite. Candidate disjointness is limited to this graph and current exposure;
it is not a claim of fully independent test recordings.

## Implementation and execution

`scripts/audit_temporac_sources.py` uses Python standard-library CPU I/O only.
Its config explicitly selects the metadata inventory, composite root, population,
three RepCount original-video roots and both existing composite-audit receipts.
It verifies live composite directory membership against that inventory, checks
that the reused receipt covers exactly the full train/val identities, lists all
source candidates, and hashes every distinct relevant resolved path. A before/
after size and mtime check detects media changes during reading. All duplicates
and ambiguous paths survive in the output.

F0 completed the actual full filename inventory: 302 source tokens, each with
one exact candidate in each of three roots (906 realpaths). The three original
trees each have 1,041 media files. A separate metadata check found 906 distinct
`(st_dev, st_ino)` identities among relevant realpaths, so none were the same
physical file eligible for mechanical hash reuse.

Full hashing completed on the server under PID **1532757**, single worker,
`nice -n 15`. Its initial idle I/O class was changed in place to
`ionice -c 2 -n 7 -p 1532757` to progress while protecting shared I/O. No other
process was changed. Raw run root:
`/mnt/4tssd/lilei/TempoRAC/runs/canonical-sources-20261104/`.
The initial command was:

```bash
nohup nice -n 15 ionice -c 3 python3 -u "$run/code/audit_temporac_sources.py" \
  --config "$run/config.json" --output "$run/full" > "$run/run.log" 2>&1 < /dev/null &
```

`launch-inputs.sha256`, `job.pid`, config, source code and logs bind that launch.
The status receipt additionally binds script/config/inventory hashes; the final
receipt binds every output artifact. Private local copies and the pre-code
specification/assumption ledger are under the E-disk `aris-work/canonical_sources`.
No credentials are included in these artifacts.

The run completed **906/906 files**, **8,405,836,827 bytes**, without read errors
in 340.464 seconds (reported hash-stage elapsed time). All 302 tokens resolve
to one byte identity each across their three copies, yielding **301 canonical
byte nodes**. The only different-token byte-alias group is `stu3_27` and
`stu4_27`, SHA256
`2a7c65f8adc0924c232d75a9dc02a0db6f44b15140c422c8fbbd551f59f6d195`.
This is directly confirmed from the six original-media files across the three
trees, independently of the four known duplicate composites. The original
RepCount copies place these two aliases in train and test respectively; these
are source-dataset split names, distinct from the MultiRep composite split.

Final A membership is **180/180**: **78 B candidates**, **102 overlap**,
**0 unknown under this byte/path audit**. The previous name-proxy 78/102
partition happens to be unchanged after source-byte alias verification; the
manifest retains each row's reasons and the alias evidence. This does not
resolve recording-level unknowns, recipe identity, or historical exposure.
The conservative full graph has **45 connected components**; **23** touch A
and only **9** touch the B candidates. Thus 78 candidate videos must not be
treated as 78 independent recording clusters for interval estimates.

Server closure verified the full artifact hashes, code/config/inventory binding,
job PID and exact 794/183/180 membership. The metadata-only evidence archive was
copied to E and every closure/artifact hash was rechecked locally. Archive SHA256:
`1e5c710c567c0cab730bcfe5c0a61eb7137b8210c97e06fc2fd14fe66e6c66d0`.
Source script SHA256:
`77e6168cf664dd78c4a36d9b3f44bec2e9c58b08d9552dd09ce386efaf23daf5`.

## Artifact interpretation

| Artifact | Content |
|---|---|
| `source_paths.json` | Every token and full lexical/resolved candidate path set; empty sets explicit. |
| `source_media.jsonl` | One byte hash/read-error record per distinct relevant realpath. |
| `canonical_graph.json` | Token/path/byte nodes; missing, read-error, distinct-byte ambiguity and resolved-byte states. Different-token byte aliases remain visible. |
| `membership.json` | Every A180 identity plus all 977 development identities and current exposure set. |
| `source_components.json` | Conservative connected components from shared tokens or any possible byte identity, including ambiguous possibilities. |
| `composite_audit_reused.json` | Receipt paths/hashes and all four known composite duplicate groups retained. |
| `status.json` | Progress, population, input/output binding, final classifications, limits and alias groups. |

An A identity is `overlap` if it shares an exact token or any possible byte ID
with current development. These two reasons are recorded separately; ambiguous
byte intersections are conservative potential overlap, not definite source
identity. Without such overlap it is `unknown` if any source token is missing,
unreadable or has multiple byte identities. Only fully resolved, byte-disjoint
identities are `B_candidate`. This keeps overlap evidence even when another
constituent is unresolved; each row still records all unresolved constituents.
All 180 identities remain in A regardless of classification.

## Acceptance evidence

| Requirement | State | Evidence |
|---|---|---|
| Preserve full population/A180 | Passed mechanically | Input/live-directory checks and closure verified all 794/183 development and 180 distinct A identities. |
| Exact path sets, explicit ambiguity | Passed mechanically | F0 inventory contains all 302 tokens and 906 paths; focused tests cover case, missing, multiple paths, distinct bytes and read errors. |
| Full relevant original SHA audit | Passed mechanically | 906/906 hash records, zero read failures; final status/closure and local archive restoration verification. |
| Reuse 977 composite hashes | Passed mechanically | Existing complete receipt checked for exact identities; no composite media bytes opened by this script. |
| No test-label/model/GPU access | Passed for implemented scope | Only config, inventory, prior audit receipts and original-media bytes opened; end-to-end fixture succeeds with invalid test annotations and absent composite media. |
| Candidate/overlap/unknown and clustering | Passed mechanically | Full A:78 candidates/102 overlap/0 byte-path unknown; 45 full components, 9 touching candidates. Fixture verifies ambiguity/missing/error semantics. |
| Tests | Passed | `python -m pytest tests/temporac/test_source_inventory.py -q`: **5 passed** (2.74 s), E-disk basetemp/cache. |
| Historical exposure and recipe verification | Partial | Explicitly unresolved; no claim that B independence is fully established. |

No model performance is produced or inferred. Fresh semantic review is requested
through the parent task; until a completed review exists, this is deterministic
execution evidence and declared assumptions, not accepted scientific assurance.

## Parent review checkpoint

Fresh Sol/xhigh technical review completed with no concrete source-script blocker. Decode-script reporting defect was fixed and rechecked in the same review session; the combined suite passed16 tests. Complete original finding and recheck: `INPUT_AUDIT_REVIEW_20261104.md`. This is same-family provisional assurance; it does not certify independent recording identity.
