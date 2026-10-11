# A005 Round 9 independent review request — candidate 20260819_195424

## Requested decision

An independent reviewer must issue exactly one fresh verdict: `ACCEPT` or
`REVISE`. This request is not a verdict and grants no authority.

The reviewer must read the complete candidate bytes from the beginning. Prior
Round 8 reviews, interrupted Round 9 work, author prechecks, file presence, and
candidate size are context only and cannot be inherited as acceptance.

## Exact review inputs

| Role | Path | Bytes | SHA-256 |
|---|---|---:|---|
| Candidate MD | `refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_20260819_195424.md` | 2,052,203 | `38e607a36d4de8f70daf5ea4ae612377a74004f860aa102819dac29b1d1585c1` |
| Candidate JSON | `refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_20260819_195424.json` | 2,024,588 | `dc8e205ad9b1ba842d692756a7a2fddc8dc1f1229533c027e522920d61242ba6` |
| Round 8 review MD | `refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_REVIEW_ROUND8.md` | 19,501 | `27e7f6080b5af427257b5cbf0bae70cd2121e43e9801df69628720031c9c5165` |
| Round 8 review JSON | `refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_REVIEW_ROUND8.json` | 15,270 | `71453a8db8eaaa9c1253e8bb9467ae2a8d8d915c585eda08e9fc0839b5e1038c` |
| Handoff | `TEMPORAC_HANDOFF_20260817.md` | repository bytes | reread current branch bytes |

Before semantic review, the reviewer must independently recompute every listed
byte count and SHA-256. Any mismatch is immediate `REVISE`.

## Mandatory closure questions

1. **Post-attestation restoration:** When FD1/FD32 and `32 -> 1` remain usable
   but only FD33 or `33 -> 2` fails, do every normative prose/JSON/mirror path
   require exactly one sequence-zero `0x7f` frame, EOF, and exit 123? Do FD32,
   `32 -> 1`, or unusable-FD1 failures remain no-frame exit 124?
2. **Persistent root poison:** Does every normative path allow authorized P00
   either to repair the same observable root in place followed by complete fresh
   validation, or to replace and explicitly rebind the observable root? Is an
   unconditional new-root-identity requirement absent?
3. **Ptrace resume state machine:** Are `ATTACH_BOOTSTRAP`,
   `AUTHORITY_TRANSITION`, and `FAILURE_TERMINATION` mutually exclusive and
   complete? Are special attach/bootstrap EXIT and cleanup/SIGKILL paths outside
   the ordinary authority-transition gate without creating an unvalidated
   userspace-resume gap?
4. **Audit outcome selection:** Is runtime row selection based only on observable
   callback inputs, minimally `(K,current stage)`? Are `action` and
   `mapped_token_or_null` outputs of the uniquely selected row everywhere,
   including catalog, builder, source manifest, uniqueness, mirror, and prose
   rules?

## Frozen invariants

The review must reject any change to scientific method, jobs, datasets, claims,
57-owner order, 1,034 stage-input rows, 54/106 pre-G5A projection, BASE19,
FINAL24, 16-token/13-origin vocabularies, or 10+64 CPython configuration fields.
It must also confirm JSON keys/order, array lengths, and all non-string values are
unchanged from candidate `20260817_134853`, except the successor timestamp is a
string change.

## Required reviewer outputs

Produce a new timestamped Markdown and JSON review pair containing:

- exact candidate paths, byte counts, and independently recomputed hashes;
- reviewer identity/family and an explicit independence declaration;
- complete findings with stable blocker IDs and severity;
- closure status for all four mandatory questions;
- frozen-invariant evidence;
- `ACCEPT` or `REVISE` with no ambiguous provisional synonym;
- an all-false authority vector and empty `authorizes` array;
- an explicit statement that even `ACCEPT` grants no experiment authority.

If and only if the independent verdict is `ACCEPT`, the successor may proceed to
the separate fixed-alias byte-copy step. It still may not rebind the effective
contract until the accepted review pair is frozen and independently rehashed.
