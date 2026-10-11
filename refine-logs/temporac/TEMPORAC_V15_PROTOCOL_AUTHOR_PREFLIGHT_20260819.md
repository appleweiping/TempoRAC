# TempoRAC v15 protocol author preflight — 2026-08-19

## Decision

`AUTHOR_PREFLIGHT_PASS / INDEPENDENT_REVIEW_REQUIRED / NO_LAUNCH_AUTHORITY`.

This document is an author-side reproducibility preflight, not an independent
review and not an `ACCEPT`. It grants no P0/P1/P2/P3/P06/S0, data, server, GPU,
training, evaluation, result-promotion, paper, git, or launch authority.

## Global contract blocker

The A005 fixed aliases remain byte-identical to the formally rejected Round 8
candidate. The unfinished Round 9 timestamp pair also remains byte-identical to
the handoff snapshot. Consequently the active 665-byte effective contract cannot
be rebound and the frozen 27-job campaign cannot be reopened.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| A005 fixed MD | 1,638,523 | `6e0ce987d88c1d3f7abbaec47d28630efdc9a3012c7192da6b692feb9f55d981` |
| A005 fixed JSON | 1,613,302 | `b135fc10c01aa6611af131d44b5ca762d802df514184ba6c76dfe5b6c7a80ef0` |
| A005 Round 9 candidate MD | 2,048,872 | `7da6647f7c37f981dd516d7b152123a156b206732a77a1ee82c7a3e492ab77bb` |
| A005 Round 9 candidate JSON | 2,021,793 | `4cff6d4ac91c0114093ee6aeed0247eb2fcda2bc6f4192c266ccfe67977ab898` |

## v15 candidate binding

v14 generated v2 landmarks but certified against v1 landmarks. v15 fixes this
candidate-only defect by copying the certificate into a new file and binding
both runtime and certificate to `full_chain_integer_landmarks_v2`. All earlier
files remain unchanged.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `full_chain_landmarks_candidate_v2.py` | 2,916 | `fa54a056602aa54ba4031f16b72bb3f5b108cbbd94c622d8231eb5d3a02b5b41` |
| `certify_target_adaptive_v7.py` | 11,779 | `3857188c5b0711e4caf8750a68ede5eeb21d9b3eceba8be4415059ff9402d833` |
| `campaign_runtime_candidate_v15_landmarks.py` | 4,252 | `61d36abb7063f689d0996aab8d6c5c2b769886ccd03cf96af2350f8457f1e713` |
| `campaign_runtime_candidate_v15.py` | 1,515 | `2b7606806bd3affd197d15705251a475ca3443ae6ba87df53ffcfe34c63930bc` |
| `smoke_temporac_pseudo_targets_v10.py` | 3,696 | `2d0f00f6d833f717d89adda8610a8687588d80021a415ac907269b5d3c3035ff` |
| `test_certify_target_adaptive_v7.py` | 896 | `afa9e6054efb17bad5b48d3c51e8f982d659fe39223ea7adc97e252acd850c10` |

Repository base commit: `49f820ff8b3a76bd6e9a630b664281427eb84389`.

## Bound evidence

- Train population: 268 identities / 110 videos; SHA-256
  `c96fc1dfa233ec4bf1af19e7480ee8c721f4cddee9f121185f4e494c7244e1eb`.
- Validation population: 134 identities / 51 videos; SHA-256
  `4064ef24dc2970e9b07de8adddf1ecd4932aecb90453a67b4f90d7bd35d10251`.
- Teacher checkpoint SHA-256:
  `5e5588cf2439154aff2ef40538e800e3035c6770c64318b623287c7c3b2bee50`.
- Ruff: PASS on all v15 candidate and test files.
- X0 frozen-ledger equivalence: `1 passed` in the server Docker environment.
- v15 train: 213/268 certified (79.48%).
- v15 validation: 109/134 certified (81.34%); numeric entrance threshold met.
- Unified result ledger SHA-256 before this preflight:
  `72326cda994c7a2e6b62aa0fcc0e6791a1ab2036dd8fc602766940252d9dfa70`.

## Mandatory independent-review questions

1. Confirm v7 changes only the natural landmark selector and remains ledger
   identical to the frozen certificate for X0.
2. Confirm runtime and certificate recompute exactly the same v2 landmark bytes.
3. Review the use of private v3 helper functions and decide whether they must be
   copied into a self-contained normative implementation.
4. Treat 109/134 only as a diagnostic result: v15 was formulated after inspecting
   this validation population. Require a fresh untouched audit population or a
   prospectively frozen replay before granting scientific entrance authority.
5. Do not inherit acceptance from v15's numeric threshold. Independently review
   A005 Round 9, rebuild the effective contract, implement the native runtime,
   and pass P00/P06 before any server campaign.

## Author conclusion

The implementation mismatch that invalidated v14 is fixed and the candidate is
reproducible. Scientific independence and global execution authority remain
unresolved. The correct next state is review hold, not campaign launch.
