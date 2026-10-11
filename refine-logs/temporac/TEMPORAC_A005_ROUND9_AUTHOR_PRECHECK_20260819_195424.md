# A005 Round 9 successor author precheck — 2026-08-19 19:54:24 +08:00

Decision: `TWO_IMPLEMENTATIONS_PASS / INDEPENDENT_REVIEW_REQUIRED / FIXED_ALIASES_UNCHANGED`.

This receipt records two read-only author-controlled terminal implementations.
It is not an independent reviewer verdict, does not authorize copying the new
pair to fixed aliases, and grants no implementation, gate, server, data,
training, evaluation, result, paper, git, or launch authority.

## Successor candidate

| Artifact | SHA-256 |
|---|---|
| Round 9 successor MD | `38e607a36d4de8f70daf5ea4ae612377a74004f860aa102819dac29b1d1585c1` |
| Round 9 successor JSON | `dc8e205ad9b1ba842d692756a7a2fddc8dc1f1229533c027e522920d61242ba6` |
| Python structural precheck | `e5feb3e1a288ebba0f150cb296134d3ab0fc043f20419b6dc04a0a8c1b69c56b` |
| PowerShell semantic precheck | `0572b99fc6db601ff621152e907e5aea26275ba747501db607aa1ed53c61796f` |

## Repairs represented

1. FD33 or `33 -> 2` as the sole restoration failure now emits the exact
   sequence-zero `0x7f` failure frame on usable FD1, then EOF, and exits 123;
   FD32/`32 -> 1` or unusable FD1 remains no-frame exit 124.
2. Persistent root poison can be repaired in place with fresh full validation,
   or the root can be replaced and explicitly rebound by authorized P00.
3. Ptrace resume authority is split into mutually exclusive
   `ATTACH_BOOTSTRAP`, `AUTHORITY_TRANSITION`, and `FAILURE_TERMINATION` states.
4. Audit outcome selection is uniquely keyed by observable `(K,current stage)`;
   action and mapped token are selected-row outputs, not selector inputs.

## Precheck evidence

- Python/Docker strict duplicate-key and recursive structural comparison: PASS.
- Changed JSON string paths: 26.
- Object keys/order, array lengths, and every non-string JSON value: unchanged.
- MD normative appendix bytes equal standalone JSON bytes: PASS.
- PowerShell fixed/old hash and obsolete/new semantic probe: PASS.
- Fixed aliases retain rejected Round 8 hashes.
- The `20260817_134853` unfinished candidate retains its handoff hashes.

## Required next action

A genuinely independent reviewer must reread the complete new MD/JSON bytes and
issue a fresh verdict. Until then, do not copy the successor to fixed aliases,
rebuild the effective contract, modify `contract.py`, or launch experiments.
