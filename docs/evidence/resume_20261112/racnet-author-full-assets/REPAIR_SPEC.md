# R1–R3 recovery repair: bounded acceptance contract

ASK=never. Base: preserved_pre_repair_r1_r3/ (standalone private packet, no Git commit mutation). Prior reviewed runtime/control/prelaunch hashes remain preserved. Same scientific contract and all 61 objects; no download, probe, training, GPU, environment install, shared status or Git edit. Only server CPU fault injections.

| Requirement | Check | Initial status |
|---|---|---|
| R1: identity-verified final file recovered after rename/state interruption; coverage and deadline retained; no request | server fixture: forced save failure after rename, reload with segment journal and final-only state; wrong-SHA final rejected | pending |
| R2: original deadline/reserve stop preserves received bytes using capacity allocated before receive; no network deadline extension | server fixture: real mocked-socket RangeResponse deadline failure and reserve failure after partial read; verify durable bytes/receipt/original failure | pending |
| R2: failing checkpoint does not skip request receipt or retry same commit | server fixture: injected journal/checkpoint failure, inspect attempted/received/durable accounting | pending |
| R3: active RC member survives nested UCF progress plus failure and explicit replay | server fixture: nested extraction failure with two simultaneous progress files; replay exact partials | pending |
| Existing prepared state remains attempts=0, no deadline, G1 coverage/bytes unchanged | explicit rebind invariants and original hashes | pending |

Interface decisions declared before code: terminal buffer flush may proceed after deadline only into the exact physical range allocated with posix_fallocate before its receive; it makes no additional network read. Failure receipts distinguish attempted, received, and journal-recoverable durable bytes; stdout carries the same receipt if metadata persistence fails. Object final path is adopted only after full expected size/SHA check. Extraction progress is scoped to the exact member path. All are operational interfaces; no new scientific assumptions or input contracts.
