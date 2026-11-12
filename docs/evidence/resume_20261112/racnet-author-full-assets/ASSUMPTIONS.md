# Prelaunch ledger

ASK=never. The scientific scope is already frozen by root/Astra; no new training decision is made.

| ID | Decision | Class | Source |
|---|---|---|---|
| A01 | Exact61objects/fixed revision; order .00→UCF→.01..59; no Countix | semantic | user/amendment |
| A02 | One writer/process; explicit --resume only, original absolute48h deadline retained across resume | interface | user/default |
| A03 | Asset budget uses conservative logical file sizes plus retained G1; volume-free checked before writes/receives. Terminal flush uses physical capacity reserved before receive as specified in A13. Original G1 remains unchanged | interface | default/sweep R2 |
| A04 | Network gaps use <=32MiB requests, connect10s/idle30s/request300s; zero retry; one new canonical fixed-revision request per missing gap | interface | default based on successful G1 |
| A05 | Commit received bytes in <=1MiB segments with fsync, SHA journal then atomic state. Replay durable segment journals on explicit resume; adopt a final object only after full expected size/SHA verification (A14). Unjournaled partial bytes remain uncovered | interface | default/sweep R1 |
| A06 | Streaming gzip cannot serialize its decoder. Resume replays existing compressed bytes/read-only member checks without re-downloading or creating a joined archive | interface | default |
| A07 | Preserve raw keys. frames OR img_feature is a candidate feature payload; missing literal frames/length only flags an interface gap. Applicable RC requires real feature payload and author gt_tsm. No alias or synthesized length/label/TSM is written | semantic | root explicit clarification |
| A08 | Native755 eligibility is future RACnet-only qualification; excluded null/source-overlap raw entries remain archived. Current TempoRAC/TransRAC756 is unchanged | semantic | amendment |
| A09 | UCF only the five frozen Handstand→HandStand names map; validate105one-to-one, scalar counts and full_N/ROI_N relation, no content-parity claim | semantic | amendment |
| A10 | Safe numeric NPZ/CRC validation,1GiB individual member/nested expansion ceiling; no pickle/object/complex payload is loaded | interface | default from known max frame sizes |
| A11 | Raw redirect headers/URLs stay private R/E; stdout/status contain no signed URLs. Public summaries require separate redaction | interface | user |
| A12 | Any malformed/missing required payload, unsafe member, network/hash/disk fault ends current finite process with durable failure evidence. No source revision fallback | interface | user |
| A13 | R2: physically allocate the next checkpoint range using posix_fallocate before receive; unsupported allocation stops before receiving it. After terminal deadline/reserve failure, only flush already received bytes into this allocated capacity, with no network deadline extension. One failed commit is not retried. Receipts independently record attempted/received/journal-recoverable durable bytes, original error and checkpoint error; metadata failures also emit the receipt to the process log | interface | sweep R2 / default |
| A14 | R1: before journal replay or missing partial creation, reconcile a present final object by exact expected byte count/SHA; recover full verified coverage and persist it without resetting deadline, requesting bytes or deleting original partial evidence | interface | sweep R1 |
| A15 | R3: each extracted member owns a progress JSON alongside its partial. Nested UCF retains the active RC checkpoint. RCReader keeps the frozen .00→UCF→.01..59 order | interface | sweep R3 |

No scientific acceptance or training-ready verdict follows from downloading archives. The review is transport/resource engineering only, same-family provisional.
