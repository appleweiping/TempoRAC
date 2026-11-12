# Resilience v2: frozen operational amendment

ASK=never. Implementation skill already fully read at pinned commit f76ec8d6b77792e2242cca049e08d0cbb7d54e19. One bounded transport-only candidate; no scientific design, source inputs, models/seeds, metrics, schema acceptance, resource cap, original total deadline or other jobs change. Root owns focused final inspection/publication and the next actual continuation. No new generic review loop or network test.

Observed production failures: first TLS handshake timeout; second TLS read timeout after3,424,256B received/durable in its final request. Preserved frontier1,244,938,240; cumulative new received=durable1,177,829,376; attempts2; originaldeadline1794673524.2427886. These demonstrate a real repeated connection/read timeout, not a schema or content-validation failure.

Frozen policy (declare before code): connect30s, idle120s, each request300s, missing-gap request chunk<=128MiB, checkpoint remains1MiB. Retry only TimeoutError/ConnectionResetError/ConnectionAbortedError raised inside the HTTPS network client; not raw errors from disk, accounting, extraction, schema, host/status/range/length validation, TLS certificate or hashing. Timeout reaching original total deadline is terminal. No retries when persistence/final flush failed.

Budget semantics: at most3 consecutive automatic retries (backoff2,5,10s), so a fourth consecutive transient failure stops; at most64 automatic retries reserved across all restarts. A successful request resets consecutive failures only; total never resets. Existing2 manually authorized attempts remain history and are not rewritten as automatic retries. Each retry is charged and persisted before backoff/network work; a crash may consume that slot but cannot restore it. Failed request's newly durable prefix remains covered. The next request is recomputed from current missing intervals within the same original bounded frontier, never from zero or the original failed offset. Backoff consumes the original48h budget; insufficient remaining time stops without another network operation. Retry counters survive explicit resume. Request and retry receipts remain private in W.

| Acceptance | Server CPU fixture | State |
|---|---|---|
| Network-only retry filter; no content/disk/schema/certificate retry | Mock socket faults and direct terminal-fault injection, zero real sockets | pending |
| Partial receive resumes actual new durable frontier | Real Engine.request with mocked RangeResponse socket; inspect requests/coverage | pending |
| Consecutive3/total64 caps survive reloaded persisted state | CPU fault fixture at exact cap boundaries; no sleep/network | pending |
| Originaldeadline excludes another retry/backoff/network read | CPU clock fixture; counter/deadline unchanged except recorded failure | pending |
| Production bytes/config/state unchanged | Before/after hashes, staged candidate CONTROL and atomic proposal only | pending |

Interface decisions: marker identifies origin of retryable errors; persistent transport_retry state records budget; new private retry journal records each reservation/terminal budget decision. No new scientific semantic rows. Prior R1-R3 recovery guarantees remain required; targeted fixture covers their relevant flush/frontier behavior without rerunning the old full fixture campaign.

Cap persistence clarification before deployment: a terminal cap/deadline stop records terminal_stop_reason and rejects future request_missing calls before network on explicit resume. The first CPU receipt checked zero additional retry reservations after reload; final assertion additionally requires zero additional network requests. Pre-tightening candidate/fixture evidence is retained in pre-terminal-cap-guard/. Thresholds and all scientific contracts remain identical.
