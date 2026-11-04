# NTR1 prepared cache execution checkpoint — 2026-11-04

This checkpoint adds persistent prepared features and derived NUDFT/routing/NOLA data, with an explicit cached runtime and guarded first-seed campaign entrypoint. The original scientific modules and plan remain byte-identical to parent commit `591b4659c5c94ed7b9538ca8b1dd4fc1bdede375`. The plan file SHA is `41405fdacb7648ab66b0d456b842085e60497967d63693c4cd3616752a96e382`; its scientific payload SHA is `8d50706a22bcfc47206b37ae399c56c2dbf70c99640c573d49ff20efa11259c3`.

## Verified scope

- Exact new source hashes and retained evidence digests: [public evidence receipt](evidence/ntr1_cached_execution_20261104.json).
- Executor evidence: 14 cache tests; nine original production-runtime CPU tests; one added stale-parity rejection check; six original handover CPU tests. Independent reviewer counts are recorded separately in the receipt and are not summed as distinct coverage.
- Bounded production parity: N=216, N=2640 and five-run N=900 each passed 84 exact-byte checks. The larger two cases used the earlier builder with the same final mathematical kernel hashes. Three workers processed six existing cache hits; this is not a full-population build or throughput claim.
- The fresh scoped review is same-family provisional; actual backend identity is unattested. Earlier stale parity and the missing campaign lock were concrete findings, fixed and preserved in the raw record. A subsequent CPU-affinity delta limits overlapping campaign work to the declared CPU set.

## Execution contract

The full inventory is 2364 identities (1878 train, 486 dev). The frozen eligible training set is 1084 natural identities plus 224 X0 views. Builders accept one to three workers, with one numerical-library thread per worker. Progress distinguishes verified inputs from durably committed cache entries; partial or failed entries never count as committed. Cache manifest order follows the frozen input order.

Training and tuning load one verified train cache. Prediction consumes only the separately completed dev cache. The first pair remains seed 20260815, canonical then capacity control; exact current R3 lease, cumulative budget, source/binding receipts and the actor-held campaign lock are required. The original source tree and earlier evidence remain available.

## Acceptance boundaries

This is a source and engineering checkpoint. Complete cache manifests, live resource admission, actual process handover, full training/tuning/prediction/scoring and scientific interpretation require their own receipts. No performance, SOTA, component contribution or full-population equivalence claim is assigned here. The initial missing hot-loop replacement, earlier stale parity, campaign-lock finding and local Torch2.12 checkpoint failure are retained; the production Torch2.0.1 nine-test runtime result is reported separately.
