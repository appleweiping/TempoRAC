# Literature delta for future TempoRAC design — 2026-11-11

Scope: fill missing coverage only; no experiment, source code, current_status, server model or frozen run matrix changed. Requested routine executor: Sol; actual backend variant is not attested. This is an evidence ledger, not a new novelty verdict.

| Acceptance requirement | Status | Evidence |
|---|---|---|
| Exactly 15 related framework entries with identity, venue, source/code, read depth, architecture, scale, metrics, ablations, protocol | PASS for ledger coverage | COVERAGE_15_20261111.json, 15 stable IDs |
| Reuse completed primary reading; do not imply all papers freshly/full read | PASS | Inherited receipt paths per row; new reading restricted to missing comparison counts/metadata/release status |
| Actual external-method counts, excluding duplicate backbones/checkpoint rows | PARTIAL | 13/15 entries have a counted table; MultiCounter+ and KSEM multi-stride full tables unavailable. ICONIP revision count remains separate from PoseRAC-v1 |
| Full scientific coverage for all 15 | PARTIAL | MultiCounter+ and multi-stride full texts missing; RepNet newly read only targeted §§6.1–6.3/T2–7 plus inherited source audit. No false 15/15 full-read claim |
| Primary-source implementation provenance | PARTIAL | Known pins retained; CountLLM/MFL/PAMS/ICONIP official runnable releases still unverified; D2-STX and TReViS official trees are placeholders |
| Inherit negative history and active freeze | PASS within this task | New outputs only in this literature folder; no experimental actions or repository mutations |
| Independent novelty/design acceptance | NOT PERFORMED | Future Astra design owns synthesis after full stage closure; no scientific PASS asserted |

## What was added

The 15 entries are RepNet, Context, Sight & Sound, TransRAC, MFL, ESCounts, CountLLM, DeTRC, D²-STX, PAMS, PoseRAC, MultiCounter, MultiCounter+, TReViS and Gan et al.'s multi-stride collaboration. This selection preserves the established comparison pool, covers RGB/audio/flow/pose/MLLM and native multi-instance settings, and keeps the closest multi-stride prior even though KSEM is not a top vision main conference. PoseRAC revisions are one family but retain separate identity/protocol within the row. The ledger does not call all 15 top-conference frameworks.

New targeted counts supplement the four already recorded papers. RepNet compares three QUVA methods but no external method on Countix test; its original OBO is **error**, so 0.3034 becomes 0.6966 accuracy only when explicitly converted. Context has four distinct cited algorithms in five rows, while its prose groups them into three families; the two Runia papers remain distinct and the Levy reimplementation does not inflate the count. Sight & Sound has three external frameworks/four rows; its UCFRep result uses sight only. ESCounts compares five RepCount, two Countix and six UCFRep frameworks; Context frozen/fine-tuned is one framework. DeTRC Table1 has eleven external frameworks across recognition/segmentation, RAC and temporal detection. MultiCounter has three MultiRep tracker+counter baselines, seven single-person transfer baselines, and two shared-MSTI period-head comparisons.

Two inherited primary PDFs were located in their original read-only temporary library and copied into this E-only folder for durability, without redoing their full read: D²-STX and PoseRAC-v1. D²-STX Table1 compares 4/6/2 frameworks on RepCount/UCFRep/Countix, Table2 five transfer frameworks. Its metrics include **RMSE and OBZ** alongside MAE/OBO, a useful correction to an MAE/OBO-only summary. PoseRAC-v1 Table3 compares seven external methods; its official evaluator's GT-count channel oracle and 21-video UCFRep-pose scope remain explicit.

TReViS Table1 has 19 external configuration rows but 15 distinct cited methods under the declared collapse rule; Table2 has 14 rows but 10 methods. Four synthesis+backbone compositions are not four new frameworks. Its three evaluation datasets are RepCount in-domain plus transfer, not three independent in-domain trainings. All TReViS mechanism and ablation facts reuse the existing full-text receipt.

## New primary-source deltas and unresolved coverage

[Gan et al., KSEM2023](https://link.springer.com/chapter/10.1007/978-3-031-40289-0_24) is now pinned to DOI10.1007/978-3-031-40289-0_24, pp295–306, first online9August2023. The publisher abstract confirms same-video multi-stride **count-result fusion**, adaptive temporal correlation, and ActCount172videos/~1870repetitions including non-human motion. The final chapter remains a subscription preview: no new full method, ablation, split or comparison-N evidence was obtained. This is direct prior against a broad “first multi-stride RAC” statement; it does not by itself settle token-level time-aligned fusion novelty.

The [official D²-STX repository](https://github.com/ZJU-Emerging-AI-Lab/D2-STX/tree/6e4614d3fce4ef0290ca6a6c6e9e7ff64901e895) is only README+LICENSE at the inspected immutable tree. A separate [wwwwwxa/D2-STX](https://github.com/wwwwwxa/D2-STX/tree/f9d23db20049bd572262a1e47c027253c21c2ef9) contains source, but direct paper-to-account authorship linkage was not established in this pass; its README has placeholder model links and says pose extraction is forthcoming. Treat it as a candidate source, not a verified official runnable baseline. The paper footnote and CVF metadata point to the ZJU repository.

[TReViS](https://github.com/yfqi/TReViS/tree/c4e355f6aa29fd66320048296fb583f22a04a092) remains a169-byte README-only tree. Its README claims IMAVIS2026, while the scientific reading is bound to arXiv2609.24367v1; final publisher identity/version remains unverified. No implementation release is inferred.

MultiCounter+ and TWCRAC retain their earlier blocked full-text receipts; no repeated failed full-paper requests were issued. TWCRAC is outside the selected15 but remains a mandatory named novelty gap, not silently deleted from the closest-prior set. The actual MultiCounter+ code interface mismatches and unknown final ablations remain visible. ESCounts/TransRAC/RepNet/PoseRAC/DeTRC immutable acquisitions are reused from the baseline agent, not redownloaded or called newly evaluated.

## Implications for later design, without changing this run

Three source-grounded overlap boundaries matter. Learned sampling/stride selection is already in Context and Sight & Sound; count-level multistride fusion is in Gan et al.; multi-scale correlation is in TransRAC. Therefore a future token-level common-clock fusion claim must explain its operation precisely and show where information is retained or lost, rather than claim multiscale sampling itself.

Training and inference mechanisms must remain distinct. PAMS's largest RepCount ablation lever is the inference counter; ESCounts's exemplar mechanism is training-time with zero-shot inference; TReViS modifies training data; DeTRC changes supervised output representation. Their gains cannot establish a contribution of an unrelated TempoRAC module. Existing full removal versus internal replacement distinctions and the less-than-three-loss requirement remain constraints for later design.

CountLLM's visual adaptation and periodicity alignment are useful causal evidence, but this task does not reactivate the cancelled LLM route. Likewise, the old mechanism notes contain historical TempoRAC numbers and proposed Qwen stages: those are **not imported as current result or authorization**. Only paper-specific source evidence is reused. The prior canonical/control negatives, PAMS reconstruction failures and WARP-PHASE half/double failures remain retained.

The later Astra design should use the closed current-stage results and this ledger together. Candidate innovation must specify a falsifiable difference from the nearest mechanism, a same-input comparison, and independent component evidence; it is not warranted to add new mid-run experiments. No new method, loss, hyperparameter grid or experiment matrix is proposed here.

## Remaining unread / unavailable

1. MultiCounter+ final full paper and relevant supplement: exact loss/call chain, long-short mechanism, all benchmarks/ablations/comparison-N.
2. Gan et al.12-page chapter: precise fusion architecture, adaptive correlations, data splits, ablations, code identity, full table counts.
3. TWCRAC full text retained from prior missing list; not repeatedly fetched without a new legal source.
4. D²-STX relevant supplement and official implementation/model/pose artifacts; PoseRAC-ICONIP official source and its distinct comparison-table count.
5. Newest-window exhaustive novelty search and fresh verdict are not claimed by this delta task. Existing dated search coverage remains bounded.

The existing dated literature numbers and notes remain source material rather than new benchmark scores. The frozen running full experiment stage remains untouched.

