# Data readiness — new server, 2026-11-04

The new server inventory found an existing larger MultiRep candidate with **794 train / 183 validation / 180 test video directories**, with corresponding MP4 and JSON counts. This is 1,157 videos in total. The inventory recorded filenames and sizes, without opening test annotations.

This is a **candidate release**, not yet a certified protocol. Matching the paper's aggregate video count does not prove release identity, a correct split, or source-video disjointness. The observed split proportions also require reconciliation with the paper's stated split description. Different composites can reuse source clips while having different video hashes.

A complete filename-only audit split composite directory names on `+`. All components matched the observed source-name patterns. It found 174 / 85 / 121 unique source-name proxies in train / validation / test: train–validation intersection 0, train–test intersection 36, validation–test intersection 42. This is strong evidence that the supplied split is not source-name-disjoint; canonical source identity still needs confirmation. No test JSON or label value was opened for this check. Preserve the original benchmark and report this limitation; do not silently delete overlapping or difficult test cases to improve a score.

Two other directories named `data/MultiRep` contain only **14 / 13 / 14** videos. They are not complete main-experiment inputs. A `MultiRep_lite` path and dummy-feature generator were excluded. These distinctions prevent accidentally treating a convenient small directory as the full benchmark.

RepCount candidates contain **758 train / 131 validation / 152 test videos**. Train and validation annotation row counts were consistent with these inventories. Test annotation contents were not opened. Pose caches found nearby are incomplete or belong to other settings; their existence is not evidence that all inputs for TempoRAC are ready.

The full read-only train/validation MP4 integrity audit has been started against all **977** candidate videos. It records file bytes, SHA-256, header findings and exact duplicate groups; it does not decode every frame, read annotations or produce model metrics. Its completion receipt is still pending. The private original inventory JSON was copied to the isolated TempoRAC server root and its SHA-256 matched the E-disk copy (`b5a866abd8e35f144f4fe7f0e8b64db7e92272ed1e5cbe0955b840ba7a57bd7a`).

Before training, reconcile the official release/split, extract source-video provenance and verify that no source crosses an independent evaluation boundary. Then build complete pose/track inputs while retaining physical frame clocks and observation support. Do not copy existing test labels into the learner or tune on this test split. There is no formal train/evaluate result in this readiness report.

Raw paths, complete file lists and process IDs remain in the private execution directory. Existing datasets and environments were inspected read-only. No STaR-Body process, other user's process or shared dataset was changed.
