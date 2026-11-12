# DeTRC prerequisites specification

Target: P1/P3/P4 of the fixed 2026-11-12 Astra TSN route. Base author commit 538abfcd24e0a81b383ca81862ff4360a2389c97 (67 verified source files); ARIS f76ec8d6b77792e2242cca049e08d0cbb7d54e19. Inputs: immutable author source, precise OpenMMLab TSN/R50 checkpoint, inherited full video metadata. Output: isolated server assets/runtime/candidate plus exact receipts and diff in this directory. Success: CPU asset/state/interface/build checks; no GPU execution. No training release, scientific redesign, source_sync/current_status/Git writes, shared environment mutation, feature computation or reduced experiment. P2/P5/P6/P8 remain separately gated. New compiler/runtime artifacts stay below the project server root.

| Requirement | Acceptance evidence | Initial |
|---|---|---|
| P1 exact TSN backbone | bytes/SHA and complete expected encoder-key/shape mapping | missing |
| P3 isolated legacy API/compiler/native ops | declarative spec, immutable package sources/build receipt; no kernel substitutions | missing |
| P4 mechanical fixes | explicit minimal diff, CPU schema/import and durable checkpoint checks | missing |
| Full extractor invocation | official raw JPEG pipeline, interval1, full T x2048; no computation now | missing |
| Existing work preserved | no writes to active jobs/shared env/immutable author source | required |
