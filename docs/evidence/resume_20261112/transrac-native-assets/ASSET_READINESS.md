# TransRAC native assets and annotation readiness

This packet implements the frozen Astra input decisions without changing the official model or launching computation. Original tree: `/root/autodl-tmp/TempoRAC/third_party/TransRAC`, clean commit `68bdd4daa60ed7c3174a7f6bf86f6537b6fa0979`. New artifacts: `/root/autodl-tmp/TempoRAC/baselines/transrac-native-assets20261112`.

| Required item | Result | Evidence |
|---|---|---|
| Official127,414,861-byte Swin-T K400 backbone | Pass at2026-11-12 14:35:19UTC | Full SHA9950e3be6b0b3763f80575dfbbe7db7cbeb18345e32363b73e8d44952ef76acc and all122 fixed chunks match known-good E source; BACKBONE_READY_RECEIPT.json |
| Original CSV758/131/152 identities | Pass | Exact SHA4cfee3cf...f857 /49695468...d148 /9779090e...12e2, bound in ANNOTATION_ASSET_RECEIPT.json |
| Prepared native CSV756/131/152 | Pass | Only stu3_27/stu4_50 removed from optimization; L fields checked finite/exact-integral and serialized as integer strings, no rounding |
| Original1041RC media availability | Pass for existence/size | All resolved paths insideR exist and match frozen byte sizes; prior full media SHA evidence reused, no redundant rehash or decode |
| Complete prepared population | Pass for identity/availability |1244 entries =RC756train+131valid+152test+105UCFtest+100QUVAtest; manifest hash below |
| Missing/zero/disputed cases | Pass | stu6_44 missing scalar remains null/empty with bounds[]; it is not relabelled0. Zero validationstu9_67/testtrain143 and disputed teststu4_5/stu4_27 retained |
| Server environment inspection | Completed, not a readiness claim | Docker absent; /root/miniconda3 conda24.4.0/basePython3.12.3; main envPython3.10.22. Isolated new environment installation belongs to implementation executor |
| GPU/model/cache work by this asset task | None | No model/framework import, checkpoint deserialization, RGB decode, feature/cache generation, metric recomputation or GPU execution |

Prepared files under server asset root:

- `inputs/train.csv` SHA `b4a7c0e360d87e47de21e0f52af5ba1250be5ccfa3a500f05dfa4dbd9ed88c83`
- `inputs/valid.csv` SHA `498087551ad3464f4bbb293e4701d16136be3f11daa37e7fd80a43ec01ab43ae`
- `inputs/test.csv` SHA `b89b7f3ba54bc29c72d460fed67d94313a4a7844011b56887da81b8774a290ec`
- `inputs/manifest.json` SHA `61cbbd8c98c7a9764284dd2f032d85d2cfb9403fedabf695fe90271f31fa2333`

Corrected source manifest `/root/autodl-tmp/TempoRAC/data/UNION_known_source_v1/manifest_dev.json` SHA `96c068ee7a06f0b952610abdfc875e78752f8fe9b5181868c8bc1cec82021fb2`. All source CSV count/bound values match the original frozen manifest; retained rows match corrected manifest exactly. The prepared manifest uses `source_video_sha256` and records filesystem availability separately from inherited content hashes.

Official backbone URL: https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_tiny_patch244_window877_kinetics400_1k.pth . This is the K400 recognition backbone, **not** a TransRAC counting task checkpoint. ExistingE asset was found in `aris-work/external-baselines-20261105/readiness/assets`; slow transfer prefixes were retained and later nonoverlapping mirror ranges continued the same known asset. All partial attempts/logs remain underR. Only the final full expected SHA permits promotion to `.pth`.

The first full-sized assembly failed its SHA before promotion (1485bf17...f6ff). Authorized static whole-file/chunk hashing confirmed the existing E asset's full9950 identity. All122 fixed1MiB chunks were compared:113 matched and exactly9 differed at offsets4,194,304..13,631,487, confined to the mixed-transfer prefix. The retained direct official first2,052,096bytes also matched. A delayed SFTP write during the switch to HTTP resume is a likely cause, not independently proven. Only that9MiB known-good E span is transferred for correction; the bad assembly, range receipts and `BACKBONE_INTEGRITY_FAILURE.json` are preserved. The final gate rechecks all122 chunks and the full expected SHA. Binary backbone/span files remain separate from the compact readiness receipts.

Correction completed only after SCP exit0. A separate copy was patched, every chunk and the full9950 SHA passed, then the usable `.pth` path was created. `ASSET_READINESS.json` SHA is `48e0a2f912966a70135570a5dff6f3e01a36de7bb38380cd3a6b32816b878097`; `BACKBONE_READY_RECEIPT.json` SHA is `882487f143dafd1bba92ec58848a6e9ea1458d89047892a3530bd3674b3370cd`. The old failed assembly was not overwritten.

The initial strict count parser stopped before writing any prepared CSV when it encountered source-empty/frozen-null stu6_44. `PREPARE_ATTEMPT1_FAILURE.json` and original helper remain; v2 accepts only exact empty↔null identity. Astra explicitly confirmed retention, and native empty-bound target semantics are left to the original function. No synthetic scalar label was introduced.

The pinned run-experiment SKILL and complete compute-env-contract reference were reread at ARIS f76ec8d6b77792e2242cca049e08d0cbb7d54e19. Existing modern environment recipe was read; no claim of original2022 numerics or current4090 readiness is made. Requested executor Sol; actual backend not independently attested. Root owns source publication, implementation review and manual GPU handoff after a complete arm boundary. The active21-arm group was untouched.
