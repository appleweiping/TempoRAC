# Targeted F1 repair — ready for same-session closure

Raw independent finding: preserved INDEPENDENT_REVIEW.md/json and before_review_fix/. Existing raw CPU receipts, contract, public allowlist and all source were copied first on both E and server. They are unchanged.

Only production-driver change is produce_features.py. H5 reuse and missing-H5 capacity now share h5_committed(), requiring an existing matching contract/population/count/frame/size/hash completion receipt. A final H5 with no valid sidecar is replayed through .h5.pending using the same validated committed pickles and unchanged FP32 value assembly, then replaced and receipted. JPEGs/pickles are not deleted or altered.

One server-only fixture exercised final-H5-present/sidecar-absent, correct replay reservation, recovery, strict array equality, committed H5 reuse and unchanged pickle/receipt byte hashes. The fixture used only2 synthetic arrays (3 and2 rows x2048); no model, real dataset decode, native GPU dispatch or production. Exact fixture source, log, JSON and raw patch are present here. No old CPU/P1/P3/P4 check was rerun.

New contract: e41f69c549ec57762f0876595fd8aed282b87a9ebd2687cd6239a4431be52c47
New produce_features.py: dfd6ad344c1792b3f6e322b20642e163280e7fef740e56cf4374e526f2eeb802
TARGETED_REVIEW_FIX.json: 101a27b8a277b35e21eaa0238d581b218b99209fe0fc454e6606dda00a23144f
Prior contract retained:43b5fb151a7b3dc4471f14cf2fe3f229e9efb8c02b87e61fbdc5614fe1094570

RUN_CONTRACT/SAFE_MANIFEST/PUBLIC_ALLOWLIST are refrozen to the one changed source identity; matrix, native numerical code, source/env assets, populations, ROI, resource policy and metrics are unchanged. Earlier CPU receipts keep their original identities; this affected-interface receipt binds the new final freeze. Requested Sol, actual backend/effort unattested, same-family provisional; reviewer followup still required. Root alone releases production.
