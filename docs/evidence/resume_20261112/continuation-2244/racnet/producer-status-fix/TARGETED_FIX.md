# Producer-status interface fix

Actual producer acquire_assets.py:344 commits STATE.status=complete_assets_not_training_ready and COMPLETION.json(status,objects_verified61,archive_bytes128629792634,training_releasedfalse,control_sha256). Publisheda50 receipt-only qualifier expectedcomplete/completed; actualauthorcompletion would be rejected beforewrite. This is a concrete unexecuted-interface defect; originalreview/publication and priorfixture remain retained.

Boundedfix: requireonlyactualproducercompletionstate and existingCOMPLETION.json status/count/bytes/control/training-notreleased semantics, retaining61verified and original receipt-only qualification. NoauthorSTATE/control/worker/science edits; no GT/decoder/oldtarget/featurepayloadread/hash. One metadata-only actualproducerstatuscase, then same-reviewer focusedclosure androotpublication before realrefresh.

I08 (interface; source=root): actualauthorcompletion usesproducer's ownfixedenumeration andCOMPLETION.json, not genericcomplete aliases. No semantic change totraining eligibility; missingQUVA remainsPENDING.

Acceptance: actualproducerstatus+matchingexistingcompletion enablesnewqualifier; originalnonqualification manifests remainbytesstable; zeroarray/science/rawpayloadcalls. ActualRACnetrefresh remainsNOTRUN untilrootreview+publication.
