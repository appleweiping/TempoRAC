# RepNet assumption ledger

ASK=never. These choices precede adapter code. Latest root instructions override generic skill smoke requirements.

| ID | Gap | Chosen | Class | Source |
|---|---|---|---|---|
| A001 | Which RepNet checkpoint | Author-released ckpt-70 updated May2025; distinct from original CVPR2020 reported result and retrained TransRAC RepNet | semantic | root |
| A002 | Population | 152 rc_test +105 ucf_test from UNION;100 QUVA test. Keep zero and both disputed RepCount test videos; labels unchanged | semantic | root |
| A003 | Inference settings | Notebook defaults: strides1,2,3,4; batch20; threshold.2; within.5; median filter true; constant-speed/fully-periodic false | semantic | default official defaults |
| A004 | Native metric | Official notebook has no benchmark scorer: preserve its continuous sum and report derived continuous MAE/OBO separately, do not name these official benchmark reproduction | semantic | default |
| A005 | Harmonized metrics | Exact checked copy of server count_metrics (np.round ties-even, positive +.1, all +.1, positive /g, zero separate) | semantic | root |
| A006 | Input decoder | Exact notebook read_video: sequential OpenCV BGR-to-RGB, resize224x224 all decoded frames; fail on unreadable/empty/truncated metadata disagreement rather than skip | semantic | default official core |
| A007 | Checkpoint mapping | Preserve literal Keras3 core and official positional mapping keys; require complete count/shape and exact post-assignment equality instead of silent zip truncation | interface | review R1 |
| A008 | Runtime | Dedicated server envs/repnet; Python3.10 TensorFlow2.16.1/Keras3.3.3/numpy1.26.4; exact-version primary PyPI metadata verifies Python3.10 support; no existing env mutation | interface | review R1 |
| A009 | Memory | Model batch20 and full RGB frames; process one video at a time; no frame caches written to disk; device cap initially8GiB, fresh resource check must pass | interface | default |
| A010 | Reuse | Explicit --resume reuses only atomic per-video JSON+trace with exact frozen input/config/source receipts; no retraining or silent skip of failed rows | interface | root |
| A011 | Restored state | get_repnet_model original constructs with random input then expect_partial; immediate complete explicit mapping, all variables checked before any data forward | interface | default |
| A012 | Full evaluation freeze | Abort on failure with failure.json and completed per-row records; no success marker until all357 predictions and finite counts obtained | semantic | default |
| A013 | UCF temporal input | Apply frozen zero-based inclusive frame_range with end clamp after exact author decoder; record full/requested/effective lengths and require existing feature metadata agreement | semantic | review R2/R3/root |
| A014 | Decode completeness | Require all357 OpenCV full decoded lengths equal existing qualified Decord len_reader/frames_full_video metadata; fail on discrepancy before that video's inference | semantic | review R4 |
| A015 | GPU authorization | GPU1 only, after full_s1/complete_known_source.json has50epochs and fixed eval populations, matching worker387997 queue_complete, worker PID absent and no other GPU1 compute owners | interface | root |

