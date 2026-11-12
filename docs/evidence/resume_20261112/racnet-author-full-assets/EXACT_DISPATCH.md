# Frozen finite dispatch contract

Status: staged, not launched. Root owns the required one bounded operational review and subsequent dispatch authorization. No user approval question is pending.

Server root: `/root/autodl-tmp/TempoRAC`; worker root: `/root/autodl-tmp/TempoRAC/baselines/racnet-author-assets20261112`.

Python: `/root/autodl-tmp/TempoRAC/env/bin/python`; existing NumPy only, no installation. `launch_once.sh` confines the process to CPU26, nice10, all numeric thread counts1, and empty CUDA_VISIBLE_DEVICES. Cache/tmp remain under R. HOME is unchanged.

Frozen repaired CONTROL SHA256: `b7d6e8dea8ce857e172ea3d80f450c50b6b4607b3dfc56051267a9c9f97bc46a`.

Prior reviewed control `03b59ea503b5e16e1da6ff88680dd3fc921df9fc693d4527590182e53d618639` and its code/state/prelaunch packet are preserved in `preserved_pre_repair_r1_r3/`. `RECOVERY_REBIND_RECEIPT.json` records this explicit unstarted R1–R3 rebind. The original negative review remains authoritative history; targeted closure is pending in the same reviewer session.

After the review is saved to a concrete project path and its full SHA is checked, invoke from SSH with actual REVIEW_FILE and REVIEW_SHA256 arguments:

```sh
cd /root/autodl-tmp/TempoRAC/baselines/racnet-author-assets20261112
nohup bash ./launch_once.sh "$REVIEW_FILE" "$REVIEW_SHA256" </dev/null >ACQUISITION_ATTEMPT1.log 2>&1 &
printf '%s\n' "$!"
```

This is one finite process, no scheduler or periodic restart. `flock` enforces one writer. The worker stores its own PID/status and original48h deadline in STATE.json. No automatic resume/retry exists. An explicitly authorized later resume must use the same control and original deadline, `acquire_assets.py --run --resume`, with one CPU and identical limits. Never rerun `--prepare`, delete state, or reset deadline.

Transfer: exactly61 immutable objects totaling128,629,792,634bytes, order RC.00→UCF→RC.01..59. Only128,551,856,160missingbytes are eligible for requests; all77,936,474G1bytes already copied and readback-hash verified. Every response must be exact206/range/length before its body is consumed; redirects only to the four explicit official mirror/storage hosts. All private HTTP headers/URLs remain on server/E, never public Git.

Resources: combined retained asset cap384GiB, volume free reserve128GiB, RC decompression≤220GiB, UCF≤3,341,174,298uncompressedbytes, one archive/NPZ member≤1GiB. No joined125GB archive. Original archives and partial evidence are preserved. Any HTTP/hash/disk/archive/schema fault stops the process without retry.

Before receiving each checkpoint, physical space is allocated at its missing object offset. A terminal failure may flush only these already received bytes after the deadline; it starts no additional read. Request receipts distinguish attempted, received and journal-recoverable durable bytes; failed checkpoint persistence is recorded without repeating the commit. Full expected size/SHA identifies an orphan final object. Extraction progress belongs to each member, including an active RC member across nested UCF extraction.

Asset rule: preserve original keys/arrays; frames OR img_feature is a candidate. Missing literal frames/length is an interface gap, not an absent asset. Applicable RC population requires actual feature payload and author gt_tsm. No aliases or synthesized counts/length/TSMs are written. Only five frozen UCF filename aliases are metadata mappings. Download completion never releases training or claims fair comparison readiness.

Prelaunch tests are bounded server-CPU fixtures and existingG1sample inspection, not model/data-target generation. No GPU, decoding, feature generation, training, new environment, repo/status/source_sync mutation, or process interruption is authorized here.
