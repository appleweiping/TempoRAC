# DeTRC native integration — frozen prelaunch packet

Source: scripts/external/detrc_native. Fixed server root: /root/autodl-tmp/TempoRAC/baselines/detrc_native_v1. Reuses prior P1/P3/P4 environment/backbone/native operators. No installation, compilation, GPU probe/forward or production is performed by preparation.

Stages: prepare -> rawframes (CPU only) -> extract (allocated GPU) -> train80 (allocated GPU) -> evaluate (one selected weight, RC131 diagnostic plus full357) -> separate CPU scorer. Root supplies a reviewed release JSON whose contract_sha256, stages and code dictionary exactly match RUN_CONTRACT/SAFE_MANIFEST. Raw labels and complete input plans stay private on the server and E: evidence directory.

CPU production entry, after root release:

    bash /root/autodl-tmp/TempoRAC/baselines/detrc_native_v1/driver/launch_cpu.sh /root/autodl-tmp/TempoRAC/baselines/detrc_native_v1/ROOT_RELEASE.json

The launcher sets CUDA_VISIBLE_DEVICES empty, first2 allowed CPUs, one worker, nice10, ionice2/7 and numerical libraries1 thread. It produces exact official short256/default-JPEG frames, full1244 (required1243) inputs and inclusive UCF ROI, one video at a time. Completed videos are atomic and reusable; incomplete video replays. Every frame has source index/size/SHA. It retains complete JPEGs and stops if64GiB derived cap or the free-space reserve is reached, without thinning frames or silently deleting inputs.

Free-space gate:128GiB PLUS pending RACnet author bytes, with total128629792634 minus STATE durable coverage (not sparse logical size or network traffic); refresh within60seconds. The CPU stage needs no free GPU. Original67 source files and prior packet remain immutable.

GPU stages require full-native features and root release, an actually idle exclusive24GiB-class GPU with >=24000MiB visible, <500MiB occupied and no compute process, >=32GiB host RAM available. Allocated phase wall budgets are16/30/2h, total48h, persisted across resumes; CPU JPEG time is separate. Phase controllers terminate only their own process group at timeout/failure. Native training remains80epochs/batch64/seed0/FP32/AdamW and native130 validation-OBO strict selection; no AMP, microbatch, operator or window replacement. GPU resource monitoring checks dynamic artifacts every5seconds; original checkpoint writes remain native atomic IO and can finish within that interval, so this packet is not a proof of exact peak byte usage or batch64 memory.

Prepared extraction changes only official extractor resume/atomic pickle IO. Numerical transforms, interval1, crop256, FP32 frame2048 and internal batch200 remain unchanged. Full H5 membership/dtype/value identities are required before train can start. Inference consumes a distinct label-free input file, keeps fixed original64-row batches across resume, passes only empty(0,3) structural GT, and uses original test windows/masks and SoftNMS sigma.1/top30/floor1e-4/count>.2. Labels enter only the separate scorer; native RC151/130 diagnostics and CSV scalar common metrics remain distinct, including RC51scalar/41cycles counterexample.

New-interface CPU checks passed once on the server: full1244 metadata/ROI/window structures,488 label-free inputs (131+357), original JPEG byte equality/ROI primitive, atomic video reuse, original SoftNMS top30 primitive. No real dataset video was decoded; only three in-memory fixture frames were encoded. Latest parent instruction added only RACnet reservation/resource wrapper and idempotent preparation; CPU_RESOURCE_DELTA_CHECK records syntax/current capacity separately instead of rerunning those passed primitives.

Publication allowlist: the11 .py/.sh files listed by PUBLIC_ALLOWLIST.json plus this README and optional SAFE_MANIFEST.json. Do not publish complete plans, CSV/labels, rawframes, feature files, checkpoints, environments or .so files. Do not launch train while JPEG/features are absent. Fresh Sol prelaunch review and actual production remain root-owned.
