# Next official external baselines — bounded availability check

2026-11-12 continuation. **No new task checkpoint was acquired.** TransRAC and RACnet are credible official-code full-training candidates; neither is currently ready for released-checkpoint evaluation. This check does not release training, feature extraction, environment installation or GPU work. RepNet/ESCounts results remain unchanged.

| Acceptance | Outcome | Evidence |
|---|---|---|
| Reuse inherited provenance, check only2 next candidates | Pass | TransRAC68bdd4d and RACnet51b8e3f remain current via official GitHub connector; inherited DeTRC blockers retained without a new acquisition campaign |
| Official task-weight bytes and SHA | Unmet for both | No checkpoint acquired; SHA=null, not substituted with backbone or internal-style weights |
| Concrete official training routes, inputs and costs | Pass for planning | Native configs/source below; wall time and4090 compatibility remain unmeasured |
| New bounded transport evidence | Pass | Server probe receipt; author-linked HF feature metadata reachable through IPv4 hf-mirror; Drive alternative unavailable, SharePoint API401; no automatic retry loop |
| CountLLM unresolved release gap | Still unresolved | Official paper v2, exact GitHub repository search and bounded web search yield no verified author release; no implementation invented |
| No scientific/environment/source changes | Pass | Static server CPU/metadata inspection only; no dependency installation, model import/forward, training, decoding, metric recomputation or feature build |

## 1. TransRAC — strongest venue fit, task checkpoint still blocked

Official source: [SvipRepetitionCounting/TransRAC](https://github.com/SvipRepetitionCounting/TransRAC/tree/68bdd4daa60ed7c3174a7f6bf86f6537b6fa0979), CVPR2022 Oral. Latest commit remains `68bdd4daa60ed7c3174a7f6bf86f6537b6fa0979`; no GitHub releases. Existing server code `/root/autodl-tmp/TempoRAC/third_party/TransRAC` reused. LICENSE contains Apache2.0 plus appended Anti996 conditions; retain the actual file, rather than claiming an Apache-only release.

The [official README](https://github.com/SvipRepetitionCounting/TransRAC/blob/68bdd4daa60ed7c3174a7f6bf86f6537b6fa0979/README.md) links a password-protected OneDrive folder and Baidu disk. A new bounded SharePoint REST shares→children request returned401; no auth bypass or author contact was attempted. [Issue42](https://github.com/SvipRepetitionCounting/TransRAC/issues/42) identifies the task file `transrac_ckpt_pytorch_171.pt`; this is evidence of an earlier released filename, not proof that we possess its bytes. `test.py` still sets `lastckpt=None`; running it unchanged would not establish a trained baseline. Task-weight SHA remains **null**.

The separate official Swin-T K400 backbone [release asset39172053](https://github.com/SwinTransformer/storage/releases/tag/v1.0.4) is127,414,861bytes; API digest is null. It is not the TransRAC task checkpoint and is absent from the server pretrained folder and bounded existing-project search. No repeat backbone download was done in this check.

Full official training entry: `python train.py`. Native defaults:200epochs, global batch32, lr8e-6,64frames, scales[1,4,8], frozen backbone inference (`OPEN=False`),4GPU hardcoded. Original RepCount-A nominal758train/131valid entails about4,800optimizer steps and repeated multiscale Swin inference; this is a step estimate, not a measured GPU-hour quote. The2GPU server would need an explicit resource-only binding while preserving global batch. Validation starts after epoch50; checkpoints include validation MAE. An eventual contract must freeze validation-only selection and the existing known-source qualifications rather than silently training on a different population.

Inputs: original fine-grained cycle CSV columns name/count/L*, raw RGB or official imgs/fps NPZ. Original videos are already present, but original RepCount CSV paths were not located by a bounded data-tree search. Native raw loader samples64frames proportionally, repeats the last frame when short, resizes224, and ToTensor gives[0,1]. Native NPZ loader subtracts127.5/divides127.5 to[-1,1]. **These paths are not numerically interchangeable.** The released-checkpoint normalization/loader contract requires resolution before execution. UCF inclusive annotated ROI and QUVA fullvideo require a thin, frozen input adapter; unchanged per-method native outputs plus harmonized full-population scoring must be retained.

Environment stated by [install.md](https://github.com/SvipRepetitionCounting/TransRAC/blob/68bdd4daa60ed7c3174a7f6bf86f6537b6fa0979/install.md): Torch1.7, MMCV-full1.3.16, CUDA11.4, MMAction and recommended apex; requirements additionally list torch1.7+cu110 and both mmcv1.4.8/mmcv_full1.3.16. This legacy stack is not proven usable on4090. No silent upgrade or approximation is authorized here.

**Hard stopper:** no task checkpoint bytes; official retraining instead requires the explicit coherent environment/input-normalization/annotation contract. Conditional route is genuine whole official-method training, not a manual head on our features.

## 2. RACnet — modern training stack, official features reachable but large

Official source: [Luoadore/RACnet](https://github.com/Luoadore/RACnet/tree/51b8e3f2decdeb10dbc7fd506a93a9f7d2d4d4fb), ICIP2024 Oral (not a CVPR main-conference method). Latest commit remains `51b8e3f2decdeb10dbc7fd506a93a9f7d2d4d4fb`; no GitHub releases. Reuse `/root/autodl-tmp/TempoRAC/external-baselines-20261111/code/RACnet-51b8e3f`. CC-BY-NC4.0 declaration retained; bundled LICENSE is an abbreviated text linking the full legalcode.

Author [README](https://github.com/Luoadore/RACnet/blob/51b8e3f2decdeb10dbc7fd506a93a9f7d2d4d4fb/README.md) ties both task weights and feature release to this implementation. Task weight is Drive file `1cmWsc7vomugrh3idvHOrkQTbujrAMZPE`; earlier drive/usercontent transport failures were not repeated identically. New docs.google alternative failed, and a bounded IPv4 version also timed out. Task-weight SHA remains **null**. Author HF model-list endpoint reports no model repositories; the feature repo has no .pt/.pth/.ckpt/.safetensors task file.

New concrete available asset: [author-linked luoadore/RACnet_feature_npy](https://huggingface.co/datasets/luoadore/RACnet_feature_npy), public/ungated metadata, immutable repo commit `2196a15bb50de973f245cff318acd1cbe997bdc8`. Direct HF server requests failed; `curl -4` to hf-mirror succeeded with HTTP200. This is an alternative public transport for the same author namespace, not an independent source of authority. Metadata file SHA `bbd0c81fd586932e9711fded2af0520329484eeab0f5f8e8f87c84914682da72`. No bulk arrays were downloaded.

| Released feature object | Compressed bytes | Availability / coverage qualification |
|---|---:|---|
| RepCountA split archive parts |125,484,960,235 across60parts| All train/valid/test content reported by README; actual per-split video counts unverified until archive inventory; no annotation CSV bundled according to README |
| UCFRep_feature_npz.zip |3,144,832,399| Test features per README; exact105/frozen ROI alignment unverified |
| Countix_feature_npz.tar.gz |42,463,087,479| Author explicitly describes an available-video subset; not automatically our frozen population |

All60part and archive LFS object SHAs are preserved in `racnet_hfmirror.json`; e.g. first RepCount part `9cc3e77b0e76d0cc4165b7f434c563343c3429e460bf6cd89f50ecb480df1e78`, UCF/Countix hashes remain in the full machine-readable tree. Current server free652GiB snapshot is capacity evidence only, not approval for125.5GB transfer/decompression.

Full training entry is **`python train.py configs/train_RACnet.yaml`**; testing **`python test.py configs/test_RACnet.yaml`** (README .py suffix is inconsistent with actual YAML files). Native defaults:200epochs, batch16, Adam lr6.4e-5, MultiStepLR40epoch decay.9,4stages/10layers/64maps, gt_mode=start, tsm_error=True,lambda4tsm1e-5,H-norm,no feature normalization,seed2333. Original nominal758train requires about9,600optimizer steps, excluding feature acquisition/extraction. Original full-length features and per-frame square TSMs dominate memory; no memory or wall-time measurement was performed. Model selection saves both bestvalidationMAE and bestvalidationOBO; primary-selection rule must be frozen before training.

Inputs: official Swin feature maps flattened7×7×768 (37,632values/frame), `img_feature`,length and training `gt_tsm`; original cycle CSV. Our existing STMAE features are **not compatible substitutes**. Either acquire and identity-check the author arrays or run the literal official RGB extractor (Swin backbone/MMCV/MMAction additionally required); both are future work. QUVA has no released feature archive in this repo; full357 would need the native RGB path or exact official feature extraction for100QUVA and any mismatched UCF inputs. No Countix expansion is requested here.

Environment README Python3.11.5; requirements pin Torch2.3.1/torchvision0.18.1,scipy1.14.0 and other packages. Static code additionally imports YAML CLoader but PyYAML is absent from requirements. Native RGB test4video.py additionally imports legacy MMCV/MMAction not covered by basic feature-only requirements. No new env was installed or declared ready.

Native RepCount test GT comes from peak counting annotation-derived target_start, not directly the CSV scalar; harmonized annotated-count scoring must be saved separately. Native thresholds/prominence and official count conversion must not be replaced by a custom head. Fullvideo arrays and known-source training overlap still require an explicit frozen protocol; author feature availability is not proof of leakage-free training.

**Hard stopper:** no task checkpoint; the viable native-training route needs125.5GB feature acquisition or faithful official extraction plus fine-grained annotations and fixed input/selection contract. It is a conditional second official comparator, not a silent replacement for an unavailable fixed-pool method.

## CountLLM and deferred third option

[CountLLM primary v2](https://arxiv.org/html/2503.17690v2) remains no verified runnable release in this bounded gapfill. GitHub connector exact repository query `CountLLM` returned0; primary-paper and web results did not establish author code/weights. A search-engine related-paper snippet pointing to Darcyddx/Video-LLM is not author linkage and was not promoted. No new CountLLM source exists in this evidence set; retain authors-reported classification and inherited8A100/three-stage cost, not an internal reconstruction.

DeTRC current commit remains538abfcd24e0a81b383ca81862ff4360a2389c97. Its inherited Baidu checkpoint, TSN/VideoMAE identity and package-name/config-path blockers remain. No new DeTRC download/install attempt was spent. PoseRAC's native GT-count oracle remains excluded from a common fair scorer.

## Machine evidence and action boundary

`PROVENANCE.json` records candidate pins, task-weight nulls, primary URLs and decision boundary. `summary.json` carries exact current source-file hashes, public feature repo revision and sizes; `racnet_hfmirror.json` retains all LFS SHAs; `probe_receipt.json` retains bounded network outcomes. These files are mirrored under server `/root/autodl-tmp/TempoRAC/external-baselines-20261112/next-availability` where applicable. The verified project root remains `/root/autodl-tmp/TempoRAC`.

Requested executor Sol; actual backend identity not independently attested. This is availability/planning evidence only; no new performance result, environment readiness, full-paper reread or scientific acceptance is claimed. Root/Astra chooses any separately frozen next group. Shared repo/current_status/source_sync and completed RepNet assets/results were untouched.
