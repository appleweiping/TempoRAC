# Official external baseline readiness — 2026-11-11

Scope: bounded official-source/asset acquisition and static adapter preparation. No environment installation, training, model forward, feature extraction or evaluation was performed. No Git, paper or shared status changes.

## Acceptance evidence

| Requirement | Status | Evidence |
|---|---|---|
| Distinguish reported / released / official-training / internal-style | Pass | Matrix below; internal-style implementations never count as external reproductions |
| Current primary sources and immutable versions | Pass within bounded scope | Official GitHub/author/CVF/arXiv links below; negative availability findings mean not verified, not proof of absence |
| At least two credible next executable comparators | Partial | RepNet full source+weights and reviewed-before-run adapter staged; TransRAC official-training route has source but task weight unavailable; PoseRAC native evaluator is oracle-assisted |
| Acquire assets without duplicate large downloads | Pass | RepNet new309.204MB checkpoint; reused existing PoseRAC information; no other large assets downloaded |
| Exact provenance and SHA | Pass | acquisition_receipt.json; source notebook and asset files retained |
| Run full external evaluation | Not run | Root requires fresh Sol review then GPU1 only after full_s1 training+fixed evaluation finishes |
| Six datasets × seven frameworks / SOTA | Not achieved by this task | No new performance claims or numbers |

## Provenance matrix

| Method | Authors' reported result | Released executable / checkpoint | Own official training path | Current classification |
|---|---|---|---|---|
| ESCounts ACCV24 | RepCount .213/.563; UCF .216/.704 | Official source f18fcf1 + task weights already server; inherited evaluation exists | Official exemplar_counting_train.py | Existing official released-checkpoint evidence; not newly evaluated here |
| RepNet CVPR20, updated2025 release | Original papers and TransRAC retrained values are different protocols | Official notebook commit60d48fc... + complete ckpt-70 acquired | Notebook does not release matching full training recipe | Highest ready new checkpoint candidate; updated2025 identity retained; full357 adapter pending review/runtime |
| TransRAC CVPR22 | RepCount .4431/.2913; cross UCF .6401/.324 | Source68bdd4d already server; task weight behind author OneDrive/Baidu; previously acquired127MB file is only Swin backbone and currently not in server pretrained folder | train.py:200 epochs,lr8e-6,batch32,64frames,scales1/4/8; raw RGB loader | Credible official-code full-training route; no task checkpoint acquired; legacy runtime needed |
| DeTRC IJCV26 | Published v4 ViT result .2622/.5430 | Official source538abfcd acquired; author task checkpoint Baidu link inaccessible through this transport | tools/train.py and DeTRC/config/repcount_tsn_feature_enc_contrastive.py | Not execution-ready; release/source/backbone variant must not be conflated with ViT paper number |
| PoseRAC arXiv23 | RepCount-pose .236/.560 | Source469590b... +11MB task weight already server | pre_train.py/train.py,20epochs,33 MediaPipe landmarks xyz | Native eval.py79–130 selects action channel minimizing GT count error. Oracle-assisted author protocol only; oracle-free inherited variant must be named separately |
| RACnet ICIP24 | .4441/.3933 RepCount; .5260/.3714 UCF; .5278/.3924 Countix | Source51b8e3f... acquired, CC-BY-NC4.0; GoogleDrive task weight transport failed | train.py configs/train_RACnet.yaml, official Swin features and refTSM | Credible next official-source candidate once task weights obtained or full official training approved; not a CVPR main-conference row |
| CountLLM CVPR25 | VideoMAE .162/.639 RepCount;.116/.839 UCF;.215/.801 Countix | No verified official repo/weights in primary paper, bounded exact search or prior receipts | Paper stages: WebVid10M10epochs, RAC50epochs, instruction50epochs; Vicuna7B,8A100 | Authors-reported only; internal CountLLM-lite is not official |
| MFL WACV24 | .3841/.3860 RepCount;.3879/.5100 UCF | Author publication page currently gives Paper/Supp, no Code | RGB+flow, separate reconstruction phase | Authors-reported only; no official executable established |
| PAMS CVPR26 Findings | .129/.842 RepCount;.208/.686 UCF | No verified author repo/weights from primary-paper/prior bounded search | Self-supervised pose training details are insufficient for official artifact identity | Authors-reported only; historical in-project PAMS routes are not promoted to external reproduction |

All numbers above are author or inherited literature records, not results of this acquisition. Keep original table-specific modalities, split, train-domain and metric definitions. Literature source: existing docs/LITERATURE_NUMBERS_20261108.md and existing benchmark_scout.json; no claim all methods share one metric.

## Exact next startup/data requirements

### RepNet
Staged at /root/autodl-tmp/TempoRAC/external-baselines-20261111/repnet-runner.
Official architecture, get_score, get_counts, read_video and checkpoint mapping are copied literally from the pinned notebook; no internal approximation/head. Thin adapter adds complete restore checks, full dataset iteration and scoring.
- Fixed populations: UNION rc_test152, UNION ucf_test105, QUVA test100 (357 total).
- All RGB frames decoded sequentially in memory; no full-frame disk cache. Default official 224x224 decoder→112x112 model,64frame windows,strides1/2/3/4,batch20,threshold.2,within.5,median filter true.
- Preserve stu4_5 and stu4_27 disputed RepCount media and all labels/zero-count row. Record ambiguity per row, not silently drop.
- Model receives frames only. Scorer is a frozen exact copy of server temporac_cvpr/metrics.py: np.round ties-even, positive +.1 primary, positive /g, all +.1, zero-count separate. Raw continuous outputs and derived continuous metrics remain distinct.
- Proposed env /root/autodl-tmp/TempoRAC/envs/repnet: Python3.10, tensorflow[and-cuda]2.16.1/Keras3.3.3, numpy1.26.4, scipy1.11.4,opencv-headless4.10.0.84. Not installed or validated. Host CUDA13/cuDNN9 does not supply TF2.16 CUDA12/cuDNN8 ABI; dedicated dependencies required.
- GPU1 only AFTER full_s1 and its fixed full evaluation finish. Proposed8GiB cap, fresh10GiB-free check; do not interrupt training or trigger its resource guard.
- After fresh review and environment witness:
  envs/repnet/bin/python external-baselines-20261111/repnet-runner/evaluate_repnet.py --config external-baselines-20261111/repnet-runner/evaluation_config.json --output external-baselines-20261111/runs/repnet-full357-v1 --mode evaluate
- Preflight/witness modes exist but have NOT run. No auto restart. Corrected v2 supports explicit --resume and frozen UCF inclusive crops.

### TransRAC official training
Existing source /root/autodl-tmp/TempoRAC/third_party/TransRAC at68bdd4d.
Legacy requirements: Torch1.7/MMCV-full1.3.16, MMAction, apex recommended; cannot declare that legacy stack works on current4090/CUDA13 without a dedicated reviewed environment.
Training command is python train.py after audited path/GPU configuration; full train758/valid131, video+CSV identity freeze first.200epochs,batch32,lr8e-6,NUM_FRAME64,SCALES[1,4,8]. Task weight is absent, so python test.py with lastckpt=None is NOT a valid trained baseline. Backbone source link is in official README and prior asset receipt.

### RACnet official route
Source archive extracted to code/RACnet-51b8e3f.
Correct command is python test.py configs/test_RACnet.yaml (README mistakenly says .py). Config requires root/test featureNPZ/test.csv/task checkpoint. Official feature dimensions7x7x768; pretrained Swin encoder needed for new RGB, and referenceTSM targets for training. test4video.py supports direct RGB but hardcodes input paths and needs MMCV/MMAction not listed in its basic requirements.
Native test scoring derives RepCount GT via target-start peak counting and uses +.1; save raw predictions and use frozen annotated-count scorer for harmonized comparison. Weight GoogleDrive entry is public author-linked but both drive.google.com and drive.usercontent endpoints failed on current server. Source acquisition succeeded via one codeload fallback after a bounded GitHTTP2 failure.

### DeTRC blockers that must be fixed explicitly
README uses configs/ but actual directory config/. Data guide filename is prepare_dataset_DetTRC.md. setup.py expects mmaction/version.py, absent from repo; repcount_dataset.py imports DetRC but package is DeTRC. Feature config says TSN,input2048 while guide downloads VideoMAE checkpoint but passes TSN filename. Resolve exact encoder/variant and required upstream MMAction version; no silent manual replacement that is later called official. Baidu checkpoint unavailable in this transport, no weight SHA possible.

## Primary-source links

- RepNet [official notebook](https://github.com/google-research/google-research/blob/60d48fc1b71b6126462136cad441dd76da7460d0/repnet/repnet_colab.ipynb)
- TransRAC [official source](https://github.com/SvipRepetitionCounting/TransRAC/tree/68bdd4daa60ed7c3174a7f6bf86f6537b6fa0979)
- DeTRC [author project](https://shirleymaxx.github.io/DeTRC/), [official repo](https://github.com/lizishi/DeTRC/tree/538abfcd24e0a81b383ca81862ff4360a2389c97), [paper v4](https://arxiv.org/html/2403.01543v4)
- PoseRAC [official source](https://github.com/MiracleDance/PoseRAC/tree/469590b611bde3595eaf163b517263da634e2096)
- RACnet [official source](https://github.com/Luoadore/RACnet/tree/51b8e3f2decdeb10dbc7fd506a93a9f7d2d4d4fb)
- CountLLM [primary paper](https://arxiv.org/html/2503.17690v1)
- MFL [author publication page](https://zhengqixiaoli.github.io/), [CVF paper](https://openaccess.thecvf.com/content/WACV2024/html/Li_Repetitive_Action_Counting_With_Motion_Feature_Learning_WACV_2024_paper.html)
- PAMS [primary paper](https://openaccess.thecvf.com/content/CVPR2026F/papers/Gao_Count_What_Repeats_Period-Adaptive_Multi-Scale_Consistency_for_Self-Supervised_Repetitive_Action_CVPRF_2026_paper.pdf)

Requested executor Sol; actual backend identity not independently attested. No positive review or scientific acceptance claimed.

