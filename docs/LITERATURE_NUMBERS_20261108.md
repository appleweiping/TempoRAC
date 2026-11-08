# Published RAC results (primary-source check, 2026-11-08)

Collected by a delegated literature pass that read each number from the primary
PDF/HTML table (rendered table images where text extraction was garbled).
Spot-check key rows again before camera-ready. "own" = authors' number;
"X→Y" = trained on X, tested on Y. RepCount-A numbers are on the 152-video test
split; UCFRep on the 105-video val split; Countix on test unless noted.

| Method (venue) | Input | Data | MAE | OBO | Provenance |
|---|---|---|---|---|---|
| RepNet (CVPR20) | RGB | Countix | 0.3641 | 0.697 | own T7 (paper prints OBO error 0.3034) |
| RepNet | RGB | RepCount-A | 0.9950 | 0.0134 | TransRAC T2 reproduction; DeTRC retrain 0.5865/0.2450 |
| RepNet | RGB | UCFRep | 0.474 | 0.371 | Sight&Sound T5 (released ckpt) |
| Context (CVPR20) | RGB | UCFRep | 0.147 | 0.79 | own T2 (encoder fine-tuned) |
| Context | RGB | RepCount-A | 0.8786 | 0.1554 | TransRAC T2 reproduction |
| Sight&Sound (CVPR21) | RGB | UCFRep | 0.143 | 0.800 | own T5 |
| Sight&Sound | RGB+audio | Countix | 0.307 | 0.511 | own T5 (sight-only 0.314/0.498) |
| TransRAC (CVPR22) | RGB | RepCount-A | 0.4431 | 0.2913 | own T2 |
| TransRAC | RGB | UCFRep | 0.6401 | 0.324 | own T3, RepCount→UCFRep only; MFL in-domain re-impl 0.4409/0.4300 |
| MFL (WACV24) | RGB+flow | RepCount-A | 0.3841 | 0.3860 | own T1 |
| MFL | RGB+flow | UCFRep | 0.3879 | 0.5100 | own T2 in-domain; RepCount→UCFRep 0.5227/0.3500 |
| Full-Res (2023) | RGB | RepCount-A | 0.4103 | 0.3267 | own T2 |
| PoseRAC (2023/24) | pose | RepCount-pose | 0.236 | 0.560 | own T3; 487-video 8-class train subset, 151 test |
| ESCounts (ACCV24) | RGB (exemplars train only) | RepCount-A | 0.213 | 0.563 | own T1a, ViT-B |
| ESCounts | RGB | UCFRep | 0.216 | 0.704 | own T1c |
| ESCounts | RGB | Countix | 0.276 | 0.673 | own T1b |
| IVAC-P2L | RGB | RepCount-A | 0.4022 | 0.3444 | own T II (train+val, count augmentation) |
| DeTRC (arXiv 2025) | RGB ViT | RepCount-A | 0.2622 | 0.5430 | own T1 |
| SkimFocusNet (arXiv24) | RGB | RepCount-A | 0.2489 | 0.5166 | own T1 |
| CountLLM (CVPR25) | MLLM | RepCount-A | 0.162 | 0.639 | own T1 (VideoMAE; VSwinT 0.209/0.552); WebVid-10M + Vicuna-7B |
| CountLLM | MLLM | UCFRep | 0.116 | 0.839 | own T1; RepCount→UCFRep 0.139/0.816 |
| CountLLM | MLLM | Countix | 0.215 | 0.801 | own T2 |
| D2-STX (CVPR26F) | RGB+pose | RepCount-A | 0.255 | 0.500 | own T1 |
| D2-STX | RGB+pose | UCFRep | 0.236 | 0.777 | own T1 |
| D2-STX | RGB+pose | Countix | 0.278 | 0.734 | own T1 |
| PAMS (CVPR26F) | skeleton, self-supervised | RepCount-A | 0.129 | 0.842 | own T1; supp T6; earlier in-repo reproduction did not reach it |
| PAMS | skeleton | UCFRep | 0.208 | 0.686 | own T1 |
| HTRM-Net (TMM) | RGB | RepCount-A | 0.3543 | 0.3576 | own T II |
| RACnet (ICIP24) | RGB | RepCount-A | 0.4441 | 0.3933 | own T1 |
| MultiCounter (ECAI24) | RGB | RepCount-A | 0.232 | 0.527 | own T2 (trained on synthetic MultiRep built from RepCount) |
| FCA-RAC (arXiv24) | RGB + test-time first-cycle exemplar | RepCount-A | 0.268 | 0.47 | own T1 |
| GMFL / SPKDB-Net / BIGC-Net | – | RepCount-A | 0.216/0.210/0.215 | 0.560/0.605/0.579 | only as cited in PAMS T1 (not checked in own papers) |

Protocol notes: ESCounts' official loaders drop zero-count videos (test 151 / val 130)
and use |round(p)-g|/(g+0.1); reproducing its released checkpoint with that protocol on
our data gives test 0.2151/0.5563 (paper 0.213/0.563). RACnet reports that TransRAC-code
descendants score against GT-density sums rather than annotated counts. Several tables
mix cross-dataset UCFRep/Countix numbers with in-domain ones (see discrepancy notes in the
session record). Countix subsets differ across papers (download attrition).
