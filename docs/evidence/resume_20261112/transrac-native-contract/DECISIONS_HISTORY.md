# One-time design-choice archive

No implementation or GPU experiment was run while resolving these choices.

1. The earlier availability packet could not find cycle CSVs under data/. Direct inspection now finds the existing official ESCounts copies under third_party/EveryShotCounts/datasets/repcount; full SHA identities and inherited complete count/bound agreement are retained. No new labels are invented.
2. Raw versus NPZ normalization is resolved from actual train.py import: raw RGB [0,1]. The incompatible test.py NPZ default and separate UCF end-aligned helper remain recorded as unused branches. This one raw branch is fixed for the new raw-trained baseline.
3. Ordinary four-way gradient accumulation was rejected before code because native four-GPU DataParallel has local BatchNorm and primary-replica buffer persistence. The frozen replacement implements those exact shard/buffer semantics; it does not claim historical CUDA bitwise parity.
4. A literal legacy package installation was replaced before implementation by the already audited modern compatibility recipe. Any actual new compatibility patch must be tied to a failing executed dependency or operation, not an architecture rewrite.
5. The initial short draft described 4,800 successful updates too strongly. Native FP16 GradScaler may skip updates; final contract specifies 4,800 logical opportunities, logs every skip, and still requires all 200 epochs. No silent extra steps or training truncation.
6. Upstream saves candidates but has no explicit selected-best function. Common positive validation MAE/OBO over fixed candidates 51,54,...198 is declared before execution; native diagnostics remain separate. No test result informed these choices.
Final contract clarification before any implementation/run: actual asset preparation found stu6_44 empty scalar/null with empty boundaries, retained without relabelling; validation frequency fixed to every epoch51..199 and candidate51..198/stride3; exact current temporac_cvpr scorer SHA pinned to avoid retired half-up pams scorer. Earlier timestamped design artifacts remain. Machine RUN_CONTRACT SHA a1e06cb1079c031514db51fff983359cda50b43dcf22461936640ffec213ba1d.


Final source-operation clarification before launch: validation/final FP32 without autocast; training AMP only. Native train/validation target sum converts FloatTensor before round; final test preserves original target dtype before round. Final machine contract SHA 892109aeff36dfe0d96f3203e8f8b08936e19073111ed58ef09a36399a8a6856; preceding machine archive retained. Design stage ends at this handoff.


CORRECTION BEFORE ANY LAUNCH: the preceding statement that native validation/final forward is FP32 was wrong. Executor pointed to models/TransRAC.py:193-195, and designer verified the internal with autocast() encloses every TransferModel forward. Retain original internal autocast in all modes; validation/test outer loop has no extra autocast. Never patch model to force FP32. Archived hash892109aeff36dfe0d96f3203e8f8b08936e19073111ed58ef09a36399a8a6856 is superseded by fe9004083ee68d6ccb67eace10eb16d6c8f4bfae7663e0a8df8b3d2058edc669. No experiment ran the mistaken version. All other scope, budget, native target arithmetic, selection and lane decisions remain unchanged.


RESOURCE-ONLY PRELAUNCH AMENDMENT: root explicitly authorized isolated Python3.10.22 venv with read-only existing Torch2.5.1+cu124/CUDA/torchvision/NumPy instead of waiting hours for duplicate cp311 Torch wheels. New prefix R/envs/transrac-native310 owns all pinned dependency overlays; healthy R/env cannot be changed or uninstalled. Previous fe9004083ee68d6ccb67eace10eb16d6c8f4bfae7663e0a8df8b3d2058edc669 contract and affected files are archived. New contract SHA 1b986ab40ce69185899107aad3b28eba429a5b85fccda6a10585eeaa4deb2d8a. Official source/model/data/internal autocast/optimizer/seed/200epochs/selection/metrics/populations/manual handoff/existing21 are unchanged. Existing CPU primitive results used the same Python3.10/Torch per root; no new generic review or tests requested. Designer made only local E documentation changes and did not install or launch anything.

