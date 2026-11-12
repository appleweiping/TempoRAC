# DeTRC 前置实施交付 — 2026-11-12

本轮实际补齐精确 TSN 骨干、隔离旧 API 环境、原生算子 CPU 编译与最小机械 candidate。没有执行 GPU kernel、视频解码、特征提取、模型 forward、训练或评测；不释放80epoch，也不新增 performance 结果。

| 验收项 | 判定 | 实际证据 |
|---|---|---|
| P1 精确官方TSN权重 | 通过 | ASSET_RECEIPT/BACKBONE_STATE_MAPPING；97,579,098bytes，SHA cc665e2a86857b4ff4873747ae4e98f9cd6de3f72111cf3379954f551cc63c49；318/318 encoder状态严格映射，缺失0、shape错0，仅2个分类头键明确不用 |
| P3 隔离环境与原算子构建 | CPU范围通过；运行时整体部分 | Python3.10.22/Torch2.5.1+cu124只读继承；私有MMAction0.24.1/MMCV1.7.0、nvcc12.4.131/g++11.4；两.so编译和导入，dispatch0；ENV_LOCK/NATIVE_BUILD_RECEIPT。GPU witness、实际batch64未验证；作者原环境仍未知 |
| P4 机械兼容与恢复 | 实现及CPU定点检查通过；独立审查待办 | 18文件diff；namespace/注册/路径、np.bool/np.int、CSV iloc/空GT形状、严格零位置表、ROIAlign梯度返回数；epoch19/warmup31/optimizer/StepLR2e-5/最佳OBO状态及Python+NumPy+Torch RNG恢复；原生strict tie保留较早best，原子保存并保留旧best |
| 全量TSN调用准备 | 静态通过；输入未完成 | 1244条、954550原生帧的UntrimmedSampleFrames均精确1..T；原生JPEG/短256/crop256/interval1/batch200/FP32 T×2048。EXTRACTION_INPUT_PLAN及prepared shell；没有H5或JPEG产出 |
| 冻结科学字段/原源保护 | 通过本轮范围 | 67/67不可变作者文件SHA复核一致；69文件candidate清单；model/optimizer/lr/80epoch/window/evaluation配置一致；无Git/current_status/source_sync写入，未修改共享env/系统CUDA或当前训练源 |
| 完成DeTRC外部全量performance | 未达标 | 本轮0训练、0评价、0新增完成框架/域/SOTA；当前任务仅前置实施 |

精确命令、原始失败及修复见 COMMANDS.md 和所有log；源代码见 candidate/，完整差异见 CANDIDATE_DIFF.patch，MMAction独立NumPy兼容差异见 MMACTION_COMPAT.patch。编译错误曾为缺cusparse.h；只在构建进程加入现有只读NVIDIA header目录，未替换kernel。

A001完整语义假设：作者MMAction版本无可验证tag，使用v0.24.1兼容候选，不能称作者环境精确复原。A004完整语义假设：仅适用于从零重训的默认TSN路线；训练仍传None，位置表必须严格全零，测试修合法取值/repeat，保留作者permute(2,0,1)，不推广非零作者checkpoint或非512×512位置维度。二者均来自已冻结路线，没有新增科学设计选择；当前需Astra定夺的新问题清单为空。

剩余明确未完成：P2真实全量JPEG、UCF inclusive ROI原生帧物化、全部H5及frame/JPEG/feature身份回执；P5完整152+105+100的无标签推断包装和common/native分离输出；P6单独全80合同/资源和fresh实现审查；P8真实FP32 batch64峰值/吞吐。native test配置仍151行，不能当作完整357评测。Ba​​idu任务权重和最强ViT仍未取得；这不阻止正确标注的TSN从零重训，但不能宣称论文ViT重现。

请求Sol执行，实际backend model/effort未独立证明；same-family provisional。SILENT_ASSUMPTION_SWEEP是pending，尚无review verdict，不是PASS。原生算子构建成功不是GPU运行时ready。当前两GPU任务继续，下一次生产提取须等完整输入/资源/合同具备。
