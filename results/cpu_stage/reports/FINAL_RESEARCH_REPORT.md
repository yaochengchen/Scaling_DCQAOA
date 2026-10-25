# Training-Free Explicit CD-QAOA 最终研究报告

> 日期：2026-10-11（Asia/Shanghai）  
> 自主执行起点：21:03:26；初次完整封包：21:37:15；墙钟 33 分 49 秒，未触达 10 小时上限。  
> 定位：CPU 中小规模、可复现、允许负结论的普通 SCI 论文证据闭环。GPU 扩展按用户要求留给合作者。

## 1. 执行摘要

本轮完成了从物理正确性、离线多初值训练、参数集中与尺度规律、冻结公式、独立 OOD 测试、双比特门预算审计，到图表、论文草稿和复现材料的完整 CPU 闭环。核心结论不是“CD 在所有条件下更好”，而是一个边界清楚的结果：**固定参数显式 CD-QAOA 在相同层数下显著优于固定参数 QAOA，但在本项目测试的 n≤20、p=1/2 范围内，优势不足以抵偿额外的双比特门成本。**

主确认实验含 81 个训练、54 个验证、81 个测试实例，1,458 条离线优化记录、305,737 次离线目标函数调用和 162 个独立测试实例-深度配对。相同 p 时 CD 的平均归一化配对增益为 **+0.079818**，95% CI **[+0.071876,+0.087575]**，胜率 **96.30%**；等未优化 Pauli 分解 CX 预算（CD p 对 QAOA 3p）时平均增益变为 **−0.085661**，95% CI **[−0.092810,−0.078986]**，胜率 **1/162=0.62%**。两个预先限定的诊断都没有逆转负结论。

## 2. P0–P5 完成情况

| 阶段 | 本轮完成工作 | 证据文件 | 评价 |
|---|---|---|---|
| P0 物理与基线 | 冻结 Hamiltonian、little-endian 位序、门序、CSV 解析、Qulacs/Qiskit 符号；dense expm、Qulacs、Qiskit 三重交叉验证；alpha=0 退化；尺度协变；Qiskit CX 编译 | `tests/`, `reports/BACKENDS.md`, `results/codex_pilot_recheck/` | 基本完成 |
| P1 离线发现 | n=8/10/12、p=1/2，3 restarts；保留所有重启、能量、eval 数与收敛状态；归一化集中分析 | `results/codex_cpu10h_confirmatory_locked/offline_optima.json`, `concentration_summary.json` | 主要完成；缺 p=4 和严格等价分支对齐 |
| P2 公式和参数库 | 比较 CSV/RMS/训练 medoid、BCH、CD reference、degree-corrected proposed；验证集选择；冻结 CSV/JSON 和模型 | `FROZEN_MODEL.json`, `assets/cd_transfer_table_v1.{csv,json}`, `assets/CD_TRANSFER_V1_FROZEN_MODEL.json` | 完成首版可用公式 |
| P3 独立验证 | 81 个 n=12/14/16 OOD 测试；162 配对；分层 CI；256 shots；Qiskit level 0/1/3；n=18/20 额外诊断 | 主结果目录及两个 `analysis/` 目录 | 主要完成；缺具体硬件拓扑/噪声及 p≥4 |
| P4 GPU 扩展 | 未执行，符合用户“暂时不用开展”要求 | 本报告第 10 节交接清单 | 延后 |
| P5 论文与封包 | 6 张论文图、4 张表、英文论文草稿、中文总报告、复现报告、打包脚本和 ZIP | `paper/`, `reports/`, `dist/` | 本轮完成 |

## 3. 实际实验、规模、参数与运行时间

### 3.1 本地计算资源

- Windows 11 `10.0.26200`，Lenovo 83DG。
- Intel Core i7-14650HX，16 核/24 线程，34,053,414,912 bytes RAM。
- Python 3.13.9；NumPy 2.3.5；SciPy 1.16.3；NetworkX 3.5；Matplotlib 3.10.6。
- Qiskit 2.0.0、Aer 0.17.0、Qulacs 0.6.14。
- 机器有 RTX 4060 Laptop GPU，但本轮**没有使用 GPU**。

### 3.2 P0 复验

1. `python -m pytest -q`：最终 **12 passed**。
2. `scripts/08_check_scale_law.py`：s={0.25,0.5,1,2,4} 的最大态矢误差均为 **0**。
3. `scripts/00_validate_and_benchmark_backends.py --sizes 6 8 10 --repetitions 5`：
   - Qulacs 对 NumPy 最大误差 `1.55e-16`；
   - Qiskit 对 NumPy 最大误差 `1.13e-15`；
   - n=10 实测吞吐：NumPy 749.82 eval/s、Qulacs 1288.76 eval/s、Qiskit 89.03 eval/s。
4. 新目录 pilot：8/8/4 个 train/val/test，48 个优化记录，4 个测试配对；等预算 CD 平均增益 −0.120813，CI [−0.166475,−0.075150]。

### 3.3 第一套锁定小确认实验

- 配置：`configs/codex_fallback_small.json`。
- 规模：24 train / 16 val / 36 test；CD p=1/2；2 restarts；maxiter=26；288 个优化记录；72 个测试配对。
- OOD 测试：n=10/12/14，regular3/ER3，bimodal/lognormal。
- 文件时间跨度：**28.85 s**。
- 结果：等分解 CX 预算平均增益 **−0.081182**，95% CI **[−0.087694,−0.074635]**，0/72 胜；同深度增益 **+0.094317**，CI **[+0.084739,+0.104473]**，72/72 胜。
- 路径：`results/codex_fallback_confirmatory_locked/`。

### 3.4 主确认实验

- 配置：`configs/codex_cpu10h.json`，seed 20261018；manifest SHA256 `b9addf9e543d3ca0ea491dc416a3c284cdc219d3f2da58fe628842b389bf2fd0`。
- 训练：n=8/10/12；regular3、ER3、ring；unit、uniform、signed_normal；每 cell 3 个，共 81 个实例。
- 验证：相同 cell、独立 seed；每 cell 2 个，共 54 个实例。
- 测试：n=12/14/16；regular3、ER3、ER6；bimodal、lognormal、heavy_tail；每 cell 3 个，共 81 个实例。测试权重族与训练完全分离，ER6 还构成度数 OOD。
- 深度：CD p=1/2；QAOA 离线训练 p=1/2/3/6；3 restarts；L-BFGS-B maxiter=45。
- 训练记录：1,458；目标函数调用 305,737；optimizer success 1,310/1,458=89.85%。
- 测试：162 个实例-深度配对；每个固定协议额外 256 shots。
- 完整流水线文件时间跨度：**328.37 s**。
- 路径：`results/codex_cpu10h_confirmatory_locked/`。

### 3.5 两轮限定诊断

1. **实际编译预算敏感性**：主比较在 Qiskit optimization_level 0/1 的 CX 完全相等；level 3 对 CD 有额外约消，因此在新冻结诊断中改为 CD p 对 QAOA 2p，QAOA 方法仍只在验证集选择。162 配对平均增益 −0.032450，CI [−0.039210,−0.026334]，胜率 17.28%。路径：`results/codex_cpu10h_confirmatory_locked/analysis/compiled_level3_sensitivity/`。
2. **跨规模诊断**：配置 `configs/codex_crossscale_diagnostic_round2.json`；n=18/20，regular3/ER6，bimodal/lognormal/heavy_tail，12 个测试实例、24 配对；完整流水线 643.74 s。平均增益 −0.093541，CI [−0.117415,−0.074551]，0/24 胜。路径：`results/codex_crossscale_diagnostic_round2/`。

第二轮完成后已按预案停止方法和测试集调整。

## 4. 定量结果

### 4.1 主结果表

所有增益定义为 `(E_QAOA − E_candidate)/sum|J|`，正值表示 candidate 更好。

| 比较 | n | 平均增益 | 中位数 | 95% bootstrap CI | candidate 胜率 |
|---|---:|---:|---:|---:|---:|
| fixed CD-proposed vs fixed QAOA，同 p | 162 | +0.079818 | +0.072173 | [+0.071876,+0.087575] | 96.30% |
| fixed CD-proposed vs fixed QAOA，3p 等分解 CX | 162 | −0.085661 | −0.074202 | [−0.092810,−0.078986] | 0.62% |
| level-3 编译重配：CD p vs QAOA 2p | 162 | −0.032450 | −0.028510 | [−0.039210,−0.026334] | 17.28% |
| n=18/20 跨规模诊断，3p | 24 | −0.093541 | −0.072035 | [−0.117415,−0.074551] | 0% |
| offline optimized CD vs optimized QAOA，同 p | 162 | +0.086289 | +0.091164 | [+0.063437,+0.109790] | 88.89% |

原始机器可读统计：`results/codex_cpu10h_confirmatory_locked/analysis/final_statistics.json`。

### 4.2 失败区域

- 主确认集中所有报告子组的等预算平均增益都为负。
- family：ER3 最差，平均 −0.102466；regular3 −0.089049；ER6 −0.065470。
- weight：heavy_tail 最差，−0.097633；bimodal −0.080757；lognormal −0.078594。
- depth：p=2 为 −0.096847，p=1 为 −0.074476。
- size：n=16 为 −0.092949，未出现随规模增大接近正值的迹象。
- n=18/20 诊断中 heavy_tail 进一步降到 −0.134215。

完整分组 CSV：`results/codex_cpu10h_confirmatory_locked/analysis/subgroup_results.csv`。

### 4.3 有限 shots 与精确基态间隙

- 256-shot best-of-K：平均配对增益 −0.000250，CI [−0.000788,+0.000063]；97.53% 为 tie。这是因为小规模下两种电路通常都至少采到一次基态，不能据此否定期望能量差。
- 精确基态归一化平均 gap：QAOA 3p 为 0.172129，CD p 为 0.257790；QAOA 更接近基态。

### 4.4 原生基组门数

代表 n=12、m=18 测试实例，统一 basis `{cx,rz,sx,x}`：

| 配对 | opt level 0 CX | opt level 1 CX | opt level 1 depth（CD/QAOA） |
|---|---:|---:|---:|
| CD p=1 / QAOA p=3 | 108 / 108 | 108 / 108 | 101 / 75 |
| CD p=2 / QAOA p=6 | 216 / 216 | 216 / 216 | 190 / 144 |

level 3 会把 CD CX 降到 72/144，而 QAOA 3p 仍是 108/216；改用 QAOA 2p 后双方分别精确匹配为 72 和 144 CX，但 QAOA 电路深度仍更低（52 vs 74；98 vs 136）。数据：`analysis/native_gate_counts.json` 和 `analysis/compiled_level3_sensitivity/summary.json`。这些结果是无路由、all-to-all 编译，不能代表具体芯片。

## 5. 参数集中、尺度规律、最终公式与角度表

### 5.1 集中规律

主训练集中，按每实例最佳 restart 并乘 RMS 后：

- p=1：`alpha_unit` 中位数 −0.183703、IQR 0.120835；到 medoid 的中位距离 0.105429。
- p=2：两层 `alpha_unit` 中位数 [−0.150503,−0.038568]，IQR [0.079725,0.067176]；medoid 中位距离 0.317456。
- gamma/beta/alpha 的全局标准差很大，图中存在落在远端分支或边界的离群最优解；因此不能宣称唯一强集中。
- 一个紧的主分支和少量远端分支同时存在。medoid/质量迁移比直接角度均值更可靠。

数据：`concentration_summary.json` 与 `analysis/concentration_diagnostics.json`；图：`paper/figures/fig02_parameter_concentration.png`。

### 5.2 最终经验公式

设 `D=2m/n`、`s_J=sqrt(mean(J^2))`，并定义

`f_D(D)=1`（D≤1），否则 `atan(1/sqrt(D-1))`。

最终冻结公式：

`gamma_l = gamma_unit_l * f_D(D) / s_J`

`beta_l = beta_table_l`

`alpha_l = alpha_unit_l / s_J * (D/3)^(-0.5)`

验证集同时冻结：CSV gamma sign = −1，alpha multiplier = 1，BCH c = +0.5。`J -> sJ` 且 `gamma,alpha -> gamma/s,alpha/s` 是当前电路的精确恒等式，不是经验发现。

### 5.3 可直接使用的固定角度表

| p | layer | gamma_unit | beta | alpha_unit |
|---:|---:|---:|---:|---:|
| 1 | 1 | −0.4956227054 | 0.4609495592 | −0.1767317913 |
| 2 | 1 | −0.4070376889 | 0.5977566203 | −0.1423353688 |
| 2 | 2 | −0.8494824053 | 0.2177565565 | −0.0355714209 |

适用域仅限当前证据支持的二体无外场 weighted Ising、p=1/2、近似稀疏到中等度数图。不要外推为 p≥4、含外场、高阶 HUBO 或硬件噪声下的已验证表。

消融：proposed 相比 CD-reference 平均改善 +0.019413，CI [+0.015196,+0.023850]；相比 BCH +0.037683，CI [+0.031000,+0.043641]。前者中位数只有 `5.37e-5` 且 39.5% tie，说明度数修正主要改善特定分布，而非普遍逐实例改善。

## 6. 科学假设判断

- **H1 参数集中：部分支持。** RMS 归一化后出现紧主分支，固定 medoid 在 OOD 上产生强同深度收益；但远端多模态/周期分支造成巨大标准差，尚不能声称唯一或全局集中。
- **H2 alpha 随整体权重 1/s 缩放：在当前 raw CD 电路中严格成立。** 多个倍率态矢误差为 0；这是构造恒等式，不是完整最优角定理。跨权重形状表现仍需经验检验。
- **H3 少数静态特征足够改进 naive CD：得到支持，但幅度有限。** degree exponent −0.5 的 proposed 在验证集冻结后显著优于 BCH 和 reference；仍未跨越强等预算 QAOA。
- **H4 等两比特门预算优势：在已测域受到明确反证。** 主确认、level-3 重配和 n=18/20 两个诊断都为负。此结论限定于本文 ansatz、固定参数协议、p≤2、n≤20 和无噪声模拟；不构成一般 no-go theorem。

## 7. 论文与工程完成度

### 7.1 投稿完成度：**68%**（主观、证据加权）

| 阶段权重 | 本轮得分 | 依据与缺口 |
|---|---:|---|
| P0 12% | 12 | 三后端/矩阵一致，测试与门数审计完成 |
| P1 23% | 16 | 81 个训练实例、3 restarts、集中分析；缺 p=4/6、系统分支对齐和更多图族 |
| P2 20% | 17 | 公式、消融、冻结表完整；缺跨深度连续曲线和独立复现实验 |
| P3 25% | 17 | 162 主配对+24 跨规模、CI、OOD、编译敏感性；缺拓扑路由/噪声、p≥4 和外部复现 |
| P4 10% | 0 | GPU n≥24 按要求延后 |
| P5 10% | 6 | 草稿、6 图、4 表、报告、复现、ZIP 已有；仍需作者润色、期刊格式、引用逐条核对和补充实验 |

该 68% 不表示正命题已经成立；它表示负边界论文已经形成自洽骨架，但尚不足以稳妥投稿。距离投稿约 **32%**，主要是更深 p、拓扑/噪声、GPU 规模、分支对齐和外部复现，而不是继续在同一测试集调参。

### 7.2 代码工程完成度：**86%**

已有：可恢复流水线、不可变 manifest、验证集冻结、纯预测接口、三模拟器一致、测试、原始 JSON、统计、图表、参数表和封包。缺口：统一实验元数据/墙钟自动记录、集群/GPU executor、设备拓扑路由批处理、cluster bootstrap、严格图同构去重、p=4/6 的性能优化。

## 8. 论文图表与原始数据索引

论文图：

1. `paper/figures/fig01_training_free_protocol.png`：冻结协议。
2. `paper/figures/fig02_parameter_concentration.png`：最佳 restart 归一化参数与多模态离群。
3. `paper/figures/fig03_weight_scale_identity.png`：整体尺度精确协变。
4. `paper/figures/fig04_fixed_method_comparison.png`：全部固定方法。
5. `paper/figures/fig05_equal_budget_subgroups.png`：子组均值与 CI。
6. `paper/figures/fig06_same_depth_vs_budget.png`：同深度优势与等预算劣势的逐点关系。

论文表：`paper/tables/table1_dataset.csv`、`table2_method_results.csv`、`table3_fixed_angles.csv`、`table4_native_gate_counts.csv`。

最关键原始数据：

- `manifest.json`：全部 seed、图、权重、fingerprint 和 split。
- `offline_optima.json`：全部 1,458 个 restart。
- `FROZEN_MODEL.json`：测试前冻结模型。
- `heldout_results.json`：162 个测试配对的所有方法能量、shots 和门成本。
- `analysis/final_statistics.json`：论文统计。
- `analysis/subgroup_results.csv`：分层 CI。

## 9. 局限和审慎表述

1. 训练只到 p=2 的 CD，不能把结论外推到 p=4/6。
2. n=20 仍是中小规模；Liu–Wang 使用的 ansatz、训练和规模不同，本项目没有复现其全部 crossover 条件。
3. 10.15% 优化调用未报告 success；多初值缓解但不保证全局最优。
4. bootstrap 的基本单位是实例-深度行，同一实例两个深度存在相关性；投稿前应补 cluster bootstrap。
5. 指纹排除了完全相同的图+权重，但没有做严格图同构去重。
6. Qiskit 编译没有固定真实 coupling map，因此只证明统一 basis 下的门数，不是硬件路由公平性。
7. 256-shot best-of-K 在小规模出现天花板式 tie，不能替代更困难规模/成功概率曲线。
8. 本文是“固定参数迁移的边界研究”，不能表述为普遍 CD 量子优势。

## 10. 最值得开展的下一步（不超过 5 项）

1. **最小必要：p=4 的 CPU/GPU 桥接。** 训练 n=10/12 的 20–30 个实例、5 restarts，并在 n=14–20 的固定 OOD 集上测试 CD p=4 对 QAOA p=8/12；停止点是每子组至少 30 个配对或 CI 明确离开 0。
2. **cluster bootstrap 与分支对齐。** 按实例而非实例-深度行重采样；beta 按已知周期对齐，对 gamma/alpha 只做能量验证过的聚类，不假设未知全局周期。
3. **固定真实拓扑公平比较。** 选择 1–2 个公开 coupling map 和固定 Qiskit 版本/seed，报告 routed CX、SWAP、depth，并用实际相近 CX 而非固定 2p/3p 再配对。
4. **GPU 合作者任务。** Qulacs-GPU、Aer GPU 或 cuStateVec 在 n=20/24/28 做 FP64 重叠校验；批量运行冻结参数 p=1/2/4 的 100–300 个实例；只对小型训练子集做离线优化，测试实例仍保持 0 目标调用。
5. **独立新 seed 复现。** 在方法完全不变的前提下生成新 manifest，复验主负结论；若 p=4/GPU 仍无等资源优势，即以“迁移可行但资源优势域受限”的论文定位投稿，不再扩展模型复杂度。

## 11. 复现、论文和封包

- 英文论文草稿：`paper/manuscript_draft.md`。
- 完整环境和 PowerShell 命令：`reports/REPRODUCIBILITY.md`。
- 固定参数资产：`assets/cd_transfer_table_v1.csv`、`assets/cd_transfer_table_v1.json`、`assets/CD_TRANSFER_V1_FROZEN_MODEL.json`。
- 完整 ZIP：`dist/TrainingFree_CDQAOA_Codex_20261011_Complete.zip`。
- SHA256：见同目录 `TrainingFree_CDQAOA_Codex_20261011_Complete.zip.sha256.txt`，并用 `Get-FileHash` 独立核对。

本报告所有数字均来自上述 JSON/CSV 和实际命令输出；未运行的大规模 GPU、噪声和 p≥4 结果没有填入论文。
