# TQE_update_2026-10-28：相对 Clean 新增的代码与结果

来源：`Jiapei_final_draft/TQE_reproducibility_2026-10-28/TQE_reproducibility_2026-10-28/`（共 864 个文件）。
本文件夹只放**相对 Clean 新增或内容不同**的分析代码和结果文件，目录结构与原包一致，所以原包 `README.md`、`RESULTS_INDEX.md` 里的相对路径仍然有效。稿件（`manuscript/`）不放在这里。Clean 里已有的文件没有任何改动。

## 比对方法

逐文件按 SHA-256 比对（原包自带 `SHA256.json`；Clean 一侧用 `Optimal_Success_Probability/SHA256.json`，其余候选文件逐个计算哈希；`runs.zip` 解压后逐条比对）。

| 类别 | 文件数 | 处理 |
|---|---:|---|
| 新增 / 内容不同的代码与结果 | 67 + EMAIL_DRAFT.txt | 放在本文件夹 |
| 与 Clean 中某文件逐字节相同 | 50 | 未复制，位置见 `IDENTICAL_IN_CLEAN.json` |
| `inputs/optimizer_benchmark/runs/`（720 条轨迹） | 720 | 与 Clean 的 `runs.zip` 内容逐条相同，未复制 |
| `manuscript/` | 26 | 不放 |

## 新增内容（按用途）

**参数方式（U1 幂律）与拟合**
- `models/A_joint_candidate.json`：历史候选模型（仅溯源用）。当前规则 `models/U1_power_law.json` 已在 `Scaling_DCQAOA/Optimal_Success_Probability/study/models/`，未重复放。
- `refit_parameters.py`、`code/study.py`、`code/render_shared_column_titles.py`、`code/render_exponents.py`：从训练角度重拟合 64 个 cost/CD 幅度与指数，绘制图 2、S1、S2。

**成功概率结果（PennyLane）**
- `results/pennylane_probability_1050/`：1,050 图 × 深度 1–4 的**完整 4,200 条**配对结果（`paired_probabilities_1050.csv`、`summary.json`、`verification.json`）。Clean 里 `Optimal_Success_Probability` 只有当时 3,610/4,200 的快照；其 `source_manifest.json` 与本包的 `manifest.json` 相同，所以 manifest 未重复放。
- `results/pennylane_probability_low_degree_p6_600/`：600 个小图 × 5 个深度（含 p=6）以及 11 个低度图的结果（图 S5、S6）。
- `figures/figure3_depth_average_probability/statistics.json`、`figures/figure6_depth_average_probability/statistics_and_negative_audit.json`：图 3/7/S7 的统计与负值审计。
- `code/probability_statistics.py`、`figure3_depth_average_review.py`、`figure6_depth_average_review.py`、`render_probability_supp.py`、`render_single_column_trial.py`、`circuit_core.py`。

**优化实验（图 5、6）**
- `results/benchmark_12graphs.json`：12 图在**当前公式**下的参考值；`results/benchmark_new_summary.json`：重新计算的 720 条轨迹汇总。
- `results/figure5_ajoint_c128_cpu/`：12 个 U1 公式起点 + 240 个随机起点的 adjoint 梯度精修（252 条链）。
- `code/portable_optimizer.py`、`rerun_optimization.py`。

**硬件（IBM job dbgjqlgte5oc738iauug）**
- `code/render_hardware.py`、`code/render_main_hardware.py`：从原始 counts 重算 84 个电路的成功概率/近似比，生成图 4、S3、S4。
- `hardware/DOWNLOAD_MANIFEST.json`。其余硬件文件（plan、results、raw counts、qpy、calibration 等）与 `Clean/cdqaoa_hw/` 下相同，未复制。

**复现入口与溯源**
- `reproduce.py`、`rerun_probabilities.py`、`requirements-*.txt`、`README.md`、`RESULTS_INDEX.md`、`VERIFICATION.md`、`SHA256.json`。
- `code/figure1/main.tex`：图 1 的 TikZ 源码。
- `code/reference/cdtransfer/experiment.py`、`code/reference_analysis/prr_figures.py`：新版的这两个文件与 Clean 里同名文件内容略有不同，放在本文件夹自己的路径下，不影响 Clean 原文件。
- `provenance/`：原始实现（溯源，含历史路径），以及可移植化改动说明 `adaptations.md`。

## 在本地跑之前：先补齐省略的文件

```
cd D:\software\Jiapei\CD_QAOA_FIRST\Clean\TQE_update_2026-10-28
python restore_from_clean.py --check   # 只检查，不写文件
python restore_from_clean.py           # 从 Clean 复制 50 个相同文件并解压 720 条轨迹
python reproduce.py figures
python refit_parameters.py
```

`restore_from_clean.py` 默认把上一级目录当作 Clean 根目录，每个文件都对照 `SHA256.json` 校验，不会覆盖已存在的文件。

因为不含 `manuscript/`：`reproduce.py figures`、`refit_parameters.py`、`rerun_*.py` 可以正常用；`reproduce.py build` 不可用；`reproduce.py verify` 前半部分（图数、均值检查）会通过，但随后逐个校验 `SHA256.json` 时会在第一个 manuscript 文件处报缺失。
