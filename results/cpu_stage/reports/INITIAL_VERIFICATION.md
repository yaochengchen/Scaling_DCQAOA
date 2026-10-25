# 初始代码交付与验证报告（2026-10-11）

## 交付目标

已开发可运行的 CPU 原型，使得从 user-supplied QAOA `transfer_data.csv` 出发，在显式 CD-QAOA 中通过离线角度训练、参数归一化/代表值、验证集拟合以及独立 OOD 测试完成完整闭环。GPU/大型多设备数值实验未开展，后续由合作者补充。此报告仅证明项目管线能运行，**尚不证明 CD-QAOA 的收益命题**。

## 代码已实现

1. 纯 NumPy 有限深度精确态矢：逐边的 YZ、ZY 显式 CD 一阶旋转；按固定次序执行。支持标准 QAOA (`alpha=0`)。
2. 修复 CSV 的 gamma/beta 读取顺序；使用验证集一次性选择 legacy sign (+/-)，测试数据没有用于选择。
3. 生成带权 Ising: ring / regular-3 / ER / complete，unit/uniform/bimodal/signed_normal/lognormal/heavy_tail；按样本 seed 分割，记录 fingerprint。
4. 离线 QAOA/CD 优化；CD 只在 train 里多次优化，每条 restart checkpoint 持久化。
5. Medoid 角度模板 + RMS/平均度静态缩放 + BCH 系数和低维指数超参；全部在验证集冻结。
6. 等层数、等**未优化 Pauli 分解 CX**预算两类评测，独立测试能量、有限 shots、bootstrap CI、图表、报告和 CSV/JSON 参数表。
7. 可选 Qulacs、Qiskit 电路导出、Qiskit CX 编译与交叉测试接口，依赖不存在时正确跳过。

## 实际测试情况

- 执行命令：`python -m pytest -q`
- **8 passed, 2 skipped**：2 skipped 因当前环境缺少 Qulacs/Qiskit。已实际完成与 `scipy.linalg.expm` 两量子比特密集矩阵对照，以及反绝热交换子符号验证。
- `scripts/08_check_scale_law.py`: 对全局权重倍率 s={0.25,0.5,1,2,4} 的电路态矢误差均为 0（浮点精度内）。这属于构造恒等式，不是经验新成果。
- `reports/BACKENDS.md`：在此 CPU 环境下，NumPy 参考实现微基准成功，n=6,8,10；尚未在 GPU 上 benchmark。

## 实际运行的 pilot（主要记录在 results/pilot/）

- 训练集 8 个实例，验证集 8 个，测试集 4 个（n=6,8；test regular-3 + bimodal，CD p=1）。
- 训练优化记录共 48 个：含 CD p=1、标准 QAOA p=1 和 p=3 的 2 restarts；maxiter=14。测试是从未参与训练和拟合的实例。
- 验证阶段固定：CSV gamma sign=-1, alpha degree exponent=-0.5, alpha multiplier=1.5, BCH c=1.0。
- **等未优化 CX 门分解预算**：CD p=1 vs QAOA p'=3；归一化能量配对增益 `(E_QAOA - E_CD)/sum |J|` 的平均为 **-0.099655**，95% 简单实例 bootstrap CI **[-0.134581, -0.064728]**，CD win rate **0/4**。
- **这不支持“CD 固定参数系统优于 QAOA”的命题**，但 4 个实例、深度 p=1、小训练集不能确定该命题在更广范围的真伪；无需做假阳性表述。
- 输出文件：`manifest.json`, `offline_optima.json`, `FROZEN_MODEL.json`, `cd_transfer_table_v1.csv`, `cd_transfer_table_v1.json`, `heldout_results.json`, `summary.json`, `EXPERIMENT_REPORT.md`, 3 张 PNG。

## 第二个烟雾测试：results/smoke_multidepth/

- 测试 CD p=1/2、matched QAOA p=3/6 的完整端到端功能；快速优化 `maxiter=5`（极浅训练，仅验证支持多深度而非性能）。
- 训练 2、验证 2、测试 2 实例；4 个实例-深度测试对。归一化平均配对收益约 -0.6982，低质量小样本训练结果不可用于科学推广。

## 当前重要未完成任务

- 真正的多初值优化收敛、多模态角度对齐与统计规律，需要 n=8–16 规模下更广泛训练。
- 比较真实 native basis 编译后的双比特门数；当前“等预算”严格限定为无取消/无路由固定门分解。
- 对 CSV 基线 Hamiltonian sign 的源论文解释需核实；本原型仅根据历史代码并使用 val 选择，避免错误地强行固定符号。
- 构建可充分支持论文结论的 graph-isomorphism-aware 拆分、独立 OOD 样本和更强固定 QAOA 对照。
- 真实可迁移角度表、跨权重形状而非全局倍率的实验发现，需要 Codex 后续生成。
- GPU n>=24 由合作者以后补充，不作为本次是否交付完整原型的条件。

## 主观完成度（**非经验证的自动评分**）

| 内容 | 初始完成度估计 | 理由 |
|---|---:|---|
| 工程代码原型 | **35%** | 主要环节跑通，但缺真实后端交叉验证、优化收敛/大样本和严谨编译成本；离可投稿开源项目尚远 |
| 论文投稿科学证据 | **12%** | P0 起步已完成，P1–P3 的核心可推广实证证据尚缺失；P4 GPU 尚未做；P5 仅骨架 |

这些比例用于规划 Codex 的后续工作，不能被理解为已经完成 12% 的物理论证。
