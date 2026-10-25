# 等深 CD-QAOA GPU 扩展实验：使用说明

在 8 × RTX PRO 6000 上用 PennyLane Lightning-GPU，对更大的 n、更深的 p、更多图族和正权分布生成优化后的 QAOA / CD-QAOA 参数；随后检验参数集中性和缩放规律，拟合并冻结新的固定参数公式，在独立测试集上与 QAOA 同层固定参数、当前 V1 公式做配对比较。所有比较都是**等深**：CD-QAOA p 层 对 QAOA p 层。

## 1. 安装（只需一次）

驱动需支持 CUDA 12.8+（`nvidia-smi` 右上角 CUDA Version ≥ 12.8）。CUDA 运行库和 cuStateVec 由 pip 装进 conda 环境，不需要系统 CUDA Toolkit。

```bash
unzip CDQAOA_EqualDepth_LightningGPU_8x_*.zip && cd CDQAOA_EqualDepth_LightningGPU_8x
conda env create -f environment.yml
conda activate cdqaoa
python -m pip install -e '.[gpu,dev]'
python -c "import pennylane as qml; qml.device('lightning.gpu', wires=2); print('lightning.gpu OK')"
```

如果报找不到 `libcustatevec.so` / `libcudart.so`，把 pip 装的 NVIDIA 库加进库路径（可写入 `$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh`）：

```bash
export LD_LIBRARY_PATH=$(python -c "import site,pathlib;print(':'.join(sorted({str(p.parent) for r in site.getsitepackages() for p in pathlib.Path(r).rglob('lib*.so*') if 'nvidia' in str(p) or 'custatevec' in str(p)})))"):${LD_LIBRARY_PATH:-}
```

## 2. 运行步骤

所有命令都在项目根目录、`conda activate cdqaoa` 之后、**tmux 里**运行，这样 SSH 断线不会中断任务：

```bash
tmux new -s cdqaoa          # 新建；断线后重新登录用  tmux attach -t cdqaoa  回到原窗口
```

**中断后重跑同一条命令即续跑**，已完成的优化不会重算。实例、深度、优化器等科学配置一旦用过就不能改（要换新输出目录）；GPU 编号、每卡进程数这类调度参数可以随时改。

如果重跑时提示 `... is still being trained by another process`：说明上一次的进程还活着（例如断线前没用 tmux）。本项目的进程名都是 `cdqaoa-launch`（主进程）和 `cdqaoa-worker`（GPU 子进程），用 `ps -e | grep cdqaoa` 查看，`pkill cdqaoa` 全部结束，再重跑同一命令即可，已完成的结果都保留。主进程退出时其 GPU 子进程也会自动退出。

### 第 1 步：冒烟测试（约十几分钟）

```bash
bash scripts/run_gpu_8x.sh configs/eqdepth_smoke.json results/eqdepth_smoke
```

先在 GPU 0 上核对 Lightning-GPU 与 NumPy 参考实现的态矢量、能量、伴随梯度（不一致直接停止），再完整跑一遍 train → fit → test → report。看 `results/eqdepth_smoke/EXPERIMENT_REPORT.md` 和 `figures/` 确认正常。

### 第 2 步：测每张卡同时跑几个进程最快（约 3–5 分钟，只需一次）

```bash
python scripts/20_quick_bench.py --gpus 0,1,2,3,4,5,6,7
```

不跑训练流程：对 n=18、22 分别在每卡同时开 1/2/4 个进程，每个进程反复调用“能量+伴随梯度”（CD p=3）20 秒，统计 8 张卡的总调用速率（不含进程启动时间），最后打印 `Use: OPTS="..."`。想多测几个值可以加参数，例如 `--n 18 20 22 24 --levels 1 2 4 8 --seconds 30`。`scripts/19_tune_gpu_workers.py` 是用完整训练流程测的慢速版本，一般不需要。

含义：每卡若干“槽位”，n 小的实例占 1 个槽（一张卡同时跑多个），n 大的占多个或整卡（未测的 n 默认独占一张卡）。调度器大实例优先，小实例填满剩余槽位。

速度说明：n ≤ 22 时每次“能量+梯度”调用的耗时主要花在 PennyLane 的 Python 层（每次重建 QNode、autograd 追踪、把 PauliRot 重新分解），GPU 本身只占小部分。现在的代码对每个 (p, CD/QAOA) 只建一次线路 tape，之后只更新角度并直接调用 Lightning 的 C++ 伴随梯度（`engine: "tape"`，预检会与 NumPy 参考和普通 QNode 结果逐项核对）；CPU 上实测每次调用快 2.5–3.7 倍。旧版本测到的每卡进程数结论（n=18：1 个进程 19598、2 个 12858、4 个 8629 次/小时；n=20、2 个进程 5549 次/小时）不再适用，需要用新代码重测。

可选：开启 CUDA MPS 再测一次（同一个 shell 里）：

```bash
source scripts/mps_start.sh 0,1,2,3,4,5,6,7
python scripts/19_tune_gpu_workers.py --out results/tune_slots_mps
# 若 MPS 更快，之后的训练也在这个 shell 里运行；全部结束后：bash scripts/mps_stop.sh
```

进程总数 = 8 × 槽位数，每个进程占 1 个 CPU 核，先用 `nproc` 确认核数够用。

把推荐结果存进变量，后面的命令都用它（没做第 2 步就用默认值）：

```bash
OPTS="--slots-per-gpu 1"          # 或第 2 步打印的那一行，例如 "--slots-per-gpu 4 --task-weights 18:1,20:1,22:2"
```

### 第 3 步：计时预实验 + 估算总耗时

```bash
bash scripts/run_gpu_8x.sh configs/eqdepth_pilot_timing.json results/eqdepth_pilot_timing \
     $OPTS
python scripts/22_estimate_cost.py --pilot results/eqdepth_pilot_timing --config configs/eqdepth_stage20.json
python scripts/22_estimate_cost.py --pilot results/eqdepth_pilot_timing --config configs/eqdepth_full50.json
```

`$OPTS` 是第 2 步定下的调度参数（见第 2 步末尾）。根据估算决定是直接跑 50 个/单元，还是先跑 20 个；时间太长时可以先把配置里的 `depths` 改成 `[1,2,3,4]`（存为新配置），之后再加 p=6 用 `--reuse-from` 续上。

### 第 4 步：起步规模（每单元 20/10/10）

```bash
bash scripts/run_gpu_8x.sh configs/eqdepth_stage20.json results/eqdepth_stage20 \
     $OPTS
```

### 第 5 步：目标规模（每单元训练 50），复用第 4 步

```bash
bash scripts/run_gpu_8x.sh configs/eqdepth_full50.json results/eqdepth_full50 \
     $OPTS --reuse-from results/eqdepth_stage20
```

前 20 个训练实例与第 4 步完全相同，直接复制，只计算新增的 30 个。

### 第 6 步：参数集中性与缩放规律分析

```bash
python scripts/21_eqdepth_concentration.py --run results/eqdepth_full50
```

### 可选

```bash
# 带符号 Ising 数据集（单独目录，不与 MaxCut 近似比混合）
bash scripts/run_gpu_8x.sh configs/eqdepth_ising_signed.json results/eqdepth_ising_signed $OPTS
# 更大规模 n=26,28（每个实例独占一张卡）
bash scripts/run_gpu_8x.sh configs/eqdepth_large_26_28.json results/eqdepth_large_26_28
```

### 分阶段运行 / 监控

```bash
python -m cdtransfer.cli train    --config CONFIG --out OUT [--slots-per-gpu 4 --task-weights ...] [--reuse-from DIR]
python -m cdtransfer.cli fit      --config CONFIG --out OUT        # 训练全部完成后才允许
python -m cdtransfer.cli evaluate --config CONFIG --out OUT
python -m cdtransfer.cli report   --config CONFIG --out OUT
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw --format=csv -l 5   # 另开窗口看利用率
```

训练日志每完成一个实例打印一行（已完成/总优化次数、粗略剩余时间），同时写入 `OUT/launcher.log`；GPU 利用率记录在 `OUT/gpu_utilization_*.csv`。某个实例出错时其他实例继续，错误写在 `OUT/errors/`，修好后重跑同一命令。

## 3. 实验设计

| 配置 | n | 图族 | 权重 | p | 训练/验证/测试 每单元 | 起点数 |
|---|---|---|---|---|---|---|
| eqdepth_smoke | 18 | regular3, er05 | uniform | 1,2,3 | 4/1/1 | 2 |
| eqdepth_pilot_timing | 18, 22 | ring, regular3, regular6, er05 | uniform, heavy_tail | 1,2,3,4,6 | 2/1/1 | 2 |
| eqdepth_stage20 | 18,20,22,24 | 10 族 | 5 种正权 | 1,2,3,4,6 | 20/10/10 | 8 |
| eqdepth_full50 | 18,20,22,24 | 10 族 | 5 种正权 | 1,2,3,4,6 | 50/10/20 | 8 |
| eqdepth_large_26_28 | 26, 28 | regular3, er3, er05 | uniform, lognormal | 1,2,3,4 | 10/5/5 | 4 |
| eqdepth_ising_signed | 18–24 | 10 族 | signed_normal | 1,2,3,4,6 | 20/10/10 | 8 |

- **10 个图族**：随机树 `tree`、环 `ring`、随机 3/4/5/6-正则、稀疏 ER `er3`（平均度≈3）、稠密 ER `er05`（连边概率 0.5）、两社区 SBM `community`、随机二部图 `bipartite`。
- **5 种正权**：`unit`、`uniform`(0.5–1.5)、`bimodal`{0.5, 1.5}、`lognormal`(σ=0.65)、`heavy_tail`(对数正态 σ=1.2，截断到 10)。`ring × unit` 的实例彼此同构，只差顶点编号。
- **p=1,2** 很便宜：作为插值链的起点，并且可以在测试集上与 V1 公式（只有 p=1,2 的 CD 模板）配对比较。
- **每个实例按 p 从小到大链式优化**（L-BFGS-B + 伴随梯度），每个 (p, 方法) 8 个起点：QAOA = 线性 ramp、由 p−1 最优解插值、6 个独立随机；CD = 从同层 QAOA 最优解出发（α=0）、由 p−1 插值（p=1 时为 ramp）、6 个独立随机。所有起点的结果都保留。由于 CD 在 α=0 时就是 QAOA，CD 的最优值不会比同层 QAOA 差，`optimizer_audit.json` 会列出例外。
- **精确参考值**：n ≤ 24（大规模配置 ≤ 28）时，在 CPU 上穷举 2^(n−1) 个自旋态求精确 E_min / E_max，近似比是精确的。
- **数据隔离**：训练集 → 模板；验证集 → 超参数（CSV 符号、固定 QAOA 方法、α 的度修正指数和倍率、BCH 系数）；冻结 `FROZEN_MODEL.json` 并写 SHA-256；测试集只评估一次，测试实例上不做任何优化。
- 等深比较中 CD 每层的双比特旋转数是 QAOA 的 3 倍（结果里的 `cx_*` 列），这不是等门预算比较。`eval_calls` 是模拟器上“能量 + 伴随梯度”的调用次数，不等于硬件线路评估次数。

## 4. 输出文件（OUT 目录）

| 文件 | 内容 |
|---|---|
| manifest.json, split_*.csv | 不可变的 train/val/test 划分、图指纹、各划分 SHA-256、静态图特征 |
| checkpoints/<实例>/p{p}_{qaoa,cd}_r{k}.json | 每次优化：起点类型、初始/最终参数、能量、精确近似比、评估次数、优化器状态、耗时、GPU |
| checkpoints/<实例>/instance.json | 精确 E_min / E_max |
| traces/<实例>/*.jsonl.gz | 每次能量+梯度调用的参数、能量、best-so-far、梯度范数、时间 |
| offline_optima.jsonl | 所有训练记录（一行一条） |
| optimizer_audit.json | 未收敛列表、CD 比 QAOA 差的例外、总评估次数和总耗时 |
| environment_*.json, gpu_utilization_*.csv | conda/软件版本、GPU 型号/驱动、CPU 核数、调度参数、源码 SHA-256；GPU 利用率、显存、功耗 |
| FROZEN_MODEL.json, FROZEN_MODEL.sha256 | 冻结的新公式 |
| test_results/, heldout_results.json | 测试集逐实例结果：各方法能量、近似比、CX 数 |
| summary.json, summary_table.csv, EXPERIMENT_REPORT.md | 按 p × 图族 / 权重 / n 的配对增益、95% bootstrap CI、胜率、近似比；训练集优化上限 |
| figures/ | fig01 图族、fig02 权重分布、fig01b 比特数（同层比较，含 V1）、fig03 训练集优化上限 |
| analysis/ | 第 6 步：集中性表、IQR 随 n 的变化、归一化参数与图特征的相关性、对度数的幂律指数、参数曲线图 |

基于 `analysis/` 提出新的公式形式后，应只用训练/验证集确定，写成新的冻结模型，再在测试集上评估一次。
