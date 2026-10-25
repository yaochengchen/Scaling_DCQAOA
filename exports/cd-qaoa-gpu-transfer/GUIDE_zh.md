# 统一 GPU 角度表：CD-QAOA 固定参数迁移使用指南

本包提供**一条公式、一张逐深度角度表、一个共同的参考平均度 \(D_0\)**。给定图后可直接计算 CD-QAOA 参数，不在目标图上优化，也不运行量子模拟。本表只由既有 GPU **训练**记录离线构造；随后已用冻结的统一表，在原有 7,000 张保留测试图上完成 32,000 次固定角度线路评估。结果见 [统一表评估报告](../../analysis/outputs_unified/REPORT.md)。模型 JSON 和公式代码在这次评估中均未修改。

## 1. 适用问题与线路约定

正权加权 MaxCut：\(n\) 个顶点，边权 \(J_{ij}>0\)，

$$H_C=\sum_{(i,j)\in E}J_{ij}Z_iZ_j,\qquad H_B=\sum_i X_i.$$

初态为 \(|+\rangle^{\otimes n}\)。第 \(l\) 层先施加 \(e^{-i\gamma_lH_C}\)，再按边的字典序依次施加 \(e^{-i\alpha_lJ_{ij}Y_iZ_j}\) 与 \(e^{-i\alpha_lJ_{ij}Z_iY_j}\)，最后施加 \(e^{-i\beta_lH_B}\)。核对所用软件框架的旋转门是否把指数角度定义为参数的一半；否则不能直接照搬数字。负 \(\gamma\) 源于这里的哈密顿量和指数符号约定，不要擅自取绝对值。

只有 \(p=1,2,3,4,6\) 的模板。\(p=5\) 与 \(p>6\) 没有现成模板。训练范围：\(p\leq4\) 覆盖 \(n=10,12,14,16,18,20,22\)；\(p=6\) 覆盖 \(n=10,12,14,16,18\)。负权、零权、有向边、重复边不在本包定义域。

## 2. 迁移公式

从目标图计算平均度 \(D\)、边权均方根 \(s_J\)：

$$
D=\frac{2|E|}{n},\qquad
s_J=\sqrt{\frac{1}{|E|}\sum_{(i,j)\in E}J_{ij}^{\,2}},\qquad
f_D(D)=
\begin{cases}
\arctan(1/\sqrt{D-1}),&D>1,\\
1,&D\leq1.
\end{cases}
$$

共同参考值：\(D_0=3.4545454545454546\)，\(b=-1/2\)，\(c_\alpha=1\)。对所选深度 \(p\) 的第 \(l=1,\ldots,p\) 层：

$$
\boxed{\displaystyle
\gamma_l=\gamma_l^{\rm unit}\frac{f_D(D)}{s_J},\qquad
\beta_l=\beta_l^{\rm unit},\qquad
\alpha_l=\frac{\alpha_l^{\rm unit}}{s_J}
\left(\frac{D}{D_0}\right)^{-1/2}.}
$$

**所有规模都用同一张表与同一个 \(D_0\)**，不按 \(n\) 切换旧实验模型。公式形式与原 GPU 主实验相同；这里只重新汇总了表值并统一了 \(D_0\)。

## 3. 统一角度表

下表保留六位小数；实际计算读取 [全精度模型](assets/unified_gpu_model.json)，不要用舍入后的值。

| \(p\) | \(\boldsymbol\gamma^{\rm unit}\) | \(\boldsymbol\beta^{\rm unit}\) | \(\boldsymbol\alpha^{\rm unit}\) | 训练图数 |
|---:|---|---|---|---:|
| 1 | −0.484390 | 0.479780 | −0.155486 | 7,000 |
| 2 | −0.408528, −0.867625 | 0.613549, 0.206408 | −0.120026, −0.025438 | 7,000 |
| 3 | −0.342069, −0.839489, −0.680321 | 0.719497, 0.331893, 0.162807 | −0.105171, −0.059108, −0.017935 | 7,000 |
| 4 | −0.310922, −0.785077, −0.728243, −0.649480 | 0.754922, 0.428698, 0.253991, 0.140993 | −0.101272, −0.048531, −0.040444, −0.009732 | 7,000 |
| 6 | −0.252953, −0.689333, −0.644973, −0.652261, −0.664684, −0.612785 | 0.788766, 0.489933, 0.348545, 0.262893, 0.179982, 0.108804 | −0.081611, −0.054337, −0.050842, −0.050044, −0.036952, −0.005273 | 5,000 |

## 4. 使用

解压 ZIP，以包目录为当前目录运行：

~~~powershell
python scripts/transfer.py predict --graph examples/graph.json --p 1
~~~

图文件顶点编号为 \(0,\ldots,n-1\)，边写成 [较小端点, 较大端点, 正权重]。[示例图](examples/graph.json) 是 10 顶点单位权环。输出三组最终角度与 \(D,s_J,D_0\)。此命令只计算静态图特征。

在 Python 中调用：

~~~python
import json
from pathlib import Path
from scripts.transfer import predict_angles

model = json.loads(Path("assets/unified_gpu_model.json").read_text(encoding="utf-8"))
graph = json.loads(Path("examples/graph.json").read_text(encoding="utf-8"))
angles = predict_angles(model, graph["n"], graph["edges"], p=2)
print(angles["gamma"], angles["beta"], angles["alpha"])
~~~

使用冻结模型推断只依赖 Python 标准库。其他框架可直接按公式实现，但需保持相同的哈密顿量与门顺序。

## 5. 离线构造与复算

包中附了三份既有 GPU 实验的压缩**训练最优角度记录**，不是新计算的结果：

- data/small_best_optima.csv.gz：\(n=10\)–16。
- data/large_best_optima.csv.gz：\(n=18\)–22，\(p\leq4\)。
- data/extra_best_optima.csv.gz：\(n=18\) 的 \(p=6\) 补充实验。

对 \(p=1,2,3,4\) 合并 small 的 4,000 张图与 large 的 3,000 张图；对 \(p=6\) 合并 small 的 4,000 张图与 extra 的 1,000 张图。extra 的 \(p\leq4\) 记录与 large 重复，未加入。每图每深度使用现有 8 个起点中能量最好的 CD 记录。若 \(\gamma\) 平均值为正，同时翻转 \(\gamma,\beta\) 符号来对齐等能量分支。

规范化每组角度为 \((\boldsymbol\gamma s_J/f_D(D),\boldsymbol\beta,\boldsymbol\alpha s_J)\)。每个深度选到同深度其他规范化角度的欧氏距离之和最小的 **medoid**：它是一组真实训练图的角度，不是逐坐标平均。\(D_0\) 取 small+large 共 7,000 张不重复训练图平均度的中位数，所有深度共用。\(b=-1/2,c_\alpha=1\) 保持原 GPU 验证集选定值，没有重新拟合。

复算模型需要 NumPy、SciPy，但**不需要 GPU 或量子模拟**：

~~~powershell
python -m pip install numpy scipy
python scripts/transfer.py build
python -m unittest discover -s tests -v
~~~

构建脚本写出模型 JSON 与 SHA-256 文件。模型哈希：f2be3446fd992a826dcfec8f6048b27a8ccf21fff146d06faa84fd507d7f1eb4。JSON 内另含三份源文件的 SHA-256，供核对来源。JSON 中的 `status` 字段保留**建表时**“尚未测试”的历史标记，以免更改冻结文件与哈希；当前评估状态以本指南和上述报告为准。

## 6. 理论分析与论文用语边界

**精确性质：整体权重缩放协变。** 若 \(J_{ij}\mapsto cJ_{ij}\)（\(c>0\)），则 \(s_J\mapsto cs_J\)。规则让 \(\gamma,\alpha\) 各除以 \(c\)，\(\beta\) 不变；因此加权生成元的指数参数及指定门顺序的整条线路完全不变。这是代数关系，不依赖数据拟合。

**经验假设：度缩放与角度集中。** \(f_D(D)\) 与 \((D/D_0)^{-1/2}\) 并非已证明的普遍最优式；规范化角度跨图集中、pooled medoid 有代表性，也只是统计现象。旧 GPU 实验的验证集选过 \(b=-1/2,c_\alpha=1\)；本次统一表测试沿用它们，并检验了这一完整规则的预测效果，但没有重新寻找这些系数。

**统一表成绩只来自统一表自己的评估。** 原 GPU 实验分别冻结并测试了各自的角度表；它们的近似比和胜率不能平均或拼接成统一表的性能。现在可引用统一表在原有 7,000 张保留测试图上的新增评估结果：单层平均近似比由同层固定 QAOA 的 0.771860 提高到统一表 DC-QAOA 的 0.835305。统一表是在原两组测试分析之后汇总、在本次评估之前锁定；不应称这些图在整个研究历史中都从未被看过。更大于 22 量子比特的性能未实测；同层数优势也不能等同于同双量子比特门预算优势。

给理论合作者的分工线索：整体权重尺度协变可严格证明；度因子的来源、参数集中机制及异质图/深层失效条件仍待解释。
