#!/usr/bin/env python
"""从已下载的原始记录生成错误率 / 保真度表格（不连接 IBM，不消耗机时）。

用法（在 cdqaoa_hw 文件夹下）：
    python error_tables.py raw_dbgb8v5rd2mc73cucek0      # 旧规则那次
    python error_tables.py raw_dbgjqlgte5oc738iauug      # 新规则那次
需要 raw_<job>/ 里的 calibration.json、isa_circuits.qpy、circuits.csv、stats.csv、job_metadata.json。
输出到 raw_<job>/error_tables/：
    device_summary.csv     整台机器的中位数/分位数（CZ、SX、读出误差，T1、T2）
    qubits.csv             每个物理比特：T1、T2、读出误差、SX/X 误差、被多少条线路用到
    cz_pairs.csv           每个比特对：CZ 误差、门长、被用到的次数
    circuit_fidelity.csv   每条线路：实际用到的门的误差汇总、按校准估计的保真度、实测结果
    fidelity_by_n.csv      按 (n, 方法) 平均的上表

估计保真度 = Π(1-CZ误差) × Π(1-单比特门误差) × Π(1-读出误差)，按线路实际用到的每个门和比特计算。
这是只看门误差的粗略模型：不含退相干空闲误差、串扰、相干误差，也不反映 DD/twirling 的效果，
只用作相对比较，不能当作线路的真实保真度。
"""
import csv, json, sys
from pathlib import Path

import numpy as np
from qiskit import qpy

raw = Path(sys.argv[1])
out = raw / "error_tables"
out.mkdir(exist_ok=True)
cal = json.loads((raw / "calibration.json").read_text(encoding="utf-8"))
meta = json.loads((raw / "job_metadata.json").read_text(encoding="utf-8"))


def par(params, name):
    for p in params:
        if p["name"] == name:
            return p["value"]
    return None


# ---- 校准数据
qubit = {}
for q, ps in enumerate(cal["qubits"]):
    qubit[q] = dict(T1_us=par(ps, "T1"), T2_us=par(ps, "T2"), readout_error=par(ps, "readout_error"),
                    p_meas0_prep1=par(ps, "prob_meas0_prep1"), p_meas1_prep0=par(ps, "prob_meas1_prep0"))
one_q = {}                         # (gate, qubit) -> error
cz = {}                            # (q0, q1) -> (error, length_ns)   只取 gate == "cz"
for g in cal["gates"]:
    err, length = par(g["parameters"], "gate_error"), par(g["parameters"], "gate_length")
    if g["gate"] == "cz":
        cz[tuple(sorted(g["qubits"]))] = (err, length)
    elif len(g["qubits"]) == 1 and g["gate"] in ("sx", "x"):
        one_q[(g["gate"], g["qubits"][0])] = err
        qubit[g["qubits"][0]][f"{g['gate']}_error"] = err

# ---- 线路
circs = qpy.load(open(raw / "isa_circuits.qpy", "rb"))
with open(raw / "circuits.csv", encoding="utf-8") as f:
    crow = list(csv.DictReader(f))
with open(raw / "stats.csv", encoding="utf-8") as f:
    srow = list(csv.DictReader(f))
assert len(circs) == len(crow) == len(srow)
old_summary = raw.parent / "hw_results.csv"           # 旧规则那次的理想值在这里
if "ratio_ideal" not in srow[0] and old_summary.exists() and "dbgb8v5rd2mc73cucek0" in raw.name:
    with open(old_summary, encoding="utf-8") as f:
        ideal = list(csv.DictReader(f))
    for s, h in zip(srow, ideal):
        s["ratio_ideal"] = h["ratio_ideal"]

q_use, cz_use, rows = {}, {}, []
for k, (qc, cr, sr) in enumerate(zip(circs, crow, srow)):
    e2, e1, measured = [], [], []
    for ins in qc.data:
        name = ins.operation.name
        qs = [qc.find_bit(q).index for q in ins.qubits]
        if len(qs) == 2:
            pair = tuple(sorted(qs))
            e2.append(cz[pair][0])
            cz_use[pair] = cz_use.get(pair, 0) + 1
        elif name in ("sx", "x"):
            e1.append(one_q[(name, qs[0])])
        elif name == "measure":
            measured.append(qs[0])
    used = sorted({qc.find_bit(q).index for ins in qc.data for q in ins.qubits
                   if ins.operation.name not in ("barrier", "delay")})
    for q in used:
        q_use[q] = q_use.get(q, 0) + 1
    e2, e1 = np.array(e2), np.array(e1)
    ro = np.array([qubit[q]["readout_error"] for q in measured])
    f2, f1, fro = np.prod(1 - e2), np.prod(1 - e1), np.prod(1 - ro)
    r = dict(index=k, id=sr["id"], n=sr["n"], p=sr.get("p", cr.get("p", "2")), method=sr["method"],
             twoq=len(e2), twoq_depth=cr["twoq_depth"], oneq_sx_x=len(e1), n_qubits=len(used),
             cz_err_mean=e2.mean(), cz_err_median=np.median(e2), cz_err_max=e2.max(), cz_err_sum=e2.sum(),
             sx_err_mean=e1.mean() if len(e1) else 0, readout_err_mean=ro.mean(),
             T1_us_mean=np.mean([qubit[q]["T1_us"] for q in used]),
             T2_us_mean=np.mean([qubit[q]["T2_us"] for q in used]),
             F_est_2q=f2, F_est_1q=f1, F_est_readout=fro, F_est_total=f2 * f1 * fro)
    for key in ("p_opt", "p_opt_ideal", "p_opt_random", "ratio", "ratio_ideal", "ratio_random"):
        if key in sr:
            r[key] = float(sr[key])
    if "ratio_ideal" in r:          # 实测保留比例 (hw-random)/(ideal-random)
        r["ratio_retention"] = (r["ratio"] - r["ratio_random"]) / (r["ratio_ideal"] - r["ratio_random"])
    if "p_opt_ideal" in r:
        r["p_opt_retention"] = (r["p_opt"] - r["p_opt_random"]) / (r["p_opt_ideal"] - r["p_opt_random"])
    r["physical_qubits"] = " ".join(map(str, used))
    rows.append(r)


def write(path, rows_):
    cols = list(rows_[0])
    for r in rows_:
        cols += [c for c in r if c not in cols]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows_)


write(out / "circuit_fidelity.csv", rows)
write(out / "qubits.csv", [dict(qubit=q, **qubit[q], used_by_circuits=q_use.get(q, 0)) for q in sorted(qubit)])
write(out / "cz_pairs.csv", [dict(q0=a, q1=b, cz_error=e, cz_length_ns=l, uses=cz_use.get((a, b), 0))
                             for (a, b), (e, l) in sorted(cz.items())])

# ---- 按 (n, 方法) 平均
groups = {}
for r in rows:
    groups.setdefault((int(r["n"]), r["method"]), []).append(r)
num = [c for c in rows[0] if isinstance(rows[0][c], (int, float, np.floating)) and c not in ("index", "n")]
agg = [dict(n=n, method=m, graphs=len(rs), **{c: float(np.mean([float(r[c]) for r in rs])) for c in num
                                              if all(c in r for r in rs)})
       for (n, m), rs in sorted(groups.items())]
write(out / "fidelity_by_n.csv", agg)

# ---- 整机汇总
def stats(name, v):
    v = np.array([x for x in v if x is not None], float)
    return dict(quantity=name, count=len(v), median=np.median(v), p10=np.percentile(v, 10),
                p90=np.percentile(v, 90), mean=v.mean(), min=v.min(), max=v.max())
used_pairs = [p for p in cz if cz_use.get(p)]
write(out / "device_summary.csv", [
    stats("CZ error (all pairs)", [cz[p][0] for p in cz]),
    stats("CZ error (pairs used in this job)", [cz[p][0] for p in used_pairs]),
    stats("SX error", [qubit[q].get("sx_error") for q in qubit]),
    stats("readout error", [qubit[q]["readout_error"] for q in qubit]),
    stats("T1 (us)", [qubit[q]["T1_us"] for q in qubit]),
    stats("T2 (us)", [qubit[q]["T2_us"] for q in qubit]),
])
print(f"backend {meta.get('backend')}  calibration {cal.get('last_update_date')}  job {meta.get('job_id')}")
print(f"-> {out}/ : device_summary.csv, qubits.csv, cz_pairs.csv, circuit_fidelity.csv, fidelity_by_n.csv")
