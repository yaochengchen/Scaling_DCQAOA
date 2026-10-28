#!/usr/bin/env python
"""取回新规则（U1）IBM job 的全部原始记录（不重新提交、不消耗 QPU 时间）。
与旧规则的 fetch_raw_job.py 输出相同的一套文件。

在放 tqe_hw_run.json、tqe_hw_plan.json 的同一个文件夹运行：
    python fetch_raw_tqe_job.py
输出到 raw_<job_id>/：
    counts.json          每条线路的 counts、实际 shots、图 ID、方法、深度
    bitstrings.npz       每条线路逐 shot 的原始比特（BitArray 打包数组）
    stats.csv            由 counts 计算：最优解概率及标准误、平均近似比及标准误、最好样本、随机基线、论文理想值
    isa_circuits.qpy     job 实际执行的编译后线路（从 job 输入取回）
    circuits.csv         每条线路的双比特门数、双比特深度、总深度、使用的物理比特
    job_metadata.json    job 信息、用时、metrics、执行选项、结果元数据
    calibration.json     执行当时的后端校准数据（若可取得）
"""
import json
from pathlib import Path

import numpy as np
from qiskit import qpy
from qiskit_ibm_runtime import QiskitRuntimeService

TWO_Q = {"cz", "ecr", "cx", "rzz"}


def energies(x, edges, w):
    e = np.zeros(len(x))
    for (i, j), wij in zip(edges, w):
        e += wij * (1.0 - 2.0 * (((x >> i) ^ (x >> j)) & 1))
    return e


def jdump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def try_(f, default=None):
    try:
        return f()
    except Exception as exc:
        print(f"  [skip] {exc}")
        return default


st = json.loads(Path("tqe_hw_run.json").read_text())
items = json.loads(Path("tqe_hw_plan.json").read_text())["items"]
service = QiskitRuntimeService()
job = service.job(st["job_id"])
out = Path(f"raw_{st['job_id']}")
out.mkdir(exist_ok=True)
print("job", st["job_id"], "status", job.status())
res = job.result()
assert len(res) == len(items), "pub 数与 tqe_hw_plan.json 不一致"

# ---- counts、逐 shot 比特、统计量
counts_out, arrays, rows, extrema = [], {}, [], {}
for k, (it, pr) in enumerate(zip(items, res)):
    ba = pr.data.meas
    counts = ba.get_counts()
    shots = int(ba.num_shots)
    assert sum(counts.values()) == shots
    method = f"{it['method']}_p{it['p']}"
    counts_out.append(dict(index=k, id=it["id"], n=it["n"], p=it["p"], method=method,
                           shots=shots, counts=counts, pub_metadata=try_(lambda: dict(pr.metadata), {})))
    arrays[f"{k:02d}|{it['id']}|{method}"] = ba.array
    n, edges, w = it["n"], it["edges"], it["weights"]
    if it["id"] not in extrema:
        ev = energies(np.arange(1 << n, dtype=np.int64), edges, w)
        emin_ = float(ev.min())
        extrema = {it["id"]: (emin_, float(ev.max()), int((np.abs(ev - emin_) < 1e-9).sum()))}
    emin, emax, nopt = extrema[it["id"]]
    x = np.array([int(b, 2) for b in counts], dtype=np.int64)
    c = np.array(list(counts.values()), float)
    e = energies(x, edges, w)
    mean = float(c @ e / shots)
    var = float(c @ (e - mean) ** 2 / (shots - 1))
    span = emax - emin
    p_hw = float(c[np.abs(e - emin) < 1e-9].sum() / shots)
    rows.append(dict(index=k, id=it["id"], n=n, p=it["p"], method=method, shots=shots,
                     p_opt=p_hw, p_opt_se=(p_hw * (1 - p_hw) / shots) ** 0.5,
                     p_opt_ideal=it["ideal_probability"], p_opt_random=nopt / 2 ** n,
                     ratio=(emax - mean) / span, ratio_se=(var / shots) ** 0.5 / span,
                     ratio_ideal=it["ideal_ratio"], ratio_random=emax / span,
                     best_ratio=float((emax - e.min()) / span),
                     optimal_states=nopt, e_min=emin, e_max=emax))
jdump(out / "counts.json", counts_out)
np.savez_compressed(out / "bitstrings.npz", **arrays)
cols = list(rows[0])
with open(out / "stats.csv", "w") as f:
    f.write(",".join(cols) + "\n")
    for r in rows:
        f.write(",".join(str(r[c]) for c in cols) + "\n")
print(f"counts/bitstrings/stats: {len(rows)} 条线路")

# ---- 实际执行的编译后线路
inputs = try_(lambda: job.inputs, {}) or {}
pubs = inputs.get("pubs", [])
circs = []
for p in pubs:
    qc = getattr(p, "circuit", None)
    if qc is None:
        qc = p[0] if isinstance(p, (list, tuple)) else p
    circs.append(qc)
if circs:
    with open(out / "isa_circuits.qpy", "wb") as f:
        qpy.dump(circs, f)
    with open(out / "circuits.csv", "w") as f:
        f.write("index,id,p,method,twoq,twoq_depth,depth,physical_qubits\n")
        for k, (it, qc) in enumerate(zip(items, circs)):
            ops = qc.count_ops()
            twoq = sum(v for g, v in ops.items() if g in TWO_Q)
            d2 = qc.depth(lambda ins: ins.operation.num_qubits == 2)
            used = sorted({qc.find_bit(q).index for ins in qc.data for q in ins.qubits
                           if ins.operation.name not in ("barrier", "delay")})
            f.write(f"{k},{it['id']},{it['p']},{it['method']}_p{it['p']},{twoq},{d2},{qc.depth()},"
                    f"\"{' '.join(map(str, used))}\"\n")
    print(f"isa_circuits.qpy / circuits.csv: {len(circs)} 条")
else:
    print("  [skip] job 输入中没有取到线路")

# ---- job 元数据
meta = dict(
    job_id=st["job_id"],
    backend=try_(lambda: job.backend().name, st.get("backend")),
    status=str(job.status()),
    creation_date=try_(lambda: job.creation_date),
    tags=try_(lambda: job.tags),
    usage_seconds=try_(lambda: job.usage()),
    metrics=try_(lambda: job.metrics()),
    options={k: v for k, v in inputs.items() if k != "pubs"},
    result_metadata=try_(lambda: dict(res.metadata)),
)
jdump(out / "job_metadata.json", meta)
print("job_metadata.json")

# ---- 执行当时的校准数据
backend = try_(lambda: service.backend(meta["backend"]))
if backend is not None:
    props = try_(lambda: backend.properties(datetime=job.creation_date))
    if props is not None:
        jdump(out / "calibration.json", props.to_dict())
        print("calibration.json")
print("完成 ->", out)
