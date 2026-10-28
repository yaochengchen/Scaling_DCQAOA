#!/usr/bin/env python
"""IBM 真机：论文现行规则（U1 幂律 DC-QAOA）vs 论文冻结 QAOA 对照，等深 p=1、2。

图、角度、无噪声参考值全部来自 tqe_hw_plan.json（由 TQE_reproducibility_2026-10-27 的
results/pennylane_probability_1050/manifest.json 与 paired_probabilities_1050.csv 生成；
角度逻辑与 code/circuit_core.py::predict 相同，已核对理想最优解概率与论文数据一致）。
21 张 regular3-unit 测试图（n=10..22，每个 n 3 张，即论文 1050 图中的这一格）。

步骤（与 tqe_hw_plan.json 放在同一文件夹）：
  python ibm_tqe_hw.py submit --backend ibm_fez   # 提交
  python ibm_tqe_hw.py result                     # 结果 -> tqe_output_results.txt, tqe_hw_results.csv
  python fetch_raw_tqe_job.py                     # 原始记录 -> raw_<job_id>/
"""
import argparse, json
from pathlib import Path
import numpy as np

PLAN = Path("tqe_hw_plan.json")
STATE = Path("tqe_hw_run.json")
SHOTS = 8000


def circuit(n, edges, w, ang):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for g, b, a in zip(ang["gamma"], ang["beta"], ang["alpha"]):
        for (i, j), wij in zip(edges, w):
            qc.rzz(2 * g * wij, i, j)                       # exp(-i g J Z_i Z_j)
        if a != 0:
            for (i, j), wij in zip(edges, w):
                for y, z in ((i, j), (j, i)):               # YZ 然后 ZY：exp(-i a J Y_y Z_z)
                    qc.sdg(y); qc.h(y); qc.rzz(2 * a * wij, y, z); qc.h(y); qc.s(y)
        qc.rx(2 * b, range(n))                              # exp(-i b X)
    qc.measure_all()
    return qc


def energies(x, edges, w):
    e = np.zeros(len(x))
    for (i, j), wij in zip(edges, w):
        e += wij * (1.0 - 2.0 * (((x >> i) ^ (x >> j)) & 1))
    return e


def submit(args):
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    items = json.loads(PLAN.read_text())["items"]
    backend = QiskitRuntimeService().backend(args.backend)
    circs = [circuit(it["n"], it["edges"], it["weights"], it["angles"]) for it in items]
    isa = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=1).run(circs)
    twoq = [int(sum(v for k, v in qc.count_ops().items() if k in ("cz", "ecr", "cx"))) for qc in isa]
    print(f"{len(isa)} 条线路，平均双比特门数：")
    print("  n |  p=1 QAOA  DC(U1) |  p=2 QAOA  DC(U1)")
    for n in sorted({it["n"] for it in items}):
        cell = lambda p, m: np.mean([t for it, t in zip(items, twoq) if it["n"] == n and it["p"] == p and it["method"] == m])
        print(f" {n:2d} | {cell(1,'qaoa'):8.0f} {cell(1,'u1'):7.0f} | {cell(2,'qaoa'):8.0f} {cell(2,'u1'):7.0f}")
    if input(f"提交到 {args.backend}（{len(isa)}×{args.shots} shots，机时上限 {args.max_seconds}s）？[y/N] ") != "y":
        return
    sampler = SamplerV2(mode=backend)
    sampler.options.default_shots = args.shots
    sampler.options.max_execution_time = args.max_seconds
    sampler.options.dynamical_decoupling.enable = True
    sampler.options.twirling.enable_gates = True
    sampler.options.twirling.enable_measure = True
    job = sampler.run(isa)
    STATE.write_text(json.dumps(dict(job_id=job.job_id(), backend=args.backend, shots=args.shots, twoq=twoq)))
    print(f"已提交 job {job.job_id()}，完成后运行：python ibm_tqe_hw.py result")


def result(args):
    from qiskit_ibm_runtime import QiskitRuntimeService
    items = json.loads(PLAN.read_text())["items"]
    st = json.loads(STATE.read_text())
    job = QiskitRuntimeService().job(st["job_id"])
    status = str(job.status())
    lines = []
    out = lambda s="": (print(s), lines.append(s))
    out(f"job 状态: {status}")
    if not status.upper().endswith("DONE"):
        return

    rows, cache = [], {}
    for it, pr, tq in zip(items, job.result(), st["twoq"]):
        counts = pr.data.meas.get_counts()
        n, E, W = it["n"], it["edges"], it["weights"]
        if it["id"] not in cache:
            ev = energies(np.arange(1 << n, dtype=np.int64), E, W)
            cache = {it["id"]: (ev.min(), ev.max(), int((np.abs(ev - ev.min()) < 1e-9).sum()))}
        emin, emax, nopt = cache[it["id"]]
        assert nopt == it["optimal_states"], "最优解个数与论文数据不一致"
        x = np.array([int(k, 2) for k in counts], dtype=np.int64)
        c = np.array(list(counts.values()), float)
        e = energies(x, E, W)
        shots = c.sum()
        p_hw = float(c[np.abs(e - emin) < 1e-9].sum() / shots)
        mean = float(c @ e / shots)
        rows.append(dict(id=it["id"], n=n, p=it["p"], method=f"{it['method']}_p{it['p']}", twoq=tq,
                         shots=int(shots), p_opt_hw=p_hw, p_opt_se=(p_hw * (1 - p_hw) / shots) ** 0.5,
                         p_opt_ideal=it["ideal_probability"], p_opt_random=nopt / 2 ** n,
                         ratio_hw=(emax - mean) / (emax - emin), ratio_ideal=it["ideal_ratio"],
                         ratio_random=emax / (emax - emin)))

    cols = list(rows[0])
    with open("tqe_hw_results.csv", "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in cols) + "\n")

    m = lambda n, p, meth, k: np.mean([r[k] for r in rows if r["n"] == n and r["method"] == f"{meth}_p{p}"])
    for p in (1, 2):
        out(f"\n=== p={p}：最优解概率（%，每个 n 3 张图平均）真机 / 理想；近似比 真机 / 理想 ===")
        out("  n |  QAOA P 真机/理想  |  DC(U1) P 真机/理想 | P 比值 DC/QAOA 真机/理想 | 近似比 QAOA 真机/理想 | 近似比 DC 真机/理想 | 随机P")
        for n in sorted({r["n"] for r in rows}):
            qh, qi, dh, di = (m(n, p, "qaoa", "p_opt_hw"), m(n, p, "qaoa", "p_opt_ideal"),
                              m(n, p, "u1", "p_opt_hw"), m(n, p, "u1", "p_opt_ideal"))
            out(f" {n:2d} | {100*qh:6.3f} / {100*qi:6.3f} | {100*dh:6.3f} / {100*di:6.3f} | "
                f"{dh/qh if qh else float('nan'):6.2f} / {di/qi:5.2f} | "
                f"{m(n,p,'qaoa','ratio_hw'):.4f} / {m(n,p,'qaoa','ratio_ideal'):.4f} | "
                f"{m(n,p,'u1','ratio_hw'):.4f} / {m(n,p,'u1','ratio_ideal'):.4f} | {100*m(n,p,'qaoa','p_opt_random'):.4f}")
        by = {(r["id"], r["method"]): r for r in rows}
        ids = sorted({r["id"] for r in rows})
        dp = [by[(i, f"u1_p{p}")]["p_opt_hw"] - by[(i, f"qaoa_p{p}")]["p_opt_hw"] for i in ids]
        dr = [by[(i, f"u1_p{p}")]["ratio_hw"] - by[(i, f"qaoa_p{p}")]["ratio_hw"] for i in ids]
        out(f"  p={p} 真机 DC−QAOA：最优解概率 {100*np.mean(dp):+.3f} 个百分点（DC 更高 {sum(d > 0 for d in dp)}/{len(dp)} 张图）；"
            f"近似比 {np.mean(dr):+.4f}（DC 更高 {sum(d > 0 for d in dr)}/{len(dr)}）")
    try:
        out(f"\nQPU 用时: {job.usage()} s")
    except Exception:
        pass
    out("逐图结果 -> tqe_hw_results.csv；原始记录请运行 python fetch_raw_tqe_job.py")
    Path("tqe_output_results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit")
    s.add_argument("--backend", required=True)
    s.add_argument("--shots", type=int, default=SHOTS)
    s.add_argument("--max-seconds", type=int, default=420)
    sub.add_parser("result")
    a = ap.parse_args()
    submit(a) if a.cmd == "submit" else result(a)
