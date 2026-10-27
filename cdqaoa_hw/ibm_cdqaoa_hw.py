#!/usr/bin/env python
"""IBM 真机：等深比较 CD-QAOA p=2 vs QAOA p=2（固定公式参数，与论文配置相同，不做优化）。

两步：
  python ibm_cdqaoa_hw.py submit --backend ibm_fez     # 生成图+角度+线路，转译，提交一个 job
  python ibm_cdqaoa_hw.py result                       # 取回结果，与无噪声理想值对比

图：GPU 测试集 test-n{n}-regular3-unit-000..003（base seed 20261013），n=10..22。
角度：CD = 统一角度表 (unified_gpu_model.json)；QAOA = 论文固定基线 qaoa_csv (sign -1)。
门序与 cdtransfer/simulator.py 相同。
"""
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np

N_LIST = [10, 12, 14, 16, 18, 20, 22]
PER_N = 4
P = 2
SHOTS = 4000
BASE_SEED = 20261013
STATE_FILE = Path("hw_run.json")

# 统一角度表 p=2（全精度）与论文 QAOA 固定基线 p=2
CD_P2 = dict(gamma_unit=[-0.40852793167186546, -0.8676245822864975],
             beta=[0.6135485151877298, 0.20640819505269356],
             alpha_unit=[-0.1200255745655971, -0.025438229216823774])
D0 = 3.4545454545454546
QAOA_P2 = dict(gamma=[0.3817, 0.6655], beta=[0.496, 0.269])


def graph(n, j):
    """与 cdtransfer.instances.make_instance(n, 'regular3', 'unit', seed) 相同。"""
    import networkx as nx
    token = f"{BASE_SEED}:2:{n}:regular3:unit:{j}".encode()          # 2 = test split
    seed = int.from_bytes(hashlib.sha256(token).digest()[:4], "big")
    g = nx.random_regular_graph(3, n, seed=seed)
    perm = np.random.default_rng(seed).permutation(n)
    edges = sorted(tuple(sorted((int(perm[a]), int(perm[b])))) for a, b in g.edges())
    return [list(e) for e in edges], [1.0] * len(edges)


def angles(kind, n, edges, w):
    d = 2 * len(edges) / n
    s = math.sqrt(np.mean(np.square(w)))
    fd = math.atan(1 / math.sqrt(d - 1)) if d > 1 else 1.0
    if kind == "cd":
        return dict(gamma=[g * fd / s for g in CD_P2["gamma_unit"]], beta=CD_P2["beta"],
                    alpha=[a / s * (d / D0) ** -0.5 for a in CD_P2["alpha_unit"]])
    return dict(gamma=[-g * fd / s for g in QAOA_P2["gamma"]], beta=QAOA_P2["beta"], alpha=[0.0] * P)


def circuit(n, edges, w, ang):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for g, b, a in zip(ang["gamma"], ang["beta"], ang["alpha"]):
        for (i, j), wij in zip(edges, w):
            qc.rzz(2 * g * wij, i, j)                       # exp(-i g J ZZ)
        if a != 0:
            for (i, j), wij in zip(edges, w):
                for y, z in ((i, j), (j, i)):               # exp(-i a J Y_y Z_z)
                    qc.sdg(y); qc.h(y); qc.rzz(2 * a * wij, y, z); qc.h(y); qc.s(y)
        qc.rx(2 * b, range(n))                              # exp(-i b X)
    qc.measure_all()
    return qc


def energies(x, edges, w):
    e = np.zeros(len(x))
    for (i, j), wij in zip(edges, w):
        e += wij * (1.0 - 2.0 * (((x >> i) ^ (x >> j)) & 1))
    return e


def ideal_energy(n, edges, w, ang, ev):
    """精确态矢量（比特 i = 量子比特 i），返回 <H_C>。"""
    psi = np.full(1 << n, (1 << n) ** -0.5, dtype=complex)
    t = psi.reshape((2,) * n)                                # 轴 k <-> 比特 n-1-k
    sl = lambda ax, v: tuple(slice(v, v + 1) if k == ax else slice(None) for k in range(n))
    for g, b, a in zip(ang["gamma"], ang["beta"], ang["alpha"]):
        psi *= np.exp(-1j * g * ev)
        if a != 0:
            for (i, j), wij in zip(edges, w):
                for y, z in ((i, j), (j, i)):
                    ay, az = n - 1 - y, n - 1 - z
                    zs = np.array([1.0, -1.0]).reshape([2 if k == az else 1 for k in range(n)])
                    c, s = math.cos(a * wij), math.sin(a * wij)
                    x0, x1 = t[sl(ay, 0)], t[sl(ay, 1)]
                    t[sl(ay, 0)], t[sl(ay, 1)] = c * x0 - s * zs * x1, s * zs * x0 + c * x1
        c, s = math.cos(b), math.sin(b)
        for q in range(n):
            ax = n - 1 - q
            x0, x1 = t[sl(ax, 0)], t[sl(ax, 1)]
            t[sl(ax, 0)], t[sl(ax, 1)] = c * x0 - 1j * s * x1, -1j * s * x0 + c * x1
    return float(np.abs(psi) ** 2 @ ev)


def submit(args):
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    backend = QiskitRuntimeService().backend(args.backend)
    items, circs = [], []
    for n in N_LIST:
        for j in range(PER_N):
            edges, w = graph(n, j)
            for kind in ("qaoa", "cd"):
                ang = angles(kind, n, edges, w)
                items.append(dict(id=f"test-n{n}-regular3-unit-{j:03d}", n=n, kind=kind,
                                  edges=edges, w=w, angles=ang))
                circs.append(circuit(n, edges, w, ang))
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=1)
    isa = pm.run(circs)
    for it, qc in zip(items, isa):
        it["twoq"] = int(sum(v for k, v in qc.count_ops().items() if k in ("cz", "ecr", "cx")))
    print(f"{len(isa)} 条线路；平均双比特门数：")
    for n in N_LIST:
        q = np.mean([i["twoq"] for i in items if i["n"] == n and i["kind"] == "qaoa"])
        c = np.mean([i["twoq"] for i in items if i["n"] == n and i["kind"] == "cd"])
        print(f"  n={n:2d}  QAOA p2 {q:5.0f}   CD p2 {c:5.0f}")
    if input(f"提交到 {args.backend}（{len(isa)}×{SHOTS} shots，机时上限 {args.max_seconds}s）？[y/N] ") != "y":
        return
    sampler = SamplerV2(mode=backend)
    sampler.options.default_shots = SHOTS
    sampler.options.max_execution_time = args.max_seconds
    sampler.options.dynamical_decoupling.enable = True
    sampler.options.twirling.enable_gates = True
    sampler.options.twirling.enable_measure = True
    job = sampler.run(isa)
    STATE_FILE.write_text(json.dumps(dict(job_id=job.job_id(), backend=args.backend, items=items)))
    print(f"已提交 job {job.job_id()}，完成后运行：python ibm_cdqaoa_hw.py result")


def result(args):
    from qiskit_ibm_runtime import QiskitRuntimeService
    st = json.loads(STATE_FILE.read_text())
    job = QiskitRuntimeService().job(st["job_id"])
    status = job.status()
    print("job 状态:", status)
    if str(status).upper().split(".")[-1] != "DONE":
        return
    rows = []
    cache = {}
    for it, pr in zip(st["items"], job.result()):
        n, edges, w = it["n"], it["edges"], it["w"]
        if it["id"] not in cache:
            ev = energies(np.arange(1 << n, dtype=np.int64), edges, w)
            cache = {it["id"]: (ev, ev.min(), ev.max())}
        ev, emin, emax = cache[it["id"]]
        counts = pr.data.meas.get_counts()
        x = np.array([int(k, 2) for k in counts], dtype=np.int64)
        c = np.array(list(counts.values()), float)
        e_hw = float(c @ energies(x, edges, w) / c.sum())
        e_id = ideal_energy(n, edges, w, it["angles"], ev)
        rows.append(dict(id=it["id"], n=n, kind=it["kind"], twoq=it["twoq"],
                         ratio_hw=(emax - e_hw) / (emax - emin), ratio_ideal=(emax - e_id) / (emax - emin),
                         ratio_random=emax / (emax - emin)))
    with open("hw_results.csv", "w") as f:
        f.write("id,n,method,twoq,ratio_hw,ratio_ideal,ratio_random\n")
        for r in rows:
            f.write(f"{r['id']},{r['n']},{r['kind']}_p2,{r['twoq']},{r['ratio_hw']:.6f},"
                    f"{r['ratio_ideal']:.6f},{r['ratio_random']:.6f}\n")
    m = lambda n, k, key: np.mean([r[key] for r in rows if r["n"] == n and r["kind"] == k])
    print("\n近似比（每个 n 平均 4 张图）")
    print(" n | QAOA p2 真机 / 理想 | CD p2 真机 / 理想 | CD−QAOA 真机 / 理想 | 随机")
    for n in N_LIST:
        q_hw, q_id, c_hw, c_id = m(n, "qaoa", "ratio_hw"), m(n, "qaoa", "ratio_ideal"), \
                                 m(n, "cd", "ratio_hw"), m(n, "cd", "ratio_ideal")
        print(f"{n:2d} | {q_hw:.4f} / {q_id:.4f} | {c_hw:.4f} / {c_id:.4f} | "
              f"{c_hw - q_hw:+.4f} / {c_id - q_id:+.4f} | {m(n, 'qaoa', 'ratio_random'):.4f}")
    try:
        print(f"\nQPU 用时: {job.usage()} s")
    except Exception:
        pass
    print("逐图结果 -> hw_results.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit")
    s.add_argument("--backend", required=True)
    s.add_argument("--max-seconds", type=int, default=300)
    sub.add_parser("result")
    a = ap.parse_args()
    submit(a) if a.cmd == "submit" else result(a)
