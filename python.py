# -*- coding: utf-8 -*-
"""
Light multi-objective battery pack sizing (ns/np) with 4 algorithms:
NSGA-II, NSGA-III, MOPSO, MOSOA

Constraint:
    300V <= V_pack <= 350V

Objectives (Pareto):
    Min  mass_pack
    Max  energy_kWh
    Max  power_kW  (proxy from C-rate)

Outputs:
    - Excel: Pareto + Picks (8 unique-ns) per algo & cell + Summary sheet
    - Plots: ns-np Pareto comparison per cell (all algos) + highlighted picks

Install:
    pip install pymoo numpy matplotlib pandas openpyxl
"""

import os
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1) CELL DATA (EDIT IF NEEDED)
# =========================
CELLS = [
    dict(name="IFP22290120A-96Ah", V=3.21, Ah=96.0,  mass_kg=1.73, C_cont=1.0),
    dict(name="LF100LA-100Ah",     V=3.20, Ah=100.0, mass_kg=1.95, C_cont=1.0),
    dict(name="LF280K-280Ah",      V=3.20, Ah=280.0, mass_kg=5.30, C_cont=0.5),
]

# =========================
# 2) CONSTRAINT + BOUNDS
# =========================
V_MIN, V_MAX = 300.0, 350.0      # voltage constraint (ONLY constraint)
NP_MIN, NP_MAX = 1, 20           # keep small => light
OVERHEAD = 0.15                  # pack overhead mass factor (module+busbars+housing...)

# Lightweight algorithm settings
NSGA2_POP, NSGA2_GEN = 90, 90
NSGA3_POP, NSGA3_GEN = 100, 90
MOPSO_PARTICLES, MOPSO_ITERS = 70, 130
MOSOA_POP, MOSOA_ITERS = 70, 130

PICKS_PER_ALGO = 8

OUT_DIR = "pareto_light_voltage_300_350_excel"
os.makedirs(OUT_DIR, exist_ok=True)

EXCEL_PATH = os.path.join(OUT_DIR, "battery_pareto_picks.xlsx")

SEED = 10
random.seed(SEED)
np.random.seed(SEED)

# =========================
# 3) METRICS
# =========================
def ns_window(cell):
    ns_min = math.ceil(V_MIN / cell["V"])
    ns_max = math.floor(V_MAX / cell["V"])
    return int(ns_min), int(ns_max)

def pack_metrics(cell, ns, np_):
    V_pack = ns * cell["V"]
    Ah_pack = np_ * cell["Ah"]
    E_kWh = (V_pack * Ah_pack) / 1000.0

    n_cells = ns * np_
    mass = (n_cells * cell["mass_kg"]) * (1.0 + OVERHEAD)

    # power proxy: P = V * I ; I = C_cont * Ah_pack
    I_cont = cell["C_cont"] * Ah_pack
    P_kW = (V_pack * I_cont) / 1000.0

    return dict(ns=int(ns), np=int(np_), V=float(V_pack), E=float(E_kWh), P=float(P_kW), mass=float(mass))

def objectives(m):
    # minimize mass, maximize energy, maximize power
    # convert max to min => negative
    return np.array([m["mass"], -m["E"], -m["P"]], dtype=float)

def dominates(a, b):
    return np.all(a <= b) and np.any(a < b)

def pareto_filter(sol_list):
    """sol_list: list of dicts {"M": metrics, "F": objectives}"""
    if not sol_list:
        return []
    nd = []
    for s in sol_list:
        dom = False
        for t in sol_list:
            if t is s:
                continue
            if dominates(t["F"], s["F"]):
                dom = True
                break
        if not dom:
            nd.append(s)

    # dedup by (ns,np)
    seen = set()
    uniq = []
    for s in nd:
        key = (s["M"]["ns"], s["M"]["np"])
        if key not in seen:
            seen.add(key)
            uniq.append(s)
    return uniq

def clamp_int(x, lo, hi):
    return int(max(lo, min(hi, int(round(float(x))))))

def pick_k_unique_ns(front, k=8):
    """
    Pick up to k solutions with different ns (spread across ns range),
    keeping best (min mass) per ns.
    """
    if not front:
        return []

    by_ns = {}
    for s in front:
        ns = s["M"]["ns"]
        if ns not in by_ns or s["M"]["mass"] < by_ns[ns]["M"]["mass"]:
            by_ns[ns] = s

    uniq = sorted(by_ns.values(), key=lambda s: s["M"]["ns"])
    if len(uniq) <= k:
        return uniq

    idxs = np.linspace(0, len(uniq)-1, k).round().astype(int)
    picked = [uniq[i] for i in idxs]

    out, seen = [], set()
    for s in picked:
        ns = s["M"]["ns"]
        if ns not in seen:
            out.append(s)
            seen.add(ns)

    if len(out) < k:
        for s in uniq:
            ns = s["M"]["ns"]
            if ns not in seen:
                out.append(s)
                seen.add(ns)
            if len(out) == k:
                break

    return out

# =========================
# 4) NSGA2 / NSGA3 (pymoo) - light
# =========================
def run_nsga2(cell, seed=1):
    from pymoo.core.problem import Problem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM

    ns_min, ns_max = ns_window(cell)
    xl = np.array([ns_min, NP_MIN], dtype=float)
    xu = np.array([ns_max, NP_MAX], dtype=float)

    class P(Problem):
        def __init__(self):
            super().__init__(n_var=2, n_obj=3, n_ieq_constr=0, xl=xl, xu=xu, elementwise=False)

        def _evaluate(self, X, out, *args, **kwargs):
            F = []
            for x in X:
                ns = clamp_int(x[0], ns_min, ns_max)
                np_ = clamp_int(x[1], NP_MIN, NP_MAX)
                m = pack_metrics(cell, ns, np_)
                F.append(objectives(m))
            out["F"] = np.array(F, dtype=float)

    algo = NSGA2(
        pop_size=NSGA2_POP,
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )

    res = minimize(P(), algo, termination=get_termination("n_gen", NSGA2_GEN), seed=seed, verbose=False)
    if res.X is None:
        return []

    sols = []
    for x in res.X:
        ns = clamp_int(x[0], ns_min, ns_max)
        np_ = clamp_int(x[1], NP_MIN, NP_MAX)
        m = pack_metrics(cell, ns, np_)
        sols.append(dict(M=m, F=objectives(m)))
    return pareto_filter(sols)

def run_nsga3(cell, seed=2):
    from pymoo.core.problem import Problem
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination
    from pymoo.util.ref_dirs import get_reference_directions
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM

    ns_min, ns_max = ns_window(cell)
    xl = np.array([ns_min, NP_MIN], dtype=float)
    xu = np.array([ns_max, NP_MAX], dtype=float)

    # fewer ref dirs => faster
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=8)
    pop_size = max(NSGA3_POP, len(ref_dirs))

    class P(Problem):
        def __init__(self):
            super().__init__(n_var=2, n_obj=3, n_ieq_constr=0, xl=xl, xu=xu, elementwise=False)

        def _evaluate(self, X, out, *args, **kwargs):
            F = []
            for x in X:
                ns = clamp_int(x[0], ns_min, ns_max)
                np_ = clamp_int(x[1], NP_MIN, NP_MAX)
                m = pack_metrics(cell, ns, np_)
                F.append(objectives(m))
            out["F"] = np.array(F, dtype=float)

    algo = NSGA3(
        pop_size=pop_size,
        ref_dirs=ref_dirs,
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )

    res = minimize(P(), algo, termination=get_termination("n_gen", NSGA3_GEN), seed=seed, verbose=False)
    if res.X is None:
        return []

    sols = []
    for x in res.X:
        ns = clamp_int(x[0], ns_min, ns_max)
        np_ = clamp_int(x[1], NP_MIN, NP_MAX)
        m = pack_metrics(cell, ns, np_)
        sols.append(dict(M=m, F=objectives(m)))
    return pareto_filter(sols)

# =========================
# 5) MOPSO / MOSOA (simple, light)
# =========================
def random_feasible(cell):
    ns_min, ns_max = ns_window(cell)
    ns = random.randint(ns_min, ns_max)
    np_ = random.randint(NP_MIN, NP_MAX)
    return ns, np_

def run_mopso(cell, seed=3):
    random.seed(seed)
    np.random.seed(seed)
    ns_min, ns_max = ns_window(cell)

    particles = []
    for _ in range(MOPSO_PARTICLES):
        ns, np_ = random_feasible(cell)
        m = pack_metrics(cell, ns, np_)
        particles.append(dict(x=np.array([ns, np_], float), v=np.zeros(2), pbest=m, pF=objectives(m)))

    archive = pareto_filter([dict(M=p["pbest"], F=p["pF"]) for p in particles])

    w, c1, c2 = 0.6, 1.6, 1.6

    def pick_leader():
        return random.choice(archive)["M"] if archive else random.choice([p["pbest"] for p in particles])

    for _ in range(MOPSO_ITERS):
        leader = pick_leader()
        L = np.array([leader["ns"], leader["np"]], float)

        for p in particles:
            r1 = np.random.rand(2)
            r2 = np.random.rand(2)
            PB = np.array([p["pbest"]["ns"], p["pbest"]["np"]], float)

            p["v"] = w*p["v"] + c1*r1*(PB - p["x"]) + c2*r2*(L - p["x"])
            p["x"] = p["x"] + p["v"]

            ns = clamp_int(p["x"][0], ns_min, ns_max)
            np_ = clamp_int(p["x"][1], NP_MIN, NP_MAX)

            m = pack_metrics(cell, ns, np_)
            F = objectives(m)

            if dominates(F, p["pF"]):
                p["pbest"] = m
                p["pF"] = F

        archive = pareto_filter(archive + [dict(M=p["pbest"], F=p["pF"]) for p in particles])

    return archive

def run_mosoa(cell, seed=4):
    random.seed(seed)
    np.random.seed(seed)
    ns_min, ns_max = ns_window(cell)

    pop = []
    for _ in range(MOSOA_POP):
        ns, np_ = random_feasible(cell)
        m = pack_metrics(cell, ns, np_)
        pop.append(dict(M=m, F=objectives(m)))

    archive = pareto_filter(pop)

    for t in range(1, MOSOA_ITERS+1):
        if not archive:
            archive = pareto_filter(pop)

        leader = random.choice(archive)["M"]
        L = np.array([leader["ns"], leader["np"]], float)
        A = 2.0*(1.0 - t/max(1, MOSOA_ITERS))

        new_pop = []
        for s in pop:
            x = np.array([s["M"]["ns"], s["M"]["np"]], float)
            B = np.random.uniform(-A, A, size=2)
            x1 = x + B*(L - x)

            theta = np.random.uniform(0, 2*math.pi)
            r = np.random.uniform(0, 1)
            spiral = r*np.array([math.cos(theta), math.sin(theta)], float)
            x2 = L + spiral*(x1 - L)

            ns = clamp_int(x2[0], ns_min, ns_max)
            np_ = clamp_int(x2[1], NP_MIN, NP_MAX)

            m = pack_metrics(cell, ns, np_)
            new_pop.append(dict(M=m, F=objectives(m)))

        pop = new_pop
        archive = pareto_filter(archive + pop)

    return archive

# =========================
# 6) PLOTS (ns,np comparison per cell)
# =========================
def plot_ns_np(cell_name, fronts, picks_by_algo):
    plt.figure(figsize=(7.4, 5.4))
    markers = {"NSGA2":"o","NSGA3":"s","MOPSO":"^","MOSOA":"*"}
    for algo, front in fronts.items():
        xs = [s["M"]["ns"] for s in front]
        ys = [s["M"]["np"] for s in front]
        plt.scatter(xs, ys, s=22, marker=markers.get(algo,"o"), label=f"{algo} Pareto (n={len(front)})")

        # highlight picks
        picks = picks_by_algo.get(algo, [])
        if picks:
            px = [s["M"]["ns"] for s in picks]
            py = [s["M"]["np"] for s in picks]
            plt.scatter(px, py, s=110, facecolors="none", edgecolors="k", linewidths=1.2)

    plt.xlabel("ns")
    plt.ylabel("np")
    plt.title(f"{cell_name} | Pareto ns vs np\nConstraint: V in [{V_MIN},{V_MAX}]")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(OUT_DIR, f"{cell_name}_ns_np_compare.png".replace("/","_"))
    plt.savefig(fname, dpi=220)
    plt.show()
    print("Saved plot ->", fname)

# =========================
# 7) EXCEL HELPERS
# =========================
def front_to_df(front, algo, cell_name):
    rows = []
    for s in front:
        m = s["M"]
        rows.append(dict(
            algo=algo,
            cell=cell_name,
            ns=m["ns"], np=m["np"],
            V_pack=m["V"], energy_kWh=m["E"], power_kW=m["P"], mass_kg=m["mass"]
        ))
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["ns", "np"]).reset_index(drop=True)
    return df

def safe_sheet_name(name: str) -> str:
    # Excel sheet name limit 31 chars
    return name[:31]

# =========================
# 8) RUN
# =========================
def main():
    summary_pick_rows = []

    with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl") as writer:
        for cell in CELLS:
            ns_min, ns_max = ns_window(cell)
            print(f"\n{cell['name']} | ns range for voltage: [{ns_min},{ns_max}]")

            f2 = run_nsga2(cell, seed=SEED)
            f3 = run_nsga3(cell, seed=SEED+1)
            fp = run_mopso(cell, seed=SEED+2)
            fa = run_mosoa(cell, seed=SEED+3)

            fronts = {"NSGA2": f2, "NSGA3": f3, "MOPSO": fp, "MOSOA": fa}

            picks_by_algo = {}
            for algo, front in fronts.items():
                picks = pick_k_unique_ns(front, k=PICKS_PER_ALGO)
                picks_by_algo[algo] = picks

                print(f"\n  {algo} -> {len(picks)} diverse designs (unique ns):")
                if not picks:
                    print("    (no solutions)")
                    continue

                for s in picks:
                    m = s["M"]
                    print(f"    ns={m['ns']}, np={m['np']} | V={m['V']:.1f}V | "
                          f"E={m['E']:.2f}kWh | P={m['P']:.1f}kW | mass={m['mass']:.1f}kg")
                    summary_pick_rows.append(dict(
                        algo=algo, cell=cell["name"],
                        ns=m["ns"], np=m["np"],
                        V_pack=m["V"], energy_kWh=m["E"], power_kW=m["P"], mass_kg=m["mass"]
                    ))

                # write PICKS sheet
                df_picks = front_to_df(picks, algo, cell["name"])
                sheet_picks = safe_sheet_name(f"PICKS_{algo}_{cell['name']}")
                df_picks.to_excel(writer, sheet_name=sheet_picks, index=False)

                # write PARETO sheet
                df_front = front_to_df(front, algo, cell["name"])
                sheet_front = safe_sheet_name(f"PARETO_{algo}_{cell['name']}")
                df_front.to_excel(writer, sheet_name=sheet_front, index=False)

            # plots per cell
            plot_ns_np(cell["name"], fronts, picks_by_algo)

        # Summary sheet
        df_sum = pd.DataFrame(summary_pick_rows)
        if not df_sum.empty:
            df_sum = df_sum.sort_values(["cell", "algo", "ns"]).reset_index(drop=True)
        df_sum.to_excel(writer, sheet_name="SUMMARY_PICKS", index=False)

    print("\nExcel saved ->", EXCEL_PATH)
    print("All outputs ->", OUT_DIR)

if __name__ == "__main__":
    main()