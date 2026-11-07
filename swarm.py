"""
Fair comparison of evolutionary algorithms on a noisy quadratic.

Two tracks, because "fit 3 numbers" and "discover an expression" are
different problems and should not be mixed in one table:

  Track A - PARAMETRIC FIT   : y = a*x^2 + b*x + c
      GA vs DE vs Random Search (+ np.polyfit as a non-budgeted reference)

  Track B - SYMBOLIC REGRESSION : same grammar, same genotype->phenotype map
      GE vs DE-on-genome (grammatical DE) vs Random Search

Fairness measures
  1. Equal budget       : every optimizer gets exactly N fitness evaluations
                          (enforced by the Objective wrapper, not by generations).
  2. Many seeds         : N_SEEDS independent runs, report median / IQR / mean.
  3. Independent RNGs   : each (seed, algorithm) pair has its own Generator.
  4. Same data per seed : all algorithms see the identical noisy training set.
  5. Pure MSE fitness   : no hidden penalty terms; curves are best-so-far MSE.
  6. Held-out testing   : error vs the CLEAN target, inside AND outside the
                          training range (extrapolation).
  7. Baselines          : random search (and polyfit) show what is "easy".
  8. Statistics         : pairwise Wilcoxon signed-rank tests across seeds.
"""

import argparse
import csv
import itertools
import os
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless-safe; figures are saved to disk
import matplotlib.pyplot as plt

try:
    from scipy.stats import wilcoxon
except ImportError:  # scipy optional
    wilcoxon = None

# =====================================================
# 1. CONFIGURATION
# =====================================================

N_SEEDS = 30
BUDGET_A = 5_000       # fitness evaluations per run, Track A
BUDGET_B = 10_000      # fitness evaluations per run, Track B
NOISE_STD = 0.5
FAIL_MSE = 1e12

# Search bounds for Track A. Deliberately wide and NOT centred on the
# true parameters (3, -2, 5) so the problem is not trivially easy.
A_LO, A_HI = -50.0, 50.0

# Grammar for Track B
MAX_DEPTH = 4
CONSTS = [1, 2, 3, 4, 5]
GENOME_LEN = 50
CODON_MAX = 255


def target_function(x):
    return 3 * x**2 - 2 * x + 5


# Held-out evaluation grids (always compared with the CLEAN target)
X_TRAIN = np.linspace(-5, 5, 81)
X_TEST_IN = np.linspace(-5, 5, 401)
X_TEST_OUT = np.concatenate([np.linspace(-10, -5.05, 100), np.linspace(5.05, 10, 100)])


def mse(y_true, y_pred):
    y_pred = np.asarray(y_pred, dtype=float)
    if not np.all(np.isfinite(y_pred)):
        return FAIL_MSE
    return float(min(np.mean((y_true - y_pred) ** 2), FAIL_MSE))


# =====================================================
# 2. BUDGETED OBJECTIVE
# =====================================================

class BudgetExhausted(Exception):
    pass


class Objective:
    """Wraps a fitness function; counts evaluations and records best-so-far."""

    def __init__(self, fn, budget):
        self.fn = fn
        self.budget = budget
        self.n = 0
        self.best = np.inf
        self.best_x = None
        self.curve = np.full(budget, np.nan)

    def __call__(self, x):
        if self.n >= self.budget:
            raise BudgetExhausted
        f = self.fn(x)
        if f < self.best:
            self.best, self.best_x = f, np.array(x, copy=True)
        self.curve[self.n] = self.best
        self.n += 1
        return f


# =====================================================
# 3. TRACK A OPTIMIZERS (parametric: x = [a, b, c])
# =====================================================

def tournament(rng, fit, k=3):
    idx = rng.choice(len(fit), k, replace=False)
    return idx[np.argmin(fit[idx])]


def ga_real(obj, rng, pop_size=50):
    """Real-coded GA: tournament, BLX-0.5 crossover, Gaussian mutation, elitism."""
    sigma = 0.01 * (A_HI - A_LO)
    try:
        P = rng.uniform(A_LO, A_HI, (pop_size, 3))
        f = np.array([obj(p) for p in P])
        while True:
            new_P, new_f = [P[f.argmin()].copy()], [f.min()]
            while len(new_P) < pop_size:
                p1, p2 = P[tournament(rng, f)], P[tournament(rng, f)]
                d = np.abs(p1 - p2)
                lo, hi = np.minimum(p1, p2) - 0.5 * d, np.maximum(p1, p2) + 0.5 * d
                child = rng.uniform(lo, hi)
                mask = rng.random(3) < 0.3
                child = child + mask * rng.normal(0, sigma, 3)
                child = np.clip(child, A_LO, A_HI)
                new_P.append(child)
                new_f.append(obj(child))
            P, f = np.array(new_P), np.array(new_f)
    except BudgetExhausted:
        pass
    return obj.best_x


def de_real(obj, rng, pop_size=50, F=0.7, CR=0.9):
    """DE/rand/1/bin with cached fitness (one evaluation per trial vector)."""
    try:
        P = rng.uniform(A_LO, A_HI, (pop_size, 3))
        f = np.array([obj(p) for p in P])
        while True:
            for i in range(pop_size):
                cand = [j for j in range(pop_size) if j != i]
                r1, r2, r3 = rng.choice(cand, 3, replace=False)
                mutant = np.clip(P[r1] + F * (P[r2] - P[r3]), A_LO, A_HI)
                cross = rng.random(3) < CR
                cross[rng.integers(3)] = True
                trial = np.where(cross, mutant, P[i])
                ft = obj(trial)
                if ft <= f[i]:
                    P[i], f[i] = trial, ft
    except BudgetExhausted:
        pass
    return obj.best_x


def random_search_real(obj, rng):
    try:
        while True:
            obj(rng.uniform(A_LO, A_HI, 3))
    except BudgetExhausted:
        pass
    return obj.best_x


# =====================================================
# 4. TRACK B: SHARED GRAMMAR (genotype -> expression tree)
# =====================================================
# <expr>  ::= (<expr> + <expr>) | (<expr> - <expr>) | (<expr> * <expr>) | x | <const>
# <const> ::= 1 | 2 | 3 | 4 | 5

OPS = ("+", "-", "*")


def decode(genome, max_depth=MAX_DEPTH):
    n = len(genome)
    pos = 0

    def nxt():
        nonlocal pos
        v = int(genome[pos % n])
        pos += 1
        return v

    def expand(depth):
        choice = 3 + nxt() % 2 if depth == 0 else nxt() % 5
        if choice < 3:
            left = expand(depth - 1)
            right = expand(depth - 1)
            return (OPS[choice], left, right)
        if choice == 3:
            return "x"
        return float(CONSTS[nxt() % len(CONSTS)])

    return expand(max_depth)


def eval_tree(t, x):
    """Safe recursive evaluation (no eval())."""
    if isinstance(t, tuple):
        op, l, r = t
        a, b = eval_tree(l, x), eval_tree(r, x)
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        return a * b
    if t == "x":
        return x
    return t


def predict_tree(t, x):
    return np.broadcast_to(np.asarray(eval_tree(t, x), dtype=float), np.shape(x)).copy()


def tree_str(t):
    if isinstance(t, tuple):
        return f"({tree_str(t[1])} {t[0]} {tree_str(t[2])})"
    return "x" if t == "x" else f"{t:g}"


def tree_size(t):
    return 1 + tree_size(t[1]) + tree_size(t[2]) if isinstance(t, tuple) else 1


def ge_standard(obj, rng, pop_size=100, p_cx=0.9, p_mut=0.02):
    """Grammatical Evolution: int genomes, tournament, one-point crossover,
    codon mutation, elitism."""
    try:
        P = rng.integers(0, CODON_MAX + 1, (pop_size, GENOME_LEN))
        f = np.array([obj(g) for g in P])
        while True:
            new_P, new_f = [P[f.argmin()].copy()], [f.min()]
            while len(new_P) < pop_size:
                p1, p2 = P[tournament(rng, f)], P[tournament(rng, f)]
                child = p1.copy()
                if rng.random() < p_cx:
                    pt = rng.integers(1, GENOME_LEN)
                    child = np.concatenate([p1[:pt], p2[pt:]])
                m = rng.random(GENOME_LEN) < p_mut
                child[m] = rng.integers(0, CODON_MAX + 1, m.sum())
                new_P.append(child)
                new_f.append(obj(child))
            P, f = np.array(new_P), np.array(new_f)
    except BudgetExhausted:
        pass
    return obj.best_x


def ge_de(obj, rng, pop_size=100, F=0.7, CR=0.9):
    """Grammatical DE: real vectors in [0, 256) are floored to codons, then
    mapped through the SAME grammar as ge_standard."""
    hi = CODON_MAX + 0.999
    try:
        P = rng.uniform(0, hi, (pop_size, GENOME_LEN))
        f = np.array([obj(p) for p in P])
        while True:
            for i in range(pop_size):
                cand = [j for j in range(pop_size) if j != i]
                r1, r2, r3 = rng.choice(cand, 3, replace=False)
                mutant = np.clip(P[r1] + F * (P[r2] - P[r3]), 0, hi)
                cross = rng.random(GENOME_LEN) < CR
                cross[rng.integers(GENOME_LEN)] = True
                trial = np.where(cross, mutant, P[i])
                ft = obj(trial)
                if ft <= f[i]:
                    P[i], f[i] = trial, ft
    except BudgetExhausted:
        pass
    return obj.best_x


def random_search_ge(obj, rng):
    try:
        while True:
            obj(rng.integers(0, CODON_MAX + 1, GENOME_LEN))
    except BudgetExhausted:
        pass
    return obj.best_x


# =====================================================
# 5. RUNNERS
# =====================================================

def make_data(seed):
    rng = np.random.default_rng([seed, 999])  # same data for every algorithm
    y = target_function(X_TRAIN) + rng.normal(0, NOISE_STD, X_TRAIN.shape)
    return y


def poly_predict(p, x):
    a, b, c = p
    return a * x**2 + b * x + c


def evaluate_solution(predict, solution):
    return (
        mse(target_function(X_TEST_IN), predict(solution, X_TEST_IN)),
        mse(target_function(X_TEST_OUT), predict(solution, X_TEST_OUT)),
    )


def run_one(track, name, fn, seed, algo_id, y_train, budget):
    rng = np.random.default_rng([seed, algo_id])
    if track == "A":
        fitness = lambda p: mse(y_train, poly_predict(p, X_TRAIN))
        predict = poly_predict
        describe = lambda p: f"y = {p[0]:.4f}x^2 + ({p[1]:.4f})x + ({p[2]:.4f})"
        size = lambda p: 3
    else:
        fitness = lambda g: mse(y_train, predict_tree(decode(np.floor(g).astype(int)), X_TRAIN))
        predict = lambda g, x: predict_tree(decode(np.floor(g).astype(int)), x)
        describe = lambda g: "y = " + tree_str(decode(np.floor(g).astype(int)))
        size = lambda g: tree_size(decode(np.floor(g).astype(int)))

    obj = Objective(fitness, budget)
    t0 = time.perf_counter()
    best = fn(obj, rng)
    elapsed = time.perf_counter() - t0
    test_in, test_out = evaluate_solution(predict, best)
    return {
        "track": track, "algorithm": name, "seed": seed,
        "evals": obj.n, "train_mse": obj.best,
        "test_in_mse": test_in, "test_out_mse": test_out,
        "time_s": elapsed, "size": size(best), "equation": describe(best),
        "curve": obj.curve, "best": best,
    }


def run_polyfit(seed, y_train):
    p = np.polyfit(X_TRAIN, y_train, 2)
    t_in, t_out = evaluate_solution(poly_predict, p)
    return {
        "track": "A", "algorithm": "PolyFit (ref)", "seed": seed, "evals": 0,
        "train_mse": mse(y_train, poly_predict(p, X_TRAIN)),
        "test_in_mse": t_in, "test_out_mse": t_out, "time_s": 0.0, "size": 3,
        "equation": f"y = {p[0]:.4f}x^2 + ({p[1]:.4f})x + ({p[2]:.4f})",
        "curve": None, "best": p,
    }


# =====================================================
# 6. REPORTING
# =====================================================

def fmt(v):
    return f"{v:.3e}"


def summarize(rows, track):
    algos = []
    for r in rows:
        if r["track"] == track and r["algorithm"] not in algos:
            algos.append(r["algorithm"])

    print("\n" + "=" * 100)
    print(f"TRACK {track} SUMMARY  (median [Q1, Q3] over seeds; test errors vs CLEAN target)")
    print("=" * 124)
    print(f"{'Algorithm':<16}{'Train MSE':<36}{'Test MSE (in-range)':<36}{'Test MSE (extrapolate)':<36}")
    print("-" * 124)
    for a in algos:
        sub = [r for r in rows if r["track"] == track and r["algorithm"] == a]
        cells = []
        for key in ("train_mse", "test_in_mse", "test_out_mse"):
            v = np.array([r[key] for r in sub])
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            cells.append(f"{fmt(med)} [{fmt(q1)},{fmt(q3)}]")
        print(f"{a:<16}{cells[0]:<36}{cells[1]:<36}{cells[2]:<36}")

    print("-" * 124)
    for a in algos:
        sub = [r for r in rows if r["track"] == track and r["algorithm"] == a]
        solved = sum(r["test_in_mse"] < 1e-6 for r in sub)
        t = np.mean([r["time_s"] for r in sub])
        print(f"{a:<16} solved {solved}/{len(sub)} runs | mean time {t:.2f}s | "
              f"median size {np.median([r['size'] for r in sub]):.0f}")
    print("  (solved = in-range test MSE vs clean target < 1e-6)")

    if wilcoxon is not None:
        print("\nPairwise Wilcoxon signed-rank on in-range TEST MSE (paired by seed):")
        for a, b in itertools.combinations(algos, 2):
            va = np.array([r["test_in_mse"] for r in rows if r["track"] == track and r["algorithm"] == a])
            vb = np.array([r["test_in_mse"] for r in rows if r["track"] == track and r["algorithm"] == b])
            if np.allclose(va, vb):
                print(f"  {a} vs {b}: identical")
                continue
            try:
                p = wilcoxon(va, vb).pvalue
                winner = a if np.median(va) < np.median(vb) else b
                print(f"  {a:<14} vs {b:<14} p = {p:.2e}  (lower median: {winner})")
            except ValueError:
                print(f"  {a} vs {b}: test not applicable")
    else:
        print("\n(scipy not installed - skipping Wilcoxon tests)")


def save_csvs(rows, outdir):
    path = os.path.join(outdir, "runs.csv")
    keys = ["track", "algorithm", "seed", "evals", "train_mse", "test_in_mse",
            "test_out_mse", "time_s", "size", "equation"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nPer-run results saved to {path}")


def plot_track(rows, track, budget, outdir, title):
    algos = []
    for r in rows:
        if r["track"] == track and r["algorithm"] not in algos:
            algos.append(r["algorithm"])

    # --- convergence (best-so-far train MSE vs evaluations) ---
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = np.arange(1, budget + 1)
    for a in algos:
        curves = [r["curve"] for r in rows if r["track"] == track and r["algorithm"] == a
                  and r["curve"] is not None]
        if not curves:
            continue
        C = np.maximum(np.array(curves), 1e-12)
        med = np.median(C, axis=0)
        q1, q3 = np.percentile(C, [25, 75], axis=0)
        ax.plot(xs, med, label=a)
        ax.fill_between(xs, q1, q3, alpha=0.2)
    if track == "A":
        ax.axhline(NOISE_STD**2, color="gray", ls=":", label="noise floor (sigma^2)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Fitness evaluations")
    ax.set_ylabel("Best-so-far train MSE")
    ax.set_title(f"{title}: convergence (median, IQR band)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"convergence_{track}.png"), dpi=150)
    plt.close(fig)

    # --- boxplots of test error ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, key, lab in zip(axes, ("test_in_mse", "test_out_mse"),
                            ("In-range test MSE", "Extrapolation test MSE")):
        data = [np.maximum([r[key] for r in rows if r["track"] == track and r["algorithm"] == a], 1e-12)
                for a in algos]
        try:
            ax.boxplot(data, tick_labels=algos, showfliers=True)  # matplotlib >= 3.9
        except TypeError:
            ax.boxplot(data, labels=algos, showfliers=True)
        ax.set_yscale("log")
        ax.set_ylabel(lab + " (log)")
        ax.grid(True, which="both", alpha=0.3)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle(f"{title}: generalisation across {N_SEEDS} seeds")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"boxplot_{track}.png"), dpi=150)
    plt.close(fig)


def plot_fits(rows, track, outdir, title, seed=0):
    xs = np.linspace(-10, 10, 400)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xs, target_function(xs), "k", lw=3, label="True function")
    ax.axvspan(-5, 5, color="gray", alpha=0.1, label="training range")
    for r in rows:
        if r["track"] != track or r["seed"] != seed:
            continue
        if track == "A":
            y = poly_predict(r["best"], xs)
        else:
            y = predict_tree(decode(np.floor(r["best"]).astype(int)), xs)
        ax.plot(xs, y, "--", label=r["algorithm"])
    ax.set_ylim(-20, 350)
    ax.set_title(f"{title}: fitted curves (seed {seed})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"fits_{track}.png"), dpi=150)
    plt.close(fig)


# =====================================================
# 7. MAIN
# =====================================================

def main():
    global N_SEEDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--budget-a", type=int, default=BUDGET_A)
    ap.add_argument("--budget-b", type=int, default=BUDGET_B)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    N_SEEDS = args.seeds
    os.makedirs(args.outdir, exist_ok=True)

    track_a = [("GA", ga_real, 1), ("DE", de_real, 2), ("RandomSearch", random_search_real, 3)]
    track_b = [("GE", ge_standard, 4), ("DE-GE", ge_de, 5), ("RandomSearch", random_search_ge, 6)]

    rows = []
    print(f"Running {N_SEEDS} seeds | Track A budget={args.budget_a} | Track B budget={args.budget_b}")
    t_all = time.perf_counter()
    for seed in range(N_SEEDS):
        y_train = make_data(seed)
        rows.append(run_polyfit(seed, y_train))
        for name, fn, aid in track_a:
            rows.append(run_one("A", name, fn, seed, aid, y_train, args.budget_a))
        for name, fn, aid in track_b:
            rows.append(run_one("B", name, fn, seed, aid, y_train, args.budget_b))
        print(f"  seed {seed + 1}/{N_SEEDS} done ({time.perf_counter() - t_all:.0f}s elapsed)")

    # sanity check: every budgeted run used exactly the same number of evaluations
    for track, budget in (("A", args.budget_a), ("B", args.budget_b)):
        used = {r["evals"] for r in rows if r["track"] == track and r["algorithm"] != "PolyFit (ref)"}
        assert used == {budget}, f"Unequal budgets in track {track}: {used}"

    summarize(rows, "A")
    summarize(rows, "B")
    save_csvs(rows, args.outdir)

    plot_track(rows, "A", args.budget_a, args.outdir, "Track A (parametric fit)")
    plot_track(rows, "B", args.budget_b, args.outdir, "Track B (symbolic regression)")
    plot_fits(rows, "A", args.outdir, "Track A")
    plot_fits(rows, "B", args.outdir, "Track B")
    print(f"Figures saved to {args.outdir}/")


if __name__ == "__main__":
    main()