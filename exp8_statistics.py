"""
Experiment 8. Uncertainty of the main comparisons.

Works on the per-window results of experiments 2 to 5, so no audio is needed. Neighbouring windows overlap and
follow the same flight, so a plain binomial interval or McNemar test would be too optimistic. We report
    circular block bootstrap 95% intervals; blocks are drawn within each flight (5 s blocks, also 2.5 and 10 s)
    sign-flip test           two-sided p value that flips whole non-overlapping 5 s blocks of paired differences
    exact McNemar            on all windows, for comparison only, since it assumes independent windows
    effective sample size    from the autocorrelation of the paired difference
Controlled mixtures are resampled by direction (24 clusters) and LOCATA by recording (3 per task).

Usage: python exp8_statistics.py      (reads results/, or the folder named in PAPER_RESULTS)
Output in results_review/: exp8_statistics.json
"""
import csv
import json
import os

import numpy as np
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(HERE, os.environ.get("PAPER_RESULTS", "results"))
SHARED = os.path.join(HERE, "results")      # experiments 4 and 5, which do not use method B
OUT = os.path.join(HERE, "results_review")
SEED, N_BOOT = 0, 10000
HOP_SEC = 0.25
A, B, C, D, E = ("A_SRP-PHAT", "B_SRP-PHAT+RPMmask", "C_MUSIC", "D_GEVD-MUSIC_globalNCM", "E_GEVD-MUSIC_rpmNCM")


def read(name, folder=PAPER):
    with open(os.path.join(folder, name)) as f:
        return list(csv.DictReader(f))


def block_indices(rng, n, length):
    """Circular block resample of 0..n-1. Wrapping gives every window the same weight; plain moving blocks
    under-sample the start and end of a flight, which are its hardest windows here, and so bias the mean upward."""
    starts = rng.integers(0, n, size=int(np.ceil(n / length)))
    return np.concatenate([np.arange(s, s + length) % n for s in starts])[:n]


def interval(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def block_boot(values, groups, length, stat=np.mean):
    """values (N,) or (N, k). Blocks are drawn within each group."""
    rng = np.random.default_rng(SEED)
    idx = [np.where(groups == g)[0] for g in np.unique(groups)]
    out = []
    for _ in range(N_BOOT):
        pick = np.concatenate([i[block_indices(rng, len(i), length)] for i in idx])
        out.append(stat(values[pick]))
    return np.array(out)


def cluster_boot(values, clusters, stat=np.mean):
    rng = np.random.default_rng(SEED)
    ids = np.unique(clusters)
    members = [np.where(clusters == c)[0] for c in ids]
    out = []
    for _ in range(N_BOOT):
        pick = np.concatenate([members[k] for k in rng.integers(0, len(ids), len(ids))])
        out.append(stat(values[pick]))
    return np.array(out)


def signflip_p(diff, groups, length):
    """Two-sided p value of the mean paired difference; whole blocks keep or flip their sign together."""
    rng = np.random.default_rng(SEED)
    blocks = []
    for g in np.unique(groups):
        d = diff[groups == g]
        blocks += [d[s:s + length] for s in range(0, len(d), length)]
    sums = np.array([b.sum() for b in blocks])
    obs = abs(sums.sum())
    flips = rng.choice([-1.0, 1.0], size=(N_BOOT, len(sums)))
    return float((1 + np.sum(np.abs(flips @ sums) >= obs - 1e-12)) / (N_BOOT + 1))


def mcnemar_p(a, b):
    """Exact two-sided McNemar test on paired booleans."""
    n01, n10 = int(np.sum(~a & b)), int(np.sum(a & ~b))
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return {"only_first": n10, "only_second": n01, "p": float(p)}


def effective_n(diff, groups):
    """n / (1 + 2 sum rho_k), summed over groups, with the sum stopped at the first non-positive autocorrelation."""
    total = 0.0
    for g in np.unique(groups):
        d = diff[groups == g].astype(float)
        d = d - d.mean()
        if d.var() == 0:
            continue
        s = 0.0
        for k in range(1, len(d) // 2):
            r = np.mean(d[:-k] * d[k:]) / d.var()
            if r <= 0:
                break
            s += r
        total += len(d) / (1 + 2 * s)
    return float(total)


def flights():
    rows = read("exp2_per_window.csv")
    fl = np.array([r["flight"] for r in rows])
    err = {m: np.array([float(r[f"{m}_err"]) for r in rows]) for m in (A, B, C, D, E)}
    out = {"n": len(rows), "methods": {}, "pairs": {}, "per_flight": {}}
    length = int(5 / HOP_SEC)
    for m, e in err.items():
        ok = (e <= 10).astype(float)
        out["methods"][m] = {"within10": 100 * ok.mean(), "ci": [100 * v for v in interval(block_boot(ok, fl, length))],
                             "median": float(np.median(e)), "median_ci": interval(block_boot(e, fl, length, np.median))}
    for first, second in ((A, D), (A, E), (A, B), (A, C), (C, D), (D, E), (B, E), (B, D)):
        a, b = err[first] <= 10, err[second] <= 10
        diff = b.astype(float) - a.astype(float)
        cis = {f"{s}s": [100 * v for v in interval(block_boot(diff, fl, int(s / HOP_SEC)))] for s in (2.5, 5, 10)}
        res = {"diff": 100 * diff.mean(), "ci": cis, "signflip_p_5s": signflip_p(diff, fl, length),
               "mcnemar_all_windows": mcnemar_p(a, b), "effective_n": effective_n(diff, fl),
               "by_flight": {g: 100 * float(diff[fl == g].mean()) for g in np.unique(fl)}}
        if first == D and second == E:
            res["by_flight_ci_5s"] = {g: [100 * v for v in interval(block_boot(diff[fl == g], fl[fl == g], length))]
                                      for g in np.unique(fl)}
            res["mean_error_diff"] = float(err[D].mean() - err[E].mean())
            res["mean_error_diff_ci"] = interval(block_boot(err[D] - err[E], fl, length))
        out["pairs"][f"{first[0]}_vs_{second[0]}"] = res
    for g in np.unique(fl):
        out["per_flight"][g] = {m[0]: 100 * float(np.mean(e[fl == g] <= 10)) for m, e in err.items()}
    return out


def sweep():
    rows = read("exp3_per_trial.csv")
    cl = np.array([f"{r['gt_az']}_{r['gt_el']}" for r in rows])
    snr = np.array([int(float(r["snr_db"])) for r in rows])
    err = {m: np.array([float(r[f"{m}_err"]) for r in rows]) for m in (A, B, C, D, E)}
    out = {}
    for s in np.unique(snr)[::-1]:
        k = snr == s
        lev = {"n": int(k.sum()), "methods": {}, "pairs": {}}
        for m, e in err.items():
            ok = (e[k] <= 10).astype(float)
            lev["methods"][m[0]] = {"within10": 100 * ok.mean(), "ci": [100 * v for v in interval(cluster_boot(ok, cl[k]))]}
        for first, second in ((A, D), (A, E), (D, E), (C, D)):
            a, b = err[first][k] <= 10, err[second][k] <= 10
            diff = b.astype(float) - a.astype(float)
            lev["pairs"][f"{first[0]}_vs_{second[0]}"] = {"diff": 100 * diff.mean(),
                                                          "ci": [100 * v for v in interval(cluster_boot(diff, cl[k]))],
                                                          "mcnemar": mcnemar_p(a, b)}
        out[str(int(s))] = lev
    return out


def enhancement():
    rows = read("exp4_snr_gain.csv", SHARED)
    cl = np.array([f"{r['az']}_{r['el']}" for r in rows])
    g = {k: np.array([float(r[f"{k}_gain_db"]) for r in rows]) for k in ("delay_and_sum", "mvdr_global", "mvdr_rpm")}
    out = {k: {"median": float(np.median(v)), "ci": interval(cluster_boot(v, cl, np.median))} for k, v in g.items()}
    for x, y in (("delay_and_sum", "mvdr_global"), ("mvdr_global", "mvdr_rpm"), ("delay_and_sum", "mvdr_rpm")):
        d = g[y] - g[x]
        out[f"{y}_minus_{x}"] = {"median": float(np.median(d)), "ci": interval(cluster_boot(d, cl, np.median)),
                                 "share_positive": float(np.mean(d > 0))}
    return out


def locata():
    rows = read("exp5_locata_per_window.csv", SHARED)
    task = np.array([int(r["task"]) for r in rows])
    rec = np.array([f"{r['task']}_{r['recording']}" for r in rows])
    keys = {"A": A, "C": C, "D": D, "D0": "D0_GEVD-MUSIC_noguard"}
    err = {k: np.array([float(r[f"{m}_err"]) for r in rows]) for k, m in keys.items()}
    out = {}
    for t in (1, 3, 5):
        k = task == t
        res = {"n": int(k.sum()), "methods": {}, "pairs": {}, "per_recording": {}}
        for m, e in err.items():
            ok = (e[k] <= 10).astype(float)
            res["methods"][m] = {"within10": 100 * ok.mean(), "ci": [100 * v for v in interval(cluster_boot(ok, rec[k]))]}
        for first, second in (("A", "D"), ("C", "D"), ("A", "D0"), ("C", "D0"), ("A", "C")):
            a, b = err[first][k] <= 10, err[second][k] <= 10
            diff = b.astype(float) - a.astype(float)
            res["pairs"][f"{first}_vs_{second}"] = {"diff": 100 * diff.mean(),
                                                    "ci": [100 * v for v in interval(cluster_boot(diff, rec[k]))],
                                                    "mcnemar": mcnemar_p(a, b)}
        for r in np.unique(rec[k]):
            j = rec == r
            res["per_recording"][r] = {m: 100 * float(np.mean(e[j] <= 10)) for m, e in err.items()} | {"n": int(j.sum())}
        out[str(t)] = res
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {"paper_results": os.path.basename(PAPER), "seed": SEED, "n_boot": N_BOOT, "flights": flights(),
           "sweep": sweep(), "enhancement": enhancement(), "locata": locata()}
    with open(os.path.join(OUT, "exp8_statistics.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res["flights"]["methods"], indent=0))
    print(json.dumps(res["flights"]["pairs"]["D_vs_E"], indent=0))


if __name__ == "__main__":
    main()
