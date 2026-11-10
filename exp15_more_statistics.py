"""
Experiment 15. Uncertainty of the comparisons in experiments 12 to 14, with the resampling of experiment 8.

    flights     5 s circular blocks within each flight, on the 391 windows of the DREGON flight table
    moving      5 s circular blocks, on the motors-off recording mixed with flight noise
    sweep       resampling the 24 directions, on the mixtures of the SNR-sweep table
    agh         exact McNemar on the trials, since two speech directions are too few to resample
    locata      resampling the recordings of each task

Usage: python exp15_more_statistics.py [part ...]      (reads results/, or the folder named in PAPER_RESULTS)
Output in results_review/: exp15_statistics.json
"""
import csv
import json
import os
import sys

import numpy as np

import exp8_statistics as e8

OUT = e8.OUT
BLOCK = 20    # 5 s at a hop of 0.25 s


def load(name):
    with open(os.path.join(OUT, name)) as f:
        return json.load(f)


def diff_row(a, b, groups, thr=10):
    """Gain of b over a in points, with a block interval and the block sign-flip test."""
    d = 100.0 * ((np.asarray(b) <= thr).astype(float) - (np.asarray(a) <= thr).astype(float))
    return {"gain": float(d.mean()), "ci": e8.interval(e8.block_boot(d, groups, BLOCK)), "p": e8.signflip_p(d, groups, BLOCK)}


def part_flights():
    w = load("exp12_wf.json")
    rows = e8.read("exp2_per_window.csv")
    flights = np.array(w["flights"])
    assert len(rows) == w["windows"]
    table = {"A": e8.A, "D": e8.D, "E": e8.E}
    old = {k: np.array([float(r[f"{v}_err"]) for r in rows]) for k, v in table.items()}
    assert np.array_equal(np.array(["high" in r["flight"] for r in rows]), flights == 0)
    new = {k: np.array(w[k]["errors"]) for k in ("F_global", "F_speed", "F_wide_global", "F_wide_speed", "F_16k_all", "GEVD_16k_all")}
    out = {"level": {}}
    for k, v in new.items():
        ok = 100.0 * (v <= 10)
        out["level"][k] = {"within10": float(ok.mean()), "ci": e8.interval(e8.block_boot(ok, flights, BLOCK))}
    pairs = [("A", "F_global"), ("D", "F_global"), ("E", "F_global"), ("F_global", "F_speed"), ("F_wide_global", "F_wide_speed"), ("GEVD_16k_all", "F_16k_all")]
    allv = {**old, **new}
    out["pairs"] = {f"{b}-{a}": diff_row(allv[a], allv[b], flights) for a, b in pairs}
    for g, label in ((0, "loud"), (1, "quiet")):
        m = flights == g
        out["pairs"][f"F_speed-F_global_{label}"] = diff_row(allv["F_global"][m], allv["F_speed"][m], flights[m])
        out["pairs"][f"F_global-E_{label}"] = diff_row(allv["E"][m], allv["F_global"][m], flights[m])
    return out


def part_moving():
    m = load("exp12_mixed.json")
    out = {}
    for snr, row in m["levels"].items():
        g = np.zeros(m["windows"], dtype=int)
        out[snr] = {f"{b}-{a}": diff_row(row[a]["errors"], row[b]["errors"], g) for a, b in (("A", "D"), ("A", "F"), ("D", "E"), ("D", "F"), ("F", "F_speed"))}
    return out


def part_sweep():
    s = load("exp12_sweep.json")
    dirs = np.array(s["directions"])
    with open(os.path.join(e8.PAPER, "exp3_per_trial.csv")) as f:
        trials = list(csv.DictReader(f))
    out = {}
    for level in ("-10", "-15"):
        tr = [r for r in trials if float(r["snr_db"]) == float(level)]
        errs = {k: np.array(s["levels"][level][k]["errors"]) for k in ("A", "F", "F_speed")}
        errs["D"] = np.array([float(r[f"{e8.D}_err"]) for r in tr])
        errs["E"] = np.array([float(r[f"{e8.E}_err"]) for r in tr])
        row = {}
        for a, b in (("A", "F"), ("D", "F"), ("E", "F"), ("F", "F_speed")):
            d = 100.0 * ((errs[b] <= 10).astype(float) - (errs[a] <= 10).astype(float))
            row[f"{b}-{a}"] = {"gain": float(d.mean()), "ci": e8.interval(e8.cluster_boot(d, dirs)),
                               "mcnemar": e8.mcnemar_p(errs[a] <= 10, errs[b] <= 10)}
        out[level] = row
    return out


def part_agh():
    rows = load("exp13_mixtures.json")["rows"]
    out = {}
    for kind in ("speech", "white"):
        out[kind] = {}
        for snr in sorted({r["snr"] for r in rows}, reverse=True):
            rr = [r for r in rows if r["signal"] == kind and r["snr"] == snr]
            ok = {m: np.array([r[m] <= 10 for r in rr]) for m in ("A", "D", "E", "F", "F_state")}
            out[kind][str(snr)] = {"n": len(rr), **{m: float(100 * v.mean()) for m, v in ok.items()},
                                   **{f"{b}-{a}": e8.mcnemar_p(ok[a], ok[b]) for a, b in (("A", "D"), ("D", "E"), ("F", "F_state"), ("D", "F"))}}
    out["per_file"] = {str(f): float(np.median([r["A"] for r in rows if r["file"] == f and r["snr"] == 0])) for f in range(10)}
    return out


def part_locata():
    out = {}
    for part in ("head_dev", "head_eval", "eigen_dev", "eigen_eval"):
        path = os.path.join(OUT, f"exp14_{part}.json")
        if not os.path.exists(path):
            continue
        res = load(f"exp14_{part}.json")
        out[part] = {"summary": res["summary"]}
        for task in ("3", "5"):
            rr = [r for r in res["rows"] if str(r["task"]) == task]
            cl = np.array([r["recording"] for r in rr])
            row = {}
            for a, b in (("C", "D"), ("A", "D"), ("A", "F")):
                d = 100.0 * ((np.array([r[b] for r in rr]) <= 10).astype(float) - (np.array([r[a] for r in rr]) <= 10).astype(float))
                row[f"{b}-{a}"] = {"gain": float(d.mean()), "ci": e8.interval(e8.cluster_boot(d, cl))}
            out[part][task] = row
    return out


PARTS = {"flights": part_flights, "moving": part_moving, "sweep": part_sweep, "agh": part_agh, "locata": part_locata}

if __name__ == "__main__":
    path = os.path.join(OUT, "exp15_statistics.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    for p in sys.argv[1:] or PARTS:
        res[p] = PARTS[p]()
        print(p, json.dumps(res[p], indent=0)[:6000], flush=True)
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
