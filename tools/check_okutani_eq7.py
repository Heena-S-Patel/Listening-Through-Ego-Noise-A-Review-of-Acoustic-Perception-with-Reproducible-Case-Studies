"""D with Okutani's Eq. 7 taken literally: E are the eigenvectors of K^-1/2 R K^-1/2, applied directly in Eq. 4.

Original-papers setting of the ablation table, on the 391 flight windows. The paper reports the result (71.1%).
Usage: python tools/check_okutani_eq7.py      Output: results_review/exp9_okutani_eq7.json
"""
import os
import sys

os.environ["PROTOCOL"] = "original"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.environ.setdefault("RESULTS_DIR", os.path.join(ROOT, "results_review", "scratch"))
sys.path.insert(0, ROOT)
import numpy as np
import scipy.linalg

import common
import config
import dregon_io
import localisation as loc

noise = dregon_io.load_recording(config.NOISE_ONLY_FLIGHT)
fs = noise["fs"]
L = loc.Localiser(fs)
ncm = common.build_noise_models(L, noise)
al = {"offset": 93, "az_sign": 1, "el_sign": 1}
err = []
for name in config.SOURCE_FLIGHTS:
    rec = dregon_io.load_recording(name)
    x = rec["audio"]
    starts, w = common.window_starts(len(x), fs)
    t = rec["audio_ts"][starts + w // 2]
    gaz, gel, ok = dregon_io.source_dir_at(rec, t)
    sp = dregon_io.speeds_at(rec, t).mean(axis=1)
    for i in np.where(ok & (sp > .3 * np.median(sp)))[0]:
        Rx = L.group_cov(x[starts[i]:starts[i] + w], "CD")
        F, M, _ = Rx.shape
        out = np.zeros(len(L.az))
        for f in range(F):
            N = ncm["CD"].global_ncm[f] + config.DIAG_LOADING * np.real(np.trace(ncm["CD"].global_ncm[f])) / M * np.eye(M)
            w2, U = np.linalg.eigh(N)
            Ki = (U / np.sqrt(w2)) @ U.conj().T
            _, V = np.linalg.eigh(Ki @ Rx[f] @ Ki)
            out += 1.0 / (np.sum(np.abs(L.sub_g["CD"].A[f] @ V[:, :M - 1].conj()), axis=1) + 1e-12)
        a, e = loc.apply_alignment(*L.peak(out), al)
        err.append(float(loc.angular_error(a, e, gaz[i], gel[i])))
print("D, Okutani Eq. 7 literal", round(100 * np.mean(np.array(err) <= 10), 1), "n", len(err))
import json  # noqa: E402
with open(os.path.join(ROOT, "results_review", "exp9_okutani_eq7.json"), "w") as f:
    json.dump({"D_within10": round(100 * float(np.mean(np.array(err) <= 10)), 1), "windows": len(err)}, f, indent=1)
