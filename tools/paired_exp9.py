"""Paired intervals for the tone mask combined with D and E, and for the Gaussian process baseline.

Reads results_review/exp9_combined.json and results_review/exp9_gp.json (from exp9_flight_sensitivity.py)
and writes results_review/exp9_paired.json. Usage: python tools/paired_exp9.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import exp8_statistics as s  # noqa: E402

REV = os.path.join(ROOT, "results_review")
c = json.load(open(os.path.join(REV, "exp9_combined.json")))
g = json.load(open(os.path.join(REV, "exp9_gp.json")))
fl = np.array(c["without_mask"]["E"]["flights"])
D = np.array(c["without_mask"]["D"]["errors"])
E = np.array(c["without_mask"]["E"]["errors"])
pairs = [
    ("E+mask1024 vs E", E, c["mask1024"]["E"]["errors"]),
    ("E+mask4096 vs E", E, c["mask4096"]["E"]["errors"]),
    ("D+mask1024 vs D", D, c["mask1024"]["D"]["errors"]),
    ("D+mask4096 vs D", D, c["mask4096"]["D"]["errors"]),
    ("GP rotor vs E", E, g["rotor"]["flight"]["errors"]),
    ("GP rotor+imu vs E", E, g["rotor_imu"]["flight"]["errors"]),
    ("GP rotor vs D", D, g["rotor"]["flight"]["errors"]),
    ("GP rotor+imu vs D", D, g["rotor_imu"]["flight"]["errors"]),
]
out = {}
for name, a, b in pairs:
    A, B = a <= 10, np.array(b) <= 10
    d = B.astype(float) - A.astype(float)
    out[name] = {"first": round(100 * A.mean(), 1), "second": round(100 * B.mean(), 1), "diff": round(100 * d.mean(), 1),
                 "ci5s": [round(100 * x, 1) for x in s.interval(s.block_boot(d, fl, 20))],
                 "signflip_p": round(s.signflip_p(d, fl, 20), 4), "mcnemar": s.mcnemar_p(A, B)}
    print(name, out[name])
json.dump(out, open(os.path.join(REV, "exp9_paired.json"), "w"), indent=1)
