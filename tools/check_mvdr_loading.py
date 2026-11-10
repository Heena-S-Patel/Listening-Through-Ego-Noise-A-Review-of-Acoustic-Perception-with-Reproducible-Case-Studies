"""MVDR gain of experiment 4 for other diagonal loadings (10^-2 in the paper), and for other input SNRs."""
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
os.environ.setdefault("RESULTS_DIR", os.path.join(ROOT, "results_review", "scratch"))

import common
import config
import exp4_input_output as e4
import localisation as loc

locz = loc.Localiser(config.FS)
plt = common.plot_style()
out = {}
for snr, loads in ((-10, (1e-4, 1e-3, 1e-2, 1e-1)), (-5, (1e-2,)), (-15, (1e-2,))):
    e4.MIX_SNR_DB = snr
    for d in loads:
        e4.MVDR_LOADING = d
        res = e4.snr_gain(locz, 1.0, plt)[0]
        out[f"{snr}_{d}"] = {k: res[k]["median_gain_db"] for k in ("delay_and_sum", "mvdr_global", "mvdr_rpm")}
        print(snr, d, out[f"{snr}_{d}"], flush=True)
with open(os.path.join(ROOT, "results_review", "exp4_mvdr_loading.json"), "w") as f:
    json.dump(out, f, indent=1)
