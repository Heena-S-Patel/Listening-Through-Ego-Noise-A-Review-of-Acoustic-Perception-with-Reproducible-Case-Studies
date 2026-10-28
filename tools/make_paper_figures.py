"""Redraw Figs. 1 to 4 of the ICCI 2026 paper from saved results (no recomputation), with publication labels:
no in-figure titles, method names as in the paper, and fonts sized for a 0.6-0.75 text-width figure."""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("RESULTS_DIR", os.path.normpath(os.path.join(HERE, "..", "results")))
OUT = os.path.normpath(os.path.join(HERE, "..", "papers", "icci2026", "fig"))
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "savefig.dpi": 300,
                     "axes.spines.top": False, "axes.spines.right": False})


def tracking():
    with open(os.path.join(RES, "exp2_per_window.csv")) as f:
        rows = list(csv.DictReader(f))
    flights = [("DREGON_free-flight_whitenoise-high_room1", "Loud source"),
               ("DREGON_free-flight_whitenoise-low_room1", "Quiet source")]
    methods = [("A_SRP-PHAT", "A  SRP-PHAT", "#7f7f7f"), ("E_GEVD-MUSIC_rpmNCM", "E  GEVD-MUSIC, speed-binned", "#d62728")]
    fig, axes = plt.subplots(2, 2, figsize=(5.2, 3.0), sharex="col")
    for c, (name, title) in enumerate(flights):
        rr = [r for r in rows if r["flight"] == name]
        t = np.array([float(r["time_s"]) for r in rr])
        for i, key in enumerate(("az", "el")):
            ax = axes[i, c]
            gt = np.array([float(r[f"gt_{key}"]) for r in rr])
            if key == "az":                                   # do not draw a line across the +-180 deg wrap
                gt[np.r_[False, np.abs(np.diff(gt)) > 180]] = np.nan
            ax.plot(t, gt, "-", color="k", lw=1.2, label="true direction")
            for m, lab, col in methods:
                ax.plot(t, [float(r[f"{m}_{key}"]) for r in rr], ".", ms=2.5, color=col, label=lab)
            ax.set(ylabel=("azimuth" if key == "az" else "elevation") + " (deg)",
                   ylim=(-180, 180) if key == "az" else (-90, 90))
            ax.set_yticks([-180, -90, 0, 90, 180] if key == "az" else [-90, -45, 0, 45, 90])
            if i == 0:
                ax.set_title(title, fontsize=9)
        axes[1, c].set_xlabel("time (s)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, markerscale=3, fontsize=8, handlelength=1.5, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.07, 1, 1), h_pad=0.4, w_pad=0.6)
    fig.savefig(os.path.join(OUT, "dregon_tracking.png"))
    plt.close(fig)


def snr_by_frequency():
    d = json.load(open(os.path.join(RES, "exp4_metrics.json")))["snr_gain"]["snr_by_third_octave_db"]
    curves = [("input", "single microphone (input)", "k", "--"), ("delay_and_sum", "delay-and-sum", "#7f7f7f", "-"),
              ("mvdr_global", "MVDR, one noise model", "#ff7f0e", "-"), ("mvdr_rpm", "MVDR, speed-binned", "#d62728", "-")]
    fig, ax = plt.subplots(figsize=(4.0, 2.3))
    for key, lab, col, ls in curves:
        fc = np.array([float(k) for k in d[key]])
        ax.semilogx(fc, list(d[key].values()), ls, color=col, lw=1.4, label=lab)
    ax.set(xlabel="frequency (Hz)", ylabel="SNR (dB)", ylim=(-45, 17))
    ax.set_xticks([125, 250, 500, 1000, 2000, 4000, 8000], ["125", "250", "500", "1k", "2k", "4k", "8k"])
    ax.set_xticks([], minor=True)
    ax.legend(fontsize=7, loc="lower right", handlelength=1.8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "dregon_snr_by_frequency.png"))
    plt.close(fig)


def multidrone():
    rows = list(csv.DictReader(open(os.path.join(RES, "exp6_multidrone.csv"))))
    not_flying = {"KU_Leuven", "AGH"}                     # held above the floor / stationary in the chamber
    names = []
    for r in rows:
        name = r["drone"].replace(" (DREGON)", "\n(DREGON)").replace("Quadrotor in anechoic chamber", "Anechoic quad\n(AGH)")
        names.append(name + "*" if r["name"] in not_flying else name)
    fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.75), gridspec_kw={"width_ratios": [1, 1.25]})
    y = np.arange(len(rows))
    axes[0].barh(y, [100 * float(r["tonal_share"]) for r in rows], color="#4c72b0")
    axes[0].set_yticks(y, names, fontsize=8)
    axes[0].set_title("Tonal share of\nego-noise energy (%)", fontsize=8.5)
    ax = axes[1]
    ax.barh(y - 0.18, [float(r["learned_rejection_db"]) for r in rows], 0.34, color="#4c72b0", label="measured ego-noise")
    ref = [float(r["diffuse_rejection_db"]) if r["diffuse_rejection_db"] else np.nan for r in rows]
    ax.barh(y + 0.18, ref, 0.34, color="#bbbbbb", label="diffuse noise, same array")
    for i, r in enumerate(rows):
        if not r["diffuse_rejection_db"]:
            ax.text(0.2, i + 0.18, "geometry not published", fontsize=6.5, va="center", color="#555555")
    ax.set_yticks(y, [""] * len(rows))
    ax.set_title("Rejection by a subspace learned\non the other half (dB)", fontsize=8.5)
    for a in axes:
        a.set_ylim(len(rows) - 0.3, -0.6)
        a.tick_params(labelsize=8)
    fig.legend(*ax.get_legend_handles_labels(), fontsize=8, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(OUT, "multidrone_egonoise.png"))
    plt.close(fig)


def jitter():
    rows = list(csv.DictReader(open(os.path.join(RES, "exp7_per_window.csv"))))
    spec = list(csv.DictReader(open(os.path.join(RES, "exp7_aligned_spectra.csv"))))
    m = json.load(open(os.path.join(RES, "exp7_metrics.json")))
    groups = [("middle", "middle half", "#aaaaaa"), ("calmest", "calmest quarter", "#4c72b0"),
              ("most_jittery", "most variable quarter", "#c44e52")]
    x_cap = 1.8                                          # one window at about 3.9 rev/s is drawn at the edge
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.45), gridspec_kw={"width_ratios": [1, 1.35]})

    ax = axes[0]
    for g, lab, col in groups:
        rr = [r for r in rows if r["group"] == g]
        j = np.array([float(r["jitter_std_rps"]) for r in rr])
        s = 100 * np.array([float(r["tonal_share"]) for r in rr])
        ax.scatter(np.minimum(j, x_cap), s, s=9, color=col, marker="o", label=lab, zorder=3)
        for jj, ss in zip(j[j > x_cap], s[j > x_cap]):
            ax.annotate(f"{jj:.1f}$\\rightarrow$", (x_cap, ss), xytext=(-2, 4), textcoords="offset points",
                        ha="right", fontsize=6.5, color=col)
        if g != "middle":
            ax.plot([np.minimum(j, x_cap).min(), np.minimum(j, x_cap).max()], [s.mean()] * 2, color=col, lw=1.6)
    ax.text(0.97, 0.97, f"$\\rho$ = {m['spearman_jitter']['rho']:.2f}\np = {m['spearman_jitter']['p']:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7)
    ax.set(xlabel="rotor-speed std in window (rev/s)", ylabel="tonal share (%)", xlim=(0.6, x_cap + 0.05))

    ax = axes[1]
    o = np.array([float(r["order"]) for r in spec])
    calm = np.array([float(r["calmest_db"]) for r in spec])
    shaky = np.array([float(r["most_jittery_db"]) for r in spec])
    ax.plot(o, calm, color="#4c72b0", lw=1.0)
    ax.plot(o, shaky, color="#c44e52", lw=1.0)
    ax.fill_between(o, shaky, calm, where=(calm > shaky) & (calm > 2.0), color="#4c72b0", alpha=0.15, lw=0)
    ax.set(xlabel="frequency / (2 $\\times$ mean rotor speed)", ylabel="level above floor (dB)", xlim=(o[0], o[-1]),
           yticks=[-4, -2, 0, 2, 4, 6])

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=7.5, markerscale=1.6, handletextpad=0.3,
               bbox_to_anchor=(0.5, -0.01))
    for a, lab in zip(axes, ("(a)", "(b)")):            # subfigure labels below each panel
        a.text(0.5, -0.33, lab, transform=a.transAxes, ha="center", va="top", fontsize=8.5)
    for a in axes:
        a.tick_params(labelsize=7.5)
        a.xaxis.label.set_size(8)
        a.yaxis.label.set_size(8)
    fig.tight_layout(rect=(0, 0.04, 1, 1), w_pad=2.0)
    fig.savefig(os.path.join(OUT, "dregon_jitter_tonal.png"))
    plt.close(fig)


tracking()
snr_by_frequency()
multidrone()
jitter()
print("figures written to", OUT)
