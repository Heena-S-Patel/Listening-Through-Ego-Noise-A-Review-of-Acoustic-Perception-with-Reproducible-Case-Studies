"""
Experiment 7. Does rotor-speed jitter in flight lower the tonal share?

Experiment 6 finds tonal shares from 1% to 70% across five quadrotors. The two highest come from drones
that were not flying (KU Leuven: held about 0.5 m above the floor; AGH: stationary in an anechoic
chamber), the three lowest from drones in flight. A plausible cause: in flight the controller keeps
adjusting rotor speeds, which smears the rotor harmonics in a 1 s spectrum and pushes them under the
6 dB peak threshold. Only DREGON logs rotor speeds, so the within-drone test uses DREGON alone.

    (a) tonal share of the five drones (from exp6_multidrone.csv), marked flying or not flying
    (b) per 1 s window of steady DREGON flight: within-window rotor-speed std against tonal share
    (c) mean spectrum of the calmest and the most jittery quarter of those windows, 1 to 2 kHz
    (d) tonal share against chunk length, with white-noise controls for the estimator's own bias:
        shorter chunks average fewer spectra, so random peaks alone start to count as tonal

Outputs in results/: exp7_tonal_vs_flight.png, exp7_jitter_vs_tonal_share.png and
         exp7_smeared_harmonics.png (panels b and c alone), exp7_metrics.json,
         exp7_per_window.csv and exp7_aligned_spectra.csv (the plotted data)
Run exp6_multidrone_egonoise.py first.
"""
import csv

import numpy as np
import scipy.signal
import scipy.stats
from scipy.ndimage import median_filter

import common
import config
import dregon_io
from exp6_multidrone_egonoise import tonal_share

STEADY = (15.0, 65.0)          # same steady-flight segment as experiment 6
CHUNKS = (1.0, 0.5, 0.25)
CALM_COLOR, SHAKY_COLOR = "#4c72b0", "#c44e52"
ORDERS = (6.0, 12.0)          # blade-passing harmonic orders shown in panel (c), about 1 to 2 kHz in hover
FLYING = {"DREGON": True, "Diagonal_Unloading": True, "Shout_COOEE": True, "AGH": False, "KU_Leuven": False}
CONDITION = {"DREGON": "free flight", "Diagonal_Unloading": "hover, outdoors", "Shout_COOEE": "manual hover",
             "AGH": "stationary, anechoic", "KU_Leuven": "held 0.5 m above floor"}


def mean_spectrum_db(x, fs):
    """Channel-mean 8192-point Welch spectrum in dB, relative to its 300 Hz running median."""
    f, P = scipy.signal.welch(x.T, fs=fs, nperseg=8192, axis=-1)
    db = 10 * np.log10(P.mean(axis=0) + 1e-20)
    return f, db - median_filter(db, size=int(300 / (f[1] - f[0])), mode="nearest")


def main():
    plt = common.plot_style()
    print("Experiment 7  tonal share and rotor-speed jitter")
    with open(common.results_path("exp6_multidrone.csv")) as f:
        drones = list(csv.DictReader(f))

    rec = dregon_io.load_recording(config.NOISE_ONLY_FLIGHT)
    x, fs = rec["audio"], rec["fs"]
    ts, sp, _ = dregon_io.motor_speeds(rec)
    ts = ts - rec["audio_ts"][0]

    starts = np.arange(STEADY[0], STEADY[1] - 1 + 1e-9, 1.0)
    jitter, spread, speed, share = [], [], [], []
    for s in starts:
        w = sp[(ts >= s) & (ts < s + 1)]
        jitter.append(w.std(axis=0).mean())        # within-window std of each rotor, averaged over rotors
        spread.append(w.mean(axis=0).std())        # spread of the mean speed between the four rotors
        speed.append(w.mean())
        share.append(tonal_share(x[int(s * fs): int((s + 1) * fs)], fs))
    jitter, spread, speed, share = map(np.array, (jitter, spread, speed, share))
    rho_j = scipy.stats.spearmanr(jitter, share)
    rho_s = scipy.stats.spearmanr(spread, share)
    q1, q3 = np.quantile(jitter, [0.25, 0.75])
    calm, shaky = jitter <= q1, jitter >= q3
    print(f"  {len(starts)} windows, mean rotor speed {np.mean(speed):.1f} rev/s, "
          f"within-window std median {np.median(jitter):.2f} rev/s")
    print(f"  Spearman jitter vs tonal share rho={rho_j.statistic:.2f} p={rho_j.pvalue:.3f}; "
          f"between-rotor spread rho={rho_s.statistic:.2f} p={rho_s.pvalue:.3f}")
    print(f"  tonal share calmest quarter {100 * share[calm].mean():.1f}%, most jittery {100 * share[shaky].mean():.1f}%")
    print(f"  mean rotor speed calmest quarter {speed[calm].mean():.1f} rev/s (std over windows {speed[calm].std():.2f}), "
          f"most jittery {speed[shaky].mean():.1f} rev/s ({speed[shaky].std():.2f})")

    seg = x[int(STEADY[0] * fs): int(STEADY[1] * fs)]
    rng = np.random.default_rng(0)
    white8 = rng.standard_normal((len(seg), 8))
    white1 = rng.standard_normal((len(seg), 1))
    chunk_rows = []
    for c in CHUNKS:
        row = {"chunk_s": c, "dregon": tonal_share(seg, fs, chunk_sec=c),
               "white_8ch": tonal_share(white8, fs, chunk_sec=c), "white_1ch": tonal_share(white1, fs, chunk_sec=c)}
        chunk_rows.append(row)
        print(f"  chunk {c:4.2f} s  DREGON {100 * row['dregon']:5.1f}%  white 8 ch {100 * row['white_8ch']:5.1f}%  "
              f"white 1 ch {100 * row['white_1ch']:5.1f}%")

    metrics = {"windows": len(starts), "mean_rotor_speed_rps": float(np.mean(speed)),
                      "jitter_std_rps": {"median": float(np.median(jitter)), "q1": float(q1), "q3": float(q3),
                                         "max": float(jitter.max())},
                      "between_rotor_spread_rps_median": float(np.median(spread)),
                      "spearman_jitter": {"rho": float(rho_j.statistic), "p": float(rho_j.pvalue)},
                      "spearman_spread": {"rho": float(rho_s.statistic), "p": float(rho_s.pvalue)},
                      "tonal_share_calmest_quarter": float(share[calm].mean()),
                      "tonal_share_most_jittery_quarter": float(share[shaky].mean()),
                      "mean_speed_calmest_quarter_rps": float(speed[calm].mean()),
                      "mean_speed_most_jittery_quarter_rps": float(speed[shaky].mean()),
                      "chunk_length": chunk_rows}
    common.save_json(metrics, "exp7_metrics.json")

    # each window on its own blade-passing order axis (f / 2 f0), so that windows at different mean speeds
    # line up and only the within-window smearing is left. f0 is the mean of the four rotors; DREGON's two
    # rotor pairs run at about 75 and 85 rev/s, so their harmonics sit near 0.93 k and 1.06 k, not at integers
    orders = np.linspace(ORDERS[0], ORDERS[1], 1200)
    aligned = {}
    for key, mask in (("calm", calm), ("shaky", shaky)):
        dbs = []
        for s, f0 in zip(starts[mask], speed[mask]):
            f, db = mean_spectrum_db(x[int(s * fs): int((s + 1) * fs)], fs)
            dbs.append(np.interp(orders, f / (2 * f0), db))
        aligned[key] = np.mean(dbs, axis=0)

    # plotted data, so that the figures can be redrawn without the audio (tools/make_paper_figures.py)
    with open(common.results_path("exp7_per_window.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["start_s", "jitter_std_rps", "between_rotor_spread_rps", "mean_speed_rps", "tonal_share", "group"])
        for i, s in enumerate(starts):
            group = "calmest" if calm[i] else "most_jittery" if shaky[i] else "middle"
            wr.writerow([s, jitter[i], spread[i], speed[i], share[i], group])
    with open(common.results_path("exp7_aligned_spectra.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["order", "calmest_db", "most_jittery_db"])
        wr.writerows(zip(orders, aligned["calm"], aligned["shaky"]))

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.0))

    ax = axes[0, 0]
    order = sorted(drones, key=lambda r: float(r["tonal_share"]))
    y = np.arange(len(order))
    ax.barh(y, [100 * float(r["tonal_share"]) for r in order],
            color=["#4c72b0" if FLYING[r["name"]] else "#dd8452" for r in order])
    ax.set_yticks(y, [f"{r['drone'].replace('Quadrotor in anechoic chamber', 'Quadrotor (AGH)')}\n"
                      f"{CONDITION[r['name']]}" for r in order], fontsize=8)
    for i, r in enumerate(order):
        ax.text(100 * float(r["tonal_share"]) + 1, i, f"{100 * float(r['tonal_share']):.1f}%", va="center", fontsize=8)
    ax.set_xlim(0, 85)
    ax.set_xlabel("Tonal share (%)")
    ax.set_title("(a) Five drones: flying (blue) vs not flying (orange)", fontsize=10)

    ax = axes[0, 1]
    ax.scatter(jitter, 100 * share, s=18, color="#4c72b0")
    ax.set_xlabel("Rotor-speed std within the 1 s window, mean of 4 rotors (rev/s)")
    ax.set_ylabel("Tonal share (%)")
    ax.set_title(f"(b) DREGON, {len(starts)} windows of 1 s\nSpearman $\\rho$ = {rho_j.statistic:.2f}, "
                 f"p = {rho_j.pvalue:.3f}", fontsize=10)

    ax = axes[1, 0]
    for key, label, color in (("calm", f"calmest quarter ({100 * share[calm].mean():.1f}%)", CALM_COLOR),
                              ("shaky", f"most jittery quarter ({100 * share[shaky].mean():.1f}%)", SHAKY_COLOR)):
        ax.plot(orders, aligned[key], color=color, lw=0.9, label=label)
    ax.axhline(6, color="#555555", ls="--", lw=0.8, label="6 dB tonal threshold")
    ax.set_xlim(*ORDERS)
    ax.set_ylim(-4.5, 10.5)
    ax.set_xlabel(f"Frequency / (2 x mean rotor speed)   (about {2 * ORDERS[0] * speed.mean() / 1000:.1f} to "
                  f"{2 * ORDERS[1] * speed.mean() / 1000:.1f} kHz)")
    ax.set_ylabel("Level above 300 Hz median (dB)")
    ax.set_title("(c) DREGON spectra, aligned to rotor speed", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper center", ncol=2)

    ax = axes[1, 1]
    cx = np.arange(len(CHUNKS))
    for key, label, color, dx in (("dregon", "DREGON ego-noise", "#4c72b0", -0.25),
                                  ("white_8ch", "white noise, 8 channels", "#bbbbbb", 0.0),
                                  ("white_1ch", "white noise, 1 channel", "#777777", 0.25)):
        vals = [100 * r[key] for r in chunk_rows]
        ax.bar(cx + dx, vals, 0.25, color=color, label=label)
        for i, v in enumerate(vals):
            ax.text(i + dx, v + 0.6, f"{v:.1f}", ha="center", fontsize=7)
    ax.set_xticks(cx, [f"{c:g} s" for c in CHUNKS])
    ax.set_xlabel("Analysis chunk length")
    ax.set_ylabel("Tonal share (%)")
    ax.set_title("(d) Shorter chunks, with estimator-bias controls", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(common.results_path("exp7_tonal_vs_flight.png"), dpi=200)
    plt.close(fig)
    print("  figure written to", common.results_path("exp7_tonal_vs_flight.png"))

    # the jitter result on its own, as two separate figures
    # figure 1: correlation. One window lies far out (about 3.9 rev/s); it is drawn at the right edge so the
    # other 49 are readable
    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    x_cap = 1.8
    jx = np.minimum(jitter, x_cap)
    mid = ~(calm | shaky)
    ax.scatter(jx[mid], 100 * share[mid], s=22, color="#aaaaaa", label="middle half")
    ax.scatter(jx[calm], 100 * share[calm], s=26, color=CALM_COLOR,
               label=f"calmest quarter (mean {100 * share[calm].mean():.1f}%)")
    ax.scatter(jx[shaky & (jitter <= x_cap)], 100 * share[shaky & (jitter <= x_cap)], s=26, color=SHAKY_COLOR,
               label=f"most jittery quarter (mean {100 * share[shaky].mean():.1f}%)")
    for j, v in zip(jitter[jitter > x_cap], share[jitter > x_cap]):
        ax.scatter([x_cap], [100 * v], s=40, color=SHAKY_COLOR, marker=">")
        ax.annotate(f"{j:.1f} rev/s", (x_cap, 100 * v), xytext=(-8, -3), textcoords="offset points",
                    ha="right", fontsize=7.5, color=SHAKY_COLOR)
    for mask, color in ((calm, CALM_COLOR), (shaky, SHAKY_COLOR)):
        ax.plot([jx[mask].min(), jx[mask].max()], [100 * share[mask].mean()] * 2, color=color, lw=2.2, alpha=0.8)
    ax.set_xlim(0.6, x_cap + 0.08)
    ax.text(0.97, 0.97, f"Spearman $\\rho$ = {rho_j.statistic:.2f}\np = {rho_j.pvalue:.3f}, n = {len(starts)}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox={"boxstyle": "round", "fc": "white", "ec": "#cccccc"})
    ax.set_xlabel("Rotor-speed std within the 1 s window, mean of 4 rotors (rev/s)")
    ax.set_ylabel("Tonal share (%)")
    ax.set_title(f"DREGON, {len(starts)} windows: more jitter, lower tonal share", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, frameon=False,
              title="horizontal lines: mean of each quarter", title_fontsize=7.5)
    fig.tight_layout()
    fig.savefig(common.results_path("exp7_jitter_vs_tonal_share.png"), dpi=200)
    plt.close(fig)
    print("  figure written to", common.results_path("exp7_jitter_vs_tonal_share.png"))

    # figure 2: the smeared harmonics behind the correlation
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for key, mask, color, name in (("calm", calm, CALM_COLOR, "calmest quarter"),
                                   ("shaky", shaky, SHAKY_COLOR, "most jittery quarter")):
        ax.plot(orders, aligned[key], color=color, lw=1.2,
                label=f"{name}: median jitter {np.median(jitter[mask]):.2f} rev/s, mean speed {speed[mask].mean():.1f} rev/s")
    ax.fill_between(orders, aligned["shaky"], aligned["calm"],
                    where=(aligned["calm"] > aligned["shaky"]) & (aligned["calm"] > 2.0),
                    color=CALM_COLOR, alpha=0.15, lw=0, label="harmonic peak height lost in jittery windows")
    ax.set_xlim(*ORDERS)
    ax.set_ylim(-4.5, 9.5)
    ax.set_xlabel(f"Frequency / (2 x mean rotor speed)   (about {2 * ORDERS[0] * speed.mean() / 1000:.1f} to "
                  f"{2 * ORDERS[1] * speed.mean() / 1000:.1f} kHz)")
    ax.set_ylabel("Level above 300 Hz median (dB)")
    ax.set_title("DREGON: same mean rotor speed, smeared harmonics", fontsize=10)
    ax.legend(fontsize=8, loc="upper center")
    fig.tight_layout()
    fig.savefig(common.results_path("exp7_smeared_harmonics.png"), dpi=200)
    plt.close(fig)
    print("  figure written to", common.results_path("exp7_smeared_harmonics.png"))
    return metrics


if __name__ == "__main__":
    main()
