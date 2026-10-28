"""
Animated GIFs for the README. Needs no dataset: the concept and smearing animations are synthetic, and the
tracking animation replays results/exp2_per_window.csv from case study 2.

    python tools/make_animations.py          # writes docs/media/*.gif

    concept.gif     top view of the drone: rotor noise and a distant target reach the microphones; the baseline
                    is pulled towards the rotors, a learned noise model finds the target
    tracking.gif    the quiet-source DREGON flight: true direction, SRP-PHAT (A) and speed-binned GEVD-MUSIC (E)
    smearing.gif    rotor-speed variation flattens the harmonics of a synthetic rotor tone and lowers its tonal share
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.ndimage import median_filter
import scipy.signal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import config  # noqa: E402

RES = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results"))
OUT = os.path.join(ROOT, "docs", "media")
GRAY, RED, BLUE, ORANGE = "#7f7f7f", "#d62728", "#1f77b4", "#ff7f0e"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def save(anim, name, fps):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    anim.save(path, writer=PillowWriter(fps=fps), dpi=80)
    print(f"  {name}  {os.path.getsize(path) / 1e6:.1f} MB")


def concept():
    rotors = config.ROTOR_POS[:, :2] - config.ROTOR_POS[:, :2].mean(axis=0)
    mics = config.MIC_POS[:, :2] * 1.3                     # slightly enlarged so the array is visible
    src_dir = np.deg2rad(140)
    u = np.array([np.cos(src_dir), np.sin(src_dir)])
    n_frames, split = 120, 60

    fig, ax = plt.subplots(figsize=(8, 4.6))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.02)

    def frame(i):
        ax.clear()
        ax.set(xlim=(-1.25, 1.25), ylim=(-0.72, 0.72), aspect="equal", xticks=[], yticks=[])
        for s in ax.spines.values():
            s.set_visible(False)
        model_on = i >= split
        fade = min(1.0, (i - split) / 15) if model_on else 0.0
        # plane wave fronts from the distant target (quiet: thin lines)
        perp = np.array([-u[1], u[0]])
        for k in range(8):
            d = 1.6 - ((i * 0.03 + k * 0.4) % 3.2)
            c = u * d
            p = np.array([c - perp * 2, c + perp * 2])
            ax.plot(p[:, 0], p[:, 1], color=BLUE, lw=1.0, alpha=0.55)
        # expanding rings from each rotor (loud: thick, many); faded out once the noise model is on
        for r in rotors:
            for k in range(5):
                rad = (i * 0.02 + k * 0.12) % 0.6
                ax.add_patch(plt.Circle(r, rad, fill=False, color=ORANGE, lw=2.2 * (1 - rad / 0.6),
                                        alpha=(1 - rad / 0.6) * (1 - 0.85 * fade)))
        # drone frame, rotors and microphones
        for r in rotors:
            ax.plot([0, r[0]], [0, r[1]], color="k", lw=3, solid_capstyle="round")
            ax.add_patch(plt.Circle(r, 0.06, color="#444444"))
            ax.text(r[0] * 1.45, r[1] * 1.45, "rotor", ha="center", va="center", fontsize=8)
        ax.add_patch(plt.Circle((0, 0), 0.13, color="#dddddd", zorder=4))
        ax.scatter(mics[:, 0], mics[:, 1], s=12, color="k", zorder=5)
        ax.text(0, -0.34, "8 microphones", ha="center", fontsize=8, zorder=6)
        ax.annotate("target\n(far and quiet)", xy=u * 1.05, ha="center", va="center", color=BLUE, fontsize=9)
        # direction estimate
        if not model_on:
            est, col, lab = rotors[1] / np.linalg.norm(rotors[1]), GRAY, "SRP-PHAT points at a rotor"
        else:
            est, col, lab = u, RED, "noise model points at the target"
        ax.annotate("", xy=est * 0.62, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=3))
        ax.text(1.22, -0.66, lab, ha="right", color=col, fontsize=10, fontweight="bold")
        title = ("Rotor noise (orange) is loud and close. The target (blue) is far away.\n"
                 "A plain localizer follows the loudest sound: the rotors."
                 if not model_on else
                 "A noise model learned from a noise-only flight\n"
                 "suppresses the rotor directions, and the target becomes visible.")
        fig.suptitle(title, fontsize=10.5)
        return []

    save(FuncAnimation(fig, frame, frames=n_frames, interval=80), "concept.gif", fps=12)
    plt.close(fig)


def tracking():
    with open(os.path.join(RES, "exp2_per_window.csv")) as f:
        rows = [r for r in csv.DictReader(f) if "whitenoise-low" in r["flight"]]
    t = np.array([float(r["time_s"]) for r in rows])
    gt = np.array([[float(r["gt_az"]), float(r["gt_el"])] for r in rows])
    est = {m: np.array([[float(r[f"{m}_az"]), float(r[f"{m}_el"])] for r in rows])
           for m in ("A_SRP-PHAT", "E_GEVD-MUSIC_rpmNCM")}
    err = {m: np.array([float(r[f"{m}_err"]) for r in rows]) for m in est}
    idx = np.arange(0, len(rows), 2)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.6, 3.9), gridspec_kw={"width_ratios": [2.2, 1]})
    fig.subplots_adjust(left=0.08, right=0.97, top=0.84, bottom=0.14, wspace=0.35)

    def frame(j):
        i = idx[j]
        lo = max(0, i - 10)
        ax.clear()
        ax.set(xlim=(-180, 180), ylim=(-90, 90), xlabel="azimuth (deg)", ylabel="elevation (deg)")
        ax.axhspan(40, 55, color=GRAY, alpha=0.12)
        ax.text(-175, 47, "rotor directions", fontsize=8, color=GRAY, va="center")
        ax.plot(gt[lo:i + 1, 0], gt[lo:i + 1, 1], "-", color="k", lw=1.2, alpha=0.5)
        for m, col, lab in (("A_SRP-PHAT", GRAY, "A  SRP-PHAT"), ("E_GEVD-MUSIC_rpmNCM", RED, "E  noise model + motor speed")):
            ax.scatter(est[m][lo:i, 0], est[m][lo:i, 1], s=12, color=col, alpha=0.35)
            ax.scatter(*est[m][i], s=70, color=col, edgecolor="k", lw=0.5, label=lab, zorder=4)
        ax.scatter(*gt[i], s=160, marker="+", color="k", lw=2.5, label="true direction", zorder=5)
        ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
        ax.set_title(f"DREGON flight, quiet source, t = {t[i]:.1f} s", fontsize=10)
        bx.clear()
        acc = [100 * np.mean(err[m][:i + 1] <= 10) for m in est]
        bars = bx.bar([0, 1], acc, color=[GRAY, RED])
        for b, v in zip(bars, acc):
            bx.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}%", ha="center", fontsize=10)
        bx.set(xticks=[0, 1], xticklabels=["A", "E"], ylim=(0, 100), ylabel="within 10° so far (%)")
        fig.suptitle("Real result: the noise model follows the source, the baseline is pulled towards the rotors", fontsize=10.5)
        return []

    save(FuncAnimation(fig, frame, frames=len(idx), interval=100), "tracking.gif", fps=10)
    plt.close(fig)


def smearing():
    fs, dur, f0 = 16000, 1.0, 80.0
    n = int(fs * dur)
    rng = np.random.default_rng(0)
    wobble = scipy.signal.lfilter([1.0], [1.0, -0.995], rng.standard_normal(n))
    wobble = (wobble - wobble.mean()) / wobble.std()                 # slow, unit-variance speed fluctuation
    floor_noise = rng.standard_normal(n) * 0.6
    sigmas = np.r_[np.linspace(0, 2.0, 40), np.full(12, 2.0)]

    def spectrum(sigma):
        phase = 2 * np.pi * np.cumsum(f0 + sigma * wobble) / fs      # shaft angle
        x = sum(np.cos(2 * k * phase) / k ** 0.3 for k in range(1, 26)) + floor_noise
        f, P = scipy.signal.welch(x, fs=fs, nperseg=4096)
        db = 10 * np.log10(P)
        above = db - median_filter(db, size=int(300 / (f[1] - f[0])), mode="nearest")
        band = (f >= 100) & (f <= 4000)
        share = P[band & (above >= 6)].sum() / P[band].sum()
        return f, above, share

    f, ref, ref_share = spectrum(0.0)
    fig, ax = plt.subplots(figsize=(8, 3.9))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.8, bottom=0.15)
    k = (f >= 1000) & (f <= 2000)

    def frame(i):
        sigma = sigmas[i]
        _, above, share = spectrum(sigma)
        ax.clear()
        ax.plot(f[k], ref[k], color=BLUE, lw=1, alpha=0.35, label=f"constant speed (tonal share {100 * ref_share:.0f}%)")
        ax.plot(f[k], above[k], color=RED, lw=1.6, label=f"speed std {sigma:.1f} rev/s (tonal share {100 * share:.0f}%)")
        ax.axhline(6, color="k", ls="--", lw=0.8)
        ax.text(1995, 6.6, "6 dB: counted as a tone", ha="right", fontsize=8)
        ax.set(xlim=(1000, 2000), ylim=(-6, 32), xlabel="frequency (Hz)", ylabel="level above floor (dB)")
        ax.legend(loc="upper left", fontsize=8.5)
        fig.suptitle("Synthetic example. A flying drone keeps adjusting its rotor speeds. The harmonics move,\n"
                     "so they spread out and fall below the tone threshold: the tonal share drops.", fontsize=10)
        return []

    save(FuncAnimation(fig, frame, frames=len(sigmas), interval=150), "smearing.gif", fps=7)
    plt.close(fig)


if __name__ == "__main__":
    print("Writing animations to", OUT)
    concept()
    tracking()
    smearing()
