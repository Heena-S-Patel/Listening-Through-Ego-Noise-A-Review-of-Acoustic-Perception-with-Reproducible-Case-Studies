"""
Experiment 6. Is drone ego-noise alike across drones?

Experiments 1 to 5 use one drone (DREGON). This experiment measures three properties of ego-noise on
five quadrotors with 8-microphone arrays, using noise-only recordings: the DREGON noise-only flight and
four public recordings from the IEEE Signal Processing Cup 2019 ego-noise collection
(http://dregon.inria.fr/datasets/the-spcup19-egonoise-dataset/). The measures need no array geometry,
so drones with different arrays can be compared.

    tonal share       share of 100-4000 Hz noise energy in narrow spectral peaks (>= 6 dB above a 300 Hz
                      running-median floor, 8192-point spectra on 1 s chunks). High = rotor tones dominate.
    spatial rank      share of 300-4000 Hz noise energy in the 4 strongest eigenvectors of the 8x8 spatial
                      covariance (one per rotor on a quadrotor). Spatially white noise would give 4/8 = 50%.
    learned rejection noise power removed (dB) from the SECOND half of a recording by projecting out the
                      4-dimensional subspace learned on the FIRST half. Spatially white noise gives 3 dB.
                      This is the room a learned spatial noise model has to work with.

Outputs in results/: exp6_multidrone.png, exp6_multidrone.csv, exp6_metrics.json
"""
import csv
import os

import numpy as np
import scipy.signal
from scipy.ndimage import median_filter

import common
import config
import dregon_io

SPCUP = os.path.join(os.path.dirname(config.DATA_DIR), "SPCup19_egonoise")

# name, drone, list of (path, start fraction, end fraction); fractions trim take-off and landing
RECORDINGS = [
    ("DREGON", "Quadrotor (DREGON)", None),
    ("Diagonal_Unloading", "DJI Phantom 4 Pro",
     [("Diagonal_Unloading/recordings/static/static_2m.wav", 0.05, 0.95),
      ("Diagonal_Unloading/recordings/static/static_10m.wav", 0.05, 0.95)]),
    ("AGH", "Quadrotor in anechoic chamber",
     [(f"AGH/ego-noise/mic array/{k}.wav", 0.1, 0.9) for k in range(5)]),
    ("KU_Leuven", "MK Quadro V3",
     [("KU_Leuven/SPCUP19_KU_Leuven_Team_1_recording.wav", 0.1, 0.9),
      ("KU_Leuven/SPCUP19_KU_Leuven_Team_2_recording.wav", 0.1, 0.9)]),
    ("Shout_COOEE", "Intel Aero",
     [("Shout_COOEE!/SPCUP19_Shout COOEE_StaticSubmission1.wav", 0.3, 0.7),
      ("Shout_COOEE!/SPCUP19_Shout COOEE_StaticSubmission2.wav", 0.3, 0.7)]),
]
K_SPATIAL = 4
NFFT = 1024
BAND = (300.0, 4000.0)

# Array geometries published with the recordings (metres). AGH and Intel Aero did not publish theirs.
GEOMETRY = {
    "DREGON": config.MIC_POS,
    "Diagonal_Unloading": np.array([[0.0426, 0.0213, 0], [0.0424, 0.0424, 0], [0.0213, 0.0426, 0], [0.0002, 0.0424, 0],
                                    [0.0000, 0.0213, 0], [0.0002, 0.0002, 0], [0.0213, 0.0000, 0], [0.0424, 0.0002, 0]]),
    "KU_Leuven": np.array([[-0.0345, -0.0595, -0.0345], [0.0345, -0.0595, 0.0345], [0.0595, -0.0345, -0.0345],
                           [0.0595, 0.0345, 0.0345], [0.0345, 0.0595, -0.0345], [-0.0345, 0.0595, 0.0345],
                           [-0.0595, 0.0345, -0.0345], [-0.0595, -0.0345, 0.0345]]),
}


def diffuse_covariance(mic_pos, freqs):
    """Spatial coherence of a spherically isotropic (diffuse) noise field: sinc(2 pi f d / c) per pair."""
    d = np.linalg.norm(mic_pos[:, None, :] - mic_pos[None, :, :], axis=-1)
    return np.sinc(2 * freqs[:, None, None] * d[None] / config.SPEED_OF_SOUND).astype(complex)


def segments(name, spec):
    """Yield (audio[samples, 8], fs) noise-only segments for one drone."""
    if spec is None:
        rec = dregon_io.load_recording(config.NOISE_ONLY_FLIGHT)
        x, fs = rec["audio"], rec["fs"]
        yield x[int(15 * fs): int(65 * fs)], fs          # steady flight, take-off and landing excluded
        return
    for rel, a, b in spec:
        x, fs = dregon_io.read_wav(os.path.join(SPCUP, rel))
        n = len(x)
        yield x[int(a * n): int(b * n)], fs


def tonal_share(x, fs, band=(100, 4000), chunk_sec=1.0):
    """Measured on 1 s chunks, where rotor speeds are nearly constant, then averaged:
    over long spans the tones drift with rotor speed and smear into a broadband floor."""
    n = int(chunk_sec * fs)
    shares = []
    for s in range(0, len(x) - n + 1, n):
        f, P = scipy.signal.welch(x[s:s + n].T, fs=fs, nperseg=8192, axis=-1)
        P = P.mean(axis=0)
        k = (f >= band[0]) & (f <= band[1])
        db = 10 * np.log10(P + 1e-20)
        # floor = running median over 300 Hz: wider than a cluster of tones from four rotors at similar speeds
        floor = median_filter(db, size=max(5, int(300 / (f[1] - f[0]))), mode="nearest")
        peak = (db - floor) >= 6.0
        shares.append(P[k & peak].sum() / P[k].sum())
    return float(np.mean(shares))


def covariances(x, fs, band=BAND):
    """Per-bin spatial covariances in the band, each normalised to unit trace so every bin counts equally."""
    f, _, X = scipy.signal.stft(x.T, fs=fs, nperseg=NFFT, noverlap=NFFT // 2)
    k = (f >= band[0]) & (f <= band[1])
    X = X[:, k, :]
    R = np.einsum("mft,nft->fmn", X, X.conj()) / X.shape[2]
    return R / np.real(np.trace(R, axis1=1, axis2=2))[:, None, None], f[k]


def spatial_rank(R):
    """Mean over bins of the share of energy in the K strongest eigenvectors."""
    w = np.linalg.eigvalsh(R)[:, ::-1]
    return float(np.mean(w[:, :K_SPATIAL].sum(axis=1) / w.sum(axis=1)))


def learned_rejection_db(R_train, R_test):
    """Mean over bins of the power removed from R_test by projecting out the K-dim subspace of R_train."""
    _, V = np.linalg.eigh(R_train)
    U = V[:, :, -K_SPATIAL:]
    M = R_test.shape[-1]
    P = np.eye(M)[None] - U @ np.conj(np.transpose(U, (0, 2, 1)))
    left = np.real(np.einsum("fij,fjk,fki->f", P, R_test, P)) / np.real(np.trace(R_test, axis1=1, axis2=2))
    return float(10 * np.log10(1.0 / np.mean(left)))


def main():
    plt = common.plot_style()
    print("Experiment 6  ego-noise across drones")
    rows = []
    for name, drone, spec in RECORDINGS:
        tonal, rank, rej, dur, freqs = [], [], [], 0.0, None
        for x, fs in segments(name, spec):
            if x.shape[1] != 8:
                print(f"  skipping a {x.shape[1]}-channel file of {name}")
                continue
            dur += len(x) / fs
            half = len(x) // 2
            tonal.append(tonal_share(x, fs))
            R_all, freqs = covariances(x, fs)
            rank.append(spatial_rank(R_all))
            rej.append(learned_rejection_db(covariances(x[:half], fs)[0], covariances(x[half:], fs)[0]))
        if not tonal:
            continue
        row = {"name": name, "drone": drone, "seconds": round(dur, 1), "files": len(tonal),
               "tonal_share": float(np.mean(tonal)), "spatial_rank4": float(np.mean(rank)),
               "learned_rejection_db": float(np.mean(rej)), "diffuse_rank4": None, "diffuse_rejection_db": None}
        if name in GEOMETRY:
            G = diffuse_covariance(GEOMETRY[name], freqs)
            G = G / np.real(np.trace(G, axis1=1, axis2=2))[:, None, None]
            row["diffuse_rank4"] = spatial_rank(G)
            row["diffuse_rejection_db"] = learned_rejection_db(G, G)
        rows.append(row)
        ref = (f"(diffuse field on this array: {100 * row['diffuse_rank4']:4.1f}%, {row['diffuse_rejection_db']:4.1f} dB)"
               if row["diffuse_rank4"] is not None else "(array geometry not published)")
        print(f"  {name:20s} {drone:30s} {dur:6.1f} s  tonal {100 * row['tonal_share']:5.1f}%  "
              f"top-4 spatial {100 * row['spatial_rank4']:5.1f}%  learned rejection {row['learned_rejection_db']:5.1f} dB  {ref}")

    with open(common.results_path("exp6_multidrone.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    common.save_json({"recordings": rows, "k_spatial": K_SPATIAL, "band_hz": list(BAND),
                      "white_noise_reference": {"spatial_rank4": K_SPATIAL / 8,
                                                "learned_rejection_db": float(10 * np.log10(8 / (8 - K_SPATIAL)))}},
                     "exp6_metrics.json")

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.7), gridspec_kw={"width_ratios": [1, 1.25]})
    names = [r["drone"].replace(" (DREGON)", "\n(DREGON)").replace("Quadrotor in anechoic chamber", "Anechoic quad\n(AGH)")
             for r in rows]
    y = np.arange(len(rows))
    axes[0].barh(y, [100 * r["tonal_share"] for r in rows], color="#4c72b0")
    axes[0].set_yticks(y, names, fontsize=9)
    axes[0].set_title("Tonal share of\nego-noise energy (%)", fontsize=10)
    ax = axes[1]
    ax.barh(y - 0.18, [r["learned_rejection_db"] for r in rows], 0.34, color="#4c72b0", label="measured ego-noise")
    ref = [r["diffuse_rejection_db"] if r["diffuse_rejection_db"] is not None else np.nan for r in rows]
    ax.barh(y + 0.18, ref, 0.34, color="#bbbbbb", label="diffuse noise, same array")
    for i, r in enumerate(rows):
        if r["diffuse_rejection_db"] is None:
            ax.text(0.2, i + 0.18, "geometry not published", fontsize=8, va="center", color="#555555")
    ax.set_yticks(y, [""] * len(rows))
    ax.set_title("Rejection by a subspace learned\non the other half (dB)", fontsize=10)
    fig.legend(*ax.get_legend_handles_labels(), fontsize=9, loc="lower center", ncol=2, frameon=False)
    ax.set_ylim(len(rows) - 0.3, -0.6)
    axes[0].set_ylim(len(rows) - 0.3, -0.6)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(common.results_path("exp6_multidrone.png"))
    plt.close(fig)
    print("  figures written to", config.RESULTS_DIR)
    return rows


if __name__ == "__main__":
    main()
