"""
Experiment 5. Do the localization results carry over to another platform and array?

LOCATA (Evers et al., IEEE/ACM TASLP 2020; data: https://doi.org/10.5281/zenodo.3630470, ODC-BY licence)
records a 12-microphone pseudo-spherical array built into a NAO robot head, in a reverberant room
(T60 about 0.5 s), with OptiTrack ground truth for array and source poses and per-sample voice activity labels.
In Task 5 the head is moved on a trolley. We use the single-source tasks of the development set:

    task 1  static loudspeaker,  static array
    task 3  moving talker,       static array
    task 5  moving talker,       moving array

Only the robot-head files are needed; tools/locata_partial.py fetches them from the Zenodo archive.

Methods, on the same 3 degree grid and 300-4000 Hz band as experiment 2:
    A  SRP-PHAT
    C  MUSIC
    D  GEVD-MUSIC whitened with a noise covariance learned from the frames of the same recording in which the
       voice activity labels say the source is silent, at least 0.5 s (about one reverberation time) away from
       any speech so that the reverberant tail does not leak in; reported also without this guard (D0)
B and E need motor speeds, which this platform does not have.

Ground truth is computed in the array's own frame from the tracked array pose, so no frame alignment is fitted.

Outputs in results/: exp5_locata_per_window.csv, exp5_locata_metrics.json, exp5_locata.png
"""
import csv
import glob
import os

import numpy as np
import soundfile as sf

import common
import config
import localisation as loc

LOCATA_DIR = os.path.join(os.path.dirname(config.DATA_DIR), "LOCATA", "dev")
TASKS = {1: "static source, static array", 3: "moving talker, static array", 5: "moving talker, moving array"}
ARRAY = "benchmark2"
ACTIVE = 0.8      # a window is scored if the source is active in at least this share of its samples
GUARD_SEC = 0.5           # noise frames must be this far from any speech (about one reverberation time)
MIN_NOISE_FRAMES = 20     # fewer guarded silent frames than this: fall back to plain silent frames
METHODS = ["A_SRP-PHAT", "C_MUSIC", "D_GEVD-MUSIC_globalNCM", "D0_GEVD-MUSIC_noguard"]
LABELS = {"A_SRP-PHAT": "A  SRP-PHAT", "C_MUSIC": "C  MUSIC", "D_GEVD-MUSIC_globalNCM": "D  GEVD-MUSIC, 0.5 s guard",
          "D0_GEVD-MUSIC_noguard": "D  GEVD-MUSIC, no guard"}


def seconds(table):
    """Columns year month day hour minute second -> seconds of the day."""
    return table[:, 3] * 3600 + table[:, 4] * 60 + table[:, 5]


def first_timestamp(path):
    with open(path) as f:
        f.readline()
        return seconds(np.array([[float(v) for v in f.readline().split()]]))[0]


def load_recording(rec_dir):
    x, fs = sf.read(os.path.join(rec_dir, f"audio_array_{ARRAY}.wav"), always_2d=True)
    t_audio = first_timestamp(os.path.join(rec_dir, f"audio_array_timestamps_{ARRAY}.txt")) + np.arange(len(x)) / fs
    pa = np.loadtxt(os.path.join(rec_dir, f"position_array_{ARRAY}.txt"), skiprows=1)
    src_files = sorted(glob.glob(os.path.join(rec_dir, "position_source_*.txt")))
    if len(src_files) != 1:
        raise ValueError(f"expected one source in {rec_dir}, found {len(src_files)}")
    src = os.path.basename(src_files[0])[len("position_source_"):-4]
    ps = np.loadtxt(src_files[0], skiprows=1)
    vad = np.loadtxt(os.path.join(rec_dir, f"VAD_{ARRAY}_{src}.txt"), skiprows=1)
    n = min(len(x), len(vad))
    return {"x": x[:n].astype(np.float32), "fs": fs, "t": t_audio[:n], "vad": vad[:n], "pa": pa, "ps": ps,
            "t_pa": seconds(pa), "t_ps": seconds(ps), "source": src}


def array_pose(rec, t):
    """Position (3,), rotation (3, 3) local-to-global, and local microphone positions (M, 3) at time t."""
    k = int(np.argmin(np.abs(rec["t_pa"] - t)))
    row = rec["pa"][k]
    pos = row[6:9]
    R = row[12:21].reshape(3, 3)
    mics = row[21:].reshape(-1, 3)
    return pos, R, (mics - pos) @ R          # R^T (m - pos) for each microphone


def source_dir_local(rec, t, pos, R):
    k = int(np.argmin(np.abs(rec["t_ps"] - t)))
    v = R.T @ (rec["ps"][k, 6:9] - pos)
    v = v / np.linalg.norm(v)
    return np.rad2deg(np.arctan2(v[1], v[0])), np.rad2deg(np.arcsin(np.clip(v[2], -1, 1)))


def noise_covariance(locz, x, vad, fs, guard_sec=None):
    """Spatial covariance from STFT frames in which the source is silent, and has been silent for guard_sec,
    so that the reverberant tail of the speech (T60 about 0.5 s) does not leak into the noise model."""
    guard_sec = GUARD_SEC if guard_sec is None else guard_sec
    _, X = loc.stft_multi(x, fs)
    hop, nfft = config.HOP, config.NFFT
    g = int(guard_sec * fs)
    busy = np.convolve(vad > 0, np.ones(2 * g + 1), mode="same") > 0 if g > 0 else vad > 0
    starts = np.arange(X.shape[2]) * hop
    silent = np.array([not busy[s:s + nfft].any() for s in starts if s + nfft <= len(vad)])
    if silent.sum() < MIN_NOISE_FRAMES and guard_sec > 0:       # too little silence: fall back to no guard
        Phi, n, _ = noise_covariance(locz, x, vad, fs, guard_sec=0.0)
        return Phi, n, 0.0
    Xs = X[:, locz.sel, :len(silent)][:, :, silent]
    return np.einsum("mft,nft->fmn", Xs, Xs.conj()) / max(1, Xs.shape[2]), int(silent.sum()), guard_sec


def top_k_share(R, k=4):
    """Mean over bins of the share of energy in the k strongest eigenvectors (same measure as experiment 6)."""
    w = np.linalg.eigvalsh(R)[:, ::-1]
    return float(np.mean(w[:, :k].sum(axis=1) / w.sum(axis=1)))


def diffuse_top_k_share(mic_pos, freqs, k=4):
    d = np.linalg.norm(mic_pos[:, None, :] - mic_pos[None, :, :], axis=-1)
    G = np.sinc(2 * freqs[:, None, None] * d[None] / config.SPEED_OF_SOUND)
    return top_k_share(G, k)


def main():
    plt = common.plot_style()
    print("Experiment 5  LOCATA robot head (NAO, 12 microphones)")
    rows, diag = [], []
    for task, label in TASKS.items():
        for rec_dir in sorted(glob.glob(os.path.join(LOCATA_DIR, f"task{task}", "recording*", ARRAY))):
            rec = load_recording(rec_dir)
            fs = rec["fs"]
            _, _, mics0 = array_pose(rec, rec["t"][0])
            locz = loc.Localiser(fs, mic_pos=mics0)
            Phi, n_noise, used_guard = noise_covariance(locz, rec["x"], rec["vad"], fs)
            Phi0, n_noise0, _ = noise_covariance(locz, rec["x"], rec["vad"], fs, guard_sec=0.0)
            starts, w = common.window_starts(len(rec["x"]), fs)
            name = os.path.basename(os.path.dirname(rec_dir))
            kept = 0
            p_active = []
            for s in starts:
                if rec["vad"][s:s + w].mean() < ACTIVE:
                    continue
                t_mid = rec["t"][s + w // 2]
                pos, R, mics = array_pose(rec, t_mid)
                if task == 5:     # the array moves, so the local geometry is refreshed (it is rigid, so this is a check)
                    locz = loc.Localiser(fs, mic_pos=mics) if np.abs(mics - mics0).max() > 0.005 else locz
                gaz, gel = source_dir_local(rec, t_mid, pos, R)
                xw = rec["x"][s:s + w]
                res = locz.run(xw, methods=["A_SRP-PHAT", "C_MUSIC"])
                _, X = loc.stft_multi(xw, fs)
                Rx = loc.spatial_cov(X, locz.sel)
                res["D_GEVD-MUSIC_globalNCM"] = locz.peak(locz.sub.spectrum(Rx, Phi))
                res["D0_GEVD-MUSIC_noguard"] = locz.peak(locz.sub.spectrum(Rx, Phi0))
                p_active.append(np.real(np.trace(Rx, axis1=1, axis2=2)).mean())
                row = {"task": task, "recording": name, "time_s": float(t_mid - rec["t"][0]), "gt_az": gaz, "gt_el": gel}
                for m, (az, el) in res.items():
                    row[f"{m}_az"], row[f"{m}_el"] = float(az), float(el)
                    row[f"{m}_err"] = float(loc.angular_error(az, el, gaz, gel))
                rows.append(row)
                kept += 1
            p_noise = np.real(np.trace(Phi0, axis1=1, axis2=2)).mean()
            snr = 10 * np.log10(max(np.mean(p_active) - p_noise, 1e-12) / p_noise)
            diag.append({"task": task, "recording": name, "windows": kept, "noise_frames": n_noise,
                         "guard_used_s": used_guard, "snr_db": float(snr), "noise_top4_share": top_k_share(Phi0),
                         "diffuse_top4_share": diffuse_top_k_share(mics0, locz.freqs[locz.sel])})
            print(f"  task {task} {name}: {kept} active windows, noise model from {n_noise} frames "
                  f"(guard {used_guard} s), SNR {snr:5.1f} dB, noise energy in 4 strongest directions "
                  f"{100 * diag[-1]['noise_top4_share']:4.1f}% (diffuse {100 * diag[-1]['diffuse_top4_share']:4.1f}%)")

    with open(common.results_path("exp5_locata_per_window.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    methods = METHODS
    summary = {"by_task": {}, "overall": {}, "recordings": diag,
               "noise_top4_share_white_reference": 4 / 12, "guard_s": GUARD_SEC}
    for task in TASKS:
        rr = [r for r in rows if r["task"] == task]
        summary["by_task"][str(task)] = {m: loc.summarise_errors([r[f"{m}_err"] for r in rr]) for m in methods}
    for m in methods:
        summary["overall"][m] = loc.summarise_errors([r[f"{m}_err"] for r in rows])
    summary["tasks"] = {str(k): v for k, v in TASKS.items()}
    common.save_json(summary, "exp5_locata_metrics.json")

    print(f"\n  Located within 10 degrees (%), median error (deg)")
    for task in TASKS:
        line = "  ".join(f"{m.split('_')[0]} {100 * summary['by_task'][str(task)][m]['acc@10deg']:5.1f}% "
                         f"{summary['by_task'][str(task)][m]['median_deg']:5.1f}" for m in methods)
        print(f"  task {task} ({TASKS[task]:28s}) n={summary['by_task'][str(task)][methods[0]]['n']:4d}  {line}")

    fig, ax = plt.subplots(figsize=(7, 3.2))
    xi = np.arange(len(TASKS))
    colors = {**common.METHOD_COLORS, "D0_GEVD-MUSIC_noguard": "#fdbf6f"}
    for j, m in enumerate(methods):
        vals = [100 * summary["by_task"][str(t)][m]["acc@10deg"] for t in TASKS]
        bars = ax.bar(xi + (j - 1.5) * 0.2, vals, 0.19, color=colors[m], label=LABELS[m])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}", ha="center", fontsize=7)
    ax.set_xticks(xi, [f"Task {t}\n{TASKS[t]}" for t in TASKS], fontsize=8)
    ax.set(ylabel="windows within 10° (%)", ylim=(0, 105), title="LOCATA robot head, single-source tasks")
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(common.results_path("exp5_locata.png"))
    plt.close(fig)
    print("  figures written to", config.RESULTS_DIR)
    return summary


if __name__ == "__main__":
    main()
