"""Small helpers shared by the experiment scripts."""
import json
import os

import numpy as np
import scipy.signal

import config
import dregon_io
import localisation as loc
import rotor


def results_path(*parts):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    return os.path.join(config.RESULTS_DIR, *parts)


def save_json(obj, name):
    with open(results_path(name), "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def load_json(name):
    p = results_path(name)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def get_speed_scale(noise_rec):
    cached = load_json("speed_scale.json")
    if cached and config.SPEED_SCALE is None:
        return cached["scale"]
    s = rotor.estimate_speed_scale(noise_rec)
    save_json({"scale": s}, "speed_scale.json")
    return s


def stft_bins(x, fs, sel, chunk_sec=10.0):
    """STFT of a long multichannel signal, keeping only the bins in `sel`, computed chunk by chunk."""
    nfft, hop = config.NFFT, config.HOP
    chunk = int(chunk_sec * fs) // hop * hop
    out, centres = [], []
    for s in range(0, len(x) - nfft, chunk):
        seg = x[s: s + chunk + nfft - hop]
        if len(seg) < nfft:
            break
        _, X = loc.stft_multi(seg, fs)
        out.append(X[:, sel, :].astype(np.complex64))
        centres.append(s + nfft // 2 + hop * np.arange(X.shape[2]))
    return np.concatenate(out, axis=2), np.concatenate(centres)


def build_ncm_model(rec, sel, sample_range=None):
    """Global and RPM binned noise covariance from a noise-only recording."""
    x = rec["audio"]
    offset = 0
    if sample_range is not None:
        offset = sample_range[0]
        x = x[sample_range[0]: sample_range[1]]
    X, centres = stft_bins(x, rec["fs"], sel)
    t = rec["audio_ts"][np.clip(centres + offset, 0, len(rec["audio_ts"]) - 1)]
    mean_speed = dregon_io.speeds_at(rec, t).mean(axis=1)
    running = mean_speed > 0.3 * np.median(mean_speed)
    return loc.NoiseCovModel(X[:, :, running], np.arange(len(sel)), mean_speed[running], config.NCM_RPM_BINS)


def window_starts(n_samples, fs, frame_sec=None, hop_sec=None):
    frame_sec = frame_sec or config.FRAME_SEC
    hop_sec = hop_sec or config.FRAME_HOP_SEC
    w, h = int(frame_sec * fs), int(hop_sec * fs)
    return np.arange(0, n_samples - w, h), w


def window_stft(x, fs):
    _, X = loc.stft_multi(x, fs)
    return X


def plot_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                         "axes.spines.top": False, "axes.spines.right": False})
    return plt


METHOD_LABELS = {
    "A_SRP-PHAT": "A  SRP-PHAT (baseline)",
    "B_SRP-PHAT+RPMmask": "B  SRP-PHAT + RPM tone mask",
    "C_MUSIC": "C  MUSIC (no noise model)",
    "D_GEVD-MUSIC_globalNCM": "D  GEVD-MUSIC, one average NCM",
    "E_GEVD-MUSIC_rpmNCM": "E  GEVD-MUSIC, RPM-binned NCM",
}
METHOD_COLORS = {
    "A_SRP-PHAT": "#7f7f7f",
    "B_SRP-PHAT+RPMmask": "#1f77b4",
    "C_MUSIC": "#bcbd22",
    "D_GEVD-MUSIC_globalNCM": "#ff7f0e",
    "E_GEVD-MUSIC_rpmNCM": "#d62728",
}
