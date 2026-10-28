"""
Experiment 4. Input versus output: what does the drone hear before and after processing?

Experiments 1 to 3 only report directions. This one shows the effect on the signal itself.

    Part 1  Listening demo. A stretch of the loud-source flight. The input is microphone 1.
            The outputs steer the array at the direction method E estimates (no ground truth used):
              delay-and-sum           plain steering, no noise model
              MVDR + RPM noise model  steering that also cancels the rotor noise, with the noise
                                      covariance picked by current motor speed (same idea as E)
            Writes WAV files and a spectrogram figure.
    Part 2  Objective gain. Clean recordings mixed with flight noise (as in experiment 3). A beamformer
            is linear, so the signal part and the noise part can be passed through it separately and
            the output SNR measured exactly. Noise model from the first half of the noise flight,
            mixing noise from the second half.
    Part 3  Tracking figure. True direction against methods A and E over time, from experiment 2.

Outputs in results/
    audio/exp4_input_mic1.wav, audio/exp4_output_delay_and_sum.wav, audio/exp4_output_mvdr_rpm.wav
    exp4_spectrograms.png, exp4_snr_gain.png, exp4_tracking.png, exp4_metrics.json
"""
import csv
import os

import numpy as np
import scipy.signal
import soundfile as sf

import common
import config
import dregon_io
import localisation as loc

NFFT, HOP = config.NFFT, config.HOP
LISTEN_FLIGHT = config.SOURCE_FLIGHTS[0]   # loud source
LISTEN_START_S, LISTEN_SEC = 20.0, 10.0
MIX_SNR_DB = -10                           # input SNR for part 2, where experiment 3 shows the methods split
MVDR_LOADING = 1e-2                        # relative diagonal loading for the beamformer
GAIN_BAND_HZ = (200.0, 8000.0)             # band for the SNR numbers


def stft(x, fs):
    """x (samples, channels) -> X (channels, freqs, frames), invertible with istft."""
    _, _, X = scipy.signal.stft(x.T, fs=fs, nperseg=NFFT, noverlap=NFFT - HOP)
    return X


def istft(Y, fs):
    _, y = scipy.signal.istft(Y, fs=fs, nperseg=NFFT, noverlap=NFFT - HOP)
    return y


class FullBandNoiseModel:
    """Noise covariance for every STFT bin, global and per motor speed bin (like loc.NoiseCovModel)."""

    def __init__(self, rec, sample_range=None, n_bins=config.NCM_RPM_BINS, chunk_sec=10.0):
        x = rec["audio"]
        lo, hi = sample_range or (0, len(x))
        fs = rec["fs"]
        chunk = int(chunk_sec * fs) // HOP * HOP
        frames, speeds = [], []
        for s in range(lo, hi - NFFT, chunk):
            # chunks hop by `chunk` samples, keep only the frames that start inside this chunk
            X = stft(x[s: min(s + chunk + NFFT, hi)], fs)[:, :, : chunk // HOP]
            centres = s + HOP * np.arange(X.shape[2])
            t = rec["audio_ts"][np.clip(centres, 0, len(rec["audio_ts"]) - 1)]
            speeds.append(dregon_io.speeds_at(rec, t).mean(axis=1))
            frames.append(X.astype(np.complex64))
        X, speeds = np.concatenate(frames, axis=2), np.concatenate(speeds)
        running = speeds > 0.3 * np.median(speeds)
        X, speeds = X[:, :, running], speeds[running]
        self.global_ncm = np.einsum("mft,nft->fmn", X, X.conj()) / X.shape[2]
        edges = np.quantile(speeds, np.linspace(0, 1, n_bins + 1))
        self.centres, self.bins = [], []
        for b in range(n_bins):
            idx = np.where((speeds >= edges[b]) & (speeds <= edges[b + 1]))[0]
            if len(idx) >= 10:
                Xb = X[:, :, idx]
                self.bins.append(np.einsum("mft,nft->fmn", Xb, Xb.conj()) / len(idx))
                self.centres.append(speeds[idx].mean())
        self.centres = np.array(self.centres)

    def for_speed(self, mean_speed):
        return self.bins[int(np.argmin(np.abs(self.centres - mean_speed)))]


def steering(freqs, az, el):
    """(F, M) phase of a far-field source at (az, el) in the array frame, same convention as localisation.py."""
    tau = (loc.unit_vectors(az, el) @ config.MIC_POS.T) / config.SPEED_OF_SOUND
    return np.exp(2j * np.pi * freqs[:, None] * tau[None, :])


def das_weights(a):
    return a / a.shape[1]


def mvdr_weights(a, Phi):
    M = a.shape[1]
    load = MVDR_LOADING * np.real(np.trace(Phi, axis1=1, axis2=2)) / M
    Phi = Phi + load[:, None, None] * np.eye(M)[None]
    x = np.linalg.solve(Phi, a[:, :, None])[:, :, 0]
    return x / np.sum(a.conj() * x, axis=1, keepdims=True)


def apply(w, X):
    """w (F, M), X (M, F, T) -> (F, T)."""
    return np.einsum("fm,mft->ft", w.conj(), X)


def band_power(Y, freqs):
    k = (freqs >= GAIN_BAND_HZ[0]) & (freqs <= GAIN_BAND_HZ[1])
    return np.mean(np.abs(Y[k]) ** 2)


# ---------------------------------------------------------------------------
# Part 1, listening demo on a real flight
# ---------------------------------------------------------------------------
def listening_demo(locz, ncm_band, ncm_full, scale, plt):
    rec = dregon_io.load_recording(LISTEN_FLIGHT)
    fs = rec["fs"]
    # the clip is moved earlier (and shortened if needed) when the recording is too short, e.g. synthetic data
    n = min(int(LISTEN_SEC * fs), len(rec["audio"]))
    s0 = min(int(LISTEN_START_S * fs), len(rec["audio"]) - n)
    start_s, seconds = s0 / fs, n / fs
    x = rec["audio"][s0: s0 + n]
    X = stft(x, fs)
    freqs = np.fft.rfftfreq(NFFT, 1 / fs)
    frame_t = np.arange(X.shape[2]) * HOP / fs                     # frame centres, seconds into the clip

    # direction from method E on each 0.5 s window (hop 0.25 s), then used for the frames it covers
    starts, w = common.window_starts(n, fs)
    Y_das = np.zeros(X.shape[1:], dtype=complex)
    Y_mvdr = np.zeros_like(Y_das)
    al = common.load_json("frame_alignment_flight.json")
    errs = []
    for k, s in enumerate(starts):
        t_mid = rec["audio_ts"][s0 + s + w // 2]
        sp = dregon_io.speeds_at(rec, np.array([t_mid]))[0]
        r = locz.run(x[s:s + w], f0s=sp * scale, ncm_model=ncm_band, mean_speed=sp.mean(),
                     methods=["E_GEVD-MUSIC_rpmNCM"])
        az, el = r["E_GEVD-MUSIC_rpmNCM"]
        if al is not None:
            gaz, gel, ok = dregon_io.source_dir_at(rec, np.array([t_mid]))
            a2, e2 = loc.apply_alignment(az, el, al)
            if ok[0]:
                errs.append(float(loc.angular_error(a2, e2, gaz[0], gel[0])))
        a = steering(freqs, az, el)
        lo = (s + w / 2) / fs - config.FRAME_HOP_SEC / 2
        hi = lo + config.FRAME_HOP_SEC
        if k == 0:
            lo = -1
        if k == len(starts) - 1:
            hi = np.inf
        sel = (frame_t >= lo) & (frame_t < hi)
        Y_das[:, sel] = apply(das_weights(a), X[:, :, sel])
        Y_mvdr[:, sel] = apply(mvdr_weights(a, ncm_full.for_speed(sp.mean())), X[:, :, sel])

    sigs = {"input_mic1": x[:, 0], "output_delay_and_sum": istft(Y_das, fs)[:n],
            "output_mvdr_rpm": istft(Y_mvdr, fs)[:n]}
    # same loudness for fair listening, then one common gain so nothing clips
    ref = np.sqrt(np.mean(sigs["input_mic1"] ** 2))
    sigs = {k: v * ref / (np.sqrt(np.mean(v ** 2)) + 1e-12) for k, v in sigs.items()}
    peak = max(np.max(np.abs(v)) for v in sigs.values())
    os.makedirs(common.results_path("audio"), exist_ok=True)
    for k, v in sigs.items():
        sf.write(common.results_path("audio", f"exp4_{k}.wav"), (0.9 * v / peak).astype(np.float32), fs)

    titles = {"input_mic1": "Input, microphone 1 (rotor noise + loudspeaker)",
              "output_delay_and_sum": "Output, delay-and-sum steered at E's estimate (no noise model)",
              "output_mvdr_rpm": "Output, MVDR with RPM-binned noise model steered at E's estimate"}
    fig, axes = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True)
    for ax, (k, v) in zip(axes, sigs.items()):
        f, t, Z = scipy.signal.stft(v, fs=fs, nperseg=2048, noverlap=1536)
        kk = f <= 6000
        ax.pcolormesh(t + start_s, f[kk], 20 * np.log10(np.abs(Z[kk]) + 1e-9), shading="auto",
                      cmap="magma", vmin=-110, vmax=-40)
        ax.set(ylabel="frequency (Hz)", title=titles[k])
        ax.grid(False)
    axes[-1].set_xlabel(f"time in {LISTEN_FLIGHT} (s)")
    fig.tight_layout()
    fig.savefig(common.results_path("exp4_spectrograms.png"))
    plt.close(fig)

    out = {"flight": LISTEN_FLIGHT, "start_s": start_s, "seconds": seconds,
           "windows": len(starts), "E_within_10deg_in_clip": float(np.mean(np.array(errs) <= 10)) if errs else None,
           "E_median_err_in_clip_deg": float(np.median(errs)) if errs else None}
    # tonal hum reduction, rotor band below 1 kHz versus the rest (same loudness normalisation as the WAVs)
    for k, v in sigs.items():
        f, P = scipy.signal.welch(v, fs=fs, nperseg=4096)
        out[f"{k}_share_below_1kHz"] = float(P[(f > 50) & (f < 1000)].sum() / P[(f > 50) & (f < 8000)].sum())
    print(f"  listening example, {seconds:.0f} s of {LISTEN_FLIGHT} from {start_s:.0f} s, "
          f"E within 10 deg in {100 * (out['E_within_10deg_in_clip'] or 0):.0f}% of windows")
    return out


# ---------------------------------------------------------------------------
# Part 2, objective SNR gain on mixtures with a known clean signal
# ---------------------------------------------------------------------------
def snr_gain(locz, scale, plt):
    rng = np.random.default_rng(config.RANDOM_SEED)
    noise_rec = dregon_io.load_recording(config.NOISE_ONLY_FLIGHT)
    fs = noise_rec["fs"]
    N = len(noise_rec["audio"])
    half = N // 2
    ncm_band = common.build_ncm_model(noise_rec, locz.sel, sample_range=(0, half))
    ncm_full = FullBandNoiseModel(noise_rec, sample_range=(0, half))
    freqs = np.fft.rfftfreq(NFFT, 1 / fs)
    w = int(1.0 * fs)

    starts = np.arange(half, N - w, w // 2)
    t_mid = noise_rec["audio_ts"][starts + w // 2]
    sp_all = dregon_io.speeds_at(noise_rec, t_mid)
    good = sp_all.mean(axis=1) > 0.3 * np.median(sp_all.mean(axis=1))
    starts, sp_all = starts[good], sp_all[good]

    rows = []
    # signal and noise power per frequency, summed over trials, for the SNR-by-frequency figure
    spec = {k: [np.zeros(len(freqs)), np.zeros(len(freqs))] for k in ("input", "delay_and_sum", "mvdr_global", "mvdr_rpm")}
    for az_gt, el_gt, dist, p in dregon_io.list_clean_recordings():
        x, _ = dregon_io.read_wav(p)
        mid = len(x) // 2
        seg = x[max(0, mid - w // 2): max(0, mid - w // 2) + w]
        if len(seg) < w:
            continue
        for j in range(config.SEGMENTS_PER_DIRECTION):
            k = rng.integers(len(starts))
            noise = noise_rec["audio"][starts[k]: starts[k] + w]
            sp = sp_all[k]
            g = np.sqrt(np.mean(noise ** 2) / np.mean(seg ** 2) * 10 ** (MIX_SNR_DB / 10))
            S, Nn = stft(seg * g, fs), stft(noise, fs)
            r = locz.run(seg * g + noise, f0s=sp * scale, ncm_model=ncm_band, mean_speed=sp.mean(),
                         methods=["E_GEVD-MUSIC_rpmNCM"])
            a = steering(freqs, *r["E_GEVD-MUSIC_rpmNCM"])
            snr_in = 10 * np.log10(np.mean([band_power(S[m], freqs) / band_power(Nn[m], freqs) for m in range(S.shape[0])]))
            row = {"az": az_gt, "el": el_gt, "dist": dist, "segment": j, "snr_in_db": snr_in}
            spec["input"][0] += np.mean(np.abs(S) ** 2, axis=(0, 2))
            spec["input"][1] += np.mean(np.abs(Nn) ** 2, axis=(0, 2))
            for name, wts in (("delay_and_sum", das_weights(a)),
                              ("mvdr_global", mvdr_weights(a, ncm_full.global_ncm)),
                              ("mvdr_rpm", mvdr_weights(a, ncm_full.for_speed(sp.mean())))):
                YS, YN = apply(wts, S), apply(wts, Nn)
                snr_out = 10 * np.log10(band_power(YS, freqs) / band_power(YN, freqs))
                row[f"{name}_gain_db"] = snr_out - snr_in
                spec[name][0] += np.mean(np.abs(YS) ** 2, axis=1)
                spec[name][1] += np.mean(np.abs(YN) ** 2, axis=1)
            rows.append(row)

    with open(common.results_path("exp4_snr_gain.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    names = [("delay_and_sum", "Delay-and-sum", "#7f7f7f"), ("mvdr_global", "MVDR, one average NCM", "#ff7f0e"),
             ("mvdr_rpm", "MVDR, RPM-binned NCM", "#d62728")]
    out = {"input_snr_db": MIX_SNR_DB, "trials": len(rows), "band_hz": list(GAIN_BAND_HZ)}
    for key, _, _ in names:
        v = np.array([r[f"{key}_gain_db"] for r in rows])
        out[key] = {"median_gain_db": float(np.median(v)), "mean_gain_db": float(np.mean(v)),
                    "p10_db": float(np.percentile(v, 10)), "p90_db": float(np.percentile(v, 90))}

    fig, ax = plt.subplots(figsize=(7, 3.4))
    data = [[r[f"{k}_gain_db"] for r in rows] for k, _, _ in names]
    bp = ax.boxplot(data, orientation="horizontal", patch_artist=True, widths=0.5)
    for patch, (_, _, c) in zip(bp["boxes"], names):
        patch.set(facecolor=c, alpha=0.6)
    ax.set_yticks(range(1, len(names) + 1), [lab for _, lab, _ in names])
    ax.axvline(0, color="k", lw=0.8)
    ax.set(xlabel="SNR gain over a single microphone (dB)",
           title=f"SNR gain, {len(rows)} mixtures at {MIX_SNR_DB} dB input, steered at E's estimate")
    ax.title.set_fontsize(9)
    fig.tight_layout()
    fig.savefig(common.results_path("exp4_snr_gain.png"))
    plt.close(fig)

    # SNR by frequency in third-octave bands, before (single mic) and after each beamformer
    centres = 1000 * 2.0 ** (np.arange(-10, 10) / 3)
    centres = centres[(centres >= 100) & (centres <= 8000)]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    curves = [("input", "Input, single microphone", "k", "--")] + [(k, lab, c, "-") for k, lab, c in names]
    for key, lab, c, ls in curves:
        ps, pn = spec[key]
        snr = [10 * np.log10(ps[(freqs >= fc / 2 ** (1 / 6)) & (freqs < fc * 2 ** (1 / 6))].sum() /
                             pn[(freqs >= fc / 2 ** (1 / 6)) & (freqs < fc * 2 ** (1 / 6))].sum()) for fc in centres]
        ax.semilogx(centres, snr, ls, color=c, lw=1.8, label=lab)
        out.setdefault("snr_by_third_octave_db", {})[key] = dict(zip([f"{fc:.0f}" for fc in centres], map(float, snr)))
    ax.set(xlabel="frequency (Hz)", ylabel="SNR (dB)", title="Before and after, SNR by frequency (higher is better)")
    ax.set_xticks([125, 250, 500, 1000, 2000, 4000, 8000], ["125", "250", "500", "1k", "2k", "4k", "8k"])
    ax.set_xticks([], minor=True)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(common.results_path("exp4_snr_by_frequency.png"))
    plt.close(fig)
    print(f"  SNR gain at {MIX_SNR_DB} dB input, median over {len(rows)} mixtures: " +
          ", ".join(f"{lab} {out[k]['median_gain_db']:+.1f} dB" for k, lab, _ in names))
    return out, ncm_band, ncm_full


# ---------------------------------------------------------------------------
# Part 3, tracking figure from experiment 2
# ---------------------------------------------------------------------------
def tracking_figure(plt):
    p = common.results_path("exp2_per_window.csv")
    if not os.path.exists(p):
        print("  tracking figure skipped, run experiment 2 first")
        return
    with open(p) as f:
        rows = list(csv.DictReader(f))
    flights = list(dict.fromkeys(r["flight"] for r in rows))
    fig, axes = plt.subplots(2, len(flights), figsize=(5.2 * len(flights), 5.6), sharex="col", squeeze=False)
    for c, name in enumerate(flights):
        rr = [r for r in rows if r["flight"] == name]
        t = np.array([float(r["time_s"]) for r in rr])
        for row_i, key in enumerate(("az", "el")):
            ax = axes[row_i, c]
            ax.plot(t, [float(r[f"gt_{key}"]) for r in rr], "-", color="k", lw=1.5, label="true (Vicon)")
            for m in ("A_SRP-PHAT", "E_GEVD-MUSIC_rpmNCM"):
                ax.plot(t, [float(r[f"{m}_{key}"]) for r in rr], ".", ms=4, color=common.METHOD_COLORS[m],
                        label=common.METHOD_LABELS[m])
            ax.set(ylabel=("azimuth" if key == "az" else "elevation") + " (deg)",
                   ylim=(-180, 180) if key == "az" else (-90, 90))
            if row_i == 0:
                ax.set_title(name.replace("DREGON_", "").replace("_room1", ""), fontsize=9)
        axes[1, c].set_xlabel("time (s)")
    axes[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Where the sound is, over time: truth vs baseline vs motor-informed noise model", fontsize=10)
    fig.tight_layout()
    fig.savefig(common.results_path("exp4_tracking.png"))
    plt.close(fig)


def main():
    plt = common.plot_style()
    print("Experiment 4  input versus output")
    noise_rec = dregon_io.load_recording(config.NOISE_ONLY_FLIGHT)
    scale = common.get_speed_scale(noise_rec)
    del noise_rec
    locz = loc.Localiser(config.FS)
    metrics = {}
    metrics["snr_gain"], _, _ = snr_gain(locz, scale, plt)
    # full noise flight for the listening example, as in experiment 2
    noise_rec = dregon_io.load_recording(config.NOISE_ONLY_FLIGHT)
    ncm_band = common.build_ncm_model(noise_rec, locz.sel)
    ncm_full = FullBandNoiseModel(noise_rec)
    del noise_rec
    metrics["listening"] = listening_demo(locz, ncm_band, ncm_full, scale, plt)
    tracking_figure(plt)
    common.save_json(metrics, "exp4_metrics.json")
    print("  figures and audio written to", config.RESULTS_DIR)
    return metrics


if __name__ == "__main__":
    main()
