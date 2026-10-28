"""
Experiment 1. What does drone ego-noise look like, and how much of it can the motor log predict?

Uses only the DREGON noise-only free flight (no target source).

Outputs in results/
    exp1_motor_speeds.png        the four rotor speeds during the flight
    exp1_spectrogram_rpm.png     spectrogram with the blade passing harmonics predicted from the motor log
    exp1_energy_by_band.png      share of noise energy sitting on predicted rotor tones, per frequency band
    exp1_coherence.png           how similar the noise is across microphones, per frequency
    exp1_metrics.json
"""
import numpy as np
import scipy.signal

import common
import config
import dregon_io
import rotor


def main():
    plt = common.plot_style()
    print("Experiment 1  ego-noise analysis")
    rec = dregon_io.load_recording(config.NOISE_ONLY_FLIGHT)
    fs = rec["fs"]
    scale = common.get_speed_scale(rec)
    ts, sp, kind = dregon_io.motor_speeds(rec)
    t0 = rec["audio_ts"][0]
    metrics = {"recording": rec["name"], "motor_speed_kind": kind, "speed_scale": scale,
               "duration_s": len(rec["audio"]) / fs}

    # 1. motor speeds
    fig, ax = plt.subplots(figsize=(8, 3))
    for m in range(sp.shape[1]):
        ax.plot(ts - t0, sp[:, m] * scale, lw=0.8, label=f"rotor {m + 1}")
    ax.set(xlabel="time (s)", ylabel="shaft rate (turns/s)", title="Rotor speeds from the motor log")
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(common.results_path("exp1_motor_speeds.png"))
    plt.close(fig)

    # 2. spectrogram of one microphone with predicted harmonics
    x = rec["audio"][:, 0]
    nfft, hop = 8192, 2048
    f, tt, Z = scipy.signal.stft(x, fs=fs, nperseg=nfft, noverlap=nfft - hop)
    P = np.abs(Z) ** 2
    t_abs = rec["audio_ts"][np.clip((tt * fs).astype(int), 0, len(x) - 1)]
    f0 = dregon_io.speeds_at(rec, t_abs) * scale                        # (frames, 4)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    fmax_plot = 2000
    k = f <= fmax_plot
    ax.pcolormesh(tt, f[k], 10 * np.log10(P[k] + 1e-12), shading="auto", cmap="magma")
    bpf = config.N_BLADES * f0.mean(axis=1)
    for h in range(1, 8):
        ax.plot(tt, h * bpf, color="cyan", lw=0.7, alpha=0.8, label="predicted blade passing harmonics" if h == 1 else None)
    ax.set(xlabel="time (s)", ylabel="frequency (Hz)", ylim=(0, fmax_plot),
           title="Microphone 1 during flight, with rotor tones predicted from the motor log")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(common.results_path("exp1_spectrogram_rpm.png"))
    plt.close(fig)

    # 3. how much energy sits on predicted tones, per band
    bands = [(50, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]
    running = f0.mean(axis=1) > 0.3 * np.median(f0.mean(axis=1))
    rows = []
    for lo, hi in bands:
        bmask = (f >= lo) & (f < hi)
        e_bpf = e_shaft = e_tot = b_bpf = b_shaft = 0.0
        for j in np.where(running)[0]:
            mb = rotor.comb_bins(f, f0[j], config.N_BLADES, hi, 1) & bmask
            ms = rotor.comb_bins(f, f0[j], 1, hi, 1) & bmask
            e_tot += P[bmask, j].sum()
            e_bpf += P[mb, j].sum()
            e_shaft += P[ms, j].sum()
            b_bpf += mb.sum() / bmask.sum()
            b_shaft += ms.sum() / bmask.sum()
        n = running.sum()
        rows.append({"band_hz": f"{lo}-{hi}", "energy_share_bpf_tones": e_bpf / e_tot, "bin_share_bpf_tones": b_bpf / n,
                     "energy_share_all_shaft_tones": e_shaft / e_tot, "bin_share_all_shaft_tones": b_shaft / n})
    metrics["energy_by_band"] = rows

    fig, ax = plt.subplots(figsize=(8, 3.6))
    xi = np.arange(len(rows))
    ax.bar(xi - 0.3, [r["energy_share_bpf_tones"] * 100 for r in rows], 0.2, label="energy on blade tones", color="#d62728")
    ax.bar(xi - 0.1, [r["bin_share_bpf_tones"] * 100 for r in rows], 0.2, label="bins covered by blade tones", color="#f4a3a3")
    ax.bar(xi + 0.1, [r["energy_share_all_shaft_tones"] * 100 for r in rows], 0.2, label="energy on all shaft tones", color="#1f77b4")
    ax.bar(xi + 0.3, [r["bin_share_all_shaft_tones"] * 100 for r in rows], 0.2, label="bins covered by all shaft tones", color="#a6c8e6")
    ax.set_xticks(xi, [r["band_hz"] + " Hz" for r in rows])
    ax.set(ylabel="percent", title="How much of the noise sits exactly where the motor log says it should")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(common.results_path("exp1_energy_by_band.png"))
    plt.close(fig)

    # 4. inter-microphone coherence of the ego-noise
    seg = rec["audio"][int(5 * fs): int(min(len(rec["audio"]) / fs - 5, 35) * fs)]
    cohs = []
    for i in range(8):
        for j in range(i + 1, 8):
            fc, c = scipy.signal.coherence(seg[:, i], seg[:, j], fs=fs, nperseg=2048)
            cohs.append(c)
    cohs = np.array(cohs)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    kk = fc <= 8000
    ax.plot(fc[kk], cohs.mean(0)[kk], color="k", lw=1, label="mean over 28 mic pairs")
    ax.fill_between(fc[kk], np.percentile(cohs, 10, 0)[kk], np.percentile(cohs, 90, 0)[kk], alpha=0.2, color="k",
                    label="10 to 90 percent range")
    ax.set(xlabel="frequency (Hz)", ylabel="coherence", ylim=(0, 1),
           title="How predictable the noise is from one microphone to another")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(common.results_path("exp1_coherence.png"))
    plt.close(fig)
    for lo, hi in bands:
        sel = (fc >= lo) & (fc < hi)
        metrics.setdefault("mean_coherence_by_band", {})[f"{lo}-{hi}"] = float(cohs.mean(0)[sel].mean())

    # 5. IMU rate check (can the flight controller IMU see rotor vibration at all?)
    if "imu" in rec:
        dt = np.median(np.diff(rec["imu"]["timestamps"].ravel()))
        imu_rate = 1.0 / dt if dt > 0 else float("nan")
        lowest_bpf = float(np.percentile(config.N_BLADES * f0[running].min(axis=1), 5))
        metrics["imu_rate_hz"] = imu_rate
        metrics["imu_nyquist_hz"] = imu_rate / 2
        metrics["lowest_blade_passing_freq_hz"] = lowest_bpf
        print(f"  IMU logged at about {imu_rate:.0f} Hz, so it can only see vibration below {imu_rate / 2:.0f} Hz. "
              f"The lowest blade passing tone is about {lowest_bpf:.0f} Hz.")

    common.save_json(metrics, "exp1_metrics.json")
    for r in rows:
        print(f"  {r['band_hz']:>10} Hz  energy on blade tones {100 * r['energy_share_bpf_tones']:5.1f}%  "
              f"(covering {100 * r['bin_share_bpf_tones']:4.1f}% of bins)")
    print("  figures written to", config.RESULTS_DIR)
    return metrics


if __name__ == "__main__":
    main()
