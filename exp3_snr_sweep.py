"""
Experiment 3. How does each method cope as the drone gets louder?

DREGON provides clean loudspeaker recordings (array on a tripod, named AZ_EL_DIST.wav)
and real in-flight noise. Sound adds up, so we can mix them at any chosen SNR and
know the true direction exactly. This gives a fair, repeatable test from 0 dB down
to -25 dB.

To keep training and test noise apart, the noise covariance is learned from the FIRST half of the
noise-only flight and the noise used for mixing is taken from the SECOND half.
"""
import csv

import numpy as np

import common
import config
import dregon_io
import localisation as loc


def main():
    plt = common.plot_style()
    print("Experiment 3  SNR sweep with clean recordings + real ego-noise")
    rng = np.random.default_rng(config.RANDOM_SEED)
    noise_rec = dregon_io.load_recording(config.NOISE_ONLY_FLIGHT)
    fs = noise_rec["fs"]
    scale = common.get_speed_scale(noise_rec)
    locz = loc.Localiser(fs)
    N = len(noise_rec["audio"])
    half = N // 2
    ncm = common.build_ncm_model(noise_rec, locz.sel, sample_range=(0, half))

    clean = dregon_io.list_clean_recordings()
    if not clean:
        raise FileNotFoundError(f"No clean recordings found ({config.CLEAN_RECORDINGS})")
    w = int(1.0 * fs)

    # frame alignment on the clean files themselves (no noise added)
    est, gt, sigs = [], [], []
    for az, el, dist, p in clean:
        x, _ = dregon_io.read_wav(p)
        mid = len(x) // 2
        seg = x[max(0, mid - w // 2): max(0, mid - w // 2) + w]
        if len(seg) < w:
            continue
        sigs.append((az, el, seg))
        r = locz.run(seg, methods=["A_SRP-PHAT"])
        est.append(r["A_SRP-PHAT"])
        gt.append((az, el))
    est, gt = np.array(est), np.array(gt)
    al = loc.fit_frame_alignment(est[:, 0], est[:, 1], gt[:, 0], gt[:, 1])
    common.save_json(al, "frame_alignment_clean.json")
    print(f"  {len(sigs)} clean recordings, frame alignment offset {al['offset']} deg, "
          f"median clean error {al['median_err']:.1f} deg")

    # candidate noise segments from the second half, with running motors
    starts = np.arange(half, N - w, w // 2)
    t_mid = noise_rec["audio_ts"][starts + w // 2]
    sp = dregon_io.speeds_at(noise_rec, t_mid)
    good = sp.mean(axis=1) > 0.3 * np.median(sp.mean(axis=1))
    starts, sp = starts[good], sp[good]

    rows = []
    for az, el, seg in sigs:
        p_s = np.mean(seg ** 2)
        for j in range(config.SEGMENTS_PER_DIRECTION):
            k = rng.integers(len(starts))
            noise = noise_rec["audio"][starts[k]: starts[k] + w]
            p_n = np.mean(noise ** 2)
            for snr in config.SNR_LIST_DB:
                mix = noise + seg * np.sqrt(p_n / p_s * 10 ** (snr / 10))
                res = locz.run(mix, f0s=sp[k] * scale, ncm_model=ncm, mean_speed=sp[k].mean())
                row = {"gt_az": az, "gt_el": el, "segment": j, "snr_db": snr}
                for m, (a, e) in res.items():
                    a2, e2 = loc.apply_alignment(a, e, al)
                    row[f"{m}_err"] = float(loc.angular_error(a2, e2, az, el))
                rows.append(row)
        print(f"    direction az {az:6.1f} el {el:5.1f} done")

    with open(common.results_path("exp3_per_trial.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    methods = [m for m in loc.METHODS if f"{m}_err" in rows[0]]
    summary = {"frame_alignment": al, "by_snr": {}}
    for snr in config.SNR_LIST_DB:
        rr = [r for r in rows if r["snr_db"] == snr]
        summary["by_snr"][str(snr)] = {m: loc.summarise_errors([r[f"{m}_err"] for r in rr]) for m in methods}
    common.save_json(summary, "exp3_metrics.json")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for m in methods:
        acc = [100 * summary["by_snr"][str(s)][m]["acc@10deg"] for s in config.SNR_LIST_DB]
        med = [summary["by_snr"][str(s)][m]["median_deg"] for s in config.SNR_LIST_DB]
        axes[0].plot(config.SNR_LIST_DB, acc, "o-", color=common.METHOD_COLORS[m], label=common.METHOD_LABELS[m])
        axes[1].plot(config.SNR_LIST_DB, med, "o-", color=common.METHOD_COLORS[m], label=common.METHOD_LABELS[m])
    axes[0].set(xlabel="SNR (dB), drone gets louder to the right", ylabel="located within 10° (%)", ylim=(0, 105),
                title="Accuracy as the drone noise grows")
    axes[1].set(xlabel="SNR (dB)", ylabel="median error (deg)", title="Median error as the drone noise grows")
    for ax in axes:
        ax.invert_xaxis()
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(common.results_path("exp3_snr_sweep.png"))
    plt.close(fig)

    print(f"\n  Located within 10 degrees (%)")
    print("  " + f"{'method':38s}" + "".join(f"{s:>7d}dB" for s in config.SNR_LIST_DB))
    for m in methods:
        print("  " + f"{common.METHOD_LABELS[m]:38s}" +
              "".join(f"{100 * summary['by_snr'][str(s)][m]['acc@10deg']:8.0f}%" for s in config.SNR_LIST_DB))
    print("  figures written to", config.RESULTS_DIR)
    return summary


if __name__ == "__main__":
    main()
