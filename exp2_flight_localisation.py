"""
Experiment 2. Where is the sound coming from, during real flight?

Compares five localisation methods on the DREGON free flight recordings with a
loudspeaker source, scored against Vicon ground truth. Two of the methods use
the motor log (B and E).

Steps
    1. Learn the noise covariance (global and per motor speed bin) from the noise-only flight.
    2. Align the array frame with the ground truth frame using the silent flight (motors off).
    3. Run every method on 0.5 s windows of each source flight and compare with ground truth.

Usage
    python exp2_flight_localisation.py            # full run
    python exp2_flight_localisation.py --quick    # every 4th window, for a fast check
"""
import argparse
import csv

import numpy as np

import common
import config
import dregon_io
import localisation as loc


def calibrate(locz, rec_cal):
    starts, w = common.window_starts(len(rec_cal["audio"]), rec_cal["fs"])
    est, gt = [], []
    for s in starts:
        t_mid = rec_cal["audio_ts"][s + w // 2]
        az, el, ok = dregon_io.source_dir_at(rec_cal, np.array([t_mid]))
        if not ok[0]:
            continue
        r = locz.run(rec_cal["audio"][s:s + w], methods=["A_SRP-PHAT"])
        est.append(r["A_SRP-PHAT"])
        gt.append((az[0], el[0]))
    est, gt = np.array(est), np.array(gt)
    return loc.fit_frame_alignment(est[:, 0], est[:, 1], gt[:, 0], gt[:, 1])


def main(quick=False):
    plt = common.plot_style()
    print("Experiment 2  in-flight localisation")
    noise_rec = dregon_io.load_recording(config.NOISE_ONLY_FLIGHT)
    fs = noise_rec["fs"]
    scale = common.get_speed_scale(noise_rec)
    locz = loc.Localiser(fs)
    print("  learning noise covariance from the noise-only flight")
    ncm = common.build_ncm_model(noise_rec, locz.sel)
    del noise_rec

    # frame alignment
    try:
        rec_cal = dregon_io.load_recording(config.CALIBRATION_FLIGHT)
        al = calibrate(locz, rec_cal)
        al["source"] = config.CALIBRATION_FLIGHT
        del rec_cal
    except FileNotFoundError:
        print("  silent flight not found, aligning frames on the first 20% of the first source flight instead")
        rec0 = dregon_io.load_recording(config.SOURCE_FLIGHTS[0])
        n = int(0.2 * len(rec0["audio"]))
        rec0["audio"] = rec0["audio"][:n]
        al = calibrate(locz, rec0)
        al["source"] = config.SOURCE_FLIGHTS[0] + " (first 20%)"
        del rec0
    print(f"  frame alignment  offset {al['offset']} deg, az sign {al['az_sign']}, el sign {al['el_sign']}, "
          f"median error after alignment {al['median_err']:.1f} deg")
    common.save_json(al, "frame_alignment_flight.json")

    rows = []
    example = None
    for name in config.SOURCE_FLIGHTS:
        try:
            rec = dregon_io.load_recording(name)
        except FileNotFoundError:
            print(f"  skipping {name} (not found)")
            continue
        starts, w = common.window_starts(len(rec["audio"]), fs)
        if quick:
            starts = starts[::4]
        t_mid = rec["audio_ts"][starts + w // 2]
        gaz, gel, ok = dregon_io.source_dir_at(rec, t_mid)
        speeds = dregon_io.speeds_at(rec, t_mid)
        mean_sp = speeds.mean(axis=1)
        running = mean_sp > 0.3 * np.median(mean_sp)
        keep = np.where(ok & running)[0]
        print(f"  {name}  {len(keep)} windows")
        for n_done, i in enumerate(keep):
            s = starts[i]
            xw = rec["audio"][s:s + w]
            res = locz.run(xw, f0s=speeds[i] * scale, ncm_model=ncm, mean_speed=mean_sp[i])
            row = {"flight": name, "time_s": float(t_mid[i] - rec["audio_ts"][0]), "gt_az": float(gaz[i]),
                   "gt_el": float(gel[i]), "mean_shaft_rate": float(mean_sp[i] * scale),
                   "rpm_mask_share": float(getattr(locz, "last_mask_share", np.nan))}
            for mth, (az, el) in res.items():
                a, e = loc.apply_alignment(az, el, al)
                row[f"{mth}_az"], row[f"{mth}_el"] = float(a), float(e)
                row[f"{mth}_err"] = float(loc.angular_error(a, e, gaz[i], gel[i]))
            rows.append(row)
            if example is None and n_done == len(keep) // 3:
                example = (xw, speeds[i] * scale, mean_sp[i], gaz[i], gel[i], name, row["time_s"])
            if (n_done + 1) % 50 == 0:
                print(f"    {n_done + 1}/{len(keep)}")
        del rec

    if not rows:
        raise RuntimeError("No source flights found")

    with open(common.results_path("exp2_per_window.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    methods = [m for m in loc.METHODS if f"{m}_err" in rows[0]]
    summary = {"frame_alignment": al, "per_flight": {}, "overall": {}}
    for m in methods:
        summary["overall"][m] = loc.summarise_errors([r[f"{m}_err"] for r in rows])
    for name in config.SOURCE_FLIGHTS:
        rr = [r for r in rows if r["flight"] == name]
        if rr:
            summary["per_flight"][name] = {m: loc.summarise_errors([r[f"{m}_err"] for r in rr]) for m in methods}
    common.save_json(summary, "exp2_metrics.json")

    print("\n  Overall results (lower error and higher accuracy are better)")
    print(f"  {'method':38s} {'median err':>10s} {'acc@10':>8s} {'acc@20':>8s}")
    for m in methods:
        s = summary["overall"][m]
        print(f"  {common.METHOD_LABELS[m]:38s} {s['median_deg']:9.1f}° {100 * s['acc@10deg']:7.1f}% {100 * s['acc@20deg']:7.1f}%")

    # figure, accuracy bars
    fig, ax = plt.subplots(figsize=(8, 3.6))
    xi = np.arange(len(methods))
    for k, t in enumerate(config.ACC_THRESHOLDS):
        vals = [100 * summary["overall"][m][f"acc@{t}deg"] for m in methods]
        bars = ax.bar(xi + (k - 0.5) * 0.38, vals, 0.36, color=[common.METHOD_COLORS[m] for m in methods],
                      alpha=1.0 if k == 0 else 0.45, label=f"within {t}°")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}", ha="center", fontsize=7)
    ax.set_xticks(xi, [common.METHOD_LABELS[m].split("  ")[1] for m in methods], rotation=15, ha="right", fontsize=8)
    ax.set(ylabel="windows located correctly (%)", ylim=(0, 105), title="In-flight localisation on DREGON")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(common.results_path("exp2_accuracy.png"))
    plt.close(fig)

    # figure, error CDF
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for m in methods:
        e = np.sort([r[f"{m}_err"] for r in rows])
        ax.plot(e, np.arange(1, len(e) + 1) / len(e) * 100, color=common.METHOD_COLORS[m], label=common.METHOD_LABELS[m])
    ax.set(xlabel="angular error (degrees)", ylabel="share of windows (%)", xlim=(0, 90),
           title="Error distribution (curves further left are better)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(common.results_path("exp2_error_cdf.png"))
    plt.close(fig)

    # figure, error over time for the first flight
    first = [r for r in rows if r["flight"] == rows[0]["flight"]]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for m in ("A_SRP-PHAT", "E_GEVD-MUSIC_rpmNCM"):
        if m in methods:
            ax.plot([r["time_s"] for r in first], [r[f"{m}_err"] for r in first], ".", ms=3,
                    color=common.METHOD_COLORS[m], label=common.METHOD_LABELS[m])
    ax.set(xlabel="time (s)", ylabel="angular error (deg)", ylim=(0, 180), title=f"Error over time, {rows[0]['flight']}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(common.results_path("exp2_error_over_time.png"))
    plt.close(fig)

    # figure, angular spectra for one example window
    if example is not None:
        xw, f0s, msp, gaz, gel, name, tsec = example
        X = common.window_stft(xw, fs)
        Rx = loc.spatial_cov(X, locz.sel)
        base_w = locz.band.astype(float)
        maps = {"A  SRP-PHAT (baseline)": locz.srp.spectrum(X, base_w),
                "E  GEVD-MUSIC, RPM-binned NCM": locz.sub.spectrum(Rx, ncm.for_speed(msp))}
        naz = len(np.unique(locz.az))
        nel = len(np.unique(locz.el))
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
        for ax, (lab, mp) in zip(axes, maps.items()):
            img = mp.reshape(naz, nel).T
            ax.imshow(img, origin="lower", aspect="auto", cmap="viridis",
                      extent=[locz.az.min(), locz.az.max(), locz.el.min(), locz.el.max()])
            # ground truth back in the array frame
            ga = (al["az_sign"] * (gaz - al["offset"]) + 180) % 360 - 180
            ge = al["el_sign"] * gel
            ax.plot(ga, ge, "w+", ms=14, mew=2, label="true direction")
            ax.set(title=lab, xlabel="azimuth (deg, array frame)", ylabel="elevation (deg)")
            ax.legend(fontsize=7, loc="upper right")
        fig.suptitle(f"Where each method thinks the sound is, {name} at {tsec:.1f} s", fontsize=10)
        fig.tight_layout()
        fig.savefig(common.results_path("exp2_example_maps.png"))
        plt.close(fig)

    print("  figures written to", config.RESULTS_DIR)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(quick=ap.parse_args().quick)
