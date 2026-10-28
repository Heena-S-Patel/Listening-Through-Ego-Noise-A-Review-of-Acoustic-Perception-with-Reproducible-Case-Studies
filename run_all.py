"""
Run all case studies and write results/summary.md.

    python run_all.py           # full run on DREGON
    python run_all.py --quick   # faster, uses every 4th window in experiment 2
"""
import argparse
import datetime
import os
import time

import common
import config
import exp1_ego_noise
import exp2_flight_localisation
import exp3_snr_sweep
import exp4_input_output
import exp5_locata
import exp6_multidrone_egonoise
import exp7_tonal_vs_flight
import localisation as loc


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main(quick=False):
    t0 = time.time()
    m1 = exp1_ego_noise.main()
    m2 = exp2_flight_localisation.main(quick=quick)
    try:
        m3 = exp3_snr_sweep.main()
    except FileNotFoundError as e:
        print("  experiment 3 skipped,", e)
        m3 = None
    try:
        m4 = exp4_input_output.main()
    except FileNotFoundError as e:
        print("  experiment 4 skipped,", e)
        m4 = None
    try:
        m5 = exp5_locata.main()
    except (FileNotFoundError, IndexError, OSError) as e:
        print("  experiment 5 skipped (LOCATA robot-head data not found, see tools/locata_partial.py):", e)
        m5 = None
    try:
        m6 = exp6_multidrone_egonoise.main()
    except (FileNotFoundError, OSError, RuntimeError) as e:
        print("  experiment 6 skipped (SP Cup ego-noise data not found):", e)
        m6 = None
    m7 = exp7_tonal_vs_flight.main() if m6 else None

    L = []
    L.append("# Results summary")
    L.append(f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M} from the data in `{os.path.basename(config.DATA_DIR)}`.")
    L.append("")
    L.append("## Experiment 1. How much of the ego-noise does the motor log predict?")
    L.append(f"Motor speeds used are the **{m1['motor_speed_kind']}** values, converted with scale "
             f"{m1['speed_scale']:.4g} to turns per second.")
    rows = [(r["band_hz"] + " Hz", f"{100 * r['energy_share_bpf_tones']:.1f}%", f"{100 * r['bin_share_bpf_tones']:.1f}%",
             f"{100 * r['energy_share_all_shaft_tones']:.1f}%", f"{100 * r['bin_share_all_shaft_tones']:.1f}%")
            for r in m1["energy_by_band"]]
    L.append(md_table(["band", "energy on blade tones", "bins they cover", "energy on all shaft tones", "bins they cover"], rows))
    L.append("")
    L.append("Reading the table. If a small share of bins holds a large share of energy, the motor log explains that "
             "band well. Where the two numbers are close, the noise is broadband and RPM alone does not help much.")
    if "imu_rate_hz" in m1:
        L.append("")
        L.append(f"The flight controller IMU is logged at about {m1['imu_rate_hz']:.0f} Hz, so it cannot see vibration "
                 f"above {m1['imu_nyquist_hz']:.0f} Hz. The lowest blade passing tone is about "
                 f"{m1['lowest_blade_passing_freq_hz']:.0f} Hz, so the IMU can follow the first few rotor tones but "
                 f"not the {config.BAND_HZ[0]:.0f} to {config.BAND_HZ[1]:.0f} Hz band used for localisation. "
                 f"Sensing vibration there needs a dedicated high-rate sensor.")
    L.append("")
    L.append("![](exp1_spectrogram_rpm.png)")
    L.append("![](exp1_energy_by_band.png)")
    L.append("![](exp1_coherence.png)")
    L.append("")
    L.append("## Experiment 2. In-flight localisation against Vicon ground truth")
    rows = []
    for m, s in m2["overall"].items():
        rows.append((common.METHOD_LABELS[m], f"{s['median_deg']:.1f}°", f"{100 * s['acc@10deg']:.1f}%",
                     f"{100 * s['acc@20deg']:.1f}%", s["n"]))
    L.append(md_table(["method", "median error", "within 10°", "within 20°", "windows"], rows))
    L.append("")
    L.append("![](exp2_accuracy.png)")
    L.append("![](exp2_error_cdf.png)")
    L.append("![](exp2_example_maps.png)")
    if m3:
        L.append("")
        L.append("## Experiment 3. Accuracy (within 10°) as the drone gets louder")
        methods = list(next(iter(m3["by_snr"].values())).keys())
        rows = [[common.METHOD_LABELS[m]] + [f"{100 * m3['by_snr'][str(s)][m]['acc@10deg']:.0f}%"
                                             for s in config.SNR_LIST_DB] for m in methods]
        L.append(md_table(["method"] + [f"{s} dB" for s in config.SNR_LIST_DB], rows))
        L.append("")
        L.append("![](exp3_snr_sweep.png)")
    if m4:
        g = m4["snr_gain"]
        L.append("")
        L.append("## Experiment 4. Input versus output")
        L.append(f"Beamformers steered at method E's estimated direction. SNR gain over a single microphone, "
                 f"{g['trials']} clean-plus-flight-noise mixtures at {g['input_snr_db']} dB input, "
                 f"{g['band_hz'][0]:.0f} to {g['band_hz'][1]:.0f} Hz.")
        rows = [(lab, f"{g[k]['median_gain_db']:+.1f} dB", f"{g[k]['p10_db']:+.1f} to {g[k]['p90_db']:+.1f} dB")
                for k, lab in (("delay_and_sum", "Delay-and-sum"), ("mvdr_global", "MVDR, one average NCM"),
                               ("mvdr_rpm", "MVDR, RPM-binned NCM"))]
        L.append(md_table(["beamformer", "median gain", "10th to 90th percentile"], rows))
        li = m4["listening"]
        L.append("")
        L.append(f"Listening example: {li['seconds']:.0f} s of `{li['flight']}` from {li['start_s']:.0f} s, in `audio/`. "
                 f"Method E is within 10° in {100 * li['E_within_10deg_in_clip']:.0f}% of the windows in this clip.")
        L.append("")
        L.append("![](exp4_tracking.png)")
        L.append("![](exp4_snr_gain.png)")
        L.append("![](exp4_snr_by_frequency.png)")
        L.append("![](exp4_spectrograms.png)")
    if m5:
        L.append("")
        L.append("## Experiment 5. Robot-head array in a reverberant room (LOCATA)")
        L.append("Share of windows within 10° / median error, single-source development tasks.")
        rows = [[exp5_locata.LABELS[m]] + [f"{100 * m5['by_task'][t][m]['acc@10deg']:.1f}% / "
                                           f"{m5['by_task'][t][m]['median_deg']:.1f}°" for t in ("1", "3", "5")]
                for m in exp5_locata.METHODS]
        L.append(md_table(["method", "task 1", "task 3", "task 5"], rows))
        L.append("")
        L.append("![](exp5_locata.png)")
    if m6:
        L.append("")
        L.append("## Experiment 6. Ego-noise across five drones")
        rows = [(r["drone"], f"{100 * r['tonal_share']:.1f}%", f"{r['learned_rejection_db']:.1f} dB",
                 f"{r['diffuse_rejection_db']:.1f} dB" if r["diffuse_rejection_db"] is not None else "geometry not published")
                for r in m6]
        L.append(md_table(["drone", "tonal share", "learned rejection", "diffuse reference"], rows))
        L.append("")
        L.append("![](exp6_multidrone.png)")
    if m7:
        L.append("")
        L.append("## Experiment 7. Rotor-speed variation and tonal share (DREGON)")
        L.append(f"{m7['windows']} steady windows of 1 s. Spearman correlation between within-window rotor-speed "
                 f"std and tonal share: rho = {m7['spearman_jitter']['rho']:.2f}, p = {m7['spearman_jitter']['p']:.3f}. "
                 f"Tonal share {100 * m7['tonal_share_calmest_quarter']:.1f}% in the calmest quarter of windows and "
                 f"{100 * m7['tonal_share_most_jittery_quarter']:.1f}% in the most variable quarter, at mean rotor "
                 f"speeds of {m7['mean_speed_calmest_quarter_rps']:.1f} and {m7['mean_speed_most_jittery_quarter_rps']:.1f} rev/s.")
        L.append("")
        L.append("![](exp7_jitter_vs_tonal_share.png)")
        L.append("![](exp7_smeared_harmonics.png)")
    L.append("")
    L.append(f"Total run time {(time.time() - t0) / 60:.1f} minutes.")
    with open(common.results_path("summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\nSummary written to", common.results_path("summary.md"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(quick=ap.parse_args().quick)
