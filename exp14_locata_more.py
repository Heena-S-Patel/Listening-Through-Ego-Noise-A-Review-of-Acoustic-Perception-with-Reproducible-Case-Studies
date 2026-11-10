"""
Experiment 14. Does the loss from a noise model (RQ3) carry over to more LOCATA data and to an array without a robot head?

Same protocol as experiment 5, on tasks 1, 3 and 5, with one added method:
    F  SRP-PHAT after a multichannel Wiener filter, with the noise covariance of method D (see experiment 12)

    head_dev     robot head, development set, the recordings of the LOCATA table
    head_eval    robot head, evaluation set, which experiment 5 does not use
    eigen_dev    32-microphone spherical Eigenmike in the same room, development set
    eigen_eval   Eigenmike, evaluation set

LOCATA_ROOT is the folder that holds dev/ and eval/; the default is data/LOCATA.
Usage: python exp14_locata_more.py [part ...]     Output in results_review/: exp14_<part>.json
"""
import glob
import json
import os
import sys

os.environ.setdefault("RESULTS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_review"))

import numpy as np

import common
import config
import exp5_locata as e5
import exp12_dregon_controls as e12
import localisation as loc

ROOT = os.environ.get("LOCATA_ROOT", os.path.join(os.path.dirname(config.DATA_DIR), "LOCATA"))
PARTS = {"head_dev": ("benchmark2", "dev"), "head_eval": ("benchmark2", "eval"), "eigen_dev": ("eigenmike", "dev"), "eigen_eval": ("eigenmike", "eval")}
METHODS = ("A", "C", "D", "F")


def noise_frames(X, vad, hop, nfft, fs, guard_sec):
    """STFT frames in which the source is silent and has been silent for guard_sec, as in experiment 5."""
    g = int(guard_sec * fs)
    busy = np.convolve(vad > 0, np.ones(2 * g + 1), mode="same") > 0 if g > 0 else vad > 0
    starts = np.arange(X.shape[2]) * hop
    return np.array([s + nfft <= len(vad) and not busy[s:s + nfft].any() for s in starts])


def run(part):
    array, split = PARTS[part]
    e5.ARRAY = array
    az, el = loc.direction_grid()
    rows, recordings = [], []
    for task in e5.TASKS:
        for rec_dir in sorted(glob.glob(os.path.join(ROOT, split, f"task{task}", "recording*", array))):
            try:
                rec = e5.load_recording(rec_dir)
            except (OSError, ValueError) as err:
                print("skipped", rec_dir, err, flush=True)
                continue
            fs = rec["fs"]
            _, _, mics0 = e5.array_pose(rec, rec["t"][0])
            su = e12.Setup(fs, config.NFFT, config.HOP, config.BAND_HZ, az, el, mic_pos=mics0)
            sub_sel = np.arange(0, len(su.sel), config.MUSIC_BIN_STEP)
            sub = loc.SubspaceLocaliser(mics0, su.freqs[su.sel][sub_sel], az, el)
            X = su.stft(rec["x"])
            silent = noise_frames(X, rec["vad"], config.HOP, config.NFFT, fs, e5.GUARD_SEC)
            guard = e5.GUARD_SEC
            if silent.sum() < e5.MIN_NOISE_FRAMES:
                silent, guard = noise_frames(X, rec["vad"], config.HOP, config.NFFT, fs, 0.0), 0.0
            if silent.sum() == 0:
                print("no silent frames", rec_dir, flush=True)
                continue
            Rn = e12.covariance(X[:, su.sel][:, :, silent])
            del X
            starts, w = common.window_starts(len(rec["x"]), fs)
            name = os.path.basename(os.path.dirname(rec_dir))
            kept = 0
            for s in starts:
                if rec["vad"][s:s + w].mean() < e5.ACTIVE:
                    continue
                t_mid = rec["t"][s + w // 2]
                pos, R, _ = e5.array_pose(rec, t_mid)
                gaz, gel = e5.source_dir_local(rec, t_mid, pos, R)
                Xw = su.stft(rec["x"][s:s + w])
                Rx = e12.covariance(Xw[:, su.sel][:, sub_sel])
                est = {"A": su.srp_of(Xw), "C": su.peak(sub.spectrum(Rx)), "D": su.peak(sub.spectrum(Rx, Rn[sub_sel])), "F": su.filtered(Xw, Rn)}
                row = {"task": task, "recording": name}
                row.update({m: float(loc.angular_error(*v, gaz, gel)) for m, v in est.items()})
                rows.append(row)
                kept += 1
            recordings.append({"task": task, "recording": name, "windows": kept, "noise_frames": int(silent.sum()), "guard_s": guard})
            print(part, recordings[-1], flush=True)
            del rec
    summary = {}
    for task in e5.TASKS:
        rr = [r for r in rows if r["task"] == task]
        if rr:
            summary[str(task)] = {"n": len(rr), "recordings": len({r["recording"] for r in rr}),
                                  **{m: float(100 * np.mean([r[m] <= 10 for r in rr])) for m in METHODS},
                                  **{f"{m}_median": float(np.median([r[m] for r in rr])) for m in METHODS}}
    common.save_json({"summary": summary, "recordings": recordings, "rows": rows}, f"exp14_{part}.json")
    print(part, json.dumps(summary), flush=True)


if __name__ == "__main__":
    for p in sys.argv[1:] or PARTS:
        run(p)
