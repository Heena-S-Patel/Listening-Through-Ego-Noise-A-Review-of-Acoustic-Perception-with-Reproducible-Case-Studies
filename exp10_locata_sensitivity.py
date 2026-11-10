"""
Experiment 10. Why does the noise model lower accuracy on the LOCATA robot head?

Controls for the LOCATA table of the paper, on the same windows as experiment 5:
    frames      noise frames per recording, and the power of the guarded and plain silent frames
    loading     diagonal loading of D (large loading should bring D back towards MUSIC)
    guard       guard around speech when the noise frames are selected
    local       noise covariance from the silent frames nearest in time, since the array moves in task 5
    scaling     generalized eigenvectors scaled to unit norm (the paper) or left as returned by the solver
    background  tonal share and coherence of the silent frames, as a measurement of possible fan noise

Output in results_review/: exp10_locata.json
"""
import glob
import json
import os

os.environ.setdefault("RESULTS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_review"))

import numpy as np
import scipy.linalg
import scipy.signal

import common
import config
import exp5_locata as e5
import exp6_multidrone_egonoise as e6
import localisation as loc

LOADINGS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
GUARDS = (0.0, 0.25, 0.5, 1.0)
LOCAL_FRAMES = (100, 400)


def silent_mask(vad, n_frames, guard_sec, fs):
    g = int(guard_sec * fs)
    busy = np.convolve(vad > 0, np.ones(2 * g + 1), mode="same") > 0 if g > 0 else vad > 0
    starts = np.arange(n_frames) * config.HOP
    return np.array([not busy[s:s + config.NFFT].any() for s in starts if s + config.NFFT <= len(vad)])


def unscaled_spectrum(sub, Rx, Phi):
    """As the shared style of localisation.py, without the unit-norm scaling."""
    F, M, _ = Rx.shape
    out = np.zeros(sub.A.shape[1])
    for f in range(F):
        N = Phi[f] + config.DIAG_LOADING * np.real(np.trace(Phi[f])) / M * np.eye(M)
        _, V = scipy.linalg.eigh(Rx[f], N)
        p = 1.0 / (np.sum(np.abs(sub.A[f] @ V[:, :M - 1].conj()) ** 2, axis=1) + 1e-12)
        out += p / p.max()
    return out


def score(items, phi_of, spectrum=None):
    """Within 10 degrees per task."""
    errs = {}
    for it in items:
        phi = phi_of(it)
        spec = spectrum(it["sub"], it["Rx"], phi) if spectrum else it["sub"].spectrum(it["Rx"], phi)
        a, e = it["locz"].peak(spec)
        errs.setdefault(it["task"], []).append(float(loc.angular_error(a, e, it["gaz"], it["gel"])))
    return {str(t): float(100 * np.mean(np.array(v) <= 10)) for t, v in sorted(errs.items())}


def coherence(R):
    M = R.shape[1]
    p = np.real(np.einsum("fii->fi", R))
    iu = np.triu_indices(M, 1)
    return float(np.mean(np.abs(R[:, iu[0], iu[1]]) ** 2 / (p[:, iu[0]] * p[:, iu[1]])))


def main():
    items, recs = [], []
    for task in e5.TASKS:
        for rec_dir in sorted(glob.glob(os.path.join(e5.LOCATA_DIR, f"task{task}", "recording*", e5.ARRAY))):
            rec = e5.load_recording(rec_dir)
            fs = rec["fs"]
            _, _, mics0 = e5.array_pose(rec, rec["t"][0])
            locz = loc.Localiser(fs, mic_pos=mics0)
            _, X = loc.stft_multi(rec["x"], fs)
            Xs = X[:, locz.sel, :].astype(np.complex64)
            T = Xs.shape[2]
            info = {"task": task, "recording": os.path.basename(os.path.dirname(rec_dir)), "seconds": len(rec["x"]) / fs,
                    "frames_total": T, "phi": {}, "frames": {}, "power_db": {}}
            for g in GUARDS:
                m = silent_mask(rec["vad"], T, g, fs)
                info["frames"][str(g)] = int(m.sum())
                if m.sum():
                    Xn = Xs[:, :, :len(m)][:, :, m]
                    info["phi"][g] = np.einsum("mft,nft->fmn", Xn, Xn.conj()) / Xn.shape[2]
                    info["power_db"][str(g)] = float(10 * np.log10(np.mean(np.abs(Xn) ** 2)))
            # same rule as experiment 5: with too few guarded frames, fall back to plain silent frames
            used = 0.5 if info["frames"]["0.5"] >= e5.MIN_NOISE_FRAMES else 0.0
            info["used_guard"] = used
            info["coherence_silent"] = coherence(info["phi"][0.0])
            info["coherence_diffuse"] = coherence(e6.diffuse_covariance(mics0, locz.freqs[locz.sel]))
            # background measurement on the longest stretch of guarded silence
            sil = rec["vad"] == 0
            quiet = np.flatnonzero(np.diff(np.r_[0, sil.astype(int), 0]))
            spans = quiet.reshape(-1, 2)
            s0, s1 = spans[np.argmax(spans[:, 1] - spans[:, 0])]
            info["longest_silence_s"] = float((s1 - s0) / fs)
            if s1 - s0 > fs:
                info["tonal_share_silence"] = e6.tonal_share(rec["x"][s0:s1], fs)
            silent_idx = np.where(silent_mask(rec["vad"], T, used, fs))[0]
            starts, w = common.window_starts(len(rec["x"]), fs)
            kept = 0
            for s in starts:
                if rec["vad"][s:s + w].mean() < e5.ACTIVE:
                    continue
                t_mid = rec["t"][s + w // 2]
                pos, R, mics = e5.array_pose(rec, t_mid)
                lz = locz
                if task == 5 and np.abs(mics - mics0).max() > 0.005:
                    lz = loc.Localiser(fs, mic_pos=mics)
                gaz, gel = e5.source_dir_local(rec, t_mid, pos, R)
                _, Xw = loc.stft_multi(rec["x"][s:s + w], fs)
                items.append({"task": task, "rec": info, "Rx": loc.spatial_cov(Xw, lz.sel).astype(np.complex64),
                              "sub": lz.sub, "locz": lz, "gaz": gaz, "gel": gel, "centre": (s + w // 2) // config.HOP,
                              "silent": silent_idx, "Xs": Xs})
                kept += 1
            info["windows"] = kept
            recs.append(info)
            print(task, info["recording"], kept, info["frames"], flush=True)

    res = {"windows": len(items), "recordings": [{k: v for k, v in r.items() if k != "phi"} for r in recs]}
    res["music"] = score(items, lambda it: None)
    res["D_as_table"] = score(items, lambda it: it["rec"]["phi"][it["rec"]["used_guard"]])
    res["D0_as_table"] = score(items, lambda it: it["rec"]["phi"][0.0])
    print(res["music"], res["D_as_table"], res["D0_as_table"], flush=True)

    res["loading"] = {}
    for d in LOADINGS:
        config.DIAG_LOADING = d
        res["loading"][str(d)] = score(items, lambda it: it["rec"]["phi"][it["rec"]["used_guard"]])
    config.DIAG_LOADING = 1e-3
    res["guard"] = {str(g): score(items, lambda it, g=g: it["rec"]["phi"].get(g, it["rec"]["phi"][0.0])) for g in GUARDS}

    res["local"] = {}
    for n in LOCAL_FRAMES:
        def local(it, n=n):
            idx = it["silent"]
            near = idx[np.argsort(np.abs(idx - it["centre"]))[:n]]
            Xn = it["Xs"][:, :, near]
            return np.einsum("mft,nft->fmn", Xn, Xn.conj()) / Xn.shape[2]
        res["local"][str(n)] = score(items, local)
    res["unscaled"] = score(items, lambda it: it["rec"]["phi"][it["rec"]["used_guard"]], unscaled_spectrum)
    common.save_json(res, "exp10_locata.json")
    print(json.dumps({k: v for k, v in res.items() if k != "recordings"}, indent=0, default=float))


if __name__ == "__main__":
    main()
