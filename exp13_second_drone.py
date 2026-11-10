"""
Experiment 13. Do the DREGON results carry over to a second drone and to recorded speech?

The AGH set of the SP Cup 2019 (data/SPCup19_egonoise/AGH) was recorded with a drone in an anechoic chamber. It holds
white noise and speech from labelled directions with the motors off, ego-noise at five labelled throttle settings, and
two recordings with white noise and running motors together. The throttle is the only state information.

The set does not publish the geometry of its cube array, so the microphone positions are measured from the time
differences on the motors-off recordings. To keep this independent of the scored recording, each direction is scored
with a geometry fitted only to white-noise recordings from other directions, at least 15 degrees away.

    geometry   fit on all white-noise recordings, residuals, and the check on the speech recordings
    mixtures   real speech or white noise mixed with held-out ego-noise of the same drone, at equal SNR in the band
    real       the two recordings with source and motors together
    rejection  the learned rejection G and the coherence of the ego-noise, on the measured geometry

Usage: python exp13_second_drone.py [part ...]     Output in results_review/: exp13_<part>.json
"""
import json
import os
import sys

os.environ.setdefault("RESULTS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_review"))

import numpy as np
import scipy.signal
import soundfile as sf

import common
import config
import exp6_multidrone_egonoise as e6
import exp12_dregon_controls as e12
import localisation as loc

AGH = os.path.join(os.path.dirname(config.DATA_DIR), "SPCup19_egonoise", "AGH")
# info.xlsx of the set: file, signal, azimuth, elevation
CLEAN = [(0, "white", -1.56222492, -26.34785187), (1, "white", 41.11209044, -20.46998328), (2, "white", 101.19511143, -29.11368844),
         (3, "white", 124.91183022, -24.96386054), (4, "white", -178.70783311, -22.27745199), (5, "white", -132.72591123, -15.53745485),
         (6, "white", -99.24611275, -22.63571875), (7, "white", -53.72338064, -21.22083241), (8, "speech", -135.0, -19.47122063),
         (9, "speech", 124.91183022, -24.96386054)]
THROTTLES = {100: 0, 80: 1, 60: 2, 40: 3, 20: 4}       # percent: file
CORRUPTED = [(0, "very high speed", 124.91183022, -24.96386054), (1, "medium speed", 44.40934279, -21.66648178)]
SNRS = (0, -5, -10, -15, -20, -25, -30, -35)
MIX_THROTTLES = (100, 80, 60)
TRIALS = 16
MIN_SEPARATION = 15.0
WINDOW_SEC = 0.5


def save(obj, part):
    common.save_json(obj, f"exp13_{part}.json")
    print(part, json.dumps(e12.brief(obj), default=float)[:3500], flush=True)


def read(*parts):
    x, fs = sf.read(os.path.join(AGH, *parts), dtype="float32", always_2d=True)
    return x, fs


def active(x, fs):
    """Samples where the loudspeaker plays, by level."""
    e = np.convolve((x ** 2).mean(axis=1), np.ones(fs // 10) / (fs // 10), mode="same")
    return e > 0.25 * np.percentile(e, 95)


def tdoas(x, fs, nfft=8192, interp=32):
    """Pairwise time differences t_i - t_j from GCC-PHAT averaged over the recording."""
    f, _, X = scipy.signal.stft(x.T, fs, nperseg=nfft, noverlap=nfft // 2, boundary=None, padded=False)
    band = (f >= 300) & (f <= 8000)
    M = x.shape[1]
    T = np.zeros((M, M))
    for i in range(M):
        for j in range(i + 1, M):
            R = (X[i] * X[j].conj()).mean(axis=1)
            c = np.fft.fftshift(np.fft.irfft(R / (np.abs(R) + 1e-12) * band, nfft * interp))
            T[i, j] = (np.argmax(c) - nfft * interp // 2) / (fs * interp)
    return T


def fit_geometry(T, dirs):
    """Least squares with robust weights for t_i - t_j = (r_j - r_i).u / c, centred on the array barycentre."""
    M = 8
    A, b = [], []
    for Tk, (az, el) in zip(T, dirs):
        u = loc.unit_vectors(az, el)
        for i in range(M):
            for j in range(i + 1, M):
                row = np.zeros(3 * M)
                row[3 * j:3 * j + 3], row[3 * i:3 * i + 3] = u / config.SPEED_OF_SOUND, -u / config.SPEED_OF_SOUND
                A.append(row)
                b.append(Tk[i, j])
    A, b = np.array(A), np.array(b)
    w = np.ones(len(b))
    for _ in range(20):
        sol = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)[0]
        r = A @ sol - b
        s = np.median(np.abs(r)) + 1e-9
        w = 1 / np.maximum(1, np.abs(r) / (3 * s))
    P = sol.reshape(M, 3)
    return P - P.mean(axis=0), float(np.median(np.abs(r)))


def predicted(P, az, el):
    u = loc.unit_vectors(az, el)
    return np.array([[(P[j] - P[i]) @ u / config.SPEED_OF_SOUND for j in range(8)] for i in range(8)])


def clean_data():
    sigs, T = {}, {}
    for k, kind, az, el in CLEAN:
        x, fs = read("static clean", f"{k}.wav")
        sigs[k] = x[active(x, fs)]
        T[k] = tdoas(sigs[k], fs)
    return sigs, T, fs


def geometry_for(T, az, el):
    """White-noise recordings from directions at least MIN_SEPARATION away from the scored one."""
    use = [c for c in CLEAN if c[1] == "white" and loc.angular_error(c[2], c[3], az, el) >= MIN_SEPARATION]
    return fit_geometry([T[c[0]] for c in use], [(c[2], c[3]) for c in use])[0], len(use)


def part_geometry():
    _, T, _ = clean_data()
    white = [c for c in CLEAN if c[1] == "white"]
    P, res = fit_geometry([T[c[0]] for c in white], [(c[2], c[3]) for c in white])
    d = np.linalg.norm(P[:, None] - P[None], axis=-1)
    iu = np.triu_indices(8, 1)
    out = {"positions_cm": (100 * P).round(2).tolist(), "residual_us": res * 1e6, "aperture_cm": float(100 * d.max()), "held_out": []}
    for k, kind, az, el in CLEAN:
        Pk, n = geometry_for(T, az, el)
        out["held_out"].append({"file": k, "signal": kind, "fit_files": n, "median_tdoa_error_us": float(np.median(np.abs(predicted(Pk, az, el) - T[k])[iu]) * 1e6)})
    save(out, "geometry")


def models(fs, su_sel):
    """Noise covariance from the first half of each throttle file; the second half is kept for mixing."""
    per, sec = {}, {}
    frames = []
    for pct, k in THROTTLES.items():
        x, _ = read("ego-noise", "mic array", f"{k}.wav")
        x = x[int(0.1 * len(x)):int(0.9 * len(x))]
        half = len(x) // 2
        X = loc.stft_multi(x[:half], fs)[1][:, su_sel]
        per[pct] = e12.covariance(X)
        frames.append(X)
        sec[pct] = x[half:]
    pooled = e12.covariance(np.concatenate(frames, axis=2))
    return per, pooled, sec


def band_power(x, fs, band=config.BAND_HZ):
    f, P = scipy.signal.welch(x.T, fs, nperseg=1024, axis=-1)
    return float(P[:, (f >= band[0]) & (f <= band[1])].sum(axis=1).mean())


def localise_all(su, sub_sel, sub, X, Rn_pool, Rn_state):
    """A, C, D, E, F, F with state on one window. Rn_* hold all bins of the band."""
    Rx = e12.covariance(X[:, su.sel][:, sub_sel])
    peak = su.peak
    return {"A": su.srp_of(X), "C": peak(sub.spectrum(Rx)), "D": peak(sub.spectrum(Rx, Rn_pool[sub_sel])),
            "E": peak(sub.spectrum(Rx, Rn_state[sub_sel])), "F": su.filtered(X, Rn_pool), "F_state": su.filtered(X, Rn_state)}


def part_mixtures():
    sigs, T, fs = clean_data()
    az_g, el_g = loc.direction_grid()
    rng = np.random.default_rng(config.RANDOM_SEED)
    w = int(WINDOW_SEC * fs)
    rows = []
    per, pooled, sec = None, None, None
    for k, kind, az, el in CLEAN:
        P, _ = geometry_for(T, az, el)
        su = e12.Setup(fs, config.NFFT, config.HOP, config.BAND_HZ, az_g, el_g, mic_pos=P)
        sub_sel = np.arange(0, len(su.sel), config.MUSIC_BIN_STEP)
        sub = loc.SubspaceLocaliser(P, su.freqs[su.sel][sub_sel], az_g, el_g)
        if per is None:
            per, pooled, sec = models(fs, su.sel)
        x = sigs[k]
        for pct in MIX_THROTTLES:
            for t in range(TRIALS):
                seg = x[rng.integers(len(x) - w):][:w]
                n = sec[pct][rng.integers(len(sec[pct]) - w):][:w]
                gain = band_power(n, fs) / band_power(seg, fs)
                for snr in SNRS:
                    X = su.stft(n + seg * np.sqrt(gain * 10 ** (snr / 10)))
                    res = localise_all(su, sub_sel, sub, X, pooled, per[pct])
                    row = {"file": k, "signal": kind, "throttle": pct, "trial": t, "snr": snr}
                    row.update({m: float(loc.angular_error(*v, az, el)) for m, v in res.items()})
                    rows.append(row)
        print(k, kind, "done", flush=True)
    methods = ["A", "C", "D", "E", "F", "F_state"]
    summary = {}
    for kind in ("white", "speech"):
        summary[kind] = {}
        for snr in SNRS:
            rr = [r for r in rows if r["signal"] == kind and r["snr"] == snr]
            summary[kind][str(snr)] = {m: float(100 * np.mean([r[m] <= 10 for r in rr])) for m in methods}
            summary[kind][str(snr)]["n"] = len(rr)
    save({"summary": summary, "rows": rows}, "mixtures")


def part_real():
    sigs, T, fs = clean_data()
    az_g, el_g = loc.direction_grid()
    out = {}
    per, pooled = None, None
    for k, label, az, el in CORRUPTED:
        P, n_fit = geometry_for(T, az, el)
        su = e12.Setup(fs, config.NFFT, config.HOP, config.BAND_HZ, az_g, el_g, mic_pos=P)
        sub_sel = np.arange(0, len(su.sel), config.MUSIC_BIN_STEP)
        sub = loc.SubspaceLocaliser(P, su.freqs[su.sel][sub_sel], az_g, el_g)
        if per is None:
            per, pooled, _ = models(fs, su.sel)
        x, _ = read("static corrupted", f"{k}.wav")
        # the throttle of these recordings is only described in words, so the closest level of the noise files is taken
        levels = {pct: float(np.mean(read("ego-noise", "mic array", f"{f}.wav")[0] ** 2)) for pct, f in THROTTLES.items()}
        pct = min(levels, key=lambda p: abs(np.log(levels[p] / np.mean(x ** 2))))
        starts, w = common.window_starts(len(x), fs)
        errs = {m: [] for m in ("A", "C", "D", "E", "F", "F_state")}
        for s in starts:
            res = localise_all(su, sub_sel, sub, su.stft(x[s:s + w]), pooled, per[pct])
            for m, v in res.items():
                errs[m].append(float(loc.angular_error(*v, az, el)))
        out[label] = {"windows": len(starts), "matched_throttle": pct, **{m: e12.stats(v) for m, v in errs.items()}}
    save(out, "real")


def part_rejection():
    _, T, fs = clean_data()
    white = [c for c in CLEAN if c[1] == "white"]
    P, _ = fit_geometry([T[c[0]] for c in white], [(c[2], c[3]) for c in white])
    out = {}
    rej, frames = {k: [] for k in (1, 2, 3, 4, 5, 6)}, []
    for pct, f in THROTTLES.items():
        x, _ = read("ego-noise", "mic array", f"{f}.wav")
        x = x[int(0.1 * len(x)):int(0.9 * len(x))]
        half = len(x) // 2
        R1, R2 = e6.covariances(x[:half], fs)[0], e6.covariances(x[half:], fs)[0]
        for k in rej:
            e6.K_SPATIAL = k
            rej[k].append(e6.learned_rejection_db(R1, R2))
        e6.K_SPATIAL = 4
        R, freqs = e6.covariances(x, fs)
        p = np.real(np.einsum("fii->fi", R))
        iu = np.triu_indices(8, 1)
        c = np.abs(R[:, iu[0], iu[1]]) ** 2 / (p[:, iu[0]] * p[:, iu[1]])
        frames.append((c.mean(), c[freqs < 1000].mean()))
    G = e6.diffuse_covariance(P, freqs)
    Gn = G / np.real(np.trace(G, axis1=1, axis2=2))[:, None, None]
    pd = np.real(np.einsum("fii->fi", G))
    cd = np.abs(G[:, iu[0], iu[1]]) ** 2 / (pd[:, iu[0]] * pd[:, iu[1]])
    out["rejection"] = {str(k): float(np.mean(v)) for k, v in rej.items()}
    out["per_throttle_K4"] = {str(p): float(v) for p, v in zip(THROTTLES, rej[4])}
    out["diffuse_rejection"] = {}
    for k in rej:
        e6.K_SPATIAL = k
        out["diffuse_rejection"][str(k)] = e6.learned_rejection_db(Gn, Gn)
    e6.K_SPATIAL = 4
    out["coherence"] = {"mean": float(np.mean([c[0] for c in frames])), "below_1kHz": float(np.mean([c[1] for c in frames])),
                        "diffuse_mean": float(cd.mean()), "diffuse_below_1kHz": float(cd[freqs < 1000].mean())}
    save(out, "rejection")


PARTS = {"geometry": part_geometry, "mixtures": part_mixtures, "real": part_real, "rejection": part_rejection}

if __name__ == "__main__":
    for part in sys.argv[1:] or PARTS:
        PARTS[part]()
