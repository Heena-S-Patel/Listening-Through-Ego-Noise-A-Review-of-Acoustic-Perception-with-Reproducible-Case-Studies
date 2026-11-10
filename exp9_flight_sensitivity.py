"""
Experiment 9. How robust are the DREGON results to the choices made in the paper?

Each part runs on the 391 flight windows of experiment 2, and where stated on the controlled mixtures of experiment 3
(noise model from the first half of the noise flight, mixing noise from the second half).

    blades      enrichment of noise energy at multiples of 1, 2, 3 and 4 times the shaft rate
    loading     diagonal loading of D and E
    bins        number of speed bins of E
    duration    length of the noise-only recording, at several places in the flight
    gp          state-informed baseline after Furukawa et al. 2013, Sect. II: Gaussian process regression of the
                Cholesky factor of the noise covariance from the motor speeds, and optionally the IMU
    combined    tone mask followed by D or E
    source      motors-off recordings with real speech envelopes against the same recordings as white noise,
                at equal SNR in the 300 to 4000 Hz band (semi-synthetic, not a recorded speech flight)

Usage: python exp9_flight_sensitivity.py [part ...]     Output in results_review/: exp9_<part>.json
"""
import glob
import json
import os
import sys

os.environ.setdefault("RESULTS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_review"))

import numpy as np
import scipy.signal
import soundfile as sf

import common
import config
import dregon_io
import localisation as loc
import rotor

AL_OFFSET = {"offset": 93, "az_sign": 1, "el_sign": 1}    # from the motors-off recording; same as the DREGON flight table
LOADINGS = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
BIN_COUNTS = (2, 3, 6, 12, 24)
DURATIONS = (5, 10, 20, 35)
PLACEMENTS = 5
SWEEP_LEVELS = (-10, -15)
GP_GAMMAS = (0.1, 0.3, 1, 3, 10)
GP_NOISES = (0.01, 0.1, 1.0)


def save(obj, part):
    common.save_json(obj, f"exp9_{part}.json")
    print(part, json.dumps(obj, indent=0, default=float)[:3000], flush=True)


def within(errs):
    return float(100 * np.mean(np.array(errs) <= 10))


def imu_at(rec, t, half=0.25):
    """Mean angular velocity and acceleration (6 values) over each window."""
    ts, g, acc = rec["imu"]["timestamps"].ravel(), rec["imu"]["angular_velocity"], rec["imu"]["acceleration"]
    g = g.T if g.shape[0] == 3 else g
    acc = acc.T if acc.shape[0] == 3 else acc
    out = []
    for c in t:
        k = (ts >= c - half) & (ts <= c + half)
        out.append(np.r_[g[k].mean(axis=0), acc[k].mean(axis=0)] if k.any() else np.zeros(6))
    return np.array(out)


class Flights:
    """Covariances and logs of the 391 flight windows, so that each variant only redoes the localisation."""

    def __init__(self, locz):
        self.locz, self.items = locz, []
        for name in config.SOURCE_FLIGHTS:
            rec = dregon_io.load_recording(name)
            starts, w = common.window_starts(len(rec["audio"]), rec["fs"])
            t = rec["audio_ts"][starts + w // 2]
            gaz, gel, ok = dregon_io.source_dir_at(rec, t)
            sp = dregon_io.speeds_at(rec, t)
            ms = sp.mean(axis=1)
            imu = imu_at(rec, t)
            for i in np.where(ok & (ms > 0.3 * np.median(ms)))[0]:
                xw = rec["audio"][starts[i]:starts[i] + w]
                _, X = loc.stft_multi(xw, rec["fs"])
                _, Xh = loc.stft_multi(xw, rec["fs"], 4096, 1024)
                self.items.append({"flight": name, "Rx": loc.spatial_cov(X, locz.sel).astype(np.complex64),
                                   "psd": np.mean(np.abs(X) ** 2, axis=(0, 2)), "psd_h": np.mean(np.abs(Xh) ** 2, axis=(0, 2)),
                                   "sp": sp[i], "ms": ms[i], "imu": imu[i], "gaz": gaz[i], "gel": gel[i]})

    def score(self, phi_of):
        """phi_of(item) returns the noise covariance of that window. Returns overall and per-flight accuracy."""
        errs = []
        for it in self.items:
            a, e = loc.apply_alignment(*self.locz.peak(self.locz.sub.spectrum(it["Rx"], phi_of(it))), AL_OFFSET)
            errs.append(float(loc.angular_error(a, e, it["gaz"], it["gel"])))
        return summary(errs, [it["flight"] for it in self.items])


def summary(errs, flights):
    errs, flights = np.array(errs), np.array(flights)
    res = {"within10": within(errs), "median": float(np.median(errs)), "mean": float(np.mean(errs)),
           "errors": [round(float(v), 3) for v in errs], "flights": [0 if "high" in g else 1 for g in flights]}
    for g in config.SOURCE_FLIGHTS:
        res["loud" if "high" in g else "quiet"] = within(errs[flights == g])
    return res


class Sweep:
    """The mixtures of experiment 3, with the same random draws, at the chosen SNR levels."""

    def __init__(self, locz, noise_rec, levels=SWEEP_LEVELS):
        self.locz = locz
        rng = np.random.default_rng(config.RANDOM_SEED)
        fs, N = noise_rec["fs"], len(noise_rec["audio"])
        half, w = N // 2, int(fs)
        starts = np.arange(half, N - w, w // 2)
        t_mid = noise_rec["audio_ts"][starts + w // 2]
        sp = dregon_io.speeds_at(noise_rec, t_mid)
        imu = imu_at(noise_rec, t_mid, 0.5)
        good = sp.mean(axis=1) > 0.3 * np.median(sp.mean(axis=1))
        starts, sp, imu = starts[good], sp[good], imu[good]
        est, gt, sigs = [], [], []
        for az, el, _, p in dregon_io.list_clean_recordings():
            x, _ = dregon_io.read_wav(p)
            seg = x[len(x) // 2 - w // 2: len(x) // 2 - w // 2 + w]
            if len(seg) == w:
                sigs.append((az, el, seg))
                est.append(locz.reference_A(seg))
                gt.append((az, el))
        est, gt = np.array(est), np.array(gt)
        self.al = loc.fit_frame_alignment(est[:, 0], est[:, 1], gt[:, 0], gt[:, 1])
        self.items = []
        for az, el, seg in sigs:
            p_s = np.mean(seg ** 2)
            for _ in range(config.SEGMENTS_PER_DIRECTION):
                k = rng.integers(len(starts))
                noise = noise_rec["audio"][starts[k]: starts[k] + w]
                for snr in levels:
                    mix = noise + seg * np.sqrt(np.mean(noise ** 2) / p_s * 10 ** (snr / 10))
                    _, X = loc.stft_multi(mix, fs)
                    self.items.append({"snr": snr, "Rx": loc.spatial_cov(X, locz.sel).astype(np.complex64), "sp": sp[k],
                                       "ms": sp[k].mean(), "imu": imu[k], "gaz": az, "gel": el})

    def score(self, phi_of):
        errs = {}
        for it in self.items:
            a, e = loc.apply_alignment(*self.locz.peak(self.locz.sub.spectrum(it["Rx"], phi_of(it))), self.al)
            errs.setdefault(it["snr"], []).append(float(loc.angular_error(a, e, it["gaz"], it["gel"])))
        return {str(s): within(v) for s, v in errs.items()}


def noise_frames(locz, rec, lo=0, hi=None):
    """STFT frames of the band and their mean rotor speed, running motors only."""
    hi = hi or len(rec["audio"])
    X, centres = common.stft_bins(rec["audio"][lo:hi], rec["fs"], locz.sel)
    t = rec["audio_ts"][np.clip(centres + lo, 0, len(rec["audio_ts"]) - 1)]
    ms = dregon_io.speeds_at(rec, t).mean(axis=1)
    run = ms > 0.3 * np.median(ms)
    return X[:, :, run], ms[run]


def model(X, ms, bins=config.NCM_RPM_BINS):
    return loc.NoiseCovModel(X, np.arange(X.shape[1]), ms, bins)


# ---------------------------------------------------------------------------
def part_blades(noise_rec):
    freqs, spectra, speeds = rotor._segment_spectra(noise_rec, n_segments=40)
    band = (freqs > 50) & (freqs < 4000)
    res = {}
    for mult in (1, 2, 3, 4):
        vals = []
        for P, f0 in zip(spectra, speeds * config.SPEED_SCALE):
            m = rotor.comb_bins(freqs, f0, mult, 4000.0, 1) & band
            vals.append((P[m].sum() / P[band].sum()) / (m.sum() / band.sum()))
        res[str(mult)] = {"enrichment": float(np.mean(vals))}
    # odd and even harmonics separately: a two-blade rotor concentrates energy on the even ones
    for label, k in (("odd", np.arange(1, 41, 2)), ("even", np.arange(2, 41, 2)), ("third", np.arange(3, 41, 3))):
        vals = []
        for P, f0 in zip(spectra, speeds * config.SPEED_SCALE):
            m = np.zeros(len(freqs), dtype=bool)
            for f in f0:
                j = np.round(k * f / (freqs[1] - freqs[0])).astype(int)
                m[j[j < len(freqs)]] = True
            m &= band
            vals.append((P[m].sum() / P[band].sum()) / (m.sum() / band.sum()))
        res[label] = float(np.mean(vals))
    # height of each shaft-rate order above its neighbourhood (dB), which large low-frequency tones cannot dominate
    df = freqs[1] - freqs[0]
    span = int(30 / df)
    db = 10 * np.log10(spectra + 1e-20)
    res["order_height_db"] = {}
    for k in range(1, 25):
        vals = []
        for P, f0 in zip(db, speeds * config.SPEED_SCALE):
            for f in f0:
                j = int(round(k * f / df))
                if j + span < len(freqs) and k * f < 4000:
                    vals.append(P[max(0, j - 1):j + 2].max() - np.median(P[max(0, j - span):j + span + 1]))
        if vals:
            res["order_height_db"][str(k)] = float(np.mean(vals))
    save(res, "blades")


def part_blade_mask(locz, fs):
    """Method B of the DREGON flight table (4096-point STFT) with combs at 1, 2, 3 and 4 times the shaft rate."""
    f4 = np.fft.rfftfreq(4096, 1 / fs)
    band4 = (f4 >= config.BAND_HZ[0]) & (f4 <= config.BAND_HZ[1])
    srp = loc.SRPPHAT(config.MIC_POS, fs, 4096, locz.az, locz.el, interp=2)
    errs = {m: [] for m in (1, 2, 3, 4)}
    flights, share = [], {m: [] for m in errs}
    for name in config.SOURCE_FLIGHTS:
        rec = dregon_io.load_recording(name)
        starts, w = common.window_starts(len(rec["audio"]), fs)
        t = rec["audio_ts"][starts + w // 2]
        gaz, gel, ok = dregon_io.source_dir_at(rec, t)
        sp = dregon_io.speeds_at(rec, t)
        ms = sp.mean(axis=1)
        for i in np.where(ok & (ms > 0.3 * np.median(ms)))[0]:
            _, Xh = loc.stft_multi(rec["audio"][starts[i]:starts[i] + w], fs, 4096, 1024)
            psd = np.mean(np.abs(Xh) ** 2, axis=(0, 2))
            flights.append(name)
            for m in errs:
                config.MASK_MULTIPLE = m
                mask = rotor.tonal_mask(f4, sp[i] * config.SPEED_SCALE, psd, band4)
                share[m].append(mask.sum() / band4.sum())
                a, e = loc.apply_alignment(*locz.peak(srp.spectrum(Xh, band4 * (~mask))), AL_OFFSET)
                errs[m].append(float(loc.angular_error(a, e, gaz[i], gel[i])))
    config.MASK_MULTIPLE = config.N_BLADES
    save({str(m): dict(summary(v, flights), bins_masked=float(np.mean(share[m]))) for m, v in errs.items()}, "blade_mask")


def part_loading(fl, ncm):
    res = {}
    for d in LOADINGS:
        config.DIAG_LOADING = d
        res[str(d)] = {"D": fl.score(lambda it: ncm.global_ncm), "E": fl.score(lambda it: ncm.for_speed(it["ms"]))}
    config.DIAG_LOADING = 1e-3
    save(res, "loading")


def part_bins(fl, sw, X, ms, Xh, msh):
    res = {}
    for n in BIN_COUNTS:
        m, mh = model(X, ms, n), model(Xh, msh, n)
        res[str(n)] = {"bins_used": len(m.bins), "flight": fl.score(lambda it: m.for_speed(it["ms"])),
                       "sweep": sw.score(lambda it: mh.for_speed(it["ms"]))}
    save(res, "bins")


def part_duration(fl, sw, X, ms, Xh, msh, fs):
    """Contiguous spans at evenly spaced places. Flights use the whole noise flight, mixtures only its first half."""
    fr = fs / config.HOP
    res = {}
    for dur in DURATIONS:
        n = int(dur * fr)
        runs = []
        for s_f, s_h in zip(np.linspace(0, X.shape[2] - n, PLACEMENTS).astype(int),
                            np.linspace(0, max(0, Xh.shape[2] - n), PLACEMENTS).astype(int)):
            m = model(X[:, :, s_f:s_f + n], ms[s_f:s_f + n])
            mh = model(Xh[:, :, s_h:s_h + n], msh[s_h:s_h + n])
            runs.append({"D": fl.score(lambda it: m.global_ncm), "E": fl.score(lambda it: m.for_speed(it["ms"])),
                         "D_sweep": sw.score(lambda it: mh.global_ncm), "E_sweep": sw.score(lambda it: mh.for_speed(it["ms"]))})
        out = {}
        for k in ("D", "E"):
            v = [r[k]["within10"] for r in runs]
            out[k] = {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))}
            for lev in map(str, SWEEP_LEVELS):
                v = [r[f"{k}_sweep"][lev] for r in runs]
                out[f"{k}_sweep_{lev}"] = {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))}
        res[str(dur)] = out
    m, mh = model(X, ms), model(Xh, msh)
    res["all"] = {"seconds_flight": X.shape[2] / fr, "seconds_first_half": Xh.shape[2] / fr,
                  "D": fl.score(lambda it: m.global_ncm)["within10"], "E": fl.score(lambda it: m.for_speed(it["ms"]))["within10"],
                  "D_sweep": sw.score(lambda it: mh.global_ncm), "E_sweep": sw.score(lambda it: mh.for_speed(it["ms"]))}
    save(res, "duration")


# ---------------------------------------------------------------------------
class GaussianProcessNCM:
    """Furukawa et al. 2013, Eqs. 1-10. The dependent variable is the upper Cholesky factor of the noise covariance with
    square roots on its diagonal, relative to the identity, so that the prior mean of the covariance is the identity.
    The kernel is Eq. 10 with each feature scaled by its variance. gamma and the noise variance are not given in the paper,
    so they are selected on a held-out last fifth of the noise recording, as far as this data allows."""

    def __init__(self, X, feats, gamma, rho2):
        self.scale = np.real(np.trace(np.einsum("mft,nft->fmn", X, X.conj()) / X.shape[2], axis1=1, axis2=2)) / X.shape[1]
        self.feats, self.mu, self.sd = feats, feats.mean(axis=0), feats.std(axis=0) + 1e-9
        self.gamma, self.Y = gamma, self.to_factors(X)
        K = self.kernel(feats)
        self.alpha = np.linalg.solve(K + rho2 * np.eye(len(K)), np.eye(len(K)))

    def to_factors(self, X):
        """X (M, F, T) in windows of 42 frames (0.5 s) with 21 frames hop -> (windows, F, M, M)."""
        M, F, T = X.shape
        out = []
        for s in window_slices(T):
            Xs = X[:, :, s]
            Q = np.einsum("mft,nft->fmn", Xs, Xs.conj()) / Xs.shape[2] / self.scale[:, None, None]
            Q = Q + 1e-3 * np.eye(M)
            L = np.conj(np.transpose(np.linalg.cholesky(Q), (0, 2, 1)))     # Q = L^H L with L upper triangular
            d = np.sqrt(np.real(np.einsum("fii->fi", L)))
            for i in range(M):
                L[:, i, i] = d[:, i] - 1
            out.append(L.astype(np.complex64))
        return np.array(out)

    def kernel(self, u):
        a, b = (u - self.mu) / self.sd, (self.feats - self.mu) / self.sd
        d = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1) / a.shape[1]
        return np.exp(-self.gamma * d)

    def predict(self, u):
        w = self.kernel(u[None])[0] @ self.alpha
        L = np.tensordot(w, self.Y, axes=1)
        M = L.shape[-1]
        for i in range(M):
            L[:, i, i] = (1 + np.real(L[:, i, i])) ** 2
        return (np.conj(np.transpose(L, (0, 2, 1))) @ L) * self.scale[:, None, None]


def window_slices(T, n=42, hop=21):
    return [np.arange(s, s + n) for s in range(0, T - n + 1, hop)]


def window_features(rec, locz, use_imu, lo=0, hi=None):
    hi = hi or len(rec["audio"])
    X, centres = common.stft_bins(rec["audio"][lo:hi], rec["fs"], locz.sel)
    t = rec["audio_ts"][np.clip(centres + lo, 0, len(rec["audio_ts"]) - 1)]
    ms = dregon_io.speeds_at(rec, t).mean(axis=1)
    X, t = X[:, :, ms > 0.3 * np.median(ms)], t[ms > 0.3 * np.median(ms)]
    tw = np.array([t[s].mean() for s in window_slices(X.shape[2])])
    u = dregon_io.speeds_at(rec, tw)
    if use_imu:
        u = np.hstack([u, imu_at(rec, tw)])
    return X, u


def select_gp(X, u):
    """Held-out last fifth: the setting whose predicted covariance is closest to the measured one."""
    T = X.shape[2]
    cut = int(0.8 * T)
    nwin = len(window_slices(cut))
    best, table = None, {}
    probe = GaussianProcessNCM(X[:, :, :cut], u[:nwin], 1.0, 1.0)
    test = []
    for s in window_slices(T - cut):
        Xs = X[:, :, cut:][:, :, s]
        test.append(np.einsum("mft,nft->fmn", Xs, Xs.conj()) / Xs.shape[2])
    ut = u[-len(test):]
    for g in GP_GAMMAS:
        for r in GP_NOISES:
            gp = GaussianProcessNCM.__new__(GaussianProcessNCM)
            gp.scale, gp.feats, gp.mu, gp.sd, gp.Y = probe.scale, probe.feats, probe.mu, probe.sd, probe.Y
            gp.gamma = g
            K = gp.kernel(gp.feats)
            gp.alpha = np.linalg.solve(K + r * np.eye(len(K)), np.eye(len(K)))
            e = np.mean([np.sum(np.abs(gp.predict(a) - q) ** 2) / np.sum(np.abs(q) ** 2) for a, q in zip(ut, test)])
            table[f"{g}_{r}"] = float(e)
            if best is None or e < best[0]:
                best = (float(e), g, r)
    # the baseline to beat on the same held-out data: one covariance and no regression
    Xt = X[:, :, :cut]
    Q0 = np.einsum("mft,nft->fmn", Xt, Xt.conj()) / Xt.shape[2]
    baseline = float(np.mean([np.sum(np.abs(Q0 - q) ** 2) / np.sum(np.abs(q) ** 2) for q in test]))
    return best, table, baseline


def part_gp(fl, sw, noise_rec, locz):
    res = {}
    half = len(noise_rec["audio"]) // 2
    for label, use_imu in (("rotor", False), ("rotor_imu", True)):
        X, u = window_features(noise_rec, locz, use_imu)
        best, table, base = select_gp(X, u)
        gp = GaussianProcessNCM(X, u, best[1], best[2])
        res[label] = {"selected_gamma": best[1], "selected_noise": best[2], "heldout_error": best[0],
                      "heldout_error_one_covariance": base, "heldout_table": table,
                      "flight": fl.score(lambda it: gp.predict(np.r_[it["sp"], it["imu"]] if use_imu else it["sp"]))}
        Xh, uh = window_features(noise_rec, locz, use_imu, 0, half)
        bh = select_gp(Xh, uh)[0]
        gph = GaussianProcessNCM(Xh, uh, bh[1], bh[2])
        res[label]["sweep"] = sw.score(lambda it: gph.predict(np.r_[it["sp"], it["imu"]] if use_imu else it["sp"]))
        res[label]["sweep_selected"] = [bh[1], bh[2]]
    save(res, "gp")


# ---------------------------------------------------------------------------
def masked_spectrum(sub, Rx, Phi, keep):
    part = loc.SubspaceLocaliser.__new__(loc.SubspaceLocaliser)
    part.A = sub.A[keep]
    return part.spectrum(Rx[keep], Phi[keep])


def part_combined(fl, ncm, locz, fs):
    f1, f4 = np.fft.rfftfreq(1024, 1 / fs), np.fft.rfftfreq(4096, 1 / fs)
    band4 = (f4 >= config.BAND_HZ[0]) & (f4 <= config.BAND_HZ[1])
    res = {}

    def run(kind, which):
        errs, share = [], []
        for it in fl.items:
            f0 = it["sp"] * config.SPEED_SCALE
            if kind == "mask1024":
                masked = rotor.tonal_mask(f1, f0, it["psd"], locz.band)
                drop = masked[locz.sel]
            else:                                   # the 4096-point mask of the DREGON flight table, mapped to the nearest 1024-point bin
                m4 = rotor.tonal_mask(f4, f0, it["psd_h"], band4)
                near = np.zeros(len(f1), dtype=bool)
                near[np.round(f4[m4] / (f1[1] - f1[0])).astype(int)] = True
                drop = near[locz.sel]
            keep = np.where(~drop)[0]
            share.append(1 - len(keep) / len(drop))
            phi = ncm.global_ncm if which == "D" else ncm.for_speed(it["ms"])
            a, e = loc.apply_alignment(*locz.peak(masked_spectrum(locz.sub, it["Rx"], phi, keep)), AL_OFFSET)
            errs.append(float(loc.angular_error(a, e, it["gaz"], it["gel"])))
        return dict(summary(errs, [it["flight"] for it in fl.items]), bins_removed_share=float(np.mean(share)))

    for kind in ("mask1024", "mask4096"):
        res[kind] = {w: run(kind, w) for w in ("D", "E")}
    res["without_mask"] = {"D": fl.score(lambda it: ncm.global_ncm), "E": fl.score(lambda it: ncm.for_speed(it["ms"]))}
    save(res, "combined")


# ---------------------------------------------------------------------------
def speech_segments(n, fs, length):
    """Real talker excerpts: one microphone of the LOCATA moving-talker recordings, where the labels say speech is active."""
    out = []
    root = os.path.join(os.path.dirname(config.DATA_DIR), "LOCATA", "dev", "task3")
    for d in sorted(glob.glob(os.path.join(root, "recording*", "benchmark2"))):
        x, f2 = sf.read(os.path.join(d, "audio_array_benchmark2.wav"), always_2d=True)
        vad = np.loadtxt(glob.glob(os.path.join(d, "VAD_benchmark2_*.txt"))[0], skiprows=1)
        y = scipy.signal.resample_poly(x[:, 0], fs, f2)
        act = scipy.signal.resample_poly(vad[: len(x)].astype(float), fs, f2) > 0.5
        s = 0
        while s + length < len(y) and len(out) < 500:
            if act[s:s + length].mean() >= 0.8:
                out.append(y[s:s + length].astype(np.float32))
                s += length
            else:
                s += length // 4
    rng = np.random.default_rng(config.RANDOM_SEED)
    return [out[i] for i in rng.permutation(len(out))[:n]], len(out)


def band_power(x, fs):
    f, P = scipy.signal.welch(x.T, fs=fs, nperseg=1024)
    return float(P[:, (f >= config.BAND_HZ[0]) & (f <= config.BAND_HZ[1])].sum(axis=1).mean())


def impose_envelope(seg, talker, fs):
    """Keeps the phases, and so the direction, of the multichannel recording and takes the magnitude envelope of the talker."""
    _, _, X = scipy.signal.stft(seg.T, fs=fs, nperseg=1024)
    _, _, S = scipy.signal.stft(talker, fs=fs, nperseg=1024)
    T = min(X.shape[2], S.shape[1])
    _, y = scipy.signal.istft(X[:, :, :T] * np.abs(S[:, :T])[None], fs=fs, nperseg=1024)
    y = y.T[: len(seg)]
    return np.pad(y, ((0, len(seg) - len(y)), (0, 0))).astype(np.float32)


def part_source(locz, noise_rec, Xh, msh):
    fs = noise_rec["fs"]
    N, half, w = len(noise_rec["audio"]), len(noise_rec["audio"]) // 2, int(fs)
    ncm = model(Xh, msh)
    rng = np.random.default_rng(config.RANDOM_SEED)
    starts = np.arange(half, N - w, w // 2)
    sp = dregon_io.speeds_at(noise_rec, noise_rec["audio_ts"][starts + w // 2])
    good = sp.mean(axis=1) > 0.3 * np.median(sp.mean(axis=1))
    starts, sp = starts[good], sp[good]
    clean, est, gt = [], [], []
    for az, el, _, p in dregon_io.list_clean_recordings():
        x, _ = dregon_io.read_wav(p)
        seg = x[len(x) // 2 - w // 2: len(x) // 2 - w // 2 + w]
        clean.append((az, el, seg))
        est.append(locz.reference_A(seg))
        gt.append((az, el))
    est, gt = np.array(est), np.array(gt)
    al = loc.fit_frame_alignment(est[:, 0], est[:, 1], gt[:, 0], gt[:, 1])
    n_trials = len(clean) * config.SEGMENTS_PER_DIRECTION
    talkers, available = speech_segments(n_trials, fs, w)
    levels = (5, 0, -5, -10, -15, -20)
    errs = {src: {s: {m: [] for m in "ACDE"} for s in levels} for src in ("white", "speech")}
    fullband = {"white": [], "speech": []}
    trial = 0
    for az, el, seg in clean:
        for _ in range(config.SEGMENTS_PER_DIRECTION):
            k = rng.integers(len(starts))
            noise = noise_rec["audio"][starts[k]: starts[k] + w]
            pn = band_power(noise, fs)
            for src, sig in (("white", seg), ("speech", impose_envelope(seg, talkers[trial % len(talkers)], fs))):
                ps = band_power(sig, fs)
                g0 = np.sqrt(pn / ps)
                fullband[src].append(10 * np.log10(np.mean((sig * g0) ** 2) / np.mean(noise ** 2)))
                for snr in levels:
                    mix = noise + sig * g0 * 10 ** (snr / 20)
                    res = locz.run(mix, ncm_model=ncm, mean_speed=sp[k].mean(), methods=["A_SRP-PHAT", "C_MUSIC", "D_GEVD-MUSIC_globalNCM", "E_GEVD-MUSIC_rpmNCM"])
                    for m, (a, e) in res.items():
                        a2, e2 = loc.apply_alignment(a, e, al)
                        errs[src][snr][m[0]].append(float(loc.angular_error(a2, e2, az, el)))
            trial += 1
    out = {"trials_per_level": n_trials, "talker_excerpts_available": available, "alignment": al,
           "fullband_snr_when_band_snr_is_0dB": {k: float(np.median(v)) for k, v in fullband.items()}}
    for src in errs:
        out[src] = {str(s): {m: within(v) for m, v in errs[src][s].items()} for s in levels}
    save(out, "source")


def main(parts):
    noise_rec = dregon_io.load_recording(config.NOISE_ONLY_FLIGHT)
    fs = noise_rec["fs"]
    locz = loc.Localiser(fs)
    if "blades" in parts:
        part_blades(noise_rec)
    if "blade_mask" in parts:
        part_blade_mask(locz, fs)
    X, ms = noise_frames(locz, noise_rec)
    half = len(noise_rec["audio"]) // 2
    Xh, msh = noise_frames(locz, noise_rec, 0, half)
    if "source" in parts:
        part_source(locz, noise_rec, Xh, msh)
    if set(parts) & {"loading", "bins", "duration", "gp", "combined"}:
        fl = Flights(locz)
        ncm = model(X, ms)
        base = {"D": fl.score(lambda it: ncm.global_ncm), "E": fl.score(lambda it: ncm.for_speed(it["ms"]))}
        print("baseline", base, flush=True)
        assert abs(base["D"]["within10"] - 60.4) < 0.06 and abs(base["E"]["within10"] - 69.8) < 0.06
        sw = Sweep(locz, noise_rec) if set(parts) & {"bins", "duration", "gp"} else None
        if "loading" in parts:
            part_loading(fl, ncm)
        if "combined" in parts:
            part_combined(fl, ncm, locz, fs)
        if "bins" in parts:
            part_bins(fl, sw, X, ms, Xh, msh)
        if "duration" in parts:
            part_duration(fl, sw, X, ms, Xh, msh, fs)
        if "gp" in parts:
            part_gp(fl, sw, noise_rec, locz)


if __name__ == "__main__":
    main(sys.argv[1:] or ["blades", "blade_mask", "loading", "bins", "duration", "gp", "combined", "source"])
