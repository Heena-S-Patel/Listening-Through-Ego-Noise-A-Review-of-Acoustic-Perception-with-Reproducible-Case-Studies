"""
Experiment 12. Controls on the DREGON recordings that the earlier experiments did not include.

Method F is the best baseline of the DREGON paper (Strauss et al. 2018, Sects. IV-V): a multichannel Wiener filter
before SRP-PHAT (the paper uses a toolbox variation of SRP-PHAT with GCC-PHAT spectra).

    wf       F on the 391 windows of the DREGON flight table, with the settings of that paper (16 kHz, 64 ms frames, 50% overlap, 5 degree
             grid, noise covariance from 7 s of free flight, at five places since the paper does not say which 7 s),
             and with our settings. GCC-PHAT and GEVD-MUSIC with the settings of that paper are included for comparison.
             "printed" follows the matrix order of Eq. 2 of that paper, W = Rxx^-1 (Rxx - Rnn), for the record.
    sweep    F on the controlled mixtures of the SNR-sweep table, with the same random draws
    silent   the recording in which the motors are off and a person carries the drone, which also holds about 12 s of
             room silence. Tests the methods without ego-noise, and whether a noise model hurts on this array at high SNR
    mixed    the same recording mixed with held-out flight noise at chosen SNRs, as a third moving-array condition

Usage: python exp12_dregon_controls.py [part ...]     Output in results_review/: exp12_<part>.json
"""
import json
import os
import sys

os.environ.setdefault("RESULTS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_review"))

import numpy as np
import scipy.signal

import common
import config
import dregon_io
import localisation as loc

AL = {"offset": 93, "az_sign": 1, "el_sign": 1}    # from the motors-off recording; same as the DREGON flight table
FS_WF = 16000
NFFT_WF, HOP_WF = 1024, 512       # 64 ms and 50% at 16 kHz
SPAN_WF = 7.0
PLACES = 5
MIX_LEVELS = (0, -5, -10, -15)
WIDE_BAND = (300.0, 8000.0)


def save(obj, part):
    common.save_json(obj, f"exp12_{part}.json")
    print(part, json.dumps(brief(obj), default=float)[:3500], flush=True)


def brief(o):
    if isinstance(o, dict):
        return {k: brief(v) for k, v in o.items() if k not in ("errors", "flights", "times")}
    return o


def stats(errs, az_err=None, el_err=None):
    errs = np.asarray(errs)
    out = {"within10": float(100 * np.mean(errs <= 10)), "within20": float(100 * np.mean(errs <= 20)),
           "median": float(np.median(errs)), "mean": float(np.mean(errs)), "errors": [round(float(v), 3) for v in errs]}
    if az_err is not None:      # the success rule of the DREGON paper
        out["both_within10"] = float(100 * np.mean((np.asarray(az_err) <= 10) & (np.asarray(el_err) <= 10)))
    return out


def resample(x, fs, fs_new):
    return scipy.signal.resample_poly(x, fs_new, fs, axis=0).astype(np.float32)


def stft16(x):
    return loc.stft_multi(x, FS_WF, NFFT_WF, HOP_WF)[1]


def covariance(X):
    return np.einsum("mft,nft->fmn", X, X.conj()) / X.shape[2]


def printed_order(X, Rnn):
    Rxx = covariance(X)
    M = Rxx.shape[1]
    load = 1e-3 * np.real(np.trace(Rxx, axis1=1, axis2=2))[:, None, None] / M
    return np.einsum("fmn,nft->mft", np.linalg.solve(Rxx + load * np.eye(M), Rxx - Rnn), X)


class Setup:
    """SRP-PHAT over the given band, with the Wiener filter computed on the same bins."""

    def __init__(self, fs, nfft, hop, band, az, el, mic_pos=None):
        self.fs, self.nfft, self.hop, self.az, self.el = fs, nfft, hop, az, el
        mic_pos = config.MIC_POS if mic_pos is None else mic_pos
        self.freqs = np.fft.rfftfreq(nfft, 1 / fs)
        self.sel = np.where((self.freqs >= band[0]) & (self.freqs <= band[1]))[0]
        self.weight = np.isin(np.arange(len(self.freqs)), self.sel).astype(float)
        self.srp = loc.SRPPHAT(mic_pos, fs, nfft, az, el)
        self.sub = loc.SubspaceLocaliser(mic_pos, self.freqs[self.sel], az, el)

    def peak(self, s):
        k = int(np.argmax(s))
        return self.az[k], self.el[k]

    def stft(self, x):
        return loc.stft_multi(x, self.fs, self.nfft, self.hop)[1]

    def srp_of(self, X):
        return self.peak(self.srp.spectrum(X, self.weight))

    def filtered(self, X, Rn, clip=True):
        Y = np.zeros_like(X)
        Y[:, self.sel] = loc.wiener_filter(X[:, self.sel], Rn, clip)
        return self.peak(self.srp.spectrum(Y, self.weight))

    def music(self, X, Rn):
        return self.peak(self.sub.spectrum(covariance(X[:, self.sel]), Rn))


def windows(rec):
    starts, w = common.window_starts(len(rec["audio"]), rec["fs"])
    t = rec["audio_ts"][starts + w // 2]
    gaz, gel, ok = dregon_io.source_dir_at(rec, t)
    ms = dregon_io.speeds_at(rec, t).mean(axis=1)
    return starts, w, t, gaz, gel, ms, np.where(ok & (ms > 0.3 * np.median(ms)))[0]


def score(items, fn, flights):
    errs, azs, els = [], [], []
    for it in items:
        a, e = loc.apply_alignment(*fn(it), AL)
        errs.append(float(loc.angular_error(a, e, it["gaz"], it["gel"])))
        azs.append(abs((a - it["gaz"] + 180) % 360 - 180))
        els.append(abs(e - it["gel"]))
    errs = np.array(errs)
    res = stats(errs, azs, els)
    res["loud"], res["quiet"] = [float(100 * np.mean(errs[flights == g] <= 10)) for g in (0, 1)]
    print("   ", {k: round(v, 1) for k, v in res.items() if k != "errors"}, flush=True)
    return res


def part_wf():
    noise = dregon_io.load_recording(config.NOISE_ONLY_FLIGHT)
    fs = noise["fs"]
    # settings of the DREGON paper
    su16 = Setup(FS_WF, NFFT_WF, HOP_WF, (1.0, FS_WF / 2), *loc.direction_grid(5, 5, (-90, 90)))
    x16 = resample(noise["audio"], fs, FS_WF)
    Xn = stft16(x16)[:, su16.sel]
    centres = (NFFT_WF // 2 + HOP_WF * np.arange(Xn.shape[2])) * fs // FS_WF
    ms = dregon_io.speeds_at(noise, noise["audio_ts"][np.clip(centres, 0, len(noise["audio_ts"]) - 1)]).mean(axis=1)
    idx = np.where(ms > 0.3 * np.median(ms))[0]
    n_span = int(SPAN_WF * FS_WF / HOP_WF)
    places = np.linspace(0, len(idx) - n_span, PLACES).astype(int)
    R16 = {f"7s_{k}": covariance(Xn[:, :, idx[p:p + n_span]]) for k, p in enumerate(places)}
    R16["all"] = covariance(Xn[:, :, idx])
    # our settings
    su = Setup(fs, config.NFFT, config.HOP, config.BAND_HZ, *loc.direction_grid())
    su8 = Setup(fs, config.NFFT, config.HOP, WIDE_BAND, *loc.direction_grid())
    nm, nm8 = common.build_ncm_model(noise, su.sel), common.build_ncm_model(noise, su8.sel)
    del noise, x16, Xn

    items = []
    for name in config.SOURCE_FLIGHTS:
        rec = dregon_io.load_recording(name)
        starts, w, t, gaz, gel, ms, keep = windows(rec)
        for i in keep:
            xw = rec["audio"][starts[i]:starts[i] + w]
            items.append({"flight": name, "X16": stft16(resample(xw, fs, FS_WF)).astype(np.complex64),
                          "X": su.stft(xw).astype(np.complex64), "ms": ms[i], "gaz": gaz[i], "gel": gel[i], "t": float(t[i] - rec["audio_ts"][0])})
        del rec
    flights = np.array([0 if "high" in it["flight"] else 1 for it in items])
    print(len(items), "windows", flush=True)

    out = {"windows": len(items), "flights": flights.tolist(), "times": [it["t"] for it in items]}
    run = lambda key, fn: (print(key, flush=True), out.__setitem__(key, score(items, fn, flights)))
    run("GCC-PHAT_16k", lambda it: su16.srp_of(it["X16"]))
    for k, R in R16.items():
        run(f"F_16k_{k}", lambda it: su16.filtered(it["X16"], R))
        run(f"GEVD_16k_{k}", lambda it: su16.music(it["X16"], R))
    run("F_16k_all_noclip", lambda it: su16.filtered(it["X16"], R16["all"], clip=False))

    def printed(it):
        Y = np.zeros_like(it["X16"])
        Y[:, su16.sel] = printed_order(it["X16"][:, su16.sel], R16["all"])
        return su16.peak(su16.srp.spectrum(Y, su16.weight))
    run("F_16k_all_printed_order", printed)
    run("F_global", lambda it: su.filtered(it["X"], nm.global_ncm))
    run("F_speed", lambda it: su.filtered(it["X"], nm.for_speed(it["ms"])))
    run("F_wide_global", lambda it: su8.filtered(it["X"], nm8.global_ncm))
    run("F_wide_speed", lambda it: su8.filtered(it["X"], nm8.for_speed(it["ms"])))
    run("A_wide", lambda it: su8.srp_of(it["X"]))
    save(out, "wf")


def part_sweep():
    noise = dregon_io.load_recording(config.NOISE_ONLY_FLIGHT)
    fs, N = noise["fs"], len(noise["audio"])
    su = Setup(fs, config.NFFT, config.HOP, config.BAND_HZ, *loc.direction_grid())
    locz = loc.Localiser(fs)
    half, w = N // 2, int(fs)
    nm = common.build_ncm_model(noise, su.sel, sample_range=(0, half))
    rng = np.random.default_rng(config.RANDOM_SEED)    # the draws of experiment 3
    starts = np.arange(half, N - w, w // 2)
    sp = dregon_io.speeds_at(noise, noise["audio_ts"][starts + w // 2])
    good = sp.mean(axis=1) > 0.3 * np.median(sp.mean(axis=1))
    starts, sp = starts[good], sp[good]
    sigs, est, gt = [], [], []
    for az, el, _, p in dregon_io.list_clean_recordings():
        x, _ = dregon_io.read_wav(p)
        seg = x[len(x) // 2 - w // 2: len(x) // 2 - w // 2 + w]
        if len(seg) == w:
            sigs.append((az, el, seg))
            est.append(locz.reference_A(seg))
            gt.append((az, el))
    est, gt = np.array(est), np.array(gt)
    al = loc.fit_frame_alignment(est[:, 0], est[:, 1], gt[:, 0], gt[:, 1])
    errs = {str(s): {"A": [], "F": [], "F_speed": []} for s in config.SNR_LIST_DB}
    dirs = []
    for d, (az, el, seg) in enumerate(sigs):
        p_s = np.mean(seg ** 2)
        for _ in range(config.SEGMENTS_PER_DIRECTION):
            k = rng.integers(len(starts))
            n = noise["audio"][starts[k]:starts[k] + w]
            dirs.append(d)
            for snr in config.SNR_LIST_DB:
                X = su.stft(n + seg * np.sqrt(np.mean(n ** 2) / p_s * 10 ** (snr / 10)))
                for key, val in (("A", su.srp_of(X)), ("F", su.filtered(X, nm.global_ncm)), ("F_speed", su.filtered(X, nm.for_speed(sp[k].mean())))):
                    a, e = loc.apply_alignment(*val, al)
                    errs[str(snr)][key].append(float(loc.angular_error(a, e, az, el)))
    out = {"alignment": al, "directions": dirs, "levels": {s: {k: stats(v) for k, v in row.items()} for s, row in errs.items()}}
    save(out, "sweep")


def source_on(rec, w_s=0.25):
    """Frames of the motors-off recording split into room silence and source, by level."""
    x, fs = rec["audio"], rec["fs"]
    w = int(w_s * fs)
    p = np.array([np.mean(x[s:s + w] ** 2) for s in range(0, len(x) - w, w)])
    thr = 10 ** (0.5 * (np.log10(np.percentile(p, 5)) + np.log10(np.percentile(p, 95))))
    return p < thr, w


def active_windows(rec, quiet, wp):
    starts, w = common.window_starts(len(rec["audio"]), rec["fs"])
    t = rec["audio_ts"][starts + w // 2]
    gaz, gel, ok = dregon_io.source_dir_at(rec, t)
    on = np.array([not quiet[min(s // wp, len(quiet) - 1):(s + w) // wp].any() for s in starts])
    return starts, w, gaz, gel, np.where(ok & on)[0]


def part_silent():
    rec = dregon_io.load_recording(config.CALIBRATION_FLIGHT)
    fs = rec["fs"]
    su = Setup(fs, config.NFFT, config.HOP, config.BAND_HZ, *loc.direction_grid())
    locz = loc.Localiser(fs)
    quiet, wp = source_on(rec)
    Xq = np.concatenate([su.stft(rec["audio"][k * wp:(k + 1) * wp]) for k in np.where(quiet)[0]], axis=2)
    Rroom_all = covariance(Xq[:, su.sel])
    Rroom = Rroom_all[:: config.MUSIC_BIN_STEP]
    noise = dregon_io.load_recording(config.NOISE_ONLY_FLIGHT)
    nm = common.build_ncm_model(noise, su.sel)
    del noise
    starts, w, gaz, gel, keep = active_windows(rec, quiet, wp)
    errs = {k: [] for k in ("A", "C", "D_room", "D_flight", "F_room", "F_flight")}
    snrs = []
    for i in keep:
        X = su.stft(rec["audio"][starts[i]:starts[i] + w])
        Rx = loc.spatial_cov(X, locz.sel)
        est = {"A": su.srp_of(X), "C": locz.peak(locz.sub.spectrum(Rx)), "D_room": locz.peak(locz.sub.spectrum(Rx, Rroom)),
               "D_flight": locz.peak(locz.sub.spectrum(Rx, nm.global_ncm[:: config.MUSIC_BIN_STEP])),
               "F_room": su.filtered(X, Rroom_all), "F_flight": su.filtered(X, nm.global_ncm)}
        for k, v in est.items():
            a, e = loc.apply_alignment(*v, AL)
            errs[k].append(float(loc.angular_error(a, e, gaz[i], gel[i])))
        snrs.append(10 * np.log10(np.real(np.trace(Rx, axis1=1, axis2=2)).mean() / np.real(np.trace(Rroom, axis1=1, axis2=2)).mean()))
    out = {"windows": len(keep), "room_silence_s": float(quiet.sum() * 0.25), "snr_db_median": float(np.median(snrs)),
           "snr_db_range": [float(np.min(snrs)), float(np.max(snrs))]}
    out.update({k: stats(v) for k, v in errs.items()})
    save(out, "silent")


def part_mixed():
    rec = dregon_io.load_recording(config.CALIBRATION_FLIGHT)
    fs = rec["fs"]
    su = Setup(fs, config.NFFT, config.HOP, config.BAND_HZ, *loc.direction_grid())
    locz = loc.Localiser(fs)
    quiet, wp = source_on(rec)
    noise = dregon_io.load_recording(config.NOISE_ONLY_FLIGHT)
    half = len(noise["audio"]) // 2
    nm = common.build_ncm_model(noise, su.sel, sample_range=(0, half))
    Dglobal = nm.global_ncm[:: config.MUSIC_BIN_STEP]
    w = int(config.FRAME_SEC * fs)
    ns = np.arange(half, len(noise["audio"]) - w, w // 2)
    sp = dregon_io.speeds_at(noise, noise["audio_ts"][ns + w // 2]).mean(axis=1)
    good = sp > 0.3 * np.median(sp)
    ns, sp = ns[good], sp[good]
    starts, _, gaz, gel, keep = active_windows(rec, quiet, wp)
    draws = np.random.default_rng(config.RANDOM_SEED).integers(len(ns), size=len(keep))
    out = {"windows": len(keep), "levels": {}}
    for snr in MIX_LEVELS:
        errs = {k: [] for k in ("A", "C", "D", "E", "F", "F_speed")}
        for j, i in enumerate(keep):
            xw = rec["audio"][starts[i]:starts[i] + w]
            n = noise["audio"][ns[draws[j]]:ns[draws[j]] + w]
            X = su.stft(n + xw * np.sqrt(np.mean(n ** 2) / np.mean(xw ** 2) * 10 ** (snr / 10)))
            Rx = loc.spatial_cov(X, locz.sel)
            Rb = nm.for_speed(sp[draws[j]])
            est = {"A": su.srp_of(X), "C": locz.peak(locz.sub.spectrum(Rx)), "D": locz.peak(locz.sub.spectrum(Rx, Dglobal)),
                   "E": locz.peak(locz.sub.spectrum(Rx, Rb[:: config.MUSIC_BIN_STEP])),
                   "F": su.filtered(X, nm.global_ncm), "F_speed": su.filtered(X, Rb)}
            for k, v in est.items():
                a, e = loc.apply_alignment(*v, AL)
                errs[k].append(float(loc.angular_error(a, e, gaz[i], gel[i])))
        out["levels"][str(snr)] = {k: stats(v) for k, v in errs.items()}
        print(snr, {k: round(v["within10"], 1) for k, v in out["levels"][str(snr)].items()}, flush=True)
    save(out, "mixed")


def part_bias():
    """Is the frame alignment of the DREGON flight table fair to the subspace methods? It is fitted with SRP-PHAT. Here each method's own
    estimates on the motors-off recording give its own offset, and the flight windows are rescored with that offset."""
    rec = dregon_io.load_recording(config.CALIBRATION_FLIGHT)
    fs = rec["fs"]
    su = Setup(fs, config.NFFT, config.HOP, config.BAND_HZ, *loc.direction_grid())
    locz = loc.Localiser(fs)
    quiet, wp = source_on(rec)
    starts, w, gaz, gel, keep = active_windows(rec, quiet, wp)
    est = {"A": [], "C": []}
    for i in keep:
        X = su.stft(rec["audio"][starts[i]:starts[i] + w])
        est["A"].append(su.srp_of(X))
        est["C"].append(locz.peak(locz.sub.spectrum(loc.spatial_cov(X, locz.sel))))
    est = {k: np.array(v) for k, v in est.items()}
    out = {"motors_off": {}}
    fits = {}
    for k, v in est.items():
        fits[k] = loc.fit_frame_alignment(v[:, 0], v[:, 1], gaz[keep], gel[keep])
        a, e = loc.apply_alignment(v[:, 0], v[:, 1], AL)
        out["motors_off"][k] = {"fit": fits[k], "signed_az_deg": float(np.median((a - gaz[keep] + 180) % 360 - 180)),
                                "signed_el_deg": float(np.median(e - gel[keep])),
                                "within10_table3_offset": float(100 * np.mean(loc.angular_error(a, e, gaz[keep], gel[keep]) <= 10))}
    # elevation offset of the subspace estimates, as a second fitted constant
    a, e = loc.apply_alignment(est["C"][:, 0], est["C"][:, 1], fits["C"])
    el_shift = float(np.median(gel[keep] - e))
    out["motors_off"]["C"]["within10_own_offset"] = float(100 * np.mean(loc.angular_error(a, e, gaz[keep], gel[keep]) <= 10))
    out["motors_off"]["C"]["within10_own_offset_and_el"] = float(100 * np.mean(loc.angular_error(a, e + el_shift, gaz[keep], gel[keep]) <= 10))
    out["motors_off"]["C"]["el_shift"] = el_shift
    import csv
    folder = os.path.join(os.path.dirname(config.RESULTS_DIR), os.environ.get("PAPER_RESULTS", "results"))
    with open(os.path.join(folder, "exp2_per_window.csv")) as f:
        rows = list(csv.DictReader(f))
    gt = np.array([[float(r["gt_az"]), float(r["gt_el"])] for r in rows])
    delta = fits["C"]["offset"] - AL["offset"]
    out["flights"] = {"delta_az_deg": delta, "el_shift_deg": el_shift}
    for name, col in (("C", "C_MUSIC"), ("D", "D_GEVD-MUSIC_globalNCM"), ("E", "E_GEVD-MUSIC_rpmNCM")):
        az = np.array([float(r[f"{col}_az"]) for r in rows]) + delta
        el = np.array([float(r[f"{col}_el"]) for r in rows])
        out["flights"][name] = {"table3": float(100 * np.mean(np.array([float(r[f"{col}_err"]) for r in rows]) <= 10)),
                                "own_offset": float(100 * np.mean(loc.angular_error(az, el, gt[:, 0], gt[:, 1]) <= 10)),
                                "own_offset_and_el": float(100 * np.mean(loc.angular_error(az, el + el_shift, gt[:, 0], gt[:, 1]) <= 10)),
                                "within20_table3": float(100 * np.mean(np.array([float(r[f"{col}_err"]) for r in rows]) <= 20))}
    for name, col in (("A", "A_SRP-PHAT"),):
        out["flights"][name] = {"table3": float(100 * np.mean(np.array([float(r[f"{col}_err"]) for r in rows]) <= 10)),
                                "within20_table3": float(100 * np.mean(np.array([float(r[f"{col}_err"]) for r in rows]) <= 20))}
    w_res = json.load(open(common.results_path("exp12_wf.json")))
    out["flights"]["F"] = {"table3": w_res["F_global"]["within10"], "within20_table3": w_res["F_global"]["within20"]}
    save(out, "bias")


PARTS = {"wf": part_wf, "sweep": part_sweep, "silent": part_silent, "mixed": part_mixed, "bias": part_bias}

if __name__ == "__main__":
    for part in sys.argv[1:] or PARTS:
        PARTS[part]()
