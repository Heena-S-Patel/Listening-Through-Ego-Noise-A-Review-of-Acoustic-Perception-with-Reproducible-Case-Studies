"""
Experiment 16. Can the outdoor speech flights of the KumamoTech set (SP Cup 2019) be scored?

The two flights hold 16-channel audio with a whistle at about 2.9 kHz and a voice from two loudspeakers, with the logged
position and attitude of the drone and the source positions. The voice has no activity labels, so a voice score could
only be trusted if the ground truth is first shown to work. The whistle allows this check: it is clearly audible and
narrowband, and its on and off periods can be read from the audio. Frames are aligned per flight on the whistle itself,
which favours the check. D and F use a noise covariance from whistle-off frames of the same flight, in the whistle band.

Needs the rosbags package to read the flights. Output in results_review/: exp16_outdoor_flight.json
"""
import json
import os
import re
import tempfile

os.environ.setdefault("RESULTS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_review"))

import numpy as np
import scipy.signal

import common
import config
import exp12_dregon_controls as e12
import localisation as loc

SET = os.path.join(os.path.dirname(config.DATA_DIR), "SPCup19_egonoise", "KumamoTech")
CACHE = os.environ.get("KUMAMOTO_CACHE", tempfile.gettempdir())
FS, HOP = 16000, 160                                   # each message holds 512 samples, of which 160 are new
SPANS = {"data1": (8, 83), "data2": (14, 135)}         # steady flight with the sources playing
WHISTLE, BESIDE = (2750, 3050), ((2400, 2700), (3100, 3400))
ON_DB, OFF_DB = 10.0, 2.0
WINDOW = 8000


def extract(name):
    """Audio, pose and source positions of one flight, cached outside the repository."""
    path = os.path.join(CACHE, f"kumamoto_{name}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return {k: d[k] for k in d.files}
    from pathlib import Path
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore, get_types_from_msg
    store = get_typestore(Stores.ROS1_NOETIC)
    frames, stamps, att, pos, src = [], [], [], [], []
    with Reader(Path(SET) / f"{name}.bag") as r:
        types = {}
        for c in r.connections:
            types.update(get_types_from_msg(c.msgdef.data, c.msgtype))
        store.register(types)
        for c, t, raw in r.messages():
            m = store.deserialize_ros1(raw, c.msgtype)
            if c.topic == "/HarkWave":
                frames.append(np.stack([np.asarray(v.wavedata, dtype=np.float32) for v in m.src]))
                stamps.append(t / 1e9)
            elif c.topic == "/attitude":
                att.append((t / 1e9, m.x, m.y, m.z, m.w))
            elif c.topic == "/position":
                pos.append((t / 1e9, m.x, m.y, m.z))
            elif c.topic == "/sound_sources_position":
                src.append([float(v) for v in re.findall(r"\(([-\d.]+),([-\d.]+)\)", m.data)[0]])
    frames = np.stack(frames)
    x = np.concatenate([frames[0].T[:512 - HOP], frames[:, :, 512 - HOP:].transpose(0, 2, 1).reshape(-1, frames.shape[1])]) / 32768
    out = {"x": x.astype(np.float32), "t0": stamps[0], "att": np.array(att), "pos": np.array(pos), "src": np.array(src)}
    np.savez(path, **out)
    return out


def rotation(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def truth(d, t, sources):
    """Source directions in the body frame at time t. The source height is not logged, so ground level is assumed."""
    tb = t + float(d["t0"])
    p = [np.interp(tb, d["pos"][:, 0], d["pos"][:, c]) for c in (1, 2, 3)]
    R = rotation(d["att"][np.argmin(np.abs(d["att"][:, 0] - tb)), 1:])
    out = []
    for s in sources:
        v = R.T @ np.array([s[0] - p[0], s[1] - p[1], -p[2]])
        v /= np.linalg.norm(v)
        out.append((np.rad2deg(np.arctan2(v[1], v[0])), np.rad2deg(np.arcsin(np.clip(v[2], -1, 1)))))
    return out


def main():
    mic = np.array([[float(v) for v in re.findall(r'"(-?[\d.]+)"', l)] for l in open(os.path.join(SET, "microphones.txt")) if l.startswith("ch")])
    az, el = loc.direction_grid()
    su = e12.Setup(FS, config.NFFT, config.HOP, WHISTLE, az, el, mic_pos=mic)
    res = {"chance_azimuth_within20": 100 * 40 / 360}
    for name, (t0, t1) in SPANS.items():
        d = extract(name)
        x = d["x"]
        f, tt, Z = scipy.signal.stft(x[:, :8].mean(axis=1), FS, nperseg=1024, noverlap=512)
        P = np.abs(Z) ** 2
        on_band = (f > WHISTLE[0]) & (f < WHISTLE[1])
        off_band = ((f > BESIDE[0][0]) & (f < BESIDE[0][1])) | ((f > BESIDE[1][0]) & (f < BESIDE[1][1]))
        prom = 10 * np.log10(P[on_band].mean(axis=0) / P[off_band].mean(axis=0))
        starts = np.arange(int(t0 * FS), int(t1 * FS) - WINDOW, FS // 4)
        p = np.array([prom[(tt >= s / FS) & (tt <= (s + WINDOW) / FS)].mean() for s in starts])
        on, off = starts[p > ON_DB], starts[p < OFF_DB]
        Rn = e12.covariance(np.concatenate([su.stft(x[s:s + WINDOW]) for s in off], axis=2)[:, su.sel])
        est = {"A": [], "D": [], "F": []}
        gts = []
        for s in on:
            X = su.stft(x[s:s + WINDOW])
            est["A"].append(su.srp_of(X))
            est["D"].append(su.music(X, Rn))
            est["F"].append(su.filtered(X, Rn))
            gts.append(truth(d, (s + WINDOW / 2) / FS, d["src"]))
        gts = np.array(gts)
        row = {"whistle_windows": len(on), "whistle_off_windows": len(off)}
        for m, v in est.items():
            v = np.array(v)
            row[m] = {}
            for k in range(2):
                al = loc.fit_frame_alignment(v[:, 0], v[:, 1], gts[:, k, 0], gts[:, k, 1])
                a, e = loc.apply_alignment(v[:, 0], v[:, 1], al)
                daz = np.abs((a - gts[:, k, 0] + 180) % 360 - 180)
                row[m][f"source{k + 1}"] = {"offset": al["offset"], "median_azimuth_error": float(np.median(daz)), "azimuth_within20": float(100 * np.mean(daz <= 20)),
                                            "within20": float(100 * np.mean(loc.angular_error(a, e, gts[:, k, 0], gts[:, k, 1]) <= 20))}
        res[name] = row
        print(name, json.dumps(row), flush=True)
    common.save_json(res, "exp16_outdoor_flight.json")


if __name__ == "__main__":
    main()
