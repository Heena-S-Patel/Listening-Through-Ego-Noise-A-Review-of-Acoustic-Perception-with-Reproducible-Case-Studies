"""
Loading DREGON recordings and logs.

DREGON stores each recording as
    <name>.wav                 8 channel audio at 44.1 kHz
    <name>_audiots.mat         audio_timestamps, one per sample
    <name>_motors.mat          motor.timestamps, motor.measured (optional), motor.command
    <name>_imu.mat             imu.timestamps, imu.angular_velocity, imu.acceleration
    <name>_sourcepos.mat       source_position.timestamps, .azimuth, .elevation, .distance
The files can sit directly in DATA_DIR or inside a sub folder per recording,
the loader searches recursively.
"""
import glob
import os
import re

import numpy as np
import scipy.io

try:
    import soundfile as sf
except ImportError:  # pragma: no cover
    sf = None

import config


def _find(stem_with_ext, root=None):
    root = root or config.DATA_DIR
    hits = glob.glob(os.path.join(root, "**", stem_with_ext), recursive=True)
    return sorted(hits)[0] if hits else None


def read_wav(path):
    if sf is not None:
        x, fs = sf.read(path, dtype="float32", always_2d=True)
    else:
        fs, x = scipy.io.wavfile.read(path)
        x = x.astype(np.float32)
        if x.ndim == 1:
            x = x[:, None]
        if np.issubdtype(x.dtype, np.integer):
            x /= np.iinfo(x.dtype).max
    return x, fs


def _mat_struct_to_dict(obj):
    if isinstance(obj, dict):
        return {k: np.asarray(v) for k, v in obj.items() if not k.startswith("__")}
    if hasattr(obj, "_fieldnames"):
        return {k: np.asarray(getattr(obj, k)) for k in obj._fieldnames}
    raise TypeError("Unexpected MATLAB struct type")


def load_mat_struct(path, name):
    """Return the fields of the MATLAB struct `name` as a dict of numpy arrays."""
    try:
        m = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
        if name in m:
            obj = m[name]
            return obj if isinstance(obj, np.ndarray) and obj.dtype != object else _mat_struct_to_dict(obj)
        keys = [k for k in m if not k.startswith("__")]
        if len(keys) == 1:
            obj = m[keys[0]]
            return obj if isinstance(obj, np.ndarray) and obj.dtype != object else _mat_struct_to_dict(obj)
        raise KeyError(f"{name} not found in {path}, keys are {keys}")
    except NotImplementedError:
        # MATLAB v7.3 files are HDF5
        import h5py
        with h5py.File(path, "r") as f:
            g = f[name]
            if isinstance(g, h5py.Dataset):
                return np.array(g).T.squeeze()
            return {k: np.array(g[k]).T.squeeze() for k in g.keys()}


def load_recording(name, root=None):
    """Load audio plus every log that exists for one DREGON recording."""
    wav = _find(f"{name}.wav", root)
    if wav is None:
        raise FileNotFoundError(f"Could not find {name}.wav under {root or config.DATA_DIR}")
    x, fs = read_wav(wav)
    rec = {"name": name, "audio": x, "fs": fs}

    p = _find(f"{name}_audiots.mat", root)
    if p:
        rec["audio_ts"] = np.asarray(load_mat_struct(p, "audio_timestamps"), dtype=np.float64).ravel()
    else:
        rec["audio_ts"] = np.arange(len(x)) / fs

    p = _find(f"{name}_motors.mat", root)
    if p:
        m = load_mat_struct(p, "motor")
        rec["motors"] = {k: np.asarray(v, dtype=np.float64) for k, v in m.items()}

    p = _find(f"{name}_imu.mat", root)
    if p:
        m = load_mat_struct(p, "imu")
        rec["imu"] = {k: np.asarray(v, dtype=np.float64) for k, v in m.items()}

    p = _find(f"{name}_sourcepos.mat", root)
    if p:
        s = load_mat_struct(p, "source_position")
        s = {k: np.asarray(v, dtype=np.float64).ravel() for k, v in s.items()}
        # DREGON angles are expected in degrees. Convert if they look like radians.
        for k in ("azimuth", "elevation"):
            if np.nanmax(np.abs(s[k])) <= 2 * np.pi + 0.1:
                s[k] = np.rad2deg(s[k])
        rec["sourcepos"] = s
    return rec


def motor_speeds(rec):
    """Return (timestamps, speeds[T,4], kind) using measured speeds when available."""
    m = rec.get("motors")
    if m is None:
        return None, None, None
    kind = "measured" if "measured" in m and np.size(m["measured"]) > 0 else "command"
    sp = np.atleast_2d(m[kind])
    if sp.shape[0] == 4 and sp.shape[1] != 4:
        sp = sp.T
    return m["timestamps"].ravel(), sp, kind


def speeds_at(rec, t):
    """Interpolate the four motor speeds (raw log units) at times t."""
    ts, sp, _ = motor_speeds(rec)
    return np.stack([np.interp(t, ts, sp[:, i]) for i in range(sp.shape[1])], axis=-1)


def source_dir_at(rec, t):
    """Interpolate ground truth azimuth and elevation (degrees) at times t."""
    s = rec["sourcepos"]
    az = np.unwrap(np.deg2rad(s["azimuth"]))
    az_t = np.rad2deg(np.interp(t, s["timestamps"], az))
    el_t = np.interp(t, s["timestamps"], s["elevation"])
    valid = (t >= s["timestamps"].min()) & (t <= s["timestamps"].max())
    return (az_t + 180) % 360 - 180, el_t, valid


def list_clean_recordings(folder_name=None, root=None):
    """Return a list of (az, el, dist, path) for the tripod recordings named AZ_EL_DIST.wav."""
    root = root or config.DATA_DIR
    folder_name = folder_name or config.CLEAN_RECORDINGS
    dirs = [d for d in glob.glob(os.path.join(root, "**", folder_name), recursive=True) if os.path.isdir(d)]
    search = dirs[0] if dirs else root
    out = []
    pat = re.compile(r"^(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)\.wav$")
    for p in sorted(glob.glob(os.path.join(search, "**", "*.wav"), recursive=True)):
        mm = pat.match(os.path.basename(p))
        if mm:
            out.append((float(mm.group(1)), float(mm.group(2)), float(mm.group(3)), p))
    return out
