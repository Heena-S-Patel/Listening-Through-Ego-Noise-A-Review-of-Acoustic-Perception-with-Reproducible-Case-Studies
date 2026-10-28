"""
Create a small synthetic dataset with exactly the same file layout as DREGON.

This is ONLY for checking that the code runs end to end (for example on a laptop
before the real data is downloaded). The physics is simplified, so the numbers it
produces say nothing about real drones. Report results only from the real DREGON files.

Usage
    python make_synthetic_dregon.py            # writes to data/SYNTH_DREGON
    DREGON_DIR=data/SYNTH_DREGON python run_all.py
"""
import os
import sys

import numpy as np
import scipy.io
import soundfile as sf

import config

FS = config.FS
C = config.SPEED_OF_SOUND
OUT = os.path.join(os.path.dirname(__file__), "data", "SYNTH_DREGON")
GT_AZ_OFFSET = 90.0          # pretend the ground truth frame is rotated, to exercise frame alignment
SPEED_UNIT = 60.0            # store motor speeds in rpm, to exercise speed unit detection
T0 = 1.5e9
rng = np.random.default_rng(1)


def unit(az, el):
    az, el = np.deg2rad(az), np.deg2rad(el)
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def delay_signal(sig, delays_s, gains=None):
    """Apply per microphone fractional delays in the frequency domain. Returns (N, M)."""
    n = len(sig)
    S = np.fft.rfft(sig)
    f = np.fft.rfftfreq(n, 1 / FS)
    gains = np.ones(len(delays_s)) if gains is None else gains
    return np.stack([g * np.fft.irfft(S * np.exp(-2j * np.pi * f * d), n=n) for d, g in zip(delays_s, gains)], axis=1)


def rotor_noise(dur, rps_tracks):
    """Harmonic plus broadband noise from four rotors, propagated to the eight microphones."""
    n = int(dur * FS)
    t_log = np.arange(rps_tracks.shape[0]) / 100.0
    t = np.arange(n) / FS
    mics = np.zeros((n, 8))
    for m in range(4):
        rps = np.interp(t, t_log, rps_tracks[:, m])
        phase = 2 * np.pi * np.cumsum(rps) / FS
        s = np.zeros(n)
        kmax = int(4500 / rps.max())
        for k in range(1, kmax + 1):
            a = (3.0 if k % config.N_BLADES == 0 else 1.0) / k ** 0.6
            s += a * np.cos(k * phase + rng.uniform(0, 2 * np.pi))
        bb = np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) / np.maximum(np.fft.rfftfreq(n, 1 / FS), 50) ** 0.5, n=n)
        s += 0.8 * bb / bb.std() * s.std()
        d = np.linalg.norm(config.MIC_POS - config.ROTOR_POS[m], axis=1)
        mics += delay_signal(s, d / C, 1.0 / d)
    # uncorrelated wind and sensor noise
    wind = np.fft.irfft(np.fft.rfft(rng.standard_normal((n, 8)), axis=0) *
                        (1 / np.maximum(np.fft.rfftfreq(n, 1 / FS), 30))[:, None], n=n, axis=0)
    mics += 0.3 * mics.std() * wind / wind.std() + 0.02 * mics.std() * rng.standard_normal((n, 8))
    return mics


def moving_source(dur, az_track, el_track, block_sec=2.0):
    """Far field white noise source whose direction changes every block (with crossfades)."""
    n = int(dur * FS)
    src = rng.standard_normal(n)
    out = np.zeros((n, 8))
    blk = int(block_sec * FS)
    fade = int(0.02 * FS)
    for b, s0 in enumerate(range(0, n, blk)):
        s1 = min(n, s0 + blk)
        a0, a1 = max(0, s0 - fade), min(n, s1 + fade)
        u = unit(az_track[b], el_track[b])
        delays = -(config.MIC_POS @ u) / C
        seg = delay_signal(src[a0:a1], delays)
        w = np.ones(a1 - a0)
        if a0 < s0:
            w[: s0 - a0 + fade] = np.linspace(0, 1, s0 - a0 + fade)
        if a1 > s1:
            w[-(a1 - s1 + fade):] = np.linspace(1, 0, a1 - s1 + fade)
        out[a0:a1] += seg * w[:, None]
    return out


def speed_tracks(dur):
    t = np.arange(int(dur * 100)) / 100.0
    base = 70 + 8 * np.sin(2 * np.pi * t / 17) + 4 * np.sin(2 * np.pi * t / 5.3)
    return np.stack([base + o + 0.8 * np.sin(2 * np.pi * t / (3 + m)) for m, o in enumerate((-2, 1, 3, -1))], axis=1)


def save(name, audio, motors=None, source=None, with_imu=True):
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    sf.write(os.path.join(d, f"{name}.wav"), audio.astype(np.float32), FS, subtype="FLOAT")
    n = len(audio)
    scipy.io.savemat(os.path.join(d, f"{name}_audiots.mat"), {"audio_timestamps": T0 + np.arange(n) / FS})
    if motors is not None:
        ts = T0 + np.arange(motors.shape[0]) / 100.0
        scipy.io.savemat(os.path.join(d, f"{name}_motors.mat"),
                         {"motor": {"timestamps": ts, "measured": motors * SPEED_UNIT, "command": motors * SPEED_UNIT}})
    if with_imu:
        k = int(n / FS * 100)
        scipy.io.savemat(os.path.join(d, f"{name}_imu.mat"),
                         {"imu": {"timestamps": T0 + np.arange(k) / 100.0,
                                  "angular_velocity": 0.05 * rng.standard_normal((k, 3)),
                                  "acceleration": 0.2 * rng.standard_normal((k, 3)) + [0, 0, 9.81]}})
    if source is not None:
        az_b, el_b, block_sec = source
        k = int(n / FS * 100)
        t = np.arange(k) / 100.0
        b = np.minimum((t / block_sec).astype(int), len(az_b) - 1)
        az_gt = (np.asarray(az_b)[b] + GT_AZ_OFFSET + 180) % 360 - 180
        scipy.io.savemat(os.path.join(d, f"{name}_sourcepos.mat"),
                         {"source_position": {"timestamps": T0 + t, "azimuth": az_gt,
                                              "elevation": np.asarray(el_b)[b], "distance": 2.0 + 0 * t}})


def main():
    os.makedirs(OUT, exist_ok=True)
    print("Writing synthetic DREGON-like data to", OUT)

    dur = 40.0
    sp = speed_tracks(dur)
    noise = rotor_noise(dur, sp)
    gain = 0.1 / noise.std()
    save(config.NOISE_ONLY_FLIGHT, noise * gain, motors=sp)
    print("  noise-only flight done")

    for name, snr in zip(config.SOURCE_FLIGHTS, (-12.0, -16.0)):
        dur = 24.0
        sp = speed_tracks(dur) + rng.uniform(-3, 3)
        noise = rotor_noise(dur, sp) * gain
        nb = int(dur / 2) + 1
        az_b = rng.uniform(-180, 180, nb)
        el_b = rng.uniform(-70, -10, nb)
        src = moving_source(dur, az_b, el_b)
        src *= np.sqrt(np.mean(noise ** 2) / np.mean(src ** 2) * 10 ** (snr / 10))
        save(name, noise + src, motors=sp, source=(az_b, el_b, 2.0))
        print(f"  {name} done (SNR {snr} dB)")

    dur = 16.0
    nb = int(dur / 2) + 1
    az_b, el_b = rng.uniform(-180, 180, nb), rng.uniform(-70, -10, nb)
    src = moving_source(dur, az_b, el_b)
    src = src / src.std() * 0.05 + 0.001 * rng.standard_normal(src.shape)
    save(config.CALIBRATION_FLIGHT, src, motors=None, source=(az_b, el_b, 2.0), with_imu=False)
    print("  silent calibration flight done")

    d = os.path.join(OUT, config.CLEAN_RECORDINGS)
    os.makedirs(d, exist_ok=True)
    for az in (-135, -90, -45, 0, 45, 90, 135, 180):
        for el in (-60, -30):
            s = moving_source(3.0, [az, az], [el, el], block_sec=3.0)
            s = s / s.std() * 0.05
            az_name = (az + GT_AZ_OFFSET + 180) % 360 - 180
            sf.write(os.path.join(d, f"{int(az_name)}_{el}_2.wav"), s.astype(np.float32), FS, subtype="FLOAT")
    print("  clean tripod recordings done")


if __name__ == "__main__":
    sys.exit(main())
