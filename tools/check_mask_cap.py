"""Share of the band removed by the tone mask of method B on DREGON, to check whether its 40% cap is reached.

Uses the same windows, 4096-point STFT (hop 1024) and two-blade comb as part_blade_mask in exp9_flight_sensitivity.py.
Usage: python tools/check_mask_cap.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import common  # noqa: E402
import config  # noqa: E402
import dregon_io  # noqa: E402
import localisation as loc  # noqa: E402
import rotor  # noqa: E402

shares, capped = [], 0
for name in config.SOURCE_FLIGHTS:
    rec = dregon_io.load_recording(name)
    fs = rec["fs"]
    f4 = np.fft.rfftfreq(4096, 1 / fs)
    band4 = (f4 >= config.BAND_HZ[0]) & (f4 <= config.BAND_HZ[1])
    cap = int(config.MASK_MAX_COVERAGE * band4.sum())
    starts, w = common.window_starts(len(rec["audio"]), fs)
    t = rec["audio_ts"][starts + w // 2]
    gaz, gel, ok = dregon_io.source_dir_at(rec, t)
    sp = dregon_io.speeds_at(rec, t)
    ms = sp.mean(axis=1)
    for i in np.where(ok & (ms > 0.3 * np.median(ms)))[0]:
        _, Xh = loc.stft_multi(rec["audio"][starts[i]:starts[i] + w], fs, 4096, 1024)
        psd = np.mean(np.abs(Xh) ** 2, axis=(0, 2))
        m = rotor.tonal_mask(f4, sp[i] * config.SPEED_SCALE, psd, band4)
        shares.append(m.sum() / band4.sum())
        capped += int(m.sum() >= cap)
shares = np.array(shares)
print(f"windows {len(shares)}, masked share mean {100 * shares.mean():.1f}%, max {100 * shares.max():.1f}%, "
      f"windows at the {100 * config.MASK_MAX_COVERAGE:.0f}% cap: {capped}")
