"""
Rotor noise helpers that use the motor speed logs.

The key idea is simple. A rotor spinning at f0 turns per second produces tones at
f0, 2 f0, 3 f0 and so on, and the strongest ones are usually at multiples of the
blade passing frequency (N_BLADES x f0). If we know f0 from the motor log, we know
where those tones are, without listening.
"""
import numpy as np
from scipy.ndimage import percentile_filter

import config
import dregon_io


def comb_bins(freqs, f0s, multiple, fmax, half_width_bins):
    """Boolean mask of STFT bins that lie near k * multiple * f0 for any rotor."""
    df = freqs[1] - freqs[0]
    mask = np.zeros(len(freqs), dtype=bool)
    for f0 in np.atleast_1d(f0s):
        if f0 <= 0:
            continue
        step = multiple * f0
        tones = np.arange(step, fmax, step)
        idx = np.round(tones / df).astype(int)
        for d in range(-half_width_bins, half_width_bins + 1):
            j = idx + d
            j = j[(j >= 0) & (j < len(freqs))]
            mask[j] = True
    return mask


def _segment_spectra(rec, n_segments=12, seg_sec=0.5, nfft=8192):
    x = rec["audio"]
    fs = rec["fs"]
    seg = int(seg_sec * fs)
    starts = np.linspace(int(2 * fs), len(x) - seg - int(2 * fs), n_segments).astype(int)
    win = np.hanning(nfft)
    freqs = np.fft.rfftfreq(nfft, 1 / fs)
    spectra, speeds = [], []
    for s in starts:
        chunk = x[s:s + seg, 0]
        frames = [chunk[i:i + nfft] * win for i in range(0, seg - nfft + 1, nfft // 2)]
        P = np.mean([np.abs(np.fft.rfft(f)) ** 2 for f in frames], axis=0)
        t_mid = rec["audio_ts"][s + seg // 2]
        sp = dregon_io.speeds_at(rec, np.array([t_mid]))[0]
        if np.all(sp > 0):
            spectra.append(P)
            speeds.append(sp)
    return freqs, np.array(spectra), np.array(speeds)


def estimate_speed_scale(rec, fmax=2000.0, verbose=True):
    """
    Find the factor that converts the logged motor speeds into turns per second.

    We try many factors and keep the one whose predicted blade passing harmonics
    line up best with the peaks in the noise-only spectrum. The score is
    "energy share at predicted bins" divided by "bin share", so a comb that
    simply covers more bins does not win. Among near ties the smallest factor is
    kept, which avoids locking onto every second harmonic.
    """
    if config.SPEED_SCALE is not None:
        return float(config.SPEED_SCALE)
    freqs, spectra, speeds = _segment_spectra(rec)
    if len(spectra) == 0:
        raise RuntimeError("No segments with running motors found for speed scale estimation")
    band = (freqs > 50) & (freqs < fmax)
    candidates = np.logspace(-3, 3, 1200)
    scores = np.full(len(candidates), -np.inf)
    for ci, s in enumerate(candidates):
        f0 = speeds * s
        if np.median(f0) < 10 or np.median(f0) > 300:   # plausible shaft rates for small multirotors
            continue
        sc = []
        for P, f0_seg in zip(spectra, f0):
            m = comb_bins(freqs, f0_seg, config.N_BLADES, fmax, 1) & band
            # a comb that covers most of the band always scores about 1, so it can not tell us anything
            if m.sum() == 0 or m.sum() > 0.25 * band.sum():
                continue
            e_frac = P[m].sum() / P[band].sum()
            b_frac = m.sum() / band.sum()
            sc.append(e_frac / b_frac)
        if sc:
            scores[ci] = np.mean(sc)
    best = np.nanmax(scores)
    good = np.where(scores >= 0.85 * best)[0]
    coarse = candidates[good.min()]

    # fine search around the coarse value, using more harmonics so small unit errors show up
    fine = coarse * np.linspace(0.97, 1.03, 601)
    fine_scores = np.array([_comb_score(freqs, spectra, speeds * s, 2 * fmax) for s in fine])
    scale = fine[int(np.argmax(fine_scores))]
    if verbose:
        print(f"  Estimated motor speed scale {scale:.5g} (median shaft rate {np.median(speeds * scale):.1f} turns/s, "
              f"comb enrichment {fine_scores.max():.2f}x)")
    return float(scale)


def _comb_score(freqs, spectra, f0, fmax):
    band = (freqs > 50) & (freqs < fmax)
    sc = []
    for P, f0_seg in zip(spectra, f0):
        m = comb_bins(freqs, f0_seg, config.N_BLADES, fmax, 1) & band
        if m.any():
            sc.append((P[m].sum() / P[band].sum()) / (m.sum() / band.sum()))
    return np.mean(sc) if sc else -np.inf


def tonal_mask(freqs, f0s, psd, band_mask):
    """
    RPM-guided tonal mask for one analysis window.

    Bins near predicted rotor harmonics are masked, but only where the spectrum
    really rises above the local noise floor (prominence check), and never more than MASK_MAX_COVERAGE
    of the band.
    """
    cand = comb_bins(freqs, f0s, config.MASK_MULTIPLE, config.MASK_FMAX, config.MASK_HALF_WIDTH_BINS) & band_mask
    if not cand.any():
        return cand
    psd_db = 10 * np.log10(psd + 1e-20)
    # height above the local noise floor (20th percentile of nearby bins)
    prominence = psd_db - percentile_filter(psd_db, 20, size=15, mode="nearest")
    m = cand & (prominence > config.MASK_PROMINENCE_DB)
    max_bins = int(config.MASK_MAX_COVERAGE * band_mask.sum())
    if m.sum() > max_bins:
        idx = np.where(m)[0]
        keep = idx[np.argsort(prominence[idx])[::-1][:max_bins]]
        m = np.zeros_like(m)
        m[keep] = True
    return m
