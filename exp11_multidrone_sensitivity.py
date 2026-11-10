"""
Experiment 11. Controls for the ego-noise comparison across five drones (Fig. 3 of the paper).

    rejection   learned rejection G for 1 to 6 eigenvectors, against diffuse and white noise on the same array
    coherence   measured magnitude-squared coherence between microphones against the diffuse-field value,
                to test directly whether the noise is uncorrelated between microphones
    tonal       tonal share for other peak thresholds, median widths and spectral resolutions

Output in results_review/: exp11_multidrone.json
"""
import json
import os

os.environ.setdefault("RESULTS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_review"))

import numpy as np
import scipy.signal
from scipy.ndimage import median_filter
from scipy.stats import spearmanr

import common
import exp6_multidrone_egonoise as e6

KS = (1, 2, 3, 4, 5, 6)
THRESHOLDS = (3.0, 6.0, 10.0)
WIDTHS = (150, 300, 600)
RESOLUTIONS = (4096, 8192, 16384)


def tonal_share(x, fs, thr, width, nperseg, band=(100, 4000), chunk_sec=1.0):
    n = max(int(chunk_sec * fs), nperseg)
    shares = []
    for s in range(0, len(x) - n + 1, n):
        f, P = scipy.signal.welch(x[s:s + n].T, fs=fs, nperseg=nperseg, axis=-1)
        P = P.mean(axis=0)
        k = (f >= band[0]) & (f <= band[1])
        db = 10 * np.log10(P + 1e-20)
        floor = median_filter(db, size=max(5, int(width / (f[1] - f[0]))), mode="nearest")
        shares.append(P[k & (db - floor >= thr)].sum() / P[k].sum())
    return float(np.mean(shares))


def pair_coherence(R, live):
    """Mean magnitude-squared coherence per bin over the pairs of live microphones. R (F, M, M)."""
    R = R[:, live][:, :, live]
    M = R.shape[1]
    p = np.real(np.einsum("fii->fi", R))
    iu = np.triu_indices(M, 1)
    return np.abs(R[:, iu[0], iu[1]]) ** 2 / (p[:, iu[0]] * p[:, iu[1]])


def main():
    out = {}
    for name, drone, spec in e6.RECORDINGS:
        rej = {k: [] for k in KS}
        tonal = {}
        coh, freqs, frames = [], None, []
        live = np.zeros(8, dtype=bool)
        for x, fs in e6.segments(name, spec):
            if x.shape[1] != 8:
                continue
            live |= np.any(x != 0, axis=0)
            half = len(x) // 2
            R1, R2 = e6.covariances(x[:half], fs)[0], e6.covariances(x[half:], fs)[0]
            for k in KS:
                e6.K_SPATIAL = k
                rej[k].append(e6.learned_rejection_db(R1, R2))
            for thr in THRESHOLDS:
                for w in WIDTHS:
                    for res in RESOLUTIONS:
                        if (thr == 6.0 and w == 300) or res == 8192:   # one setting varied at a time around the paper's
                            if res != 8192 and not (thr == 6.0 and w == 300):
                                continue
                            tonal.setdefault(f"{thr}_{w}_{res}", []).append(tonal_share(x, fs, thr, w, res))
            R, freqs = e6.covariances(x, fs)
            c = pair_coherence(R, np.any(x != 0, axis=0))
            coh.append((c.mean(), c[freqs < 1000].mean(), c[freqs >= 1000].mean()))
            frames.append(len(x))
        e6.K_SPATIAL = 4
        c = np.average(np.array(coh), axis=0, weights=np.array(frames))
        row = {"drone": drone, "live_mics": int(live.sum()), "rejection": {str(k): float(np.mean(v)) for k, v in rej.items()},
               "white_rejection": {str(k): float(10 * np.log10(live.sum() / (live.sum() - k))) for k in KS},
               "tonal": {k: float(np.mean(v)) for k, v in tonal.items()},
               "coherence": {"mean": float(c[0]), "below_1kHz": float(c[1]), "above_1kHz": float(c[2])}}
        if name in e6.GEOMETRY:
            G = e6.diffuse_covariance(e6.GEOMETRY[name][live], freqs)
            d = np.linalg.norm(e6.GEOMETRY[name][live][:, None] - e6.GEOMETRY[name][live][None], axis=-1)
            row["aperture_cm"] = float(100 * d.max())
            cd = pair_coherence(G, np.ones(int(live.sum()), dtype=bool))
            row["coherence"]["diffuse_mean"] = float(cd.mean())
            row["coherence"]["diffuse_below_1kHz"] = float(cd[freqs < 1000].mean())
            row["coherence"]["diffuse_above_1kHz"] = float(cd[freqs >= 1000].mean())
            Gn = G / np.real(np.trace(G, axis1=1, axis2=2))[:, None, None]
            row["diffuse_rejection"] = {}
            for k in KS:
                e6.K_SPATIAL = k
                row["diffuse_rejection"][str(k)] = e6.learned_rejection_db(Gn, Gn)
            e6.K_SPATIAL = 4
        out[name] = row
        print(name, json.dumps(row, default=float)[:900], flush=True)

    # is the ordering of the drones stable?
    names = list(out)
    base = [out[n]["tonal"]["6.0_300_8192"] for n in names]
    stability = {}
    for key in out[names[0]]["tonal"]:
        vals = [out[n]["tonal"][key] for n in names]
        stability[key] = {"spearman_with_paper_setting": float(spearmanr(base, vals)[0]),
                          "order": [names[i] for i in np.argsort(vals)[::-1]]}
    common.save_json({"drones": out, "tonal_stability": stability}, "exp11_multidrone.json")
    print(json.dumps(stability, indent=0))


if __name__ == "__main__":
    main()
