"""
Sound source localisation methods compared in the case studies.

A  SRP-PHAT               classic baseline (same family as the SP Cup 2019 baseline)
B  SRP-PHAT + RPM mask    ignore the frequency bins where the motor log predicts rotor tones
                          (uses a finer 4096 point STFT so single rotor tones can be resolved)
C  MUSIC                  subspace method with no noise model
D  GEVD-MUSIC global NCM  subspace method that whitens with one average noise covariance
E  GEVD-MUSIC RPM NCM     same, but the noise covariance is picked by current motor speed

C, D and E use the same direction grid and frequency band as A and B.
"""
import numpy as np
import scipy.linalg
import scipy.signal

import config

METHODS = ["A_SRP-PHAT", "B_SRP-PHAT+RPMmask", "C_MUSIC", "D_GEVD-MUSIC_globalNCM", "E_GEVD-MUSIC_rpmNCM"]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def direction_grid(az_step=None, el_step=None, el_bounds=None):
    az_step = az_step or config.AZ_STEP
    el_step = el_step or config.EL_STEP
    el_bounds = el_bounds or config.EL_BOUNDS
    az = np.arange(-180, 180, az_step)
    el = np.arange(el_bounds[0], el_bounds[1] + 1e-9, el_step)
    AZ, EL = np.meshgrid(az, el, indexing="ij")
    return AZ.ravel().astype(float), EL.ravel().astype(float)


def unit_vectors(az_deg, el_deg):
    az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
    return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=-1)


def angular_error(az1, el1, az2, el2):
    u1, u2 = unit_vectors(az1, el1), unit_vectors(az2, el2)
    return np.rad2deg(np.arccos(np.clip(np.sum(u1 * u2, axis=-1), -1, 1)))


# ---------------------------------------------------------------------------
# STFT
# ---------------------------------------------------------------------------
def stft_multi(x, fs, nfft=None, hop=None):
    """x has shape (samples, channels). Returns freqs, X with shape (channels, freqs, frames)."""
    nfft = nfft or config.NFFT
    hop = hop or config.HOP
    f, _, X = scipy.signal.stft(x.T, fs=fs, nperseg=nfft, noverlap=nfft - hop, boundary=None, padded=False)
    return f, X


# ---------------------------------------------------------------------------
# A and B, SRP-PHAT through GCC lag lookup
# ---------------------------------------------------------------------------
class SRPPHAT:
    def __init__(self, mic_pos, fs, nfft, az, el, interp=8):
        self.pairs = [(i, j) for i in range(len(mic_pos)) for j in range(i + 1, len(mic_pos))]
        self.L = nfft * interp
        self.nfft = nfft
        u = unit_vectors(az, el)
        # Arrival time at mic m is t_m = -(r_m . u) / c, and the GCC of pair (i, j)
        # peaks at lag t_i - t_j = (r_j - r_i) . u / c (in upsampled samples below)
        self.lag_idx = np.stack([
            np.round(((mic_pos[j] - mic_pos[i]) @ u.T) / config.SPEED_OF_SOUND * fs * interp).astype(int) % self.L
            for i, j in self.pairs])

    def spectrum(self, X, weight):
        """X (M, F, T) for one window, weight (F,) in [0, 1]. Returns SRP over the grid."""
        out = np.zeros(self.lag_idx.shape[1])
        for p, (i, j) in enumerate(self.pairs):
            R = np.sum(X[i] * np.conj(X[j]), axis=1)
            R = R / (np.abs(R) + 1e-12) * weight
            gcc = np.fft.irfft(R, n=self.L)
            out += gcc[self.lag_idx[p]]
        return out


# ---------------------------------------------------------------------------
# C, D, E, subspace methods
# ---------------------------------------------------------------------------
class SubspaceLocaliser:
    def __init__(self, mic_pos, freqs_sel, az, el):
        u = unit_vectors(az, el)
        delays = (u @ mic_pos.T) / config.SPEED_OF_SOUND          # (D, M), positive means earlier arrival
        self.A = np.exp(2j * np.pi * freqs_sel[:, None, None] * delays[None]).astype(np.complex64)  # (F, D, M)

    def spectrum(self, Rx, Phi=None, n_src=1):
        """Rx and Phi have shape (F, M, M). Phi None means plain MUSIC."""
        F, M, _ = Rx.shape
        out = np.zeros(self.A.shape[1])
        for f in range(F):
            R = Rx[f]
            if Phi is None:
                w, V = np.linalg.eigh(R)
                En = V[:, : M - n_src]
            else:
                N = Phi[f] + config.DIAG_LOADING * np.real(np.trace(Phi[f])) / M * np.eye(M)
                w, V = scipy.linalg.eigh(R, N)
                En = V[:, : M - n_src]
                En = En / np.linalg.norm(En, axis=0, keepdims=True)
            proj = np.sum(np.abs(self.A[f] @ En.conj()) ** 2, axis=1)
            p = 1.0 / (proj + 1e-12)
            out += p / p.max()
        return out


def spatial_cov(X, sel):
    """X (M, F, T) -> (F_sel, M, M) averaged over frames."""
    Xs = X[:, sel, :]
    return np.einsum("mft,nft->fmn", Xs, Xs.conj()) / Xs.shape[2]


# ---------------------------------------------------------------------------
# Noise covariance models (global and RPM binned)
# ---------------------------------------------------------------------------
class NoiseCovModel:
    def __init__(self, X_noise, sel, mean_speed_per_frame, n_bins):
        self.sel = sel
        Xs = X_noise[:, sel, :]
        self.global_ncm = np.einsum("mft,nft->fmn", Xs, Xs.conj()) / Xs.shape[2]
        q = np.quantile(mean_speed_per_frame, np.linspace(0, 1, n_bins + 1))
        self.edges = q
        self.centres = []
        self.bins = []
        for b in range(n_bins):
            idx = np.where((mean_speed_per_frame >= q[b]) & (mean_speed_per_frame <= q[b + 1]))[0]
            if len(idx) < 10:
                continue
            Xb = Xs[:, :, idx]
            self.bins.append(np.einsum("mft,nft->fmn", Xb, Xb.conj()) / len(idx))
            self.centres.append(np.mean(mean_speed_per_frame[idx]))
        self.centres = np.array(self.centres)

    def for_speed(self, mean_speed):
        b = int(np.argmin(np.abs(self.centres - mean_speed)))
        return self.bins[b]


# ---------------------------------------------------------------------------
# Running all methods on one analysis window
# ---------------------------------------------------------------------------
class Localiser:
    def __init__(self, fs, az=None, el=None, mic_pos=None):
        """mic_pos (M, 3) in metres; defaults to the DREGON array in config.MIC_POS."""
        self.fs = fs
        mic_pos = config.MIC_POS if mic_pos is None else np.asarray(mic_pos, dtype=float)
        self.az, self.el = (az, el) if az is not None else direction_grid()
        self.freqs = np.fft.rfftfreq(config.NFFT, 1 / fs)
        self.band = (self.freqs >= config.BAND_HZ[0]) & (self.freqs <= config.BAND_HZ[1])
        sel = np.where(self.band)[0][:: config.MUSIC_BIN_STEP]
        self.sel = sel
        self.srp = SRPPHAT(mic_pos, fs, config.NFFT, self.az, self.el)
        self.sub = SubspaceLocaliser(mic_pos, self.freqs[sel], self.az, self.el)
        # higher resolution SRP for the RPM mask method
        self.freqs_hr = np.fft.rfftfreq(config.MASK_NFFT, 1 / fs)
        self.band_hr = (self.freqs_hr >= config.BAND_HZ[0]) & (self.freqs_hr <= config.BAND_HZ[1])
        self.srp_hr = SRPPHAT(mic_pos, fs, config.MASK_NFFT, self.az, self.el, interp=2)

    def peak(self, spec):
        k = int(np.argmax(spec))
        return self.az[k], self.el[k]

    def run(self, x, f0s=None, ncm_model=None, mean_speed=None, methods=METHODS):
        """
        x is one analysis window, shape (samples, channels).
        f0s are the four rotor shaft rates in turns/s for this window.
        Returns {method: (azimuth, elevation)} in the array frame.
        """
        _, X = stft_multi(x, self.fs)
        res = {}
        base_w = self.band.astype(float)
        if "A_SRP-PHAT" in methods:
            res["A_SRP-PHAT"] = self.peak(self.srp.spectrum(X, base_w))
        if "B_SRP-PHAT+RPMmask" in methods and f0s is not None:
            import rotor
            _, Xh = stft_multi(x, self.fs, config.MASK_NFFT, config.MASK_HOP)
            psd = np.mean(np.abs(Xh) ** 2, axis=(0, 2))
            m = rotor.tonal_mask(self.freqs_hr, f0s, psd, self.band_hr)
            self.last_mask_share = m.sum() / self.band_hr.sum()
            res["B_SRP-PHAT+RPMmask"] = self.peak(self.srp_hr.spectrum(Xh, self.band_hr * (~m)))
        need_sub = any(k in methods for k in METHODS[2:])
        if need_sub:
            Rx = spatial_cov(X, self.sel)
            if "C_MUSIC" in methods:
                res["C_MUSIC"] = self.peak(self.sub.spectrum(Rx))
            if ncm_model is not None and "D_GEVD-MUSIC_globalNCM" in methods:
                res["D_GEVD-MUSIC_globalNCM"] = self.peak(self.sub.spectrum(Rx, ncm_model.global_ncm))
            if ncm_model is not None and mean_speed is not None and "E_GEVD-MUSIC_rpmNCM" in methods:
                res["E_GEVD-MUSIC_rpmNCM"] = self.peak(self.sub.spectrum(Rx, ncm_model.for_speed(mean_speed)))
        return res


# ---------------------------------------------------------------------------
# Coordinate frame alignment between array frame and ground truth frame
# ---------------------------------------------------------------------------
def fit_frame_alignment(est_az, est_el, gt_az, gt_el):
    """
    Find az_gt = s * az_est + offset and el_gt = e * el_est that best match.
    Only 3 global parameters, used to align the array frame with the ground truth frame.
    """
    best = (np.inf, 0, 1, 1)
    est_az, est_el, gt_az, gt_el = map(np.asarray, (est_az, est_el, gt_az, gt_el))
    for s in (1, -1):
        for e in (1, -1):
            for off in range(0, 360, 1):
                a = (s * est_az + off + 180) % 360 - 180
                err = np.median(angular_error(a, e * est_el, gt_az, gt_el))
                if err < best[0]:
                    best = (err, off, s, e)
    return {"median_err": best[0], "offset": best[1], "az_sign": best[2], "el_sign": best[3]}


def apply_alignment(az, el, al):
    a = (al["az_sign"] * np.asarray(az) + al["offset"] + 180) % 360 - 180
    return a, al["el_sign"] * np.asarray(el)


def summarise_errors(errs):
    errs = np.asarray(errs)
    out = {"n": int(len(errs)), "median_deg": float(np.median(errs)), "mean_deg": float(np.mean(errs))}
    for t in config.ACC_THRESHOLDS:
        out[f"acc@{t}deg"] = float(np.mean(errs <= t))
    return out
