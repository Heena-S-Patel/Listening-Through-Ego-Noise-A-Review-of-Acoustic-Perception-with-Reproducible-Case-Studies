"""
Central settings for the case studies.

Everything that depends on the dataset or on a design choice lives here,
so the experiment scripts stay short and easy to read.
"""
import os
import numpy as np

# ---------------------------------------------------------------------------
# Where the data and results live
# ---------------------------------------------------------------------------
# Point DREGON_DIR at the folder where you unzipped the DREGON files.
# You can also set it from the shell:  export DREGON_DIR=/path/to/DREGON
DATA_DIR = os.environ.get("DREGON_DIR", os.path.join(os.path.dirname(__file__), "data", "DREGON"))
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(os.path.dirname(__file__), "results"))

# Recording names used by the experiments (folder or file stem in DREGON)
NOISE_ONLY_FLIGHT = "DREGON_free-flight_nosource_room1"        # pure ego-noise, same room as the source flights
SOURCE_FLIGHTS = [
    "DREGON_free-flight_whitenoise-high_room1",
    "DREGON_free-flight_whitenoise-low_room1",
]
CALIBRATION_FLIGHT = "DREGON_silent-flight_whitenoise-low_room1"  # motors off, used only to align coordinate frames
CLEAN_RECORDINGS = "DREGON_clean_recordings_whitenoise"           # tripod recordings named AZ_EL_DIST.wav

# ---------------------------------------------------------------------------
# Physics and hardware
# ---------------------------------------------------------------------------
FS = 44100            # DREGON audio sample rate (Hz)
SPEED_OF_SOUND = 343.0
N_BLADES = 2          # propeller blades per rotor (blade passing frequency = N_BLADES * shaft rate)

# Official DREGON microphone positions (metres, array barycentre at origin), from micPos.txt
MIC_POS = np.array([
    [ 0.0615, -0.0420, -0.0410],
    [ 0.0615,  0.0420,  0.0410],
    [ 0.0420,  0.0615, -0.0410],
    [-0.0420,  0.0615,  0.0410],
    [-0.0615,  0.0420, -0.0410],
    [-0.0615, -0.0420,  0.0410],
    [-0.0420, -0.0615, -0.0410],
    [ 0.0420, -0.0615,  0.0410],
])

# Approximate rotor positions in the array frame (from the SP Cup 2019 plot_structure.m)
_r = 0.485 / 2
ROTOR_POS = np.array([[_r * np.cos(np.deg2rad(a + 90)) + 0.0115,
                       _r * np.sin(np.deg2rad(a + 90)),
                       0.1915] for a in (45, 135, 225, 315)])

# Motor speed units in DREGON logs are not documented on the website.
# None = detect automatically from the noise-only recording (see rotor.estimate_speed_scale).
# If you know the unit, set the factor that converts log values to turns per second.
# Checked on the noise-only flight: the logged values are already turns per second
# (shaft harmonics line up best at a factor of 0.997, about 2.8x energy enrichment).
SPEED_SCALE = 1.0

# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------
NFFT = 1024            # STFT size (about 23 ms at 44.1 kHz)
HOP = 512
FRAME_SEC = 0.5        # localisation window, same as the SP Cup 2019 flight task
FRAME_HOP_SEC = 0.25
BAND_HZ = (300.0, 4000.0)   # frequency band used for localisation
MUSIC_BIN_STEP = 2          # use every 2nd STFT bin for the subspace methods (speed)

# Direction search grid (degrees)
AZ_STEP = 3
EL_STEP = 3
EL_BOUNDS = (-90, 90)

# Rotor harmonic masking (method B)
MASK_NFFT = 4096            # finer frequency resolution (about 11 Hz) so individual rotor tones can be separated
MASK_HOP = 1024
MASK_MULTIPLE = N_BLADES    # mask blade passing harmonics (k x N_BLADES x shaft rate)
MASK_FMAX = 4000.0          # mask predicted rotor tones up to this frequency
MASK_HALF_WIDTH_BINS = 0    # +- bins around each predicted tone
MASK_PROMINENCE_DB = 3.0    # only mask predicted tones that rise above the local noise floor
MASK_MAX_COVERAGE = 0.4     # never mask more than this share of the band

# Noise covariance model
NCM_RPM_BINS = 6            # number of motor speed bins for the RPM-informed noise covariance
DIAG_LOADING = 1e-3         # relative diagonal loading for numerical stability

# SNR sweep (experiment 3)
SNR_LIST_DB = [0, -5, -10, -15, -20, -25]
SEGMENTS_PER_DIRECTION = 2
RANDOM_SEED = 0

# Accuracy thresholds for reporting (degrees)
ACC_THRESHOLDS = (10, 20)
