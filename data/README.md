# Data folder

The datasets are not included in this repository. Download them as described in the main
[README](../README.md#data) and place them here. The expected layout is:

```
data/
├── DREGON/                      DREGON recordings (any sub-folder layout, the loader searches recursively)
├── SPCup19_egonoise/            IEEE SP Cup 2019 ego-noise sets
│   ├── AGH/ego-noise/mic array/0.wav ... 4.wav        case studies 6, 7 and experiments 11, 13
│   ├── AGH/static clean/*.wav                          experiment 13 (clean speech and white noise)
│   ├── AGH/static corrupted/*.wav                      experiment 13 (real recordings with rotor noise)
│   ├── Diagonal_Unloading/recordings/static/static_2m.wav, static_10m.wav
│   ├── KU_Leuven/SPCUP19_KU_Leuven_Team_1_recording.wav, ..._Team_2_recording.wav
│   ├── Shout_COOEE!/SPCUP19_Shout COOEE_StaticSubmission1.wav, ...Submission2.wav
│   └── KumamoTech/data1.bag, data2.bag, microphones.txt   experiment 16 only (ground-truth check)
├── LOCATA/dev/task{1,3,5}/recording*/benchmark2/      LOCATA robot head (case study 5, experiment 10)
├── LOCATA/dev/.../eigenmike/, LOCATA/eval/...          LOCATA evaluation set and Eigenmike (experiment 14)
└── LOCATA_download/             working folder for tools/locata_partial.py (can be deleted afterwards)
```

The exact SP Cup files are listed in `RECORDINGS` at the top of `exp6_multidrone_egonoise.py`. Experiment 14 looks
for LOCATA in `LOCATA_ROOT` (default `data/LOCATA`), which holds `dev/` and `eval/`.

Everything in this folder except this file is ignored by git.
