# Data folder

The datasets are not included in this repository. Download them as described in the main
[README](../README.md#data) and place them here. The expected layout is:

```
data/
├── DREGON/                      DREGON recordings (any sub-folder layout, the loader searches recursively)
├── SPCup19_egonoise/            IEEE SP Cup 2019 ego-noise sets (case studies 6 and 7)
│   ├── AGH/ego-noise/mic array/0.wav ... 4.wav
│   ├── Diagonal_Unloading/recordings/static/static_2m.wav, static_10m.wav
│   ├── KU_Leuven/SPCUP19_KU_Leuven_Team_1_recording.wav, ..._Team_2_recording.wav
│   └── Shout_COOEE!/SPCUP19_Shout COOEE_StaticSubmission1.wav, ...Submission2.wav
├── LOCATA/dev/task{1,3,5}/recording*/benchmark2/   LOCATA robot-head files (case study 5)
└── LOCATA_download/             working folder for tools/locata_partial.py (can be deleted afterwards)
```

Everything in this folder except this file is ignored by git.
