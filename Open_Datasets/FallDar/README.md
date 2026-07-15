# FallDar dataset preprocessing

This directory contains the scripts used to preprocess the FallDar Wi-Fi CSI
dataset for RF-CRATE.

## Dataset availability

The raw FallDar dataset is **not included in this repository** because it was
created and distributed by its original authors. Users must obtain the dataset
from the original source or contact the authors, and must follow the original
dataset's license and terms of use.

Please cite the original paper when using this dataset:

> Zheng Yang, Yi Zhang, and Qian Zhang, "Rethinking Fall Detection With
> Wi-Fi," *IEEE Transactions on Mobile Computing*, vol. 22, no. 10,
> pp. 6126–6143, Oct. 2023.

## Python preprocessing

The `preprocess_csi.py` script reads Intel 5300 CSI Tool `.dat` files, extracts
their complex CSI matrices, applies CSI normalization, phase cleaning, and
signal filtering, and saves the processed matrices as NumPy files. It also
provides `get_csi_dfs` for converting the processed CSI into Doppler frequency
spectrum (DFS) representations.

### 1. Install dependencies

Use a Python environment with the following packages:

```bash
pip install numpy scipy torch tqdm CSIKit
```

### 2. Arrange the raw data

Create a `raw_data` directory here and retain the date-based directory
structure of the original dataset. The complete directory structure before and
after preprocessing is:

```text
FallDar/
├── README.md
├── Intro_preprocess_vis.ipynb       # Dataset exploration, preprocessing, and visualization
├── preprocess_csi.py
├── metas.csv
├── matlab_scripts/
│   ├── main_fall_prep.m
│   ├── myacf.m
│   └── preproc_matrix.m
├── raw_data/                        # Not distributed with this repository
│   ├── 20200525/
│   │   ├── zhangyi-1-1-r.dat
│   │   └── ...
│   ├── 20200526/
│   │   └── ...
│   └── ...
├── CSI_Processed/                   # Generated processed complex CSI
│   ├── 20200525/
│   │   ├── zhangyi-1-1-r.npy
│   │   └── ...
│   └── ...
├── DFS/                             # Generated Doppler spectra, when enabled
│   ├── 20200525/
│   │   ├── zhangyi-1-1-r.npy
│   │   └── ...
│   └── ...
└── preprocessing_log.txt            # Per-directory processing summary
```

Raw filenames are expected to follow this convention:

```text
<participant>-<activity-index>-<repeat-index>-r.dat
```

For example, `zhangyi-1-1-r.dat` is the first repetition of activity 1
performed by `zhangyi`.

The date directories used by the released metadata are:

```text
20200525  20200526  20200601  20200614  20200623  20200705
20200706  20200708  20200709  20200712  20200713
```

### 3. Run preprocessing

The script uses paths relative to this directory, so run it from
`Open_Datasets/FallDar`:

```bash
cd Open_Datasets/FallDar
python preprocess_csi.py
```

The script processes every date directory under `raw_data`. Files containing
fewer than 1,000 CSI frames are skipped. Existing output files are also skipped,
so the command can be rerun without processing completed files again.

### 4. Processed output

Processed files are written using the same directory and filename structure:

```text
CSI_Processed/
├── 20200525/
│   ├── zhangyi-1-1-r.npy
│   └── ...
└── ...
```

Each `.npy` file contains a `complex64` array with shape:

```text
(number_of_frames, number_of_subcarriers, number_of_Rx_antennas, number_of_Tx_antennas)
```

Load a processed sample with:

```python
import numpy as np

csi = np.load("CSI_Processed/20200525/zhangyi-1-1-r.npy")
print(csi.shape, csi.dtype)
```

The number of successful and failed files is appended to
`preprocessing_log.txt`.

### 5. Doppler frequency spectrum (DFS)

The `get_csi_dfs` function converts processed CSI into a time-frequency
representation using a short-time Fourier transform (STFT). Its default
parameters are:

- CSI sampling rate: 1,000 Hz
- STFT window size: 256 samples
- Window step: 10 samples
- FFT size: 1,000
- Retained Doppler range: -60 Hz to 60 Hz

Before calling the function, the receive and transmit antenna dimensions are
flattened from `(Rx, Tx)` to `Rx * Tx`. The resulting DFS tensor has shape:

```text
(time_bins, frequency_bins, subcarriers, Rx_times_Tx, 2)
```

The final dimension stores the real and imaginary components of the complex
STFT result. When converted to `float32` and saved, a DFS sample can be loaded
with:

```python
dfs = np.load("DFS/20200525/zhangyi-1-1-r.npy")
print(dfs.shape, dfs.dtype)
```

The checked-in main loop creates the `DFS/<date>` directories, while the calls
that calculate and save DFS files are commented out in `preprocess_csi.py`.
Uncomment the `get_csi_dfs` call, NumPy conversion, and DFS `np.save` operation
to generate these files alongside `CSI_Processed`. Save each file to
`DFS/<date>/<original-name>.npy`, matching the directory layout shown above.

## Labels and metadata

`metas.csv` records the participant names and the activity indices associated
with fall and non-fall recordings for each date directory:

- `Nonfall_index`: activity indices labeled as non-fall (`0`)
- `Fall_index`: activity indices labeled as fall (`1`)
- `Non`: no recordings of that class in the corresponding directory

The activity index is the second field in a raw filename. For example, the
activity index in `bionic-3-10-r.dat` is `3`.

## MATLAB scripts

The `matlab_scripts` directory contains the original exploratory MATLAB
preprocessing code. It documents the segmentation used in earlier experiments,
including 1.5-second windows and motion-based localization of fall events.
These scripts may require path changes and additional functions or intermediate
data supplied with the original dataset. The Python workflow above is the
recommended entry point for reproducing the CSI conversion included in this
repository.
