"""Real EEG data loaders.

Currently implements:

* ``physionet_eegbci`` – downloads + parses subject runs from the
  PhysioNet EEG Motor Movement/Imagery Database (eegmmidb v1.0.0).
  Source: https://physionet.org/content/eegmmidb/1.0.0/
  64 channels, 160 Hz, BCI2000 annotations (T0/T1/T2).
"""

from .physionet_eegbci import (
    DATASET_FS,
    DATASET_N_CHANNELS,
    download_run,
    load_run,
    run_to_windows,
    PHYSIONET_BASE,
)

__all__ = [
    "DATASET_FS",
    "DATASET_N_CHANNELS",
    "download_run",
    "load_run",
    "run_to_windows",
    "PHYSIONET_BASE",
]
