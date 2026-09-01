# sEMG Signal Processing & Activity Classification

Python/Jupyter pipeline for filtering, feature extraction, and classification of surface EMG (sEMG) signals recorded from the forearm.

## Overview

This notebook processes raw two-channel EMG data (forearm and back-of-forearm) recorded via LabChart, and trains classifiers to recognize different muscle activities from the signal.

1. **Load data** — reads raw LabChart recording (`lab_data.adicht`) and extracts two EMG channels.
2. **Filter (Part A)** — applies a 1st-order Butterworth lowpass filter (1 Hz cutoff, zero-phase via `filtfilt`) to both channels; compares power spectral density before/after filtering using Welch's method.
3. **Preprocess (Part B)** — combines both filtered channels, splits into train/test sets (80/20), and normalizes with `StandardScaler`.
4. **Window & label (Part C)** — segments the signal into overlapping windows (~0.21 s length, ~0.004 s step at 320 Hz) and labels each window into one of 5 activity classes based on time bins.
5. **Extract features (Part D)** — computes standard EMG time-domain features per window: MAV, STD, VAR, and IAV.
6. **Classify (Part E)** — trains and evaluates K-Nearest Neighbors and Linear Discriminant Analysis classifiers, reporting confusion matrices and accuracy.

## Requirements

- Python 3
- `adi` (LabChart file reader)
- `numpy`, `pandas`, `scipy`, `matplotlib`
- `scikit-learn`
