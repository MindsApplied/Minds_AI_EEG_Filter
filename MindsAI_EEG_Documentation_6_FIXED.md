# MindsAI EEG Real‑Time Filtering & Metrics — Documentation (_6)

This document covers configuration, licensing, quick‑start code, math (with GitHub‑rendered LaTeX), and how configuration choices affect the signal and console output. It also includes copy‑pasteable code snippets.

> **Tip:** If you are connecting a real headset, **disable synthetic noise** (set all `INJECT_* = False`).

---

## 1) Quick‑Start / README‑Style Config

### License
```python
import mindsai_filter_python
mindsai_filter_python.initialize_mindsai_license('YOUR-LICENSE-KEY')
print(mindsai_filter_python.get_mindsai_license_message())
```

### Core Config (edit at top of your script)
```python
# Filter strength (regularization). Lower = stronger denoising (risk over‑smooth).
filterHyperparameter = 1e-25   # aka λ (lambda)

# Channel to focus in the main plot (0-based index within EEG set).
channel_idx = 4

# Rolling window length (seconds) used for metrics/plots.
window_seconds = 1

# Synthetic-noise toggles (FOR TESTING ONLY — set False for real boards)
ENABLE_NOISE_HIGHLIGHT = True
INJECT_BURST  = False
INJECT_FLAT   = False
INJECT_SINE   = False
INJECT_WHITE  = False

# Optional bandpass BEFORE MindsAI
ENABLE_BANDPASS = True
BP_LOW_HZ  = 1.0
BP_HIGH_HZ = 40.0
BP_ORDER   = 4
BP_TYPE    = FilterTypes.BUTTERWORTH  # zero-phase = FilterTypes.BUTTERWORTH_ZERO_PHASE

# Console “tag” thresholds (drive the inline labels)
ARTIFACT_SUPPRESSION_THRESH = 20.0  # % peak drop
DRIFT_THRESH_UV             = 5.0   # |mean| or |median| shift (μV)
VARIANCE_SMOOTHING_THRESH   = 5.0   # % variance drop

# Choose SNR variant for console interpretation
SNR_METHOD = "amplitude_ratio"  # also: "variance_ratio" | "power_ratio"
```

### Sampling Rate & Board selection (BrainFlow vs manual)
```python
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

USE_BRAINFLOW  = True
USE_SYNTHETIC  = False
params         = BrainFlowInputParams()
board_id       = BoardIds.SYNTHETIC_BOARD.value if USE_SYNTHETIC else BoardIds.CYTON_DAISY_BOARD.value

# If using BrainFlow, get fs from the board definition; else set MANUAL_FS
MANUAL_FS = 125
fs = BoardShim.get_sampling_rate(board_id) if USE_BRAINFLOW else MANUAL_FS

# Serial port only required for physical OpenBCI boards
if not USE_SYNTHETIC:
    params.serial_port = "COM3"

# Unit scaling into the MindsAI filter (volts in, volts out)
UNIT_SCALE_IN = 1e-6 if not USE_SYNTHETIC else 1.0  # μV→V for real boards; synthetic already V

# Window size in samples
window_size = fs * window_seconds
```

### EEG‑only averaging (exclude AUX/EMG/ACC)
```python
# raw_plot_uv, filtered_plot_uv: shape = (channels, timepoints) in microvolts
eeg_ch = BoardShim.get_eeg_channels(board_id)  # BrainFlow’s EEG subset (indices into the board’s channel list)

# Average across EEG electrodes only
avg_raw  = raw_plot_uv[eeg_ch, :].mean(axis=0)
avg_filt = filtered_plot_uv[eeg_ch, :].mean(axis=0)
```

### UTC axis tick labels (millisecond precision)
```python
from matplotlib import dates as mdates
from matplotlib.ticker import FuncFormatter
import datetime as dt

ax.xaxis.set_major_locator(mdates.AutoDateLocator())

def utc_ms_formatter(x, pos=None):
    d = mdates.num2date(x, tz=dt.timezone.utc)
    return d.strftime('%H:%M:%S.%f')[:-3]  # HH:MM:SS.mmm

ax.xaxis.set_major_formatter(FuncFormatter(utc_ms_formatter))
```

---

## 2) Mathematical Background (GitHub‑rendered)

> Notation: `s` = filtered signal, `n` = removed noise (`raw - filtered`).  
> We avoid `\left`/`\right` pairs to prevent renderer errors and keep formulas robust on GitHub.

### 2.1 Signal‑to‑Noise Ratio (SNR)

General definition (decibels):
```math
\mathrm{SNR}_{\mathrm{dB}} = 10 \,\log_{10}\!\Big( \frac{P_{\mathrm{signal}}}{P_{\mathrm{noise}}} \Big)
```

Choices for \( P_{\mathrm{signal}}, P_{\mathrm{noise}} \):

**Variance ratio**
```math
\mathrm{SNR}_{\mathrm{dB}} = 10 \,\log_{10}\!\Big( \frac{\mathrm{Var}(s)}{\mathrm{Var}(n)} \Big)
```

**Power ratio**
```math
\mathrm{SNR}_{\mathrm{dB}} = 10 \,\log_{10}\!\Big( \frac{\mathbb{E}[\,s^2\,]}{\mathbb{E}[\,n^2\,]} \Big)
```

**Amplitude ratio**
```math
\mathrm{SNR}_{\mathrm{dB}} = 20 \,\log_{10}\!\Big( \frac{\mathbb{E}[\,|s|\,]}{\mathbb{E}[\,|n|\,]} \Big)
```

Convert to linear and “signal power fraction” used in the console:
```math
\mathrm{SNR}_{\mathrm{lin}} = 10^{\mathrm{SNR}_{\mathrm{dB}}/10}
\qquad
\text{Signal Fraction} = \frac{\mathrm{SNR}_{\mathrm{lin}}}{1+\mathrm{SNR}_{\mathrm{lin}}}
```

### 2.2 Artifact Suppression (Peak drop)
```math
\Delta_{\text{peak}} = \max_t |x_{\text{raw}}(t)| \;-\; \max_t |x_{\text{filt}}(t)|
\qquad
\%\Delta_{\text{peak}} = 100 \cdot \frac{\Delta_{\text{peak}}}{\max_t |x_{\text{raw}}(t)|}
```

### 2.3 Baseline Drift (Mean/Median)
```math
\Delta_{\mu} = \mu_{\text{filt}} - \mu_{\text{raw}}
\qquad
\Delta_{\mathrm{med}} = \mathrm{med}_{\text{filt}} - \mathrm{med}_{\text{raw}}
```

### 2.4 Variance Smoothing
```math
\%\Delta\sigma^2 = 100 \cdot \frac{\sigma^2_{\text{raw}} - \sigma^2_{\text{filt}}}{\sigma^2_{\text{raw}}}
```

### 2.5 Multi‑Channel vs Single‑Channel
For a window with \(C\) EEG channels and \(T\) time samples, arrays are shape \((C \times T)\).

* **Single‑channel** metric uses one row (the focused channel).  
* **Average across EEG channels** (exclude AUX/EMG/ACC):
```math
\bar{x}(t) = \frac{1}{|S|}\sum_{c\in S} x_c(t),
\qquad t=1,\dots,T
```
where \(S\) is the EEG‑only index set (e.g., BrainFlow `BoardShim.get_eeg_channels(board_id)`).

---

## 3) How Configuration Choices Affect Signal & Console

* **λ (lambda, `filterHyperparameter`)**  
  * Lower λ → stronger denoising (larger peak/variance drops), potentially more baseline pull and alpha/beta attenuation.  
  * Higher λ → lighter touch; preserves subtle activity but leaves more noise.

* **Thresholds** (drive inline tags in the console)  
  * `ARTIFACT_SUPPRESSION_THRESH` ↑ → fewer “Artifact Suppression” tags; ↓ → more sensitive.  
  * `DRIFT_THRESH_UV` ↑ → fewer “Drift Correction” tags; ↓ → will flag smaller offsets.  
  * `VARIANCE_SMOOTHING_THRESH` ↑ → only large smoothing gets tagged; ↓ → sensitive to minor variance drops.

* **SNR method**  
  * `variance_ratio` is robust for EEG variance shifts.  
  * `power_ratio` interprets energy directly; similar to variance for zero‑mean.  
  * `amplitude_ratio` can be more responsive to bursty artifacts.

* **Bandpass**  
  * ON (e.g., 1–40 Hz) reduces slow drift/high‑frequency noise before MindsAI — often improves SNR and variance drops.  
  * OFF keeps raw spectral content for comparison.

* **Window length (`window_seconds`)**  
  * Short windows react faster (more jittery).  
  * Longer windows stabilize metrics but respond more slowly.

---

## 4) Copy‑Paste Code Snippets

### 4.1 Formatting the console (always show SNR + thresholded tags)
```python
def format_metrics_console_inline(
    snr_db, impact, snr_method="variance_ratio",
    bp_enabled=False, bp_low=1.0, bp_high=40.0,
    artifact_suppression_thresh=20.0,
    drift_thresh_uv=5.0,
    variance_smoothing_thresh=5.0,
    window_time_prefix=""
):
    import numpy as np

    # SNR interpretation
    if np.isinf(snr_db):
        snr_text = "∞ dB (noise≈0)"
        ratio_text = "Signal ≫ noise"
        sig_pct_text = "≈100% signal power"
    else:
        lin = 10 ** (snr_db / 10.0)
        sig_frac = lin / (1.0 + lin)
        snr_text = f"{snr_db:.2f} dB"
        ratio_text = f"Signal ~{lin:.1f}× stronger than noise"
        sig_pct_text = f"≈{sig_frac*100:.0f}% signal power"

    # Core metrics
    peak_before = impact['peak_before']
    peak_after  = impact['peak_after']
    peak_drop   = impact['peak_reduction']
    peak_pct    = (peak_drop / peak_before * 100.0) if peak_before > 0 else 0.0

    mean_shift   = impact['mean_shift']
    median_shift = impact['median_shift']
    var_drop_pct = impact['artifact_variance_reduction_pct']

    # Inline tags
    peak_tag  = " | Artifact Suppression" if peak_pct >= artifact_suppression_thresh else ""
    drift_tag = " | Drift Correction" if (abs(mean_shift) >= drift_thresh_uv or abs(median_shift) >= drift_thresh_uv) else ""
    var_tag   = " | Smoothing Effect" if var_drop_pct >= variance_smoothing_thresh else ""

    bp_text = f"[BP={'ON' if bp_enabled else 'OFF'} {bp_low}-{bp_high}Hz]"

    line1 = (f"{window_time_prefix}"
             f"[SNR: {snr_text} | {ratio_text} | {sig_pct_text}]  "
             f"[Peak: {peak_before:.2f}→{peak_after:.2f} μV (↓{peak_drop:.2f} μV, {peak_pct:.0f}%){peak_tag}]  "
             f"[Variance ↓{var_drop_pct:.1f}%{var_tag}]  {bp_text}")

    line2 = (f"[Baseline Shift: mean {mean_shift:+.2f} μV | median {median_shift:+.2f} μV{drift_tag}]  "
             f"[SNR method: {snr_method}]")

    return line1 + "\n" + line2
```

### 4.2 Apply bandpass safely before MindsAI
```python
from brainflow.data_filter import DataFilter

def apply_optional_bandpass(window_ch_by_time, fs, enable=True,
                            low_hz=1.0, high_hz=40.0, order=4, ftype=FilterTypes.BUTTERWORTH):
    if not enable:
        return window_ch_by_time
    arr = window_ch_by_time.copy()
    for ch_idx in range(arr.shape[0]):
        DataFilter.perform_bandpass(arr[ch_idx], fs, low_hz, high_hz, order, ftype, 0)
    return arr
```

### 4.3 Detrend (constant) in volts prior to filtering
```python
from brainflow.data_filter import DetrendOperations
for ch in range(raw_window_for_filter.shape[0]):
    DataFilter.detrend(raw_window_for_filter[ch], DetrendOperations.CONSTANT.value)
```

---

## 5) Full Script (Reference Pointer)

Keep the full acquisition script in your repository (e.g., `DLL_Real-time_Analysis_Python.py`) and run it directly — avoid launching hardware from notebooks. The notebook or docs can import helper functions only.

```python
# Example (do not auto-run hardware in docs/notebooks)
# from DLL_Real-time_Analysis_Python import main  # keep commented in docs
```

---

## 6) Practical Notes

* **Real headset:** turn all `INJECT_*` to `False`. The highlight overlays in the demo mark *synthetic* events only. (Automatic real-artifact detection is outside the scope of this doc.)
* **Units:** convert μV→V before calling the MindsAI filter (`UNIT_SCALE_IN`), then convert back to μV for metrics/plots.
* **EEG‑only averaging:** Always slice with `BoardShim.get_eeg_channels(board_id)` to exclude non‑EEG inputs.
* **Versioning:** Save this file as `MindsAI_EEG_Documentation_6.md` (or `_6_FIXED.md`) in your repo.
