# MindsAI EEG Real-Time Filtering & Analysis

This document explains how to configure and use the MindsAI real-time EEG filtering pipeline.
It covers:
- Configuration options (license, thresholds, noise injection)
- Console metrics and interpretation
- Mathematical background (with LaTeX math that renders correctly on GitHub)
- Code references to help you modify or extend the pipeline

---

## 1. Configuration Overview

### License
Before running, initialize with your MindsAI license key:
```python
mindsai_filter_python.initialize_mindsai_license('YOUR-KEY-HERE')
print(mindsai_filter_python.get_mindsai_license_message())
```

### Hyperparameters & Windows
- `filterHyperparameter`: Controls sensitivity of MindsAI filter (default `1e-25`)
- `channel_idx`: Which EEG channel to plot (0-based index, usually `0–15` for OpenBCI)
- `window_seconds`: How long each processing window lasts (e.g., 1 second or 5 seconds)

### Noise Injection (for synthetic testing)
```python
INJECT_BURST = False   # affects only channel_idx
INJECT_FLAT  = False   # affects only channel_idx
INJECT_SINE  = False   # affects all channels
INJECT_WHITE = False   # affects all channels
```
> **Important**: If connecting to a real device, turn all `INJECT_*` to `False`.

### Bandpass Filtering
```python
ENABLE_BANDPASS = True
BP_LOW_HZ  = 1.0
BP_HIGH_HZ = 40.0
BP_ORDER   = 4
BP_TYPE    = FilterTypes.BUTTERWORTH
```

### Thresholds (for console display tags)
- `ARTIFACT_SUPPRESSION_THRESH = 20.0` (% peak drop)
- `DRIFT_THRESH_UV = 5.0` (µV baseline shift)
- `VARIANCE_SMOOTHING_THRESH = 5.0` (% variance drop)

### SNR Method
Choose one:
- `"variance_ratio"` (default)
- `"power_ratio"`
- `"amplitude_ratio"`

### BrainFlow / Manual Sampling Rate
```python
USE_BRAINFLOW = True
USE_SYNTHETIC = False
if USE_BRAINFLOW:
    fs = BoardShim.get_sampling_rate(board_id)
else:
    fs = MANUAL_FS
```

---

## 2. Mathematical Background

### 2.1 Signal-to-Noise Ratio (SNR)

The SNR measures the relative strength of signal vs noise in decibels:

$$
\mathrm{SNR}_{dB} = 10 \cdot \log_{10}\left( \frac{P_{signal}}{P_{noise}} \right)
$$

Choices for $P_{signal}, P_{noise}$:

- **Variance ratio**  
  $$
  \mathrm{SNR}_{dB} = 10 \cdot \log_{10}\left( \frac{\mathrm{Var}[s]}{\mathrm{Var}[n]} \right)
  $$

- **Power ratio**  
  $$
  \mathrm{SNR}_{dB} = 10 \cdot \log_{10}\left( \frac{\mathbb{E}[s^2]}{\mathbb{E}[n^2]} \right)
  $$

- **Amplitude ratio**  
  $$
  \mathrm{SNR}_{dB} = 20 \cdot \log_{10}\left( \frac{\mathbb{E}[|s|]}{\mathbb{E}[|n|]} \right)
  $$

Where:
- $s =$ filtered signal  
- $n =$ removed noise (raw – filtered)

The linear fraction of signal power in the total is:

$$
\text{Signal Fraction} = \frac{\mathrm{SNR}_{lin}}{1 + \mathrm{SNR}_{lin}}
$$

---

### 2.2 Artifact Suppression (Peaks)

Peak reduction per window:

$$
\Delta Peak = \max_t |x_{raw}(t)| - \max_t |x_{filt}(t)|
$$

Percent drop:

$$
\%Drop = 100 \times \frac{\Delta Peak}{\max_t |x_{raw}(t)|}
$$

Crossing ≥ 20% (configurable) → tag as **Artifact Suppression**.

---

### 2.3 Baseline Drift Correction (Mean/Median)

Shift in baseline:

$$
\Delta Mean = Mean(filt) - Mean(raw)
$$

$$
\Delta Median = Median(filt) - Median(raw)
$$

Crossing ±5 µV → tag as **Drift Correction**.

---

### 2.4 Variance Smoothing

Variance reduction:

$$
\Delta Var\% = \frac{Var(raw) - Var(filt)}{Var(raw)} \times 100
$$

Crossing ≥ 5% → tag as **Smoothing Effect**.

---

### 2.5 Multi-Channel vs Single-Channel

For a window with $C$ EEG channels and $T$ samples: shape = $(C \times T)$.

- **Single-channel metric** operates on one vector $x_c(t)$.
- **Cross-channel average** (EEG-only) is:

$$
\bar{x}(t) = \frac{1}{|S|} \sum_{c \in S} x_c(t), \quad t = 1, ..., T
$$

Where $S$ is the EEG channel set (exclude AUX/EMG/Accel).  
In code, slice only EEG rows before averaging:

```python
eeg_channels = BoardShim.get_eeg_channels(board_id)
avg_raw  = np.mean(raw_window[eeg_channels, :], axis=0)
avg_filt = np.mean(filtered[eeg_channels, :], axis=0)
```

---

## 3. How Configurations Affect Results

- **SNR Method**: Changes the reference metric (variance, power, amplitude). Console description adapts.  
- **Artifact Suppression Threshold**: Lower = more sensitive, higher = stricter. Affects when `[Artifact Suppression]` tag prints.  
- **Drift Threshold (µV)**: Too low = normal fluctuations tagged; too high = real drifts ignored.  
- **Variance Threshold**: Controls when `[Smoothing Effect]` tag prints.  
- **Lambda (filterHyperparameter)**: Lower → less aggressive (keeps noise), higher → stronger suppression (may attenuate signal).  

Console display reflects these choices in real-time with contextual tags.

---

## 4. Code References

- **Noise injection**: `inject_noise()`  
- **Metrics**: `calculate_snr()`, `calculate_filter_impact()`  
- **Console formatting**: `format_metrics_console_inline()`  
- **Bandpass prefilter**: `apply_optional_bandpass()`  
- **Plot setup**: `create_dark_plot()`  

---
