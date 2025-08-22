# MindsAI EEG Real-time Filtering Documentation

## 1. Quick Start

- **License Init**
```python
import mindsai_filter_python
mindsai_filter_python.initialize_mindsai_license('MINDS-TEST-001')
print(mindsai_filter_python.get_mindsai_license_message())
```

- **Configurable Parameters**
```python
filterHyperparameter = 1e-25
channel_idx = 4
window_seconds = 1

ENABLE_NOISE_HIGHLIGHT = True
INJECT_BURST = False
INJECT_FLAT  = False
INJECT_SINE  = False
INJECT_WHITE = False

ENABLE_BANDPASS = True
BP_LOW_HZ       = 1.0
BP_HIGH_HZ      = 40.0
BP_ORDER        = 4

ARTIFACT_SUPPRESSION_THRESH = 20.0
DRIFT_THRESH_UV             = 5.0
VARIANCE_SMOOTHING_THRESH   = 5.0
SNR_METHOD = "amplitude_ratio"
```

---

## 2. Mathematical Background

### 2.1 Signal-to-Noise Ratio (SNR)

The SNR measures the relative strength of signal vs noise in decibels:

```math
\mathrm{SNR}_{dB} = 10 \cdot \log_{10}\left(rac{P_{signal}}{P_{noise}}ight)
```

Choices for $P_{signal}, P_{noise}$:

- **Variance ratio**  
```math
\mathrm{SNR}_{dB} = 10 \cdot \log_{10}\Big(rac{\mathrm{Var}[s]}{\mathrm{Var}[n]}\Big)
```

- **Power ratio**  
```math
\mathrm{SNR}_{dB} = 10 \cdot \log_{10}\Big(rac{\mathbb{E}[s^2]}{\mathbb{E}[n^2]}\Big)
```

- **Amplitude ratio**  
```math
\mathrm{SNR}_{dB} = 20 \cdot \log_{10}\Big(rac{\mathbb{E}[|s|]}{\mathbb{E}[|n|]}\Big)
```

Where:

- $s =$ filtered signal  
- $n =$ removed noise (raw – filtered)  

The linear fraction of signal power in the total is:

```math
	ext{Signal Fraction} = rac{\mathrm{SNR}_{lin}}{1 + \mathrm{SNR}_{lin}}
```

---

### 2.2 Artifact Suppression (Peaks)

Peak reduction per window:

```math
\Delta Peak = \max_t |x_{raw}(t)| - \max_t |x_{filt}(t)|
```

Percent drop:

```math
\%	ext{Drop} = 100 	imes rac{\Delta Peak}{\max_t |x_{raw}(t)|}
```

Crossing ≥ 20% (configurable) → tag as **Artifact Suppression**.

---

### 2.3 Baseline Drift Correction (Mean/Median)

Shifts in mean or median amplitude suggest drift:

```math
\Delta Mean = Mean(filtered) - Mean(raw)
```

```math
\Delta Median = Median(filtered) - Median(raw)
```

Crossing ±5 μV indicates **Drift Correction**.

---

### 2.4 Variance Smoothing

Variance reduction is computed as:

```math
\Delta Var\% = rac{Var(raw) - Var(filtered)}{Var(raw)} 	imes 100
```

Crossing ≥ 5% (configurable) → tag as **Smoothing Effect**.

---

### 2.5 Multi-Channel vs Single-Channel

For a window with (C) EEG channels and (T) time samples, arrays are shape (C × T).  

- **Single-channel metric** operates on one row $x \in \mathbb{R}^T$.  
- **Average across EEG channels** (excluding AUX/EMG):

```math
ar{x}(t) = rac{1}{|S|}\sum_{c \in S} x_c(t), \quad t = 1, ..., T
```

Where (S) is the EEG-only set (e.g., BrainFlow `BoardShim.get_eeg_channels(board_id)`).

---

## 3. How Configurations Affect Output

- **SNR Method** (`variance_ratio`, `power_ratio`, `amplitude_ratio`) changes the scale of the reported SNR and how “signal vs noise” is defined.
- **Artifact Suppression Threshold**: Lower threshold → more frequent tagging of suppression events. Higher threshold → only strong peak drops trigger.
- **Drift Threshold**: Smaller values → console more sensitive to slow baseline shifts. Larger values → ignores small baseline drift.
- **Variance Threshold**: Lower → tags smoothing frequently. Higher → only tags strong variance reduction.
- **Lambda (filterHyperparameter)**: Controls strength of the MindsAI filter. Very small → aggressive noise reduction, risk of amplitude shrink. Larger → more conservative, preserving raw fluctuations.

Console summaries always display SNR. Other tags (suppression, drift correction, smoothing) appear **inline next to their metric** when thresholds are crossed.

---

## 4. Code References

- **License init & config** → see section 1.  
- **Noise injection toggles** → controlled by `INJECT_*` switches.  
- **Bandpass filter** → applied via BrainFlow before MindsAI.  
- **Metrics** → functions `calculate_snr()` and `calculate_filter_impact()`.  
- **Console formatting** → `format_metrics_console_inline()`.  
- **Plotting** → UTC time on x-axis, EEG amplitudes (µV) on y-axis.

---

## 5. Notes for Real Devices

- If using **synthetic data**, noise can be injected (`INJECT_* = True`).  
- If using a **real EEG device**, disable synthetic noise injection.  
- BrainFlow automatically provides EEG channel indices (use these to exclude AUX/EMG channels for averages).  
- Sampling rate is either:  
  - From BrainFlow (`BoardShim.get_sampling_rate(board_id)`)  
  - Manual (`MANUAL_FS`) if using an unsupported device.

---
