# MindsAI EEG Real-time Analysis Documentation

This document provides both a **quick-start configuration guide** and an **in-depth explanation** of the methods, metrics, and math used in the MindsAI EEG real-time filtering script.

---

## 1. Quick Start (README-style)

### License Initialization
```python
import mindsai_filter_python
mindsai_filter_python.initialize_mindsai_license('YOUR-LICENSE-KEY')
print(mindsai_filter_python.get_mindsai_license_message())
```

### Configurable Parameters
- **filterHyperparameter**: Strength of the MindsAI filter.
- **channel_idx**: EEG channel index to plot.
- **window_seconds**: Length of rolling window (e.g., 1s, 30s).
- **Noise injection toggles**: `INJECT_BURST`, `INJECT_FLAT`, `INJECT_SINE`, `INJECT_WHITE` (for testing only, turn off for real devices).
- **Bandpass filter**: Enable with `ENABLE_BANDPASS` and configure `(BP_LOW_HZ, BP_HIGH_HZ, BP_ORDER)`.
- **SNR Method**: `"variance_ratio"`, `"power_ratio"`, `"amplitude_ratio"`.
- **Console thresholds**: Artifact suppression, baseline drift correction, variance smoothing.

### Device Connection
```python
USE_BRAINFLOW = True
USE_SYNTHETIC = False
board_id = BoardIds.CYTON_DAISY_BOARD.value

if USE_BRAINFLOW:
    fs = BoardShim.get_sampling_rate(board_id)
else:
    fs = MANUAL_FS
```

---

## 2. Mathematical Background

### Signal-to-Noise Ratio (SNR)
The SNR measures the relative strength of signal vs noise.

- **Variance ratio**:
$$
SNR_{dB} = 10 \cdot \log_{10} \left( \frac{Var(signal)}{Var(noise)} \right)
$$

- **Power ratio**:
$$
SNR_{dB} = 10 \cdot \log_{10} \left( \frac{E[signal^2]}{E[noise^2]} \right)
$$

- **Amplitude ratio**:
$$
SNR_{dB} = 20 \cdot \log_{10} \left( \frac{E[|signal|]}{E[|noise|]} \right)
$$

---

### Artifact Suppression
We measure peak reduction as:
$$
\Delta Peak = Peak_{before} - Peak_{after}
$$

Threshold crossing (e.g., 20%) indicates **Artifact Suppression**.

---

### Baseline Drift Correction
Shifts in mean or median amplitude suggest baseline drift:
$$
\Delta Mean = Mean(filtered) - Mean(raw)
$$
$$
\Delta Median = Median(filtered) - Median(raw)
$$

Crossing ±5 µV indicates **Drift Correction**.

---

### Variance Smoothing
Variance reduction is computed as:
$$
\Delta Var\% = \frac{Var(raw) - Var(filtered)}{Var(raw)} \times 100
$$

Above 5% suggests **Smoothing Effect**.

---

## 3. Multi-Channel vs Single-Channel

- **Single-channel metrics**: Applied directly to `channel_idx` (e.g., SNR, peaks).
- **Multi-channel averaging**: Only EEG electrodes (`BoardShim.get_eeg_channels(board_id)`) are averaged, avoiding non-EEG channels.

---

## 4. Usage Notes
- Turn off synthetic noise when using **real devices**.
- Bandpass filtering can be enabled before MindsAI for comparison.
- Adjust `window_seconds` for responsiveness vs stability:
  - Short windows → faster updates, more noise.
  - Long windows → smoother results, less responsive.

---

## 5. Example Console Output

```
[Window: 12:00:00.000 → 12:00:01.000] [SNR: 9.19 dB | Signal ~8.3× stronger than noise | ≈89% signal power]
[Peak: 439.94→344.11 μV (↓95.84 μV, 22%) | Artifact Suppression] [Variance ↓15.6% | Smoothing Effect]
[Baseline Shift: mean -2.1 μV | median -1.8 μV | Drift Correction] [SNR method: variance_ratio]
```

---

## 6. References
- BrainFlow: https://brainflow.org
- MindsAI Filter: Proprietary sensor-fusion based EEG noise suppression
- Signal processing basics: SNR, variance, detrending, and bandpass filtering
