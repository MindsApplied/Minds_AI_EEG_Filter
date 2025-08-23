# MindsAI EEG Real-Time Filtering & Metrics Documentation

## Quick Start (README-style)

### License
Before using the filter, a valid license key must be initialized:

```python
import mindsai_filter_python
mindsai_filter_python.initialize_mindsai_license('YOUR-LICENSE-KEY')
print(mindsai_filter_python.get_mindsai_license_message())
```

### Configuration Overview
All key settings can be configured at the top of the script:

- **filterHyperparameter (λ)**: Controls the filter’s strength.  
  - Lower values (e.g., `1e-25`) = stronger denoising, risk of over-smoothing.  
  - Higher values = weaker denoising, more raw signal preserved.

- **channel_idx**: Index of EEG channel to display (0-based).

- **window_seconds**: Length of rolling analysis window.

- **Noise Toggles** (for synthetic data only, must be `False` for real boards):  
  - `INJECT_BURST`, `INJECT_FLAT`, `INJECT_SINE`, `INJECT_WHITE`

- **Bandpass filter** (applied before MindsAI filter):  
  - Enable/disable with `ENABLE_BANDPASS`  
  - Default range: 1–40 Hz, 4th order Butterworth

- **Thresholds (console tags)**:  
  - `ARTIFACT_SUPPRESSION_THRESH` (default 20% peak drop)  
  - `DRIFT_THRESH_UV` (default 5 µV)  
  - `VARIANCE_SMOOTHING_THRESH` (default 5% reduction)

- **SNR Method**: `"variance_ratio" | "power_ratio" | "amplitude_ratio"`

- **Board Parameters**:  
  - `USE_BRAINFLOW = True` → sampling rate from BrainFlow (`BoardShim.get_sampling_rate()`)  
  - `USE_BRAINFLOW = False` → manually set `MANUAL_FS`  

---

## Console Metrics & Their Meaning

Each processing window prints contextual metrics with tags:

```
[Window: 12:34:56.001 → 12:34:57.001]
[SNR: 9.19 dB | Signal ~8.3× stronger than noise | ≈89% signal power]  
[Peak: 439.94→344.11 μV (↓95.84 μV, 22%) | Artifact Suppression]  
[Variance ↓15.6% | Smoothing Effect] [BP=ON 1.0-40.0Hz]
[Baseline Shift: mean -87.52 μV | median -89.15 μV | Drift Correction]  
[SNR method: variance_ratio]
```

### Tags & Triggers
- **Artifact Suppression** → Peak reduction ≥ `ARTIFACT_SUPPRESSION_THRESH`  
- **Drift Correction** → |mean/median shift| ≥ `DRIFT_THRESH_UV`  
- **Smoothing Effect** → Variance reduction ≥ `VARIANCE_SMOOTHING_THRESH`  

---

## Math (GitHub-rendered)

### Signal-to-Noise Ratio (SNR)

- **Power Ratio**  
  ```math
  SNR = 10 \cdot \log_{10}\!\left(rac{	ext{mean}(x^2)}{	ext{mean}(n^2)}ight)
  ```

- **Variance Ratio**  
  ```math
  SNR = 10 \cdot \log_{10}\!\left(rac{\mathrm{Var}(x)}{\mathrm{Var}(n)}ight)
  ```

- **Amplitude Ratio**  
  ```math
  SNR = 20 \cdot \log_{10}\!\left(rac{\mathrm{mean}(|x|)}{\mathrm{mean}(|n|)}ight)
  ```

Where `x` = filtered signal, `n` = removed noise.

---

### Artifact Suppression (Peak Drop)
```math
\Delta_{peak} = \max(|x_{raw}|) - \max(|x_{filtered}|)
```

Threshold applied as % drop relative to raw peak.

---

### Baseline Drift (Mean/Median Shift)
```math
\Delta_{\mu} = \mu_{filtered} - \mu_{raw}
```

```math
\Delta_{median} = median(filtered) - median(raw)
```

---

### Variance Smoothing
```math
\Delta_{\sigma^2} = rac{\sigma^2_{raw} - \sigma^2_{filtered}}{\sigma^2_{raw}} 	imes 100\%
```

---

## Effect of Configuration Changes

- **λ (Lambda, filterHyperparameter)**  
  - Lower λ = more aggressive filtering, greater noise removal but risk of attenuating small signals.  
  - Higher λ = lighter filtering, retains more high-frequency noise.

- **Thresholds (artifact, drift, variance)**  
  - Raising thresholds = fewer tags triggered → stricter interpretation.  
  - Lowering thresholds = more frequent tagging → sensitive but possibly noisy console output.

- **SNR method**  
  - `variance_ratio` → emphasizes variability (useful for EEG).  
  - `power_ratio` → interprets energy directly.  
  - `amplitude_ratio` → better for short bursts/artifacts.

---

## Single vs Multi-Channel Behavior

- **Single Channel (demo plots)** → Metrics (SNR, drift, etc.) are computed on that channel.  

- **Multi-Channel (averages)** → Across all EEG electrodes, average is computed:  

```math
x_{avg}(t) = rac{1}{N} \sum_{i=1}^{N} x_i(t)
```

where \(N\) is number of EEG channels (subset defined via `BoardShim.get_eeg_channels(board_id)`).

---

## Full Script (Reference)

The full Python script is included for reference only (see repository).  
**Warning**: Running inside this notebook may attempt to open real hardware connections.  

```python
# Example reference import
from DLL_Real-time_Analysis_Python import main
```

---
