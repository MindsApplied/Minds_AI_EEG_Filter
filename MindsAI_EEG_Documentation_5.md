# MindsAI EEG Documentation (v5)

## 1. Configuration and Setup

### License Initialization
Before using the MindsAI Filter, initialize your license key:
```python
mindsai_filter_python.initialize_mindsai_license('YOUR-LICENSE-KEY')
print(mindsai_filter_python.get_mindsai_license_message())
```

### Configurable Parameters
- **filterHyperparameter (λ):** Controls strength of synchrony enforcement. Smaller = gentler filtering; larger = stronger artifact removal.  
- **window_seconds:** Size of rolling buffer window. Larger windows = smoother, slower response.  
- **Noise toggles (INJECT_BURST, FLAT, etc.):** Use only with synthetic data, disable with real devices.  
- **Bandpass filter:** Optional pre-filter. Adjust `BP_LOW_HZ`, `BP_HIGH_HZ`, and order depending on device specs.  
- **Thresholds (console tags):**  
  - Artifact suppression (% peak drop)  
  - Drift correction (µV mean/median shift)  
  - Variance smoothing (% variance reduction)

### BrainFlow vs Manual Sampling Rate
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

### 2.1 Signal-to-Noise Ratio (SNR)

The SNR measures the relative strength of signal vs noise in decibels:

$$
\mathrm{SNR}_{dB} = 10 \cdot \log_{10}\left(\frac{P_{signal}}{P_{noise}} \right)
$$

Choices for $P_{signal}, P_{noise}$:

- **Variance ratio**
  $$
  SNR_{dB} = 10 \cdot \log_{10}\left( \frac{\mathrm{Var}[s]}{\mathrm{Var}[n]} \right)
  $$

- **Power ratio**
  $$
  SNR_{dB} = 10 \cdot \log_{10}\left( \frac{\mathbb{E}[s^2]}{\mathbb{E}[n^2]} \right)
  $$

- **Amplitude ratio**
  $$
  SNR_{dB} = 20 \cdot \log_{10}\left( \frac{\mathbb{E}[|s|]}{\mathbb{E}[|n|]} \right)
  $$

Where:
- $s$ = filtered signal  
- $n$ = removed noise (raw – filtered)

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

We measure baseline drift as:

$$
\Delta Mean = Mean(filtered) - Mean(raw)
$$

$$
\Delta Median = Median(filtered) - Median(raw)
$$

Crossing ±5 µV (configurable) → tag as **Drift Correction**.

---

### 2.4 Variance Smoothing

Variance reduction across the window:

$$
\Delta Var\% = \frac{Var(raw) - Var(filtered)}{Var(raw)} \times 100
$$

Crossing ≥ 5% (configurable) → tag as **Smoothing Effect**.

---

### 2.5 Multi-Channel vs Single-Channel

For a window with (C) EEG channels and (T) time samples, arrays are shape $(C \times T)$.

- **Single-channel metric** operates on one row $x \in \mathbb{R}^T$.  
- **Average across EEG channels** (excluding AUX/EMG):

$$
\bar{x}(t) = \frac{1}{|S|} \sum_{c \in S} x_c(t), \quad t=1,\ldots,T
$$

Where (S) is the EEG-only set (e.g., BrainFlow `BoardShim.get_eeg_channels(board_id)`).

---

## 3. Console Display Behavior

- **SNR tags** update live depending on chosen method (variance, power, amplitude).  
- **Artifact suppression tag** triggers if % peak drop exceeds threshold.  
- **Drift correction tag** triggers if mean/median shift exceeds µV threshold.  
- **Smoothing tag** triggers if variance reduction exceeds threshold.  

Adjusting thresholds makes the console more/less sensitive.  
Changing λ (filterHyperparameter) alters balance between artifact removal and signal fidelity.

---

## 4. Practical Notes

- Disable synthetic noise injection when using a **real device**.  
- Averaging should be applied **only across EEG channels**, not auxiliary sensors.  
- Use BrainFlow sampling rate unless working with unsupported hardware.  
- λ too small → minimal effect. λ too large → risk of amplitude distortion.  
