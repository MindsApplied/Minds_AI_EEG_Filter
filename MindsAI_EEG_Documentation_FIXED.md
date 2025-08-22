# MindsAI EEG Real-time Filtering & Metrics Documentation

## 1. Configuration Overview

Before running the script, configure the following items:

- **License**  
  The MindsAI filter requires a valid license key.  
  Example:  
  ```python
  mindsai_filter_python.initialize_mindsai_license('MINDS-TEST-001')
  print(mindsai_filter_python.get_mindsai_license_message())
  ```

- **Filter Hyperparameter (`lambda`)**  
  Controls the strength of the synchronization filter.  
  - Smaller values (e.g., `1e-25`) → more aggressive noise suppression, risk of distorting true signal.  
  - Larger values (e.g., `1e-6`) → gentler filtering, preserves raw features but may leave noise.

- **Channel Index (`channel_idx`)**  
  Sets which EEG channel is displayed in the main plot. (0-based index)

- **Window Length (`window_seconds`)**  
  Defines analysis window size. Larger windows → more stability, slower responsiveness. Smaller windows → more responsive but noisier.

- **Noise Injection Toggles** (for synthetic testing only)  
  ```python
  INJECT_BURST = False   # spikes on single channel
  INJECT_FLAT  = False   # flatline dropouts
  INJECT_SINE  = False   # 60 Hz hum (all channels)
  INJECT_WHITE = False   # Gaussian noise (all channels)
  ```
  > Turn **all off** when connecting to a real board.

- **Bandpass Filter (optional pre-filter)**  
  ```python
  ENABLE_BANDPASS = True
  BP_LOW_HZ  = 1.0
  BP_HIGH_HZ = 40.0
  BP_ORDER   = 4
  BP_TYPE    = FilterTypes.BUTTERWORTH
  ```
  Helps isolate EEG-relevant frequencies before MindsAI filtering.

- **Console Thresholds**  
  ```python
  ARTIFACT_SUPPRESSION_THRESH = 20.0  # % peak drop
  DRIFT_THRESH_UV             = 5.0   # µV mean/median shift
  VARIANCE_SMOOTHING_THRESH   = 5.0   # % variance reduction
  ```

- **SNR Method**  
  Choose: `"variance_ratio"`, `"power_ratio"`, `"amplitude_ratio"`  
  This determines how Signal-to-Noise Ratio is computed in the console.

- **BrainFlow vs Manual Mode**  
  ```python
  USE_BRAINFLOW = True
  USE_SYNTHETIC = False
  ```  
  - If `USE_BRAINFLOW = True`: Sampling rate from board (`BoardShim.get_sampling_rate`).  
  - If `USE_BRAINFLOW = False`: Use `MANUAL_FS` and custom input.

---

## 2. Mathematical Background

### 2.1 Signal-to-Noise Ratio (SNR)

The SNR measures the relative strength of signal vs noise in decibels:

$$ \mathrm{SNR}_{dB} = 10 \cdot \log_{10}\left( \frac{P_{signal}}{P_{noise}} \right) $$

Choices for $P_{signal}, P_{noise}$:

- **Variance ratio**  
  $$ \mathrm{SNR}_{dB} = 10 \cdot \log_{10}\left( \frac{\mathrm{Var}[s]}{\mathrm{Var}[n]} \right) $$

- **Power ratio**  
  $$ \mathrm{SNR}_{dB} = 10 \cdot \log_{10}\left( \frac{\mathbb{E}[s^2]}{\mathbb{E}[n^2]} \right) $$

- **Amplitude ratio**  
  $$ \mathrm{SNR}_{dB} = 20 \cdot \log_{10}\left( \frac{\mathbb{E}[|s|]}{\mathbb{E}[|n|]} \right) $$

Where:
- $s$ = filtered signal  
- $n$ = removed noise (raw – filtered)  

The linear fraction of signal power in the total is:

$$ \text{Signal Fraction} = \frac{\mathrm{SNR}_{lin}}{1 + \mathrm{SNR}_{lin}} $$

---

### 2.2 Artifact Suppression (Peaks)

We quantify peak reduction as:

$$ \Delta \mathrm{Peak} = \max_t |x_{raw}(t)| - \max_t |x_{filt}(t)| $$

Percent drop:

$$ \%\,\mathrm{Drop} = 100 \times \frac{\Delta \mathrm{Peak}}{\max_t |x_{raw}(t)|} $$

If above threshold (default 20%), we tag **Artifact Suppression**.

---

### 2.3 Baseline Drift Correction (Mean/Median)

We check if mean or median amplitude shifts beyond ±5 µV:

$$ \Delta \mathrm{Mean} = \mathrm{Mean}[x_{filt}] - \mathrm{Mean}[x_{raw}] $$

$$ \Delta \mathrm{Median} = \mathrm{Median}[x_{filt}] - \mathrm{Median}[x_{raw}] $$

Crossing threshold triggers **Drift Correction**.

---

### 2.4 Variance Smoothing

We compute reduction in signal variance:

$$ \Delta \mathrm{Var}\% = \frac{\mathrm{Var}[x_{raw}] - \mathrm{Var}[x_{filt}]}{\mathrm{Var}[x_{raw}]} \times 100 $$

Crossing threshold indicates **Smoothing Effect**.

---

### 2.5 Multi-Channel vs Single-Channel

For a window with $C$ EEG channels and $T$ time samples:  
Shape = ($C \times T$).

- **Single-channel metric:** works on vector $x \in \mathbb{R}^T$ (chosen `channel_idx`)  
- **Multi-channel average (EEG only):**

$$ \bar{x}(t) = \frac{1}{|S|} \sum_{c \in S} x_c(t), \quad t = 1, \dots, T $$

where $S$ = EEG channel set (exclude AUX/EMG/accelerometer).  
In code, use BrainFlow’s:

```python
eeg_channels = BoardShim.get_eeg_channels(board_id)
raw_avg = np.mean(raw_window[eeg_channels, :], axis=0)
```

---

## 3. Configuration Effects on Console Output

- **Lambda (filterHyperparameter):**  
  - Smaller → console shows bigger SNR boosts, larger peak suppression.  
  - Larger → console metrics move closer to raw signal, less variance reduction.

- **SNR Method:**  
  - `variance_ratio` → sensitive to variance-based noise.  
  - `power_ratio` → energy-based, good for sinusoidal artifacts.  
  - `amplitude_ratio` → interpretable for spikes or drifts.

- **Thresholds:**  
  - Artifact Suppression = % peak drop.  
  - Drift Correction = µV shift.  
  - Variance Smoothing = % variance reduction.  

Changing thresholds directly affects when tags (`Artifact Suppression`, `Drift Correction`, `Smoothing Effect`) appear in the console.

---

## 4. Example Console Output

```
[Window: 12:34:56.001 → 12:34:57.001]
[SNR: 9.12 dB | Signal ~8.1× stronger than noise | ≈89% signal power]
[Peak: 420.0→310.0 μV (↓110.0 μV, 26%) | Artifact Suppression]
[Variance ↓12.5% | Smoothing Effect] [BP=ON 1.0-40.0Hz]
[Baseline Shift: mean -6.2 μV | median -5.8 μV | Drift Correction] [SNR method: variance_ratio]
```

---

## 5. Best Practices

- Disable **synthetic noise** when using real devices.  
- Use correct **EEG channel indices** (`BoardShim.get_eeg_channels`).  
- Configure thresholds carefully: too tight may flag normal shifts; too loose may miss artifacts.  
- Experiment with **window size**:  
  - Short (1s) = fast updates, noisier.  
  - Long (5–30s) = smoother, more stable metrics.

---
