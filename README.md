# Minds AI Filter for EEG Documentation

*Your* Minds AI Signal Filter relies on sensor fusion to recognize the physics of true brain signal and, in turn, filter out artifacts and supress noise. “AI” is included in the naming convention due to its part in a larger artificial intelligence
pipeline, but the filter does not require prior training or deep learning.
- License: Polyform Noncommercial 1.0.0 — [Contact MindsApplied](https://www.minds-applied.com/contact) for commercial usage. See [LICENSE](LICENSE).
- Patent status: **Patent pending** (US provisional filed 2025-06-30). See [PATENTS](PATENTS.md).
- Cite this software: see [CITATION](CITATION.cff) and the preprint below.
- Preprint: https://doi.org/10.1101/2025.09.24.675953.
- [Live demo of the filter and below application](https://www.youtube.com/watch?v=YgEt1vKYDc4).

---

## 1) Filter Package Installation

**Package only**
```bash
pip install "git+https://github.com/MindsApplied/Minds_AI_EEG_Filter@main"
```
**Package and App**
```bash
pip install "mindsai-filter-python[examples] @ git+https://github.com/MindsApplied/Minds_AI_EEG_Filter@main""
```

### 1.1 Implementation
After adding the mindsai_filter_python file to your project, and ensuring version compatibility, it can be called using the following:
```python
import mindsai_filter_python as mai
# data: (channels x timepoints) array
data = [
    [5, 6, 5, 4, 3, 2, 3, 4],     # ch0
    [-2, -1, 0, 1, 0, -1, -2, -1] # ch1
]
tailoring_lambda = 1e-25
filtered_data = mai.mindsai_python_filter(data, tailoring_lambda)
```
It's that easy! An intialization key is no longer required.  It expects `data` to be a 2-D continuous array of **channels x time** and relies on one hyperparameter. It should be applied to the data as a whole, prior to other filters or indiviudal electrode analysis. It can be applied to large trials or looped for real-time usage.

### 1.2 Tightening Lambda  

![Tailoring Lambda Description Visual](images/MAI_Filter_Lambda_Funnel_labled.png)

The hyperparameter integer, `tailoring_lambda`, controls how much your Minds AI Filter modifies the original signal and should be input on a logarithmic scale between `0` and `0.1`. A lower `lambda` value like the default `1e-25` causes the filter to make bolder adjustments for more complex transformations that highlight the structure across `channels`, such as for real-time filtering (1 second windows). A higher `lambda` value like `1e-40` works best with more data (such as 60-second trials) for still helpful, but more conservative adjustments.

---

## 2) Demo App Documentation (Not required for filter usage)

The app script used in the above live demo can be found in the examples folder as `Minds_AI_Filter_Real-time_Signal_Analysis.py`. After installing this package and its dependencies, the script can be called via Python in a CLI. It visualizes noise and artifact removal by your Minds AI Filter for a single channel and all channels on average, in real-time. It also provides various SNR metrics, explained more below. It can connect to any headset via Brainflow, using synthetic data by default, and allows you to upload a file.

### 2.1 Optional Configurations
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

# Console “flag” thresholds 
ARTIFACT_SUPPRESSION_THRESH = 20.0  # % peak drop
DRIFT_THRESH_UV             = 5.0   # |mean| or |median| shift (μV)
VARIANCE_SMOOTHING_THRESH   = 5.0   # % variance drop

# Console Signal to Noise ratio variants 
SNR_METHOD = "power_ratio"  # also: "amplitude_ratio" | "variance_ratio"
```

### 2.2 Sampling Rate & Board Selection (BrainFlow vs manual)
```python
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

USE_BRAINFLOW  = True
USE_SYNTHETIC  = False
params         = BrainFlowInputParams()
board_id       = BoardIds.SYNTHETIC_BOARD.value if USE_SYNTHETIC else BoardIds.CYTON_DAISY_BOARD.value

# If using BrainFlow, get fs from the board definition; else set MANUAL_FS
MANUAL_FS = 125
fs = BoardShim.get_sampling_rate(board_id) if USE_BRAINFLOW else MANUAL_FS

# Serial port only required for devices using an input port like usb
if not USE_SYNTHETIC:
    params.serial_port = "COM3"

# Unit scaling required for boards streaming microvolts such via brainflow. Not required for standalone filter. μV→V→mindsai_python_filter→μV→Plot
if USE_BRAINFLOW: 
    UNIT_SCALE_IN = 1e-6 else 1.0 

# Window size in samples
window_size = fs * window_seconds
```
> **Detrending (constant)** removes per‑channel DC offsets (in volts) before filtering so MindsAI isn’t fighting baseline bias.
### 2.3 EEG‑Only Averaging
```python
# raw_plot_uv, filtered_plot_uv: shape = (channels, timepoints) in microvolts
eeg_ch = BoardShim.get_eeg_channels(board_id)  # BrainFlow’s EEG subset (indices into the board’s channel list)

# Average across EEG electrodes only
avg_raw  = raw_plot_uv[eeg_ch, :].mean(axis=0)
avg_filt = filtered_plot_uv[eeg_ch, :].mean(axis=0)
```
> **Small Filtered Average (green):** MAI removes cross-channel-incoherent artifacts; when you average across electrodes, incoherent energy no longer inflates the mean, so the trace appears visually “sparser.”

---

## 3) Noise Models Used in the Demo (for synthetic tests)

We optionally inject simple, interpretable noise terms to stress‑test the filter. Let \(x(t)\) be the clean input and \(y(t)\) the observed (noisy) signal.

**White noise (Gaussian):**
```math
y(t) = x(t) + \epsilon(t), \qquad \epsilon(t) \sim \mathcal{N}(0,\sigma^2)
```

**Power‑line sinusoid (50/60 Hz):**
```math
y(t) = x(t) + A \sin(2\pi f t + \phi), \qquad f \in \{50,\;60\}\ \mathrm{Hz}
```

**Burst / spike segment** of length \(L\) samples, starting at random \(s\):
```math
y(t) = \begin{cases}
x(t) + b(t), & s \le t < s+L \\
x(t), & \text{otherwise}
\end{cases}
\qquad
b(t) \sim \mathcal{N}(\mu,\tau^2)
```

**Flatline segment** (sensor stall) of length \(L\) samples:
```math
y(t) = \begin{cases}
0, & s \le t < s+L \\
x(t), & \text{otherwise}
\end{cases}
```
> In the live demo, synthetic bursts/flatlines are highlighted only when **injected**. Real artifacts are not auto‑tagged.

---

## 4) Mathematical Background

> Notation: \( s \) = filtered signal, \( n = \text{raw} - \text{filtered} \) (removed noise).  

### 4.1 Signal‑to‑Noise Ratio (SNR)

General definition (decibels):
```math
\mathrm{SNR}_{\mathrm{dB}} = 10 \log_{10}\!\Big( \frac{P_{\mathrm{signal}}}{P_{\mathrm{noise}}} \Big)
```

Choices for \( P_{\mathrm{signal}}, P_{\mathrm{noise}} \):

**Power ratio** - Reflects how much mean-square power 'energy' is in the signal vs noise. High amp artifacts will make the SNR worse.
```math
\mathrm{SNR}_{\mathrm{dB}} = 10 \log_{10}\!\Big( \frac{\mathbb{E}[\,s^2\,]}{\mathbb{E}[\,n^2\,]} \Big)
```

**Amplitude ratio** - Contrasts absolute amplitude of the signal and noise. Less spike sensitive than power but more affected by baseline magnitude.
```math
\mathrm{SNR}_{\mathrm{dB}} = 20 \log_{10}\!\Big( \frac{\mathbb{E}[\,|s|\,]}{\mathbb{E}[\,|n|\,]} \Big)
```

**Variance ratio** - Compares the spread of the filtered signal to the residual noise (how much it wiggles around its average). Ignores DC powerline offset but sensitive to big spikes.
```math
\mathrm{SNR}_{\mathrm{dB}} = 10 \log_{10}\!\Big( \frac{\mathrm{Var}(s)}{\mathrm{Var}(n)} \Big)
```

Linearization and “signal power fraction” used in the console:
```math
\mathrm{SNR}_{\mathrm{lin}} = 10^{\mathrm{SNR}_{\mathrm{dB}}/10}
\qquad
\text{Signal Fraction} = \frac{\mathrm{SNR}_{\mathrm{lin}}}{1 + \mathrm{SNR}_{\mathrm{lin}}}
```

### 4.2 Artifact Suppression (peak drop)
```math
\Delta_{\text{peak}} = \max_t |x_{\text{raw}}(t)| - \max_t |x_{\text{filt}}(t)|,
\qquad
\%\Delta_{\text{peak}} = 100 \frac{\Delta_{\text{peak}}}{\max_t |x_{\text{raw}}(t)|}
```

### 4.3 Baseline Drift (mean/median)
```math
\Delta_{\mu} = \mu_{\text{filt}} - \mu_{\text{raw}},
\qquad
\Delta_{\mathrm{med}} = \mathrm{med}_{\text{filt}} - \mathrm{med}_{\text{raw}}
```

### 4.4 Variance Smoothing
```math
\%\Delta\sigma^2 = 100 \frac{\sigma^2_{\text{raw}} - \sigma^2_{\text{filt}}}{\sigma^2_{\text{raw}}}
```

### 4.5 Multi‑Channel vs Single‑Channel
For a window with \( C \) EEG channels and \( T \) time samples, arrays are shape \( (C \times T) \).

* **Single‑channel** metrics operate on one channel \( x \in \mathbb{R}^T \).
* **Average across EEG channels** (exclude AUX/EMG/ACC):
```math
\bar{x}(t) = \frac{1}{|S|}\sum_{c\in S} x_c(t), \qquad t = 1,\dots,T
```
where \( S \) is the EEG‑only index set (e.g., BrainFlow `BoardShim.get_eeg_channels(board_id)`).

---

## 5) Console Output: What Each Field Means

A typical two‑line console block per window is:

```
[Window: 12:34:56.001 → 12:34:57.001] [SNR: 9.19 dB | Signal ~8.3× stronger than noise | ≈89% signal power]  [Peak: 440.0→344.1 μV (↓95.9 μV, 22%) | Artifact Suppression]  [Variance ↓15.6% | Smoothing Effect]
[Baseline Shift: mean -3.1 μV | median -2.8 μV]  [SNR method: amplitude_ratio]
```

**Field mapping to math and config**

* **SNR (dB)** → Section 3.1. The *linear* value is \(10^{\mathrm{SNR}_{\mathrm{dB}}/10}\). The “signal power” fraction is \(\mathrm{SNR}_{\mathrm{lin}}/(1+\mathrm{SNR}_{\mathrm{lin}})\).
  * **Config:** `SNR_METHOD` chooses variance, power, or amplitude version.
* **Peak a→b (↓Δ, %)** → Section 3.2. If `%Δ_peak ≥ ARTIFACT_SUPPRESSION_THRESH`, the tag **Artifact Suppression** appears.
* **Variance ↓%** → Section 3.4. If `%Δσ² ≥ VARIANCE_SMOOTHING_THRESH`, the tag **Smoothing Effect** appears.
* **Baseline Shift (mean/median)** → Section 3.3. If `|Δ| ≥ DRIFT_THRESH_UV` for mean or median, the tag **Drift Correction** appears.
* **Window times** → Start/End of the plotted/metrics window (UTC, millisecond precision).

**How config changes affect the console**

* Lower **λ** (`filterHyperparameter`) → typically larger peak and variance drops (more tags), sometimes larger baseline shifts.
* Raise **thresholds** → fewer tags; lower thresholds → more sensitive.
* Change **SNR_METHOD** → dB and “signal power” readouts may differ across methods.

---

## Patent Status

**Patent pending.** Portions of this work are covered by a US provisional application filed **2025-06-30**.  
For details, see [PATENTS.md](PATENTS.md). For commercial licensing, contact contact@minds-applied.com.

## How to Cite

If you use this software, please cite the preprint:

> Wesierski, J-M., Rodriguez, N. *A lightweight, physics-based, sensor-fusion filter for real-time EEG.* bioRxiv (2025).  
> https://doi.org/10.1101/2025.09.24.675953

A machine-readable citation file is provided: [CITATION.cff](CITATION.cff).

