# MindsAI EEG Filter — Quick Guide + Math

Real-time EEG demo using the **MindsAI** filter with BrainFlow input (or manual sampling rate).  
Includes configurable thresholds, UTC-time plotting, optional bandpass, and human-readable console metrics.

---

## 1) Quick Configuration (what to change first)

```python
# --- License (required) ---
mindsai_filter_python.initialize_mindsai_license('YOUR-LICENSE-KEY')
print(mindsai_filter_python.get_mindsai_license_message())

# --- Core knobs ---
filterHyperparameter = 1e-25   # λ: filter strength (↓ keeps more signal, ↑ removes more noise)
channel_idx = 4                # which EEG channel to spotlight (0-based)
window_seconds = 1             # rolling window length for plots/metrics

# --- Noise toggles (testing only; turn OFF on real boards) ---
ENABLE_NOISE_HIGHLIGHT = True
INJECT_BURST  = False
INJECT_FLAT   = False
INJECT_SINE   = False
INJECT_WHITE  = False

# --- Optional bandpass BEFORE MindsAI ---
ENABLE_BANDPASS = True
BP_LOW_HZ  = 1.0
BP_HIGH_HZ = 40.0
BP_ORDER   = 4
BP_TYPE    = FilterTypes.BUTTERWORTH  # gentler than zero-phase for some boards

# --- Console thresholds (affect the tags you see printed) ---
ARTIFACT_SUPPRESSION_THRESH = 20.0   # % peak drop → tag "Artifact Suppression"
DRIFT_THRESH_UV             = 5.0    # |mean| or |median| shift in µV → tag "Drift Correction"
VARIANCE_SMOOTHING_THRESH   = 5.0    # % variance drop → tag "Smoothing Effect"

# --- SNR method: "variance_ratio" | "power_ratio" | "amplitude_ratio"
SNR_METHOD = "amplitude_ratio"
```

### BrainFlow vs manual sampling rate

```python
USE_BRAINFLOW = True
USE_SYNTHETIC = False
params = BrainFlowInputParams()
board_id = BoardIds.SYNTHETIC_BOARD.value if USE_SYNTHETIC else BoardIds.CYTON_DAISY_BOARD.value

MANUAL_FS = 125  # only used if USE_BRAINFLOW=False

if USE_BRAINFLOW:
    fs = BoardShim.get_sampling_rate(board_id)
else:
    fs = MANUAL_FS

if not USE_SYNTHETIC:
    params.serial_port = "COM3"

window_size = fs * window_seconds
```

### Units (Volts vs µV)

- BrainFlow Cyton/Cyton-Daisy typically provides **µV**; the MindsAI filter expects **Volts**.  
- In the reference code we convert to V before filtering, then back to µV for plotting/metrics:

```python
UNIT_SCALE_IN = 1e-6  # µV → V for real boards; 1.0 if synthetic already in V
raw_window_for_filter = raw_window * UNIT_SCALE_IN
# ... run filter in volts ...
filtered_plot_uv = filtered_volts * 1e6  # back to µV for display
```

### Selecting **only EEG** channels for averaging

BrainFlow returns a channel list for EEG:

```python
eeg_channels = BoardShim.get_eeg_channels(board_id)
avg_raw  = np.mean(raw_plot_uv[eeg_channels, :], axis=0)
avg_filt = np.mean(filtered_plot_uv[eeg_channels, :], axis=0)
```

> Do **not** average over aux/accel/EMG/etc. Use `get_eeg_channels` so your “All-Channel Avg” truly reflects EEG only.

---

## 2) What each configuration changes in practice

- **λ (filterHyperparameter)**  
  *Effect on signal*: larger λ = stronger regularization → more artifact removal, but can damp small genuine features.  
  *Effect on console*: typically **higher SNR**, larger *Peak drop %*, more **Variance ↓%**; drift tag may appear if baseline is corrected.

- **SNR method**  
  - `variance_ratio`: robust for broadband noise.  
  - `power_ratio`: equivalent when signals are zero-mean.  
  - `amplitude_ratio`: more intuitive for spiky artifacts, sometimes less sensitive to mild oscillatory noise.  
  *Effect on console*: SNR dB number and derived “Signal ~× stronger” and “% signal power” will change accordingly.

- **Artifact/Drift/Variance thresholds**  
  - Lower thresholds → tags appear more often (more “sensitive”).  
  - Higher thresholds → tags appear only for strong effects.

- **Bandpass**  
  - Tightening passband (e.g., 1–30 Hz) reduces high-freq noise but can reduce sharp transients (including true spikes).  
  - Increasing order increases steepness; too high may cause ringing.  
  *Effect on console*: often increases **Variance ↓%** and may raise SNR; can also increase **Peak drop %** if spikes lie out of band.

- **Noise toggles**  
  - For synthetic tests only. Turn **all OFF** on real hardware.  
  - Burst/flat apply only to the focused channel; sine/white apply to all channels.

---

## 3) Math (GitHub-rendered)

### 3.1 Signal-to-Noise Ratio (SNR)

```math
\mathrm{SNR}_{\mathrm{dB}} = 10 \log_{10}\!\left(rac{P_{	ext{signal}}}{P_{	ext{noise}}}ight)
```

Choices for \(P_{	ext{signal}}, P_{	ext{noise}}\):

- **Variance ratio**
```math
\mathrm{SNR}_{\mathrm{dB}} = 10 \log_{10}\!\left(rac{\mathrm{Var}[s]}{\mathrm{Var}[n]}ight)
```

- **Power ratio**
```math
\mathrm{SNR}_{\mathrm{dB}} = 10 \log_{10}\!\left(rac{\mathbb{E}[s^2]}{\mathbb{E}[n^2]}ight)
```

- **Amplitude ratio**
```math
\mathrm{SNR}_{\mathrm{dB}} = 20 \log_{10}\!\left(rac{\mathbb{E}[|s|]}{\mathbb{E}[|n|]}ight)
```

We also show linear and “signal power fraction”:

```math
\mathrm{SNR}_{\mathrm{lin}} = 10^{\mathrm{SNR}_{\mathrm{dB}}/10},\qquad
	ext{SignalFraction} = rac{\mathrm{SNR}_{\mathrm{lin}}}{1 + \mathrm{SNR}_{\mathrm{lin}}}.
```

### 3.2 Artifact Suppression (Peaks)

```math
\Delta 	ext{Peak} = \max_t|x_{	ext{raw}}(t)| - \max_t|x_{	ext{filt}}(t)|,
\qquad
\%	ext{Drop} = 100 	imes rac{\Delta	ext{Peak}}{\max_t|x_{	ext{raw}}(t)|}.
```

Tag **Artifact Suppression** iff \(\%	ext{Drop} \ge 	ext{ARTIFACT\_SUPPRESSION\_THRESH}\).

### 3.3 Drift Correction (Mean/Median)

```math
\Delta \mu = \mu_{	ext{filt}} - \mu_{	ext{raw}}, \qquad
\Delta 	ilde{x} = 	ilde{x}_{	ext{filt}} - 	ilde{x}_{	ext{raw}}.
```

Tag **Drift Correction** iff \(|\Delta \mu| \ge 	ext{DRIFT\_THRESH\_UV}\) or \(|\Delta 	ilde{x}| \ge 	ext{DRIFT\_THRESH\_UV}\).

### 3.4 Variance Smoothing

```math
\%	ext{Var}\downarrow = 100 	imes rac{\mathrm{Var}[x_{	ext{raw}}] - \mathrm{Var}[x_{	ext{filt}}]}{\mathrm{Var}[x_{	ext{raw}}]}.
```

Tag **Smoothing Effect** iff \(\%	ext{Var}\downarrow \ge 	ext{VARIANCE\_SMOOTHING\_THRESH}\).

### 3.5 Multi-Channel vs Single-Channel

Single-channel metrics use the focused row \(x_c(t)\).  
A cross-channel average (EEG only) is

```math
ar{x}(t) = rac{1}{|S|} \sum_{c \in S} x_c(t),
```

where \(S\) is the EEG channel set from BrainFlow
(`BoardShim.get_eeg_channels(board_id)`).

---

## 4) Code: metrics + console strings (copy-ready)

```python
snr_db = calculate_snr(filt_ch, diff_ch, method=SNR_METHOD)
impact = calculate_filter_impact(raw_ch, filt_ch)

print(format_metrics_console_inline(
    snr_db, impact, snr_method=SNR_METHOD,
    bp_enabled=ENABLE_BANDPASS, bp_low=BP_LOW_HZ, bp_high=BP_HIGH_HZ,
    artifact_suppression_thresh=ARTIFACT_SUPPRESSION_THRESH,
    drift_thresh_uv=DRIFT_THRESH_UV,
    variance_smoothing_thresh=VARIANCE_SMOOTHING_THRESH,
    window_time_prefix=time_prefix
))
print()  # blank line between windows
```

**Example output**

```
[Window: 12:34:56.001 → 12:34:57.001]
[SNR: 9.2 dB | Signal ~8.3× stronger than noise | ≈89% signal power]
[Peak: 440.0→344.1 μV (↓95.9 μV, 22%) | Artifact Suppression]
[Variance ↓15.6% | Smoothing Effect]
[Baseline Shift: mean -3.2 μV | median -2.8 μV]
[BP=ON 1.0-40.0Hz]
```

Tags appear **only** when thresholds are exceeded.

---

## 5) Practical notes for real devices

- Turn **OFF** all synthetic noise toggles.  
- Ensure unit scaling to **Volts** before sending to the MindsAI filter, then back to **µV** for plots.  
- Detrend (constant) per channel before bandpass/filtering if needed.  
- Use `get_eeg_channels()` for any “All-Channel Avg” or multi-channel metric.  
- Tighten/loosen bandpass conservatively; too-high order can ring or oversuppress.

---

## 6) Where to look in the code (map to functions)

- `inject_noise(...)` — synthetic artifact generator (off for real boards)  
- `apply_optional_bandpass(...)` — BrainFlow bandpass on a copy of the window  
- `calculate_snr(...)` — SNR (three methods)  
- `calculate_filter_impact(...)` — peaks/mean/median/variance metrics  
- `format_metrics_console_inline(...)` — human-readable console lines with threshold tags  
- Main loop — acquisition → (optional) noise → bandpass → MindsAI → metrics/plot

---
