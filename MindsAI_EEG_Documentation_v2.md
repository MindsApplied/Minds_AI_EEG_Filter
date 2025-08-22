# MindsAI EEG Filter Documentation (v2)

## 1. Configurable Parameters

The MindsAI filter requires a **valid license key** for initialization.  
Users can configure parameters for filtering and console metrics:

```python
filterHyperparameter = 1e-25   # Lambda value controlling filter strength
ARTIFACT_SUPPRESSION_THRESH = 20.0   # % peak drop
DRIFT_THRESH_UV             = 5.0    # µV shift
VARIANCE_SMOOTHING_THRESH   = 5.0    # % variance drop
SNR_METHOD = "variance_ratio"        # or "power_ratio", "amplitude_ratio"
```

---

## 2. Mathematical Background

### 2.1 Signal-to-Noise Ratio (SNR)

The SNR measures the relative strength of signal vs noise in decibels:

```math
SNR_{dB} = 10 \cdot \log_{10}\left(rac{P_{signal}}{P_{noise}}ight)
```

Choices for $P_{signal}, P_{noise}$:

- **Variance ratio**  
  ```math
  SNR_{dB} = 10 \cdot \log_{10}\left(rac{Var[s]}{Var[n]}ight)
  ```

- **Power ratio**  
  ```math
  SNR_{dB} = 10 \cdot \log_{10}\left(rac{E[s^2]}{E[n^2]}ight)
  ```

- **Amplitude ratio**  
  ```math
  SNR_{dB} = 20 \cdot \log_{10}\left(rac{E[|s|]}{E[|n|]}ight)
  ```

Where:

- $s$ = filtered signal  
- $n$ = removed noise (raw – filtered)  

The linear fraction of signal power in the total is:

```math
SignalFraction = rac{SNR_{lin}}{1 + SNR_{lin}}
```

---

### 2.2 Artifact Suppression (Peaks)

We measure peak reduction as:

```math
\Delta Peak = \max_t |x_{raw}(t)| - \max_t |x_{filt}(t)|
```

Percent drop:

```math
\%Drop = 100 	imes rac{\Delta Peak}{\max_t |x_{raw}(t)|}
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

Crossing ±5 µV indicates **Drift Correction**.

---

### 2.4 Variance Smoothing

Variance reduction is computed as:

```math
\Delta Var\% = rac{Var(raw) - Var(filtered)}{Var(raw)} 	imes 100
```

Crossing ≥ 5% indicates **Smoothing Effect**.

---

### 2.5 Multi-Channel vs Single-Channel

For a window with $(C)$ EEG channels and $(T)$ time samples, our data arrays are shape $(C 	imes T)$.

- A **single-channel metric** operates on a vector $x \in \mathbb{R}^T$.
- An **average across channels** uses only EEG channels (exclude AUX/EMG/accel).

```math
ar{x}(t) = rac{1}{|S|} \sum_{c \in S} x_c(t), \quad t = 1, \ldots, T
```

Where $(S)$ is the EEG set you configure (e.g., BrainFlow `BoardShim.get_eeg_channels(board_id)`).

---

## 3. How Configurations Affect Output

- **Lambda (`filterHyperparameter`)**: Controls the strength of the MindsAI filter. Smaller values preserve more detail; larger values enforce stronger smoothing.  
- **SNR Method**: Chooses how signal-to-noise is quantified (variance, power, or amplitude). Affects console interpretation but not filtering.  
- **Artifact Suppression Threshold**: If peak drop exceeds threshold, console prints “Artifact Suppression.” Lower thresholds → more frequent tagging.  
- **Drift Threshold**: Defines what counts as baseline drift (mean/median shifts). Lower thresholds make console more sensitive.  
- **Variance Threshold**: Governs when the console reports “Smoothing Effect.”  

These settings only affect **console reporting** tags, not the actual MindsAI filter mathematics.

---

## 4. Code Reference Example

Snippet for SNR calculation in code:

```python
def calculate_snr(signal, noise, method="variance_ratio"):
    if method == "power_ratio":
        signal_power = np.mean(signal ** 2)
        noise_power  = np.mean(noise ** 2)
    elif method == "variance_ratio":
        signal_power = np.var(signal)
        noise_power  = np.var(noise)
    elif method == "amplitude_ratio":
        signal_power = np.mean(np.abs(signal))
        noise_power  = np.mean(np.abs(noise))
    else:
        raise ValueError(f"Unknown SNR method: {method}")
    return 10 * np.log10(signal_power / noise_power)
```

---

## 5. Notes

- Always disable synthetic noise (`INJECT_*`) when using real hardware.  
- Use only EEG channels (exclude AUX/accel/other).  
- Ensure units are consistent: µV → V before passing to MindsAI filter.
