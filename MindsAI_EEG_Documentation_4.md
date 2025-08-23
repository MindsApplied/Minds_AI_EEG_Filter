# MindsAI EEG Filter Documentation

## 2. Mathematical Background

### 2.1 Signal-to-Noise Ratio (SNR)

The SNR measures the relative strength of signal vs noise in decibels:

```math
\mathrm{SNR}_{dB} = 10 \cdot \log_{10}\left(\frac{P_{signal}}{P_{noise}} \right)
```

Choices for $P_{signal}, P_{noise}$:

- **Variance ratio**  
```math
\mathrm{SNR}_{dB} = 10 \cdot \log_{10}\left( \frac{\mathrm{Var}[s]}{\mathrm{Var}[n]} \right)
```

- **Power ratio**  
```math
\mathrm{SNR}_{dB} = 10 \cdot \log_{10}\left( \frac{\mathbb{E}[s^2]}{\mathbb{E}[n^2]} \right)
```

- **Amplitude ratio**  
```math
\mathrm{SNR}_{dB} = 20 \cdot \log_{10}\left( \frac{\mathbb{E}[|s|]}{\mathbb{E}[|n|]} \right)
```

Where:  
- $s =$ filtered signal  
- $n =$ removed noise (raw – filtered)

The linear fraction of signal power in the total is:

```math
\text{Signal Fraction} = \frac{\mathrm{SNR}_{lin}}{1 + \mathrm{SNR}_{lin}}
```

---

### 2.2 Artifact Suppression (Peaks)

Peak reduction per window:

```math
\Delta Peak = \max_t |x_{raw}(t)| - \max_t |x_{filt}(t)|
```

Percent drop:

```math
\%Drop = 100 \times \frac{\Delta Peak}{\max_t |x_{raw}(t)|}
```

Crossing ≥ 20% (configurable) → tag as **Artifact Suppression**.

---

### 2.3 Baseline Drift Correction (Mean/Median)

We measure baseline drift correction by comparing means and medians:

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
\Delta Var\% = (\mathrm{Var}[raw] - \mathrm{Var}[filtered]) / \mathrm{Var}[raw] \times 100
```

Crossing ≥ 5% (configurable) → tag as **Smoothing Effect**.

---

### 2.5 Multi-Channel vs Single-Channel

For a window with $(C)$ EEG channels and $(T)$ time samples, arrays are shape $(C \times T)$.

- **Single-channel metric** operates on one row $x \in \mathbb{R}^T$.  
- **Average across EEG channels** (excluding AUX/EMG):

```math
\bar{x}(t) = \frac{1}{|S|} \sum_{c \in S} x_c(t), \quad t = 1, \ldots, T
```

Where $(S)$ is the EEG-only set (e.g., BrainFlow `BoardShim.get_eeg_channels(board_id)`).

---
