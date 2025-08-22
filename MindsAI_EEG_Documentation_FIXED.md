
# MindsAI EEG – Configuration & Technical Notes

> **Tip:** This README is formatted so that **GitHub** and **Jupyter** render the math using MathJax.
> If you still see the raw symbols (`$$`, `\frac{}`, etc.), open the file in **GitHub’s Preview**, or in **Jupyter Notebook/Lab**, or use a Markdown viewer with MathJax enabled. VS Code also renders it in the Markdown Preview pane.

---

## 1) Quick Start

- **License (required)**  
  ```python
  import mindsai_filter_python
  mindsai_filter_python.initialize_mindsai_license('YOUR-LICENSE-KEY')
  print(mindsai_filter_python.get_mindsai_license_message())
  ```

- **Sampling rate (`fs`)**  
  Use BrainFlow when possible:
  ```python
  USE_BRAINFLOW = True
  USE_SYNTHETIC = False
  from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
  params = BrainFlowInputParams()
  board_id = BoardIds.SYNTHETIC_BOARD.value if USE_SYNTHETIC else BoardIds.CYTON_DAISY_BOARD.value

  if USE_BRAINFLOW:
      fs = BoardShim.get_sampling_rate(board_id)  # authoritative
  else:
      fs = 125  # MANUAL_FS fallback
  ```

- **Noise toggles (turn OFF for real devices):**
  ```python
  INJECT_BURST = False   # only channel_idx
  INJECT_FLAT  = False   # only channel_idx
  INJECT_SINE  = False   # global
  INJECT_WHITE = False   # global
  ```

- **Optional bandpass (before MindsAI):**
  ```python
  ENABLE_BANDPASS = True
  BP_LOW_HZ, BP_HIGH_HZ, BP_ORDER = 1.0, 40.0, 4
  # Choose a type you prefer, e.g. FilterTypes.BUTTERWORTH or _ZERO_PHASE
  ```

- **Console thresholds & SNR method:**
  ```python
  ARTIFACT_SUPPRESSION_THRESH = 20.0  # % peak drop
  DRIFT_THRESH_UV             = 5.0   # μV shift
  VARIANCE_SMOOTHING_THRESH   = 5.0   # % variance drop
  SNR_METHOD = "variance_ratio"       # or "power_ratio", "amplitude_ratio"
  ```

---

## 2) Mathematical Background

### 2.1 Signal‑to‑Noise Ratio (SNR)

The SNR measures the relative strength of signal vs. noise. We compute **dB** by

$$
\mathrm{SNR}_{\mathrm{dB}} \;=\; 10 \log_{10}\!\left(\frac{P_{\text{signal}}}{P_{\text{noise}}}\right).
$$

Concrete choices for \( P_{\text{signal}} \) and \( P_{\text{noise}} \):

**Variance ratio**
$$
\mathrm{SNR}_{\mathrm{dB}} \;=\; 10 \log_{10}\!\left(\frac{\mathrm{Var}(\text{signal})}{\mathrm{Var}(\text{noise})}\right).
$$

**Power ratio**
$$
\mathrm{SNR}_{\mathrm{dB}} \;=\; 10 \log_{10}\!\left(\frac{\mathbb{E}[\text{signal}^2]}{\mathbb{E}[\text{noise}^2]}\right).
$$

**Amplitude ratio** (uses mean absolute amplitude)
$$
\mathrm{SNR}_{\mathrm{dB}} \;=\; 20 \log_{10}\!\left(\frac{\mathbb{E}[|\text{signal}|]}{\mathbb{E}[|\text{noise}|]}\right).
$$

The linear SNR is \( \mathrm{SNR}_{\mathrm{lin}} = 10^{\mathrm{SNR}_{\mathrm{dB}}/10} \), and the **fraction of signal power** in the total is

$$
\text{Signal Power Fraction} \;=\; \frac{\mathrm{SNR}_{\mathrm{lin}}}{1+\mathrm{SNR}_{\mathrm{lin}}}.
$$

---

### 2.2 Artifact Suppression (Peaks)

We quantify peak reduction per window as
$$
\Delta \mathrm{Peak} \;=\; \max_t |x_{\text{raw}}(t)| \;-\; \max_t |x_{\text{filt}}(t)|,
$$
and the **percent drop**
$$
\%\mathrm{Drop} \;=\; 100 \times \frac{\Delta \mathrm{Peak}}{\max_t |x_{\text{raw}}(t)|}.
$$

If \(\%\mathrm{Drop} \geq 20\%\) (configurable), we tag **Artifact Suppression** in the console.

---

### 2.3 Baseline Drift Correction (Mean/Median)

We assess baseline shift by
$$
\Delta \mu \;=\; \mu_{\text{filt}} - \mu_{\text{raw}}, \qquad
\Delta \tilde{x} \;=\; \tilde{x}_{\text{filt}} - \tilde{x}_{\text{raw}}
$$
where \( \mu \) is the mean and \( \tilde{x} \) the median.  
If \( |\Delta \mu| \) or \( |\Delta \tilde{x}| \) exceeds a threshold (default \(5\,\mu\mathrm{V}\)), we tag **Drift Correction**.

---

### 2.4 Variance Smoothing

Smoothing is the relative drop in variance:
$$
\Delta \mathrm{Var}\% \;=\; 100 \times \frac{\mathrm{Var}(x_{\text{raw}}) - \mathrm{Var}(x_{\text{filt}})}{\mathrm{Var}(x_{\text{raw}})}.
$$
If this exceeds the threshold (default \(5\%\)), we tag **Smoothing Effect**.

---

### 2.5 Multi‑Channel vs Single‑Channel

For a window with \(C\) EEG channels and \(T\) time samples, our data arrays are shape \((C \times T)\).  
A **single‑channel** metric operates on a vector \(x \in \mathbb{R}^T\).

An **average across channels** uses only EEG channels (e.g., OpenBCI indices \(1\!:\!16\)):

$$
\bar{x}(t) \;=\; \frac{1}{|S|}\sum_{c \in S} x_c(t), \quad t=1,\dots,T,
$$

where \(S\) is the **EEG set** you configure (exclude AUX/EMG/accel). In code, slice only the EEG rows before averaging.

---

## 3) Code Notes (Key Snippets)

- **Restrict average to EEG channels only** (avoid AUX/other rows):
  ```python
  # board-reported EEG channels (safe default)
  eeg_rows = BoardShim.get_eeg_channels(board_id)

  # OR explicitly override for OpenBCI: 0..15 (first 16 channels)
  # eeg_rows = list(range(16))

  avg_raw  = np.mean(raw_plot_uv[eeg_rows, :], axis=0)
  avg_filt = np.mean(filtered_plot_uv[eeg_rows, :], axis=0)
  ```

- **SNR printing with interpretation** is built into `format_metrics_console_inline(...)`.

- **Band‑pass before MindsAI** (optional) is applied to a copy of the window.

- **Turn off synthetic noise** for real hardware:
  ```python
  INJECT_BURST = INJECT_FLAT = INJECT_SINE = INJECT_WHITE = False
  ```

---

## 4) Troubleshooting Rendering of Math

If you still see raw TeX like `$$ ... $$`:
1. **GitHub** – click “**Raw**” to view the text; to see equations, use the normal preview (not raw) and ensure you’re not looking at a code block. Math renders when `$$` is on its **own line with blank lines around it** (as used here).
2. **VS Code** – open **Markdown Preview** (`Ctrl+Shift+V`) or use the “Markdown: Open Preview to the Side” command.  
3. **Jupyter** – open the `.md` in **JupyterLab** or paste the sections into Markdown cells; LaTeX renders automatically.

---

Happy analyzing!
