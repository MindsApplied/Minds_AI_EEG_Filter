# PATENT NOTICE — MindsAI Signal Filter

**Owner:** MindsApplied Incorporated  
**Status:** Patent pending — US provisional filed **2025-06-30**  
**Contact:** contact@minds-applied.com  
**License of this repo:** Polyform Noncommercial 1.0.0 (commercial use requires a separate license)

> Required Notice: © 2023–2025 MindsApplied Incorporated (https://www.minds-applied.com/)

---

## 1) Non-Confidential Summary

This work concerns **methods and systems** for filtering multi-channel time-series data (e.g., EEG) using **physics-informed sensor fusion** and **synchrony-aware processing**. The approach combines:
- a **connectivity matrix** encoding spatial/functional relationships among sensors,
- **frequency-specific synchrony estimation** (global and regional),
- **adaptive sensor-fusion weights** derived from synchrony and noise estimates,
- **artifact/noise suppression** driven by physiologically plausible coherence,
- **boundary and windowing corrections** for real-time use.

The methodology is applicable beyond EEG (e.g., MEG, EMG, multi-sensor arrays, IoT grids). It is designed for **real-time and offline** operation and to improve downstream **AI/ML performance** (e.g., state classification, neurofeedback, BCI, interactive neuro-art).

---

## 2) Scope — Methodology (Process + System)

This filing covers both the **algorithmic workflow** and the **system architecture** that implements it.

### 2.1 Process (Method)
1. **Data Ingestion**  
   Receive multichannel time-series \(X \in \mathbb{R}^{C \times T}\) with optional sensor metadata (positions, regions, priors).

2. **Connectivity Modeling**  
   Build or import a **connectivity matrix** \(C \in \mathbb{R}^{C \times C}\) capturing spatial/functional relationships (e.g., montage adjacency, atlas/region mapping, historical correlation/coherence).

3. **Frequency-Aware Phase/Synchrony Estimation**  
   Compute instantaneous phase/analytic signals per channel and **per band** (e.g., δ, θ, α, β, γ). Estimate **global and regional synchrony** weighted by \(C\). (Kuramoto-style or equivalent synchrony models can be used; parameters may differ by band to reflect long- vs short-range coupling.)

4. **Adaptive Sensor Fusion**  
   Derive **per-channel (and optionally per-time/band) weights** from synchrony, coherence, SNR estimates, and artifact priors. Fuse channels to enhance plausible neural components while attenuating inconsistent/noisy contributions.

5. **Artifact/Noise Suppression & Boundary Handling**  
   Detect and suppress artifacts using synchrony-coherence outlier criteria and apply **window/edge corrections** to reduce filter edge effects in streaming.

6. **Output & Telemetry**  
   Emit a cleaned multichannel signal (same shape as input) plus optional **quality metrics** (e.g., SNR deltas, variance reduction, peak suppression, drift correction) for auditability and downstream AI.

### 2.2 System (Implementation)
- **Processing pipeline**: Modular stages (ingest → phase/synchrony → fusion → artifact suppression → output) operable in real-time windows.  
- **Parameterization**: A hyperparameter (e.g., λ) governs adaptation strength; additional band-specific parameters and region definitions can be configured.  
- **Deployment**: Runs on commodity hardware; supports streaming (BCI, neurofeedback) and batch (offline analysis). Provides language bindings and integration hooks for ML/visualization.

---

## 3) Representative (Non-Limiting) Use Cases

- **Real-time EEG feedback / BCI**: Improve robustness under motion/ocular artifacts and low electrode counts.  
- **Affective computing**: Enhance signals for **valence/arousal** or related state prediction.  
- **Clinical/Research pipelines**: Preprocessing prior to ML or statistical modeling.  
- **Multi-sensor arrays / IoT**: Synchrony-guided denoising where spatial/functional relationships exist.

---

## 4) Relationship to Open-Source Distribution

This repository is licensed **non-commercially** (Polyform Noncommercial 1.0.0).  
Any commercial use (including within for-profit internal tools, products, SaaS, or paid services) requires a **separate commercial license** from MindsApplied.  
Please contact **contact@minds-applied.com** to discuss terms.

---

## 5) Priorities and Dates

- **US Provisional Filing Date:** **2025-06-30**  
- Subsequent non-provisional filings and international phases (if any) will be added here when public.

---

## 6) Citations / Acknowledgments

If you use this software, please cite the preprint:
Wesierski, J-M., Rodriguez, N. A lightweight, physics-based, sensor-fusion filter for real-time EEG. bioRxiv (2025).
https://doi.org/10.1101/2025.09.24.675953

A machine-readable citation file is provided: CITATION.cff.

---

## 7) Contact for Licensing & Collaboration

- **Commercial licensing:** contact@minds-applied.com  
- **Research collaborations / OEM / field-of-use:** contact@minds-applied.com

---

## 8) Disclaimer

This PATENTS.md provides a **non-confidential** overview for notice and collaboration purposes.  
It is **not** a legal specification, does **not** waive any rights, and does **not** disclose enabling details beyond what is appropriate for public notice. The scope of any granted claims will be defined by issued patents and applications on file.
