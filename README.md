# Minds AI Filter for EEG Documentation

*Your* Minds AI Signal Filter uses spatial coupling and band-aware weighting to condition the signal based on the physics of true neural activity. “AI” is included in the naming convention due to its utility in artificial intelligence pipelines, but the filter itself does not use deep learning. It also doesn't require pretraining or clean stretches of data. It can work standalone or in combination with other filters (such as for specific noise) and is equipped for alternative or multi-modal time-series data.
Specifically the MAI Filter:
- Suppresses artifacts like transient motion/ocular bursts while preserving underlying rhythms across channels
- Reduces high-frequency noise (>40 Hz) and sharpens low-frequency activity (~3-7Hz)
- Smooths variance and reduces baseline-drift
- Interpolates flatlines

Additional References:
- Preprint: https://doi.org/10.1101/2025.09.24.675953.
- [Live demo of the filter and below application](https://www.youtube.com/watch?v=YgEt1vKYDc4).
- License: Polyform Noncommercial 1.0.0 — [Contact MindsApplied](https://www.minds-applied.com/contact) for commercial usage. See [LICENSE](LICENSE).
- Patent status: **Patent pending** (US provisional filed 2025-06-30). See [PATENTS](PATENTS.md).
- Cite this software: see [CITATION](CITATION.cff) and the preprint below.
- Reconstructs flatlines based on surrounding neural activity 
---

## 1) Filter Package Installation

**Package only**
```bash
pip install "git+https://github.com/MindsApplied/Minds_AI_EEG_Filter@main"
```
**Package and Demo Apps**
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
It's that easy! It expects `data` to be a 2-D continuous array of **channels x time** and relies on one hyperparameter. It should be applied to the data as a whole, prior to other filters or indiviudal electrode analysis. It can be applied to large trials or looped for real-time usage. 
> An intialization key is no longer required. 

<p align="center">
  <img src="images/MAI_Filter_Lambda_Funnel_labled.png" width="700" alt="Tailoring Lambda Description Visual">
</p>

### 1.2 Tightening Lambda 

The hyperparameter integer, `tailoring_lambda`, controls how much your Minds AI Filter modifies the original signal and should be input on a logarithmic scale between `0` and `0.1`. A lower `lambda` value like the default `1e-25` causes the filter to make bolder adjustments for more complex transformations that highlight the structure across `channels`, such as for real-time filtering (1 second windows). A higher `lambda` value like `1e-40` works best with more data (such as 60-second trials) for still helpful, but more conservative adjustments.

## 2) Real-time and Offline Demo Apps (Not required for filter usage)
We provide 2 apps that make it easy to test and see signal quality improvement from the Minds AI Filter. One for real-time streaming directly from your headset, and the other for feeding segments of prerecorded data. Both apps visualize the signal and removed noise, as well as provide SNR metrics for signal quality improvement.

### 2.1 Real-Time Streaming
<p align="center">
  <img src="images/MAI_Online_Demo_UI.png" alt="Real-time app UI Image">
</p>

The real-time demo app, used in the video above, can be found in the examples folder as `Minds_AI_Filter_Real-time_Signal_Analysis.py`.
Documentation for testing and configuration can be found [here in our wiki](https://github.com/MindsApplied/Minds_AI_EEG_Filter/wiki/Real%E2%80%90time-Minds-AI-Filter-Demo-Application)

### 2.2 Offline Upload
<p align="center">
  <img src="images/MAI_Offline_Demo_UI.png" alt="Offline app UI Image">
</p>

The offline demo app can be found in the examples folder as  `Minds_AI_Filter_Offline_Signal_Analysis.py`.
Documentation for testing and configuration can be found [here in our wiki](https://github.com/MindsApplied/Minds_AI_EEG_Filter/wiki/Offline-Minds-AI-Filter-Demo-Application)

---

## 3) Patent Status

**Patent pending.** Portions of this work are covered by a US provisional application filed **2025-06-30**.  
For details, see [PATENTS.md](PATENTS.md). For commercial licensing, contact contact@minds-applied.com.

## 4) How to Cite

If you use this software, please cite the preprint:

> Wesierski, J-M., Rodriguez, N. *A lightweight, physics-based, sensor-fusion filter for real-time EEG.* bioRxiv (2025).  
> https://doi.org/10.1101/2025.09.24.675953

A machine-readable citation file is provided: [CITATION.cff](CITATION.cff).

