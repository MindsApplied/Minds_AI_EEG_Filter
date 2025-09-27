# © 2023–2025 MindsApplied Incorporated — Patent pending (US provisional filed 2025-06-30)
# Licensed under the Polyform Noncommercial License 1.0.0.
# Commercial use is not permitted and requires a seperate license. See LICENSE and PATENTS.md. Contact: contact@minds-applied.com

import numpy as np
from scipy.signal import hilbert
from sklearn.base import BaseEstimator, TransformerMixin

class MindsAIFilter(BaseEstimator, TransformerMixin):
    def __init__(self, lambd: float = 1e-25, renorm: str = "diag"):
        self.lambd = float(lambd)
        self.renorm = renorm

    def _extract_phases(self, data_txc: np.ndarray) -> np.ndarray:
        analytic = hilbert(data_txc, axis=0)
        return np.angle(analytic)

    def _kuramoto_operator(self, phases_txc: np.ndarray) -> np.ndarray:
        T, C = phases_txc.shape
        P = np.zeros((C, C), dtype=np.float64)
        for i in range(C):
            for j in range(i + 1, C):
                d = np.sin(phases_txc[:, i] - phases_txc[:, j])
                val = np.sum(d * d)
                P[i, j] = val
                P[j, i] = val
        if self.renorm == "diag":
            EPS = 1e-12
            d = np.sqrt(np.clip(np.diag(P), EPS, None))
            Dinv = np.diag(1.0 / d)
            P = Dinv @ P @ Dinv
        return P

    def _closed_form_filter(self, y_cxt: np.ndarray, P_cxc: np.ndarray) -> np.ndarray:
        C = P_cxc.shape[0]
        I = np.eye(C, dtype=np.float64)
        inverse_term = np.linalg.inv(I + self.lambd * (P_cxc.T @ P_cxc))
        return inverse_term @ y_cxt

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)           # (channels, timepoints)
        trial_txc = X.T                                # (timepoints, channels)
        phases_txc = self._extract_phases(trial_txc)
        P = self._kuramoto_operator(phases_txc)
        return self._closed_form_filter(X, P)         # (channels, timepoints)

def mindsai_python_filter(data: np.ndarray, lambda_val: float = 1e-25) -> np.ndarray:
    return MindsAIFilter(lambd=lambda_val).fit_transform(data)


