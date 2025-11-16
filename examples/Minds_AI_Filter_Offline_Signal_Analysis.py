# -*- coding: utf-8 -*-
import os
import sys
import json
import math
import traceback
import webbrowser
import datetime
import csv
import re

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
from tkinter import font as tkfont
from tkinter import scrolledtext

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ---- MAI filter ----
import mindsai_filter_python as mai

"""
Minds AI Filter Offline testing app. Created by JM Wesierski

Features:
- Load EDF or CSV/TSV/TXT EEG data (up to ~30 minutes).
- Let the user select EEG channels by 1-based column indices and ranges
  (e.g. "1-21,23-26,28-31").
- Apply MindsAI filter in repeated windows (1–60 s) across the full recording.
- Display only the first 2 minutes in the GUI for visualization.
- Export:
    * For EDF input: new EDF with all channels preserved but EEG channels replaced
      by filtered data.
    * For CSV/TSV/TXT input: CSV with all original columns and EEG columns overwritten
      by filtered data.
"""

# ======================= CONSTANTS / THEME =======================

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(APP_DIR, "data", "eeg.edf")  # csv works also

LAMBDA_DEFAULT = 34

FILTERED_COLOR  = "#45c98f"  # Minds AI Filtered data
RAW_COLOR       = "#ffffff"  # Raw Signal
RMV_NOISE_COLOR = "#889eea"  # Noise

MAX_BYTES    = 10 * 1024 * 1024

# Recording duration limits: allow up to 30 minutes of data
MIN_SEC, MAX_SEC = 4, 1800
MIN_FS,  MAX_FS  = 50, 512
MIN_CH,  MAX_CH  = 4, 64

# Only show the first 2 minutes in the UI
DISPLAY_MAX_SEC = 120

PLOT_FONT_DELTA = 2

ARTIFACT_SUPPRESSION_THRESH = 20.0
DRIFT_THRESH_UV             = 5.0
VARIANCE_SMOOTHING_THRESH   = 5.0


# ======================= USER MADE ERROR =======================

def human_err(msg: str, add_not_related: bool = False) -> str:
    tail = "\nThis is not related to the MAI Filter." if add_not_related else ""
    return f"{msg}{tail}"


# ======================= CSV INTAKE =======================

def read_numeric_csv(path: str):
    with open(path, "rb") as fb:
        blob = fb.read()
        if len(blob) > MAX_BYTES:
            raise ValueError(human_err(f"File too large (> {MAX_BYTES//(1024*1024)} MB).", True))

    text = None
    if blob.startswith(b"\xff\xfe") or blob.startswith(b"\xfe\xff"):
        try: text = blob.decode("utf-16")
        except Exception: pass
    elif blob.startswith(b"\xef\xbb\xbf"):
        try: text = blob.decode("utf-8-sig")
        except Exception: pass
    if text is None:
        for enc in ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "latin-1"):
            try:
                text = blob.decode(enc); break
            except Exception:
                continue
    if text is None:
        raise ValueError(human_err("Could not decode file in a common text encoding.", True))

    text = (text.replace("\r\n", "\n")
                .replace("\r", "\n")
                .replace("\x00", "")
                .replace("\u00A0", " ")
                .replace("\u2007", " ")
                .replace("\u202F", " "))

    sample_lines = [ln for ln in text.split("\n") if ln.strip()][:100]
    if not sample_lines:
        raise ValueError(human_err("CSV appears empty.", True))
    sample = "\n".join(sample_lines)

    delimiter = None
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample, delimiters=[",", "\t", ";", "|"])
        delimiter = dialect.delimiter
    except Exception:
        delimiter = None

    delim_counts = {
        "\t": sample.count("\t"),
        ",":  sample.count(","),
        ";":  sample.count(";"),
        "|":  sample.count("|"),
    }
    explicit_delim = max(delim_counts, key=delim_counts.get) if max(delim_counts.values()) > 0 else None
    if delimiter is None:
        delimiter = explicit_delim

    def _strip_outer_quotes(s: str):
        s = s.strip()
        if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
            return s[1:-1]
        return s

    def split_line(ln: str):
        s = ln.strip()
        if not s or s.startswith("#"):
            return []
        if delimiter is not None:
            rdr = csv.reader([s], delimiter=delimiter, quoting=csv.QUOTE_MINIMAL)
            parts = next(rdr, [])
            if len(parts) == 1:
                s2 = _strip_outer_quotes(s)
                for d in [delimiter, "\t", ",", ";", "|"]:
                    if d in s2:
                        return s2.split(d)
                return re.split(r"\s+", s2)
            return parts
        else:
            s2 = _strip_outer_quotes(s)
            return re.split(r"\s+", s2)

    rows, max_len = [], 0
    for ln in text.split("\n"):
        parts = split_line(ln)
        if not parts:
            continue
        rows.append(parts)
        max_len = max(max_len, len(parts))

    if not rows:
        raise ValueError(human_err("CSV appears empty or non-numeric.", True))

    out = np.empty((len(rows), max_len), dtype=float)
    out[:] = np.nan
    thousands_re = re.compile(r"(?<=\d),(?=\d)")

    def _to_float(tok: str):
        if tok is None:
            return np.nan
        s = tok.strip().strip('"').strip("'")
        if s == "" or s.lower() == "nan":
            return np.nan
        s = thousands_re.sub("", s)
        try:
            return float(s)
        except Exception:
            return np.nan

    for i, parts in enumerate(rows):
        for j in range(min(len(parts), max_len)):
            out[i, j] = _to_float(parts[j])

    if not np.isfinite(out).any():
        print("[DEBUG] First non-empty line after parsing looked like:", rows[0] if rows else "<none>")
        raise ValueError(human_err("CSV appears empty or non-numeric after parsing.", True))

    row_mask = np.isfinite(out).any(axis=1)
    out = out[row_mask, :]
    col_mask = np.isfinite(out).any(axis=0)
    if not col_mask.any():
        raise ValueError(human_err("CSV contains no numeric columns.", True))
    out = out[:, col_mask]

    if delimiter is not None:
        write_delim = delimiter
    else:
        write_delim = " "

    return out, write_delim


# ======================= EDF INTAKE (FULL, NO AUTO-DROP) =======================

try:
    import pyedflib
    _HAS_PYEDFLIB = True
except Exception:
    _HAS_PYEDFLIB = False


def read_edf_full(path: str):
    """
    Read EDF and return a meta dict containing all channels, sampling rates,
    signal arrays, and headers. No channels are dropped here.
    """
    if not _HAS_PYEDFLIB:
        raise RuntimeError(human_err("EDF support requires 'pyEDFlib' (pip install pyEDFlib).", True))

    import collections

    f = pyedflib.EdfReader(path)
    try:
        n_ch = f.signals_in_file
        labels_all = f.getSignalLabels()
        fs_list = [float(f.getSampleFrequency(i)) for i in range(n_ch)]
        fs_ints = [int(round(x)) for x in fs_list]
        nsamples = f.getNSamples()  # list of samples per channel

        valid_fs = [fs for fs in fs_ints if fs > 0]
        if not valid_fs:
            raise ValueError(human_err("EDF contains no channels with positive sampling rate.", True))
        counter = collections.Counter(valid_fs)
        majority_fs, _ = counter.most_common(1)[0]

        # Read all channels fully
        signals_all = [f.readSignal(i).astype(float) for i in range(n_ch)]

        try:
            units_all = [f.getPhysicalDimension(i) for i in range(n_ch)]
        except Exception:
            units_all = [""] * n_ch

        file_header    = f.getHeader()
        signal_headers = [f.getSignalHeader(i) for i in range(n_ch)]

        meta = {
            "labels": labels_all,
            "fs_list": fs_ints,
            "majority_fs": majority_fs,
            "units": units_all,
            "nsamples": nsamples,
            "signals_all": signals_all,
            "file_header": file_header,
            "signal_headers": signal_headers,
        }
        return meta
    finally:
        f._close()
        del f


# ======================= ORIENTATION & VALIDATION (CSV/TXT ONLY) =======================

def decide_orientation(arr: np.ndarray, fs: int):
    r, c = arr.shape
    candidates = []

    chA, tA = r, c
    durA = tA / fs
    okA = (MIN_CH <= chA <= MAX_CH) and (MIN_SEC <= durA <= MAX_SEC)
    candidates.append(("A", okA, chA, tA, False))

    chB, tB = c, r
    durB = tB / fs
    okB = (MIN_CH <= chB <= MAX_CH) and (MIN_SEC <= durB <= MAX_SEC)
    candidates.append(("B", okB, chB, tB, True))

    valid = [c for c in candidates if c[1]]
    if len(valid) == 1:
        _, _, _, _, flip = valid[0]
        return (arr.T if flip else arr), flip
    elif len(valid) == 2:
        center = 0.5 * (MIN_SEC + MAX_SEC)
        dA = abs((c / fs) - center)
        dB = abs((r / fs) - center)
        flip = dB < dA
        return (arr.T if flip else arr), flip
    else:
        raise ValueError(human_err(
            f"Data shape {arr.shape} with fs={fs} Hz does not fit limits: "
            f"channels {MIN_CH}-{MAX_CH}, duration {MIN_SEC}-{MAX_SEC} s.", True))


# ======================= METRICS =======================

def compute_metrics(raw_uV: np.ndarray, filt_uV: np.ndarray, method: str, ch_idx: int, fs: int):
    def snr(sig, noise, method):
        if method == "power_ratio":
            s = float(np.mean(sig**2)); n = float(np.mean(noise**2))
        elif method == "variance_ratio":
            s = float(np.var(sig)); n = float(np.var(noise))
        elif method == "amplitude_ratio":
            s = float(np.mean(np.abs(sig))); n = float(np.mean(np.abs(noise)))
        else:
            raise ValueError(human_err(f"Unknown SNR method: {method}", True))
        if n <= 0: return float("inf")
        return 10.0 * math.log10(s/n)

    diff = raw_uV - filt_uV
    ch_raw  = raw_uV[ch_idx]
    ch_filt = filt_uV[ch_idx]

    peak_before = float(np.max(np.abs(ch_raw)))
    peak_after  = float(np.max(np.abs(ch_filt)))
    var_before  = float(np.var(ch_raw))
    var_after   = float(np.var(ch_filt))

    impact_ch = dict(
        peak_before=peak_before,
        peak_after=peak_after,
        peak_reduction=peak_before - peak_after,
        mean_shift=float(np.mean(ch_filt) - np.mean(ch_raw)),
        median_shift=float(np.median(ch_filt) - np.median(ch_raw)),
        artifact_variance_reduction_pct=((var_before - var_after)/var_before*100.0) if var_before>0 else 0.0
    )

    metrics = {
        "fs_hz": fs,
        "channels": int(raw_uV.shape[0]),
        "duration_sec": float(raw_uV.shape[1] / fs),
        "lambda": None,
        "snr_method": method,
        "snr_db_channel": None,
        "impact_channel": impact_ch,
        "tags_channel": {
            "artifact_suppression": (
                impact_ch["peak_before"] > 0 and
                ((impact_ch["peak_before"] - impact_ch["peak_after"]) / impact_ch["peak_before"] * 100.0
                 >= ARTIFACT_SUPPRESSION_THRESH)
            ),
            "drift_correction": (
                abs(impact_ch["mean_shift"]) >= DRIFT_THRESH_UV or
                abs(impact_ch["median_shift"]) >= DRIFT_THRESH_UV
            ),
            "smoothing_effect": impact_ch["artifact_variance_reduction_pct"] >= VARIANCE_SMOOTHING_THRESH
        },
        "thresholds": {
            "artifact_suppression_pct": ARTIFACT_SUPPRESSION_THRESH,
            "drift_uv": DRIFT_THRESH_UV,
            "smoothing_pct": VARIANCE_SMOOTHING_THRESH
        }
    }

    snr_db = snr(ch_filt, ch_raw - ch_filt, method)
    metrics["snr_db_channel"] = (None if (snr_db == float("inf") or np.isinf(snr_db)) else float(snr_db))

    return metrics


def print_metrics_console(metrics: dict):
    snr_db = metrics["snr_db_channel"]
    if snr_db is None:
        snr_text = "∞ dB (noise≈0)"; ratio_text = "Signal ≫ noise"; sig_pct = "≈100%"
    else:
        lin = 10 ** (snr_db/10.0)
        sig_frac = lin/(1.0+lin)
        snr_text = f"{snr_db:.2f} dB"; ratio_text = f"Signal ~{lin:.1f}× stronger than noise"; sig_pct = f"≈{sig_frac*100:.0f}%"
    imp = metrics["impact_channel"]
    peak_before = imp["peak_before"]; peak_after = imp["peak_after"]
    peak_drop = imp["peak_reduction"]; peak_pct = (peak_drop/peak_before*100.0) if peak_before>0 else 0.0
    var_drop = imp["artifact_variance_reduction_pct"]
    mean_shift = imp["mean_shift"]; median_shift = imp["median_shift"]

    dur = metrics["duration_sec"]
    print(f"[Window: 00:00:00.000 → 00:00:{dur:05.2f}]")
    print(f"[SNR: {snr_text} | {ratio_text} | {sig_pct} signal power]  [SNR method: {metrics['snr_method']}]")
    print(f"[Peak: {peak_before:.2f}→{peak_after:.2f} μV (↓{peak_drop:.2f} μV, {peak_pct:.0f}%)]")
    print(f"[Baseline Shift: mean {mean_shift:+.2f} μV | median {median_shift:+.2f} μV]")
    print(f"[Variance ↓{var_drop:.1f}%]")
    print()


# ======================= EXPORT HELPERS =======================

def save_filtered_and_metrics_same_format(
    base_dir,
    base_name,
    lam_str,
    full_input_numeric_rows_cols,
    selected_indices_0based,
    filt_cxt_volts,
    flipped,
    unit_scale_in,
    write_delim,
    metrics
):
    """
    CSV/TSV/TXT export:
    - full_input_numeric_rows_cols: original (T x C) numeric matrix
    - selected_indices_0based: columns that were EEG and filtered
    - filt_cxt_volts: filtered EEG block in volts, shape (C x T) *in cable orientation*
    """
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    filtered_name = f"{base_name}_mai_filtered_{lam_str}_{ts}.csv"
    metrics_name  = f"{base_name}_mai_metrics_{lam_str}_{ts}.json"
    filtered_path = os.path.join(base_dir, filtered_name)
    metrics_path  = os.path.join(base_dir, metrics_name)

    inv_unit = 1.0 / unit_scale_in
    filt_in_orig_unit_cxt = filt_cxt_volts * inv_unit  # (C x T)

    if flipped:
        sel_filtered_rows_cols = filt_in_orig_unit_cxt.T  # (T x C)
    else:
        sel_filtered_rows_cols = filt_in_orig_unit_cxt    # (C x T)

    arr_out = np.array(full_input_numeric_rows_cols, copy=True)

    expected_shape = (arr_out.shape[0], selected_indices_0based.size)
    if sel_filtered_rows_cols.shape != expected_shape:
        raise RuntimeError(
            f"Filtered block shape {sel_filtered_rows_cols.shape} doesn't match expected {expected_shape} "
            f"(orientation/length/fs mismatch)."
        )

    arr_out[:, selected_indices_0based] = sel_filtered_rows_cols

    np.savetxt(filtered_path, arr_out, delimiter=write_delim, fmt="%.6f")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return filtered_path, metrics_path


def save_filtered_edf_and_metrics(
    edf_path,
    meta,
    eeg_indices,
    filt_cxt_volts,
    unit_scale_in,
    metrics,
    lam_str
):
    """
    EDF export:
    - edf_path: input EDF path
    - meta: dict from read_edf_full()
    - eeg_indices: 0-based EDF channel indices selected as EEG
    - filt_cxt_volts: filtered EEG block in volts, shape (C x T), where row i corresponds
                      to eeg_indices[i]
    - unit_scale_in: input-unit → volts scale factor (e.g. 1e-6 for µV)
    - metrics: JSON-able dict
    - lam_str: string like "1e-34" for filenames
    """
    if not _HAS_PYEDFLIB:
        raise RuntimeError(human_err("EDF export requires 'pyEDFlib'.", True))

    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    base_dir  = os.path.dirname(edf_path)
    base_name = os.path.splitext(os.path.basename(edf_path))[0]

    out_edf  = os.path.join(base_dir, f"{base_name}_mai_filtered_{lam_str}_{ts}.edf")
    out_json = os.path.join(base_dir, f"{base_name}_mai_metrics_{lam_str}_{ts}.json")

    labels         = meta["labels"]
    fs_list        = meta["fs_list"]
    signals_all    = meta["signals_all"]
    nsamples       = meta["nsamples"]
    file_header    = meta["file_header"]
    signal_headers = meta["signal_headers"]

    n_ch = len(labels)
    C, T = filt_cxt_volts.shape
    if C != len(eeg_indices):
        raise RuntimeError(
            f"Filtered EEG block has {C} rows but eeg_indices has {len(eeg_indices)} channels."
        )

    # Start from original signals
    signals_out = [np.array(sig, copy=True) for sig in signals_all]

    # Convert filtered data back to original units
    inv_unit = 1.0 / unit_scale_in
    filt_in_orig_unit = filt_cxt_volts * inv_unit  # (C x T)

    for row_idx, ch_idx in enumerate(eeg_indices):
        if ch_idx < 0 or ch_idx >= n_ch:
            continue
        n_orig = nsamples[ch_idx]
        n_use  = min(T, n_orig)
        signals_out[ch_idx][:n_use] = filt_in_orig_unit[row_idx, :n_use]

    writer = pyedflib.EdfWriter(
        out_edf,
        n_ch,
        file_type=pyedflib.FILETYPE_EDFPLUS
    )

    writer.setHeader(file_header)
    writer.setSignalHeaders(signal_headers)
    writer.writeSamples(signals_out)
    writer.close()

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return out_edf, out_json


# ======================= PLOTTING HELPERS =======================

def draw_dark(ax):
    ax.set_facecolor("black")
    for s in ax.spines.values():
        s.set_color("white")
    ax.tick_params(colors="white", labelsize=plt.rcParams.get("font.size", 10) + PLOT_FONT_DELTA)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.grid(True, alpha=0.2, color="white")


def _apply_view_limits(ax, t, y_list, clip_pct=None):
    if t.size == 0:
        return
    ax.set_xlim(float(t[0]), float(t[-1]))
    if clip_pct is None:
        ax.relim(); ax.autoscale_view(scalex=False, scaley=True); return
    vals = []
    for y in y_list:
        y = np.asarray(y)
        if y.size:
            vals.append(y[np.isfinite(y)])
    if not vals:
        ax.relim(); ax.autoscale_view(scalex=False, scaley=True); return
    vals = np.concatenate(vals)
    if vals.size == 0:
        ax.relim(); ax.autoscale_view(scalex=False, scaley=True); return
    lo = np.percentile(vals, clip_pct)
    hi = np.percentile(vals, 100.0 - clip_pct)
    if hi <= lo:
        ax.relim(); ax.autoscale_view(scalex=False, scaley=True); return
    pad = 0.05 * (hi - lo + 1e-9)
    ax.set_ylim(lo - pad, hi + pad)


# ======================= TKINTER GUI =======================

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Minds AI — Offline Upload Demo")
        self.configure(bg="black")
        self.geometry("1200x900")

        base_size = 13
        self.ui_font      = tkfont.Font(family="Segoe UI", size=base_size)
        self.ui_font_bold = tkfont.Font(family="Segoe UI", size=11, weight="bold")
        self.console_font = tkfont.Font(family="Consolas", size=11)

        style = ttk.Style(self)
        try: style.theme_use("clam")
        except Exception: pass
        style.configure("TLabel",   font=self.ui_font)
        style.configure("TButton",  font=self.ui_font_bold, padding=6)
        style.configure("TEntry",   padding=4)
        self.ui_font_combo = tkfont.Font(family="Segoe UI", size=base_size+4)
        style.configure("Big.TCombobox", font=self.ui_font_combo, padding=2)

        self.file_path    = tk.StringVar(value=DEFAULT_PATH)
        self.fs_entry     = tk.StringVar(value="500")            # default fs text/value
        self.snr_method   = tk.StringVar(value="power_ratio")
        self.channel_idx  = tk.IntVar(value=0)
        self.lambda_exp   = tk.IntVar(value=LAMBDA_DEFAULT)
        self.status       = tk.StringVar(value="Ready.")
        self.view_window  = tk.StringVar(value="1 s")
        # Filter window for repeated chunked filtering
        self.filter_window = tk.StringVar(value="30 s")

        # Supports multiple ranges: e.g. "1-21,23-26,28-31"
        self.col_range_str = tk.StringVar(value="1-8")

        self._timeline_var  = tk.IntVar(value=0)

        self._raw_input_orientation_timepoints_by_channels = False
        self._raw_uV_cxt = None
        self._filt_uV_cxt = None
        self._last_metrics = None
        self._last_base_dir = None
        self._last_base_name = None
        self._last_fs = None
        self._last_lambda = None
        self._last_channel = 0
        self._last_t = None
        self._last_raw_matrix = None    # for CSV/TSV/TXT
        self._last_write_delim = ","
        self._last_selected_indices = None  # for CSV/TSV/TXT
        self._unit_scale_in = 1e-6
        self._unit_name_in  = "µV"

        # EDF-specific state
        self._edf_meta = None
        self._edf_is_input = False
        self._edf_eeg_indices = None  # 0-based EDF channel indices used as EEG

        self._build_controls()
        self._build_console()
        self._build_timeline()
        self._build_plot()

        self._try_update_detected_columns(initial=True)

    # ---------- UI building ----------

    def _build_controls(self):
        frm = tk.Frame(self, bg="black")
        frm.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        tk.Label(frm, text="EDF/CSV:", fg="white", bg="black", font=self.ui_font).grid(row=0, column=0, sticky="w")
        tk.Entry(frm, textvariable=self.file_path, width=70, font=self.ui_font).grid(row=0, column=1, padx=5)
        ttk.Button(frm, text="Browse…", command=self.browse_file).grid(row=0, column=2, padx=5)
        ttk.Button(frm, text="Open folder", command=self.open_folder).grid(row=0, column=3, padx=5)

        tk.Label(frm, text="fs (Hz):", fg="white", bg="black", font=self.ui_font).grid(row=1, column=0, sticky="w")
        e = tk.Entry(frm, textvariable=self.fs_entry, width=30, fg="gray", font=self.ui_font)
        e.grid(row=1, column=1, sticky="w")
        e.bind("<FocusIn>", lambda ev: self._clear_placeholder())

        man_frm = tk.Frame(frm, bg="black")
        man_frm.grid(row=2, column=0, columnspan=4, sticky="w", pady=(6,0))
        tk.Label(
            man_frm,
            text="EEG channel columns (1-based):",
            fg="white", bg="black", font=self.ui_font
        ).pack(side=tk.LEFT)
        self.cols_entry = tk.Entry(man_frm, textvariable=self.col_range_str, width=30, font=self.ui_font)
        self.cols_entry.pack(side=tk.LEFT, padx=8)
        self.detect_label = tk.Label(man_frm, text="Detected: — columns", fg="#C8D1FF", bg="black", font=self.ui_font)
        self.detect_label.pack(side=tk.LEFT, padx=10)

        lam_frm = tk.Frame(frm, bg="black")
        lam_frm.grid(row=3, column=0, columnspan=2, sticky="w", pady=6)
        tk.Label(lam_frm, text="λ (filter strength):", fg="white", bg="black", font=self.ui_font).pack(side=tk.LEFT)
        self.lambda_scale = tk.Scale(
            lam_frm, from_=25, to=40, orient=tk.HORIZONTAL, variable=self.lambda_exp,
            length=360, bg="black", fg="white", highlightthickness=0,
            troughcolor="#333333", command=lambda v: self._update_lambda_label()
        )
        self.lambda_scale.set(LAMBDA_DEFAULT)
        self.lambda_scale.pack(side=tk.LEFT, padx=10)
        self.lam_label = tk.Label(lam_frm, text=f"1e-{LAMBDA_DEFAULT} (default)", fg="white", bg="black", font=self.ui_font)
        self.lam_label.pack(side=tk.LEFT)

        # Filter window length selector
        filt_frm = tk.Frame(frm, bg="black")
        filt_frm.grid(row=4, column=0, columnspan=2, sticky="w", pady=2)
        tk.Label(
            filt_frm, text="Filter window (sec):",
            fg="white", bg="black", font=self.ui_font
        ).pack(side=tk.LEFT)
        self.filter_box = ttk.Combobox(
            filt_frm,
            values=["1 s", "4 s", "30 s", "60 s"],
            textvariable=self.filter_window,
            width=10,
            state="readonly",
            style="Big.TCombobox"
        )
        self.filter_box.pack(side=tk.LEFT, padx=8)

        ch_frm = tk.Frame(frm, bg="black")
        ch_frm.grid(row=1, column=4, columnspan=2, sticky="w")
        tk.Label(ch_frm, text="Channel (plot index):", fg="white", bg="black", font=self.ui_font).pack(side=tk.LEFT)
        self.ch_spin = tk.Spinbox(ch_frm, from_=0, to=63, width=5, textvariable=self.channel_idx, font=self.ui_font)
        self.ch_spin.pack(side=tk.LEFT, padx=5)

        tk.Label(frm, text="SNR method:", fg="white", bg="black", font=self.ui_font).grid(row=3, column=2, sticky="e")
        self.snr_box = ttk.Combobox(
            frm, values=["power_ratio","variance_ratio","amplitude_ratio"],
            textvariable=self.snr_method, width=18, state="readonly", style="Big.TCombobox"
        )
        self.snr_box.grid(row=3, column=3, sticky="w")

        tk.Label(frm, text="View:", fg="white", bg="black", font=self.ui_font).grid(row=3, column=4, sticky="e")
        self.view_box = ttk.Combobox(
            frm, values=["1 s", "5 s", "10 s", "30 s", "All"],
            textvariable=self.view_window, width=10, state="readonly", style="Big.TCombobox"
        )
        self.view_box.grid(row=3, column=5, sticky="w")
        self.view_box.bind("<<ComboboxSelected>>", lambda e: self._on_view_changed())

        btn_frm = tk.Frame(self, bg="black")
        btn_frm.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        ttk.Button(btn_frm, text="Validate & Visualize", command=self.process).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frm, text="Export Filtered + Metrics", command=self.export_outputs).pack(side=tk.LEFT, padx=5)

    def _build_console(self):
        con_frm = tk.Frame(self, bg="black")
        con_frm.pack(side=tk.TOP, fill=tk.BOTH, padx=10, pady=6)
        tk.Label(con_frm, text="Status / Console", fg="white", bg="black", font=self.ui_font).pack(anchor="w")

        self.console = scrolledtext.ScrolledText(
            con_frm, height=6, wrap="word",
            bg="#111111", fg="#EEEEEE", insertbackground="#EEEEEE",
            borderwidth=1, relief="solid"
        )
        self.console.configure(font=self.console_font)
        self.console.pack(fill=tk.BOTH, expand=False)
        self.console.tag_configure("ok",    foreground="#A7E3A1")
        self.console.tag_configure("error", foreground="#FF9898")
        self.console.tag_configure("info",  foreground="#C8D1FF")

    def _build_timeline(self):
        tl = tk.Frame(self, bg="black")
        tl.pack(side=tk.TOP, fill=tk.X, padx=14, pady=(2, 6))

        tk.Label(tl, text="Timeline:", fg="white", bg="black", font=self.ui_font).pack(side=tk.LEFT)
        self.timeline = tk.Scale(
            tl, from_=0, to=0, resolution=1, orient=tk.HORIZONTAL, length=5400,
            sliderlength=48,
            variable=self._timeline_var, showvalue=False,
            bg="black", fg="white", highlightthickness=0, troughcolor="#333333",
            command=lambda v: self._on_timeline_change()
        )
        self.timeline.pack(side=tk.LEFT, padx=12, fill=tk.X, expand=True)

    def _build_plot(self):
        self.fig, self.axs = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)
        self.fig.patch.set_facecolor("black")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        for ax in self.axs:
            draw_dark(ax)
            ax.set_xmargin(0)

        base_fonts = plt.rcParams.get("font.size", 10)
        fs = base_fonts + PLOT_FONT_DELTA

        self.axs[0].set_title("MindsAI Filtered vs Raw EEG (Ch 0)", fontsize=fs)
        self.axs[0].set_xlabel("Time (s)", fontsize=fs); self.axs[0].set_ylabel("Amplitude (μV)", fontsize=fs)
        self.axs[1].set_title("All-Channel Avg (Raw vs Filtered)", fontsize=fs)
        self.axs[1].set_xlabel("Time (s)", fontsize=fs); self.axs[1].set_ylabel("Amplitude (μV)", fontsize=fs)

        self.lines_main = {
            "raw": self.axs[0].plot([], [], color=RAW_COLOR, label="Raw EEG")[0],
            "filt": self.axs[0].plot([], [], color=FILTERED_COLOR, label="MAI Filtered")[0],
            "diff": self.axs[0].plot([], [], color=RMV_NOISE_COLOR, linestyle="dotted", label="Removed Noise")[0],
        }

        handles0 = [self.lines_main["raw"], self.lines_main["filt"], self.lines_main["diff"]]
        self.legend0 = self.axs[0].legend(handles=handles0, loc="upper right", frameon=True)
        for t in self.legend0.get_texts(): t.set_color("white")
        self.legend0.get_frame().set_facecolor("black"); self.legend0.get_frame().set_edgecolor("white")

        self.lines_avg = {
            "raw": self.axs[1].plot([], [], color="#F3FF00", label="Raw Avg")[0],
            "filt": self.axs[1].plot([], [], color=FILTERED_COLOR, label="Filtered Avg")[0],
        }
        self.legend1 = self.axs[1].legend(loc="upper right", frameon=True)
        for t in self.legend1.get_texts(): t.set_color("white")
        self.legend1.get_frame().set_facecolor("black"); self.legend1.get_frame().set_edgecolor("white")

    # ---------- Console helpers ----------

    def _console_write(self, text: str, tag="info"):
        self.console.configure(state="normal")
        if self.console.index("end-1c") != "1.0":
            self.console.insert("end", "\n")
        self.console.insert("end", text, tag)
        self.console.see("end")
        self.console.configure(state="disabled")

    def _set_status_ok(self, msg: str):
        self.status.set(msg)
        self._console_write(msg, tag="ok")

    def _set_status_err(self, msg: str):
        self.status.set(msg)
        self._console_write(msg, tag="error")

    def _clear_placeholder(self):
        if self.fs_entry.get().strip().lower().startswith("enter sampling"):
            self.fs_entry.set("")

    def _update_lambda_label(self):
        e = int(self.lambda_exp.get())
        self.lam_label.config(text=f"1e-{e}")

    # ---------- Detection & file selection ----------

    def _try_update_detected_columns(self, initial=False):
        path = self.file_path.get().strip()
        if not path or not os.path.isfile(path):
            self.detect_label.config(text="Detected: — columns")
            self._edf_meta = None
            self._edf_is_input = False
            return
        try:
            ext = os.path.splitext(path)[1].lower()
            if ext == ".edf":
                meta = read_edf_full(path)
                self._edf_meta = meta
                self._edf_is_input = True
                n_cols = len(meta["labels"])
                # Default fs to majority fs
                try:
                    self.fs_entry.set(str(int(meta.get("majority_fs"))))
                except Exception:
                    pass
                delim = "<EDF>"
            else:
                arr, delim = read_numeric_csv(path)
                self._last_raw_matrix = arr
                n_cols = arr.shape[1]
                self._edf_meta = None
                self._edf_is_input = False

            # Default EEG selection: all channels
            self.col_range_str.set(f"1-{n_cols}")
            self.detect_label.config(text=f"Detected: {n_cols} columns — Are these all EEG?")

            if not initial:
                disp_delim = {
                    "\t": "\\t",
                    " ": "<space>",
                    ",": ",",
                    ";": ";",
                    "|": "|"
                }.get(delim, str(delim) if delim is not None else "<EDF>")
                self._console_write(
                    f"Detected total columns: {n_cols} (delimiter='{disp_delim}') — Are these all EEG?",
                    tag="info"
                )
        except Exception as e:
            self.detect_label.config(text="Detected: — columns")
            if not initial:
                self._console_write(str(e), tag="error")

    def browse_file(self):
        path = filedialog.askopenfilename(
            title="Select EEG EDF/CSV",
            initialdir=os.path.join(APP_DIR, "data"),
            initialfile="eeg.edf",
            filetypes=[("EDF / CSV / TSV / text", "*.edf *.csv *.tsv *.txt"),
                       ("CSV / TSV / text", "*.csv *.tsv *.txt"),
                       ("All files", "*.*")]
        )
        if path:
            self.file_path.set(path)
            self._try_update_detected_columns()

    def open_folder(self):
        p = os.path.dirname(self.file_path.get())
        if not p: p = os.getcwd()
        webbrowser.open(os.path.abspath(p))

    # ---------- Parsing helpers ----------

    def _get_fs(self):
        txt = self.fs_entry.get().strip()
        try:
            fs = float(txt)
        except Exception:
            raise ValueError(human_err("Sampling rate (fs) is required and must be numeric.", True))
        if not (MIN_FS <= fs <= MAX_FS):
            raise ValueError(human_err(f"fs must be between {MIN_FS} and {MAX_FS} Hz.", True))
        return int(round(fs))

    def _view_len_samples(self, fs: int, n_total: int) -> int:
        choice = (self.view_window.get() or "All").strip().lower()
        if choice == "all":
            return n_total
        try:
            seconds = float(choice.split()[0])
        except Exception:
            seconds = 1.0
        return max(1, int(round(seconds * fs)))

    def _filter_window_seconds(self) -> float:
        txt = (self.filter_window.get() or "30 s").strip().lower()
        try:
            return float(txt.split()[0])
        except Exception:
            return 30.0

    def _filter_window_len_samples(self, fs: int, n_total: int) -> int:
        sec = self._filter_window_seconds()
        win = max(1, int(round(sec * fs)))
        return min(win, n_total)

    # Accept a preferred start index to avoid snapping to end
    def _sync_timeline_bounds(self, prefer_start_idx: int = None):
        if self._last_t is None or self._last_fs is None:
            self.timeline.config(from_=0, to=0)
            self._timeline_var.set(0)
            return
        n_total = self._last_t.size
        fs = self._last_fs
        n_display = min(n_total, int(DISPLAY_MAX_SEC * fs))
        win = self._view_len_samples(fs, n_display)
        if win >= n_display:
            self.timeline.config(from_=0, to=0)
            self._timeline_var.set(0)
        else:
            self.timeline.config(from_=0, to=n_display - win)
            if prefer_start_idx is not None:
                start_idx = max(0, min(prefer_start_idx, n_display - win))
                self._timeline_var.set(start_idx)
            else:
                self._timeline_var.set(0)

    def _on_timeline_change(self):
        if self._last_t is None or self._raw_uV_cxt is None:
            return
        self._refresh_view_only(start_idx=int(self._timeline_var.get()))

    def _on_view_changed(self):
        self._sync_timeline_bounds()
        self._on_timeline_change()

    @staticmethod
    def _parse_column_selection(text: str, n_cols: int):
        """
        Parse something like:
          "1-8" or "1-21,23-26,28-31"
        into a sorted, unique 0-based index array.
        """
        s = (text or "").strip()
        if not s:
            raise ValueError(human_err("Column selection is empty. Use 1-based indices/ranges, e.g. 1-21,23-26.", True))

        tokens = [tok.strip() for tok in s.split(",") if tok.strip()]
        if not tokens:
            raise ValueError(human_err("Column selection is empty. Use 1-based indices/ranges, e.g. 1-21,23-26.", True))

        indices = []
        for tok in tokens:
            m = re.match(r"^(\d+)\s*[-:]\s*(\d+)$", tok)
            if m:
                a1 = int(m.group(1))
                b1 = int(m.group(2))
                if a1 > b1:
                    a1, b1 = b1, a1
                for k in range(a1, b1 + 1):
                    indices.append(k - 1)
                continue

            if tok.isdigit():
                k1 = int(tok)
                indices.append(k1 - 1)
                continue

            raise ValueError(human_err(
                f"Invalid token '{tok}' in column selection. Use 1-based indices or ranges like 1-8,10,12-16.",
                True
            ))

        idx = np.unique(np.array(indices, dtype=int))
        if idx.size == 0:
            raise ValueError(human_err("No valid columns selected.", True))
        if idx.min() < 0 or idx.max() >= n_cols:
            raise ValueError(human_err(
                f"Selected column index out of bounds. File has {n_cols} columns (1-based: 1–{n_cols}).",
                True
            ))
        if idx.size < MIN_CH or idx.size > MAX_CH:
            raise ValueError(human_err(
                f"Selected {idx.size} columns; require {MIN_CH}–{MAX_CH}.",
                True
            ))
        return idx

    # ---------- Core loading, filtering, metrics ----------

    def _load_and_validate(self, for_export=False):
        path = self.file_path.get().strip()
        if not path or not os.path.isfile(path):
            raise ValueError(human_err("File path is empty or does not exist.", True))

        base_dir  = os.path.dirname(path)
        base_name = os.path.splitext(os.path.basename(path))[0]
        ext = os.path.splitext(path)[1].lower()

        # Variables to fill: cxt (C x T), fs, flipped, selected indices
        flipped = False
        self._edf_eeg_indices = None
        self._edf_meta = None

        if ext == ".edf":
            # EDF path: keep all channels, filter only EEG subset at majority fs
            meta = read_edf_full(path)
            self._edf_meta = meta
            self._edf_is_input = True

            labels  = meta["labels"]
            fs_list = meta["fs_list"]
            sigs    = meta["signals_all"]
            nsamples = meta["nsamples"]
            maj_fs  = meta["majority_fs"]

            n_cols = len(labels)
            # Log channels that are NOT at the majority sampling rate
            for i, (lab, fsi) in enumerate(zip(labels, fs_list)):
                if fsi != maj_fs:
                    self._console_write(
                        f"[EDF] Channel {i+1} '{lab}' has fs={fsi} Hz (non-majority {maj_fs} Hz). "
                        f"Consider excluding this channel in the EEG column selection.",
                        tag="info"
                    )

            try:
                self.fs_entry.set(str(int(maj_fs)))
            except Exception:
                pass
            fs = self._get_fs()

            # User's selection refers to EDF channel indices (all channels)
            idx_all = self._parse_column_selection(self.col_range_str.get(), n_cols)

            # Keep only selected channels that match majority fs
            eeg_indices = []
            for ch_idx in idx_all:
                fsi = fs_list[ch_idx]
                if fsi != maj_fs:
                    self._console_write(
                        f"[EDF] Skipping channel {ch_idx+1} '{labels[ch_idx]}' at fs={fsi} Hz "
                        f"(non-majority {maj_fs} Hz) for MindsAI filtering.",
                        tag="info"
                    )
                    continue
                eeg_indices.append(ch_idx)

            if not eeg_indices:
                raise ValueError(human_err(
                    f"No selected channels are at the majority sampling rate ({maj_fs} Hz); nothing to filter.",
                    True
                ))

            # Build C x T matrix from EEG channels
            T = min(nsamples[ch_idx] for ch_idx in eeg_indices)
            if T <= 0:
                raise ValueError(human_err("EDF EEG channels appear to have zero length.", True))

            cxt = np.stack(
                [np.asarray(sigs[ch_idx][:T], dtype=float) for ch_idx in eeg_indices],
                axis=0
            )  # C x T

            num_ch = cxt.shape[0]
            dur = T / fs

            if not (MIN_CH <= num_ch <= MAX_CH):
                raise ValueError(human_err(f"Channels={num_ch} out of range ({MIN_CH}-{MAX_CH}).", True))
            if not (MIN_SEC <= dur <= MAX_SEC):
                raise ValueError(human_err(f"Duration={dur:.2f}s out of range ({MIN_SEC}-{MAX_SEC} s).", True))

            self._edf_eeg_indices = eeg_indices
            self._last_selected_indices = None
            flipped = False

        else:
            # CSV/TSV/TXT path: numeric matrix, then orientation detection
            if not (path.lower().endswith(".csv") or path.lower().endswith(".tsv") or path.lower().endswith(".txt")):
                raise ValueError(human_err("Only .csv / .tsv / .txt / .edf files are accepted.", True))
            fs = self._get_fs()
            arr_raw, delim = read_numeric_csv(path)
            self._last_raw_matrix = arr_raw
            self._last_write_delim = delim
            self._edf_is_input = False

            n_cols = arr_raw.shape[1]
            idx = self._parse_column_selection(self.col_range_str.get(), n_cols)
            self._last_selected_indices = idx

            arr_selected = arr_raw[:, idx]
            cxt, flipped = decide_orientation(arr_selected, fs)
            num_ch, T = cxt.shape
            dur = T / fs

            if not (MIN_CH <= num_ch <= MAX_CH):
                raise ValueError(human_err(f"Channels={num_ch} out of range ({MIN_CH}-{MAX_CH}).", True))
            if not (MIN_SEC <= dur <= MAX_SEC):
                raise ValueError(human_err(f"Duration={dur:.2f}s out of range ({MIN_SEC}-{MAX_SEC} s).", True))

        # cxt is now C x T, fs is chosen, flipped tells orientation for CSV export
        num_ch, T = cxt.shape
        dur = T / fs

        # Make sure plotting channel index is in range
        if self.channel_idx.get() < 0 or self.channel_idx.get() >= num_ch:
            self.channel_idx.set(0)

        # Unit detection
        median_abs = float(np.nanmedian(np.abs(cxt)))
        if median_abs > 1e4:
            self._unit_scale_in = 1e-9
            self._unit_name_in  = "nV"
        else:
            self._unit_scale_in = 1e-6
            self._unit_name_in  = "µV"

        self._console_write(
            f"[Units] Detected input: {self._unit_name_in} (scale {self._unit_scale_in:g} → V)",
            tag="info"
        )

        # Convert to volts, DC-center per channel
        raw_volts_cxt = cxt * self._unit_scale_in
        raw_volts_cxt = raw_volts_cxt - np.mean(raw_volts_cxt, axis=1, keepdims=True)

        lam_exp = int(self.lambda_exp.get())
        lam = 10.0 ** (-lam_exp)

        # Chunked filtering across full recording
        T = raw_volts_cxt.shape[1]
        win_samples = self._filter_window_len_samples(fs, T)
        filt_volts_cxt = np.zeros_like(raw_volts_cxt)
        start = 0
        n_chunks = 0
        try:
            while start < T:
                end = min(start + win_samples, T)
                seg = raw_volts_cxt[:, start:end]
                seg_filt = mai.mindsai_python_filter(seg, lam)
                filt_volts_cxt[:, start:end] = seg_filt
                start = end
                n_chunks += 1
        except Exception as e:
            raise RuntimeError(f"MindsAI filter error during chunked filtering: {e}")

        filt_win_sec = self._filter_window_seconds()
        self._console_write(
            f"[Filter] Applied MindsAI in {n_chunks} window(s) of ~{filt_win_sec:.2f} s over {dur:.2f} s total.",
            tag="info"
        )

        # For metrics / plotting, convert to µV
        raw_uV_cxt  = raw_volts_cxt  / 1e-6
        filt_uV_cxt = filt_volts_cxt / 1e-6

        ch_plot = int(self.channel_idx.get())
        method = self.snr_method.get()
        metrics = compute_metrics(raw_uV_cxt, filt_uV_cxt, method, ch_plot, fs)
        metrics["lambda"] = lam
        metrics["filter_window_sec"] = float(filt_win_sec)
        metrics["filter_chunks"] = int(n_chunks)

        # Persist state for plotting/export
        self._raw_input_orientation_timepoints_by_channels = flipped
        self._raw_uV_cxt = raw_uV_cxt
        self._filt_uV_cxt = filt_uV_cxt
        self._last_metrics = metrics
        self._last_base_dir = base_dir
        self._last_base_name = base_name
        self._last_fs = fs
        self._last_lambda = lam
        self._last_channel = ch_plot
        self._last_t = np.arange(T) / fs

        verdict = f"Accepted: {num_ch} ch, fs={fs} Hz, {dur:.2f} s, λ=1e-{lam_exp}"
        if for_export:
            verdict += " — ready to export."
        self._set_status_ok(verdict)

        return True

    # ---------- Plotting / view ----------

    def _slice_for_view(self, start_idx: int):
        n_total = self._last_t.size
        fs = self._last_fs
        n_display = min(n_total, int(DISPLAY_MAX_SEC * fs))
        win = self._view_len_samples(fs, n_display)
        i0 = max(0, min(start_idx, max(0, n_display - win)))
        i1 = min(i0 + win, n_display)
        t = self._last_t[i0:i1]
        ch = self._last_channel
        raw = self._raw_uV_cxt[ch, i0:i1]
        filt = self._filt_uV_cxt[ch, i0:i1]
        diff = raw - filt
        avg_raw  = np.mean(self._raw_uV_cxt[:, i0:i1], axis=0)
        avg_filt = np.mean(self._filt_uV_cxt[:, i0:i1], axis=0)
        return i0, i1, t, raw, filt, diff, avg_raw, avg_filt

    def _refresh_view_only(self, start_idx: int = None):
        try:
            if self._last_t is None or self._raw_uV_cxt is None or self._filt_uV_cxt is None:
                return
            n = self._last_t.size
            if start_idx is None:
                start_idx = int(self._timeline_var.get())
            start_idx = max(0, min(start_idx, n - 1))

            i0, i1, t, raw, filt, diff, avg_raw, avg_filt = self._slice_for_view(start_idx)

            self.lines_main["raw"].set_data(t, raw)
            self.lines_main["filt"].set_data(t, filt)
            self.lines_main["diff"].set_data(t, diff)

            self.lines_avg["raw"].set_data(t, avg_raw)
            self.lines_avg["filt"].set_data(t, avg_filt)

            ch = self._last_channel
            base_fonts = plt.rcParams.get("font.size", 10)
            fs = base_fonts + PLOT_FONT_DELTA
            self.axs[0].set_title(f"MindsAI Filtered vs Raw EEG (Ch {ch})", fontsize=fs)

            clip = 1.0 if (t.size and (t[-1] - t[0] <= 5.0)) else None
            _apply_view_limits(self.axs[0], t, [raw, filt, diff], clip_pct=clip)
            _apply_view_limits(self.axs[1], t, [avg_raw, avg_filt], clip_pct=clip)

            self.canvas.draw_idle()

        except Exception as e:
            self._set_status_err(str(e))
            traceback.print_exc()

    # ---------- Actions ----------

    def process(self):
        try:
            prev_t0_sec = None
            if (self._last_t is not None) and (self._last_fs is not None):
                prev_idx = int(self._timeline_var.get())
                prev_t0_sec = prev_idx / float(self._last_fs)

            ok = self._load_and_validate(for_export=False)
            if not ok:
                return

            prefer_idx = None
            if prev_t0_sec is not None and self._last_fs is not None and self._last_t is not None:
                prefer_idx = int(round(prev_t0_sec * float(self._last_fs)))

            self._sync_timeline_bounds(prefer_start_idx=prefer_idx)
            self._on_timeline_change()

            print_metrics_console(self._last_metrics)
        except Exception as e:
            self._set_status_err(str(e))
            traceback.print_exc()

    def export_outputs(self):
        try:
            prev_t0_sec = None
            if (self._last_t is not None) and (self._last_fs is not None):
                prev_idx = int(self._timeline_var.get())
                prev_t0_sec = prev_idx / float(self._last_fs)

            ok = self._load_and_validate(for_export=True)
            if not ok:
                return

            prefer_idx = None
            if prev_t0_sec is not None and self._last_fs is not None and self._last_t is not None:
                prefer_idx = int(round(prev_t0_sec * float(self._last_fs)))

            self._sync_timeline_bounds(prefer_start_idx=prefer_idx)
            self._on_timeline_change()

            lam_exp = int(self.lambda_exp.get())
            lam_str = f"1e-{lam_exp}"

            path = self.file_path.get().strip()
            ext = os.path.splitext(path)[1].lower()

            if ext == ".edf":
                if not self._edf_meta or self._edf_eeg_indices is None:
                    raise RuntimeError("EDF metadata or EEG indices missing; cannot export EDF.")
                # _filt_uV_cxt is in µV; convert to volts for export helper
                filt_cxt_volts = self._filt_uV_cxt * 1e-6
                out_edf, out_json = save_filtered_edf_and_metrics(
                    edf_path=path,
                    meta=self._edf_meta,
                    eeg_indices=self._edf_eeg_indices,
                    filt_cxt_volts=filt_cxt_volts,
                    unit_scale_in=self._unit_scale_in,
                    metrics=self._last_metrics,
                    lam_str=lam_str
                )
                self._set_status_ok(
                    f"Exported EDF:\n  {os.path.basename(out_edf)}\n  {os.path.basename(out_json)}"
                )
            else:
                if self._last_raw_matrix is None or self._last_selected_indices is None:
                    raise RuntimeError("No CSV/TSV/TXT input matrix or selected indices for export.")
                filt_cxt_volts = self._filt_uV_cxt * 1e-6
                fpath, mpath = save_filtered_and_metrics_same_format(
                    base_dir=self._last_base_dir,
                    base_name=self._last_base_name,
                    lam_str=lam_str,
                    full_input_numeric_rows_cols=self._last_raw_matrix,
                    selected_indices_0based=self._last_selected_indices,
                    filt_cxt_volts=filt_cxt_volts,
                    flipped=self._raw_input_orientation_timepoints_by_channels,
                    unit_scale_in=self._unit_scale_in,
                    write_delim=self._last_write_delim,
                    metrics=self._last_metrics
                )
                self._set_status_ok(
                    f"Exported CSV:\n  {os.path.basename(fpath)}\n  {os.path.basename(mpath)}"
                )

            print_metrics_console(self._last_metrics)

        except Exception as e:
            self._set_status_err(str(e))
            traceback.print_exc()


# ======================= MAIN =======================

if __name__ == "__main__":
    app = App()
    app.mainloop()
