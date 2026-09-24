"""
ECG Signal Analysis -- Clinical Parameter Extraction

This module performs clinical ECG analysis using NeuroKit2.

The CNN uses a 187-sample ECG beat.
For clinical analysis, a longer ECG signal is required,
so short signals are repeated before NeuroKit2 processing.

Parameters:
    HR
    RR interval
    RR variability
    PR interval
    QRS duration
    QT interval
    QTc
    ST segment
"""

import numpy as np
import neurokit2 as nk


# ============================================================
# CONFIGURATION
# ============================================================

FS = 360
BEAT_SAMPLES = 187
CLINICAL_REPETITIONS = 20


# ============================================================
# PREPARE ECG SIGNAL
# ============================================================

def prepare_clinical_signal(ecg_signal):
    """
    Prepare an ECG signal for clinical analysis.

    If the input is a short 187-sample beat,
    repeat it to create a longer signal.
    """

    ecg_signal = np.asarray(
        ecg_signal,
        dtype=float
    ).flatten()

    # Remove invalid values
    ecg_signal = np.nan_to_num(
        ecg_signal,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # Check for empty signal
    if len(ecg_signal) == 0:
        return np.array([])

    # If signal is too short, repeat it
    if len(ecg_signal) < FS * 3:

        repetitions = max(
            CLINICAL_REPETITIONS,
            int(np.ceil((FS * 5) / len(ecg_signal)))
        )

        clinical_signal = np.tile(
            ecg_signal,
            repetitions
        )

    else:

        clinical_signal = ecg_signal

    return clinical_signal


# ============================================================
# MAIN ECG ANALYSIS
# ============================================================

def analyze_ecg_signal(
    ecg_signal,
    sampling_rate=FS
):
    """
    Analyze ECG signal using NeuroKit2.

    Returns clinical ECG parameters as a dictionary.
    """

    # Prepare signal
    clinical_signal = prepare_clinical_signal(
        ecg_signal
    )

    # Empty signal protection
    if len(clinical_signal) == 0:

        return {
            "HR_bpm": None,
            "RR_interval_ms": None,
            "RR_variability_ms": None,
            "PR_interval_ms": None,
            "QRS_duration_ms": None,
            "QT_interval_ms": None,
            "QTc_ms": None,
            "ST_segment_ms": None,
            "num_beats_detected": 0,
        }

    # Run NeuroKit2
    try:

        signals, info = nk.ecg_process(
            clinical_signal,
            sampling_rate=sampling_rate
        )

    except Exception as e:

        print(
            f"[WARNING] NeuroKit2 ECG analysis failed: {e}"
        )

        return {
            "HR_bpm": None,
            "RR_interval_ms": None,
            "RR_variability_ms": None,
            "PR_interval_ms": None,
            "QRS_duration_ms": None,
            "QT_interval_ms": None,
            "QTc_ms": None,
            "ST_segment_ms": None,
            "num_beats_detected": 0,
        }

    return _extract_features(
        info,
        sampling_rate
    )


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def _extract_features(info, fs):

    r_peaks = np.array(
        info.get("ECG_R_Peaks", []),
        dtype=float
    )

    p_onsets = np.array(
        info.get("ECG_P_Onsets", []),
        dtype=float
    )

    r_onsets = np.array(
        info.get("ECG_R_Onsets", []),
        dtype=float
    )

    r_offsets = np.array(
        info.get("ECG_R_Offsets", []),
        dtype=float
    )

    t_offsets = np.array(
        info.get("ECG_T_Offsets", []),
        dtype=float
    )

    t_onsets = np.array(
        info.get("ECG_T_Onsets", []),
        dtype=float
    )

    # Remove invalid values
    r_peaks = r_peaks[np.isfinite(r_peaks)]
    p_onsets = p_onsets[np.isfinite(p_onsets)]
    r_onsets = r_onsets[np.isfinite(r_onsets)]
    r_offsets = r_offsets[np.isfinite(r_offsets)]
    t_offsets = t_offsets[np.isfinite(t_offsets)]
    t_onsets = t_onsets[np.isfinite(t_onsets)]

    # Need at least two R peaks
    if len(r_peaks) < 2:

        print(
            "[WARNING] Too few R-peaks detected."
        )

        return {
            "HR_bpm": None,
            "RR_interval_ms": None,
            "RR_variability_ms": None,
            "PR_interval_ms": None,
            "QRS_duration_ms": None,
            "QT_interval_ms": None,
            "QTc_ms": None,
            "ST_segment_ms": None,
            "num_beats_detected": len(r_peaks),
        }

    # Keep same number of beats for all arrays
    n = min(
        len(r_peaks),
        len(p_onsets),
        len(r_onsets),
        len(r_offsets),
        len(t_onsets),
        len(t_offsets)
    )

    if n == 0:

        return {
            "HR_bpm": None,
            "RR_interval_ms": None,
            "RR_variability_ms": None,
            "PR_interval_ms": None,
            "QRS_duration_ms": None,
            "QT_interval_ms": None,
            "QTc_ms": None,
            "ST_segment_ms": None,
            "num_beats_detected": 0,
        }

    # Trim all arrays
    r_peaks = r_peaks[:n]
    p_onsets = p_onsets[:n]
    r_onsets = r_onsets[:n]
    r_offsets = r_offsets[:n]
    t_onsets = t_onsets[:n]
    t_offsets = t_offsets[:n]

    # RR interval
    rr_ms = (
        np.diff(r_peaks)
        / fs
        * 1000
    )

    # Heart rate
    if len(rr_ms):

        hr_bpm = 60000 / rr_ms

    else:

        hr_bpm = np.array([])

    # PR interval
    pr_ms = (
        (r_onsets - p_onsets)
        / fs
        * 1000
    )

    # QRS duration
    qrs_ms = (
        (r_offsets - r_onsets)
        / fs
        * 1000
    )

    # QT interval
    qt_ms = (
        (t_offsets - r_onsets)
        / fs
        * 1000
    )

    # ST segment
    st_ms = (
        (t_onsets - r_offsets)
        / fs
        * 1000
    )

    # QTc using Bazett's formula
    if len(rr_ms):

        rr_for_qtc = np.append(
            rr_ms,
            rr_ms[-1]
        )

        qtc_ms = (
            qt_ms
            / np.sqrt(rr_for_qtc / 1000)
        )

    else:

        qtc_ms = np.array([])

    # Safe mean function
    def safe_mean(values):

        values = np.asarray(
            values,
            dtype=float
        )

        values = values[
            np.isfinite(values)
        ]

        if len(values) == 0:
            return None

        return round(
            float(np.mean(values)),
            1
        )

    # Return results
    return {

        "HR_bpm":
            safe_mean(hr_bpm),

        "RR_interval_ms":
            safe_mean(rr_ms),

        "RR_variability_ms":
            round(
                float(np.std(rr_ms)),
                1
            ) if len(rr_ms) else None,

        "PR_interval_ms":
            safe_mean(pr_ms),

        "QRS_duration_ms":
            safe_mean(qrs_ms),

        "QT_interval_ms":
            safe_mean(qt_ms),

        "QTc_ms":
            safe_mean(qtc_ms),

        "ST_segment_ms":
            safe_mean(st_ms),

        "num_beats_detected":
            n,
    }


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("======================================")
    print(" ECG CLINICAL ANALYSIS TEST")
    print("======================================")
    print()

    test_signal = nk.ecg_simulate(
        duration=10,
        sampling_rate=FS,
        heart_rate=75,
        noise=0.02
    )

    result = analyze_ecg_signal(
        test_signal
    )

    print("Clinical parameters:")

    for key, value in result.items():

        print(
            f"{key}: {value}"
        )