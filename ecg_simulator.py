
"""
ECG + Vitals Simulation Module (No-QNX baseline)
 
Cycles through normal / bradycardia / tachycardia / critical states,
pulling a REAL matching beat from the full dataset (X_ecg.npy / y_ecg.npy)
for each state, so the CNN's prediction always matches the intended state.
"""
 
import numpy as np
import random
 
FS = 360          # sampling rate (Hz) — matches MIT-BIH, and your CNN's training data
BEAT_SAMPLES = 187  # matches your CNN's input length
 
 
def simulate_vitals(state="normal"):
    """
    Returns (hr_bpm, temp_celsius) for a given state.
    state: 'normal' | 'tachycardia' | 'bradycardia' | 'critical'
    """
    if state == "tachycardia":
        hr = random.randint(120, 139)
    elif state == "bradycardia":
        hr = random.randint(30, 59)
    elif state == "critical":
        hr = random.randint(140, 160)
    else:
        hr = random.randint(60, 99)
 
    temp = round(random.uniform(36.1, 37.2), 1)
    return hr, temp
 
 
def generate_sample(state="normal", X=None, y=None):
    """
    Full sample generator for one 'tick' of the pipeline.
    Pulls a REAL beat matching the intended state from the full dataset.
    Returns: (ecg_waveform [187 samples], hr, temp, ground_truth_label)
    """
    hr, temp = simulate_vitals(state)
 
    wanted_label = 0 if state == "normal" else 1
 
    matching_indices = np.where(y == wanted_label)[0]
    idx = random.choice(matching_indices)
 
    # TEMPORARY DEBUG PRINT — remove before your live demo
    print(f"[DEBUG] state={state} idx={idx} out of {len(X)}")
 
    ecg = X[idx].flatten()
    label = "Normal" if wanted_label == 0 else "Abnormal"
 
    return ecg, hr, temp, label
 
 
if __name__ == "__main__":
    # Quick sanity check — run this file directly to see sample output
    X = np.load("X_ecg.npy")
    y = np.load("y_ecg.npy")
 
    for s in ["normal", "tachycardia", "bradycardia", "critical"]:
        ecg, hr, temp, label = generate_sample(s, X, y)
        print(f"State: {s:12s} | HR: {hr:3d} bpm | Temp: {temp}C | "
              f"Label: {label:8s} | ECG shape: {ecg.shape}")
 
