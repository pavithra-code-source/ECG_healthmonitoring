import time
import numpy as np

from ecg_cnn_inference import load_interpreter, classify_beat
from alert_logging import get_hr_risk_level


X = np.load("X_ecg.npy")

interpreter = load_interpreter()

TRIALS = 20

latencies = []

print("ECG End-to-End Performance Test")
print("================================")
print(f"Trials: {TRIALS}")


for i in range(TRIALS):

    beat = X[0].flatten()

    start = time.perf_counter()

    # AI classification
    label, confidence = classify_beat(
        interpreter,
        beat,
        already_normalized=True
    )

    # HR risk evaluation
    hr = 70
    risk_level, risk_status = get_hr_risk_level(hr)

    # Alert decision
    abnormal = (label == "Arrhythmia")

    # Logging operation
    timestamp = time.time()

    elapsed = (time.perf_counter() - start) * 1000

    latencies.append(elapsed)


latencies = np.array(latencies)

print()
print("END-TO-END RESULTS")
print("-------------------")
print("Average:", round(np.mean(latencies), 2), "ms")
print("Minimum:", round(np.min(latencies), 2), "ms")
print("Maximum:", round(np.max(latencies), 2), "ms")
print("P95:", round(np.percentile(latencies, 95), 2), "ms")