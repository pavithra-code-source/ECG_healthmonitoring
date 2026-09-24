import time
import numpy as np

from ecg_cnn_inference import load_interpreter, classify_beat
from network_check import local_edge_predict, cloud_ai_predict


X = np.load("X_ecg.npy")
interpreter = load_interpreter()

# Same ECG beat for all trials
beat = X[0].flatten()

TRIALS = 20

edge_times = []
cloud_times = []


print("ECG Performance Measurement")
print("===========================")
print(f"Trials: {TRIALS}")


# =============================
# EDGE-AI TEST
# =============================

for i in range(TRIALS):

    start = time.perf_counter()

    result = local_edge_predict(
        interpreter,
        beat
    )

    elapsed = (time.perf_counter() - start) * 1000

    edge_times.append(elapsed)


# =============================
# CLOUD-AI TEST
# =============================

for i in range(TRIALS):

    start = time.perf_counter()

    result = cloud_ai_predict(
        interpreter,
        beat
    )

    elapsed = (time.perf_counter() - start) * 1000

    cloud_times.append(elapsed)


# =============================
# STATISTICS
# =============================

edge_times = np.array(edge_times)
cloud_times = np.array(cloud_times)

print()
print("EDGE-AI RESULTS")
print("----------------")
print("Average:", round(np.mean(edge_times), 2), "ms")
print("Minimum:", round(np.min(edge_times), 2), "ms")
print("Maximum:", round(np.max(edge_times), 2), "ms")
print("P95:", round(np.percentile(edge_times, 95), 2), "ms")


print()
print("CLOUD-AI (SIMULATED) RESULTS")
print("-----------------------------")
print("Average:", round(np.mean(cloud_times), 2), "ms")
print("Minimum:", round(np.min(cloud_times), 2), "ms")
print("Maximum:", round(np.max(cloud_times), 2), "ms")
print("P95:", round(np.percentile(cloud_times, 95), 2), "ms")