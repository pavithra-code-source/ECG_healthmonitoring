import time
import os
import psutil
import numpy as np

from ecg_cnn_inference import load_interpreter, classify_beat


X = np.load("X_ecg.npy")
interpreter = load_interpreter()

beat = X[0].flatten()

process = psutil.Process(os.getpid())

print("ECG CPU Usage Test")
print("==================")

# Start CPU measurement
process.cpu_percent(interval=None)

start = time.perf_counter()

# Longer workload so CPU usage can be measured
for i in range(5000):
    classify_beat(
        interpreter,
        beat,
        already_normalized=True
    )

elapsed = (time.perf_counter() - start)

# Measure CPU usage during the workload
cpu_usage = process.cpu_percent(interval=0.1)

print()
print("CPU Usage:", cpu_usage, "%")
print("5000 Inferences Time:", round(elapsed * 1000, 2), "ms")
print("Average Inference Time:", round((elapsed * 1000) / 5000, 4), "ms")