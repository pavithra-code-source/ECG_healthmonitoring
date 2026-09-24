import time
import os
import psutil
import numpy as np

from ecg_cnn_inference import load_interpreter, classify_beat


X = np.load("X_ecg.npy")
interpreter = load_interpreter()

beat = X[0].flatten()

process = psutil.Process(os.getpid())

print("ECG Resource Usage Test")
print("=======================")


# Initial memory
memory_before = process.memory_info().rss / (1024 * 1024)

# CPU measurement
cpu_before = process.cpu_percent(interval=None)

start = time.perf_counter()

# Run ECG AI inference 20 times
for i in range(20):
    label, confidence = classify_beat(
        interpreter,
        beat,
        already_normalized=True
    )

elapsed = (time.perf_counter() - start) * 1000

# CPU measurement
cpu_after = process.cpu_percent(interval=0.1)

# Final memory
memory_after = process.memory_info().rss / (1024 * 1024)


print()
print("CPU Usage:", cpu_after, "%")
print("Memory Before:", round(memory_before, 2), "MB")
print("Memory After:", round(memory_after, 2), "MB")
print("Memory Increase:", round(memory_after - memory_before, 2), "MB")
print("20 Inferences Time:", round(elapsed, 2), "ms")
print("Average Inference Time:", round(elapsed / 20, 2), "ms")