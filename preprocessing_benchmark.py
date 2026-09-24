import numpy as np
import time

ECG_BUFFER_SIZE = 187
TRIALS = 20


def generate_ecg():
    ecg = np.full(ECG_BUFFER_SIZE, 500, dtype=np.float32)

    ecg[25:36] = 530
    ecg[58:63] = 470
    ecg[68:73] = 800
    ecg[78:83] = 450
    ecg[120:136] = 550

    return ecg


def preprocess_ecg(input_ecg):
    mean = np.mean(input_ecg)

    variance = np.mean((input_ecg - mean) ** 2)

    std_dev = np.sqrt(variance)

    if std_dev != 0:
        output = (input_ecg - mean) / std_dev
    else:
        output = np.zeros(ECG_BUFFER_SIZE, dtype=np.float32)

    return output


ecg = generate_ecg()

times = []

for i in range(TRIALS):

    start = time.perf_counter_ns()

    normalized = preprocess_ecg(ecg)

    end = time.perf_counter_ns()

    elapsed_us = (end - start) / 1000.0

    times.append(elapsed_us)


times = np.array(times)

average = np.mean(times)
minimum = np.min(times)
maximum = np.max(times)
p95 = np.percentile(times, 95)


print()
print("========================================")
print("     NO-QNX ECG PREPROCESSING BENCHMARK")
print("========================================")

print("ECG samples :", ECG_BUFFER_SIZE)
print("Trials      :", TRIALS)

print()
print("PREPROCESSING RESULTS")
print("---------------------")

print("Average :", round(average, 2), "microseconds")
print("Minimum :", round(minimum, 2), "microseconds")
print("Maximum :", round(maximum, 2), "microseconds")
print("P95     :", round(p95, 2), "microseconds")

print()
print("First normalized sample :", round(float(normalized[0]), 4))

print()
print("========================================")
print("Benchmark completed.")
print("========================================")