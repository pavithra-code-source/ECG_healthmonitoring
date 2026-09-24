import numpy as np
import time

# Load held-out test ECG data
X_test = np.load("benchmark_test_beats.npy")
y_test = np.load("benchmark_test_labels.npy")

print("Real-time ECG simulation started")
print("Number of ECG beats:", len(X_test))

for i in range(len(X_test)):

    beat = X_test[i].flatten()
    true_label = y_test[i]

    if true_label == 0:
        label = "Normal"
    else:
        label = "Arrhythmia"

    print(
        f"Beat {i + 1:02d} | "
        f"Samples: {len(beat)} | "
        f"True Label: {label}"
    )

    # Simulate real-time arrival
    time.sleep(1)