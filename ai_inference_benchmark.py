import time
import numpy as np
import tensorflow as tf

MODEL_PATH = "ecg_model.tflite"
DATA_PATH = "X_ecg.npy"

TRIALS = 20


print("ECG AI Inference Latency Benchmark")
print("===================================")

# Load ECG dataset
X = np.load(DATA_PATH)

print("Dataset shape:", X.shape)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_index = input_details[0]["index"]
output_index = output_details[0]["index"]

# Use one ECG beat
beat = X[0].astype(np.float32)

# Make input shape = (1, 187, 1)
input_data = beat.reshape(1, 187, 1)

# Warm-up inference
interpreter.set_tensor(input_index, input_data)
interpreter.invoke()

times = []

# Measure ONLY AI inference time
for i in range(TRIALS):

    interpreter.set_tensor(input_index, input_data)

    start = time.perf_counter()

    interpreter.invoke()

    end = time.perf_counter()

    latency_ms = (end - start) * 1000
    times.append(latency_ms)

times = np.array(times)

print()
print("Trials:", TRIALS)
print()
print("AI INFERENCE LATENCY")
print("--------------------")
print("Average:", round(np.mean(times), 4), "ms")
print("Minimum:", round(np.min(times), 4), "ms")
print("Maximum:", round(np.max(times), 4), "ms")
print("P95:", round(np.percentile(times, 95), 4), "ms")