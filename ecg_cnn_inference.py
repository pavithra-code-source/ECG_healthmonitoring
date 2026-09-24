"""
Step 4: 1D CNN TFLite Inference - Non-QNX Baseline

Loads the trained ECG TFLite model and classifies
a real 187-sample ECG beat as Normal or Arrhythmia.

Training preprocessing:
Per-beat Z-score normalization
(mean and std calculated separately for each beat)
"""

import numpy as np
import tensorflow as tf


MODEL_PATH = "ecg_model.tflite"
DATA_PATH = "X_ecg.npy"
LABEL_PATH = "y_ecg.npy"

LABELS = {
    0: "Normal",
    1: "Arrhythmia"
}


def load_interpreter(model_path=MODEL_PATH):
    """Load the TFLite ECG model."""

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    return interpreter


def normalize_beat(beat):
    """
    Apply the exact normalization used during training:

        (beat - mean) / std
    """

    beat = np.asarray(beat, dtype=np.float32)

    mean = np.mean(beat)
    std = np.std(beat)

    if std == 0:
        raise ValueError("ECG beat has zero standard deviation.")

    beat = (beat - mean) / std

    return beat


def classify_beat(interpreter, beat, already_normalized=False):
    """
    Run one 187-sample ECG beat through the TFLite model.
    """

    beat = np.asarray(beat, dtype=np.float32)

    # Make sure exactly 187 samples are present
    if beat.size != 187:
        raise ValueError(
            f"Expected 187 ECG samples, but got {beat.size}."
        )

    # X_ecg.npy is already normalized during data preparation.
    # Raw ECG beats should be normalized here.
    if not already_normalized:
        beat = normalize_beat(beat)

    # CNN input shape = (batch, samples, channel)
    input_data = beat.reshape(1, 187, 1).astype(np.float32)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Send ECG beat to TFLite model
    interpreter.set_tensor(
        input_details[0]["index"],
        input_data
    )

    # Run inference
    interpreter.invoke()

    # Get arrhythmia probability
    probability = interpreter.get_tensor(
        output_details[0]["index"]
    )[0][0]

    # Convert probability to label
    if probability >= 0.5:
        label = "Arrhythmia"
        confidence = probability * 100
    else:
        label = "Normal"
        confidence = (1 - probability) * 100

    return label, confidence


if __name__ == "__main__":

    print("ECG 1D CNN TFLite Inference")
    print("============================")

    # Load prepared ECG dataset
    X = np.load(DATA_PATH)
    y = np.load(LABEL_PATH)

    print("Dataset shape:", X.shape)
    print("Labels shape:", y.shape)

    # Load TFLite model
    interpreter = load_interpreter()

    # Select one real ECG beat
    idx = 6

    beat = X[idx].flatten()

    # True label from dataset
    true_label = LABELS[int(y[idx])]

    # X_ecg.npy is already normalized
    predicted_label, confidence = classify_beat(
        interpreter,
        beat,
        already_normalized=True
    )

    print()
    print("Sample index:", idx)
    print("True label:", true_label)
    print("Predicted:", predicted_label)
    print("Confidence:", round(confidence, 2), "%")