import sys
import time
import numpy as np

from ecg_cnn_inference import load_interpreter, classify_beat


# --------------------------------------------------
# DEMO SETTINGS
# --------------------------------------------------

CLOUD_DELAY_SEC = 0.3

# Correct ECG samples
NORMAL_SAMPLE_INDEX = 0
ARRHYTHMIA_SAMPLE_INDEX = 341


# --------------------------------------------------
# CONDITION SETTINGS
# --------------------------------------------------

conditions = {
    "normal": {
        "hr": 75,
        "risk": "GREEN",
        "status": "Normal",
        "sample": NORMAL_SAMPLE_INDEX
    },

    "bradycardia": {
        "hr": 50,
        "risk": "YELLOW",
        "status": "Caution",
        "sample": ARRHYTHMIA_SAMPLE_INDEX
    },

    "tachycardia": {
        "hr": 110,
        "risk": "YELLOW",
        "status": "Caution",
        "sample": ARRHYTHMIA_SAMPLE_INDEX
    },

    "highrisk": {
        "hr": 130,
        "risk": "ORANGE",
        "status": "High Risk",
        "sample": ARRHYTHMIA_SAMPLE_INDEX
    },

    "critical": {
        "hr": 150,
        "risk": "RED",
        "status": "Critical",
        "sample": ARRHYTHMIA_SAMPLE_INDEX
    }
}


# --------------------------------------------------
# NETWORK
# --------------------------------------------------

def cloud_ai_predict(interpreter, beat):

    time.sleep(CLOUD_DELAY_SEC)

    label, confidence = classify_beat(
        interpreter,
        beat,
        already_normalized=True
    )

    return label, confidence, "CLOUD"


def edge_ai_predict(interpreter, beat):

    label, confidence = classify_beat(
        interpreter,
        beat,
        already_normalized=True
    )

    return label, confidence, "EDGE"


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if len(sys.argv) != 3:

    print("Usage:")
    print("python network_check.py <condition> <network>")
    print()
    print("Conditions:")
    print("normal")
    print("bradycardia")
    print("tachycardia")
    print("highrisk")
    print("critical")
    print()
    print("Network:")
    print("online")
    print("offline")

    sys.exit()


condition = sys.argv[1].lower()
network = sys.argv[2].lower()


if condition not in conditions:

    print("Invalid condition")
    sys.exit()


if network not in ["online", "offline"]:

    print("Invalid network")
    sys.exit()


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

X = np.load("X_ecg.npy")

config = conditions[condition]

hr = config["hr"]
risk = config["risk"]
status = config["status"]
sample_index = config["sample"]

beat = X[sample_index].flatten()

interpreter = load_interpreter()


# --------------------------------------------------
# AI SELECTION
# --------------------------------------------------

start = time.perf_counter()

if network == "online":

    prediction, confidence, source = cloud_ai_predict(
        interpreter,
        beat
    )

else:

    prediction, confidence, source = edge_ai_predict(
        interpreter,
        beat
    )

elapsed_ms = (time.perf_counter() - start) * 1000


# --------------------------------------------------
# OUTPUT
# --------------------------------------------------

print()
print("========================================")
print("       ECG HEALTH MONITORING DEMO")
print("========================================")

print("Condition     :", condition.upper())
print("Network       :", network.upper())
print("AI Source     :", source)
print("Heart Rate    :", hr, "BPM")
print("AI Prediction :", prediction.upper())
print("Confidence    :", round(confidence, 2), "%")
print("Risk Level    :", risk)
print("Status        :", status)
print("Total Time    :", round(elapsed_ms, 2), "ms")

print("========================================")