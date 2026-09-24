from flask import Flask, jsonify, render_template
import threading
import time
import os
import smtplib
from email.mime.text import MIMEText
from datetime import datetime

import numpy as np

from ecg_simulator import generate_sample
from ecg_cnn_inference import load_interpreter, classify_beat
from clinical_analysis import analyze_ecg_signal


app = Flask(__name__)
# ==========================================
# EMAIL ALERT CONFIG
# ==========================================

SENDER_EMAIL = "tina.tonic08@gmail.com"
SENDER_APP_PASSWORD = os.environ.get("EMAIL_APP_PASSWORD")
DOCTOR_EMAIL = "tina.tonic08@gmail.com"

ALERT_COOLDOWN_SECONDS = 10
last_alert_time = 0


def send_critical_alert(hr, prediction, confidence):
    global last_alert_time

    now = time.time()

    # Avoid sending email repeatedly
    if now - last_alert_time < ALERT_COOLDOWN_SECONDS:
        print("[INFO] Email alert skipped - cooldown active.")
        return

    subject = f"CRITICAL ECG ALERT - Patient {monitoring_data['patient_id']}"

    body = (
        f"Critical condition detected.\n\n"
        f"Patient ID: {monitoring_data['patient_id']}\n"
        f"Heart Rate: {hr} BPM\n"
        f"AI Prediction: {prediction}\n"
        f"Confidence: {confidence}%\n"
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"This is an automated alert from the ECG Monitoring System."
    )

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = SENDER_EMAIL
    msg["To"] = DOCTOR_EMAIL

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, SENDER_APP_PASSWORD)

            server.sendmail(
                SENDER_EMAIL,
                DOCTOR_EMAIL,
                msg.as_string()
            )

        last_alert_time = now
        print("EMAIL ALERT SENT")

    except Exception as e:
        print(f"[WARNING] Failed to send alert email: {e}")

# ==========================================
# DEMO MODE
# ==========================================

demo_mode = False


# ==========================================
# SHARED MONITORING DATA
# ==========================================

monitoring_data = {
    "patient_id": "P001",
    "heart_rate": 75,
    "prediction": "NORMAL",
    "confidence": 99.99,
    "risk_level": "GREEN",
    "status": "Normal",

    "rr_interval": None,
    "rr_variability": None,
    "pr_interval": None,
    "qrs_duration": None,
    "qt_interval": None,
    "qtc": None,
    "st_segment": None,

    "timestamp": ""
}


# ==========================================
# RISK LEVEL LOGIC
# ==========================================

def get_risk_level(hr, prediction):

    if hr >= 140:
        risk, status = "RED", "Critical"
    elif hr < 60 or 100 <= hr < 120:
        risk, status = "YELLOW", "Caution"
    elif 120 <= hr < 140:
        risk, status = "ORANGE", "High Risk"
    else:
        risk, status = "GREEN", "Normal"

    # Escalate if AI flags Arrhythmia but HR alone looked normal
    if prediction == "ARRHYTHMIA" and risk == "GREEN":
        risk, status = "YELLOW", "Caution"

    return risk, status


# ==========================================
# ECG MONITORING LOOP
# ==========================================

def monitoring_loop():

    # Load AI model only once
    interpreter = load_interpreter()

    # Load FULL ECG dataset (not just the benchmark subset)
    X = np.load("X_ecg.npy")
    y = np.load("y_ecg.npy")

    # State cycle for a presentable demo narrative
    states = ["normal", "bradycardia", "tachycardia", "critical"]
    state_index = 0

    while True:

        # ==================================
        # CRITICAL DEMO MODE (manual trigger via /api/critical)
        # ==================================

        if demo_mode:

            monitoring_data["heart_rate"] = 160
            monitoring_data["prediction"] = "ARRHYTHMIA"
            monitoring_data["confidence"] = 95.34
            monitoring_data["risk_level"] = "RED"
            monitoring_data["status"] = "Critical"

            monitoring_data["timestamp"] = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            time.sleep(3)

            continue

        # ==================================
        # NO-QNX AI MONITORING
        # ==================================

        # Pick current state + a REAL beat matching it
        state = states[state_index]
        beat, simulated_hr, temp, ground_truth = generate_sample(state, X, y)

        # Run TFLite CNN inference
        prediction, confidence = classify_beat(
            interpreter,
            beat,
            already_normalized=True
        )

        # ==================================
        # CLINICAL ECG ANALYSIS
        # ==================================

        clinical_signal = np.tile(beat, 20)

        clinical_result = analyze_ecg_signal(
            clinical_signal,
            sampling_rate=360
        )

        monitoring_data["rr_interval"] = clinical_result["RR_interval_ms"]
        monitoring_data["rr_variability"] = clinical_result["RR_variability_ms"]
        monitoring_data["pr_interval"] = clinical_result["PR_interval_ms"]
        monitoring_data["qrs_duration"] = clinical_result["QRS_duration_ms"]
        monitoring_data["qt_interval"] = clinical_result["QT_interval_ms"]
        monitoring_data["qtc"] = clinical_result["QTc_ms"]
        monitoring_data["st_segment"] = clinical_result["ST_segment_ms"]

        # ==================================
        # UPDATE AI PREDICTION
        # ==================================

        monitoring_data["prediction"] = str(prediction).upper()
        monitoring_data["confidence"] = float(round(float(confidence), 2))

        # Heart rate driven by the current simulated state
        monitoring_data["heart_rate"] = simulated_hr

        # ==================================
        # AI PREDICTION + HR -> RISK LEVEL
        # ==================================

        monitoring_data["risk_level"], monitoring_data["status"] = get_risk_level(
            monitoring_data["heart_rate"],
            monitoring_data["prediction"]
        )

        if monitoring_data["risk_level"] == "RED":
            print("🚨 CRITICAL ALERT - Buzzer should activate")

            send_critical_alert(
              monitoring_data["heart_rate"],
              monitoring_data["prediction"],
              monitoring_data["confidence"]
        )

        # ==================================
        # TIMESTAMP
        # ==================================

        monitoring_data["timestamp"] = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # ==================================
        # MOVE TO NEXT STATE
        # ==================================

        state_index = (state_index + 1) % len(states)

        # ==================================
        # WAIT BEFORE NEXT READING
        # ==================================

        time.sleep(3)


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================================
# LIVE ECG READING API
# ==========================================

@app.route("/api/reading")
def api_reading():
    return jsonify(monitoring_data)


# ==========================================
# CRITICAL DEMO MODE (trigger)
# ==========================================

@app.route("/api/critical")
def critical_test():
    global demo_mode
    demo_mode = True
    return jsonify({"message": "Critical demo mode activated"})


# ==========================================
# START FLASK
# ==========================================

if __name__ == "__main__":

    monitoring_thread = threading.Thread(
        target=monitoring_loop,
        daemon=True
    )

    monitoring_thread.start()

    app.run(debug=True, use_reloader=False)