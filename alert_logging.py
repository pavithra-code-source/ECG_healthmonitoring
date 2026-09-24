from datetime import datetime

LOG_FILE = "ecg_log.txt"


def get_hr_risk_level(hr):
    if 60 <= hr <= 99:
        return "GREEN", "Normal"

    elif hr < 60 or 100 <= hr <= 119:
        return "YELLOW", "Caution"

    elif 120 <= hr <= 139:
        return "ORANGE", "High Risk"

    else:
        return "RED", "Critical"


def check_alert(result, clinical_features=None):

    hr = None

    if clinical_features:
        hr = clinical_features.get("HR_bpm")

    if hr not in ("", None):

        risk_level, risk_status = get_hr_risk_level(float(hr))

        print(
            f"HR Risk: {risk_level} | "
            f"HR: {hr} BPM | "
            f"{risk_status}"
        )

    if result["label"] == "Arrhythmia":

        print(
            f"CARDIAC ANOMALY DETECTED -- "
            f"confidence {result['confidence']:.2f}% "
            f"(source: {result['source']})"
        )

        return True

    print(
        f"Normal -- "
        f"confidence {result['confidence']:.2f}% "
        f"(source: {result['source']})"
    )

    return False


def log_result(result, clinical_features=None, log_file=LOG_FILE):

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    hr = ""

    if clinical_features:
        hr = clinical_features.get("HR_bpm", "")

    if hr not in ("", None):

        risk_level, risk_status = get_hr_risk_level(float(hr))

    else:

        risk_level = ""
        risk_status = ""

    fields = [
        timestamp,
        result["label"],
        f"{result['confidence']:.2f}",
        result["source"],
        hr,
        risk_level,
        risk_status
    ]

    if clinical_features:

        for key in [
            "PR_interval_ms",
            "QRS_duration_ms",
            "QTc_ms",
            "ST_segment_ms"
        ]:

            fields.append(
                str(clinical_features.get(key, ""))
            )

    with open(log_file, "a") as f:

        f.write(
            ",".join(map(str, fields)) + "\n"
        )


def alert_and_log(
    result,
    clinical_features=None,
    log_file=LOG_FILE
):

    is_abnormal = check_alert(
        result,
        clinical_features
    )

    log_result(
        result,
        clinical_features,
        log_file
    )

    return is_abnormal


if __name__ == "__main__":

    import neurokit2 as nk

    from ecg_cnn_inference import load_interpreter
    from network_check import predict
    from clinical_analysis import analyze_ecg_signal

    print("ECG Alert and Logging System")
    print("============================")

    interpreter = load_interpreter()

    for state, hr in [
        ("normal", 70),
        ("abnormal", 150)
    ]:

        print()
        print(f"Testing state: {state}")
        print(f"Simulated HR: {hr} BPM")

        ecg_strip = nk.ecg_simulate(
            duration=10,
            sampling_rate=360,
            heart_rate=hr,
            noise=0.05
        )

        features = analyze_ecg_signal(
            ecg_strip
        )

        beat = ecg_strip[:187]

        result = predict(
            interpreter,
            beat
        )

        alert_and_log(
            result,
            features
        )

    print()
    print(f"All results logged to {LOG_FILE}")