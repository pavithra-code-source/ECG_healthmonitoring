import time


def ai_process():
    print("AI process started...")

    time.sleep(1)

    # Simulate AI failure
    raise RuntimeError("Simulated AI process failure")


def recover_ai_process():
    print("Starting autonomous recovery...")

    time.sleep(1)

    print("AI process restarted successfully.")


print("Recovery Performance Test")
print("=========================")

try:

    ai_process()

except RuntimeError as error:

    fault_detected_time = time.perf_counter()

    print("Fault detected!")
    print("Error:", error)

    recover_ai_process()

    recovery_completed_time = time.perf_counter()

    recovery_time = (
        recovery_completed_time - fault_detected_time
    ) * 1000

    print()
    print(
        "Recovery Time:",
        round(recovery_time, 2),
        "ms"
    )

    print("System status: RECOVERED")