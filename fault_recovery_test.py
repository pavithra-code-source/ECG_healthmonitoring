import time


def ai_process():
    print("AI process started...")
    time.sleep(1)

    # Fault injection
    raise RuntimeError("Simulated AI process failure")


def recover_ai_process():
    print("Fault detected!")
    print("Starting autonomous recovery...")

    time.sleep(1)

    print("AI process restarted successfully.")
    return True


print("Autonomous Fault-Recovery Test")
print("==============================")

try:
    ai_process()

except RuntimeError as error:

    print("Error:", error)

    recovered = recover_ai_process()

    if recovered:
        print("System status: RECOVERED")
    else:
        print("System status: FAILURE")