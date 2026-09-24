import time

TRIALS = 20

def ai_process():
    raise RuntimeError("Simulated AI process failure")

def recover_ai_process():
    recovery_status = True
    return recovery_status

recovery_times = []
successful_recoveries = 0

print("Multiple Fault Recovery Test")
print("============================")
print("Trials:", TRIALS)

for i in range(TRIALS):

    try:
        ai_process()

    except RuntimeError:
        fault_detected_time = time.perf_counter()

        recovered = recover_ai_process()

        recovery_completed_time = time.perf_counter()

        recovery_time = (
            recovery_completed_time - fault_detected_time
        ) * 1000

        recovery_times.append(recovery_time)

        if recovered:
            successful_recoveries += 1

print()
print("Average Recovery Time:",
      round(sum(recovery_times) / len(recovery_times), 4), "ms")

print("Minimum Recovery Time:",
      round(min(recovery_times), 4), "ms")

print("Maximum Recovery Time:",
      round(max(recovery_times), 4), "ms")

success_rate = (successful_recoveries / TRIALS) * 100

print("Successful Recoveries:",
      successful_recoveries, "/", TRIALS)

print("Recovery Success Rate:",
      round(success_rate, 2), "%")