import signal
import time
import sys
import datetime

def handle_sigterm(signum, frame):
    print("SIGTERM received! Time is almost up. Wrapping up...", flush=True)
    sys.exit(0)

# Attach signal handler
signal.signal(signal.SIGTERM, handle_sigterm)

print("Starting loop. Waiting for SIGTERM...", flush=True)

while True:
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%H:%M:%S")
    print(f"Current Time: {formatted_time}", flush=True )
    time.sleep(5)
