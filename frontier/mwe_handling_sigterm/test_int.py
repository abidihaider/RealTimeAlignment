import signal
import time
import sys
import datetime

def handle_sigint(signum, frame):
    print("SIGTINT received! too bad", flush=True)
    sys.exit(0)

# Attach signal handler
signal.signal(signal.SIGINT, handle_sigint)

print("Starting loop. Waiting for SIGINT...", flush=True)

while True:
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%H:%M:%S")
    print(f"Current Time: {formatted_time}", flush=True )
    time.sleep(5)
