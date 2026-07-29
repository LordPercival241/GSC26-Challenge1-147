import subprocess
import time
import sys
import os

print("Starting server...")
server_process = subprocess.Popen(
    [sys.executable, "src/server.py"],
    stdout=sys.stdout,
    stderr=sys.stderr,
    text=True
)

time.sleep(3) # Wait for server to start

print("Starting benign client...")
client_benign = subprocess.Popen(
    [sys.executable, "src/client.py"],
    stdout=sys.stdout,
    stderr=sys.stderr,
    text=True
)

time.sleep(1)

print("Starting malicious client...")
client_malicious = subprocess.Popen(
    [sys.executable, "src/client.py", "--malicious"],
    stdout=sys.stdout,
    stderr=sys.stderr,
    text=True
)

# Wait for clients to finish (they might not finish if server is stuck, so add timeout)
print("Waiting for training to complete (timeout 60s)...")
try:
    client_benign.wait(timeout=60)
    client_malicious.wait(timeout=60)
    # Give server a moment to print its output
    time.sleep(2)
    server_process.terminate()
except subprocess.TimeoutExpired:
    print("Test timed out!")
    server_process.terminate()
    client_benign.terminate()
    client_malicious.terminate()

print("Test complete.")
