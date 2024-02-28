import time
import subprocess

while True:
    subprocess.run(["python", "gp_patient_survey/data.py"])
    time.sleep(7200)  # Wait for 7200 seconds = 2 hours
