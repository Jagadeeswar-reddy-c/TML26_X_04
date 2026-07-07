import os
import sys
import requests
from pathlib import Path

BASE_URL = "http://34.63.153.158"      # donot change
API_KEY  = "568be0178160b1148f06f177d7d56b9a"
TASK_ID  = "22-forging-task"           # donot change

FILE_PATH = Path("submission_best.zip")

SUBMIT = True   # set to True to submit

def die(msg):
    print(f"{msg}", file=sys.stderr)
    sys.exit(1)

if not os.path.isfile(FILE_PATH):
    die(f"File not found: {FILE_PATH}")

if not SUBMIT:
    print("SUBMIT=False — dry run only. Set SUBMIT=True to send.")
    sys.exit(0)

with open(FILE_PATH, "rb") as f:
    files = {"file": (FILE_PATH.name, f, "zip")}
    params = {"api_key": API_KEY, "task_id": TASK_ID}
    resp = requests.post(f"{BASE_URL}/submit", files=files, params=params, timeout=120)

if resp.status_code == 200:
    print("Result:", resp.json())
else:
    die(f"HTTP {resp.status_code}: {resp.text}")
