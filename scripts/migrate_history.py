import json
from pathlib import Path

HIST = Path("src/session_history.json")

if not HIST.exists():
    print("No history file found.")
    exit()

with HIST.open("r") as f:
    sessions = json.load(f)

# back-fill missing Mode
for s in sessions:
    if "Mode" not in s:
        # assume any pre-sentiment entry was trend-breakout / regular
        s["Mode"] = "Trend-Breakout"

with HIST.open("w") as f:
    json.dump(sessions, f, indent=2)

print(f"Patched {len(sessions)} entries with default Mode.")
