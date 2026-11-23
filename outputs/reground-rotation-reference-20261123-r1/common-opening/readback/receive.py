"""Publish the latest complete Baige reporter event atomically in the local readback directory."""
import json
from pathlib import Path
import sys

directory = Path(__file__).resolve().parent
for line in sys.stdin:
    event = json.loads(line)
    target = directory / "REQUESTS.json"
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(event, indent=2) + "\n")
    tmp.replace(target)
