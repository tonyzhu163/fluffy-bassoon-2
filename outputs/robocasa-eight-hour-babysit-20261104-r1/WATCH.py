"""Keep this chat's read-only twenty-minute observations through the requested window."""
import datetime
import json
from pathlib import Path
import subprocess
import time


def main():
    root = Path(__file__).resolve().parent
    end = datetime.datetime.fromisoformat('2026-11-05T07:05:00+08:00').timestamp()
    while True:
        began = time.time()
        try:
            process = subprocess.run(['ssh', '-p', '4952', '-o', 'BatchMode=yes', '-o',
                'ConnectTimeout=10', 'root@10.200.0.11', 'python3', '-'],
                input=(root / 'OBSERVE.py').read_text(), text=True, capture_output=True,
                timeout=45, check=True)
            value = json.loads(process.stdout)
            stamp = datetime.datetime.fromisoformat(value['checked_shanghai']).strftime('%Y%m%d-%H%M%S')
            payload = json.dumps(value, indent=2) + '\n'
            (root / ('CHECK-' + stamp + '.json')).write_text(payload)
            temporary = root / 'LIVE-CHECK.json.tmp'
            temporary.write_text(payload); temporary.replace(root / 'LIVE-CHECK.json')
            methods = {name: dict(update=row['latest_metric'].get('update'),
                phase=row['latest_metric'].get('phase', 'actor'),
                converged=bool(row['status'] and row['status'].get('converged')),
                metric_age_seconds=row['metric_age_seconds'], finite=row['recent_metrics_finite'])
                for name, row in value['methods'].items()}
            print(json.dumps(dict(checked_shanghai=value['checked_shanghai'], methods=methods,
                errors=value['error_tails'], released=value['worker_release'],
                budget=value['source_budget'], baseline_test512=value.get('baseline_test512'))), flush=True)
        except Exception as error:
            print(json.dumps(dict(checked_unix=time.time(), observation_failed=repr(error))), flush=True)
        if time.time() >= end:
            return
        time.sleep(max(0, min(end - time.time(), began + 1200 - time.time())))


if __name__ == '__main__':
    main()
