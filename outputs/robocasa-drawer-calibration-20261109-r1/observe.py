"""Read an immutable job directory; publish only to the sibling observer directory."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time


def snapshot(source):
    rows = [json.loads(p.read_text()) for p in sorted(source.glob('episode-*.json'))]
    events = source / 'events.jsonl'
    tail = events.read_text().splitlines()[-2:] if events.exists() else []
    last = next((json.loads(line) for line in reversed(tail) if line.endswith('}')), {})
    probe = source/'GPU-PROBE.json'
    exit_file = source/'exit.txt'
    code, elapsed = None, None
    if exit_file.exists():
        code, start, end = map(int, exit_file.read_text().split())
        elapsed = end - start
    arms = {}
    for row in rows:
        name = row['role'] + ':' + str(row['compensation'])
        arm = arms.setdefault(name, Counter())
        arm.update(episodes=1, successes=int(row['success']), acquired=int(row['acquired_handle']),
                   closures=int(row['closure_control'] is not None))
    pairs = sorted({r['pair'] for r in rows})
    paired = []
    for pair in pairs:
        for sign in (-1, 1):
            triple = {r['compensation']:r for r in rows if r['pair']==pair and r['role']=='query' and r['sign']==sign}
            if len(triple) != 3:
                continue
            assert len({r['paired_initial_sha256'] for r in triple.values()}) == 1
            paired.append(dict(pair=pair, sign=sign, arms={
                str(c):dict(success=r['success'], acquired=r['acquired_handle'],
                            miss_1=r['signed_handle_miss_m'].get('1'), miss_5=r['signed_handle_miss_m'].get('5'))
                for c,r in triple.items()}))
    return dict(checked_utc=datetime.now(timezone.utc).isoformat(), host=source.name,
                gpu_probe=json.loads(probe.read_text())['status'] if probe.exists() else None,
                last_event=last, completed_episodes=len(rows), completed_controls=sum(r['controls'] for r in rows),
                completed_by_arm=arms, paired_queries=paired, exit_code=code, runner_seconds=elapsed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('source', type=Path)
    p.add_argument('--monitor', type=Path)
    a = p.parse_args()
    if not a.monitor:
        print(json.dumps(snapshot(a.source), indent=2))
        return
    a.monitor.mkdir(exist_ok=False)
    stop = time.monotonic() + 6500
    due = 0
    while time.monotonic() < stop:
        if time.monotonic() >= due or (a.source/'exit.txt').exists():
            record = snapshot(a.source)
            with (a.monitor/'checks.jsonl').open('a') as fp:
                fp.write(json.dumps(record) + '\n')
            if record['exit_code'] is not None:
                return
            due = time.monotonic() + 1200
        time.sleep(10)
    raise TimeoutError('Scheduler duration plus observer grace elapsed')


if __name__ == '__main__':
    main()
