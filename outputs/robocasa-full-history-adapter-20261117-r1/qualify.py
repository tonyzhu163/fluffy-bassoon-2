"""Fresh native control qualification; no held-out student scores are inspected."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from native import load, rollout, write


def conditions():
    axis = [([m * s, 0., 0.] if d == 0 else [0., m * s, 0.])
            for m in (.02, .04, .03, .06) for d in (0, 1) for s in (-1, 1)]
    diagonal = [[x * .04 / np.sqrt(2), y * .04 / np.sqrt(2), 0.]
                for x in (-1, 1) for y in (-1, 1)]
    return axis + diagonal


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pin', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rank', type=int, required=True, choices=range(4))
    p.add_argument('--deadline', type=float, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)

    def log(kind, **data):
        row = dict(kind=kind, time=time.time(), **data)
        with (a.output / 'events.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')
        print(json.dumps(row, allow_nan=False), flush=True)

    native, motor, codec, stats, identity = load(a.pin)
    write(a.output / 'motor.json', identity)
    scenes = (982101, 982103, 982107, 983001)
    scene = scenes[a.rank]
    nominal = rollout(native, motor, codec, directory=a.output/'nominal', scene=scene,
        offset=[0.,0.,0.], fraction=0., log=log)
    if not nominal['success'] or nominal['onset'] is None:
        write(a.output/'summary.json', dict(complete=False, reason='nominal_not_qualified', nominal=nominal))
        return
    results = []
    for index in range(a.rank, len(conditions()), 4):
        offset = conditions()[index]
        for fraction in (0., 1.):
            if time.time() >= a.deadline:
                write(a.output/'summary.json', dict(complete=False, reason='deadline', results=results))
                return
            result = rollout(native, motor, codec, directory=a.output/f'condition-{index:02d}-f{int(fraction)}',
                scene=scene, offset=offset, fraction=fraction,
                target_start=nominal['start_signature'], log=log)
            results.append({k:result[k] for k in ('offset','fraction','success','controls','seconds',
                                                 'teacher_clipped_controls','onset','first_grasp','bytes')})
        write(a.output/'progress.json', dict(completed_conditions=len(results)//2, results=results))
    write(a.output/'summary.json', dict(complete=True, scene=scene, results=results,
        interpretation='Development conditions only; no inference about held-out student adaptation'))


if __name__ == '__main__':
    main()
