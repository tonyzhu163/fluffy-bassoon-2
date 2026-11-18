"""Apply demonstration-derived lag medians to already completed frozen-VLA tapes."""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median

import numpy as np

from build_pairs import CONDITIONS, write_json
from read_bowls import lag_of_prefix


def compare(demos, panels):
    anchors = {c: median(r['lag'] for r in demos if r['condition'] == c and r['lag'] is not None)
               for c in CONDITIONS}
    rows = []
    for panel in panels:
        files = sorted((panel/'episodes').glob('*/public.npz'))
        expected = json.loads((panel/'episodes/complete.json').read_text())['episodes']
        if len(files) != expected:
            raise ValueError(f'Incomplete panel {panel}')
        for path in files:
            with np.load(path) as tape:
                lag = lag_of_prefix(tape['issued'][:80], tape['state_after'][:80, :3])
            choice = None
            if lag is not None:
                distances = [abs(lag - anchors[c]) for c in CONDITIONS]
                if distances[0] != distances[1]:
                    choice = CONDITIONS[int(distances[1] < distances[0])]
            label = json.loads((path.parent/'result.json').read_text())['variant']
            rows.append({'panel': panel.name, 'episode': path.parent.name,
                         'condition': label, 'lag': lag, 'predicted': choice})
    return {'oracle_medians': anchors, 'correct_by_panel': dict(Counter(
        r['panel'] for r in rows if r['predicted'] == r['condition'])), 'rows': rows}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('bowls', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('panels', type=Path, nargs='+')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    write_json(args.output, compare(json.loads(args.bowls.read_text())['demonstrations'], args.panels))
