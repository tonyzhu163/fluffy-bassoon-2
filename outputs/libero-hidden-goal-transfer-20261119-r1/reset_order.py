"""Marginalize over every distinct choice of two practice poses and a query pose."""
import json
from pathlib import Path
import sys

import torch

from reader import KNOWN, Reachability, evaluate, load_panel


if __name__ == '__main__':
    source, run, dest = map(Path, sys.argv[1:])
    torch.set_num_threads(1)
    model = Reachability(source / 'frozen.npz')
    for name, root, triples in [('mixed', source / 'fresh', [(0, 1, g) for g in KNOWN]),
                               ('primary', run / 'episodes', [(0, 1, 5)])]:
        panel = load_panel(root, model)
        cyclic = evaluate(panel, triples, run / 'reader')
        assert cyclic == json.loads((run / f'{name}.json').read_text())
        all_orders = evaluate(panel, triples, run / 'reader', all_poses=True)
        report = dict(pose_ids=len({p for p,_ in panel}), physical_blocks=len(panel),
                      cyclic=cyclic['total'], all_orders=all_orders['total'],
                      by_support_pose=all_orders['pose'])
        (dest / f'pose-orders-{name}.json').write_text(json.dumps(report, indent=2) + '\n')
        print(name, report['all_orders'])
