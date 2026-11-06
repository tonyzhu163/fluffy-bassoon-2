"""Independent CPU readback of this job's complete, drained physical traces."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    import numpy as np
    import torch
    from fluffy_bassoon.rl.robotwin_execution import chunk_execution
    root = Path(sys.argv[1]); manifest = json.loads((root/'TRACES.json').read_text())
    assert manifest['complete'] and not (root/'READBACK.json').exists()
    cases, hashes, previous = {}, {}, {}
    previous_intent, intent = None, None
    for name in manifest['paths']:
        path = root/name
        with gzip.open(path, 'rb') as stream: row = torch.load(stream, map_location='cpu', weights_only=False)
        intent_name = name.rsplit('-lane-', 1)[0]+'-intent.pt.gz'
        if intent_name != previous_intent:
            with gzip.open(root/intent_name, 'rb') as stream:
                intent = torch.load(stream, map_location='cpu', weights_only=False)
            previous_intent = intent_name
            hashes[intent_name] = hashlib.sha256((root/intent_name).read_bytes()).hexdigest()
        batch_row = row['native_batch']['row']
        assert dict(intent['native_batch'], row=batch_row) == row['native_batch']
        for key in ('requested_actions','base_actions','public_features'):
            assert np.array_equal(np.asarray(intent[key][batch_row]), row[key])
        for key in ('main_images','wrist_images','states'):
            assert np.array_equal(intent['observations'][batch_row][key], row['observation'][key])
        case = row['case_id']; chunks = cases.setdefault(case, [])
        assert row['chunk_index'] == len(chunks)
        if chunks:
            assert not chunks[-1]['terminated'] and not chunks[-1]['truncated']
            for field in ('main_images','wrist_images','states'):
                assert np.array_equal(previous[case][field], row['observation'][field])
            assert previous[case]['task_descriptions'] == row['observation']['task_descriptions']
        assert row['requested_actions'].shape == (32,14) and np.isfinite(row['requested_actions']).all()
        assert np.array_equal(row['requested_actions'], row['base_actions'])
        assert row['public_features'].shape == (2062,) and np.isfinite(row['public_features']).all()
        execution = chunk_execution({'fb2_execution':row['execution']}, requested_targets=32,
                                    timestep_seconds=row['execution']['timestep_seconds'])
        assert execution.duration_seconds == row['physics_duration_seconds']
        assert tuple(row['measured_valid_target_mask']) == execution.valid_target_mask
        chunks.append(dict(terminated=row['terminated'],truncated=row['truncated']))
        previous[case] = row['next_observation']
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(cases) == 4 and all(v[-1]['terminated'] or v[-1]['truncated'] for v in cases.values())
    (root/'READBACK.json').write_text(json.dumps(dict(status='complete', pid=os.getpid(), episodes=len(cases),
        transitions=len(manifest['paths']), file_sha256=hashes, consecutive_observations_exact=True,
        pre_dispatch_intents_exact=True), indent=2)+'\n')


if __name__ == '__main__': main()
