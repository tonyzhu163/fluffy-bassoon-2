"""Same raw pi05 commands, paired physical displacements and full-tape command diagnostics."""
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

import numpy as np

from adapter import issue
from panel import cases, key, save
from ur5e_sim_worker import UR5eSimulator


def command_stats(raw, issued, v):
    delta = issued[:,:3].astype(float) - raw[:,:3]
    off = delta - (delta @ v)[:,None]*v
    assert np.array_equal(raw[:,3:], issued[:,3:])
    return dict(mean_norm=float(np.linalg.norm(delta,axis=-1).mean()),
                p95_norm=float(np.quantile(np.linalg.norm(delta,axis=-1),.95)),
                max_norm=float(np.linalg.norm(delta,axis=-1).max()),
                off_axis_mean=float(np.linalg.norm(off,axis=-1).mean()),
                off_axis_max=float(np.linalg.norm(off,axis=-1).max()),
                new_boundary_controls=int(np.any((abs(issued[:,:3])>=1)&(abs(raw[:,:3])<1), axis=1).sum()),
                reversed_controls=int(np.sum((issued[:,:3]*raw[:,:3]).sum(-1)<0)))


def replay(sim, case, tape, expected, arm):
    _, first = sim.call('create', dict(case, arm=arm, embodiment='nom'))
    reset = first['environment_reproducibility']
    model = first['model_identity']
    assert reset['source_pin'] == 'e5500c690c1498eca1b7050350654964190cdf70'
    assert model['arm'] == arm and model['joint_count'] == len(model['joint_names']) and model['controller']=='OSC_POSE'
    for field in ('released_row_fingerprint','scene_fingerprint','bddl_sha256','perturbation_plan_fingerprint'):
        assert reset[field] == expected[field], field
    xyz = [first['observation']['robot0_eef_pos']]
    for group in tape.reshape(3,5,7):
        step = sim.call('chunk', group)
        assert step['steps']==5 and not step['terminated'] and not step['truncated']
        xyz.extend(step['positions'])
    return np.array(xyz), model['xml_sha256'], reset['initial_state_fingerprint']


def main(out, path, archive):
    adapters = json.loads(path.read_bytes())
    assert 'pin=a8a997e7d06bb131d3f9710204c58e106f0f0904' in (archive/'launch.txt').read_text()
    rows = []
    methods = ('identity','rank1','smooth_df','full','planar','orthogonal','rank1_free','smooth_df_line')
    with UR5eSimulator(allow_mesa=True) as sim:
        for case in cases('Kinova3'):
            name = key(case)
            file = archive/(name+'-Kinova3.json')
            raw_bytes = file.read_bytes()
            scanned = json.loads(raw_bytes)
            assert scanned['case']==case
            record = scanned['record']['reset']
            chunks = scanned['chunks']
            tape = np.concatenate([c['cmd'] for c in chunks]).astype(np.float32)
            assert len(tape)>=15 and tape.shape[1]==7 and record['arm']=='Kinova3'
            ref, panda_xml, panda_physics = replay(sim, case, tape[:15], scanned['full_reset'], 'Panda')
            ref_disp = np.diff(ref[::5],axis=0)
            row = adapters["Kinova3/"+case['suite']+"/"+str(case['task_id'])]
            assert row['support']['start']==11 and row['support']['physics']!=record['physics']
            v = np.array(row['adapters']['rank1']['direction'])
            branches = {}
            for m in methods:
                action = row['adapters'][m]
                issued = issue(tape, action['matrix'], action['direction'])
                branches[m] = dict(command=command_stats(tape,issued,v))
                if m in ('identity','rank1','smooth_df','rank1_free'):
                    xyz, kinova_xml, cpu_physics = replay(sim, case, issued[:15], scanned['full_reset'], 'Kinova3')
                    if m == 'identity':
                        logged = np.array([chunks[0]['before'], *sum([c['xyz'] for c in chunks[:3]],[])])
                        assert len(logged)==16
                        kinova_physics = cpu_physics
                        replay_diff = float(np.abs(xyz-logged).max())
                    assert cpu_physics == kinova_physics, 'Within-CPU Kinova reset mismatch'
                    disp = np.diff(xyz[::5],axis=0)
                    branches[m].update(displacements_m=disp.tolist(),
                        chunk_errors_mm=(1000*np.linalg.norm(disp-ref_disp,axis=1)).tolist(),
                        endpoint_error_mm=float(1000*np.linalg.norm(disp.sum(0)-ref_disp.sum(0))))
            rows.append(dict(case=case,controls=len(tape),input=str(file),tape_file_sha256=sha256(raw_bytes).hexdigest(),
                         opening_proposal_sha256=scanned['record']['opening_proposal_sha256'],noise=record['noise_sha256'],
                         gpu_reset_physics=record['physics'], cpu_reset_physics=kinova_physics, panda_reset_physics=panda_physics, xml_sha256=dict(Panda=panda_xml,Kinova3=kinova_xml),
                         panda_displacements_m=ref_disp.tolist(),identity_replay_max_abs_m=replay_diff,methods=branches))
            save(out/'replay.json',dict(affinity=sorted(os.sched_getaffinity(0)),rows=rows))
            print(name,{m:round(np.mean(b['chunk_errors_mm']),3) for m,b in branches.items() if 'chunk_errors_mm' in b},flush=True)


if __name__=='__main__':
    main(*[Path(a) for a in sys.argv[1:]])
