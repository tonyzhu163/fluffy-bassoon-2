"""Allocated-Linux audit and paired corpus export; no simulator interaction."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from fluffy_bassoon.meta_rl.residual_history import relabel_teacher, split_learning_stream
from fluffy_bassoon.meta_rl.residual_ppo import ResidualActorCritic, ResidualPPOConfig
from fluffy_bassoon.meta_rl.vla_history import FIELDS, file_sha256, load_dataset, save_dataset

ROOT = Path('/mowenjie/fluffy-bassoon-2')
SOURCE = ROOT/'outputs/robocasa-eight-gpu-speed-20261102-r1/S8-r2'
HISTORY_SHA = '7d66810302a46ddf95477ad108a64ca031d69b0784ae211980c32714e5cc4912'
TEACHER_SHA = '84b4a25f1f0e55a22f4fdf265766fa5f6295504db2cf9a0a3ba8a0a28cc44f4d'
INITIAL = ROOT/'outputs/robocasa-residual-approved-20261101-r1/S/boundary-000.pt'
INITIAL_SHA = '0bf0a89a34ae957c7d4802f2d103e43e9ef56a0391af03ff7720a79019f2aab7'


def ref(path):
    return dict(path=str(path), sha256=file_sha256(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert sys.platform == 'linux' and not torch.cuda.is_available()
    torch.set_num_threads(4)
    trials, manifest = load_dataset(SOURCE/'history.npz', expected_sha256=HISTORY_SHA)
    assert len(trials) == 1 and manifest['transitions'] == 5760
    trial = trials[0]
    assert manifest['observation_dim'] == 2088 and manifest['noise_dim'] == 6
    assert int(trial['episode_starts'].sum()) == 384
    assert np.array_equal(np.diff(np.r_[np.flatnonzero(trial['episode_starts']), 5760]), np.full(384, 15))
    assert np.array_equal(trial['teacher_steps'], np.repeat(np.arange(6, dtype=np.int64), 960))
    assert not trial['rewards'][~trial['dones']].any()
    assert set(np.unique(trial['rewards'])) <= {0., 1.}
    assert file_sha256(SOURCE/'final.pt') == TEACHER_SHA
    assert file_sha256(INITIAL) == INITIAL_SHA
    saved = torch.load(SOURCE/'final.pt', map_location='cpu', weights_only=False)
    assert saved['final'] and saved['stage'] == 'boundary' and saved['iteration'] == 6
    assert saved['contract'] == manifest['metadata']['contract']
    teacher = ResidualActorCritic(ResidualPPOConfig(**saved['config']['ppo']))
    teacher.load_state_dict(saved['policy'], strict=True)
    contract = saved['contract']
    assert contract['task'] == 'OpenDrawer' and contract['arm'] == 'S'
    validation_slots = sorted(np.random.default_rng(90211).choice(64, 8, replace=False).tolist())
    split = split_learning_stream(trial, validation_slots=validation_slots)
    args.output.mkdir(parents=True, exist_ok=False)
    datasets = {}
    for partition, streams in split.items():
        slots, actual = zip(*streams)
        common = dict(trial_ids=[f'OpenDrawer-S8-slot-{slot:02d}' for slot in slots],
            source_history=ref(SOURCE/'history.npz'), source_manifest=ref(SOURCE/'history.json'),
            motor_sha256=contract['initializer']['model_sha256'],
            feature_contract=contract['features'], noise_contract=saved['config']['bound'],
            partition=partition, stream_slots=list(slots), validation_slots=validation_slots,
            grouping='same collection slot across six chronological versions; different scenes',
            teacher_saw_validation_states=True, rewards='actual source simulator terminal success',
            teacher=ref(SOURCE/'final.pt'), teacher_labels='deterministic mean in original pre-tanh u coordinates')
        paired = dict(ad=list(actual), dpt=relabel_teacher(actual, teacher, deterministic=True))
        for before, after in zip(paired['ad'], paired['dpt'], strict=True):
            for field in FIELDS:
                if field != 'target_noise':
                    assert np.array_equal(before[field], after[field])
        for objective, rows in paired.items():
            path = args.output/f'{objective}-{partition}.npz'
            published = save_dataset(path, rows, objective=objective, metadata=common)
            datasets[f'{objective}_{partition}'] = dict(**ref(path), trials=len(rows),
                transitions=published['transitions'], episodes=sum(int(row['episode_starts'].sum()) for row in rows))
    result = dict(completed=True, source=ref(SOURCE/'history.npz'), teacher=ref(SOURCE/'final.pt'),
        initial=ref(INITIAL), contract=contract, datasets=datasets, validation_slots=validation_slots,
        training_episodes=336, validation_episodes=48, no_new_robot_actions=True,
        source_terminal_successes=int(trial['rewards'].sum()),
        source_successes_per_version=[int(values.sum()) for values in trial['rewards'].reshape(6, -1)])
    (args.output/'assembly.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
