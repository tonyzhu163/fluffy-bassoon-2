"""Run only in the allocated CPU job; no simulator or frozen motor imports."""
import hashlib
import json
from pathlib import Path
import platform
import socket
import subprocess
import time

import torch

from fluffy_bassoon.meta_rl.robotwin_history_corpus import export_corpus
from fluffy_bassoon.meta_rl.robotwin_history import (
    FrozenNativeHistoryController, NativeHistoryConfig, load_dataset, load_policy,
)
from fluffy_bassoon.meta_rl.robotwin_history_train import fit
from fluffy_bassoon.meta_rl.vla_history import file_sha256


ROOT = Path('/mowenjie/fluffy-bassoon-2')
RUN = 'robotwin-native-history-20261030-r1'
PIN = '1aeea4f9fcc307bb147def92374bb81a3e2cb50e'
CODE = ROOT / 'worktrees' / RUN
OUT = ROOT / 'outputs' / RUN
assert platform.system() == 'Linux' and socket.gethostname().startswith('job-')
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CODE, text=True).strip() == PIN
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=CODE, text=True).strip()
torch.set_num_threads(4)
started = time.monotonic()
prefix = ROOT / 'exports/robotwin-native-prefix-20261030-r1/manifest.json'
assert file_sha256(prefix) == '05680fb416e6b8031f04b9832e903906ea2c4cb3ee091499bf4c65eb54d4c37b'
source = json.loads(prefix.read_text())['methods']['sac']
teacher = dict(teacher_path=source['recovery_path'], teacher_sha256=source['recovery_sha256'],
               teacher_method='sac', device='cpu')
export_corpus(source['corpus_root'], OUT / 'train', source_path=source['recovery_path'],
    source_sha256=source['recovery_sha256'], source_method='sac', **teacher)
ppo = ROOT / 'outputs/robotwin-wholetrial-ppo-20261030-r2/experiment'
export_corpus(ppo / 'corpus', OUT / 'validation',
    source_path=ppo / 'stages/iteration-10/ppo-recovery.pt',
    source_sha256='62a6f92819c1df3eaa1614fd0052e999f496fe23ba5f7944bf7a722ef4a0ffb9',
    source_method='ppo', partition='validation', validation_iteration=10,
    objectives=('behavior_cloning', 'dpt'), **teacher)
train_index, validation_index = OUT / 'train/index.json', OUT / 'validation/index.json'
results = []
for objective in ('ad', 'dpt'):
    train, tm = load_dataset(train_index, file_sha256(train_index), objective)
    validation, vm = load_dataset(validation_index, file_sha256(validation_index),
        'behavior_cloning' if objective == 'ad' else 'dpt')
    for mode in ('full', 'static'):
        target = OUT / f'{objective}-{mode}'
        _, summary = fit(train, tm, validation, vm, target, source_commit=PIN, mode=mode,
            config=NativeHistoryConfig(width=8, layers=1, heads=2, context_length=8),
            updates=2, batch_size=1, max_window=4, validation_batches=1,
            eval_every=1, milestones=(1, 2), max_seconds=30., device='cpu')
        path = target / 'candidate-000002.pt'
        policy, manifest = load_policy(path, file_sha256(path))
        before = {key: value.clone() for key, value in policy.state_dict().items()}
        controller = FrozenNativeHistoryController(policy, manifest, interface=tm['interface'])
        controller.reset_trial(seed=52)
        for episode in range(2):
            controller.begin_episode(episode_index=episode, episode_limit=600)
            action = controller.act(validation[0]['observations'][0], deterministic=True)
            assert action.shape == (32, 32) and abs(action).max() <= 1.5
            controller.observe(action, 0., True, executed_steps=1)
        for key, value in policy.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        results.append(dict(objective=objective, mode=mode, summary=summary,
            checkpoint_sha256=file_sha256(path), frozen_two_episode_wrapper=True))
receipt = dict(completed=True, hostname=socket.gethostname(), source_commit=PIN,
    script_sha256=file_sha256(__file__), source_manifest_sha256=file_sha256(prefix),
    train_index_sha256=file_sha256(train_index), validation_index_sha256=file_sha256(validation_index),
    train_counts=tm['counts'], validation_counts=vm['counts'],
    additional_simulator_actions=0, additional_vla_queries=0, allocated_gpus=0,
    tiny_smoke_sgd_updates=8, results=results, elapsed_seconds=time.monotonic()-started,
    status='two-update toy-width integration only; no real first-fit or robot performance claim')
(OUT / 'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
print(json.dumps(receipt), flush=True)
