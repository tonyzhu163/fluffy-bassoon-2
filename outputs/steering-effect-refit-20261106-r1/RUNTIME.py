"""Original frozen native/source/validation setup shared by this run's stages."""
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path('/mowenjie/fluffy-bassoon-2')
PARENT = ROOT / 'outputs/robocasa-three-task-kv-velocity-meta-20261105-r1'
OLD = ROOT / 'outputs/robocasa-three-task-meta-baseline-20261104-r2'
MEMBERS = ROOT / 'inputs/steering-source-members-20261105-pfs01'


class Runtime:
    def __init__(self, code, commit, probe, contract):
        if sys.platform != 'linux' or not socket.gethostname().startswith('job-'):
            raise ValueError('Require an allocated Linux scheduler container')
        sys.path[:0] = [str(code / 'src'), os.environ['RLINF_PATH']]
        import torch
        from fluffy_bassoon.baige_gpu import verify_selected_cuda, selected_device
        from fluffy_bassoon.meta_rl.rlinf_native_ppo import checked_checkout, tensor_hash
        from fluffy_bassoon.meta_rl.joint_steering_ppo_worker import _load_native, _versions
        from fluffy_bassoon.meta_rl.flow_history import DirectActionCodec, FlowDirectMotor
        from fluffy_bassoon.meta_rl.vla_history import file_sha256
        checked_checkout(code, commit)
        os.environ['LYCO_BAIGE_GPU_MAP'] = str(probe)
        verify_selected_cuda(torch)
        os.environ['MUJOCO_EGL_DEVICE_ID'] = str(selected_device()['allocation_ordinal'])
        torch.set_num_threads(4)
        self.contract, self.sha, self.tensor_hash, self.versions = contract, file_sha256, tensor_hash, _versions
        self.assembly_path = OLD / 'joint/preparation/ASSEMBLY.json'
        if file_sha256(self.assembly_path) != contract['assembly_sha256']:
            raise ValueError('Original joint source assembly changed')
        self.assembly = json.loads(self.assembly_path.read_text())
        norm = ROOT / 'inputs/robocasa365-public-pi05-c484-75000-20261101-r1/assets/norm_stats.json'
        model = ROOT / 'inputs/robocasa365-public-pi05-c484-torch-20261101-r2'
        os.environ.update(ROBOCASA_PI05_TORCH_MODEL=str(model), ROBOCASA_PI05_NATIVE_NORM=str(norm))
        self.native, self.factory, self.signature = _load_native(dict(model_directory=str(model), norm_path=str(norm),
            source_commit=commit, runtime_commit=contract['runtime_commit']), code)
        if (self.signature['model_sha256'] != contract['native_model_sha256']
                or self.signature['norm_sha256'] != contract['native_norm_sha256']):
            raise ValueError('Original native expert or normalization changed')
        stats = json.loads(norm.read_text()); self.stats = stats.get('norm_stats', stats)
        self.action_codec = DirectActionCodec.from_stats(self.stats)
        self.motor = FlowDirectMotor(self.native, self.action_codec)
        sources = {}
        for task, reference in self.assembly['source_assemblies'].items():
            path = Path(reference['path'])
            if file_sha256(path) != reference['sha256']:
                raise ValueError('Task source assembly changed')
            record = json.loads(path.read_text()); index = path.parent / 'TAR-INDEX.json'
            sources[task] = dict(index_path=str(index), index_sha256=file_sha256(index),
                                 archive=record['provenance']['bos_archive'])
        from fluffy_bassoon.meta_rl.flow_source_queries import FlowSourceQueries
        self.reader = FlowSourceQueries(sources, self.action_codec, member_cache=MEMBERS)
        self.frozen = tensor_hash(_versions(self.native.model))

    def data(self, objective, split):
        from fluffy_bassoon.meta_rl.vla_history import load_dataset
        item = self.assembly['datasets'][objective + '_' + split]
        data, manifest = load_dataset(item['path'], expected_sha256=item['sha256'])
        if len(data) != (336 if split == 'train' else 48):
            raise ValueError('Original source stream count changed')
        return data, manifest

    def prepare(self, data, manifest):
        def prepare(indices):
            import torch
            observation = self.reader.observations(indices, data, manifest['metadata']['trial_ids'])
            try:
                with torch.no_grad():
                    _, prepared = self.motor.prepare_batch(observation)
                _, mask, cache = prepared['prefix']
                return prepared['observation'], cache, mask
            finally:
                self.native._prepared = None
        return prepare

    def batch(self, policy, data, objective, rng, flow_rng):
        import torch
        from fluffy_bassoon.meta_rl.flow_history import sample_bridge_query_batch
        batch = sample_bridge_query_batch(data, policy, rng, action_codec=self.action_codec,
            native_action_stats=self.stats['actions'], batch_size=8, max_episodes=12,
            objective=objective, discount=.99)
        batch['noise'] = torch.randn(batch['target_actions'].shape, generator=flow_rng, device='cuda:0')
        batch['timesteps'] = torch.rand(8, generator=flow_rng, device='cuda:0')
        batch['weights'] = torch.ones(8, device='cuda:0')
        return batch

    def validation_plan(self, policy, data, objective):
        import numpy as np
        import torch
        settings = self.contract['diagnostic']
        rng = np.random.default_rng(settings['validation_seed'])
        flow_rng = torch.Generator(device='cuda:0').manual_seed(settings['flow_seed'])
        return [self.batch(policy, data, objective, rng, flow_rng)
                for _ in range(settings['validation_batches'])]

    def check_frozen(self):
        if self.tensor_hash(self.versions(self.native.model)) != self.frozen:
            raise ValueError('Frozen native expert changed during this stage')
