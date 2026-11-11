"""Raw-control recording with native selected-skill termination, never hidden goals."""
import hashlib
import json
from pathlib import Path

import numpy as np
from behavior_prompting.train_network.env.libero.env_wrapper import ControlEnvForRunner

from geometry import BOWLS, sha

PUBLIC_HORIZON = 562


class CustomControlEnv(ControlEnvForRunner):
    def __init__(self, *, manifest, geometry_root, output_root, candidate, **kwargs):
        super().__init__(**kwargs)
        self.metadata = None
        self.manifest = manifest
        self.geometry_root = Path(geometry_root)
        self.output_root = Path(output_root)
        self.candidate = candidate
        self.spec = None
        self.recording = False
        self.completed_indices = set()
        self.by_state = {item['state_sha256']: item for item in manifest['layouts']}

    def native_values(self):
        before = self.get_sim_state().copy()
        values = [int(bool(self.env._eval_predicate(['on', bowl, 'plate_1']))) for bowl in BOWLS]
        primary = int(all(bool(self.env._eval_predicate(goal))
                          for goal in self.env.parsed_problem['goal_state']))
        assert primary == int(self.check_success())
        assert np.array_equal(before, self.get_sim_state()), 'Native evaluation changed physics'
        return values

    def register_init_state(self, state):
        assert not self.recording, 'Previous raw record was not finalized'
        super().register_init_state(state)
        self.spec = None if state is None else self.by_state[sha(state)]

    def reset(self):
        # Native reset is called by official initializer before registering a
        # custom state. Fail immediately on invalid scenes rather than invoking
        # the upstream wrapper's exception-swallowing retry loop.
        if self.registered_init_state is None:
            return self.prepare_observation(self.env.reset())
        obs = self.prepare_observation(self.regenerate_obs_from_state(self.registered_init_state))
        assert self.spec is not None
        assert sha(self.get_sim_state()) == self.spec['state_sha256']
        # Official runner clears its video buffer with one extra reset at end.
        # That reset must not create a second experimental record.
        if self.spec['index'] in self.completed_indices:
            return obs
        assert not self.recording
        assert {k: sha(v) for k, v in obs.items()} == self.spec['observation_hashes']
        assert self.native_values() == [0, 0]
        self.observations = [{k: np.array(v, copy=True) for k, v in obs.items()}]
        self.actions = []
        self.native_trace = []
        self.body_positions = []
        self.recording = True
        self.selected_physical_index = BOWLS.index(self.spec['candidate_object_ids'][self.candidate])
        return obs

    def step(self, action):
        assert self.recording, 'Experimental controls require an admitted registered state'
        assert len(self.actions) < PUBLIC_HORIZON
        raw_obs, raw_reward, raw_done, raw_info = self.env.step(action)
        obs = self.prepare_observation(raw_obs)
        native = self.native_values()
        self.observations.append({k: np.array(v, copy=True) for k, v in obs.items()})
        self.actions.append(np.array(action, copy=True))
        self.native_trace.append(native)
        self.body_positions.append([self.sim.data.body_xpos[self.env.obj_body_id[name]].tolist()
                                    for name in BOWLS])
        selected = native[self.selected_physical_index]
        # Raw BDDL reward/done can refer to the other physical ID after an ID
        # swap. Only this preassigned selected-skill native predicate is exposed
        # to the official motor runner. No hidden rule enters this class.
        return obs, float(selected), bool(selected), {}

    def finish_record(self):
        assert self.recording and 0 < len(self.actions) <= PUBLIC_HORIZON
        index = self.spec['index']
        count = len(self.actions)
        public = {}
        for key in self.observations[0]:
            values = np.stack([obs[key] for obs in self.observations])
            padding = np.repeat(values[-1:], PUBLIC_HORIZON + 1 - len(values), axis=0)
            public[key] = np.concatenate((values, padding), axis=0)
        actions = np.stack(self.actions)
        public['actions'] = np.concatenate((actions, np.zeros((PUBLIC_HORIZON-count, 7), dtype=actions.dtype)))
        public['active_control_mask'] = np.arange(PUBLIC_HORIZON) < count
        public['candidate'] = np.asarray(self.candidate, dtype=np.int64)
        path = self.output_root / 'public' / f'record-{index:02d}-{self.candidate}.npz'
        path.parent.mkdir(parents=True, exist_ok=True)
        assert not path.exists()
        np.savez_compressed(path, **public)
        native_any = np.max(np.asarray(self.native_trace), axis=0).astype(int).tolist()
        record = {'index': index, 'candidate': self.candidate, 'actual_controls': count,
                  'public_horizon': PUBLIC_HORIZON, 'public_file': str(path.relative_to(self.output_root)),
                  'public_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                  'public_array_hashes': {k: sha(v) for k, v in public.items()},
                  'initial_state_sha256': self.spec['state_sha256'],
                  'initial_observation_hashes': {k: sha(v) for k, v in self.observations[0].items()},
                  'selected_physical_index': self.selected_physical_index,
                  'selected_skill_success': native_any[self.selected_physical_index],
                  'native_any': native_any, 'native_trace': self.native_trace,
                  'body_positions': self.body_positions,
                  'termination': 'selected_skill' if self.native_trace[-1][self.selected_physical_index] else 'fixed_horizon',
                  'hidden_rule_used_for_actions_or_termination': False}
        private = self.output_root / 'audit' / f'record-{index:02d}-{self.candidate}.json'
        private.parent.mkdir(exist_ok=True)
        private.write_text(json.dumps(record, indent=2) + '\n')
        self.recording = False
        self.completed_indices.add(index)
        print('CUSTOM_EPISODE', index, self.candidate, count, native_any, flush=True)
        return {k: v for k, v in record.items() if k not in ['body_positions', 'native_trace']}
