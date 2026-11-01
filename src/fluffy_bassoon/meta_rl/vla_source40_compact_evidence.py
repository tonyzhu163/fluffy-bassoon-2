"""Opt-in metadata for the EXISTING source40 collector; no learner or scheduler.

Default legacy execution never constructs this writer. Fresh source contracts
or exact same-contract resumes can opt in with --compact-history-evidence.
Only compact RNG/JSON evidence is added; existing raw/full-checkpoint cadence,
noise, loss, updates, episodes, physical costs and feature calls stay unchanged.
"""
from copy import deepcopy
import json
from pathlib import Path
import pickle
import random
import time

from .vla_source40_schedule import SOURCE_SUITES, content_hash

FORMAT = 'source40_compact_history_evidence_v2'
PI05_RECEIPT = 'ec064cce32f5003e2e3979c5712d7d6385018f6bfcc59cb2161105881e70d170'
ACTOR_SCOPE = 'vla_libero.structure_digest(Flax to_state_dict(full actor TrainState)); all serialized actor variables/state, not params only'
WARMUP = dict(distribution='standard_normal', shape=[32], dtype='float32',
              expansion='tied across native H10; no clipping or rescaling')
PROTOCOL = dict(format=FORMAT, actor_state_scope=ACTOR_SCOPE,
    policy_configuration='original source contract/code/recipe; canonical public PIL64/proprio8 preprocessing; full actor encoder/normalization state',
    rng_scope='original sampler/agent/global NumPy/Python/native motor states; no RNG draws for metadata',
    checkpoint_cadence='unchanged latest complete recovery and original declared case milestones/final teacher; no per-episode full snapshots')


def reference(path):
    from .vla_libero import sha256
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha256(path))


def checked(ref):
    from .vla_libero import sha256
    path = Path(ref['path'])
    if not path.is_absolute() or sha256(path) != ref['sha256']:
        raise ValueError('Original compact source evidence changed')
    return path


def _new_json(path, value):
    from .vla_libero import write_json
    if path.exists():
        raise ValueError('Immutable compact source evidence already exists')
    write_json(path, value)
    return reference(path)


def _new_pickle(path, value):
    # Same atomic temporary->replace pattern as OfficialSAC.save; no checkpoint
    # or learning state is created, moved or overwritten by this helper.
    if path.exists():
        raise ValueError('Immutable compact RNG evidence already exists')
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('xb') as stream:
        pickle.dump(value, stream, protocol=5)
    temporary.replace(path)
    return reference(path)


def actor_identity(learner):
    """Full canonical serialized actor state plus separate params diagnostic."""
    from flax import serialization
    from .vla_libero import structure_digest
    state = serialization.to_state_dict(learner.agent._actor)
    return dict(actor_state_sha256=structure_digest(state),
                actor_parameter_sha256=structure_digest(state['params']))


def rng_state(learner, sampler, motor):
    import jax
    import numpy as np
    from .vla_source40_corpus import save_motor_rng
    return dict(sampler=deepcopy(sampler.bit_generator.state),
        agent_rng=np.asarray(jax.device_get(learner.agent._rng)).copy(),
        numpy_rng=deepcopy(np.random.get_state()), python_rng=deepcopy(random.getstate()),
        motor_rng=save_motor_rng(motor.policy._rng))


def behavior_identity(learner, contract):
    warming = learner.updates == 0
    identity = dict(actor_state_sha256=None, actor_parameter_sha256=None) if warming else actor_identity(learner)
    return dict(identity, kind='gaussian_warmup' if warming else 'sac', actor_not_used=warming,
        sampler_configuration=deepcopy(WARMUP) if warming else None,
        actor_state_scope=ACTOR_SCOPE, source_contract_sha256=content_hash(contract))


class CompactHistoryEvidence:
    """Records one real episode at a time; commits only original saved traces.

    The existing collector's progress/recovery is the authoritative cursor.
    Interrupted attempts leave orphan compact files; they are never padded into
    the chronology or used to assign identities to old/legacy episodes.
    """
    def __init__(self, output, contract):
        task = contract.get('task', {})
        if (contract.get('format') != 'openpi_pi05_libero40_persistent_source_v1'
                or contract.get('source_only') is not True or contract.get('source_data_from_targets') is not False
                or task.get('suite') not in SOURCE_SUITES or type(task.get('task_id')) is not int
                or not 0 <= task['task_id'] < 10 or task.get('task_key') != f"{task['suite']}:{task['task_id']:02d}"
                or contract.get('compact_history_evidence') != PROTOCOL
                or contract.get('motor', {}).get('receipt_sha256') != PI05_RECEIPT
                or contract.get('motor', {}).get('predicted_horizon') != 10
                or contract.get('motor', {}).get('action_dim') != 32
                or contract.get('feature_contract', {}).get('observation_dim') != 4104
                or contract.get('feature_contract', {}).get('checkpoint_receipt_sha256') != PI05_RECEIPT
                or contract.get('data_role', 'source') != 'source' or contract.get('target_only') is True):
            raise ValueError('Compact producer requires a fresh original SOURCE40 opt-in contract')
        self.output, self.contract = Path(output).resolve(), deepcopy(contract)
        self.root = self.output/'compact-history'
        self.root.mkdir(parents=True, exist_ok=True)
        self.active, self.cache, self.cursor = None, {}, 0

    def begin(self, learner, sampler, motor, *, case, episode_in_case, ordinal):
        if self.active is not None or ordinal != self.cursor:
            raise ValueError('Previous compact episode has not committed')
        directory = self.root/f'episode-{ordinal:06d}-{time.time_ns()}'
        directory.mkdir(exist_ok=False)
        self.active = dict(directory=directory, ordinal=ordinal, case=deepcopy(case),
            episode_in_case=episode_in_case, behavior=behavior_identity(learner, self.contract),
            before_updates=int(learner.updates), replay_before=len(learner.replay), choices=[],
            rng_before=_new_pickle(directory/'rng-before.pkl', rng_state(learner, sampler, motor)))

    def trace_path(self, ordinal):
        if self.active is None or self.active['ordinal'] != ordinal:
            raise ValueError('Compact raw trace ordinal differs from original active episode')
        # The opt-in trace remains one flat episode payload in the same raw/
        # directory; unique attempt names preserve interrupted original refs.
        return self.output/'raw'/(self.active['directory'].name+'.pkl.gz')

    def reset(self, journal_path, result):
        active = self.active
        record = json.loads(Path(journal_path).read_text())
        if (active is None or record.get('returned') is not True
                or record['case_id'] != active['case']['case_id']
                or record['initial_state'] != active['case']['initial_state']
                or record['scene_seed'] != active['case']['scene_seed']):
            raise ValueError('Compact evidence reset differs from original episode journal')
        active['reset_ref'] = reference(journal_path)
        active['reset_identity'] = {key:result[key] for key in
            ('initial_state_vector_sha256', 'initial_observation_sha256')}

    def choose(self, wrapped):
        """Wrap the existing choose verbatim; metadata never draws any noise."""
        def choose(observation):
            from .vla_libero import structure_digest
            active, learner = self.active, wrapped.learner
            identity = behavior_identity(learner, self.contract)
            if identity != active['behavior'] or learner.updates != active['before_updates']:
                raise ValueError('Full acting state changed inside a fixed-behavior source episode')
            private = dict(sampler=deepcopy(wrapped.sampler.bit_generator.state), agent_rng=learner.agent._rng)
            rng_sha = structure_digest(private)
            noise = wrapped.choose(observation)
            pending = wrapped.pending
            if (pending['updates_before'] != active['before_updates']
                    or pending['replay_before'] != active['replay_before']+len(active['choices'])):
                raise ValueError('Original pre-action source/replay clock changed')
            evidence = dict(source_behavior_identity=deepcopy(identity), source_choice_rng_before_sha256=rng_sha)
            pending.update(evidence)
            active['choices'].append(dict(evidence, noise_sha256=structure_digest(noise)))
            return noise
        return choose

    def before_sgd(self, learner, sampler, motor, rows, summary):
        from .vla_libero import structure_digest
        active = self.active
        if (not rows or not rows[-1]['done'] or len(rows) != len(active['choices'])
                or len(learner.replay) != active['replay_before']+len(rows)
                or learner.updates != active['before_updates']
                or behavior_identity(learner, self.contract) != active['behavior']
                or any(summary[k] != v for k,v in active['reset_identity'].items())):
            raise ValueError('Closed behavior episode differs before the original SGD burst')
        for row, choice in zip(rows, active['choices'], strict=True):
            if structure_digest(row['executed_noise']) != choice['noise_sha256']:
                raise ValueError('Actual source noise differs from original pre-action evidence')
            row.update({k:v for k,v in choice.items() if k != 'noise_sha256'})
        active.update(chunks=len(rows), primitive_steps=int(summary['primitive_steps']),
            replay_after_behavior=len(learner.replay),
            rng_pre_sgd=_new_pickle(active['directory']/'rng-before-sgd.pkl', rng_state(learner, sampler, motor)))

    def after_sgd(self, learner, sampler, motor, metrics):
        active = self.active
        after = int(learner.updates)
        expected = active['before_updates']+(active['chunks']*20 if len(learner.replay)>500 else 0)
        if after != expected or len(learner.replay) != active['replay_after_behavior']:
            raise ValueError('Actual post-episode source SGD/replay count differs')
        identity = dict(actor_state_sha256=None, actor_parameter_sha256=None) if after == 0 else actor_identity(learner)
        active.update(after_updates=after, after_identity=identity,
            rng_after=_new_pickle(active['directory']/'rng-after.pkl', rng_state(learner, sampler, motor)),
            update_metrics=deepcopy(metrics))
        return dict(compact_history_format=FORMAT, behavior_identity=deepcopy(active['behavior']),
            actor_after_parameter_sha256=identity['actor_parameter_sha256'],
            actor_state_after_sha256=identity['actor_state_sha256'],
            rng_before_ref=active['rng_before'], rng_pre_sgd_ref=active['rng_pre_sgd'], rng_after_ref=active['rng_after'],
            reset_ref=active['reset_ref'], compact_episode_clocks=dict(
                before_updates=active['before_updates'], pre_sgd_updates=active['before_updates'], after_updates=after,
                replay_before=active['replay_before'], replay_after=len(learner.replay),
                chunks=active['chunks'], primitive_steps=active['primitive_steps']),
            compact_update_metrics=deepcopy(metrics))

    def commit(self, trace, receipt, payload):
        active = self.active
        if (receipt['trace_sha256'] != reference(trace)['sha256']
                or receipt['source_updates'] != active['after_updates']
                or receipt['episode_in_case'] != active['episode_in_case']
                or receipt['case_id'] != active['case']['case_id']):
            raise ValueError('Original committed raw trace/receipt differs from compact episode')
        record = dict(source_only=True, source_data_from_targets=False, format=FORMAT,
            lineage_id=self.contract['history_id'], source_contract_sha256=content_hash(self.contract),
            ordinal=active['ordinal'], raw_trace=reference(trace),
            behavior_identity=payload['behavior_identity'], actor_after_parameter_sha256=payload['actor_after_parameter_sha256'],
            actor_state_after_sha256=payload['actor_state_after_sha256'], rng_before=payload['rng_before_ref'],
            rng_pre_sgd=payload['rng_pre_sgd_ref'], rng_after=payload['rng_after_ref'], reset_ref=payload['reset_ref'],
            clocks=payload['compact_episode_clocks'], update_metrics=payload['compact_update_metrics'],
            before_recovery=None, after_recovery=None)
        ref = _new_json(active['directory']/'episode.json', record)
        self.cache[ref['sha256']] = record
        self.active = None
        return ref

    def publish(self, receipts, *, completed=False):
        """Derive metadata from the EXISTING authoritative recovery receipt cursor."""
        from .vla_libero import write_json
        records = []
        for ordinal, receipt in enumerate(receipts):
            ref = receipt.get('compact_episode_ref')
            if ref is None:
                raise ValueError('Cannot retrofit compact identities onto a legacy source episode')
            path = checked(ref)
            record = self.cache.get(ref['sha256'])
            if record is None:
                record = json.loads(path.read_text()); self.cache[ref['sha256']] = record
            expected = dict(path=str((self.output/receipt['trace']).resolve()), sha256=receipt['trace_sha256'])
            if (record.get('format') != FORMAT or record.get('ordinal') != ordinal
                    or record.get('lineage_id') != self.contract['history_id']
                    or record.get('source_contract_sha256') != content_hash(self.contract)
                    or record['raw_trace'] != expected):
                raise ValueError('Compact cursor differs from original committed raw source chronology')
            records.append(record)
        self.cursor = len(records)
        value = dict(format=FORMAT, source_only=True, source_data_from_targets=False,
            lineage_id=self.contract['history_id'], source_contract_sha256=content_hash(self.contract),
            completed=bool(completed), episodes=records)
        path = self.root/('chronology-complete.json' if completed else 'chronology-live.json')
        if completed:
            if path.exists():
                if json.loads(path.read_text()) != value:
                    raise ValueError('Immutable final compact chronology differs')
                return reference(path)
            return _new_json(path, value)
        write_json(path, value)
        return reference(path)

    def resume(self, receipts, learner, sampler, motor):
        """Validate the ORIGINAL full recovery cursor before collecting anything."""
        from .vla_libero import structure_digest
        self.publish(receipts)
        if not receipts:
            if learner.updates != 0 or learner.replay:
                raise ValueError('Empty compact source cursor has nonempty learning state')
            return
        ref = receipts[-1]['compact_episode_ref']
        record = json.loads(checked(ref).read_text())
        actual = rng_state(learner, sampler, motor)
        with checked(record['rng_after']).open('rb') as stream:
            expected = pickle.load(stream)
        identity = dict(actor_state_sha256=None, actor_parameter_sha256=None) if learner.updates == 0 else actor_identity(learner)
        if (record['clocks']['after_updates'] != learner.updates
                or record['clocks']['replay_after'] != len(learner.replay)
                or record['actor_state_after_sha256'] != identity['actor_state_sha256']
                or record['actor_after_parameter_sha256'] != identity['actor_parameter_sha256']
                or structure_digest(actual) != structure_digest(expected)):
            raise ValueError('Restored compact source actor/RNG/replay/clock identity differs')
