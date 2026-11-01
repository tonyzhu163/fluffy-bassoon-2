"""Source-only independent lineages -> the existing seven-array history format.

No collector, learner, scheduler or grants live here. Metadata admission is
stdlib-only; projection and frozen teacher inference require an explicitly named
allocated Linux host. Existing source40 admission remains mandatory.

Late-bound ABI (no concrete scientific choices are supplied by this module):
  declaration = {format: DECLARATION, source_only: true,
    source_data_from_targets: false, source_training_task_keys: ALL40,
    source_validation_task_keys: ALL40 or the existing fixed8,
    motor: MOTOR, feature_contract: the exact shared prefix-v2 contract,
    noise_contract: NOISE, teacher_gate: null or {paired_cases, minimum_success_rate},
    ad_improvement_gate: null or {lineage_ids, paired_cases, minimum_success_gain},
    locked_query_cases: [{task_key, initial_state,
      scene_seed}], lineages: [{lineage_id, task_key, condition_id, partition,
      learner_seed, cases, label_teacher_lineage_id}]}
  bindings = [{lineage_id, index: REF, motor_before: REF, motor_after: REF, chronology: REF,
               teacher_qualification: REF or null}]
REF is {path: absolute immutable file path, sha256: actual byte SHA256}.
lineage_id must equal the ORIGINAL source contract history_id. cases must equal
that original contract's declared cases, not a post-hoc filtered subset.
condition_id is audit-only and must match the original contract (null for an
ordinary unconditioned collector). Never add it to observations.

Only selected TRAIN label-teacher bindings carry teacher_qualification. Its
JSON binds lineage_id, task_key, condition_id, checkpoint_sha256,
actor_parameter_sha256, actor_state_sha256, evidence: REF,
sampling_mode='deterministic_mean', source_only=true,
source_data_from_targets=false, and paired_cases=[{case_id, task_key,
initial_state, scene_seed, baseline_success, teacher_success}]. The fixed panel
must be separate from every declared collection/query case.

chronology JSON: {source_only:true, source_data_from_targets:false, lineage_id,
  episodes:[{raw_trace:REF, behavior_identity:{kind, actor_not_used,
    actor_parameter_sha256, actor_state_sha256, actor_state_scope,
    source_contract_sha256, sampler_configuration}, actor_after_parameter_sha256,
    actor_state_after_sha256, rng_before:REF, rng_pre_sgd:REF, rng_after:REF,
    reset_ref:REF, clocks, before_recovery:REF|null,
    after_recovery:REF|null}]} in ORIGINAL episode order. The original episode
payload itself must bind behavior_identity, actor_after_parameter_sha256,
rng_before_ref and rng_after_ref; missing historical identities cannot be
retrofitted. Warmup kind='gaussian_warmup', actor_not_used=true, actor SHA=null;
SAC kind='sac', actor_not_used=false and actual full actor TrainState AND params
SHAs are required. V2 additionally binds actor_state_scope, the original source
contract hash, actor_state_after_sha256, rng_pre_sgd_ref and reset_ref. V1 params-
only evidence is not upgraded or considered equivalent.
Warmup sampler_configuration is WARMUP below; SAC sampler_configuration=null.
RNG refs are compact original pickles with sampler, agent_rng, numpy_rng,
python_rng and motor_rng (the existing jax_motor_key_v1 encoding). Full recovery
refs per episode are OPTIONAL retained archives, never rolling/latest aliases.
The original selected final full recovery remains mandatory through load_source.

AD additionally binds ad_improvement_ref to source-only original empirical
paired_cases=[{lineage_id, case_id, task_key, initial_state, scene_seed,
baseline_success, teacher_success}], lineage_ids and evidence:REF. Its assessed
lineage IDs must equal the prospectively declared assessment scope. This scope
never filters the ALL40 data. DPT competence does NOT require positive gain.

All paths, slots, seeds, reset banks, qualifications, model roots, source commit,
host and execution authorization remain UNBOUND until a separate reviewed run.
"""
from copy import deepcopy
import gzip
import hashlib
import json
import math
from pathlib import Path
import pickle
import re
import socket
import subprocess
import sys

from .vla_source40_schedule import SOURCE_SUITES, content_hash
from .vla_source40_compact_evidence import FORMAT as COMPACT_FORMAT, ACTOR_SCOPE, PROTOCOL

DECLARATION = 'source40_independent_lineage_declaration_v2'
ASSEMBLY = 'source40_independent_lineage_assembly_v2'
PI05_RECEIPT = 'ec064cce32f5003e2e3979c5712d7d6385018f6bfcc59cb2161105881e70d170'
OPENPI_COMMIT = '215abfb217dbac7d5f1273282331b9b1866c0479'
MOTOR_PARAMETER_SHA256 = 'c565dd3f0c4005be728cf7597698cba937c4011cd5bd813700a89b789f9d69e9'
MOTOR = dict(checkpoint_receipt_sha256=PI05_RECEIPT,
    parameter_sha256=MOTOR_PARAMETER_SHA256, model_config='pi05_libero',
    native_horizon=10, execute_steps=10, noise_dim=32, flow_steps=10)
NOISE = '32D tied across native H10; warmup unbounded N(0,I), then SAC clipped[-1,1]; no rescaling'
WARMUP = dict(distribution='standard_normal', shape=[32], dtype='float32',
              expansion='tied across native H10; no clipping or rescaling')
ALL40 = frozenset(f'{suite}:{i:02d}' for suite in SOURCE_SUITES for i in range(10))
FIXED8 = frozenset(f'{suite}:{i:02d}' for suite in SOURCE_SUITES for i in (0, 1))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def reference(ref):
    if (not isinstance(ref, dict) or set(ref) != {'path', 'sha256'}
            or not isinstance(ref['path'], str) or not Path(ref['path']).is_absolute()
            or not isinstance(ref['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', ref['sha256'])
            or sha(ref['path']) != ref['sha256']):
        raise ValueError('Require an original immutable absolute file reference')
    return Path(ref['path'])


def _source_role(value):
    """Reject original DEV records even if somebody wraps them in a source index."""
    if (value.get('target_only') is True or value.get('source_assembly_admission') is False
            or value.get('source_only') is False or value.get('source_data_from_targets') is True
            or ('data_role' in value and value['data_role'] != 'source')):
        raise ValueError('Target/DEV or repurposed-role records cannot become source')


def _case_key(value):
    return value['task_key'], value['initial_state'], value['scene_seed']


def validate_declaration(value):
    """Metadata-only completeness checks; supplies no execution authorization."""
    _source_role(value)
    train, val = value['source_training_task_keys'], value['source_validation_task_keys']
    if (value.get('format') != DECLARATION or value.get('source_only') is not True
            or value.get('source_data_from_targets') is not False
            or len(train) != 40 or set(train) != ALL40
            or len(val) != len(set(val)) or frozenset(val) not in (ALL40, FIXED8)
            or value.get('motor') != MOTOR or value.get('noise_contract') != NOISE):
        raise ValueError('Require original ALL40 TRAIN/full-or-fixed8 VAL and exact OpenPI40 C565 H10/32D motor')
    feature = value['feature_contract']
    if (feature.get('kind') != 'openpi_final_prefix_visual_mean_language_mean_public_state_v2'
            or feature.get('checkpoint_receipt_sha256') != PI05_RECEIPT
            or feature.get('openpi_commit') != OPENPI_COMMIT
            or feature.get('model_config') != 'pi05_libero' or feature.get('observation_dim') != 4104
            or any(feature.get(k) != v for k,v in dict(visual_dim=2048, instruction_dim=2048,
                public_state_dim=8, pooling_dtype='float32', kv_cache_retained=False).items())
            or feature.get('pooling') != 'separate NumPy float32 mask-weighted means after native hidden materialization'
            or feature.get('hidden_materialization') != 'standalone module_jit output transferred to host before reduction'
            or feature.get('trainable_feature_parameters') != 0
            or feature.get('normalization') != 'raw frozen hidden values; student fits normalization on training tasks only'):
        raise ValueError('Require one shared frozen public4104 prefix contract; no teacher-specific encoder')
    gate = value['teacher_gate']
    if gate is not None and (type(gate['paired_cases']) is not int or gate['paired_cases'] < 1
            or type(gate['minimum_success_rate']) not in (int, float)
            or not math.isfinite(gate['minimum_success_rate']) or not 0 <= gate['minimum_success_rate'] <= 1):
        raise ValueError('Bind actual paired teacher competence threshold prospectively')
    slots = value['lineages']
    if (not slots or len({s['lineage_id'] for s in slots}) != len(slots)
            or len({s['learner_seed'] for s in slots}) != len(slots)):
        raise ValueError('Whole independent lineages require unique IDs and learner seeds')
    by_id = {s['lineage_id']: s for s in slots}
    improvement = value.get('ad_improvement_gate')
    if improvement is not None:
        chosen = improvement['lineage_ids']
        minimum = improvement['minimum_success_gain']
        if (not chosen or len(set(chosen)) != len(chosen)
                or any(key not in by_id or by_id[key]['partition'] != 'train' for key in chosen)
                or type(improvement['paired_cases']) is not int or improvement['paired_cases'] < 1
                or type(minimum) not in (int, float) or not math.isfinite(minimum) or not 0 <= minimum <= 1):
            raise ValueError('AD empirical assessment scope/threshold must be explicit TRAIN lineage choices')
    locked = {_case_key(c) for c in value['locked_query_cases']}
    if not locked or len(locked) != len(value['locked_query_cases']):
        raise ValueError('Bind a nonempty unique locked query bank prospectively')
    cases, states = {p: set() for p in ('train', 'validation')}, {p: set() for p in ('train', 'validation')}
    for slot in slots:
        task, partition = slot['task_key'], slot['partition']
        if (partition not in cases or task not in ALL40 or not slot['lineage_id']
                or slot['condition_id'] is not None and not isinstance(slot['condition_id'], str)
                or type(slot['learner_seed']) is not int or not 0 <= slot['learner_seed'] < 2**32
                or not slot['cases']):
            raise ValueError('Require typed original source40 task, partition, seed and complete reset bank')
        teacher_id = slot['label_teacher_lineage_id']
        teacher = by_id.get(teacher_id)
        if teacher_id is not None and (teacher is None or teacher['partition'] != 'train'
                or (teacher['task_key'], teacher['condition_id']) != (task, slot['condition_id'])
                or teacher['label_teacher_lineage_id'] != teacher['lineage_id']):
            raise ValueError('TRAIN and VAL must bind the same predeclared same-task/condition TRAIN teacher')
        if len({c['case_id'] for c in slot['cases']}) != len(slot['cases']):
            raise ValueError('Repeated case IDs inside one lineage')
        for case in slot['cases']:
            if (case.get('split') != partition or type(case['initial_state']) is not int
                    or case['initial_state'] not in (range(40) if partition == 'train' else range(40, 50))
                    or type(case['scene_seed']) is not int or not 0 <= case['scene_seed'] < 2**31
                    or type(case['episodes']) is not int or case['episodes'] < 2
                    or case.get('learner_reset') is not False):
                raise ValueError('Preserve complete original case/state banks and persistent learner')
            key = (task, case['initial_state'], case['scene_seed'])
            if key in locked:
                raise ValueError('Locked target-query case entered source collection')
            cases[partition].add((task, case['case_id']))
            states[partition].add((task, case['initial_state']))
    if cases['train'] & cases['validation'] or states['train'] & states['validation']:
        raise ValueError('Source case/state overlap across whole-lineage split')
    if ({s['task_key'] for s in slots if s['partition'] == 'train'} != ALL40
            or {s['task_key'] for s in slots if s['partition'] == 'validation'} != set(val)):
        raise ValueError('Every declared ALL40/full-or-fixed8 task must have actual lineage slots')
    return value


def _linux(expected_hostname, source_commit):
    if sys.platform != 'linux' or socket.gethostname() != expected_hostname:
        raise ValueError('Projection/label inference requires the explicitly allocated Linux host')
    root = Path(__file__).resolve().parents[3]
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() != source_commit
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip()):
        raise ValueError('Require the clean reviewed adapter commit')


def _load_source(slot, binding, declaration):
    """Keep the old source40 motor/task/case/state/complete-prefix guards intact."""
    from .vla_source40_training import load_source, _member
    index_path = reference(binding['index'])
    original = json.loads(index_path.read_text())
    contract = json.loads((index_path.parent / 'contract.json').read_text())
    _source_role(original); _source_role(contract)
    source = load_source(index_path, slot['partition'])
    if (contract.get('compact_history_evidence') != PROTOCOL
            or binding.get('chronology') != original.get('compact_history_evidence')):
        raise ValueError('Require original opt-in V2 full-actor evidence; no legacy metadata retrofit')
    if (source['task'] != slot['task_key'] or source['history_id'] != slot['lineage_id']
            or contract.get('learner_seed') != slot['learner_seed']
            or contract.get('condition_id') != slot['condition_id']
            or contract.get('cases') != slot['cases']
            or source['feature_contract'] != declaration['feature_contract']):
        raise ValueError('Original source contract differs from prospectively declared whole lineage')
    if any(contract['motor'].get(k) != v for k,v in dict(openpi_commit=OPENPI_COMMIT,
            checkpoint='gs://openpi-assets/checkpoints/pi05_libero', flow_steps=10,
            predicted_horizon=10, action_dim=32, weights_frozen=True).items()):
        raise ValueError('Original source motor runtime/config differs from exact OpenPI40')
    recipe = contract.get('source_recipe', {})
    if any(recipe.get(k) != v for k, v in
           dict(warmup_chunks=500, batch_size=256, utd=20, gamma_per_primitive=.999).items()):
        raise ValueError('Require the original published source recipe, with post-episode SAC bursts')
    for key in ('motor_before', 'motor_after'):
        fingerprint = json.loads(reference(binding[key]).read_text())
        if fingerprint.get('sha256') != MOTOR_PARAMETER_SHA256:
            raise ValueError('Actual entire-motor fingerprint differs from officialOpenPI40 C565')
    source.update(contract=contract, index=original, root=index_path.parent, slot=slot,
                  contract_ref=dict(path=str(index_path.parent / 'contract.json'),
                                    sha256=sha(index_path.parent / 'contract.json')))
    source['dataset_manifest_refs'] = [dict(path=str(_member(source['root'], a['path']).with_suffix('.json')),
        sha256=sha(_member(source['root'], a['path']).with_suffix('.json')))
        for a in source['selected']['datasets'].values()]
    return source


def _same(left, right):
    import numpy as np
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(_same(left[k], right[k]) for k in left)
    return np.array_equal(left, right)


def _rows(source, *, require_learning=True):
    """Check closed raw episodes; never reconstruct chronology from replay alone.

    source_updates_after is the per-step pre-SGD clock. The distinct
    source_updates_after_episode is the real post-complete-episode burst clock.
    Both must survive unchanged in the original raw references/audit sidecar.
    """
    import numpy as np
    from .vla_source40_training import _member
    rows, episodes, previous_after, replay = [], [], 0, 0
    for receipt in source['index']['episode_receipts']:
        path = _member(source['root'], receipt['trace'])
        if sha(path) != receipt['trace_sha256']:
            raise ValueError('Original raw episode checksum changed')
        with gzip.open(path, 'rb') as stream:
            episode = pickle.load(stream)
        _source_role(episode); _source_role(episode['summary'])
        if episode.get('source_contract_sha256') != content_hash(source['contract']):
            raise ValueError('Raw episode was not originally generated under this source contract')
        case = next(c for c in source['slot']['cases'] if c['case_id'] == receipt['case_id'])
        summary, actual = episode['summary'], episode['rows']
        if (episode.get('case') != case or not actual or len(actual) != receipt['chunks']
                or any(summary[k] != receipt[k] for k in ('initial_state', 'scene_seed',
                    'initial_state_vector_sha256', 'initial_observation_sha256', 'success', 'primitive_steps'))):
            raise ValueError('Raw closed episode/reset identity differs from original receipt')
        after, total = receipt['source_updates'], 0
        before = actual[0]['source_updates_before']
        if type(before) is not int or before != previous_after or type(after) is not int or after < before:
            raise ValueError('Learning lineage was reset, skipped or reordered between episodes')
        for i, row in enumerate(actual):
            _source_role(row)
            duration = row['primitive_steps']
            done = i == len(actual) - 1
            noise = np.asarray(row['executed_noise'])
            features = np.asarray(row['public_features'])
            if (row.get('source_partition') != source['slot']['partition']
                    or row.get('source_task_key') != source['slot']['task_key']
                    or row.get('case_id') != receipt['case_id']
                    or row.get('source_episode_in_case') != receipt['episode_in_case']
                    or episode.get('compact_history_format') != COMPACT_FORMAT
                    or row.get('source_behavior_identity') != episode.get('behavior_identity')
                    or not re.fullmatch('[0-9a-f]{64}', row.get('source_choice_rng_before_sha256', ''))
                    or row['source_updates_before'] != before or row['source_updates_after'] != before
                    or row['source_updates_after_episode'] != after
                    or row['source_replay_before'] != replay or row['source_replay_after'] != replay + 1
                    or row['source_stage'] != ('warmup_gaussian' if before == 0 else 'sac_online')
                    or type(row['done']) not in (bool, np.bool_) or bool(row['done']) != done
                    or bool(row['terminated']) != bool(row['success'])
                    or bool(row['truncated']) != (done and not bool(row['success']))
                    or bool(row['success']) and not done
                    or type(duration) is not int or not 1 <= duration <= 10
                    or noise.dtype != np.float32 or noise.shape != (32,) or not np.isfinite(noise).all()
                    or features.dtype != np.float32 or features.shape != (4104,) or not np.isfinite(features).all()
                    or not _same(row['proposed_noise'], noise)
                    or not _same(row['native_initial_noise'], np.tile(noise, (10, 1)))
                    or before > 0 and (np.abs(noise) > 1).any()
                    or np.asarray(row['executed_actions']).shape != (duration, 7)
                    or not np.isfinite(row['executed_actions']).all()
                    or len(row['raw_rewards']) != duration
                    or not np.isfinite(row['raw_rewards']).all()
                    or row['reward'] != (0. if row['success'] else -1.)
                    or not math.isclose(row['primitive_discount'], .999 ** duration)
                    or set(row['motor_observation']) != {'image', 'wrist', 'state'}
                    or set(row['motor_next_observation']) != {'image', 'wrist', 'state'}):
                raise ValueError('Raw behavior/noise/reward/boundary/pre-vs-post-SGD chronology differs')
            if i and (not _same(actual[i-1]['motor_next_observation'], row['motor_observation'])
                      or not _same(actual[i-1]['sac_next_observation'], row['sac_observation'])):
                raise ValueError('Actual next observation does not reach the next chronological decision')
            total += duration; replay += 1
        expected_after = before + (20 * len(actual) if replay > 500 else 0)
        if after != expected_after:
            raise ValueError('Post-episode SGD count differs from replay>500/chunks*20 recipe')
        if (total != summary['primitive_steps'] or total > 400
                or bool(actual[-1]['success']) != summary['success']):
            raise ValueError('Closed episode primitive count differs')
        rows.extend(actual)
        episodes.append(dict(raw_trace=dict(path=str(path), sha256=receipt['trace_sha256']),
            case_id=receipt['case_id'], episode_in_case=receipt['episode_in_case'],
            initial_state=summary['initial_state'], scene_seed=summary['scene_seed'],
            initial_state_vector_sha256=summary['initial_state_vector_sha256'],
            initial_observation_sha256=summary['initial_observation_sha256'],
            behavior_updates_before=before, behavior_updates_after_episode=after,
            behavior_identity=episode.get('behavior_identity'),
            actor_after_parameter_sha256=episode.get('actor_after_parameter_sha256'),
            actor_state_after_sha256=episode.get('actor_state_after_sha256'),
            rng_before_ref=episode.get('rng_before_ref'), rng_after_ref=episode.get('rng_after_ref'),
            rng_pre_sgd_ref=episode.get('rng_pre_sgd_ref'), reset_ref=episode.get('reset_ref'),
            compact_episode_clocks=episode.get('compact_episode_clocks'),
            chunks=len(actual), primitive_steps=total))
        previous_after = after
    if (previous_after != source['index']['source_updates']
            or require_learning and len({r['source_updates_before'] for r in rows if r['source_updates_before'] > 0}) < 2):
        raise ValueError('Require complete advancing source behavior with at least two learned acting versions')
    from .vla_libero import structure_digest
    source['expected_replay_sha256'] = structure_digest([dict(observations=r['sac_observation'],
        next_observations=r['sac_next_observation'], actions=r['executed_noise'][None].copy(),
        next_actions=r['executed_noise'][None].copy(), rewards=np.float32(r['reward']),
        masks=np.float32(not r['success']), discount=np.float32(.999 ** r['primitive_steps'])) for r in rows])
    return rows, episodes


def _qualification(source, binding, declaration):
    value = json.loads(reference(binding['teacher_qualification']).read_text())
    _source_role(value)
    reference(value['evidence'])
    slot = source['slot']
    if (slot['partition'] != 'train' or value.get('source_only') is not True or value.get('source_data_from_targets') is not False
            or any(value.get(k) != slot[k] for k in ('lineage_id', 'task_key', 'condition_id'))
            or value.get('checkpoint_sha256') != source['selected']['source_checkpoint_sha256']
            or not re.fullmatch('[0-9a-f]{64}', value.get('actor_parameter_sha256', ''))
            or not re.fullmatch('[0-9a-f]{64}', value.get('actor_state_sha256', ''))
            or value.get('sampling_mode') != 'deterministic_mean'):
        raise ValueError('Require the exact prospectively selected qualified TRAIN mean-label actor')
    panel, gate = value['paired_cases'], declaration['teacher_gate']
    reserved = {_case_key(c) for c in declaration['locked_query_cases']}
    reserved.update((s['task_key'], c['initial_state'], c['scene_seed'])
                    for s in declaration['lineages'] for c in s['cases'])
    if (len(panel) != gate['paired_cases'] or len({c['case_id'] for c in panel}) != len(panel)
            or len({_case_key(c) for c in panel}) != len(panel)
            or any(c['task_key'] != slot['task_key'] or _case_key(c) in reserved
                   or type(c['baseline_success']) is not bool or type(c['teacher_success']) is not bool for c in panel)):
        raise ValueError('Qualification requires original fresh paired cases separate from source/query')
    final = sum(c['teacher_success'] for c in panel) / len(panel)
    if final < gate['minimum_success_rate']:
        raise ValueError('Declared DPT teacher competence threshold failed')
    return value


def _ad_improvement(ref, declaration):
    """An explicit empirical assessment; it cannot remove ANY source lineage."""
    gate = declaration.get('ad_improvement_gate')
    if gate is None or ref is None:
        raise ValueError('AD needs its separate original empirical improvement receipt and declared assessment')
    value = json.loads(reference(ref).read_text())
    _source_role(value)
    reference(value['evidence'])
    slots = {s['lineage_id']:s for s in declaration['lineages']}
    rows = value['paired_cases']
    reserved = {_case_key(c) for c in declaration['locked_query_cases']}
    reserved.update((s['task_key'], c['initial_state'], c['scene_seed'])
                    for s in declaration['lineages'] for c in s['cases'])
    if (value.get('source_only') is not True or value.get('source_data_from_targets') is not False
            or set(value['lineage_ids']) != set(gate['lineage_ids'])
            or len(value['lineage_ids']) != len(gate['lineage_ids']) or len(rows) != gate['paired_cases']
            or {r['lineage_id'] for r in rows} != set(gate['lineage_ids'])
            or len({(r['lineage_id'], r['case_id']) for r in rows}) != len(rows)
            or any(r['task_key'] != slots[r['lineage_id']]['task_key'] or _case_key(r) in reserved
                   or type(r['baseline_success']) is not bool or type(r['teacher_success']) is not bool for r in rows)):
        raise ValueError('Original AD paired outcomes differ from the declared source TRAIN assessment')
    gain = (sum(r['teacher_success'] for r in rows)-sum(r['baseline_success'] for r in rows))/len(rows)
    if gain < gate['minimum_success_gain']:
        raise ValueError('Declared empirical AD improvement threshold failed')
    return dict(reference=deepcopy(ref), evidence=value['evidence'], measured_success_gain=gain,
                assessed_lineage_ids=value['lineage_ids'], retained_data_scope='ALL40; no filtering by assessment outcome')


def _teacher(source, qualification, dsrl_root):
    """Restore the EXISTING frozen SAC label actor; no update or simulation."""
    import numpy as np
    from flax import serialization
    from .vla_libero import OfficialSAC, structure_digest
    from .vla_source40_training import _member
    path = _member(source['root'], source['selected']['source_checkpoint'])
    expected_sha = qualification['checkpoint_sha256']
    if sha(path) != expected_sha:
        raise ValueError('Qualified TRAIN recovery changed')
    learner = OfficialSAC(dsrl_root, source['slot']['learner_seed'])
    learner.restore(path, np.random.default_rng(0), source['contract'])
    actor = structure_digest(serialization.to_state_dict(learner.agent._actor.params))
    full_actor = structure_digest(serialization.to_state_dict(learner.agent._actor))
    if (actor != qualification['actor_parameter_sha256'] or full_actor != qualification['actor_state_sha256']
            or learner.updates != source['index']['source_updates']):
        raise ValueError('Restored TRAIN teacher parameters/clock differ from qualification')
    signature = structure_digest(serialization.to_state_dict(learner.agent._save_dict))
    rng = np.asarray(learner.agent._rng).copy()

    def evaluate(observations):
        labels = np.asarray(learner.agent.eval_actions(observations), np.float32).reshape(-1, 32)
        if (structure_digest(serialization.to_state_dict(learner.agent._save_dict)) != signature
                or not np.array_equal(np.asarray(learner.agent._rng), rng)):
            raise RuntimeError('Frozen deterministic label inference changed teacher/RNG/recovery')
        return labels
    return evaluate


def _recovery(ref, contract):
    """Inspect an original full checkpoint without constructing a learner."""
    from flax import serialization
    from .vla_libero import structure_digest
    with reference(ref).open('rb') as stream:
        value = pickle.load(stream)
    if (value.get('contract') != contract or value.get('boundary') != 'complete physical episode'
            or any(value.get(k) is None for k in ('agent', 'agent_rng', 'replay', 'updates',
                                                 'sampler', 'numpy_rng', 'python_rng', 'progress'))
            or value['progress'].get('motor_rng', {}).get('format') != 'jax_motor_key_v1'
            or type(value['updates']) is not int or value['updates'] < 0
            or not isinstance(value['replay'], list)):
        raise ValueError('Require original full source recovery with optimizer/replay/sampler/all RNG state')
    state = serialization.msgpack_restore(value['agent'])
    if (not {'actor', 'critic', 'target_critic_params', 'temp'} <= state.keys()
            or any(not {'params', 'opt_state', 'step'} <= state[k].keys() for k in ('actor', 'critic', 'temp'))):
        raise ValueError('Inference weights are not full source optimizer/train state')
    reference(ref)
    return dict(updates=value['updates'], replay_rows=len(value['replay']),
        actor_parameter_sha256=structure_digest(state['actor']['params']),
        actor_state_sha256=structure_digest(state['actor']),
        train_state_sha256=hashlib.sha256(value['agent']).hexdigest(),
        replay_sha256=structure_digest(value['replay']),
        rng_sha256=structure_digest({**{k:value[k] for k in
            ('agent_rng', 'sampler', 'numpy_rng', 'python_rng')}, 'motor_rng':value['progress']['motor_rng']}))


def _compact_rng(ref):
    from .vla_libero import structure_digest
    with reference(ref).open('rb') as stream:
        value = pickle.load(stream)
    if (any(value.get(k) is None for k in ('sampler', 'agent_rng', 'numpy_rng', 'python_rng', 'motor_rng'))
            or value['motor_rng'].get('format') != 'jax_motor_key_v1'):
        raise ValueError('Require original compact sampler/global/private/motor RNG evidence')
    reference(ref)
    return structure_digest({k:value[k] for k in ('sampler', 'agent_rng', 'numpy_rng', 'python_rng', 'motor_rng')})


def _chronology(source, binding, episodes, *, require_learning=True):
    value = json.loads(reference(binding['chronology']).read_text())
    _source_role(value)
    records = value['episodes']
    if (value.get('format') != COMPACT_FORMAT or value.get('completed') is not True
            or value.get('source_contract_sha256') != content_hash(source['contract'])
            or value.get('source_only') is not True or value.get('source_data_from_targets') is not False
            or value.get('lineage_id') != source['slot']['lineage_id'] or len(records) != len(episodes)):
        raise ValueError('Bind every original complete episode to its compact behavior/RNG evidence')
    previous_rng, previous_actor, previous_state, result, versions, replay = None, None, None, [], set(), 0
    for episode, record in zip(episodes, records, strict=True):
        _source_role(record)
        if (record['raw_trace'] != episode['raw_trace']
                or record['behavior_identity'] != episode['behavior_identity']
                or record['actor_after_parameter_sha256'] != episode['actor_after_parameter_sha256']
                or record['actor_state_after_sha256'] != episode['actor_state_after_sha256']
                or record['rng_before'] != episode['rng_before_ref']
                or record['rng_pre_sgd'] != episode['rng_pre_sgd_ref'] or record['reset_ref'] != episode['reset_ref']
                or record['rng_after'] != episode['rng_after_ref']):
            raise ValueError('Recovery chronology reordered or substituted an original raw episode')
        before, after, behavior = episode['behavior_updates_before'], episode['behavior_updates_after_episode'], record['behavior_identity']
        if not isinstance(behavior, dict):
            raise ValueError('Missing original behavior identity cannot be retrofitted')
        actor = behavior['actor_parameter_sha256']
        state = behavior.get('actor_state_sha256')
        after_actor = record['actor_after_parameter_sha256']
        after_state = record['actor_state_after_sha256']
        warming = before == 0
        if (behavior.get('kind') != ('gaussian_warmup' if warming else 'sac')
                or type(behavior.get('actor_not_used')) is not bool or behavior['actor_not_used'] != warming
                or behavior.get('sampler_configuration') != (WARMUP if warming else None)
                or behavior.get('actor_state_scope') != ACTOR_SCOPE
                or behavior.get('source_contract_sha256') != content_hash(source['contract'])
                or warming and (actor is not None or state is not None)
                or not warming and (not isinstance(actor, str) or not re.fullmatch('[0-9a-f]{64}', actor))
                or not warming and (not isinstance(state, str) or not re.fullmatch('[0-9a-f]{64}', state))
                or after == 0 and (after_actor is not None or after_state is not None)
                or after > 0 and (not isinstance(after_actor, str) or not re.fullmatch('[0-9a-f]{64}', after_actor))
                or after > 0 and (not isinstance(after_state, str) or not re.fullmatch('[0-9a-f]{64}', after_state))
                or result and (actor != previous_actor or state != previous_state)):
            raise ValueError('Actual SAC parameter content or explicitly actor-unused Gaussian identity differs')
        rng_before, rng_after = _compact_rng(record['rng_before']), _compact_rng(record['rng_after'])
        _compact_rng(record['rng_pre_sgd'])
        reset = json.loads(reference(record['reset_ref']).read_text())
        if (reset.get('returned') is not True or any(reset.get(k) != episode[k] for k in
                ('case_id', 'initial_state', 'scene_seed', 'initial_state_vector_sha256', 'initial_observation_sha256'))
                or record['clocks'] != episode['compact_episode_clocks']
                or record['clocks'] != dict(before_updates=before, pre_sgd_updates=before, after_updates=after,
                    replay_before=replay, replay_after=replay+episode['chunks'], chunks=episode['chunks'],
                    primitive_steps=episode['primitive_steps'])):
            raise ValueError('Original compact reset/EP/post-burst clock evidence differs')
        if previous_rng is not None and rng_before != previous_rng:
            raise ValueError('Actual sampler/global/private/motor RNG reset between episodes')
        for name, updates, count, expected_actor, expected_state, expected_rng in (
                ('before_recovery', before, replay, actor, state, rng_before),
                ('after_recovery', after, replay+episode['chunks'], after_actor, after_state, rng_after)):
            ref = record.get(name)
            if ref is not None:
                if Path(ref['path']).name in ('source-recovery.pkl', 'target-recovery.pkl', 'recovery-0.pkl', 'recovery-1.pkl'):
                    raise ValueError('Optional per-episode full refs must be retained archives, never rolling aliases')
                identity = _recovery(ref, source['contract'])
                if (identity['updates'] != updates or identity['replay_rows'] != count
                        or identity['rng_sha256'] != expected_rng
                        or expected_state is not None and identity['actor_state_sha256'] != expected_state
                        or expected_actor is not None and identity['actor_parameter_sha256'] != expected_actor):
                    raise ValueError('Optional retained full recovery differs from compact original evidence')
        if not warming: versions.add(state)
        result.append(dict(record, behavior_updates_before=before, behavior_updates_after_episode=after,
            replay_before=replay, replay_after=replay+episode['chunks'], rng_before_sha256=rng_before, rng_after_sha256=rng_after))
        previous_actor, previous_state, previous_rng = after_actor, after_state, rng_after
        replay += episode['chunks']
    from .vla_source40_training import _member
    selected = source['selected']
    final_ref = dict(path=str(_member(source['root'], selected['source_checkpoint'])),
                     sha256=selected['source_checkpoint_sha256'])
    final = _recovery(final_ref, source['contract'])
    if (not result or final['updates'] != episodes[-1]['behavior_updates_after_episode']
            or final['replay_rows'] != replay or final['actor_parameter_sha256'] != previous_actor
            or final['actor_state_sha256'] != previous_state
            or final['rng_sha256'] != previous_rng or final['replay_sha256'] != source['expected_replay_sha256']
            or require_learning and len(versions) < 2):
        raise ValueError('Final publication or two actually acting learned source versions differ')
    return result


def _project(source, rows, evaluate):
    import numpy as np
    from .vla_history import FIELDS, validate_trial
    from .vla_source40_corpus import paired_arrays
    from .vla_source40_dpt_same_teacher import sac_input
    if evaluate is None:
        result = dict(ad=paired_arrays(rows, np.stack([r['executed_noise'] for r in rows]))['ad'])
        validate_trial(result['ad'], objective='ad')
        for key in FIELDS:
            if not np.array_equal(result['ad'][key], source['datasets']['ad'][key]):
                raise ValueError('AD projection changed original actual behavior/features/boundaries')
        return result
    labels = []
    for start in range(0, len(rows), 64):
        inputs = [sac_input(r) for r in rows[start:start+64]]
        labels.append(np.asarray(evaluate({k: np.stack([o[k] for o in inputs])
                                          for k in ('pixels', 'state')}), np.float32).reshape(len(inputs), 32))
    labels = np.concatenate(labels)
    if not np.isfinite(labels).all() or (np.abs(labels) > 1).any():
        raise ValueError('Qualified same-interface mean teacher returned invalid 32D labels')
    trials = paired_arrays(rows, labels)
    for objective, trial in trials.items():
        if validate_trial(trial, objective=objective)[1:] != (4104, 32):
            raise ValueError('Require fixed shared4104 observations and actual32D noise coordinates')
        for key in set(FIELDS) - {'target_noise'}:
            if not np.array_equal(trial[key], source['datasets']['ad'][key]):
                raise ValueError('Projection changed original chronological behavior/features or boundaries')
    return trials


def assemble(declaration_ref, bindings, output, *, objective, dsrl_root, expected_hostname,
             source_commit, ad_improvement_ref=None):
    """Late-bound allocated entry; emits normal AD/DPT NPZs for vla_history_train.

    This requires fully bound inputs, and neither acquires resources nor starts
    a fit. The caller must separately authorize and
    qualify execution. All original inputs are rehashed before publication.
    """
    _linux(expected_hostname, source_commit)
    if objective not in ('ad', 'dpt'):
        raise ValueError('Admit AD and DPT separately with an explicit objective')
    declaration = validate_declaration(json.loads(reference(declaration_ref).read_text()))
    if objective == 'dpt' and declaration['teacher_gate'] is None:
        raise ValueError('DPT competence scope/threshold must be bound prospectively')
    improvement = _ad_improvement(ad_improvement_ref, declaration) if objective == 'ad' else None
    from .vla_history import save_dataset
    slots = {s['lineage_id']: s for s in declaration['lineages']}
    if len(bindings) != len(slots) or {b['lineage_id'] for b in bindings} != set(slots):
        raise ValueError('Provide exactly one original input binding for every whole lineage')
    by_id = {b['lineage_id']: b for b in bindings}
    sources = {key: _load_source(slot, by_id[key], declaration) for key, slot in slots.items()}
    # Preserve actual physical-state/case overlap guards across all independent
    # lineages, including multiple seeds for the same task/partition.
    for key in ('history_id', 'case_ids', 'states', 'state_hashes'):
        def identities(partition):
            return {v for s in sources.values() if s['slot']['partition'] == partition
                    for v in ([s[key]] if key == 'history_id' else s[key])}
        if identities('train') & identities('validation'):
            raise ValueError('Source TRAIN/VAL overlap: ' + key)
    prepared = {}
    for key, source in sources.items():
        rows, episodes = _rows(source, require_learning=objective == 'ad')
        chronology = _chronology(source, by_id[key], episodes, require_learning=objective == 'ad')
        prepared[key] = rows, episodes, chronology
    teacher_ids = {s['label_teacher_lineage_id'] for s in slots.values()} if objective == 'dpt' else set()
    if None in teacher_ids:
        raise ValueError('DPT requires all exact TRAIN label teachers bound before admission')
    teachers, qualifications = {}, {}
    for key in sorted(teacher_ids):
        qualifications[key] = _qualification(sources[key], by_id[key], declaration)
        teachers[key] = _teacher(sources[key], qualifications[key], dsrl_root)
        from .vla_source40_dpt_same_teacher import sac_input, TRAIN_LABEL_ATOL
        from .vla_libero import sac_observation
        import numpy as np
        first = prepared[key][0][0]
        obs = sac_input(first)
        if not _same(obs, sac_observation(first['motor_observation'])):
            raise ValueError('Original TRAIN pixel/proprio preprocessing differs from canonical public input')
        label = teachers[key]({k:v[None] for k,v in obs.items()}).reshape(32)
        delta = float(np.max(np.abs(label-sources[key]['datasets']['dpt']['target_noise'][0])))
        if not math.isfinite(delta) or delta > TRAIN_LABEL_ATOL:
            raise ValueError('Exact TRAIN teacher-label replay differs beyond existing backend tolerance')
        qualifications[key] = dict(qualifications[key], train_label_replay=dict(
            max_absolute_delta=delta, tolerance=TRAIN_LABEL_ATOL))
    data = {p: {objective: []} for p in ('train', 'validation')}
    ids, provenance = {p: [] for p in data}, []
    for key in sorted(sources):
        source, slot = sources[key], slots[key]
        rows, episodes, chronology = prepared[key]
        projected = _project(source, rows, teachers[slot['label_teacher_lineage_id']] if objective == 'dpt' else None)
        data[slot['partition']][objective].append(projected[objective])
        ids[slot['partition']].append(key)
        provenance.append(dict(lineage=deepcopy(slot), inputs=deepcopy(by_id[key]),
            original_contract=deepcopy(source['contract_ref']),
            original_dataset_manifests=deepcopy(source['dataset_manifest_refs']),
            episodes=episodes, chronology=chronology, costs=source['costs'],
            label_teacher_lineage_id=slot['label_teacher_lineage_id']))
    reference(declaration_ref)
    if improvement is not None:
        reference(ad_improvement_ref); reference(improvement['evidence'])
    for binding in bindings:
        for name in ('index', 'motor_before', 'motor_after', 'chronology'):
            reference(binding[name])
        if binding['lineage_id'] in teacher_ids:
            reference(binding['teacher_qualification'])
            reference(qualifications[binding['lineage_id']]['evidence'])
    for source in sources.values():
        from .vla_source40_training import _member
        reference(source['contract_ref'])
        for ref in source['dataset_manifest_refs']:
            reference(ref)
        selected = source['selected']
        if sha(_member(source['root'], selected['source_checkpoint'])) != selected['source_checkpoint_sha256']:
            raise ValueError('Original full source recovery changed during projection')
        for artifact in selected['datasets'].values():
            if sha(_member(source['root'], artifact['path'])) != artifact['sha256']:
                raise ValueError('Original source dataset changed during projection')
    for lineage in provenance:
        for episode in lineage['episodes']:
            reference(episode['raw_trace'])
        for boundary in lineage['chronology']:
            reference(boundary['rng_before']); reference(boundary['rng_pre_sgd']); reference(boundary['rng_after'])
            reference(boundary['reset_ref'])
            for key in ('before_recovery', 'after_recovery'):
                if boundary.get(key) is not None: reference(boundary[key])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(motor_sha256=PI05_RECEIPT, motor_parameter_sha256=MOTOR_PARAMETER_SHA256,
        feature_contract=declaration['feature_contract'], noise_contract=NOISE,
        declaration=deepcopy(declaration_ref), source_only=True, source_data_from_targets=False,
        normalization='Fit once on TRAIN observations only; freeze/export existing trainer buffers',
        label_rule='AD actual requests; DPT current target only, same exact qualified TRAIN teacher on TRAIN+VAL',
        teacher_qualifications=qualifications, ad_improvement=improvement,
        history_rule='actual chronological noise/reward/done; task/condition/teacher/update IDs are metadata only')
    result = dict(format=ASSEMBLY, completed=False, source_only=True, objective=objective,
        source_commit=source_commit, declaration=deepcopy(declaration_ref), lineages=provenance,
        datasets={}, additional_simulator_actions=0, source_training_task_keys=sorted(ALL40),
        source_validation_task_keys=declaration['source_validation_task_keys'])
    for partition in data:
        result['datasets'][partition] = {}
        for objective in data[partition]:
            path = output / f'{partition}-{objective}.npz'
            manifest = save_dataset(path, data[partition][objective], objective=objective,
                metadata=dict(metadata, partition=partition, trial_ids=ids[partition]))
            result['datasets'][partition][objective] = dict(path=str(path.resolve()), sha256=manifest['sha256'],
                manifest_sha256=sha(path.with_suffix('.json')))
    result['completed'] = True
    (output / 'assembly.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result
