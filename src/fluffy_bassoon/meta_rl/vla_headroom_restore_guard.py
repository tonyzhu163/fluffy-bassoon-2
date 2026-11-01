"""Exact saved-state checks for the diagnosed headroom MsgPack order mismatch.

This changes recovery admission only. Sampling, updates and the existing raw
within-process policy signature remain owned by the original DSRL driver.
"""
import hashlib
import pickle


def restore_checked(learner, sampler, motor, checkpoint, progress, contract):
    """Bind the legacy byte signature, then verify every restored state leaf."""
    import jax
    import numpy as np
    import random
    from flax import serialization
    from . import vla_target_online_dsrl as onset
    from .vla_libero import state_differences, structure_digest

    model = progress['model']
    if onset.sha256(checkpoint) != model['sha256']:
        raise ValueError('Task57 full recovery checksum differs')
    with checkpoint.open('rb') as stream:
        payload = pickle.load(stream)
    legacy_signature = hashlib.sha256(payload['agent']).hexdigest()
    if legacy_signature != model['signature']:
        raise ValueError('Stored agent bytes differ from legacy cursor signature')
    saved = learner.restore(checkpoint, sampler, contract)
    onset._restore_motor(motor, saved['motor_rng'])
    if (saved['training'] != progress['training'] or learner.updates != model['updates']
            or len(learner.replay) != model['replay_rows']):
        raise ValueError('Atomic task57 cursor/model/training recovery differs')
    expected = serialization.msgpack_restore(payload['agent'])
    actual = serialization.to_state_dict(learner.agent._save_dict)
    differences = state_differences(expected, actual, rtol=0., atol=0.)
    if differences:
        raise ValueError('Restored actor/critic/target/temperature/optimizer leaves differ: '
                         + repr(differences[:4]))
    # The existing restore must also preserve the non-network learning state.
    checks = dict(
        replay=structure_digest(payload['replay']) == structure_digest(learner.replay),
        agent_rng=not state_differences(payload['agent_rng'], np.asarray(jax.device_get(learner.agent._rng))),
        sampler=structure_digest(payload['sampler']) == structure_digest(sampler.bit_generator.state),
        numpy_rng=structure_digest(payload['numpy_rng']) == structure_digest(np.random.get_state()),
        python_rng=payload['python_rng'] == random.getstate(),
        motor_rng=structure_digest(saved['motor_rng']) == structure_digest(onset._save_motor(motor)))
    if not all(checks.values()):
        raise ValueError('Restored replay/RNG differs: ' + repr(checks))
    proof = dict(format='headroom_exact_restore_identity_v1',
        checkpoint_sha256=model['sha256'], legacy_signature=legacy_signature,
        restored_raw_signature=onset._signature(learner),
        canonical_signature=structure_digest(actual), exact_leaf_differences=0,
        non_network_checks=checks, optimizer_updates_during_restore=0)
    return saved, proof


def panel_signature_matches(actual, expected, proof):
    """Only alias the proven saved byte identity to its exact restored identity."""
    return actual == expected or (proof is not None
        and actual == proof['restored_raw_signature']
        and expected == proof['legacy_signature'])


def main(argv=None):
    """Allocated CPU dry-run of this production guard; no VLA, SGD or physics."""
    import argparse
    import json
    import os
    from pathlib import Path
    import socket
    import subprocess
    import sys
    from types import SimpleNamespace
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('profile', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--profile-sha256', required=True)
    args = parser.parse_args(argv)
    if sys.platform != 'linux' or not socket.gethostname().startswith('job-') or os.environ.get('JAX_PLATFORMS') != 'cpu':
        raise RuntimeError('Guard dry-run requires identified allocated Linux with JAX_PLATFORMS=cpu')
    from . import vla_target_online_dsrl as onset
    if onset.sha256(args.profile) != args.profile_sha256:
        raise ValueError('Guard profile SHA differs')
    profile = json.loads(args.profile.read_text())
    source_root = Path(__file__).resolve().parents[3]
    execution_source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source_root, text=True).strip()
    if (profile['execution_source'] != execution_source or
            subprocess.check_output(['git', 'status', '--porcelain'], cwd=source_root, text=True).strip()):
        raise ValueError('Guard dry-run requires exact clean execution source')
    paths = {}
    for key in ('checkpoint', 'cursor', 'contract'):
        ref = profile[key]; path = Path(ref['path'])
        if not path.is_absolute() or path.is_symlink() or onset.sha256(path) != ref['sha256']:
            raise ValueError('Guard input SHA differs: ' + key)
        paths[key] = path
    before = {key: onset.sha256(path) for key, path in paths.items()}
    import jax
    import numpy as np
    if any(device.platform != 'cpu' for device in jax.devices()):
        raise RuntimeError('Guard dry-run may use CPU only')
    learner = onset.OfficialSAC(Path(profile['dsrl_root']), 0)
    sampler = np.random.default_rng(0)
    motor = SimpleNamespace(policy=SimpleNamespace(_rng=jax.random.PRNGKey(0)))
    progress = json.loads(paths['cursor'].read_text())
    saved, proof = restore_checked(learner, sampler, motor, paths['checkpoint'], progress,
                                  json.loads(paths['contract'].read_text()))
    after = {key: onset.sha256(path) for key, path in paths.items()}
    if before != after:
        raise RuntimeError('Guard dry-run changed an original input')
    result = dict(passed=True, execution_source=execution_source, proof=proof, input_SHA_before=before, input_SHA_after=after,
        updates=learner.updates, replay_rows=len(learner.replay),
        training_episodes=saved['training']['episodes'], new_SGD=0, new_robot_actions=0,
        VLA_constructed=False, simulator_constructed=False)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')


if __name__ == '__main__':
    main()
