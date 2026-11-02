"""Actual full-physical attempt protocol through the frozen steering ports.

Native environment construction, canonical snapshots, source binding and the
durable journal stay with the qualified backend. This loop owns only causal
controller memory, real feedback and complete-attempt recovery.
"""
import hashlib

import numpy as np

from .vla_history import FrozenHistoryController


def weights_digest(policy):
    digest = hashlib.sha256()
    for name, tensor in sorted(policy.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def replay_physical_attempt(controller, rows, *, horizon):
    """Rebuild memory from verified complete real transitions, without physics."""
    if len(rows) * 50 != horizon:
        raise ValueError("Only complete bound physical attempts can be replayed")
    rewards = [float(row["reward"]) for row in rows]
    if any(reward not in (0., 1.) for reward in rewards) or sum(rewards) > 1:
        raise ValueError("Replay only actual once-per-attempt first-success feedback")
    for index, row in enumerate(rows):
        if (type(row["done"]) not in (bool, np.bool_)
                or bool(row["done"]) != (index == len(rows) - 1)
                or row.get("executed_controls") != 50
                or row.get("episode_start") != (index == 0)):
            raise ValueError("History done must be the actual physical boundary")
        control = controller.act(row["features"], deterministic=True)
        if not np.array_equal(control, np.asarray(row["control"], np.float32)):
            raise ValueError("Recovered deterministic controller differs from saved executed steering")
        controller.observe(row["control"], row["reward"], row["done"])


def collect_case_attempts(policy, spec, case, *, horizon, environment_factory, motor,
                          noise_for, before_dispatch, after_dispatch, completed_attempt,
                          attempts=5, completed_prefix=()):
    """Collect a case with isolated memory and same-case physical resets.

    ``noise_for(case, attempt, chunk)`` returns explicit FP32 native noise and
    its provenance. Durability callbacks execute before/after every actual
    dispatch. The caller must hash-verify any completed prefix before passing
    it here; partial attempts are never accepted or silently recollected.
    """
    if (type(attempts) is not int or not 1 <= attempts <= 5
            or type(horizon) is not int or horizon < 50 or horizon % 50
            or len(completed_prefix) > attempts
            or policy.config.noise_dim != spec.steering_dim
            or attempts * (horizon // 50) > policy.config.context_length):
        raise ValueError("Bind matching steering dimensions and complete attempts within memory capacity")
    frozen = weights_digest(policy)
    controller = FrozenHistoryController(policy, mode="full")
    results = []
    for attempt, previous in enumerate(completed_prefix, 1):
        if (previous["case"] != case or previous["attempt"] != attempt
                or previous["horizon"] != horizon or not previous["completed"]):
            raise ValueError("Recovery prefix changed physical case or attempt identity")
        replay_physical_attempt(controller, previous["records"], horizon=horizon)
        results.append(previous)
    for attempt in range(len(completed_prefix) + 1, attempts + 1):
        environment = environment_factory(case, attempt)
        records = []
        first_seen = False
        try:
            observation = environment.reset()
            pairing = getattr(environment, "initial_evidence", {}).get("pairing", {})
            if (environment.case != case or environment.horizon != horizon
                    or pairing.get("matched") is not True):
                raise ValueError("Qualified canonical case comparison must pass before any action")
            for chunk in range(horizon // 50):
                features, prepared = motor.prepare(observation)
                features = np.asarray(features, np.float32)
                control = controller.act(features, deterministic=True)
                noise, provenance = noise_for(case, attempt, chunk)
                noise = np.asarray(noise, np.float32)
                if noise.shape != (50, 32) or not np.isfinite(noise).all() or not provenance:
                    raise ValueError("Require explicit finite native Gaussian with saved draw provenance")
                commands = motor.execute(prepared, noise=noise, steering=control, spec=spec)
                commands = np.asarray(commands)
                if commands.shape != (50, 12) or not np.isfinite(commands).all():
                    raise ValueError("Require all finite native H50 motor commands")
                intent = dict(case=case, attempt=attempt, chunk=chunk, horizon=horizon,
                    features=features.copy(), control=control.copy(), native_noise=noise.copy(),
                    native_noise_provenance=provenance, requested_commands=np.array(commands, copy=True))
                before_dispatch(intent, environment)
                begin = environment.steps
                observation, done, _ = environment.step_chunk(commands)
                if environment.steps - begin != 50 or bool(done) != (chunk == horizon // 50 - 1):
                    raise ValueError("Every attempt must execute all bound native chunks, including its suffix")
                seen = bool(environment.success_steps)
                reward = float(seen and not first_seen)
                first_seen = first_seen or seen
                controller.observe(control, reward, bool(done))
                record = dict(intent, reward=reward, done=bool(done),
                              episode_start=chunk == 0, executed_controls=50)
                after_dispatch(record, environment)
                records.append(record)
            first = environment.success_steps[0] if environment.success_steps else None
            result = dict(completed=True, case=case, attempt=attempt, horizon=horizon,
                success_once=first_seen, first_success_step=first,
                success_at_end=bool(environment.success_steps and environment.success_steps[-1] == horizon),
                records=records, memory_tokens=int(controller.state["length"][0]),
                pairing=environment.initial_evidence)
        finally:
            environment.close()
        completed_attempt(result, controller)
        results.append(result)
    if weights_digest(policy) != frozen:
        raise ValueError("Frozen student parameters or normalization buffers changed")
    return results
