"""Score both native goals in the same Pro2 world, independently of the option."""
from multiprocessing.connection import Connection
from pathlib import Path
import sys
import traceback

import numpy as np
from libero_remote.environment.libero import LiberoEnvironmentAdapter
from libero_remote.environment.task_manifest import TaskManifest
from libero.libero.envs.bddl_utils import robosuite_parse_problem

ROOT = Path('/mowenjie/fluffy-bassoon-2/sources/libero-pro2-e5500c69')


def serve(pipe, task_ids=(4, 8)):
    manifest = TaskManifest.load(str(ROOT / 'assets/task_manifest.json'))
    env = LiberoEnvironmentAdapter()
    try:
        while True:
            op, value = pipe.recv()
            if op == 'reset':
                env.close()
                scene, variant = value
                paths = [manifest.resolve_bddl('libero_goal', task, variant) for task in task_ids]
                problems = [robosuite_parse_problem(str(p)) for p in paths]
                goals = [p['goal_state'] for p in problems]
                prompts = [' '.join(p['language_instruction']) if isinstance(p['language_instruction'], list)
                           else p['language_instruction'] for p in problems]
                initial = env.create(dict(suite='libero_goal', task_id=task_ids[0], variant=variant,
                    init_state_id=scene, seed=18180100 + scene, _bddl_path=str(paths[0]),
                    camera_names=['agentview', 'robot0_eye_in_hand'], camera_height=256,
                    camera_width=256, control_mode='relative', max_steps=300))
                world = env._env.env
                assert world.parsed_problem['goal_state'] == goals[0]
                assert not any(all(world._eval_predicate(term) for term in g) for g in goals)
                assert initial['language_instruction'].lower() == prompts[0].lower()
                result = initial['observation'], initial['environment_reproducibility']['initial_state_fingerprint'], prompts
            elif op == 'chunk':
                outcomes, states = [], []
                for action in value:
                    step = env.step(action)
                    hits = [all(world._eval_predicate(term) for term in g) for g in goals]
                    assert hits[0] == step['success']
                    outcomes.append(hits)
                    raw = step['observation']
                    states.append(np.concatenate([raw[k] for k in
                        ('robot0_eef_pos','robot0_eef_quat_xyzw','robot0_gripper_qpos')]))
                result = raw, states, outcomes
            else:
                raise ValueError(op)
            pipe.send((True, result))
    except EOFError:
        pass
    except Exception:
        pipe.send((False, traceback.format_exc()))
    finally:
        env.close()


if __name__ == '__main__':
    with Connection(int(sys.argv[1])) as connection:
        tasks = tuple(map(int, sys.argv[2:])) if len(sys.argv) > 2 else (4, 8)
        assert len(tasks) == 2
        serve(connection, tasks)
