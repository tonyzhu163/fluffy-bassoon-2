"""Static Git/AST comparison; does not import a simulator or model."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

repo = Path(sys.argv[1])
out = Path(sys.argv[2])
old = '00dd765dc868be7ca78f14adf706ce713f9f868c'
new = '29f7ce8814c1547f5af762a0997fbd4b64848dd7'
path = 'robocasa/utils/dataset_registry.py'


def read(ref):
    return subprocess.check_output(['git', '-C', str(repo), 'show', f'{ref}:{path}'], text=True)


def parse(source):
    horizons, tasks = {}, {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        name = node.targets[0].id if isinstance(node.targets[0], ast.Name) else ''
        for keyword in node.value.keywords:
            if name == 'TARGET_TASKS':
                tasks[keyword.arg] = ast.literal_eval(keyword.value)
            elif isinstance(keyword.value, ast.Call):
                for field in keyword.value.keywords:
                    if field.arg == 'horizon':
                        horizons[keyword.arg] = ast.literal_eval(field.value)
    return horizons, tasks


old_source, new_source = read(old), read(new)
before, old_tasks = parse(old_source)
after, new_tasks = parse(new_source)
assert before.keys() == after.keys()
assert old_tasks == new_tasks
assert all(after[k] * 2 == before[k] * 3 for k in before)
changed = subprocess.check_output(['git', '-C', str(repo), 'diff', '--name-only', old, new], text=True).splitlines()
assert set(changed) == {'README.md', 'robocasa/__init__.py', path, 'setup.py'}
all_tasks = sum(new_tasks.values(), [])
assert len(all_tasks) == len(set(all_tasks)) == 50
result = dict(old_commit=old, new_commit=new, changed_files=changed,
              registry_horizons=len(before), all_exactly_1_5x=True,
              task_sets_unchanged=True, task_counts={k: len(v) for k, v in new_tasks.items()},
              source_sha256={old: hashlib.sha256(old_source.encode()).hexdigest(),
                             new: hashlib.sha256(new_source.encode()).hexdigest()},
              tasks=[dict(task=t, category=k, old_horizon=before[t], new_horizon=after[t])
                     for k, names in new_tasks.items() for t in names])
out.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'tasks'}, indent=2))
