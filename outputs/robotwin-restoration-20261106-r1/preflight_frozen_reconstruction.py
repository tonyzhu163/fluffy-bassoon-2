"""Run the exact launcher admission and all actual input checks on Linux CPU."""
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

import torch

from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha, write


def main():
    assert os.uname().sysname == 'Linux' and os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert not torch.cuda.is_initialized()
    root = Path(__file__).resolve().parent
    code, output = root.parents[1], Path(os.environ['TASK_OUT'])
    assert output.name == 'cpureconstruction-preflight01'
    assert subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip() == os.environ['TASK_SOURCE']
    assert not subprocess.check_output(['git', '-C', str(code), 'status', '--porcelain', '--untracked-files=no'])
    output.mkdir()
    started = time.monotonic()
    report = dict(status='checking', source_commit=os.environ['TASK_SOURCE'],
        gpu_seconds=0, model_calls=0, optimizer_updates=0, simulator_actions=0,
        launcher_gpu_probe_not_run=True, actual_gpu_allocation_not_qualified_by_cpu=True)
    try:
        launcher = root/'reconstruct_frozen_base.sh'
        subprocess.run(['bash', '-n', str(launcher)], check=True)
        body = launcher.read_text().split("<<'PY'\n")[1].split('\nPY')[0]
        sys.argv = [str(__file__), str(root), str(launcher)]
        exec(compile(body, str(launcher)+':admission', 'exec'), {'__name__':'__cpu_admission__'})
        entry = runpy.run_path(str(root/'reconstruct_frozen_base.py'))
        entry['stage'](validate_payloads=True)
        inputs = json.loads((output/'INPUTS.json').read_bytes())
        physical = json.loads((output/'PHYSICAL-INPUTS.json').read_bytes())
        assert inputs['all_payloads_validated'] and len(inputs['packets']) == 96
        assert physical['status'] == 'passed' and physical['all_1024_episodes_complete']
        assert not torch.cuda.is_initialized()
        report.update(status='passed', exact_launcher_admission_block=True,
            launcher_sha256=sha(launcher), entry_sha256=sha(root/'reconstruct_frozen_base.py'),
            input_report_sha256=inputs['input_report_sha256'], inputs_sha256=sha(output/'INPUTS.json'),
            physical_inputs_sha256=sha(output/'PHYSICAL-INPUTS.json'),
            payloads=96, issued_command_rows=6144, complete_physical_episodes=1024,
            physical_decisions=physical['physical_decisions'],
            post_terminal_noops=physical['post_terminal_noops'],
            total_input_load_seconds=physical['total_input_load_seconds'],
            maximum_input_load_seconds=physical['maximum_input_load_seconds'],
            cuda_initialized=False, source_corpus_admitted=False)
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['seconds'] = time.monotonic()-started
        write(output/'QUALIFICATION.json', report)
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
