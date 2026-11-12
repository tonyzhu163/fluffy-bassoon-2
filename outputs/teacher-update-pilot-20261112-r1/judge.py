"""Fresh-context Astra selections; local client orchestrates remote inference."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import tempfile
import time

CONDITIONS = ('minus', 'plus', 'donor')
SCHEMA = dict(type='object', properties={
    'choice': dict(type='string', enum=['A', 'B', 'abstain']),
    'evidence_summary': dict(type='string')}, required=['choice', 'evidence_summary'], additionalProperties=False)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact(value):
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, list):
        return [compact(v) for v in value]
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items()}
    return value


def prepare(data, output):
    seeds = (120700, 120701, 120702, 120703)
    jobs = []
    for si, seed in enumerate(seeds):
        for qi, query in enumerate((100, 250)):
            source = data/str(seed)/f'query-{query}.json'
            q = json.loads(source.read_text())
            donor_seed = seeds[(si+1) % len(seeds)]
            donor_source = data/str(donor_seed)/f'query-{query}.json'
            donor = json.loads(donor_source.read_text())
            orders = [[0,1], [1,0], [0,1] if (si+qi) % 2 == 0 else [1,0]]
            random.Random(7131+seed+query).shuffle(orders)
            for rep in range(3):
                order = orders[rep]
                order_conditions = CONDITIONS[rep:] + CONDITIONS[:rep]
                for condition in order_conditions:
                    directory = output/f'{seed}-{query}-r{rep}-{condition}'
                    directory.mkdir()
                    images = [data/str(seed)/'public'/f'{query:04d}.png']
                    context = dict(task=q['prompt'], current_control=query, control_budget=q['budget'],
                        current_state=q['current_state'], state_order=list(q['current_state']),
                        candidate_commands={name:q['candidate_commands'][idx] for name,idx in zip(('A','B'),order)},
                        action_order=q['action_order'], controller=q['controller'])
                    if condition == 'minus':
                        context['history'] = 'The previous 50 controls and observations are unavailable.'
                    else:
                        h = q if condition == 'plus' else donor
                        hseed = seed if condition == 'plus' else donor_seed
                        context['history'] = dict(
                            relationship=('Actual immediately preceding experience in this same episode.' if condition == 'plus'
                                          else 'A DIFFERENT episode and scene, not the current robot\'s past. Use only as background.'),
                            task=h['prompt'], image_controls=h['history_controls'],
                            state_controls=list(range(query-50, query+1)),
                            states=h['history_states'], command_start=query-50, commands=h['history_commands'])
                        images += [data/str(hseed)/'public'/f'{step:04d}.png' for step in h['history_controls']]
                    prompt = (
                        'You are selecting a robot action in native RoboCasa OpenDrawer. Choose A or B to maximize '
                        'native task success by the 500-control budget. Each candidate is exactly five consecutive '
                        '20Hz commands proposed by the same competent frozen pi0.5 policy. After these five controls, '
                        'the same policy resumes closed-loop control every five controls with matched random draws. '
                        'You cannot inspect candidate futures. Both candidates are legitimate proposals; use abstain '
                        'when the observations do not justify a preference. Do not invent hidden object geometry. '
                        'Arm commands are normalized delta control in the robot base frame, with the listed controller '
                        'output scales, NOT absolute Cartesian positions. Gripper_close >=0.5 means close and '
                        'base_mode >=0.5 selects the native base mode. All other commands follow native Human300 '
                        'conventions. The first image is the CURRENT left/wrist/right camera triptych. Any following '
                        'images are the explicitly described chronological history, also left/wrist/right. '
                        'All provided states and commands are public observations/actions; no success feedback is '
                        'provided. Give your choice and a brief one- or two-sentence observation-grounded evidence summary.\n\n'
                        + json.dumps(compact(context), separators=(',', ':')))
                    (directory/'prompt.txt').write_text(prompt)
                    metadata = dict(seed=seed, query=query, repetition=rep, condition=condition,
                        order=order, images=[dict(path=str(p.resolve()),sha256=sha(p)) for p in images],
                        prompt_sha256=sha(directory/'prompt.txt'), source_query_sha256=sha(source),
                        donor_source_sha256=sha(donor_source) if condition == 'donor' else None)
                    (directory/'input.json').write_text(json.dumps(metadata, indent=2)+'\n')
                    jobs.append(directory)
    return jobs


def run_one(directory, args, schema, version):
    input_record = json.loads((directory/'input.json').read_text())
    started = time.monotonic()
    record = dict(**input_record, model_requested='gpt-6-astra', client_version=version,
        model_identity_source='explicit CLI model selection; no independent backend attestation',
        transport='signed-in Codex CLI', start_utc=datetime.now(timezone.utc).isoformat())
    with tempfile.TemporaryDirectory(prefix='teacher-judgment-') as cwd:
        command = [args.codex, 'exec', '--ignore-user-config', '--ephemeral', '--skip-git-repo-check',
            '--json', '--sandbox', 'read-only', '-C', cwd, '-m', 'gpt-6-astra',
            '-c', 'approval_policy="never"', '-c', 'model_reasoning_effort="medium"',
            '-c', 'web_search="disabled"', '-c', 'features.skip_host_skill_discovery=true']
        for feature in ('shell_tool', 'apps', 'plugins', 'browser_use', 'computer_use', 'multi_agent', 'image_generation'):
            command.extend(['--disable', feature])
        for item in input_record['images']:
            command.extend(['-i', item['path']])
        command.extend(['--output-schema', str(schema), '-o', str(directory/'final.json'), '-'])
        with (directory/'events.jsonl').open('x') as events, (directory/'client.log').open('x') as log:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=events, stderr=log,
                                       text=True, start_new_session=True)
            try:
                process.communicate((directory/'prompt.txt').read_text(), timeout=240)
                record['exit_code'] = process.returncode
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                record['exit_code'] = 'timeout'
    record['wall_seconds'] = time.monotonic()-started
    record['valid'] = False
    if record['exit_code'] == 0:
        try:
            answer = json.loads((directory/'final.json').read_text())
            assert answer['choice'] in ('A','B','abstain') and isinstance(answer['evidence_summary'],str)
            record.update(valid=True, answer=answer, candidate=(None if answer['choice']=='abstain'
                else input_record['order'][('A','B').index(answer['choice'])]))
        except (ValueError, KeyError, AssertionError, FileNotFoundError) as error:
            record['parse_error'] = str(error)
    (directory/'result.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({k:record.get(k) for k in ('seed','query','repetition','condition','valid','candidate','wall_seconds')}),flush=True)
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--codex', required=True)
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--limit', type=int)
    args = ap.parse_args()
    args.data, args.output = args.data.resolve(), args.output.resolve()
    args.output.mkdir(exist_ok=False, parents=True)
    schema = args.output/'schema.json'
    schema.write_text(json.dumps(SCHEMA))
    jobs = prepare(args.data,args.output)
    if args.limit is not None:
        jobs = jobs[:args.limit]
    version = subprocess.check_output([args.codex,'--version'],text=True).strip()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda p: run_one(p,args,schema,version), jobs))
    (args.output/'complete.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__ == '__main__':
    main()
