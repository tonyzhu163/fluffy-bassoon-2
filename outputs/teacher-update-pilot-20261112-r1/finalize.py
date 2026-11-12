"""Join qualified immutable branches and audit teacher-to-native data binding."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    for name in ('run','judgments','output'):
        ap.add_argument('--'+name,type=Path,required=True)
    args = ap.parse_args()
    args.output.mkdir(exist_ok=False)
    first,last = args.run/'native04/data',args.run/'native05/data'
    assert read(args.run/'native05/exit.json')['exit_code'] == 0
    a,b = read(first/'branches.json'),read(last/'branches.json')
    key = lambda r:(r['seed'],r['query'],r['candidate'])
    assert len(a)==13 and len(b)==3 and not {key(r) for r in a} & {key(r) for r in b}
    rows = [dict(r,source_native_run=run) for data,run in ((a,'native04'),(b,'native05')) for r in data]
    rows.sort(key=key)
    expected = {(s,q,c) for s in range(120700,120704) for q in (100,250) for c in (0,1)}
    assert {key(r) for r in rows} == expected
    panel = args.output/'panel'
    shutil.copytree(first,panel)
    (panel/'events.jsonl').rename(panel/'source-native04-events.jsonl')
    shutil.copy2(last/'events.jsonl',panel/'source-native05-events.jsonl')
    for row in b:
        for suffix in ('.json','.npz','-final.png'):
            name = f"branch-{row['query']}-{row['candidate']}"+suffix
            target = panel/str(row['seed'])/name
            assert not target.exists()
            shutil.copy2(last/str(row['seed'])/name,target)
    prefix_checks = 0
    for row in rows:
        assert row['fork_exact'] and row['global_rng_restored']
        assert row['baseline_continuation_exact'] == (row['candidate']==0)
        q = read(panel/str(row['seed'])/f"query-{row['query']}.json")
        with np.load(panel/str(row['seed'])/f"branch-{row['query']}-{row['candidate']}.npz") as trace:
            np.testing.assert_array_equal(trace['commands'][:5],q['candidate_commands'][row['candidate']])
            assert len(trace['commands']) == row['controls'] == 500-row['query']
            assert bool(trace['successes'].any()) == row['success_once']
            assert bool(trace['successes'][-1]) == row['final_success']
            assert float(trace['rewards'].sum()) == row['native_return']
        prefix_checks += row['prefix_state_checks']
    calls = read(args.judgments)
    assert len(calls)==72
    keys = {(r['seed'],r['query'],r['repetition'],r['condition']) for r in calls}
    assert keys == {(s,q,rep,c) for s in range(120700,120704) for q in (100,250)
                    for rep in range(3) for c in ('minus','plus','donor')}
    image_checks = 0
    context_signatures = {}
    usage_tokens = {'input_tokens':0,'cached_input_tokens':0,'output_tokens':0}
    item_types = set()
    for r in calls:
        record_dir = args.run/'teacher-records'/f"astra-{r['seed']}"/(
            f"{r['seed']}-{r['query']}-r{r['repetition']}-{r['condition']}")
        assert sha(record_dir/'prompt.txt') == r['prompt_sha256']
        context = json.loads((record_dir/'prompt.txt').read_text().split('\n\n',1)[1])
        assert set(context) == {'task','current_control','control_budget','current_state','state_order',
                               'candidate_commands','action_order','controller','history'}
        history = context.pop('history')
        if r['condition']=='minus':
            assert isinstance(history,str) and 'unavailable' in history
        else:
            assert set(history) == {'relationship','task','image_controls','state_controls','states','command_start','commands'}
            assert len(history['commands'])==50 and len(history['states'])==51
            assert max(history['image_controls'])==r['query']
            assert history['command_start']==r['query']-50
        context_signatures[(r['seed'],r['query'],r['repetition'],r['condition'])] = json.dumps(context,sort_keys=True)
        for line in (record_dir/'events.jsonl').read_text().splitlines():
            entry = json.loads(line)
            if 'item' in entry:
                item_types.add(entry['item']['type'])
            if entry.get('type')=='turn.completed':
                for key_name in usage_tokens:
                    usage_tokens[key_name] += entry.get('usage',{}).get(key_name,0)
        assert sha(panel/str(r['seed'])/f"query-{r['query']}.json") == r['source_query_sha256']
        assert len(r['images']) == (1 if r['condition']=='minus' else 7)
        for item in r['images']:
            relative = Path(*Path(item['path']).parts[-3:])
            assert sha(panel/relative) == item['sha256']
            assert int(relative.stem) <= r['query']
            image_checks += 1
    for s in range(120700,120704):
        for q in (100,250):
            for rep in range(3):
                group = [r for r in calls if (r['seed'],r['query'],r['repetition']) == (s,q,rep)]
                assert len({tuple(r['order']) for r in group})==1
                assert len({r['images'][0]['sha256'] for r in group})==1
                assert len({context_signatures[(s,q,rep,r['condition'])] for r in group})==1
    assert item_types <= {'agent_message','reasoning'}, item_types
    usage = {}
    for name in ('native01','native02','native03','render-gc01','native04','native05'):
        record = read(args.run/name/'exit.json')
        usage[name] = dict(record,worker_gpu_hours=(record['ended']-record['started'])/3600)
    wall = sum(usage[k]['ended']-usage[k]['started'] for k in ('native04','native05'))
    (panel/'branches.json').write_text(json.dumps(rows,indent=2)+'\n')
    (panel/'complete.json').write_text(json.dumps(dict(scenes=4,queries=8,branches=rows,wall_seconds=wall,
        clock='native04 plus native05 worker walls, including probes/loading and rejected reset'),indent=2)+'\n')
    judgment_dir = args.output/'judgments'
    judgment_dir.mkdir()
    shutil.copy2(args.judgments,judgment_dir/'complete.json')
    audit = dict(qualified_branches=16,complete_candidate_zero_replays=8,
        prefix_state_comparisons=prefix_checks,delivered_candidate_prefixes_verified=16,
        unique_teacher_calls=72,teacher_image_hash_checks=image_checks,
        matched_query_and_candidate_order_triples=24,unqualified_initial_resets_in_panel_run=1,
        public_only_prompts_verified=72,external_teacher_tool_calls=0,
        teacher_item_types=sorted(item_types),teacher_tokens=usage_tokens,
        worker_runs=usage,total_worker_gpu_hours=sum(x['worker_gpu_hours'] for x in usage.values()),
        conservative_reservation_gpu_hours=(max(x['ended'] for x in usage.values())-min(x['started'] for x in usage.values()))/3600,
        teacher_summary_sha256=sha(args.judgments))
    (args.output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    subprocess.run([sys.executable,str(Path(__file__).with_name('analyze.py')),'--native',str(panel),
        '--judgments',str(judgment_dir),'--output',str(args.output/'summary.json')],check=True)
    print(json.dumps(audit,indent=2))


if __name__ == '__main__':
    main()
