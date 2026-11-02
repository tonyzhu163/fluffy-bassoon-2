"""Prepare a20-video failure-focused sample on allocated Linux CPUs."""
import argparse
import ast
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')


def select(rows):
    selected=[];used=set();failure_cases=set()
    def take(group,pool,n,*,low_candidate=False,prefer_match=False):
        pool=sorted(pool,key=lambda r:(
            0 if prefer_match and (r['task_key'],r['truth']['case']['native_seed']) in failure_cases else 1,
            r['candidate']['expected_score'] if low_candidate else -r['saved_original']['expected_score'],r['id']))
        chosen=[]
        # Alternate native instruction strings where possible, rather than
        # selecting only the left drawer from one score extreme.
        instructions=sorted({r['native_instruction'] for r in pool})
        for distinct in (True,False):
            while len(chosen)<n:
                progress=False
                for instruction in instructions:
                    candidates=[r for r in pool if r['id'] not in used and r['native_instruction']==instruction
                        and (not distinct or r['truth']['success'] or
                             (r['task_key'],r['truth']['case']['native_seed']) not in failure_cases)]
                    if not candidates:continue
                    row=candidates[0];used.add(row['id']);chosen.append(row);progress=True
                    if not row['truth']['success']:failure_cases.add((row['task_key'],row['truth']['case']['native_seed']))
                    if len(chosen)==n:break
                if not progress:break
            if len(chosen)==n:break
        if len(chosen)!=n:raise ValueError(f'Insufficient cases for{group}:{len(chosen)}/{n}')
        selected.extend(dict(group=group,pair=row) for row in chosen)
    drawer=[r for r in rows if r['dataset']=='TRAIN' and r['task_key']=='OpenDrawer']
    fail=[r for r in drawer if not r['truth']['success']]
    original=lambda r:r['rerun_original']['score']==5
    revised=lambda r:r['candidate']['score']==5
    take('drawer_persistent_false_full',[r for r in fail if original(r) and revised(r)],6)
    take('drawer_new_false_full',[r for r in fail if not original(r) and revised(r)],2)
    take('drawer_corrected_failure',[r for r in fail if original(r) and not revised(r)],2)
    good=[r for r in drawer if r['truth']['success']]
    take('drawer_success_lost',[r for r in good if original(r) and not revised(r)],2,low_candidate=True,prefer_match=True)
    take('drawer_success_retained',[r for r in good if original(r) and revised(r)],2,prefer_match=True)
    stove=[r for r in rows if r['task_key']=='TurnOffStove']
    take('stove_new_false_full',[r for r in stove if not r['truth']['success'] and not original(r) and revised(r)],4)
    take('stove_success_control',[r for r in stove if r['truth']['success'] and revised(r)],2,prefer_match=True)
    assert len(selected)==20 and sum(x['pair']['truth']['success'] for x in selected)==6
    return selected


def source_audit(root):
    native=root/'setups/robocasa365-native-20261101-r1/sources/RoboCasa'
    paths=[native/'robocasa/environments/kitchen/atomic/kitchen_drawer.py',
           root/'deps/nvidia-flora-129a2ada-20261101-r1/repo/rewarder/batch_utils/core.py']
    paths.extend(path for path in sorted((native/'robocasa/models/fixtures').glob('*.py'))
                 if 'def get_door_state' in path.read_text())
    sources=[]
    for path in paths:
        source=path.read_text();tree=ast.parse(source);functions=[]
        if path.name in ('kitchen_drawer.py','core.py'):functions=[dict(name='whole_file',source=source)]
        else:
            for node in ast.walk(tree):
                if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in (
                    'get_door_state','_messages','terminal_messages','score_probabilities','terminal_regex','parse_terminal'):
                    functions.append(dict(name=node.name,line=node.lineno,source=ast.get_source_segment(source,node)))
        sources.append(dict(path=str(path),sha256=sha(path),functions=functions))
    return sources


def frames(selected,out):
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw
    canvases=[]
    # Ten diagnostic tiles/sheet; full video inputs remain untouched.
    for sheet in range(2):
        canvas=Image.new('RGB',(1920,1050),'white');draw=ImageDraw.Draw(canvas)
        for slot,item in enumerate(selected[sheet*10:(sheet+1)*10]):
            pair=item['pair'];x=(slot%4)*480;y=(slot//4)*350
            draw.text((x+5,y+2),f"{sheet*10+slot+1:02d} {item['group']}",fill='black')
            draw.text((x+5,y+16),pair['native_instruction'],fill='black')
            for view,video in enumerate(pair['videos']):
                assert sha(video['path'])==video['sha256']
                with imageio.get_reader(video['path'],format='ffmpeg') as stream:
                    for row,index in enumerate((0,pair['horizon'])):
                        image=Image.fromarray(stream.get_data(index)).resize((160,150))
                        canvas.paste(image,(x+160*view,y+40+150*row))
        path=out/f'failure-sample-{sheet+1}.jpg';canvas.save(path,quality=90)
        canvases.append(dict(name=path.name,sha256=sha(path),base64=base64.b64encode(path.read_bytes()).decode()))
    write(out/'CONTACTSHEETS.json',canvases)


def main():
    assert platform.system()=='Linux'
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(exist_ok=False)
    path=args.root/'outputs/robocasa-flora-rubric-replay-20261102-r1/PAIRED-EPISODES.json'
    selected=select(read(path))
    write(args.output/'SELECTION.json',dict(source=dict(path=str(path),sha256=sha(path)),
        sample_size=20,failures=14,success_controls=6,groups=dict(Counter(x['group'] for x in selected)),
        selected=selected,scope='failure-enriched development sample; no population accuracy inference'))
    public=[dict(id=x['pair']['id'],task_key=x['pair']['task_key'],task=x['pair']['native_instruction'],
                 horizon=x['pair']['horizon'],videos=x['pair']['videos']) for x in selected]
    write(args.output/'PUBLIC-VIDEOS.json',public)
    write(args.output/'SOURCE-AUDIT.json',source_audit(args.root))
    frames(selected,args.output)
    print(json.dumps(dict(complete=True,sample=20,failures=14,success_controls=6,new_robot_actions=0)))


if __name__=='__main__':main()
