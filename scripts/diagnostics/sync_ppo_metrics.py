#!/usr/bin/env python3
"""Mirror scalar PPO metrics over SSH and relay them to W&B from this host.

SSH routing and credentials belong in ~/.config/fluffy-bassoon-2/metrics-relay.json,
outside Git. The remote runs are read-only. No checkpoints, videos, observations,
source, console logs or offline SDK metadata are transferred or uploaded.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import time
import urllib.error
import urllib.request


REMOTE_SNAPSHOT = r'''
import hashlib,json,sys,time
from pathlib import Path
base=Path(sys.argv[1])
result=[]
for run in sorted(base.glob('robocasa-joint-steering-*-scale50-20261103-r*')):
    identity=run/'METRICS-IDENTITY.json'
    metrics=run/'metrics.jsonl'
    export=run/'METRICS-EXPORT.json'
    if not all(p.is_file() for p in (identity,metrics,export)):
        continue
    ident=json.loads(identity.read_text())
    if ident.get('arm') not in ('velocity','kv_all_layers'):
        continue
    raw=metrics.read_bytes()
    # A writer may currently be appending the final line.
    raw=raw.rsplit(b'\n',1)[0]+b'\n' if b'\n' in raw else b''
    config=json.loads(export.read_text())
    permitted=('arm','group','run_id','project','entity','config')
    config={k:config[k] for k in permitted if k in config}
    permitted_config=('arm','updates','episodes_per_update','lanes_per_GPU',
                      'horizon','frozen_VLM','frozen_action_expert')
    config['config']={k:v for k,v in config.get('config',{}).items()
                      if k in permitted_config}
    progress={}
    if (run/'PROGRESS.json').is_file():
        p=json.loads((run/'PROGRESS.json').read_text())
        progress={k:p[k] for k in ('phase','sequence','iteration','last_check_unix') if k in p}
    result.append(dict(name=run.name,identity=ident,export=config,
        progress=progress,metrics=raw.decode(),sha256=hashlib.sha256(raw).hexdigest()))
print(json.dumps(dict(captured_unix=time.time(),runs=result)))
'''


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def mirror(config, root):
    command = ['ssh', '-p', str(config['ssh_port']), '-o', 'BatchMode=yes',
               '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=10',
               '-o', 'ServerAliveCountMax=2', config['ssh_target'],
               'python3 - ' + shlex.quote(config['remote_outputs'])]
    result = subprocess.run(command, input=REMOTE_SNAPSHOT, text=True,
                            capture_output=True, timeout=60)
    if result.returncode:
        raise RuntimeError('SSH metric snapshot failed; check the configured development endpoint')
    snapshot = json.loads(result.stdout)
    if not snapshot['runs']:
        raise RuntimeError('No matching PPO metric streams found at the configured endpoint')
    summary = []
    for run in snapshot['runs']:
        if Path(run['name']).name != run['name']:
            raise ValueError('Invalid source run name')
        raw = run['metrics'].encode()
        if hashlib.sha256(raw).hexdigest() != run['sha256']:
            raise ValueError('Metric snapshot digest mismatch')
        rows = [json.loads(line) for line in run['metrics'].splitlines()]
        sequences = [row['sequence'] for row in rows]
        if sequences != sorted(set(sequences)):
            raise ValueError('Metric sequence is not strictly increasing')
        for row in rows:
            if any(row[k] != run['identity'][k] for k in ('arm','group','run_id')):
                raise ValueError('Metric identity changed inside a stream')
            if not all(isinstance(k,str) and isinstance(v,(int,float)) and math.isfinite(v)
                       for k,v in row['metrics'].items()):
                raise ValueError('Only finite named scalar metrics may be relayed')
        directory = root/'mirror'/run['name']
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory/'metrics.jsonl.tmp'
        temporary.write_bytes(raw)
        temporary.replace(directory/'metrics.jsonl')
        for key, name in [('identity','METRICS-IDENTITY.json'),
                          ('export','METRICS-EXPORT.json'), ('progress','PROGRESS.json')]:
            save(directory/name,run[key])
        summary.append(dict(source=run['name'],arm=run['identity']['arm'],
            run_id=run['identity']['run_id'],rows=len(rows),bytes=len(raw),
            sha256=run['sha256'],last_sequence=sequences[-1] if rows else 0,
            progress=run['progress']))
    save(root/'MIRROR-STATUS.json',dict(captured_unix=snapshot['captured_unix'],
                                     mirrored_unix=time.time(),runs=summary))
    return snapshot,summary


def upload(config, root, snapshot):
    # Never obtain an API key from a run profile or a metric file.
    credential = Path(config['credential_file']).expanduser()
    if credential.exists():
        if credential.stat().st_mode & 0o077:
            raise PermissionError('W&B credential file must have mode 0600')
        os.environ['WANDB_API_KEY'] = credential.read_text().strip()
    if 'WANDB_API_KEY' not in os.environ and not (Path.home()/'.netrc').exists():
        raise RuntimeError('W&B authentication is not installed on this host')
    os.environ.update(WANDB_CONSOLE='off',WANDB_DISABLE_GIT='true',
                     WANDB_DISABLE_CODE='true',WANDB_MODE='online',
                     WANDB_BASE_URL=config['base_url'])
    if config.get('https_proxy'):
        os.environ['HTTPS_PROXY'] = config['https_proxy']
        os.environ['HTTP_PROXY'] = config['https_proxy']
    # Fail quickly on an access-denied endpoint instead of waiting for SDK retries.
    import base64
    key = os.environ.get('WANDB_API_KEY')
    if key:
        request = urllib.request.Request(config['base_url']+'/graphql',
            data=json.dumps({'query':'query { viewer { username } }'}).encode(),
            headers={'Content-Type':'application/json','Authorization':'Basic '+
                     base64.b64encode(('api:'+key).encode()).decode()})
        try:
            with urllib.request.urlopen(request,timeout=15) as response:
                viewer = json.loads(response.read()).get('data',{}).get('viewer')
                if not viewer:
                    raise RuntimeError('W&B endpoint did not confirm authentication')
        except urllib.error.HTTPError as error:
            raise RuntimeError(f'W&B endpoint returned HTTP {error.code}') from None
    import wandb
    state_path = root/'UPLOAD-STATE.json'
    state = read(state_path,{'runs':{}})
    combined = {}
    for run in snapshot['runs']:
        export = run['export']
        if export['project'] != 'fluffy-bassoon-2' or export['entity'] != 'tonyzoo-brown-university':
            raise ValueError('Unexpected W&B destination')
        key = export['entity']+'/'+export['project']+'/'+export['run_id']
        target = combined.setdefault(key,dict(export=export,rows={}))
        for row in map(json.loads,run['metrics'].splitlines()):
            old = target['rows'].get(row['sequence'])
            if old is not None and old != row:
                raise ValueError('Conflicting events for the same W&B run and sequence')
            target['rows'][row['sequence']] = row
    receipts = []
    for key, item in combined.items():
        export,rows = item['export'],item['rows']
        confirmed = state['runs'].get(key,{}).get('last_confirmed_sequence',0)
        pending = [row for seq,row in sorted(rows.items()) if seq > confirmed]
        if not pending:
            receipts.append(dict(run=key,new_rows=0,last_confirmed_sequence=confirmed))
            continue
        options = dict(base_url=config['base_url'],console='off',
            disable_git=True,disable_code=True,x_disable_stats=True,
            x_disable_meta=True,x_disable_machine_info=True,x_disable_viewer=True,
            init_timeout=20,finish_timeout=30)
        if 'finish_timeout_raises' in wandb.Settings.model_fields:
            options['finish_timeout_raises'] = True
        settings = wandb.Settings(**options)
        local_sdk = root/'sdk'
        local_sdk.mkdir(exist_ok=True)
        tracking = wandb.init(id=export['run_id'],project=export['project'],
            entity=export['entity'],group=export['group'],name=export['arm'],
            resume='allow',mode='online',config=export['config'],
            dir=str(local_sdk),settings=settings)
        tracking.define_metric('ppo/update')
        tracking.define_metric('*',step_metric='ppo/update')
        # A previous upload may have reached W&B before our local receipt was saved.
        next_step = tracking.step
        for row in pending:
            if row['sequence'] >= next_step:
                tracking.log(dict(row['metrics'],_timestamp=row['unix']),step=row['sequence'])
        tracking.finish()
        api = wandb.Api(overrides={'base_url':config['base_url']},timeout=20)
        cloud = api.run(key)
        actual = {int(row['event/sequence']):row for row in cloud.scan_history(
            keys=['event/sequence','ppo/update','_step'],page_size=100)}
        for row in pending:
            match = actual.get(row['sequence'])
            if match is None or match['ppo/update'] != row['metrics']['ppo/update']:
                raise RuntimeError('W&B history readback has not confirmed all uploaded events')
        confirmed = max(row['sequence'] for row in pending)
        receipt = dict(run=key,new_rows=len(pending),last_confirmed_sequence=confirmed,
                       verified_unix=time.time(),url=cloud.url)
        state['runs'][key] = receipt
        save(state_path,state)
        receipts.append(receipt)
    return receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path.home()/'.config/fluffy-bassoon-2/metrics-relay.json')
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--local-only',action='store_true')
    args = parser.parse_args()
    args.directory.mkdir(parents=True,exist_ok=True)
    with (args.directory/'.relay.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('Another metric relay invocation is already running')
            return
        status = dict(checked_UTC=datetime.now(timezone.utc).isoformat(),
                      source_runs_read_only=True)
        try:
            config = json.loads(args.config.read_text())
            snapshot,summary = mirror(config,args.directory)
            status.update(mirror='complete',streams=summary)
            if args.local_only:
                status['upload'] = 'not_requested'
            else:
                status['uploads'] = upload(config,args.directory,snapshot)
                status['upload'] = 'verified'
        except Exception as error:
            # HTTP/SDK exception strings can contain request details; retain type only.
            status.update(error_type=type(error).__name__,upload='pending',
                          local_metrics_preserved=True)
            if isinstance(error,RuntimeError) and str(error) in (
                'W&B authentication is not installed on this host',
                'W&B history readback has not confirmed all uploaded events',
                'SSH metric snapshot failed; check the configured development endpoint',
                'No matching PPO metric streams found at the configured endpoint',
                'W&B endpoint did not confirm authentication'):
                status['reason'] = str(error)
            elif isinstance(error,RuntimeError) and str(error).startswith('W&B endpoint returned HTTP '):
                status['reason'] = str(error)
        save(args.directory/'STATUS.json',status)
        print(json.dumps(status,indent=2))
        if status.get('upload') == 'pending':
            raise SystemExit(1)


if __name__ == '__main__':
    main()
