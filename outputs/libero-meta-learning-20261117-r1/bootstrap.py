"""Prepare an isolated LIBERO runtime on Linux; no model or GPU execution."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

PIN = 'e5500c690c1498eca1b7050350654964190cdf70'
ROOT = Path('/mowenjie/fluffy-bassoon-2')
SETUP = ROOT/'setups/libero-meta-learning-20261117-r1'
SOURCE = ROOT/'sources/libero-pro2-e5500c69'
BASE = ROOT/'setups/robocasa365-native-20261101-r1/env'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    assert sys.platform == 'linux'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', action='store_true')
    parser.add_argument('--wheelhouse', type=Path)
    args = parser.parse_args()
    assert subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip() == PIN
    SETUP.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(SETUP).free
    if free < 4*2**30:
        raise RuntimeError('Runtime/assets need <2 GB; preserve another2 GB of setup headroom')
    probe = SETUP/'write-probe'; probe.write_bytes(b'probe'); probe.unlink()
    if not (SETUP/'env/bin/python').exists():
        subprocess.run(['/mowenjie/bin/uv','venv','--python',str(BASE/'bin/python'),str(SETUP/'env')],check=True)
    site = SETUP/'env/lib/python3.11/site-packages'
    (site/'shared-frozen-policy-dependencies.pth').write_text(str(BASE/'lib/python3.11/site-packages')+'\n')
    env = dict(os.environ, UV_CACHE_DIR=str(SETUP/'uv-cache'), UV_LINK_MODE='copy')
    wheel_args = ['--find-links',str(args.wheelhouse)] if args.wheelhouse else []
    subprocess.run(['/mowenjie/bin/uv','pip','install','--python',str(SETUP/'env/bin/python'),*wheel_args,
        '--no-deps','robosuite==1.4.0','mujoco==3.2.3','bddl==1.0.1','robomimic==0.2.0',
        'easydict==1.9','numpy==1.26.4','pydantic==1.10.22','httpx==0.24.1','httpcore==0.17.3',
        'h11==0.14.0','msgpack==1.0.8'],env=env,check=True)
    simulator = SOURCE/'simulator-source/libero/libero'
    # Keep the Git checkout immutable. Ordinary assets are shared by symlink;
    # LFS payloads live in this setup's separate asset tree and are SHA-checked.
    asset_root = SETUP/'assets'
    pending = []
    for path in sorted((simulator/'assets').rglob('*')):
        relative = path.relative_to(simulator/'assets')
        target = asset_root/relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        raw = path.read_bytes() if path.stat().st_size < 256 else b''
        if raw.startswith(b'version https://git-lfs.github.com/spec/v1'):
            fields = dict(line.split(' ',1) for line in raw.decode().splitlines())
            pending.append((path,target,fields['oid'].split(':')[1],int(fields['size'])))
        elif not target.exists():
            target.symlink_to(path)

    def fetch(item):
        path,target,sha,size = item
        if target.is_file() and target.stat().st_size == size and digest(target) == sha:
            return dict(path=str(target),sha256=sha,bytes=size)
        url = 'https://media.githubusercontent.com/media/LIBERO-Pro-Team/LIBERO-Pro2.0/'+PIN+'/'+str(path.relative_to(SOURCE))
        temporary = target.with_name(target.name+'.download')
        with urllib.request.urlopen(url, timeout=60) as source, temporary.open('wb') as output:
            shutil.copyfileobj(source,output)
        if temporary.stat().st_size != size or digest(temporary) != sha:
            raise RuntimeError('LFS asset size/hash mismatch: '+str(path))
        temporary.replace(target)
        return dict(path=str(target),sha256=sha,bytes=size)

    fetched = []
    if args.assets:
        with ThreadPoolExecutor(max_workers=6) as pool:
            for record in pool.map(fetch,pending):
                fetched.append(record)
                print('asset',len(fetched),'/',len(pending),flush=True)
    # Some upstream resolvers locate assets through __file__, ignoring the
    # config. Copy Python source byte-for-byte into a setup package and point
    # its source-relative asset directory at the hydrated tree. The upstream
    # Git checkout stays clean; no symlinked Python source resolves back there.
    package_root = SETUP/'simulator-source'
    (package_root/'libero').mkdir(parents=True, exist_ok=True)
    source_records = []
    for path in sorted((SOURCE/'simulator-source/libero').rglob('*')):
        relative = path.relative_to(SOURCE/'simulator-source')
        if relative.parts[:3] == ('libero','libero','assets'):
            continue
        target = package_root/relative
        if path.is_dir():
            target.mkdir(parents=True,exist_ok=True)
        elif path.suffix == '.py':
            shutil.copyfile(path,target)
            assert digest(target) == digest(path)
            source_records.append(dict(path=str(relative),sha256=digest(path)))
        elif not target.exists():
            target.symlink_to(path)
    local_simulator = package_root/'libero/libero'
    asset_link = local_simulator/'assets'
    if not asset_link.exists():
        asset_link.symlink_to(asset_root,target_is_directory=True)
    assert asset_link.resolve() == asset_root.resolve()
    config = SETUP/'libero-config'; config.mkdir(exist_ok=True)
    paths = dict(benchmark_root=str(local_simulator),bddl_files=str(simulator/'bddl_files'),
        init_states=str(simulator/'init_files'),datasets=str(SETUP),assets=str(asset_root))
    (config/'config.yaml').write_text(''.join(k+': '+json.dumps(v)+'\n' for k,v in paths.items()))
    receipt = dict(source_pin=PIN, source=str(SOURCE), setup=str(SETUP),free_bytes_before=free,
        quota='quota command unavailable; actual write probe passed',
        lfs_required=len(pending),lfs_bytes=sum(p[3] for p in pending),lfs_fetched=fetched,
        simulator_package=str(package_root),python_sources=source_records,
        inherited_policy_dependencies=str(BASE), model_loaded=False, gpu_used=False)
    (SETUP/'preparation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('lfs_fetched','python_sources')}),flush=True)


if __name__ == '__main__':
    main()
