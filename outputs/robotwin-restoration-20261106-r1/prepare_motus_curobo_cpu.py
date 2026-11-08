"""Build an isolated legacy expert backend; no simulator or GPU execution."""
import configparser
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main(identity):
    assert sys.platform == 'linux' and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not os.uname().nodename.startswith('job-'), 'Use the allocated CPU development machine'
    assert identity in ('cpumotuscurobo01','cpumotuscurobo02')
    root = Path('/mowenjie/fluffy-bassoon-2/robotwin-curobo-20261108-'+identity)
    cold = Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1')/identity
    code = Path(os.environ['TASK_CODE'])
    source = os.environ['TASK_SOURCE']
    assert subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip() == source
    assert not subprocess.check_output(['git','-C',str(code),'status','--porcelain','--untracked-files=no'])
    assert not root.exists() and not cold.exists()
    assert shutil.disk_usage(root.parent).free >= 16*2**30
    root.mkdir(); cold.mkdir()
    report = dict(status='preparing', source_commit=source, GPU_seconds=0, models_loaded=0,
        simulator_scenes_created=0, shared_environment_modified=False, GPU_qualified=False,
        expert_success_qualified=False, root=str(root), cold=str(cold), steps=[])
    started = time.time()
    def write():
        (cold/'RESULT.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', UV_CACHE_DIR=str(root/'uv-cache'),
        GIT_LFS_SKIP_SMUDGE='1', MAX_JOBS='1', OMP_NUM_THREADS='2',
        CUDA_HOME='/usr/local/cuda', TORCH_CUDA_ARCH_LIST='9.0+PTX;12.0',
        TORCH_EXTENSIONS_DIR=str(root/'torch-extensions'), TMPDIR=str(root/'tmp'))
    (root/'tmp').mkdir()
    def run(command, *, seconds=1800, capture=False):
        remaining = 2700-(time.time()-started)
        assert remaining > 5
        name = f'{len(report["steps"]):02d}.log'
        before = time.time()
        with (cold/name).open('w') as log:
            result = subprocess.run(command, cwd=root, env=env, text=True,
                stdout=subprocess.PIPE if capture else log,
                stderr=log if capture else subprocess.STDOUT, timeout=min(seconds,remaining))
            if capture: log.write(result.stdout)
        report['steps'].append(dict(command=command,log=name,returncode=result.returncode,
            seconds=time.time()-before))
        write()
        assert result.returncode == 0, f'Failed {name}: {command}'
        return result.stdout if capture else None
    try:
        probe = root/'write-probe'
        with probe.open('xb') as f:
            f.write(b'x'*2**20); f.flush(); os.fsync(f.fileno())
        assert probe.read_bytes() == b'x'*2**20
        probe.unlink()
        import torch
        assert not torch.cuda.is_initialized()
        report.update(base_python=sys.version, torch=torch.__version__, built_cuda=torch.version.cuda,
            cxx11_ABI=torch._C._GLIBCXX_USE_CXX11_ABI, nvcc=run(['/usr/local/cuda/bin/nvcc','--version'],capture=True))
        run([sys.executable,'-m','uv','venv','--python',sys.executable,str(root/'env')])
        python = str(root/'env/bin/python')
        overlay_site = Path(run([python,'-c','import sysconfig;print(sysconfig.get_paths()["purelib"])'],capture=True).strip())
        inherited = list(dict.fromkeys(p for p in sys.path if p.endswith('site-packages') and Path(p).is_dir()))
        assert inherited
        (overlay_site/'robotwin_native_readonly.pth').write_text('\n'.join(inherited)+'\n')
        report['inherited_readonly_site_packages'] = inherited
        # The observed v0.7.8 tag resolves to this legacy MotionGen API commit.
        pin = 'd64c4b005459db10c5dd867d8b30a87d5bda9bdb'
        run(['git','clone','--depth','1','--branch','v0.7.8','https://github.com/NVlabs/curobo.git',str(root/'curobo')],seconds=300)
        actual = run(['git','-C',str(root/'curobo'),'rev-parse','HEAD'],capture=True).strip()
        assert actual == pin
        report['curobo_commit'] = actual
        config = configparser.ConfigParser()
        config.read(root/'curobo/setup.cfg')
        requirements = config['options']['install_requires'].splitlines()
        requirements = [x.strip() for x in requirements if x.strip()]
        # Inspect the overlay's dependency graph, including transitive requirements.
        # Install only missing packages; refuse to upgrade an inherited package.
        inspect_code = '''import importlib.metadata as m,json,sys
from packaging.requirements import Requirement
pending=[(x,('',)) for x in json.loads(sys.argv[1])]; seen=set(); missing=[]; conflicts=[]
while pending:
 text,contexts=pending.pop(); r=Requirement(text)
 if r.marker and not any(r.marker.evaluate({'extra':x}) for x in contexts): continue
 key=(str(r),tuple(sorted(r.extras)))
 if key in seen: continue
 seen.add(key)
 try: d=m.distribution(r.name)
 except m.PackageNotFoundError: missing.append(str(r)); continue
 if r.specifier and not r.specifier.contains(d.version,prereleases=True): conflicts.append([str(r),d.version]); continue
 pending.extend((x,tuple(r.extras) or ('',)) for x in (d.requires or []))
print(json.dumps({'missing':missing,'conflicts':conflicts,'visited':len(seen)}))
'''
        requirements += ['ninja','packaging','setuptools>=45']
        # The existing vendor environment has setuptools84 although a native
        # dependency requires <82. Change the overlay build tool only, never
        # the qualified framework, numerical packages or shared environment.
        run([sys.executable,'-m','uv','pip','install','--python',python,'--no-deps','setuptools>=45,<82'],seconds=300)
        # Match the pinned RLinf installer rather than a changing Warp default.
        requirements = ['warp-lang==1.11.1' if x.startswith('warp-lang') else x for x in requirements]
        for _ in range(8):
            graph = json.loads(run([python,'-c',inspect_code,json.dumps(requirements)],capture=True))
            assert not graph['conflicts'], graph
            report['dependency_graph'] = graph
            if not graph['missing']: break
            run([sys.executable,'-m','uv','pip','install','--python',python,'--no-deps',*graph['missing']],seconds=300)
        else: raise RuntimeError('Unresolved dependency graph')
        # Build CUDA extensions on CPU with explicit architectures. This is not
        # evidence that these kernels work on any allocated accelerator.
        run([sys.executable,'-m','uv','pip','install','--python',python,'--no-deps',
            '--no-build-isolation',str(root/'curobo')],seconds=2100)
        smoke = '''import inspect,json,torch
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen,MotionGenConfig,MotionGenPlanConfig,PoseCostMetric
assert not torch.cuda.is_initialized()
print(json.dumps({'CUDA_initialized':False,'MotionGen':str(inspect.signature(MotionGen)),
'plan_single':str(inspect.signature(MotionGen.plan_single)),
'load_from_robot_config':str(inspect.signature(MotionGenConfig.load_from_robot_config))}))
'''
        report['legacy_API_import'] = json.loads(run([python,'-c',smoke],capture=True))
        report['installed_packages'] = json.loads(run([python,'-c',
            'import importlib.metadata as m,json;print(json.dumps({d.metadata["Name"]:d.version for d in m.distributions()}))'],capture=True))
        report['extension_files'] = {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in overlay_site.glob('curobo/curobolib/*.so')}
        assert len(report['extension_files']) == 5
        assert not torch.cuda.is_initialized()
        report['status'] = 'passed_isolated_CPU_build_and_legacy_API_import_only'
    except BaseException as error:
        report.update(status='failed_CPU_preparation',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        report['seconds'] = time.time()-started
        write()
        print(report['status'],hashlib.sha256((cold/'RESULT.json').read_bytes()).hexdigest(),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--identity',required=True)
    main(parser.parse_args().identity)
