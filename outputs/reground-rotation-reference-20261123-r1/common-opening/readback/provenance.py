"""Verify readback sources from the deployed Git objects; capture CPU-only runtime facts."""
import argparse
import hashlib
import os
from pathlib import Path
import platform
import subprocess
import sys

from native_read import *
from controlled import CODE, file_hash


def git(repo,*args):
    env=dict(os.environ,GIT_OPTIONAL_LOCKS="0")
    return subprocess.check_output(["git","-C",str(repo),*args],env=env)


def verified_source(repo,pin,relative):
    check(git(repo,"rev-parse","HEAD").decode().strip()==pin,repo)
    digest=file_hash(repo/relative)
    check(digest==hashlib.sha256(git(repo,"show",f"{pin}:{relative}")).hexdigest(),relative)
    return digest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output",type=Path)
    args=ap.parse_args()
    work=Path(__file__).resolve().parent
    upstream=ROOT/"upstream"
    audit_source="baac956f214fe680c120c0636e6801bda6ed8dc2"
    rel=f"outputs/{ROOT.name}/"
    sources={}
    for file in ("reground/eval/jobs.py","reground/eval/rollout.py","reground/envs/libero_env.py",
                 "reground/envs/safety.py","reground/policies/vla.py","reground/utils.py","configs/speed_envelope.json"):
        sources[file]=verified_source(upstream,PIN,file)
    for pin,files in ((CODE,("common-opening/opening.py","common-opening/runner.py")),
                      (STREAM,("reference.py","symmeta_adapter.py","evidence/rotation-jobs.json.gz")),
                      (audit_source,("native_grid.py",))):
        for file in files:
            sources[file]=verified_source(ROOT/f"src-{pin}",pin,rel+file)
    check(sources["native_grid.py"]==file_hash(work/"native_grid.py"),"native_grid copy")
    libs=Path("/proc/self/maps").read_text()
    gpu_modules=[k for k in sys.modules if k.split('.')[0] in {"torch","mujoco","jax","tensorflow","reground","lerobot","cupy"}]
    check(not gpu_modules and not any(x in libs for x in ("libcuda.so","libcudart.so","libtorch")), "unexpected GPU imports")
    headers=ROOT/"env/lib/python3.12/site-packages/torch/include/ATen"
    runtime=dict(python=sys.version,numpy=np.__version__,system=platform.platform(),machine=platform.machine(),
                 researcher_affinity={"original_reporter":[0],"controlled_readback":[1]},
                 cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],blocked_modules_loaded=gpu_modules,
                 noise_executable_dependencies=subprocess.check_output(["ldd",str(work/"noise")],text=True).splitlines(),
                 noise_build="c++ -O2 -std=c++17 -mavx2 -mfma -DCPU_CAPABILITY_AVX2 -I<torch/include> noise.cc -o noise",
                 torch_cpu_headers={str(f.relative_to(headers)):file_hash(f) for f in
                                    (headers/"native/cpu/avx_mathfun.h",headers/"native/cpu/DistributionTemplates.h",
                                     headers/"core/TransformationHelper.h")})
    save(args.output,dict(root=str(ROOT),upstream=PIN,policy=POLICY,scientific_code=STREAM,controlled_code=CODE,
                          original_predicate_revision=audit_source,source_sha256=sources,runtime=runtime,
                          audit_sha256={f:file_hash(work/f) for f in ("native_read.py","reporter.py","controlled.py",
                                       "summarize.py","noise.cc","provenance.py")}))


if __name__=="__main__":
    main()
