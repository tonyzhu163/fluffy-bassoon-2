#!/usr/bin/env python3
"""Compose the fixed demo manifest from allocated stage0 JSON; no computation."""
import argparse
import json
from pathlib import Path

from fluffy_bassoon.meta_rl.robotwin_demo_capacity import (
    bound_json, file_sha, prepare_demo_manifest,
)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',type=Path,required=True)
    p.add_argument('--inventory-sha256',required=True)
    p.add_argument('--author',type=Path,required=True)
    p.add_argument('--author-runtime-path',type=Path,required=True)
    p.add_argument('--motor-sha256',required=True)
    p.add_argument('--norm-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    inventory=bound_json(dict(path=str(args.inventory.resolve()),sha256=args.inventory_sha256))
    author=json.loads(args.author.read_text())
    if not args.author_runtime_path.is_absolute():raise ValueError('Require exact deployed artifact path')
    result=prepare_demo_manifest(inventory,args.inventory_sha256,author,
        dict(path=str(args.author_runtime_path),sha256=file_sha(args.author)),
        motor_sha=args.motor_sha256,norm_sha=args.norm_sha256)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')


if __name__=='__main__':main()
