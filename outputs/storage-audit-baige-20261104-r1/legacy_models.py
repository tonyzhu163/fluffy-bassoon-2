"""Cold-store retired model/download assets outside current runtime closures."""
import importlib.util
import json
from pathlib import Path

base=Path('/mowenjie/fluffy-bassoon-2/outputs/storage-audit-baige-20261104-r1')
spec=importlib.util.spec_from_file_location('archive',base/'archive.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.ROOT=Path('/mowenjie')
m.AUDIT=base/'legacy-models';m.AUDIT.mkdir(exist_ok=False)
m.BOS=m.BOS/'legacy-models';m.STAGE=m.STAGE/'legacy-models'
plan={'selected':[
 'fluffy-bassoon/inputs/rlinf-pi05-maniskill-rl-a0cbb898-20261028-r1',
 'fluffy-bassoon/inputs/rlinf-pi05-maniskill-sft-eb0e2c90-20261028-r1',
 'fluffy-bassoon/inputs/libero-carry-seed1000-step4000-70bdfe6b',
 'openpi_home',
 'fluffy-bassoon-2/cache/robotwin',
 'fluffy-bassoon-2/inputs/pi05-robotwin-ppo-adjust-bottle-1223d8b9-20261031-r1'],
 'basis':'Retired ManiSkill/LycoReco/WAM/RoboTwin model assets. No active or queued job uses these model paths; current RoboCasa native launcher uses independent setup, caches, torch input and normalization. Current Flora weights, current LIBERO weights, shared package bases, teacher18 and all held checkpoints stay hot. Old qualification scripts may require restoration before reuse. Full contents archived, no source or scientific-result deletion.'}
m.save(m.AUDIT/'CANDIDATES.json',plan)
m.main()
