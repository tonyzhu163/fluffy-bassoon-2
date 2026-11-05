"""Compose and validate the actual PPO configuration inside its allocation."""
import json
import os
from pathlib import Path
import sys

from hydra import compose,initialize_config_dir
from recipe import validate_sidney


root=Path(sys.argv[1]);micro=int(sys.argv[2])
with initialize_config_dir(version_base='1.1',config_dir=str(root/'configs/native_ppo')):
    cfg=compose(config_name='sidney_phone_two',overrides=[f'actor.micro_batch_size={micro}'])
contract=validate_sidney(cfg)
Path(os.environ['NATIVE_PPO_OUTPUT'],'config-validation.json').write_text(json.dumps(contract,indent=2)+'\n')
