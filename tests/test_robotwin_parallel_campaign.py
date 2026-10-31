"""Fixed parent reservations reject silent worker expansion before collection."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import pytest

_spec=importlib.util.spec_from_file_location('parallel_campaign',Path(__file__).parents[1]/'scripts/diagnostics/robotwin_parallel_campaign.py')
module=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(module)

def manifest():
 value=module.build_manifest('robotwin-wholetrial-test','a'*40,'b'*64)
 value['created_at_utc']='2026-10-30T00:00:00Z'
 return value

def test_parent_budget_composes_without_reimporting_prior():
 value=manifest();module.validate_manifest(value,source_commit='a'*40)
 assert value['maximum_new_actions']==sum(w['maximum_actions'] for w in value['workers'].values())==3_589_600
 assert value['maximum_combined_actions']==value['maximum_new_actions']+value['prior_campaign']['maximum_actions']
 assert {w['maximum_complete_source_actions'] for w in value['workers'].values()}=={1_080_000}
 assert value['seed']==20261030
 assert sum(w['native_episodes'] for w in value['workers'].values())==1

@pytest.mark.parametrize('change',[lambda x:x['workers']['sac'].__setitem__('maximum_actions',2_000_000),lambda x:x['workers']['ppo'].__setitem__('native_episodes',2),lambda x:x.__setitem__('seed',1),lambda x:x.__setitem__('maximum_combined_actions',9_000_000)])
def test_parent_rejects_worker_budget_or_recipe_expansion(change):
 value=deepcopy(manifest());change(value)
 with pytest.raises(ValueError,match='differs'):module.validate_manifest(value)

def test_parent_rejects_wrong_source_and_unknown_fields():
 value=manifest()
 with pytest.raises(ValueError,match='source commit'):module.validate_manifest(value,source_commit='c'*40)
 value['extra_allowance']=1
 with pytest.raises(ValueError,match='Unexpected'):module.validate_manifest(value)
