"""Corpus leakage, paired chronology and source accounting; allocated Linux tests."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from fluffy_bassoon.meta_rl.vla_cnd_corpus import merge_cnd_corpus
from fluffy_bassoon.meta_rl.vla_features import FEATURE_KIND
from fluffy_bassoon.meta_rl.vla_history import load_dataset
from fluffy_bassoon.meta_rl.vla_noise_distillation import collect_noise_distillation
from test_vla_noise_distillation import arguments


SPLIT = Path(__file__).resolve().parents[1] / "outputs/libero-pi05-corpus-20261029-r1/split.json"


def make_shard(tmp_path):
    frozen = json.loads(SPLIT.read_text())
    entries = []
    for task_key in ("libero_90:57", "libero_90:02"):
        task = next(row for row in frozen["tasks"] if row["task_key"] == task_key)
        root = tmp_path / task_key.replace(":", "-")
        root.mkdir()
        kwargs = arguments(root)
        kwargs.update(noise_dim=32, teacher_action=lambda obs: np.linspace(-.8, .8, 32, dtype=np.float32),
            row_feature_encoder=lambda row: np.pad(row["public_features"], (0, 4101)),
            motor_sha256="a" * 64, feature_contract=dict(kind=FEATURE_KIND, model_config="pi05_libero",
                checkpoint_receipt_sha256="a" * 64), noise_contract=dict(dimension=32, bounds=[-1, 1]),
            benchmark_contract={**{key: task[key] for key in ("suite", "task_id", "semantic_task", "bddl_sha256")},
                                "teacher_contract_sha256": "c" * 64}, export_paired_dpt=True)
        collect_noise_distillation(**kwargs)
        entries.append(dict(task_key=task_key, teacher_contract_sha256="c" * 64,
            collection_manifest=str(kwargs["output_dir"] / "paired-collections.json")))
    return dict(format="frozen_vla_cnd_corpus_shard_v1", collections=entries)


def test_partial_corpus_preserves_global_task_splits_and_counts_source_once(tmp_path):
    shard = make_shard(tmp_path)
    summary = merge_cnd_corpus(split_path=SPLIT, shard=shard, output_dir=tmp_path / "corpus")
    assert summary["partial_shard"]
    assert summary["trials_by_split"] == dict(train=2, validation=2)
    assert summary["total_source_primitive_steps"] == 216
    for partition, key in (("train", "libero_90:57"), ("validation", "libero_90:02")):
        cnd, cm = load_dataset(summary["datasets"]["cnd"][partition])
        dpt, dm = load_dataset(summary["datasets"]["dpt"][partition])
        assert cm["metadata"]["trial_ids"] == [key + "/train:0", key + "/validation:1"]
        assert cm["metadata"]["source_cost_by_task_split"] == dict(train=108, validation=108)
        assert len(cnd) == len(dpt) == 2
        assert cm["metadata"]["total_source_primitive_steps"] == dm["metadata"]["total_source_primitive_steps"] == 216
        for a, b in zip(cnd, dpt):
            assert a["executed_noise"].tobytes() == b["executed_noise"].tobytes()


@pytest.mark.parametrize("mutation", ["heldout", "duplicate", "teacher", "feature"])
def test_corpus_rejects_heldout_duplicate_or_mismatched_provenance_before_export(tmp_path, mutation):
    shard = make_shard(tmp_path)
    if mutation == "heldout":
        shard["collections"][0]["task_key"] = "libero_90:38"
    elif mutation == "duplicate":
        shard["collections"].append(copy.deepcopy(shard["collections"][0]))
    elif mutation == "teacher":
        shard["collections"][0]["teacher_contract_sha256"] = "d" * 64
    else:
        paired_path = Path(shard["collections"][1]["collection_manifest"])
        paired = json.loads(paired_path.read_text())
        paired["feature_contract_sha256"] = "d" * 64
        paired_path.write_text(json.dumps(paired))
    output = tmp_path / "corpus"
    with pytest.raises(ValueError):
        merge_cnd_corpus(split_path=SPLIT, shard=shard, output_dir=output)
    assert not output.exists()


def test_corpus_rejects_context_relabeling_even_when_targets_may_change():
    from fluffy_bassoon.meta_rl.vla_cnd_corpus import assert_paired_contexts
    trial = {field: np.zeros((2, 1), np.float32) for field in
             ("observations", "executed_noise", "rewards", "dones", "episode_starts", "teacher_steps")}
    changed = {key: value.copy() for key, value in trial.items()}
    changed["rewards"][0] = -0.
    with pytest.raises(ValueError, match="bitwise identical"):
        assert_paired_contexts([trial], [changed])
