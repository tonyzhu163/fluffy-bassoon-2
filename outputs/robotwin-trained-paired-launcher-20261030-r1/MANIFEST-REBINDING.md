# Rebinding a whole-round handoff to the qualified executor

No new runtime, control-reader or manifest schema is required. The intended
and CPU-tested operation is a new manifest containing the exact old manifest
with only its outer `execution_source_commit` changed to
`57016ba08eed79043fd67bf8b085bd6d42e6d766`. Preserve the original file and SHA.
Issue this only from the source owner's newly audited, drained whole-round
boundary; an old immutable artifact is not current ownership evidence.

The outer handoff pin identifies the executor admitted to consume that
handoff. The old corpus's `contract.json` and `execution-migration.json` retain
their original producer. In qualified `c62be996...`, `corpus_chain` compares
`metadata.execution_source_commit` to **`contract.execution_source_commit`**,
not to the outer manifest field. It checks the corpus's predecessor SHA against
its own contract, then follows that predecessor manifest. Therefore the old
e825 contract/migration pair remains consistent beneath a570-bound handoff.
The future570 output has a570 contract/migration pair, pointing to this new
handoff and then the unchanged old corpus ancestry.

The qualified570 `load_trial_continuation` deliberately checks the outer pin
against its `execution_commit` argument. `tests/test_robotwin_trained_parallel.py`
and `tests/test_robotwin_trained_pair_preflight.py` already construct target-bound
handoffs by copying the old manifest and changing this outer field. The
47-test allocated launcher gate passed; it did not require corpus producer
metadata to be rewritten.

## Exact construction and verification

Run artifact validation only on allocated Linux, with CUDA hidden and an
external90-second bound. No `CampaignBudget`, learner constructor, simulator,
old-file write or source-process control is needed. The source owner still
owns the current boundary audit, terminal allocation release and canonical
head/ownership recheck before actual admission.

1. Verify the exact old manifest SHA from the owner's boundary audit and the
   checkpoint SHA it binds. Load that checkpoint on CPU. Require an already
   trained complete-update boundary, method PPO/reset, `kind=whole_trial_boundary`,
   `trial=null`, empty `prefix`, empty expected pending-rollout digests,
   `completed_trials=6*completed_iterations`, and an iteration below100.
2. Validate the original manifest with qualified
   `load_trial_continuation(old_path, old_sha, checkpoint_path=bound_path,
   checkpoint=cpu_payload, execution_commit=old_manifest['execution_source_commit'],
   lineage_commit='db0184089fa88a5f96c153f9b80029bb91b77def')`.
3. Construct `new = deepcopy(old)` and set only
   `new['execution_source_commit']='57016ba08eed79043fd67bf8b085bd6d42e6d766'`.
   Write a **fresh** JSON file outside the old run with exclusive creation,
   fsync the file and containing directory, and calculate its SHA256.
4. Assert the dictionaries are identical after removing just that field.
   Validate the new file with the same qualified loader, same checkpoint path
   and payload, but `execution_commit=57016...`. Check method, counts and all
   saved artifacts remain identical. The existing trained-boundary and
   original-parent checks still apply in the qualified launcher preflight.
5. Import the exact qualifiedc62 control module without running its CLI and
   assert `corpus_chain(old_ref)==corpus_chain(new_ref)` in the same ordered
   sequence. Retain both lists in the owner's rebinding receipt. This directly
   checks the proposed provenance concern on the actual immutable artifacts.

Do not regenerate the checkpoint, corpus, progress, episode records, capture
qualification, LR receipt, ledger snapshot, campaign header or canonical
journal. In particular, do not change either old corpus producer pin or its
predecessor SHA. `write_boundary_manifest` is unnecessary here; copying the
existing fully bound manifest avoids touching its old episode sink.

## Owner rebinding receipt

Use a separate fresh JSON receipt, avoiding hash-reference cycles:

```json
{
  "schema": "robotwin_executor_handoff_rebinding_v1",
  "predecessor_manifest": {"path": "OLD_ABSOLUTE_PATH", "sha256": "OLD_SHA256"},
  "admitted_manifest": {"path": "NEW_ABSOLUTE_PATH", "sha256": "NEW_SHA256"},
  "predecessor_execution_source_commit": "OLD_OUTER_EXECUTOR_FULL_SHA",
  "admitted_execution_source_commit": "57016ba08eed79043fd67bf8b085bd6d42e6d766",
  "lineage_source_commit": "db0184089fa88a5f96c153f9b80029bb91b77def",
  "original_parent": {"path": "ORIGINAL_PARENT_PATH", "sha256": "1f80c08310dfa4543ec7a748ca6e2bf9aad589cd55978a402210abb0d83bdbc0"},
  "owner_boundary_audit": {"path": "AUDITED_OLD_BOUNDARY_RECEIPT", "sha256": "AUDIT_SHA256"},
  "unchanged_checkpoint": {"path": "ORIGINAL_BOUND_CHECKPOINT_PATH", "sha256": "ORIGINAL_BOUND_CHECKPOINT_SHA256"},
  "only_existing_field_changed": "execution_source_commit",
  "all_other_existing_fields_and_artifact_refs_equal": true,
  "original_loader_validation_passed": true,
  "admitted_loader_validation_passed": true,
  "corpus_chain_reader_commit": "c62be99615e797fbec108e1f1b5e232e4a591494",
  "old_corpus_roots": ["UNCHANGED_ORDERED_ROOTS"],
  "new_corpus_roots": ["IDENTICAL_ORDERED_ROOTS"],
  "source_actions_added": 0,
  "scientific_contract_changed": false,
  "canonical_ledger_modified": false
}
```

Hash-bind this receipt in the owner's terminal release row as an additional
`executor_rebinding` artifact reference. That row's `manifest` and `checkpoint`
must equal the new plan's references exactly; the checkpoint reference stays
unchanged. The compiled plan binds the release receipt and later CPU preflight
rechecks its exact canonical heads. Sequence the hashes as: old boundary audit
→new manifest→rebinding receipt→terminal release→compute binding→compiled plan.
None of those references needs to point forward and create a cycle.

This clarifies the existing qualified protocol. It does not issue a current
handoff, authorize a copied ledger, mutate an active run or claim a trained
GPU result.
