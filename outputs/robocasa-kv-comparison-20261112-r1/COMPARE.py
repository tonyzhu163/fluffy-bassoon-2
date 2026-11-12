"""Reconstruct the complete four-model KV panel from independently audited banks."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys

METHODS = ('ad', 'dpt', 'dit', 'ic_iql')
TASKS = ('OpenDrawer', 'CloseToasterOvenDoor', 'CoffeeSetupMug', 'CloseFridge', 'TurnOnMicrowave')
HORIZONS = (600, 600, 600, 900, 450)
LABELS = dict(ad='AD', dpt='DPT', dit='DIT', ic_iql='IC-IQL')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old', type=Path, required=True)
    p.add_argument('--new', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert sys.platform == 'linux', 'Run analysis on allocated Linux CPUs'
    import numpy as np
    arrays, totals, sources, candidates = {}, {}, [], {}
    index = read(args.old / 'finalized-fits-20261106-r1/INDEX.json')
    for method in METHODS:
        base = args.old if method == 'ad' else args.new
        array = np.zeros((2, 5, 50, 6), dtype=np.int8)
        seen = set()
        totals[method] = dict(episodes=0, banks=0, physical_controls=0, decoded_videos=0)
        candidate = index['models']['kv_bridge/' + method]['artifacts']['candidate.pt']['sha256']
        candidates[method] = candidate
        for split, task_indices in (('ID', range(3)), ('OOD', range(3, 5))):
            group = base / 'readouts/kv_bridge' / method / split
            binding, overall, summary = (read(group / name) for name in ('BINDING.json', 'RESULT.json', 'SUMMARY.json'))
            assert overall['completed'] and summary['completed']
            assert binding['source_commit'] == '87984718c9199797d31a158c47225cc695623afe'
            assert binding['runtime_commit'] == '8f9bfa7661928ef8c17494b04f4d7bf3125634e1'
            assert binding['model_sha256'] == '661bf172a2ae9dfca75216697eeaa543408bcbdfa53c98132759b93e0e8993a5'
            assert binding['norm_sha256'] == '4aed1af411bd0e0f49d2b0e6d6832b11b8682917231a07173fbc30fd493bbdee'
            assert binding['selected_candidate_sha256'] == summary['selected_candidate_sha256'] == candidate
            assert binding['cases_per_task'] == 50 and binding['memory_modes'] == ['full', 'episode']
            assert binding['deployment_optimizer_updates'] == overall['deployment_optimizer_updates'] == 0
            assert sha(group / 'selected-candidate.pt') == candidate
            binding_sha = sha(group / 'BINDING.json')
            for mi, mode in enumerate(('full', 'episode')):
                for ti in task_indices:
                    task, horizon = TASKS[ti], HORIZONS[ti]
                    for start in range(0, 50, 8):
                        bank = group / mode / task / f'bank-{start:03d}'
                        raw, audit, marker = (read(bank / n) for n in ('RESULT.json', 'AUDIT.json', 'PACKED-ARCHIVE.json'))
                        manifest = Path(marker['manifest'])
                        packed = read(manifest)
                        digest = sha(bank / 'RESULT.json')
                        size = min(8, 50 - start)
                        assert raw['binding_sha256'] == binding_sha
                        assert (raw['interface'], raw['method'], raw['mode'], raw['task']) == ('kv_bridge', method, mode, task)
                        assert raw['optimizer_updates'] == audit['SGD'] == 0 and raw['native_parameters_frozen']
                        assert raw['denominators'] == size and raw['actual_batch_sizes'] == [size] * (6 * horizon // 50)
                        assert audit['passed'] and audit['result_sha256'] == marker['result_sha256'] == digest
                        assert audit['all_commands_and_draws_reproduced'] and audit['all_initial_physical_public_states_match']
                        assert audit['memory_mode'] == mode and audit['all_prior_history_retained'] == (mode == 'full')
                        assert audit['episodes'] == len(raw['episodes']) == size * 6
                        assert audit['physical_controls'] == raw['controls_charged'] == size * 6 * horizon
                        assert audit['decoded_videos'] == size * 6 * 3
                        assert audit['video_frames'] == size * 6 * (horizon + 1) * 3
                        assert marker['verified'] and packed['passed'] and packed['all_stageback_bytes_verified']
                        assert packed['independent_source_audit_sha256'] == sha(bank / 'AUDIT.json')
                        assert packed['archive_sha256'] == marker['archive_sha256']
                        assert Path(packed['cold']).stat().st_size == packed['archive_bytes']
                        for ep in raw['episodes']:
                            case, query = ep['case'], ep['attempt'] - 1
                            slot = case['slot']
                            key = (mi, ti, slot, query)
                            assert ep['completed'] and ep['horizon'] == horizon and type(ep['success_once']) is bool
                            assert start <= slot < min(start + 8, 50) and 0 <= query < 6 and key not in seen
                            assert case['task'] == task and case['native_seed'] == 64105000 + ti * 10000 + slot
                            assert case['bank'] == 'joint_meta_final_' + split and case['split'] == 'target'
                            assert case['reset_ordinal'] == 0 and case['counted']
                            seen.add(key)
                            array[key] = int(ep['success_once'])
                        for name in ('episodes', 'physical_controls', 'decoded_videos'):
                            totals[method][name] += audit[name]
                        totals[method]['banks'] += 1
                        sources.append(dict(method=method, bank=str(bank), result_sha256=digest,
                            audit_sha256=sha(bank / 'AUDIT.json'), manifest_sha256=sha(manifest),
                            archive_sha256=packed['archive_sha256']))
                    assert array[mi, ti].sum(axis=0).tolist() == summary['curves'][task][mode]['first_success']
        assert len(seen) == 3000 and totals[method]['banks'] == 70
        arrays[method] = array

    # Stratify by task; the same 50-case resamples are reused across all methods,
    # modes and queries. Intervals condition on one fixed fit per algorithm.
    rng = np.random.default_rng(20261109)
    draws = rng.integers(50, size=(5, 10000, 50))

    def interval(delta, indices):
        samples = np.mean([delta[t, draws[t], :].mean(axis=1) for t in indices], axis=0) * 100
        return np.percentile(samples, [2.5, 97.5], axis=0).T.tolist()

    report = dict(completed=True, checked_shanghai=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        methods={}, candidates=candidates, verified_totals=totals, source_banks=sources,
        new_episodes=9000, reference_episodes=3000, bootstrap_seed=20261109, bootstrap_replicates=10000,
        uncertainty_scope='Paired case sampling conditional on one fixed trained model per algorithm; marginal intervals, no multiplicity correction',
        OOD_scope='Held out from the three-task meta-training corpus, not necessarily from base VLA pretraining',
        optimizer_updates_during_evaluation=0)
    for method, array in arrays.items():
        delta = array[0] - array[1]
        item = dict(tasks={}, panels={})
        for ti, task in enumerate(TASKS):
            item['tasks'][task] = dict(full_success_counts=array[0, ti].sum(axis=0).tolist(),
                reset_success_counts=array[1, ti].sum(axis=0).tolist(), denominator=50,
                full_minus_reset_pp=(delta[ti].mean(axis=0)*100).tolist(),
                paired_case_bootstrap95_pp=interval(delta, [ti]),
                first_query_discordant_cases=int(np.count_nonzero(delta[ti, :, 0])),
                sixth_query_wins=int((delta[ti, :, 5] > 0).sum()), sixth_query_losses=int((delta[ti, :, 5] < 0).sum()))
        for split, indices in (('ID', list(range(3))), ('OOD', list(range(3, 5)))):
            item['panels'][split] = dict(
                full_success_percent=(array[0, indices].mean(axis=(0, 1))*100).tolist(),
                reset_success_percent=(array[1, indices].mean(axis=(0, 1))*100).tolist(),
                full_minus_reset_pp=(delta[indices].mean(axis=(0, 1))*100).tolist(),
                paired_case_bootstrap95_pp=interval(delta, indices),
                full_minus_AD_pp=((array[0, indices]-arrays['ad'][0, indices]).mean(axis=(0, 1))*100).tolist(),
                full_minus_AD_bootstrap95_pp=interval(array[0]-arrays['ad'][0], indices))
        report['methods'][method] = item
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'RESULT.json').write_text(json.dumps(report, indent=2)+'\n')
    lines = ['# Four frozen KV transformers: matched ID/OOD readout', '',
        'All four saved models now have complete audited panels: 12,000 episodes, including the existing 3,000-episode AD reference.', '',
        'Success below is during the sixth real episode, after five preceding attempts. It is not cumulative best-of-six success.', '',
        '| Method | Split | Full success | Reset success | Full − reset (pp) | Paired 95% interval (pp) |',
        '| --- | --- | ---: | ---: | ---: | --- |']
    for method in METHODS:
        for split in ('ID', 'OOD'):
            row = report['methods'][method]['panels'][split]
            lo, hi = row['paired_case_bootstrap95_pp'][5]
            lines.append(f"| {LABELS[method]} | {split} | {row['full_success_percent'][5]:.1f}% | {row['reset_success_percent'][5]:.1f}% | {row['full_minus_reset_pp'][5]:+.1f} | [{lo:.1f}, {hi:.1f}] |")
    lines += ['', 'The intervals reflect case sampling for one fixed fit per algorithm. They do not include variation across training seeds or new task families. First-query discordance is reported in RESULT.json; small paired differences alone do not prove causal adaptation.', '',
        'ID: Drawer, Toaster, Coffee. OOD: Fridge, Microwave, held out of this meta-training corpus. All use the original frozen VLA, canonical scenes, seeds, horizons and zero deployment SGD.', '',
        '![Per-task success curves](query-success.png)', '', '![Paired history effects](history-effect.png)', '']
    (args.output / 'RESULT.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(zip(METHODS, ('#2563eb', '#d97706', '#16a34a', '#9333ea')))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for ti, ax in enumerate(axes.flat):
        if ti == 5:
            ax.axis('off')
            continue
        for method in METHODS:
            for mi, mode in enumerate(('full', 'reset')):
                ax.plot(range(6), arrays[method][mi, ti].mean(axis=0)*100,
                    color=colors[method], linestyle='-' if mi == 0 else '--', marker='o' if mi == 0 else None,
                    markersize=3, label=LABELS[method]+' '+mode)
        ax.set_title(TASKS[ti] + (' (ID)' if ti < 3 else ' (OOD)'), fontsize=10)
        ax.set_ylim(0, 100)
        ax.set_xticks(range(6))
        ax.grid(alpha=.2)
        ax.set_xlabel('Prior real episodes')
        ax.set_ylabel('Query-episode success (%)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[1, 2].legend(handles, labels, loc='center', frameon=False)
    fig.suptitle('Matched frozen KV transformers · 50 cases per task · solid full / dashed reset')
    fig.tight_layout()
    fig.savefig(args.output / 'query-success.png', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for split, ax in zip(('ID', 'OOD'), axes):
        for i, method in enumerate(METHODS):
            row = report['methods'][method]['panels'][split]
            value = row['full_minus_reset_pp'][5]
            lo, hi = row['paired_case_bootstrap95_pp'][5]
            ax.errorbar(i, value, yerr=[[value-lo], [hi-value]], color=colors[method], fmt='o', capsize=5)
        ax.axhline(0, color='#64748b', linewidth=1)
        ax.set_xticks(range(4), [LABELS[m] for m in METHODS])
        ax.set_title(split+' · after five prior episodes')
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Full − reset success (percentage points)')
    fig.suptitle('Paired case-bootstrap 95% intervals · one fixed fit per method')
    fig.tight_layout()
    fig.savefig(args.output / 'history-effect.png', dpi=170)
    plt.close(fig)
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_banks','methods')},indent=2))


if __name__ == '__main__':
    main()
