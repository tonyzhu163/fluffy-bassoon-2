"""Plot the completed paired readout without selecting models or contrasts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


METHODS = ('ad', 'dpt', 'dit', 'ic_iql')
LABELS = {'ad': 'AD', 'dpt': 'DPT', 'dit': 'DIT', 'ic_iql': 'IC-IQL'}
MODELS = tuple(f'{size}/{method}' for size in (3, 6) for method in METHODS)
MODEL_LABELS = [f'{LABELS[method]} · {size} tasks' for size in (3, 6) for method in METHODS]
PANELS = ('ID', 'OOD')
PANEL_LABELS = {'ID': 'ID: 3 tasks, 120 paired cases', 'OOD': 'OOD: 2 tasks, 80 paired cases'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    assert sys.platform == 'linux'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    data = json.loads(args.result.read_text())
    assert data['completed'] and data['verified_physical_episodes'] == 13200
    assert set(data['models']) == set(MODELS) and set(data['panels']) == set(PANELS)
    for panel in data['panels'].values():
        assert panel['cases_per_task'] == 40 and set(panel['models']) == set(MODELS)
    args.output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.spines.left': False, 'axes.spines.bottom': False,
        'axes.titleweight': 'bold', 'pdf.fonttype': 42})
    files = []
    footer = ('All 8 fixed-final fits; one training seed per cell. Intervals: marginal 95% paired case bootstrap.\n'
              'OOD is relative to meta-training tasks; uncertainty does not cover new tasks or training seeds.')

    def finish(fig, name, title, note=footer):
        fig.suptitle(title, fontsize=17, fontweight='bold', y=.98)
        fig.text(.5, .015, note, ha='center', va='bottom', fontsize=9, color='#4B5563')
        fig.tight_layout(rect=(.015, .075, .99, .95), h_pad=2.4, w_pad=2.0)
        for extension in ('png', 'pdf'):
            path = args.output / f'{name}.{extension}'
            fig.savefig(path, dpi=180, facecolor='white', bbox_inches='tight')
            files.append(dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size))
        plt.close(fig)

    def style(axis, labels, grouped=False):
        axis.set_yticks(range(len(labels)), labels)
        axis.set_ylim(len(labels) - .5, -.5)
        axis.tick_params(axis='both', length=0)
        axis.grid(axis='x', color='#E5E7EB', linewidth=.7)
        axis.set_axisbelow(True)
        if grouped:
            axis.axhline(3.5, color='#D1D5DB', linewidth=.8)

    def forest(axis, effects, labels, limit, color='#2563EB', grouped=False):
        for y, effect in enumerate(effects):
            point = effect['difference_pp']
            lo, hi = effect['paired_bootstrap95_pp']
            assert np.isfinite([point, lo, hi]).all() and lo <= hi
            axis.hlines(y, lo, hi, color=color, linewidth=2)
            axis.plot(point, y, 'o', color=color, markersize=5)
        axis.axvline(0, color='#6B7280', linewidth=1, linestyle='--')
        axis.set_xlim(-limit, limit)
        style(axis, labels, grouped=grouped)

    def extent(effects):
        # Symmetric axes include every interval; no significance filtering.
        maximum = max(abs(value) for e in effects for value in
            [e['difference_pp'], *e['paired_bootstrap95_pp']])
        return max(10, np.ceil((maximum + 2) / 5) * 5)

    history = {panel: [data['panels'][panel]['models'][name]['history_effect'] for name in MODELS]
               for panel in PANELS}
    history_limit = extent(history['ID'] + history['OOD'])
    fig, axes = plt.subplots(2, 2, figsize=(12, 10.5))
    for column, panel in enumerate(PANELS):
        values = data['panels'][panel]
        axis = axes[0, column]
        for y, name in enumerate(MODELS):
            rates = values['models'][name]['success_percent']
            axis.plot(rates['full_sixth'], y - .12, 'o', color='#1D4ED8', markersize=5,
                label='Full history, attempt 6' if y == 0 else None)
            axis.plot(rates['reset_sixth'], y + .12, 's', color='#9CA3AF', markersize=4,
                label='Reset history, attempt 6' if y == 0 else None)
        for name, color, line, label in [('original_SFT', '#374151', ':', 'SFT'),
                ('fixed_PPO30', '#B45309', '--', 'PPO30')]:
            axis.axvline(values['baseline_success_percent'][name], color=color,
                linestyle=line, linewidth=1.3, label=label)
        style(axis, MODEL_LABELS, grouped=True)
        axis.set_xlim(0, 100)
        axis.set_xlabel('Native success (%) — observed rates')
        axis.set_title(PANEL_LABELS[panel])
        axis.legend(loc='upper center', bbox_to_anchor=(.5, -.13), ncol=2, frameon=False, fontsize=8)
        forest(axes[1, column], history[panel], MODEL_LABELS, history_limit, grouped=True)
        axes[1, column].set_title('History benefit within each model')
        axes[1, column].set_xlabel('Full sixth − reset sixth (percentage points)')
    finish(fig, 'matched-performance', 'Matched RoboCasa: competence and history benefit')

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    method_labels = [LABELS[name] for name in METHODS]
    for row, (contrast, title) in enumerate([
            ('full_sixth', 'Change in full-history success'),
            ('history_effect_difference', 'Change in history benefit')]):
        effects = {panel: [data['panels'][panel]['six_minus_three'][method][contrast]
                          for method in METHODS] for panel in PANELS}
        limit = extent(effects['ID'] + effects['OOD'])
        for column, panel in enumerate(PANELS):
            axis = axes[row, column]
            forest(axis, effects[panel], method_labels, limit, color='#7C3AED')
            axis.set_title(f'{panel}: {title}')
            axis.set_xlabel('Six-task − three-task fit (percentage points)')
    finish(fig, 'matched-scaling', 'Does adding three training tasks help?',
        'Fixed training compute; adding tasks changes both diversity and data volume.\n'
        + footer)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10.5))
    for row, (baseline, label) in enumerate([('original_SFT', 'SFT'), ('fixed_PPO30', 'PPO30')]):
        effects = {panel: [data['panels'][panel]['models'][name]['versus_baselines'][baseline]['full_sixth']
                          for name in MODELS] for panel in PANELS}
        limit = extent(effects['ID'] + effects['OOD'])
        for column, panel in enumerate(PANELS):
            axis = axes[row, column]
            forest(axis, effects[panel], MODEL_LABELS, limit, color='#0F766E', grouped=True)
            axis.set_title(f'{panel}: full history versus {label}')
            axis.set_xlabel(f'Full sixth − {label} (percentage points)')
    finish(fig, 'matched-baseline-gaps', 'Do the transformers improve over the stage baselines?',
        'PPO30: task-matched teachers on ID; fixed OpenDrawer30 transfer on both OOD tasks.\n'
        + footer)
    manifest = dict(result=dict(path=str(args.result), sha256=sha(args.result)),
        source_commit=args.source_commit, script_sha256=sha(__file__),
        generated_unix=time.time(), files=files, GPU_hours=0,
        visual_review='Required before publication; not performed by this renderer',
        selection='All eight models, both panels and every declared plotted contrast; no outcome filtering')
    (args.output / 'FIGURES.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(figures=3, files=len(files), GPU_hours=0)))


if __name__ == '__main__':
    main()
