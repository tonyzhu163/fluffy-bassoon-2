"""Compare filtered response fits and command corrections without using target reward."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess

import numpy as np

import experiment as ex
from motion import correction_matrix, fit_source, fit_target
from adapter import issue, line_inverse, matched_strength, spectral_fit, spectrum

SOURCE_SHA = 'acf6150ecd7937733d77516cabfaca4a7648531009e3d7c3264c025170279f73'
MATRIX_SHA = '20a9d6800e10404ee23a542c70a3ba1fb077daeccd56cf7fea93555c7add64cf'
BASE = ('identity', 'planar', 'full', 'smooth_df', 'rank1', 'orthogonal')
INVERSES = ('rank1_free', 'smooth_df_line')
HERE = Path(__file__).resolve().parent


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def source_select(rows):
    grid = [*ex.STRENGTHS, np.inf]
    by_arm = {}
    for arm in ('Panda', 'Sawyer'):
        prior, _ = fit_source([r for r in rows if r['arm'] != arm])
        validation = [r for r in rows if r['arm'] == arm]
        inputs, response = ex.query_panel(validation)
        costs = np.zeros((2, len(grid)))
        for row in validation:
            keep = [r['task'] != row['task'] for r in validation]
            spec = spectrum(prior, row)
            for j in (0, 1):
                for k, strength in enumerate(grid):
                    coef, _ = spectral_fit(prior, spec, strength, j)
                    costs[j, k] += ex.errors(coef, inputs[keep], response[keep]).mean() / len(validation)
        by_arm[arm] = costs.tolist()
    idx = np.argmin(np.mean(list(by_arm.values()), axis=0), axis=1)
    return dict(grid=[*grid[:-1], 'infinity'], by_pseudotarget_mm=by_arm, components=['rank1','orthogonal'],
                selected={m:float(grid[i]) if i < len(grid)-1 else 'infinity'
                          for m,i in zip(('rank1', 'orthogonal'), idx)})


def make_models(row, prior, noise, strengths):
    spec = spectrum(prior, row)
    _, s, vt, _ = spec
    leading, df = spectral_fit(prior, spec, float(strengths['rank1']), 0)
    second, df2 = spectral_fit(prior, spec, float(strengths['orthogonal']), 1)
    lam_df = matched_strength(s, df)
    smooth, smooth_df = spectral_fit(prior, spec, lam_df)
    np.testing.assert_allclose(smooth_df, df, rtol=1e-12, atol=1e-12)
    fits = dict(identity=prior, planar=fit_target('planar', prior, row, strengths['planar']),
                full=fit_target('jacobian', prior, row, strengths['jacobian']), smooth_df=smooth,
                rank1=leading, orthogonal=second, rank1_free=leading, smooth_df_line=smooth)
    line = dict(rank1=vt[0], orthogonal=vt[1], smooth_df_line=vt[0])
    actions = {}
    for m, fit in fits.items():
        v = line.get(m)
        mat = correction_matrix(prior, fit, noise) if v is None else line_inverse(prior, fit, noise, v)
        actions[m] = dict(matrix=mat.tolist(), direction=None if v is None else v.tolist())
    meta = dict(singular_values=s.tolist(), rank1_df=df, orthogonal_df=df2, smooth_df=smooth_df,
                full_df=float(np.sum(s**2 / (s**2 + strengths['jacobian']))),
                smooth_lambda=lam_df if np.isfinite(lam_df) else 'infinity')
    return fits, actions, meta


def ci(grid):
    stat = ex.summary(grid)
    return [float(stat['mean_mm']), *map(float, stat['approximate_95_mm'])]


def checked_axes():
    v = np.array([1., 2., -3.]) / np.sqrt(14)
    p = np.outer(v, v)
    mat = np.eye(3) + 2*p
    samples = np.random.default_rng(113).uniform(-1, 1, (1000,7)).astype(np.float32)
    samples[:2, :3] = [[1, .2, -.4], [-1, -.3, .1]]
    out = issue(samples, mat, v)
    d = out[:,:3].astype(float) - samples[:,:3]
    np.testing.assert_allclose(d @ (np.eye(3)-p), 0, atol=1e-7)
    np.testing.assert_array_equal(out[:,3:], samples[:,3:])
    np.testing.assert_array_equal(out[:2], samples[:2])
    np.testing.assert_array_equal(issue(samples,np.eye(3)), samples)
    axis_null = np.zeros((1,7))
    axis_null[0,:3] = [.2,-.1,0]
    np.testing.assert_allclose(issue(axis_null,mat,v), axis_null, atol=1e-15)


def action_errors(x, a, axis):
    changed = a[..., :3].astype(float) - x[..., :3]
    along = (changed @ axis)[..., None] * axis
    return dict(change=np.linalg.norm(changed, axis=-1).mean(-1),
                off_axis=np.linalg.norm(changed-along,axis=-1).mean(-1),
                reversed=np.mean((a[...,:3] * x[...,:3]).sum(-1)<0, axis=-1),
                new_boundary=np.any((abs(a[...,:3])>=1) & (abs(x[...,:3])<1), axis=-1).mean(-1))


def main(args):
    checked_axes()
    args.out.mkdir(exist_ok=False)
    sibling = args.root/'frozen-vla-motion-models-20261123-r1/cpu01'
    assert digest(sibling/'source-fit.json') == SOURCE_SHA and digest(sibling/'matrices.json') == MATRIX_SHA
    old = json.loads((sibling/'source-fit.json').read_bytes())
    matrices = json.loads((sibling/'matrices.json').read_bytes())
    split = json.loads(args.splits.read_bytes())
    sources = ex.source_traces(args.root, split)
    assert [r['support'] for r in sources] == old['sources']
    prior, noise = fit_source(sources)
    np.testing.assert_allclose(prior, old['prior'], rtol=1e-12)
    np.testing.assert_allclose(noise, old['residual_power'], rtol=1e-12)
    cv = source_select(sources)
    strengths = {**old['choice'], **cv['selected']}
    ex.dump(args.out/'source.json', dict(cv=cv, strengths=strengths, prior=prior.tolist(), noise=noise,
            sibling_source_sha256=SOURCE_SHA, split_sha256=digest(args.splits), support_seed_layout=11,
            cpu_affinity=sorted(os.sched_getaffinity(0)), numpy=np.__version__,
            code_pin=subprocess.check_output(['git','-C',str(HERE),'rev-parse','HEAD'],text=True).strip()))
    targets = ex.target_traces(args.root, split)
    assert {r['task'] for r in sources}.isdisjoint(r['task'] for group in targets.values() for r in group)
    models, adapters = {}, {}
    for arm, rows in targets.items():
        for r in rows:
            key = arm+'/'+r['task']
            oldrow = matrices[key]
            assert r['support'] == oldrow['support'] and r['controls'] == 15
            fits, actions, meta = make_models(r, prior, noise, strengths)
            for m, field in [('full','matrix'),('planar','planar_matrix')]:
                np.testing.assert_allclose(actions[m]['matrix'],oldrow[field],atol=1e-12)
            models[key] = fits
            adapters[key] = dict(support=r['support'], spectrum=meta, adapters=actions)
    ex.dump(args.out/'adapters.json', adapters)
    results, pairs = {}, []
    for arm, rows in targets.items():
        inputs, response = ex.query_panel(rows)
        actions = np.stack([np.load(r['support']['path'])['issued_mean'][:3] for r in rows])
        np.testing.assert_array_equal(inputs[:,:,:3], actions[:,:,:3])
        grids = {}
        for i, row in enumerate(rows):
            key = arm+'/'+row['task']
            v = np.array(adapters[key]['adapters']['rank1']['direction'])
            for other in targets:
                key2 = other+'/'+row['task']
                for m in (*BASE, *INVERSES):
                    ad = adapters[key2]['adapters'][m]
                    issued = issue(actions, ad['matrix'], ad['direction'])
                    np.testing.assert_array_equal(issued[...,3:], actions[...,3:])
                    metric = action_errors(actions, issued, v)
                    metric['error_mm'] = ex.errors(models[key2][m], inputs, response)
                    for name, values in metric.items():
                        g = grids.setdefault((other,m,name), np.full((8,8),np.nan))
                        g[i] = values
                        g[i,i] = np.nan
        wrong = [a for a in targets if a != arm]
        results[arm] = {}
        for m in (*BASE, *INVERSES):
            results[arm][m] = {}
            for name in ('error_mm','change','off_axis','reversed','new_boundary'):
                good = grids[arm,m,name]
                mismatch = (grids[wrong[0],m,name]+grids[wrong[1],m,name])/2
                results[arm][m][name] = dict(matched=ci(good), wrong=ci(mismatch), matched_minus_wrong=ci(good-mismatch),
                                              rank1_minus_method=ci(grids[arm,'rank1',name]-good))
        for (other,m,k),g in grids.items():
            pairs.append(dict(target=arm, support_arm=other, method=m, metric=k,
                              tasks=[r['task'] for r in rows], grid=[[float(v) if np.isfinite(v) else None for v in row] for row in g]))
        print(arm, {m:round(results[arm][m]['error_mm']['matched'][0],5) for m in BASE}, flush=True)
    ex.dump(args.out/'summary.json', results)
    ex.dump(args.out/'task-pairs.json', pairs)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('root','splits','out'):
        p.add_argument('--'+name, type=Path, required=True)
    main(p.parse_args())
