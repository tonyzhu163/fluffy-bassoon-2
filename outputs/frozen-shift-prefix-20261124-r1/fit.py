"""Fit a prefix through a frozen policy. The optimiser here holds the prefix arrays and nothing else.

An engine supplies:
    digest()                      -> {leaf path: sha256} over every base parameter
    reference(row)                -> mean valid prompt embedding, shape (width,)
    value_and_grad(prefix, row, draw) -> (flow loss, {name: gradient}) with train-time augmentation
    loss(prefix_or_None, row, draw)   -> flow loss without augmentation; None is the unmodified policy
"""
import numpy as np

import prefix as px

SEED, BATCH, UPDATES, LR = 20261124, 32, 300, 1e-3
HELDOUT_DRAWS = 2
BLOCK = 10  # controls per bootstrap block on the held-out replay


class FrozenBaseChanged(RuntimeError):
    pass


def fit(engine, rows, arm, *, updates=UPDATES, batch=BATCH, lr=LR, seed=SEED, log=None):
    before = engine.digest()
    params = px.initial(arm, engine.reference(rows[0]), seed)
    start = {key: value.copy() for key, value in params.items()}
    m = {key: np.zeros_like(value) for key, value in params.items()}
    v = {key: np.zeros_like(value) for key, value in params.items()}
    rng = np.random.default_rng([seed, 47])
    order, history = [], []
    for step in range(1, updates + 1):
        total = {key: np.zeros_like(value) for key, value in params.items()}
        loss = 0.
        for _ in range(batch):
            if not order:
                order = rng.permutation(len(rows)).tolist()
            index = order.pop()
            value, grad = engine.value_and_grad(params, rows[index], (step, index))
            if set(grad) != set(params):
                raise ValueError("Engine returned gradients for something other than the prefix")
            loss += float(value)
            for key in params:
                total[key] += np.asarray(grad[key], dtype=np.float32)
        if not np.isfinite(loss):
            raise FloatingPointError(f"Loss at update {step}")
        for key in params:
            g = total[key] / batch
            m[key] = .9 * m[key] + .1 * g
            v[key] = .999 * v[key] + .001 * g * g
            params[key] = (params[key] - lr * (m[key] / (1 - .9 ** step))
                           / (np.sqrt(v[key] / (1 - .999 ** step)) + 1e-8)).astype(np.float32)
        history.append(dict(update=step, loss=loss / batch))
        if log:
            log(history[-1])
    after = engine.digest()
    changed = sorted(key for key in before.keys() | after.keys() if before.get(key) != after.get(key))
    if changed:
        raise FrozenBaseChanged(f"{len(changed)} base leaves changed, first {changed[0]}")
    receipt = dict(
        arm=arm, description=px.ARMS[arm], seed=seed, updates=updates, batch=batch, learning_rate=lr,
        fit_rows=len(rows), parameters=px.count(params), base_leaves=len(before), base_leaves_changed=0,
        base_sha256=px.digest(np.array(sorted(before.items()))),
        initial_sha256={key: px.digest(value) for key, value in sorted(start.items())},
        norm={key: float(np.linalg.norm(value)) for key, value in sorted(params.items())},
        first_loss=history[0]["loss"], last_loss=history[-1]["loss"], history=history,
    )
    return params, receipt


def heldout(engine, rows, prefixes, *, draws=HELDOUT_DRAWS, seed=SEED):
    """Same rows and noise for the unmodified policy and every prefix. Offline forward pass."""
    names = ["stock", *prefixes]
    table = np.array([[np.mean([engine.loss(None if name == "stock" else prefixes[name], row, (0, d))
                                for d in range(draws)]) for name in names] for row in rows])
    blocks = {}
    for i, row in enumerate(rows):
        blocks.setdefault((row["episode"], row["t"] // BLOCK), []).append(i)
    means = np.array([table[ids].mean(axis=0) for ids in blocks.values()])
    picks = np.random.default_rng([seed, 53]).integers(len(means), size=(2000, len(means)))
    out = dict(rows=len(rows), blocks=len(means), draws=draws, stock=float(table[:, 0].mean()), prefixes={})
    for j, name in enumerate(names[1:], 1):
        delta = means[:, j] - means[:, 0]
        out["prefixes"][name] = dict(
            loss=float(table[:, j].mean()), minus_stock=float(delta.mean()),
            block_interval95=np.quantile(delta[picks].mean(axis=1), [.025, .975]).tolist())
    return out
