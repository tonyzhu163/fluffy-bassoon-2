"""Qualify renderer cleanup against identical-state RGB, with forced collection."""
import argparse
import gc
import json
from pathlib import Path

import numpy as np


def bind_context_cleanup():
    from robosuite.utils.binding_utils import MjRenderContext
    original = MjRenderContext.__del__
    def cleanup(self):
        context = getattr(self, 'gl_ctx', None)
        if context is not None:
            context.make_current()
        original(self)
    MjRenderContext.__del__ = cleanup


def main():
    from collect import CAMERAS, reset, picture, stable_counter_regions
    ap = argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--fix',action='store_true')
    args = ap.parse_args()
    args.output.mkdir(exist_ok=False)
    # Prevent old contexts being finalized before the deliberate intervention.
    gc.disable()
    if args.fix:
        bind_context_cleanup()
    stable_counter_regions()
    wrapped, obs = reset(120700)
    env = wrapped.unwrapped.env
    def render():
        return {k: env.sim.render(width=256,height=256,camera_name=k.removeprefix('video.'))[::-1].copy()
                for k in CAMERAS}
    before = render()
    physics = env.sim.get_state().flatten().copy()
    collected = gc.collect()
    after = render()
    np.testing.assert_array_equal(env.sim.get_state().flatten(), physics)
    picture(args.output/'before.png',before,'Before garbage collection')
    picture(args.output/'after.png',after,'After garbage collection')
    result = dict(fix=args.fix,collected=collected,physics_unchanged=True,
        cameras={k:dict(equal=bool(np.array_equal(before[k],after[k])),
            differing_components=int(np.count_nonzero(before[k] != after[k])),
            max_delta=int(np.abs(before[k].astype(np.int16)-after[k].astype(np.int16)).max())) for k in CAMERAS})
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    wrapped.close()
    del env,wrapped
    gc.collect()


if __name__ == '__main__':
    main()
