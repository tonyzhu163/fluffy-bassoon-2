"""Execute the production OpenPI/simulator transport with deterministic bounded controls on CPU."""
from pathlib import Path
import sys

import numpy as np

from native import PaintedSimulator, MODES, check_pair
from panel import case
from fluffy_bassoon.xemb.embodiment import get_embodiment_spec
from ur5e_motor import run_episode, write_json
from ur5e_sim_worker import UR5eSimulator


class FixedProbe:
    def infer(self, item, *, noise):
        if item["observation/image"].shape != (224,224,3) or noise.shape != (10,32):
            raise ValueError("Unexpected motor input layout")
        return {"actions": np.tile(np.array([.25,-.1,.2,.1,0,.1,-.8], np.float32), (10,1))}


def run(out, palette):
    c = case("Kinova3", "libero_goal", 4, 16)
    records = {}
    for mode in ("stock", *MODES):
        endpoint = UR5eSimulator(allow_mesa=True) if mode == "stock" else PaintedSimulator(mode, palette, allow_mesa=True)
        with endpoint as sim:
            row = run_episode(sim, FixedProbe(), c, get_embodiment_spec("fam_nom"), max_controls=6)
        if mode != "stock":
            stock = records["stock"]
            check_pair(stock, row)
            if (stock["native_rewards"] != row["native_rewards"] or row["controls"] != 6
                    or row["opening_proposal_sha256"] != stock["opening_proposal_sha256"]):
                raise ValueError("Fixed native controls or reward differ")
            if mode == "raw" and stock["reset"] != row["reset"]:
                raise ValueError("Raw native wrapper changed stock inputs")
        records[mode] = row
    write_json(out / "ipc.json", dict(case=c, records=records, model="fixed six-control diagnostic"))


if __name__ == "__main__":
    run(Path(sys.argv[1]), Path(sys.argv[2]))
