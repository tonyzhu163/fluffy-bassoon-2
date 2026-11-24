"""Check prompt opportunities and the native simulator interface on real, held-out Panda camera tapes."""
import json
import sys
from pathlib import Path

import h5py
import numpy as np

from common import save
from phase_prompt import PhasePrompt


class CameraTape:
    def __init__(self, images, t):
        self.images = images
        self.t = t

    def call(self, op, arg):
        self.i = 0 if op == "create" else self.i+1
        response = dict(language_instruction="open the top drawer and put the bowl inside",
                        observation={"agentview_image":self.images[self.t[self.i]]})
        if op == "create":
            self.returned = response
            return {},response
        response.update(steps=5,terminated=False,truncated=False)
        self.returned = response
        return response


def check(model, source, predictions):
    report = []
    args = dict(suite="libero_goal",task_id=3,arm="Kinova3",variant="BASE")
    with h5py.File(source,"r") as f:
        for case,rec in predictions["cases"].items():
            for mode in ("raw","visual","clock"):
                tape = CameraTape(f[f"data/demo_{case}/obs/agentview_rgb"],rec["t"])
                runner = PhasePrompt(tape,model,mode)
                for i in range(len(rec["t"])):
                    response = runner.call("create",args)[1] if i==0 else runner.call("chunk",None)
                    assert response["observation"] is tape.returned["observation"]
                    assert (response["language_instruction"] != tape.returned["language_instruction"]) == runner.events[-1]["fire"]
                    assert abs(runner.events[-1]["visual"] - rec["instant"][i]) < 1e-12
                calls = [v["t"] for v in runner.events if v["fire"]]
                expected = [] if mode == "raw" else [t for t,p in zip(rec["t"],rec[mode]) if p >= 0.5][:1]
                assert calls == expected
                report.append(dict(case=case,mode=mode,controls=calls))
    return report


if __name__ == "__main__":
    m,src,pred,out = map(Path,sys.argv[1:])
    save(out,check(json.loads(m.read_text()),src,json.loads(pred.read_text())))
