"""Apply each fixed visual completion score to both native reward assignments of every history."""
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np


def main(feedback, directory, out):
    data=json.loads(feedback.read_text())
    answer={}
    for split in ("fit","development"):
        rows=data[split]["cases"]
        paired=defaultdict(lambda:np.zeros(2))
        no_feedback=defaultdict(lambda:np.zeros(2))
        for c in rows:
            a,b,q=c["a"],c["b"],c["q"]
            key=(a,b,q,*c["variants"])
            no_feedback[key]+=c["continuation_hits"]
            paired[(*key,*c["rewards"])]+=c["continuation_hits"]
        summary={"exact_history_reward_hindsight":float(sum(x.max() for x in paired.values())),
                 "exact_history_without_reward_hindsight":float(sum(x.max() for x in no_feedback.values())),
                 "assignments":len(rows)}
        for arm in ("rgb","robot","vlm"):
            reads=json.loads((directory/f"{arm}.json").read_text())["splits"][split]["tapes"]
            probs={(r["pose"],r["goal"],r["variant"]):r["score"] for r in reads}
            success=0.
            for c in rows:
                va,vb=c["variants"]
                pa,pb=probs[c["a"],0,va],probs[c["b"],1,vb]
                p=0.5 if abs(pa-pb)<=1e-12 else float(pa<pb)
                if any(c["rewards"]):
                    assert sum(c["rewards"])==1
                    p=c["rewards"][0]
                x,y=c["continuation_hits"]
                success+=p*x+(1-p)*y
            summary[arm]=success
        answer[split]=summary
    out.write_text(json.dumps(answer,indent=2)+"\n")


if __name__=="__main__":
    main(*map(Path,sys.argv[1:]))
