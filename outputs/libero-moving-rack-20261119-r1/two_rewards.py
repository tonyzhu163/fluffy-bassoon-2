"""Post hoc third-attempt replay with two independent, equally charged practice poses."""
from collections import defaultdict
from itertools import permutations, product
from pathlib import Path
import sys,json
from refit import read,OPTIONS,REGIMES


def examples(data,triples,probe,bits):
    for (s,t,q),g in product(triples,REGIMES):
        b1=int(data[s,g,probe]['hit'])
        b2=int(data[t,g,probe]['hit'])
        n=(b1+b2 if bits==2 else b2 if bits==1 else 0)
        yield n,b1+b2,q,g


def learn(data,pairs,probe,bits):
    counts=defaultdict(lambda:[0]*len(OPTIONS))
    for n,_,q,g in examples(data,pairs,probe,bits):
        for i,a in enumerate(OPTIONS):
            counts[n][i]+=data[q,g,a]['hit']
    totals=[sum(v[i] for v in counts.values()) for i in range(len(OPTIONS))]
    return [OPTIONS[max(range(len(OPTIONS)),key=counts.get(n,totals).__getitem__)] for n in range(bits+1)]


def evaluate(data,triples,probe,bits,lookup):
    scores={g:dict(n=0,hit=0,final=0,any_of_three=0,total_rewards=0) for g in REGIMES}
    for n,before,q,g in examples(data,triples,probe,bits):
        r=data[q,g,lookup[n]]
        row=scores[g]
        row['n']+=1
        row['hit']+=r['hit']
        row['final']+=r['final']
        row['any_of_three']+=bool(before or r['hit'])
        row['total_rewards']+=before+r['hit']
    return scores


if __name__=='__main__':
    root=Path(sys.argv[1])
    src=read(root/'branch01/fitting.jsonl',root/'branch01/fresh.jsonl')
    new=read(root/'test01/native.jsonl')
    probe=json.loads((root/'expanded.json').read_text())['tree'][0]
    train=list(permutations(range(6),3))+list(permutations(range(6,12),3))
    poses=list(range(12,24))
    cyclic=[(poses[(i-2)%12],poses[(i-1)%12],q) for i,q in enumerate(poses)]
    result={}
    for bits in range(3):
        lookup=learn(src,train,probe,bits)
        result[bits]=dict(lookup=lookup,cyclic=evaluate(new,cyclic,probe,bits,lookup),
                          all_triples=evaluate(new,list(permutations(poses,3)),probe,bits,lookup))
    print(json.dumps(dict(probe=probe,results=result),indent=2))
