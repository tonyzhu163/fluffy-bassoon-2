"""Render the preregistered summaries from completed receipts. No model or selection code."""
import json
from pathlib import Path
from statistics import mean
import sys

HERE=Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_bytes())


def table(headers, rows):
    print("| "+" | ".join(headers)+" |")
    print("|"+"|".join([" --- "]*len(headers))+"|")
    for r in rows:
        print("| "+" | ".join(f"{v:.3f}" if isinstance(v,float) else str(v) for v in r)+" |")
    print()


def main(src, archive):
    seal=read(src/"sealed.json")
    sel=read(src/"validation-temperature.json")
    alpha=read(src/"validation-shrinkage.json")
    test=read(src/"source-test.json")
    print("# Registered panel readback\n")
    print("## Source calibration\n")
    print(f'{seal["measured_parents"]} physically acquired training maps, {seal["signed_views"]} controller-equivalent signed views, {seal["templates"]} unique weighted matrices. Training coordinate residual variance {seal["basevar"]:.6g} in 50 mm units. Selected temperature {seal["T"]:g}, isotropic multiplier {seal["isotropic_multiplier"]:g}. Validation predictive score used {sel["valid"]}/{sel["total"]} support records.\n')
    table(["Temperature","Held physical NLL / coordinate"],zip(sel["temps"],sel["score"]))
    table(["Isotropic multiplier","Validation passive general setpoint error, mm"],zip(alpha["multipliers"],alpha["means"]))
    print("## Disjoint source tasks\n")
    metrics=["nll","coverage90","true_hand","hand_brier","ess","top","nearest_dist","nearest_mass","exact_mass","cov_trace"]
    table(["Weights / subset","Held rows",*metrics],[[k,v["held_rows"],*[v.get(m,"–") for m in metrics]] for k,v in sorted(test.items())])
    report=read(archive/"fresh-report.json")
    primary=report["general/passive"]
    print("## Fresh passive general maps\n")
    ms=["setpoint","oracle_gap","progress","shortfall","orthogonal","world_norm","effort","saturation","issued_to_box","issued_to_clip"]
    print("Distances, projections, shortfalls and norm of the world setpoint use mm. Effort is RMS normalized issued XY norm. Saturation is the fraction of actions at a box face. The two issued-space columns use mm-equivalent.\n")
    table(["Decision",*ms],[[name,*[v["metrics"][m] for m in ms]] for name,v in primary.items()])
    tasks=sorted(primary["mean_box"]["tasks"])
    table(["Comparator minus risk, mm","Mean","95% task interval",*tasks],
          [[n,v["comparator_minus_risk"],f'[{v["ci95"][0]:.3f}, {v["ci95"][1]:.3f}]',
            *[v["tasks"][t] for t in tasks]] for n,v in primary.items()])
    for split in ("fresh","developmental"):
        result=read(archive/f"{split}-report.json")
        post=read(archive/f"{split}-posterior.json")
        print(f"## {split.capitalize()} factorial\n")
        names=list(result["general/passive"])
        table(["Map class / schedule",*names],[[cell,*[tab[n]["metrics"]["setpoint"] for n in names]]
                                               for cell,tab in sorted(result.items())])
        print("### General maps by excitation and determinant\n")
        for s in ("passive","along","lateral"):
            tab=result["general/"+s]["mean_box"]
            print(f'{s} paired box-mean minus risk by body '+", ".join(f"{k} {v:.3f} mm" for k,v in sorted(tab["by_arm"].items()))+". Pulse arms "+", ".join(f"{k} {v:.3f} mm" for k,v in sorted(tab["by_pulse"].items()))+".\n")
        strat=[]
        for s in ("passive","along","lateral"):
            for h in (-1,1):
                rows=[r for r in post if r["family"].startswith("general") and r["strategy"]==s and r["hand"]==h]
                strat.append([s,h,len(rows),*[mean(r[m] for r in rows) for m in ("rank_ratio","true_hand","ess","top","cov_trace","nearest_dist","nearest_mass","exact_mass")]])
        table(["Schedule","True hand","Supports","Weak/strong","True-hand mass","ESS","Top mass","Trace C","Nearest distance","Nearest mass","Exact mass"],strat)
        print("### Error by hand and pulse\n")
        raw=read(archive/f"{split}-rows.json")
        contrasts=[]
        for arm in ("IIWA","UR5e"):
            for h in (-1,1):
                for mode in ("passive","along+","along-","orth+","orth-"):
                    cell=[r for r in raw if r["arm"]==arm and r["hand"]==h and r["family"].startswith("general") and r["mode"]==mode]
                    means={n:mean(r["setpoint"] for r in cell if r["method"]==n) for n in ("mean_box","risk","isotropic")}
                    contrasts.append([arm,h,mode,means["risk"],means["mean_box"]-means["risk"],means["isotropic"]-means["risk"]])
        table(["Body","True hand","Pulse","Risk error, mm","Box mean minus risk, mm","Isotropic minus risk, mm"],contrasts)


if __name__=="__main__":
    main(Path(sys.argv[1]),Path(sys.argv[2]))
