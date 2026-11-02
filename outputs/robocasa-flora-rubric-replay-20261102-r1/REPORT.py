"""Export the frozen replay comparison on allocated Linux CPUs."""
import csv,json,platform
from pathlib import Path
from REPLAY import OUT,read,write,ref
assert platform.system()=='Linux'
r=read(OUT/'RESULT.json');pairs=read(OUT/'PAIRED-EPISODES.json');focus=read(OUT/'FOCUS-58.json')
assert r['completed'] and len(pairs)==634 and len(focus)==58
fields=['id','dataset','task','success','saved_grade','original_grade','candidate_grade','saved_reward','original_reward','candidate_reward','original_P5','candidate_P5','focus']
def flat(p):
    a,b,o=p['rerun_original'],p['candidate'],p['saved_original']
    return dict(id=p['id'],dataset=p['dataset'],task=p['task_key'],success=p['truth']['success'],saved_grade=o['score'],
        original_grade=a['score'],candidate_grade=b['score'],saved_reward=(o['expected_score']-1)/4,
        original_reward=(a['expected_score']-1)/4,candidate_reward=(b['expected_score']-1)/4,
        original_P5=a['probabilities'].get('5'),candidate_P5=b['probabilities'].get('5'),focus=p['focus'])
for filename,items in [('PAIRED-EPISODES.csv',pairs),('FOCUS-58.csv',focus)]:
    with (OUT/filename).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(flat(p) for p in items)
def num(x):return 'undefined' if x is None else f'{x:.3f}'
lines=['# Original versus generalized visual completion rubric','',
    'Frozen comparison on634 existing videos:384 Flora-PPO training episodes and250 earlier calibration episodes. Both prompts scored every video with the same model, camera bytes/order, sampler, output format and reward map. No new robot episodes or policy updates. This is development evidence; independent qualification remains separate.','',
    '| Video bank | Saved original AUC | Rerun original AUC | Candidate AUC | Original false Score5 | Candidate false Score5 | True Score5 lost / gained |',
    '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
for cohort,v in r['cohorts'].items():
    a,b=v['rerun_original'],v['candidate'];neg=a['count']-a['successes']
    lines.append(f"| {cohort} | {num(v['saved_original']['AUROC'])} | {num(a['AUROC'])} | {num(b['AUROC'])} | {a['Score5_false_positives']}/{neg} | {b['Score5_false_positives']}/{neg} | {v['true_full_scores_lost']} / {v['true_full_scores_gained']} |")
f=r['focus_original_failed_Score5']
lines+=['',f"Of the58 TRAIN failures originally given Score5, the candidate gives a lower grade to **{f['candidate_no_longer_full_success']}** and still gives Score5 to **{f['candidate_still_full_success']}**. Rerun-original and candidate grade histograms are retained in RESULT.json, so an original replay difference is distinguishable from a rubric change.",'',
    '| Video bank | False full successes corrected / introduced | Paired two-sided p | Score5 precision: original → candidate | Score5 recall: original → candidate | Original replay grade changes |',
    '| --- | ---: | ---: | ---: | ---: | ---: |']
for cohort,v in r['cohorts'].items():
    a,b=v['rerun_original'],v['candidate']
    lines.append(f"| {cohort} | {v['paired_false_success_corrections']} / {v['paired_new_false_successes']} | {num(v['paired_false_success_p'])} | {num(a['Score5_precision'])} → {num(b['Score5_precision'])} | {num(a['Score5_recall'])} → {num(b['Score5_recall'])} | {v['original_replay_hard_grade_changes']} |")
lines+=['','All58 original full-score failures:','',
    '| Original episode | Native instruction | Saved grade | Rerun original | Candidate | Original reward | Candidate reward |',
    '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
for p in sorted(focus,key=lambda p:p['saved_original']['expected_score'],reverse=True):
    a,b=p['rerun_original'],p['candidate']
    lines.append(f"| {p['native_id']} | {p['native_instruction']} | 5 | {a['score']} | {b['score']} | {(a['expected_score']-1)/4:.4f} | {(b['expected_score']-1)/4:.4f} |")
lines+=['',
    'A greedy Score5 is a full-success claim. The expected-grade reward remains a soft five-grade quantity; a lower greedy grade does not imply binary-failure reward. AUROC, precision and recall must be considered together. All failed and successful cases remain in the exported paired records.','',
    'The text was frozen before replay. These same videos informed the prior error investigation, so this comparison cannot establish generalization to unseen videos or a PPO improvement. Retained original/rerun differences measure numerical and batch-composition reproducibility separately.','',
    '[Full paired CSV](PAIRED-EPISODES.csv) · [58-case CSV](FOCUS-58.csv) · [Machine-readable comparison](RESULT.json) · [Frozen manifest](MANIFEST.json)','']
(OUT/'RESULT.md').write_text('\n'.join(lines))
write(OUT/'REPORT-RECEIPT.json',dict(completed=True,result=ref(OUT/'RESULT.json'),csv=ref(OUT/'PAIRED-EPISODES.csv'),focus_csv=ref(OUT/'FOCUS-58.csv'),report=ref(OUT/'RESULT.md')))
print(json.dumps(dict(completed=True,focus=f,cohorts=r['cohorts'])))
