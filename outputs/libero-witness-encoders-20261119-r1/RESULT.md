# Robot tuning improved one bowl completion classification

With the same 16 fit tapes, five images per camera per episode and fixed distance-to-centroid rule, the LIBERO-tuned π0.5 vision tower classified completion on every development tape. The general PaliGemma PT-224 tower missed one and pooled RGB missed two. All three selected the correct destination on both sides of each eligible strict group. Changing the representation did not change which useful correction the reader found there.

| Frozen visual input | Completed vs missed practice on development | Strict correct retry, own / swapped history | All 32 development assignments with native feedback |
| --- | ---: | ---: | ---: |
| π0.5 LIBERO vision tower | 16/16 | 4/4 · 0/4 | 23/32 |
| General PaliGemma PT-224 vision tower | 15/16 | 4/4 · 0/4 | 24/32 |
| Mean RGB in the same 2 × 2 grid | 14/16 | 4/4 · 0/4 | 24/32 |

Twelve of the 24 possible ordered development triples also qualified, and all three readers selected 24/24 correct destinations on these reused cases. Through all 32 cyclic assignments, a hindsight decision with complete practice evidence and observed feedback could reach 24/32. With full practice history and no reward information the upper bound was 16/32. The net one-retry difference came from two dropped practices and two zeros on a query pose where only the plate was reachable. Both hidden goals remained compatible with the practice evidence. The difference does not establish superiority in recognizing the hidden goal. [Robot tower](robot.json) · [General tower](vlm.json) · [Matched pixels](rgb.json) · [Crossed feedback and information bounds](crossed.json)

## Model and task scope

The two frozen SigLIP towers each loaded all 437 tensors, 412,442,352 parameters, with matching architecture and different weight fingerprints. We used the final patch grid, pooled into quadrants and differenced successive frames from controls 0, 75, 150, 225 and 300 in both cameras. CPU inference used BF16 in both towers. The supervised reader was fitted separately per destination on four bowl poses. Evaluation used four other bowl poses with the same two goals. The 32 assignments and four strict choices share these four physical poses. The matched RGB input uses the same frames and grid. The earlier 13-frame finer-grid pixel reader is reported in each output separately. [Protocol](PLAN.md) · [Simulator and source tapes](../libero-visual-witness-20261119-r1/RESULT.md)

The motor's original actions came from OpenPI pi05_libero. For the visual probe we loaded the public LeRobot pi05_libero_finetuned_v044 conversion at `7152f13e` and Google's PaliGemma PT-224 at `35e4f464`. We obtained the complete vision tower from the authenticated official PT checkpoint. No mix checkpoint entered the comparison. The result concerns sensory features in these endpoints. It does not identify pretraining versus LIBERO fine-tuning, the quality of the full contextual VLM, or a transformer learning an update rule. The generic visual checkpoint is the PaliGemma PT model referenced by OpenPI's training loader. [Weight receipt](pt-weights.json) · [OpenPI weight loader](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/weight_loaders.py)

Baige CPU completed all three arms at revision `848334f0`. Robot, general and pixel readouts took approximately 115, 62 and 3 seconds, respectively. The crossed readout ran at `d6e3ec79`. Zero GPU-hours. [Launch](launch.txt)
