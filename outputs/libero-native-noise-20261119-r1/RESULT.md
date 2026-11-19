# Independent π0.5 samples usually started millimetres apart

On 300 identical-reset pairs from 15 standard LIBERO-90 fitting tasks, independent frozen-policy retries had a median wrist separation of **2.21 mm after five controls**, before either policy replanned. By control 50, the median was **1.15 cm** and the 90th percentile was **3.91 cm**. Exactly one retry succeeded on **20/300 pairs (6.7%)**. The snapshot includes all 20 poses per task for the 15 tasks fully published at selection time. This is an exploratory fitting-set measurement. [Sampling plan](PLAN.md) · [Per-task counts and numerical results](result.json)

| Controls executed | Pairs reaching that control | Median 3D wrist separation | 90th percentile |
| ---: | ---: | ---: | ---: |
| 1 | 300 | 0.24 mm | 0.59 mm |
| 5 | 300 | 2.21 mm | 5.77 mm |
| 25 | 300 | 0.87 cm | 1.97 cm |
| 50 | 300 | 1.15 cm | 3.91 cm |
| 100 | 289 | 2.31 cm | 15.29 cm |

Both retries succeeded on 74 poses and both failed on 206. The two one-success cells were 14 and six. A draw chosen uniformly from each recorded pair succeeded on average in 28.0% of cases (168/600 episodes). With hindsight, choosing the better **whole rollout** would give 31.3% (94/300 poses), a 3.3 percentage point difference. A reader choosing one action chunk every five controls can visit states that occur in neither recording. Its potential gain has to be measured in live rollouts.

The collector calls the same π0.5 with the same current observation and prompt twice. Each call gets an independent Gaussian starting latent and returns ten seven-dimensional controls. An independent coin selects which five-control prefix to execute during collection. At the opening decision the two stored slots differed by a median of 0.0436 in the norm of the first translation command (90th percentile 0.1227, native command units, 600 candidate pairs). The wrist distances above come directly from simulator positions in metres. The five-control distances compare the executed samples from two reset-matched streams. Later distances accumulate fresh draws and each trajectory's own observations. The historical collector required the two complete opening proposal arrays across streams to differ for publication. All statements concern the published pairs.

The figure shows the collector's original qualification example, task 0, repeat 0. The two retries began with similar reaches. One closed the drawer after 294 controls and the other reached the 400-control horizon. The case illustrates one of the 20 discordant outcomes. [Physical paths and wrist height](trajectories.png)

### Implications for the steering experiment

The logged random choices establish a baseline for sampling variation. A test of whether a specific first action matters needs both five-control prefixes executed from one simulator state with matched continuation policy and future draws. The independent retries here allow the entire continuation to vary. A test of whether history helps requires prospective choices from genuine, cleared and same-task mismatched practice with the same trained reader. For the LoRA experiment, compare the native and gated action expert on the same flow draw. Forcing the entire adapter path to zero must recover the base chunk. Measure both rescues and regressions once the model makes its own gate decisions. A memoryless LoRA fit controls for improvement from training alone. [Paper protocol](../../docs/PAPER.md)

### Reproduction

The published source is `4750c0971e6ba9b467faa47aac0dd8641f7d5076`, with the manifest seeds in each trial, simulator seed 7 and the design hash in the results. The archive worker's first analysis attempt at `8ff2f196` expected a noise digest introduced after the original collector and stopped before producing any measurements. The successful Baige CPU run at `f446bdbd4fc9ccff80ea810a25b1ede461de4372` read both original stream IDs and replayed all recorded selection coins, checked identical starting wrists, states and embeddings, matched opening commands to the chosen proposal, and verified native outcomes. The interpretation review distinguished whole-rollout hindsight from per-decision selection and specified the matched-noise test above. Zero new GPU-hours. [CPU run log](run.log) · [Analysis](measure.py) · [Figure source](figure.py)

```sh
root=/mowenjie/fluffy-bassoon-2
run=libero-native-noise-20261119-r1
cd "$root/setups/$run/source-f446bdbd/outputs/$run"
data="$root/outputs/libero90-full-action-20261119-r1/corpus01/fitting"
interp="$root/setups/libero-meta-learning-20261117-r1/env/bin/python"
out="$root/outputs/$run/cpu02"
"$interp" measure.py "$data" "$out/result.json"
"$interp" figure.py "$data" "$out/trajectories.png"
```
