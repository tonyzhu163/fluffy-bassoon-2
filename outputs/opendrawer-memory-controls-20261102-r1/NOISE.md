# Native resampling noise, before new history collection

The four completed audited native-resampling banks contain80 cases ×8 draws
(640 episodes), with no new simulator work. All cases were selected because
one anchor rollout failed; each task uses600 controls. This is conditional
failure-bank headroom, not native success across the full task distribution
or the separate750/900-control AD/DPT panel.

| Task (20 cases each) | Fail all8 | Mixed outcomes | Succeed all8 | Successes /160 | Recover at least once /20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| OpenDrawer |1|19|0|79 (49.4%)|19|
| DrawerToCounter |5|15|0|43 (26.9%)|15|
| CoffeeSetupMug |4|16|0|44 (27.5%)|16|
| TurnOffStove |11|9|0|16 (10.0%)|9|
| Total |21|59|0|182 (28.4%)|59|

OpenDrawer is particularly noisy:12/20 cases succeeded on3–5 of8 draws, and
two different draws from the same case disagree44.5% of the time on average
(unbiased empirical pairwise estimate). Corresponding disagreement rates
are31.3% DrawerToCounter,33.9% Coffee and16.8% Stove. Across tasks,28/80 cases
have3–5 successes. A redraw alone rescues19/20 OpenDrawer anchor failures,
so attempt trends must be compared against empty-memory controls on the
same streams before attributing them to memory.

Per-case k/8 estimates, all0..8 histograms, exact outcomes and95% Wilson
intervals are retained in [NOISE.json](NOISE.json) and [NOISE.csv](NOISE.csv).
Eight failures still allow p up to32.4% in that interval; eight successes
would allow p down to67.6%. Thus observed all-fail/all-success does not mean
intrinsically impossible/deterministic. Best-of8 uses simulator truth as an
oracle selector; it measures observed resampling headroom, not a deployable
strategy or an upper bound on what a learned policy could achieve.

The same script ran successfully on allocated Linux CPUs in Baige job
`job-qdtkpou7rv50`, CUDA hidden, source `a9006970`; all four task counts
matched the local audit summary. Existing native audits qualify the input
identities; missing chunk records in213 carried native rows remain disclosed
and are not converted into complete learning histories. No new GPU work,
PPO or source histories were collected for this analysis.
