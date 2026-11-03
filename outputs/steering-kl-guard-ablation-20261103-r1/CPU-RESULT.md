# First-bank KL guard replay: numerical result

Completed on the allocated Baige development container in52.73seconds, CUDA
hidden, four CPU threads, source1cb0d35a. No new simulator controls, GPU
allocation or production-policy update. All six source files (both banks,
pre-update boundaries and saved update2 checkpoints) hash unchanged afterward.
Full evidence: [CPU-RESULT.json](CPU-RESULT.json).

Each candidate starts from the same actual normalized pre-update1 policy,
empty Adam state, sampler state and complete real first rollout bank. The
only primary difference is target_kl=.02 versus None, at one epoch maximum.
The ten-epoch guarded reference finishes with exactly the same CPU model,
Adam and sampler as guarded1 for both interfaces, because both stop in epoch1.

| Interface/candidate | Gradient steps | Gradient samples / valid bank | Final exact joint KL | Final clip fraction | Critic explained variance |
| --- | ---: | ---: | ---: | ---: | ---: |
| Velocity guarded1 | 7 | 224/3850 | .02243695 | .3034 | -.4826 |
| Velocity unguarded1 | 121 | 3850/3850 | .06921244 | .5938 | .3006 |
| All-layer K/V guarded1 | 86 | 2752/3819 | .02002297 | .6156 | .1995 |
| All-layer K/V unguarded1 | 120 | 3819/3819 | .02505233 | .7484 | .2325 |

The CPU guarded replay reproduces the original GPU step/sample counts. Exact
KL differs only by about1.9e-8 for velocity and6.4e-7 for K/V; no cross-device
bitwise equivalence is claimed. Initial exact joint KL is zero in every
variant, and initial sampled KL is approximately1e-12/1e-9 respectively.

The guard is strongly binding for velocity. Removing it permits one complete
data pass with greater policy drift; its critic fit improves on this same
training batch. K/V sees a smaller exact-KL change but high in-sample clip
fractions. These are optimizer diagnostics on the fitting batch, not native
performance or evidence that either candidate generalizes. The exact Gaussian
KL and the action-sampled approximate diagnostic measure different empirical
quantities after fitting; do not interchange them or interpret.02 as2%.

The simulator readout remains required before choosing a production recipe.
Both original source runs remain human-held at update2. Any changed scientific
optimizer recipe starts under a fresh identity from the clean pre-update
initializer; it is not an in-place modification of the old recoveries.
