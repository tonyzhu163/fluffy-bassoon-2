# Next RoboTwin online PPO run

Prepared and Linux-CPU-qualified; **not admitted or launched**.

Request: raise this task's cumulative GPU cap from **50 to 61 GPU-hours**.
Closed conservative usage is179306GPU-s (49.8072h). This allocation reserves
at most39600GPU-s (11h), giving a worst-case cumulative218906GPU-s
(60.8072h), below the proposed219600GPU-s cap. Startup, cleanup and failed
attempts count; scheduler fault retries remain disabled.

The goal of this allocation is to verify the optimized full online cycle and
produce two fresh policy versions. Two iterations do not establish convergence
or finish the six-task campaign. They resolve the missing policy1 collection
and provide measured costs for a longer curve and matched GRPO.

| Item | Fixed configuration |
| --- | --- |
| Cluster | Baige Pro6K; actual device/visibility qualified inside the job |
| Allocation | 4GPU,128CPU,1024GiB host RAM,256GiB shared memory |
| Time | 150min worker limit;160min scheduler limit;165min creation-to-release watchdog |
| Motor/task | Original joint Motus revisioneffc6e1a; `place_phone_stand` only |
| PPO data | 256 environments,4 rounds/iteration,2 iterations; group8 |
| Learner | Global2048,five epochs,micro256×accum2×four devices;30 Adam updates/rank |
| Native execution | H32/H192,3cameras,14D decoded controls,K5 Flow-SDE/noise.3 |
| Engineering change | Deferred setup and reusable joined executor, shared by train/eval |
| Checkpoints | Complete native actor/Adam plus all worker sampler/RNG sidecars at0,1,2; cold readback/stageback |
| Development evaluation | Same16 source DEV cases at policies0,1,2; no SGD in evaluation |
| Reserve | At leastmax(8GiB,8%) free on each GPU; host usage below90% |
| Outputs | Freshmotusoptimized01 identity; ≤192GiB project BOS,256GiB job-local staging; no new worker PFS outputs |

The capacity check runs before every native actor microbatch, rollout call and
environment deadline/action boundary. It uses the shared job-wide observer;
failed/stale reserve reports reject the run while retaining its latest complete
checkpoint. It does not manufacture memory occupancy or change the PPO loss.
Small-scene lifecycle equality is already verified; full64-scene combined
capacity remains an actual-device qualification in this allocation.

CPU qualification used code77d585dec7b794e265c620ca6b7c7f9d2b493a4d. The exact
launcher admission body passed model/normalization/source hashes and real
Linux write/read/fsync; the full native schema passed with only placement
hardware edges mocked. Five capacity-report fixtures exercised rejection.
The unchanged source contract correctly rejects launch without budget approval.
No native model, simulator or CUDA was constructed by this preflight.
[Receipt](CPUMOTUSOPTIMIZED01.json).

Fresh queue observation at preparation showed28 free accelerators,1904CPU
cores and28952GB RAM in the target queue; listed node typeRPBZZZ6.
This is availability, not GPU health or a reservation. Recheck placement,
limits and actual GPU type before scheduling. The development host had1956GiB
local staging free and238GiB project PFS free; outputs use BOS and job-local
disk. BOS statfs is virtual and does not prove remaining account quota;
`quota -s` supplied no usable quota report. Recheck actual-job storage and
write admission; do not infer quota from statfs or move existing caches.

After approval, commit the admission metadata to main, fetch it into a fresh
detached source and verify the qualified runtime file hashes are unchanged.
Launch the checked-in command with that exact new commit:

```sh
bash outputs/robotwin-restoration-20261106-r1/train_motus_optimized.sh \
  <reviewed-admitted-commit> \
  /mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1/motusoptimized01-signals
```

Verify advancing optimizer counts and new-policy collections, then monitor
every20min with direct job ownership/status. Failures require diagnosis and a
new output identity, within the same approved cumulative cap. Successful
training still requires an independent new-process restore and a controlled
learning assessment; do not call a completed launcher a positive PPO result.
No held-out task is trained. The complete project plan remains active.
