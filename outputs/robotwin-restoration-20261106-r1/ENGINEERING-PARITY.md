# Shared RoboCasa engineering in RoboTwin

Applies to the corrected Motus PPO and GRPO controls and the source readout.
This is an audit of the shared mechanisms present in executed code. It does
not claim that every RoboCasa optimization transfers unchanged to SAPIEN.

| Mechanism | RoboTwin training | RoboTwin evaluation | Evidence / limit |
| --- | --- | --- | --- |
| Scheduler GPU attestation, UUID/PCI binding | Actor, rollout and environment workers | Same setup and allocated renderer binding | `worker-gpus/`, `renderers/`; GPU probe inside the actual step. |
| Scene offload and executor recycling | Enabled; setup deferred until needed | Enabled through the same environment setup | `setup-profiles/`; source readout calls the same helper. |
| Batched physical controller and skipped intermediate renders | Enabled | Enabled | Actual controller is hashed; native motor actions and passive physical traces retained. |
| Frozen image encoder eval mode | Reapplied after native model.train | Native model eval | Frozen shard hashes checked after each update; avoids training-mode encoder drift. |
| Bounded asynchronous evidence I/O | Enabled | Same native writer | Flush at boundaries/evaluation completion; receipts retain original arrays and hashes. |
| Local NVMe caches, logs and checkpoint staging | Enabled | Full source checkpoints stage locally | Cold outputs/checkpoints on project BOS; zero new worker PFS outputs. |
| Global batch preserved while tuning execution | Global2048, five epochs, micro32/accum8/world8 | Original native inference batch2 | B256 actor rejected by measured joint-likelihood drift. Larger memory use cannot override this numerical condition. |
| Full optimizer/scheduler/scaler/sampler/RNG boundaries | Checkpoints0/1/2 | Full24-worker restore, then read-only evaluation | Strict owned Adam semantics; independent cold hashes and actual process restore required. |
| Combined simulator/model resource reserve | Checked during actual updates and collection | Checked during sampling/panel starts; allocation-wide monitor | Eight-GPU minimum free reserve max(8GiB,8%); host peak under90%. Actor-only memory is insufficient evidence. |
| Evaluation pairing | Common seeded progress panels with training RNG restored |64 DEV scenes; actual ODE draws recorded per chunk and compared as tensors | Native batch, cameras, horizon and controller stay fixed. Confirmation uses a disjoint64 cases after method selection. |

PPO and GRPO completed two actual combined cycles and all three checkpoints.
Both independent cold checkpoint/strict Adam and whole-batch NumPy audits
passed. Each covers three complete boundaries and both actual rollout batches.
Both source DEV readouts subsequently passed full-model instrumentation, all24-worker
restores for checkpoints0/2, all8 actor-state guards, and128 paired physical episodes.
Independent CPU cold/physical/noise readbacks passed for both methods. These are
actual GPU and cold-readback qualifications. Disjoint confirmation also passed all execution and independent readback
checks, but7/64→10/64 with p=.581 failed the effectiveness gate. The shared
engineering optimizations now support valid training and evaluation; an effective
learner remains unestablished.

Compilation, different model kernels, larger actor batches or changed sampling
cadence require their own numerical qualification. They are not implicitly
covered by this parity statement. Throughput and resource measurements should
report actual useful collection/update time and peak allocation, rather than
VRAM occupancy as an objective by itself.
