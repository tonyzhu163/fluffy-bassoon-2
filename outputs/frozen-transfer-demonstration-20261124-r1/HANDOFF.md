# Baige distributed handoff

The closed-loop study uses one exclusive RTX 4090 in Baige Distributed Training, one worker, the LIBERO native image, the shared `/mowenjie` mount and a 60-minute scheduler limit. The launcher limits the worker command to 3,540 seconds plus five seconds of kill grace. Parent owns submission and monitoring.

## Pinned entry points

The parent deployed the detached study checkout at `/mowenjie/fluffy-bassoon-2/setups/symmeta-perceiver-followups-20261123-r1/src-ef165714d626`, commit `ef165714d626305fa756e10fac2b5f7c45446f1c`. Its launch and qualification scripts use the same native runtime and weight manifest as the cue trial.

CPU qualification on development cores 38/39 uses

```bash
bash /mowenjie/fluffy-bassoon-2/setups/symmeta-perceiver-followups-20261123-r1/src-ef165714d626/outputs/frozen-transfer-demonstration-20261124-r1/qualify.sh ef165714d626305fa756e10fac2b5f7c45446f1c cpu02
```

CPU02 exited zero on the pinned checkout. The dedicated scheduler command is

```bash
bash /mowenjie/fluffy-bassoon-2/setups/symmeta-perceiver-followups-20261123-r1/src-ef165714d626/outputs/frozen-transfer-demonstration-20261124-r1/launch.sh --fresh ef165714d626305fa756e10fac2b5f7c45446f1c gpu01 cpu02
```

Shared receipts land under `/mowenjie/fluffy-bassoon-2/outputs/frozen-transfer-demonstration-20261124-r1/gpu01`. The GPU probe writes `gpu.json` inside the scheduled step. Each real episode writes `episodes/<layout>-<mode>.json`, followed by its verification. The quartet checks enforce identical reset records, public observations, policy inputs, initial proposal and noise. The final reduction writes `result.json` only when all 32 native endpoints pass. The launcher records the exclusive job hostname in `launch.txt` and the exit status and wall interval in `exit.txt`. The portable evidence includes the actual native step vectors, reward and terminal flags for each control.

## Readback

On the Baige development host, the following command verifies the scheduler receipt and replays the paired computation from the saved records. The local portable reader independently scores the archived native steps and source vectors with the Python standard library.

```bash
OPENBLAS_CORETYPE=NEHALEM OPENBLAS_NUM_THREADS=1 /mowenjie/fluffy-bassoon/envs/openpi-215abfb/bin/python -B /mowenjie/fluffy-bassoon-2/setups/symmeta-perceiver-followups-20261123-r1/src-ef165714d626/outputs/frozen-transfer-demonstration-20261124-r1/reduce.py /mowenjie/fluffy-bassoon-2/outputs/frozen-transfer-demonstration-20261124-r1/gpu01
```

The host readback uses the non-fused x86 BLAS kernel to reproduce float32 XY rotation on the job host. The portable reward scorer has no BLAS dependency. [Completed study](NATIVE.md)
