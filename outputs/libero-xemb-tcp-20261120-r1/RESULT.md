# Physical finger-TCP offset passed zero-leak Gate 0 (`8/32` open-loop) and was falsified at Gate 1 (`22/48` vs `23/48` nominal)

Physical MuJoCo finger collision-geometry TCP offsets ($\delta = (\delta_x, \delta_y)$ applied post-reset to `finger1_collision`, `finger2_collision`, `finger1_pad_collision`, and `finger2_pad_collision`) passed the zero-GPUh CPU Gate 0 audit at $r^* = 0.035\text{ m}$ (`8/8` nominal BASE, `8/32` [`25.0%`] shifted open-loop nominal replay, `8/8` tasks post-contact distinguishable, `0/104` pre-contact leaks), **but failed Gate 1 motor realizability** on live closed-loop frozen JAX OpenPI $\pi_{0.5}$ across the eight held-out LIBERO-Pro2.0 development tasks (`job-gmj5rsb0xd59`, `168` paired episodes). Closed-loop $\pi_{0.5}$ survived `23/48` (`47.9%`) shifted trials nominally, while privileged grasp/release TCP compensation (`TcpCompensatedController`) scored `22/48` (`45.8%`) with grasp compensation only and `20/48` (`41.7%`) with grasp-and-release compensation, both maintaining `8/8` on unshifted BASE (`no_base_regression = true`). Because the Gate 1 thresholds (`>= 36/48` [`>= 75%`] oracle rate and `>= +45 pp` gain over nominal) were not met, this workstream is closed at `0.447 / 8.0` scheduler GPU-hours with the official 8-task test split untouched. [Registered plan](PLAN.md) · [Execution record](EXECUTION.md)

## Gate 0: Zero-leak CPU audit (`cpugate03`)

Across all `104` open-loop nominal $\pi_{0.5}$ tape replays (`8` dev tasks $\times$ nominal + `4` cardinal shifts at each of `3` radii in `291.0 s`, 0 GPUh), frame-0 RGB images, all multi-checkpoint pre-contact RGB views (`4–5` checkpoints per episode up to controls `26–82`), and all pre-contact proprioceptive channels (`robot0_joint_pos`, `robot0_joint_vel`, `robot0_eef_pos`, `robot0_eef_quat_xyzw`, `robot0_gripper_qpos`) remained **bitwise identical** to nominal (`0` anomalies). Evaluating `apply_mujoco_tcp_offset` on a temporary `mujoco.MjData` confirmed maximum collision-geom shift error `<= 2.08e-16 m`, pad-gap change `<= 4.16e-17 m`, and unchanged reset `ncon` across all `104` replays. [Gate 0 summary](evidence/cpugate03/gate0-summary.json) · [Exit receipt](evidence/cpugate03/exit.txt)

| Radius $r$ (m) | Nominal BASE | Shifted open-loop | Mean opp RGB L1 | Min opp RGB L1 | Mean opp jaw L1 (mm) | Separable tasks | Passes Gate 0 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `0.025` | `8/8` | `10/32` (`31.25%`) | `8.49` | `2.15` | `11.32` | `8/8` | `false` |
| `0.035` *(selected)* | `8/8` | `8/32` (`25.00%`) | `9.75` | `3.22` | `6.52` | `8/8` | `true` |
| `0.045` | `8/8` | `3/32` (`9.38%`) | `8.05` | `1.31` | `8.61` | `8/8` | `true` |

## Gate 1: Closed-loop $\pi_{0.5}$ motor realizability (`job-gmj5rsb0xd59`)

At $r^* = 0.035\text{ m}$ (`init_state_id = 13`, `56` task $\times$ embodiment rows, `3` paired controllers = `168` live closed-loop episodes on one `NVIDIA RPBZZZ6`), all `56` triples had zero paired reset mismatches and zero opening-proposal SHA-256 mismatches (`reset_mismatches = 0`, `opening_proposal_mismatches = 0`). Both Gate 1 success predicates failed (`shifted_oracle_rate = false`, `shifted_gain = false`), while `no_base_regression` passed (`8/8` across all three arms). [Gate 1 compact summary](evidence/gpu01gate1/gate1-summary-compact.json) · [Smoke summary](evidence/gpu01smoke/gate1-summary.json)

### Results by embodiment and regime (`n = 56` rows)

| Group | Trials | `nominal` | `oracle_grasp_only` | `oracle_grasp_and_release` |
| --- | ---: | ---: | ---: | ---: |
| `tcp_nom` (`familiar_nom`) | `8` | `8/8` (`100.0%`) | `8/8` (`100.0%`) | `8/8` (`100.0%`) |
| `tcp_px` (`+0.035, 0`) | `8` | `7/8` (`87.5%`) | `6/8` (`75.0%`) | `6/8` (`75.0%`) |
| `tcp_mx` (`-0.035, 0`) | `8` | `4/8` (`50.0%`) | `5/8` (`62.5%`) | `5/8` (`62.5%`) |
| `tcp_py` (`0, +0.035`) | `8` | `3/8` (`37.5%`) | `3/8` (`37.5%`) | `3/8` (`37.5%`) |
| `tcp_my` (`0, -0.035`) | `8` | `1/8` (`12.5%`) | `3/8` (`37.5%`) | `1/8` (`12.5%`) |
| `comp_pxpy` (`+0.0247, +0.0247`) | `8` | `4/8` (`50.0%`) | `3/8` (`37.5%`) | `3/8` (`37.5%`) |
| `comp_mxpy` (`-0.0247, +0.0247`) | `8` | `4/8` (`50.0%`) | `2/8` (`25.0%`) | `2/8` (`25.0%`) |
| `familiar_axis` (`px/mx/py/my`) | `32` | `15/32` (`46.9%`) | `17/32` (`53.1%`) | `15/32` (`46.9%`) |
| `compositional_ood` (`pxpy/mxpy`) | `16` | `8/16` (`50.0%`) | `5/16` (`31.25%`) | `5/16` (`31.25%`) |
| **All shifted** | **`48`** | **`23/48` (`47.9%`)** | **`22/48` (`45.8%`)** | **`20/48` (`41.7%`)** |
| **Overall** | **`56`** | **`31/56` (`55.4%`)** | **`30/56` (`53.6%`)** | **`28/56` (`50.0%`)** |

### Shifted results by task (`n = 6` shifted rows per task)

| Development task | `nominal` | `oracle_grasp_only` (grasp trig) | `oracle_grasp_and_release` (grasp / release trig) |
| --- | ---: | ---: | ---: |
| `libero_10-0` | `1/6` | `1/6` (`8`) | `1/6` (`8 / 6`) |
| `libero_10-3` | `2/6` | `2/6` (`6`) | `2/6` (`6 / 6`) |
| `libero_goal-0` | `4/6` | `4/6` (`0`) | `4/6` (`0 / 0`) |
| `libero_goal-1` | `5/6` | `6/6` (`6`) | `5/6` (`6 / 1`) |
| `libero_object-0` | `2/6` | `2/6` (`6`) | `2/6` (`6 / 6`) |
| `libero_object-7` | `2/6` | `3/6` (`7`) | `3/6` (`8 / 5`) |
| `libero_spatial-1` | `4/6` | `3/6` (`6`) | `2/6` (`6 / 4`) |
| `libero_spatial-7` | `3/6` | `1/6` (`6`) | `1/6` (`6 / 5`) |

## Why the Gate 1 hypothesis was falsified

1. **Closed-loop replanning outperforms open-loop tape replay on shifted TCPs:** Under fixed open-loop Gate 0 replay at $r = 0.035\text{ m}$, nominal $\pi_{0.5}$ succeeded on `8/32` (`25.0%`); under live closed-loop replanning in Gate 1, nominal $\pi_{0.5}$ succeeded on `15/32` (`46.9%`) of the same cardinal shifts (`23/48 = 47.9%` across all shifted trials), halving the potential improvement margin before adaptation.
2. **Grasp-trigger blindness on non-grasp manipulation (`libero_goal-0`):** `TcpCompensatedController` ([`src/fluffy_bassoon/xemb/tcp.py`](../../src/fluffy_bassoon/xemb/tcp.py)) only engages compensation when it observes two consecutive closing commands (`prop[6] > 0.0`) with an open jaw (`jaw > 0.05`). On `libero_goal-0`, both oracle arms recorded `0` grasp triggers and `0` release triggers across all `6` shifted trials, collapsing identically to `nominal` (`4/6`).
3. **Trigger-based wrist shifts fail to rescue and can disrupt live $\pi_{0.5}$ trajectories:** Across the remaining 7 tasks where grasp triggers fired (`6–8` triggers per task), `TcpCompensatedController` substitutes a fixed scripted sequence (`lift_clear` for 4 steps, `shift_xy` toward `_p_target = cur_pos - _delta_at_grasp` up to 12 steps, `descend` up to 8 steps, `close_jaw` for 10 steps, `lift_obj` for 5 steps) whenever $\pi_{0.5}$ begins closing. Overall, `oracle_grasp_only` gained `4` shifted trials over `nominal` (`libero_10-3/tcp_my`, `libero_goal-1/tcp_my`, `libero_object-0/tcp_mx`, `libero_object-7/comp_pxpy`) while **losing `5`** (`libero_10-3/comp_mxpy`, `libero_object-0/comp_mxpy`, `libero_spatial-1/comp_pxpy`, `libero_spatial-7/tcp_px`, `libero_spatial-7/comp_pxpy`), and `oracle_grasp_and_release` gained `3` while **losing `6`**. Even with oracle `(dx, dy)`, shifting the wrist at closure triggers fails to open useful motor headroom (`<= 45.8%` vs `47.9%` nominal).

## Conclusion

We stop the physical finger collision-offset line here rather than fitting contact readers on an unviable motor target. Next phase pivots to **held-out physical cross-morphology transfer (`Panda` + `Sawyer` 7-DoF training → held-out 6-DoF `UR5e`)**, reusing the qualified multi-arm simulator in [`outputs/libero-crossarm-20261120-r1/arm_env.py`](../libero-crossarm-20261120-r1/arm_env.py) so that `frame0_probe`, the Phase-1 7-DoF `analytic_joint_jacobian` counterexample, and `analytic_visual_sysid` (on passive task rollouts) are tested on a genuinely held-out 6-DoF morphology.
