# Paired native replay through the raw recorder

The custom-layout smoke completed eight episodes with no native successes.
The controller transported bowls, but selection and placement failed. One
remaining implementation question is whether the new reset/recording path
changes behavior even on a previously successful native initial state.

Run exactly **two episodes**: original native initialization 0 with each
unchanged fixed BPP demonstration, through `CustomControlEnv`. This is an
explicit `native_control` mode, not a custom relational layout. It has no
rule strata, artificial balance fields, or custom-geometry admission verdict.
The native source bank hash must equal the earlier panel03 receipt. The
two original BDDL goals identify bowl 1 and bowl 2 respectively; these are
fixed before execution. No scene, prompt or motor setting is tuned.

Use the same released checkpoint, strict saved tensors, global seed 7301,
official environment seed 7, prompt dataset seed 0, one environment,
128-pixel cameras, absolute controller, 12-action execution chunks and
562-control public horizon. Use latest reviewed main and a new immutable
`control01` output identity on the qualified one-GPU Pro6K runtime. The
subprocess limit is 20 minutes and scheduler limit 30 minutes, charged to
the remaining original 4 GPU-hour stage cap.

Record raw actions, both official native predicates, fixed public arrays,
state identity, outer reset observation hashes, videos and clocks. Compare
initial observation hashes and native outcomes descriptively to native
panel03 index 0 on the same GPU type. Both candidates previously succeeded
there. No success cutoff or binary motor-admission verdict is introduced.
The second candidate's policy RNG stream need not be bitwise identical to
the prior 20-episode-per-candidate panel; this is an interface control, not
an action-trajectory equality claim.

If native execution remains successful, the new custom geometry becomes
the leading explanation for the smoke failure, within the limits of this
small control. If it fails, inspect the differences before making that
attribution. In either case preserve the result. There is no automatic
32-episode panel and no reader training in this allocation.
