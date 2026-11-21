# Frozen motor check

On the predetermined LIBERO Goal 1 development start (layout 13, seed 20500101), π₀ succeeded on simulated Panda and failed on UR5e. π₀.₅ succeeded on both. All four episodes used the nominal command transform, native terminal success and five executed controls per inference. The same-arm resets matched across policies, including both camera frames. Both arms used the same released scene and prompt.

This establishes that the campaign π₀ loader and action path can produce a source-arm success. One start per cell cannot distinguish vision, body geometry or closed-loop control as the cause of the UR5e failure. The one-Pro6K Baige job `job-ev9ie3w3ki3q` exited 0 at wrapper pin `565af610`, using the official-wave dependency pin `437da7f6`. [Native outcomes and reset checks](evidence/result.json)
