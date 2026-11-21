# Stage 2D CPU result

The 15 controls span all three translation axes. With exact motion, a full linear solve recovered the injected rotation, gains (0.65, 1.25, 1.55) and inverted jaw. The maximum command error on 32 held-out commands was 1.95 × 10⁻¹⁶.

With the same exact motion vectors supplied to the reader at `437da7f6`, the estimated rotation missed by **11.73°**. Unequal actuator gains and the probe's nonorthogonal command directions bias the raw cross-covariance. Supplying the true gains to the rotation solver reduced this error to 0.000008°. The clean rotation and cleared-history checks passed. The public prediction helper also failed when it tried to unpack the three SO(3) outputs into four planar values.

This is a synthetic development result. The analytical oracle establishes observability for this probe and fixture. A learned estimator still has to separate gains from rotation and decode both before a live benchmark can test transfer. The gate ran on Baige Linux CPU in the existing OpenPI environment at harness revision `fa8968e8`. [Numerical evidence](RESULT.json)
