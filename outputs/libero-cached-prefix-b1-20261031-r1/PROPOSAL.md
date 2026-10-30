# Cached-prefix B1 qualification proposal — no reservation

This preparation is static code and prepared tests only. It creates no hold,
allocation, source actions or learner updates. The execution owner controls
the remaining original32GPU-hour LIBERO budget and must reconcile/fund any
actual qualification before launch. Neither closed systems hold is reopened.

The owner's measured native batch14 run at9bc9c9c, jobccniwh46yq9w, completed
14K4trials/56episodes,8,313actions,795valid chunks and two SGD steps. Reported
collection651.24seconds, replay4.09, update13.46 and852schedulerGPU-seconds;
sampled device peak10.55GiB and Torch peak1.25GiB. Individual/merged likelihood
and value replay errors were0, and full frozen-motor hashes were unchanged.
All56episodes were already successful; this establishes no adaptation benefit.
These are owner-reported preparation inputs, not a new measurement here.

## First proposed actual comparison

Propose one exclusive GPU with at most600schedulerGPU-seconds, including
startup, compilation, reading, parity, full model hashes and cleanup. This is
an upper bound for owner review, not a new resource authorization or a claimed
cost saving. Stop and retain partial evidence if it does not fit; no automatic
retry or larger allocation follows.

Use a hash-bound index of the795actual source inference records retained by
the measured batch14 run, preserving each current public observation, actual
instruction, explicit native initial noise, original features and generated
actions. Compare newly executed native and cached B1 inference on those same
inputs. Retain source/root/record hashes, exact original model checkpoint and
OpenPI/runtime pins, GPU probe and before/after full parameter hashes. Add a
small predeclared hash-bound set of existing SAC records including unbounded
warmup coordinates; never modify a training sampling scheme to make the test.

No simulator, robot action, PPO/SAC update or new case collection is needed
for this recorded-input kernel gate. Count and report every actual native and
cached VLA query and the entire scheduler duration; zero robot actions is not
zero compute cost. CUDA/model work must run in the owner's actual allocation,
not on macOS or the development SSH host.

Per record, run native and cached feature/action calls from the same saved
native motor RNG state and identical explicit noise. A cached feature query
must precede cached inference. Restore only for paired qualification; source
collection continues its unchanged original motor stream. Require equal final
native RNG states, feature maxerror≤2e-4 and action maxerror≤1e-5, and record
native-versus-original-artifact differences separately from cached-versus-native
differences. Cross-hardware native drift is diagnostic, never silently blamed
on caching or accepted by loosening thresholds.

Exercise both real scalar call orders: repeated PPO frozen-feature reads before
decoding, and SAC's recorded frozen-feature→infer calls with its separate
pixel actor/replay adapter. Also verify generic repeated host-feature reads
after decoding without reusing consumed KV; do not label that additional
interface check as measured source-SAC replay cost.
Keep public feature/observation/instruction identity and one-time KV
consumption; clearing/restoring a complete-boundary key must reproduce the next
feature/action/RNG result without serializing KV. Include invalid-input and
failed-decode ordering through prepared CPU fakes before model qualification.

Report matched native/cached model-phase medians and total funded cost with
their limits. This does not establish a whole-source speedup, live cached
trajectory equivalence, general reset compatibility or robot benefit. A later
already-funded actual source continuation can measure collection time and
complete-batch replay/update behavior after the owner qualifies/adopts the
adapter. Do not delay the independent SAC top-up budget request or RoboTwin
continuation while waiting for an unmeasured cache benefit.

The old c6 systems qualification remains759physical actions/411GPU-seconds
with zero source-training credit. Its released/retained balance has already
been reconciled elsewhere; none is available to this proposal.
