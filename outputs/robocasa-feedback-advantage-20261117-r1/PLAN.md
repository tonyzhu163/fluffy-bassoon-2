# Protocol: Native Information-Advantage and DIT Weighting Qualification on RoboCasa365

## Purpose
Before training or scaling any larger history-conditioned VLA recipe, qualify a small, bounded native RoboCasa365 experiment where:
1. A persistent unobserved trial variable $z \in \{-1, +1\}$ changes which attainable action succeeds while current observations and task language stay identical.
2. Matched prior-attempt feedback demonstrably helps a reference selector choose a successful action strictly better than the best fixed action and any current-observation reference.
3. A small causal history reader (`CausalDecoderCore`, 2 layers, 4 heads, width 128) trained with matching persistent-variable trial structure learns from that feedback across three independent fit seeds (`0, 1, 2`), isolating behavior cloning, DIT estimated-advantage weighting ($\tau = 1.0$ vs $\tau = 0.1$), and known-outcome weighting with 100% pre-success query coverage and 0% post-success suffix exposure.

## Stage 1: Native Reference Information-Advantage Qualification (`OpenDrawer`)

- **Motor and environment:** Frozen public Human300 $\pi_{0.5}$ (`pi05_pretrain_human300/multitask_learning/75000`, revision `c484448a`, bundled normalization), predict 50, execute 5, 10 denoising steps, $H = 600$ controls, `robocasa/OpenDrawer`, `pretrain` split, native success predicate.
- **Canonical start pairing:** Select a recurring start signature (`matched_start`) per scene before any action or reward so all branches on a scene share bit-identical initial physics state, camera poses, geometry, public state, prompt, and camera images.
- **Persistent calibration variable:** Each trial has a persistent hidden lateral approach offset $b = z \cdot B$ with $z \in \{-1, +1\}$ and $B = 0.04\text{ m}$ applied to the right arm OSC goal in base-frame Y (`controller.compute_goal_pos`), active over `decision <= t < decision + 100` while `first_grasp is None`. Here `decision` is the first 5-control replan boundary where end-effector distance to the drawer handle collision center is $\le 0.10\text{ m}$ prior to two-finger grasp.
- **Attainable candidate actions:** At `decision`, apply discrete residual $u \in \{-1, 0, +1\}$ ($a = u \cdot B \in \{-0.04, 0, +0.04\}\text{ m}$, mapped into normalized `a[1]` and clipped to controller input bounds) over the same active approach window.
- **Admission and splits:**
  - **Gate 0 (`gate01`):** Test seeds `982101` and `982103`. Advance to the 16-scene panel only if `repeat_tape` reproduces nominal bit-identically and both scenes exhibit a two-way rescue contrast under both signs (uncompensated $u=0$ fails while matched inverse $u=-z$ succeeds).
  - **16-scene panel (`panel01`):** Screen seeds sequentially (`982100..982107`, then `983000..983040`), admitting the first 16 scenes whose unperturbed nominal rollout opens the drawer and reaches `decision` before `first_grasp`. Admission depends solely on nominal unperturbed behavior before inspecting any perturbed branch outcomes. All screened seeds and failure counts are retained in `events.jsonl` and `SUMMARY.json`.
  - **Train / held-out split:** First 8 admitted scenes = training split; next 8 admitted scenes = held-out evaluation split. Held-out scenes never enter normalization, classifier fitting, critic fitting, actor fitting, or checkpoint selection.
- **Verification per admitted scene:**
  1. `nominal` ($b=0, u=0$),
  2. `repeat_tape` (exact action-tape replay verifying identical `controls`, `success`, and `terminal_sha256`),
  3. `repeat_closed_loop` (fresh closed-loop rollout from identical canonical start and noise seed `scene + 700000`, measuring empty-history repeat variation),
  4. All 6 perturbed closed-loop branches `b{sign:+d}-u{u:+d}` for $z \in \{-1, +1\}$ and $u \in \{-1, 0, +1\}$, replaying the nominal prefix `0 <= t < decision` and verifying bit-identical `physics` and `decision_observation_sha256` at `t = decision`.
- **Stage 1 reference selectors (fitted on train split only, scored on held-out split):**
  - `oracle_per_condition` ($u = -z$),
  - `best_fixed_action` (single best $u \in \{-1, 0, +1\}$ on train),
  - `uncorrected_nominal` ($u = 0$),
  - `current_obs_ref` (ridge classifier on current `decision_state`),
  - `feedback_action_reward_ref` (linear/ridge classifier on `[u_sup, 2*r_sup - 1, u_sup*(2*r_sup - 1)]` under `own`, `matched_donor`, `feedback_masked`, and `counterfactual_reward`),
  - `feedback_drift_ref` (linear classifier on uncompensated $u_{\text{sup}}=0$ support proprioceptive drift `(public_delta_eef_y, commanded_mean_y)`).

## Stage 2: Small Causal History Reader & DIT Weighting Isolation

- **Two-episode trial structure (matching train & evaluation):**
  - Episode 1 (support) uses hidden $z$ and probe $u_1 \in \{-1, +1\}$ (or $u_1 = 0$ in the drift variant), producing native outcome $r_1 \in \{0, 1\}$.
  - Episode 2 (query) shares the exact same persistent $z$ and evaluates query action $u_2 \in \{-1, +1\}$ at `decision`, with native outcome $r_2 \in \{0, 1\}$.
  - Training pairs all ordered $(s_1, s_2) \in \text{train}^2 \times z \in \{-1, +1\} \times u_1 \in \{-1, +1\} \times u_2 \in \{-1, +1\}$ (512 unique queries; 100% pre-success decision queries; 0% post-success suffixes).
- **Causal token layout:**
  - Token 0 (support): `[norm(decision_state_1), u_1, r_1, done=1, start=1]` (plus separate drift-enabled variant `[norm(decision_state_1), u_1, norm(delta_eef_y), norm(cmd_y), r_1, done=1, start=1]` for $u_1 = 0$ support).
  - Token 1 (query): `[norm(decision_state_2), 0, 0, done=0, start=1]` (strictly causal: zeroed current action/feedback/outcome).
- **Model & conditions (3 independent fit seeds `0, 1, 2`, checkpoints at `64, 128, 256, 512, 1024` updates, `batch_size=32`, AdamW `lr=3e-4`):**
  1. `dpt`: Teacher-labeled ($u^* = -z$) imitation.
  2. `uniform_bc`: Unweighted ($w = 1$) imitation of uniform random $u_2 \in \{-1, +1\}$.
  3. `dit_tau1`: DIT advantage weighting $w = \text{clip}(\exp((Q - V) / 1.0), \max=100)$ from independently fitted causal $Q(h, o_q, u_2)$ and $V(h, o_q)$ critics (500 updates against $r_2$).
  4. `dit_tau01`: Same fitted $Q, V$ critics with temperature $\tau = 0.1$.
  5. `known_outcome_tau01`: Diagnostic oracle-return weighting $w = \text{clip}(\exp((r_2 - \bar{r}) / 0.1), \max=100)$.
  6. `memoryless`: Separately trained memoryless control (support token zeroed) trained on $u^* = -z$.
- **Evaluation & controls:**
  - Evaluate held-out cross-scene transfer (disjoint support/query pairs $s_1 \ne s_2$ sharing $z$) and same-scene retry ($s_1 = s_2$) under `own`, `matched_donor` (opposite $z$), `cleared`, `feedback_masked` ($r_1$ replaced by train mean $0.5$ or zeroed), and `counterfactual_reward` ($r_1 \leftarrow 1 - r_1$).
  - Report Attempt-2 native success and Cumulative (Attempt 1 or Attempt 2) native success, retaining all failures and unrescued scenes, with 10,000-draw scene-stratified paired bootstrap 95% intervals.
