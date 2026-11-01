# Owner CUA submission and proof contract

This is a launch prerequisite, not a submission script or authority expansion.
The execution owner must bind its concrete single-call CUA sequence to the
freshly observed form/list/detail controls before launching. No browser backend
HTTP requests, guessed task IDs or resource-template receipts are allowed.

Prepare the exact reviewed form/command and local dispatcher first. Read the
fresh `dispatcher-ready.json` and verify its PID is live, its expiry is still
in the future, and its run/code/task-name match the form. The expected task
name is `mwj_1031_robotwin_sac65_cpu_r2_<first-seven-pin-characters>`.

In **one bounded CUA invocation**, with sufficient context headroom:

1. Click the actual observed Submit control. Follow the actual created task
   through visible list/detail links, identified by the exact task name. Use
   fresh CUA AX state after actions. Poll only through documented CUA UI reads
   or UI reloads; do not invent an allocation ID or detail URL.
2. Observe the identified task currently **运行中**, its actual `ID: job-...`,
   actual `job-...-master-0` instance and Worker table `1 / 不使用加速芯片 /
   8 核 / 64 GiB`. Save the current detail screenshot as `actual-running.png`
   and full current AX state as `actual-running.txt` in the armed local inbox.
   The bridge parses these actual fields and rejects a Failed task or foreign
   name/instance. Screenshot hash alone does not prove its semantic contents.
3. In that same invocation, after both proof files are saved, write and atomically
   rename `proof-ready.json` from a fresh temporary file. Its complete schema is:

   ```json
   {
     "schema": "robotwin_cpu_actual_ui_proof_ready_v1",
     "run_id": "robotwin-sac65-cpu-compat-20261031-r2",
     "code_commit": "<full reviewed pin from dispatcher readiness>",
     "nonce": "<nonce from dispatcher readiness>",
     "dispatcher_pid": 123,
     "captured_at_unix": 0
   }
   ```

   Set the actual dispatcher PID and current capture Unix time in seconds.
   No CPU/GPU/RAM values or guessed allocation identity belong in this marker.
   Use the existing CUA filesystem binding for these artifact writes. Publish
   no marker on a UI mismatch, failed screenshot/write or expired readiness.

The bridge handles all later transfer/ticket work without another model turn.
It independently reads the actual `bootstrap-intent.json`, matches actual host,
code/run/output, and rejects late intent. Gate wait and every later phase still
share the original reduced 1,400-second deadline.

Keep the complete CUA operation under 60 seconds, including the UI watch and
proof writes. An observed queue/UI delay does not create a new wait budget.
If it cannot reach verified Running before the bounded capture cutoff, the
same owner-controlled sequence should stop/cancel its identified job through
fresh observed UI controls and retain evidence; if UI control fails, the gate
still refuses unproved work and times out. Reconcile actual scheduler cost and
do not resubmit automatically. A successful bounded proof capture is required
before saying the bootstrap timing failure has been corrected in execution.
