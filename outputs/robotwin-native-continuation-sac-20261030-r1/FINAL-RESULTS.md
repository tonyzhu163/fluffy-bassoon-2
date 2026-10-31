# RoboTwin SAC final result

Baige `job-gv77bzhyeroo` completed successfully at08:04:24Shanghai Oct31,2026, exit0. All100source rounds/600completeK4 trials/2400source episodes and all10/25/50/100evaluation stages are complete. Source collection used1,003,790actions and638,500updates EACH actor/critic/temperature optimizer (1,915,500total). Evaluation target-clone updates add no source SGD. This is feature-conditioned DSRL-style SAC, not an authors’ PixelSAC reproduction.

Stage100source validation was2/24. On the two development targets, successes per8 were:

| Condition | adjust_bottle | move_stapler_pad | Target actions | Additional joint target updates |
| --- | ---: | ---: | ---: | ---: |
| Frozen SAC |6|2|4400|0|
| Online SAC, fresh target replay |5|0|5120|3320|

Both conditions use stochastic action sampling from the same independently restored source actor/private action RNG. Their first noise, motor action, next observation and executed duration match on both tasks. Online20SGDafterchunk1 changes later behavior, so wholeE1 equality is not required. All56stage100source-validation/target resets exactly match PPO/reset controls’ actual seeds, initial features and raw observations. Episode1/2/4/8 cumulative outcomes and full8episode traces are in `stage100-targets.csv`. This one-seed, two-development-task, one-K8-per-condition result establishes no adaptation or generalization advantage.

Independent final audit passed final coordinator/four target recovery SHA/ZIPCRCs, all4completed stage manifests with source RNG restored,61clean child exits, frozen motor and closed canonical pending0/uncertain0. All640native masks from56episodes independently reconcile to19,700physical actions:10,180source-validation and9,520target. The complete SAC child ledger contains1,088,029actions:1,003,790source,41,032source-validation and43,207targets, within its unchanged1,194,400cap. Full replay/Adam/samplers/private+globalRNG/source ancestry and target recoveries remain protected. RGB payloads were retained but not all rehashed in this final integrity audit.

Final coordinator SHA`f551e3f28d14702556392e67dafdf62f094c824f9069511219f8dc5af4db9fd8`; fixed stage100source snapshot SHA`59588f07be18d4a1339af0c7536c62ef735a713f3c933d7e372fb3f4f4fa7e90`. The final owner-issued export-provenance manifest binds the actual0061validation root through its unchanged execution migration to0060; it authorizes no source resume. The62-root final ancestry is retained in the trained-continuation run’s `handoffs/`. Full source100 byte projection/export/teacher competence/student fits remain separate proposals.

Actual scheduler Running23:07:14Oct30→Success08:04:24Oct31 was32,230GPU-seconds; pod32,224 and supervisor32,209.948seconds are separate. The known terminal GPU subtotal is now125,014seconds; full cumulative historical total remains unknown because reported630optimizationseconds are unverified. No original-goal job remains active. Exact evidence and raw small receipts are under `checks/` and `receipts/`.
