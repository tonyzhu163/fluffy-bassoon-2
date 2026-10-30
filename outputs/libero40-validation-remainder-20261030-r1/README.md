# Reconciled frozen source-validation remainder

Parent validation is terminal exit0:32K4trials/128episodes;full63/64,episode-reset61/64,eachfirsttwoattempts16/16. All32trialtraceSHAs and fullrecovery independently verified;weightsunchanged,zeroSGD. Used22753actions/1203conservativeGPU-seconds; UI terminalduration1150s. Parentindex and clock hashes are bound in sac-alias.json.

Original56k physical envelope is preserved. The per-task SAC teacher competence child aliases20000unspentactions, so parent+child ceiling42753<=56000;13247physicalactions remain unassigned within that envelope. No duplicate canonical reservation.

Original4h compute hold is reconciled to1253s(parent1203+50margin),3600sSACdiagnostic,7200sfrozenPPOtargetwave;2347sreleased. Combined12053s<=14400s. PPOtargetwave has a separate140k target-only action ledger; it never enters sourcepretraining or consumes the56k sourcevalidation physical envelope. Overall32h unchanged.

Unexpected studentdeficit and sourceceiling/failurecoverage analysis recorded in docs/frozen-vla-steering-summary.md before additional collection. These diagnostic/evaluation runs resolve the existing baseline gaps.
