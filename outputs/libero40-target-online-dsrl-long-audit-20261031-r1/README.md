# Longer ordinary DSRL development cases: terminal

Both fixed240-episode continuations finished exit0 atcee1ea0. Task0:72/240
training successes,8,154chunks/153,480updates,87,576cumulativecaseactions.
Task40:0/240,9,600chunks/182,400updates,102,520actions. Every query failed
except task0's80-episode checkpoint (249primitives); its160and240queries
failed. One query/checkpoint does not estimate a success probability.

Task0 job-pytyo9urlxrj charged5,065GPU-s (UI5,012); task40job-5rqftpqn4wxi
charged5,406 (UI5,351). Newactions78,546+93,490=172,036. Count the original
six-case parent's35,558actions/1,851GPU-s once:207,594uniqueactions and
12,322GPU-s for onset pluscontinuations. Copiedprefixes are notnewdata.

All250training/querytracehashes percase and final240query/fullrecoveries
were rehashed read-only. receipt.json preserves exact immutable hashes,
queryweights/replay guards, episode budgets and counters. No additional
model computation or rollout occurred during this audit.

Both cases are outcome-selected development diagnostics, not a confirmatory
full-suite comparison or a faithful original-pi0 paper reproduction. Native
OpenPIpi0.5H10/ex10 is a declared motor/interface change. The released recipe
stores gamma**query_freq; our actual-duration gamma agrees for full chunks,
while handling shorter chunks explicitly. No extra gamma is applied.

Next proposed check freezes80/160/240 and native on the same physical task0
state with eight new paired stochastic seeds. It remains unlaunched and
requires its own bounded reservation; it will not select a best checkpoint
or turn this developmentcase into a heldout performance claim.
