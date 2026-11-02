# Six retained trajectories now have checked first-success clocks

The three newly replayed success controls128/72/264 all match their15 saved
normalized public16D state boundaries exactly (maximum error0), match terminal
success, and match all native scene metadata after sorting only unordered
`groups_containing_sampled_obj` tags. The earlier excluded128 output is kept
unchanged. No full MuJoCo initial state was saved; retain that limitation.

| Case | Native first success control | Seconds at20Hz | Terminal |
| --- | ---: | ---: | --- |
|112 (prior checked replay)|Never|—|Failure; opens drawer below target|
|64 (prior checked replay)|542|27.1|Failure; success later undone|
|128|384|19.2|Success|
|72|208|10.4|Success|
|264|216|10.8|Success|
|144 (prior checked replay)|222|11.1|Success|

New CPU work physically replays2250 retained controls; the prior qualified
replays contribute2250. No new motor calls, newly sampled robot actions or
GPU model inference. Three new action extracts and their checkpoint hashes
are in SOURCE-SURVEY.json. Exact replay JSON files were copied from Linux with
all hashes verified in CPU-COPIED-HASHES.json. CPU source pin c8cd3134; GPU
prefix selection/scoring will use a separate subsequently reviewed pin.

The prefix contract was refined before any inference to include one additional
50-control post-crossing probe per successful trajectory. This avoids treating
a crossing just before the first probe as proof that a detector cannot respond
within a normal chunk delay. At most35 prefixes/70 grades on the same six
retained videos, unchanged900GPU-s ceiling. No full-bank replay.
