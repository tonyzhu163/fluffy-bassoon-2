# LIBERO40 source versus LIBERO90 goal audit

Pinned LIBERO source: `87edbd1b823e464bc66557d213cc45e9f88aa543`. All 40 source and 90 target candidates use task order 0.

This audit assigns no training or final-test split. LIBERO90 remains target-only for the new arm; the prior split is not imported.

| Quantity | Count |
| --- | ---: |
| source40 | 40 |
| target90 | 90 |
| source_target_matching_pairs | 72 |
| target90_literal_goal_binding_overlap | 19 |
| target90_typed_alpha_goal_overlap | 22 |
| target90_instruction_connected_only_overlap | 3 |
| target90_source40_family_overlap | 25 |
| target90_without_source40_family_overlap | 65 |
| target90_new_goal_without_reserved_development_family | 63 |

There are 19 literal typed-goal matches. Instance renaming adds IDs12/14/15, giving 22 direct typed-goal overlaps. The conservative instruction-connected rule additionally flags IDs25/29/31: their white-cabinet goals differ in fixture type from source wooden-cabinet goals, but identical instructions connect them through LIBERO90 siblings. These three are candidate family overlaps, not identical typed goals.

Families join equal normalized full goals with typed bindings, equal full goals after type-preserving instance renaming, or equal normalized full instructions. The typed signature preserves predicates, complete conjunctions, object types, region suffixes, and entity identity. The instruction-connected family can span differing typed goals and is reported separately. A shared primitive alone does not join families.

Prior development IDs: `libero_90:01`, `libero_90:02`, `libero_90:47`, `libero_90:57`. Related family members are flagged separately in the JSON; this document does not select a final evaluation set.

| Target90 | Direct typed-goal match | Instruction-connected only | Source40 family members | Prior-development family members |
| --- | --- | --- | --- | --- |
| libero_90:00 | no | no | none | none |
| libero_90:01 | no | no | none | libero_90:01 |
| libero_90:02 | yes | no | libero_goal:03 | libero_90:02 |
| libero_90:03 | no | no | none | none |
| libero_90:04 | no | no | none | none |
| libero_90:05 | no | no | none | none |
| libero_90:06 | no | no | none | none |
| libero_90:07 | no | no | none | none |
| libero_90:08 | no | no | none | none |
| libero_90:09 | yes | no | libero_goal:08, libero_spatial:00, libero_spatial:01, libero_spatial:02, libero_spatial:03, libero_spatial:04, libero_spatial:05, libero_spatial:06, libero_spatial:07, libero_spatial:08, libero_spatial:09 | none |
| libero_90:10 | yes | no | libero_goal:04 | none |
| libero_90:11 | no | no | none | none |
| libero_90:12 | yes | no | libero_goal:08, libero_spatial:00, libero_spatial:01, libero_spatial:02, libero_spatial:03, libero_spatial:04, libero_spatial:05, libero_spatial:06, libero_spatial:07, libero_spatial:08, libero_spatial:09 | none |
| libero_90:13 | yes | no | libero_goal:08, libero_spatial:00, libero_spatial:01, libero_spatial:02, libero_spatial:03, libero_spatial:04, libero_spatial:05, libero_spatial:06, libero_spatial:07, libero_spatial:08, libero_spatial:09 | none |
| libero_90:14 | yes | no | libero_goal:08, libero_spatial:00, libero_spatial:01, libero_spatial:02, libero_spatial:03, libero_spatial:04, libero_spatial:05, libero_spatial:06, libero_spatial:07, libero_spatial:08, libero_spatial:09 | none |
| libero_90:15 | yes | no | libero_goal:04 | none |
| libero_90:16 | no | no | none | none |
| libero_90:17 | no | no | none | none |
| libero_90:18 | no | no | none | none |
| libero_90:19 | no | no | none | none |
| libero_90:20 | yes | no | libero_goal:07 | none |
| libero_90:21 | no | no | none | none |
| libero_90:22 | no | no | none | none |
| libero_90:23 | no | no | none | none |
| libero_90:24 | no | no | none | none |
| libero_90:25 | no | yes | libero_goal:04 | none |
| libero_90:26 | no | no | none | none |
| libero_90:27 | yes | no | libero_goal:09 | none |
| libero_90:28 | no | no | none | none |
| libero_90:29 | no | yes | libero_goal:03 | libero_90:02 |
| libero_90:30 | yes | no | libero_goal:08, libero_spatial:00, libero_spatial:01, libero_spatial:02, libero_spatial:03, libero_spatial:04, libero_spatial:05, libero_spatial:06, libero_spatial:07, libero_spatial:08, libero_spatial:09 | none |
| libero_90:31 | no | yes | libero_goal:04 | none |
| libero_90:32 | no | no | none | none |
| libero_90:33 | no | no | none | none |
| libero_90:34 | no | no | none | none |
| libero_90:35 | no | no | none | none |
| libero_90:36 | no | no | none | none |
| libero_90:37 | no | no | none | none |
| libero_90:38 | yes | no | libero_10:02 | none |
| libero_90:39 | no | no | none | none |
| libero_90:40 | no | no | none | none |
| libero_90:41 | no | no | none | none |
| libero_90:42 | no | no | none | none |
| libero_90:43 | no | no | none | none |
| libero_90:44 | yes | no | libero_goal:07 | none |
| libero_90:45 | no | no | none | none |
| libero_90:46 | yes | no | libero_object:00 | none |
| libero_90:47 | yes | no | libero_object:01 | libero_90:47 |
| libero_90:48 | yes | no | libero_object:04 | none |
| libero_90:49 | yes | no | libero_object:05 | none |
| libero_90:50 | yes | no | libero_object:00 | none |
| libero_90:51 | yes | no | libero_object:06 | none |
| libero_90:52 | yes | no | libero_object:07 | none |
| libero_90:53 | yes | no | libero_object:09 | none |
| libero_90:54 | yes | no | libero_object:05 | none |
| libero_90:55 | no | no | none | none |
| libero_90:56 | no | no | none | none |
| libero_90:57 | no | no | none | libero_90:57 |
| libero_90:58 | no | no | none | none |
| libero_90:59 | no | no | none | none |
| libero_90:60 | no | no | none | none |
| libero_90:61 | no | no | none | none |
| libero_90:62 | no | no | none | none |
| libero_90:63 | no | no | none | none |
| libero_90:64 | no | no | none | none |
| libero_90:65 | no | no | none | none |
| libero_90:66 | no | no | none | none |
| libero_90:67 | no | no | none | none |
| libero_90:68 | no | no | none | none |
| libero_90:69 | no | no | none | none |
| libero_90:70 | no | no | none | none |
| libero_90:71 | no | no | none | none |
| libero_90:72 | no | no | none | none |
| libero_90:73 | no | no | none | none |
| libero_90:74 | no | no | none | none |
| libero_90:75 | no | no | none | none |
| libero_90:76 | no | no | none | none |
| libero_90:77 | yes | no | libero_10:05 | none |
| libero_90:78 | no | no | none | none |
| libero_90:79 | no | no | none | none |
| libero_90:80 | no | no | none | none |
| libero_90:81 | no | no | none | none |
| libero_90:82 | no | no | none | none |
| libero_90:83 | no | no | none | none |
| libero_90:84 | no | no | none | none |
| libero_90:85 | no | no | none | none |
| libero_90:86 | no | no | none | none |
| libero_90:87 | no | no | none | none |
| libero_90:88 | no | no | none | none |
| libero_90:89 | no | no | none | none |

## Reproduction and provenance

The snapshot contains static BDDL/API/task-map text plus SHA-256, byte counts, and pinned Git blob identities for every source file and initial-state file. It contains no environment observations or executed trajectories. The copied original parser is imported without calling its split-building main function.

```sh
python build_audit.py --snapshot source_snapshot.json --parser original_build_split.py --output .
```

Source inventory SHA-256: `5a38925fd484a28b6aac775b437d7a5069d8303ad488f58e4382b9ce8ad64cef`.

Parser SHA-256: `16a6bbf11b0f8fd7141b9da6bcc663df16381e331ed0f306ac3568ebbe7e638f`.

Audit content SHA-256: `f55decd3deb9102969ca38dc654fe46720009aaf40554f5dcc6ce704279f5a51`.

Static goal families are conservative candidates: no language synonyms or object types are collapsed. Goal novelty does not establish novelty of scenes, primitives, or the pretrained motor's data.
