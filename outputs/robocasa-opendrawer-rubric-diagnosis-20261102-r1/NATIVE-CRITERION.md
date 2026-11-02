# Native OpenDrawer correction

The retained pinned RoboCasa source, exported on allocated Linux CPU in
SOURCE-SURVEY.json and REPLAY-SOURCE.json, establishes:

- Drawer._create_cab sets slidejoint.range to [-self.size[1],0].
- Drawer.get_door_state normalizes -qpos by self.size[1]*0.55.
- normalize_joint_value returns (raw-min)/(max-min), without clipping.
- OpenDrawer._check_success requires the target drawer normalized value>=0.95.

The native threshold is therefore0.5225 of fixture depth, approximately half
of mechanical slide range. “95% of travel”, “almost the mechanical end stop”
and “fully extended” are inaccurate translations. A still extendable drawer
can already satisfy the task. This is a source correction, not evidence that
any particular false-positive video is truly successful.

Revision1.2 already included the52% number, but surrounded it with “full-open”
and “almost all” language. Its negative sample result does not justify simply
repeating that formulation. The next revision must remove the contradictory
physical requirement and diagnose target identity and endpoint separately.

A four-episode CPU replay will use retained executed actions. Qualification
requires all retained native scene metadata to match, all15 normalized public
16D state boundaries to agree within1e-4, and matching terminal truth. There is
no saved full MuJoCo state, so retain this limitation even if checks pass.
