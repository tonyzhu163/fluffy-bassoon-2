"""Measure the effect of allowing more control targets and changing the prompt."""
import numpy as np


class PhoneTrace:
    def __init__(self, task):
        self.start_z = float(task.phone.get_pose().p[2])
        self.max_height = 0.
        self.closest = float('inf')
        self.closest_open = None
        self.ever_in_position = False
        self.at_400 = None

    def observe(self, task):
        self.max_height = max(self.max_height, float(task.phone.get_pose().p[2]) - self.start_z)
        phone = np.asarray(task.phone.get_functional_point(0))[:3]
        stand = np.asarray(task.stand.get_functional_point(0))[:3]
        delta = np.abs(phone - stand)
        ratio = float(np.max(delta / np.array([.045, .04, .04])))
        if ratio < self.closest:
            self.closest = ratio
            self.closest_open = [bool(task.is_left_gripper_open()), bool(task.is_right_gripper_open())]
        self.ever_in_position |= ratio < 1
        if task.take_action_cnt == 400:
            self.at_400 = self.summary()

    def summary(self):
        return dict(max_phone_lift_m=self.max_height, min_position_error_over_tolerance=self.closest,
                    grippers_open_at_closest=self.closest_open, inside_position_tolerance=self.ever_in_position)


def diagnose(out, frozen_cases, native, proc, scratch, report, count):
    from envs.place_phone_stand import place_phone_stand
    from envs.utils import UnStableError
    from run import environment_args, episode, save
    config = environment_args(native, 'demo_clean')
    config.update(policy_name='sidney_lerobot', ckpt_setting='sidney_release')
    modes = ('scene_frozen', 'short_frozen', 'short_sequential')
    report.update(modes=list(modes), settings={m: [] for m in modes}, target_limit=1200, internal_cutoff=400)
    for index, case in enumerate(frozen_cases[:count]):
        for mode in modes[index % 3:] + modes[:index % 3]:
            sequential = mode == 'short_sequential'
            row = dict(case, seed=1000+index) if sequential else case
            short = mode != 'scene_frozen'
            directory = out/f"{mode}-{index:03d}-seed-{row['seed']}"
            task = place_phone_stand()
            task.suc = task.test_num = 0
            try:
                value = episode(task, config, native, row, proc, scratch, directory,
                                horizon=1200, short_prompt=short, trace_class=PhoneTrace)
            except UnStableError as exc:
                if not sequential or directory.exists():
                    raise
                # Release default sequential seeds have no expert feasibility screening.
                value = dict(index=index, seed=row['seed'], reset_error=str(exc), success=None)
                task.close_env()
            report['settings'][mode].append(value)
            save(out/'RESULT.json', report)
            del task
