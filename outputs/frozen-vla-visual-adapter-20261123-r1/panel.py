TASKS = (("libero_goal", 3), ("libero_10", 8), ("libero_goal", 6))
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")


def case(arm, suite, task, start):
    return dict(arm=arm, suite=suite, task_id=task, init_state_id=start, variant="BASE",
                seed=41800000 + 10000 * SUITES.index(suite) + 100 * task + start)
