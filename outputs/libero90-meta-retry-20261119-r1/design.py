"""Generate the nominal LIBERO-90 task split and paired reset manifest."""
import json
import random
from pathlib import Path

SEED = 2026111907


def cases():
    tasks = list(range(90))
    random.Random(SEED).shuffle(tasks)
    partitions = [
        ("fitting", tasks[:48], 20),
        ("development", tasks[48:60], 12),
        ("test", tasks[60:], 24),
    ]
    for split, partition, repeats in partitions:
        for task in sorted(partition):
            poses = list(range(1, 50))
            random.Random(SEED + task + 1).shuffle(poses)
            for repeat in range(repeats):
                yield dict(split=split, task_id=task, repeat=repeat,
                           support_id=poses[2 * repeat], query_id=poses[2 * repeat + 1],
                           seed=27110100 + 100 * task + 2 * repeat)


if __name__ == "__main__":
    out = Path(__file__).with_name("cases.json")
    rows = list(cases())
    assert len(rows) == 1824
    out.write_text("[\n" + ",\n".join(json.dumps(row) for row in rows) + "\n]\n")
