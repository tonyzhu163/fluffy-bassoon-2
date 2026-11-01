"""Fixed eight-way annotation of one complete64-episode PPO batch.

No learner or simulator dependency. A subset can be scored independently,
but only all64 official outputs can produce the training label map.
"""
from .rlinf_flora_ppo import flora_jobs, parse_outputs
from .rlinf_robocasa_ppo import validate_protocol


def shard_episode_bank(episodes, rank, world, proto):
    validate_protocol(proto, training=True)
    if (type(rank) is not int or type(world) is not int or world != 8
            or not 0 <= rank < world or len(episodes) != 64):
        raise ValueError("One complete64-episode bank and eight fixed scorer ranks required")
    flora_jobs(episodes, proto)
    return episodes[rank * 8:(rank + 1) * 8]


def gather_terminal_outputs(episodes, shards, proto):
    """Keep official grades/probabilities unchanged; fail before any partial label map."""
    if set(shards) != set(range(8)):
        raise ValueError("All eight terminal-output shards are required before GAE")
    by_id = {}
    for rank in range(8):
        subset = shard_episode_bank(episodes, rank, 8, proto)
        rows = shards[rank]
        parse_outputs(subset, rows, proto)
        for row in rows:
            if row['id'] in by_id:
                raise ValueError("A terminal label cannot appear in two shards")
            by_id[row['id']] = row
    ordered = [by_id[episode['id']] for episode in episodes]
    return ordered, parse_outputs(episodes, ordered, proto)


def baseline_episode_shard(episodes, rank, proto):
    """Unequal7/7/6/6/6/6/6/6 partitions of a complete native50 bank."""
    validate_protocol(proto, training=True)
    if (proto['baseline_count'] != 50 or len(episodes) != 50
            or type(rank) is not int or not 0 <= rank < 8):
        raise ValueError('Exactly fifty native baseline cases and eight scorer ranks required')
    flora_jobs(episodes, proto)
    return episodes[rank::8]


def gather_baseline_outputs(episodes, shards, proto):
    """No fifty-case quality result from a partial or misrouted annotation bank."""
    if set(shards) != set(range(8)):
        raise ValueError('All eight baseline shards are required')
    by_id = {}
    for rank in range(8):
        subset, rows = baseline_episode_shard(episodes, rank, proto), shards[rank]
        parse_outputs(subset, rows, proto)
        for row in rows:
            if row['id'] in by_id:
                raise ValueError('A native case cannot occur in two baseline shards')
            by_id[row['id']] = row
    ordered = [by_id[episode['id']] for episode in episodes]
    return ordered, parse_outputs(episodes, ordered, proto)
