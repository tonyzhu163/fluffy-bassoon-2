"""Fixed eight-way annotation of one complete64-episode PPO batch.

No learner or simulator dependency. A subset can be scored independently,
but only all64 official outputs can produce the training label map.
"""
from .rlinf_flora_ppo import flora_jobs, parse_outputs
from .rlinf_robocasa_ppo import PPO_TASKS, protocol, validate_protocol


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


def baseline_episode_shard(episodes, rank, proto, *, world=8):
    """Complete native50 bank, split over four or eight calibration scorers."""
    validate_protocol(proto, training=True)
    if (proto['baseline_count'] != 50 or len(episodes) != 50
            or type(world) is not int or world not in (4,8)
            or type(rank) is not int or not 0 <= rank < world):
        raise ValueError('Exactly fifty native baseline cases and four or eight scorer ranks required')
    flora_jobs(episodes, proto)
    return episodes[rank::world]


def gather_baseline_outputs(episodes, shards, proto, *, world=8):
    """No fifty-case quality result from a partial or misrouted annotation bank."""
    if type(world) is not int or world not in (4,8) or set(shards) != set(range(world)):
        raise ValueError('All bound baseline shards are required')
    by_id = {}
    for rank in range(world):
        subset, rows = baseline_episode_shard(episodes, rank, proto, world=world), shards[rank]
        parse_outputs(subset, rows, proto)
        for row in rows:
            if row['id'] in by_id:
                raise ValueError('A native case cannot occur in two baseline shards')
            by_id[row['id']] = row
    ordered = [by_id[episode['id']] for episode in episodes]
    return ordered, parse_outputs(episodes, ordered, proto)


def joint_episode_shard(episodes, rank, episode_protocols):
    """One complete interleaved80 bank, ten cases per fixed scorer rank."""
    if (type(rank) is not int or not 0 <= rank < 8 or len(episodes) != 80
            or not isinstance(episode_protocols, dict)
            or set(episode_protocols) != {episode['id'] for episode in episodes}
            or any(not isinstance(value,dict) for value in episode_protocols.values())):
        raise ValueError('Exact eighty-case joint bank and per-episode native protocol map required')
    expected=list(PPO_TASKS)*16
    if ([episode_protocols[episode['id']].get('task') for episode in episodes] != expected
            or any(episode_protocols[episode['id']] != protocol(task,baseline_count=50)
                   for episode,task in zip(episodes,expected))):
        raise ValueError('Sixteen interleaved cases of each of the five declared PPO tasks required')
    flora_jobs(episodes,episode_protocols=episode_protocols)
    subset=episodes[rank*10:(rank+1)*10]
    return subset,{episode['id']:episode_protocols[episode['id']] for episode in subset}


def gather_joint_terminal_outputs(episodes, shards, episode_protocols):
    """All eighty native-task grades are required before any joint label map."""
    if set(shards) != set(range(8)):
        raise ValueError('All eight joint terminal-output shards are required')
    by_id={}
    for rank in range(8):
        subset,protocols=joint_episode_shard(episodes,rank,episode_protocols)
        rows=shards[rank]
        parse_outputs(subset,rows,episode_protocols=protocols)
        for row in rows:
            if row['id'] in by_id:raise ValueError('A joint terminal case cannot appear in two shards')
            by_id[row['id']]=row
    ordered=[by_id[episode['id']] for episode in episodes]
    return ordered,parse_outputs(episodes,ordered,episode_protocols=episode_protocols)
