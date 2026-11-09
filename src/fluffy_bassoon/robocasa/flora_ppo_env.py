"""Keep simulator truth outside the learned-reward actor."""


def strip_training_truth(infos):
    """Only transition validity survives; simulator labels never reach F's actor."""
    return {key:value for key,value in infos.items() if key=='valid_action_mask'}
