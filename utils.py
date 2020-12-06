from collections import OrderedDict

def unwrap_state_dict(state_dict):
    """ cleans up the keys of a model state dictionary

    Methods and classes such as convert_sync_batchnorm or DataParallel wrap their
    respective module, which also alters the state dictionary's keys.
    This functions removes the leading module. string of the keys.

    """

    unwrap = lambda x: ".".join(x.split(".")[1:])

    # PyTorch state dicionaries are just regulard ordered dictionaries
    return OrderedDict((unwrap(k), v) for k, v in state_dict.items())
