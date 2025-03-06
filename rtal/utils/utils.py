"""
"""
import copy

import torch
from torch import nn

def extract_name_kwargs(obj):
    """
    """
    if isinstance(obj, dict):
        obj    = copy.copy(obj)
        name   = obj.pop('name')
        kwargs = obj
    else:
        name   = obj
        kwargs = {}

    return (name, kwargs)


def get_norm_layer3d(norm, features):
    name, kwargs = extract_name_kwargs(norm)

    if name is None:
        return nn.Identity(**kwargs)

    if name == 'layer':
        return nn.LayerNorm((features,), **kwargs)

    if name == 'batch':
        return nn.BatchNorm3d(features, **kwargs)

    if name == 'instance':
        return nn.InstanceNorm3d(features, **kwargs)

    raise ValueError(f"Unknown Layer: {name}")


def get_norm_layer2d(norm, features):
    name, kwargs = extract_name_kwargs(norm)

    if name is None:
        return nn.Identity(**kwargs)

    if name == 'layer':
        return nn.LayerNorm((features,), **kwargs)

    if name == 'batch':
        return nn.BatchNorm2d(features, **kwargs)

    if name == 'instance':
        return nn.InstanceNorm2d(features, **kwargs)

    raise ValueError(f"Unknown Layer: {name}")


def get_activ_layer(activ):
    name, kwargs = extract_name_kwargs(activ)

    if (name is None) or (name == 'linear'):
        return nn.Identity()

    if name == 'gelu':
        return nn.GELU(**kwargs)

    if name == 'relu':
        return nn.ReLU(**kwargs)

    if name == 'leakyrelu':
        return nn.LeakyReLU(**kwargs)

    if name == 'tanh':
        return nn.Tanh()

    if name == 'sigmoid':
        return nn.Sigmoid()

    raise ValueError(f"Unknown activation: {name}")


def get_jit_input(tensor, batch_size, device):
    """
    Get a dummy input for jit tracing
    """
    dummy = torch.ones_like(tensor)
    shape = (batch_size, ) + (1, ) * tensor.dim()
    dummy = dummy.repeat(shape)
    return dummy.to(device)


def get_lr(optim):
    """
    Get the current learning rate
    """
    for param_group in optim.param_groups:
        return param_group['lr']


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_data_root():
    """
    Get root to data as an environment variable
    """
    # Get the value of DATAROOT
    data_root = os.getenv('DATAROOT')

    assert data_root is not None, \
        ('DATAROOT is not set!'
         'please run export DATAROOT=/path/to/your/data '
         'to set the environment variable.')

    print(f'\nDATAROOT is set to: {data_root}\n')

    return data_root


def flatten_dict(mydict):
    """
    flatten hierarchical dictionary
    """
    result = {}
    for key, val in mydict.items():
        if isinstance(val, dict):
            flattened = flatten(val)
            for sub_key, sub_val in flattened.items():
                result[f'{key}-{sub_key}'] = sub_val
        else:
            result[key] = val
    return result

