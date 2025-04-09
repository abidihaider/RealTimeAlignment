"""
An MLP-based network to predict misalignment from readout
"""
import math
import torch
from torch import nn

class SineActivation(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, data):
        return torch.sin(self.factor * data)


def sine_init(layer):
    with torch.no_grad():
        if hasattr(layer, 'weight'):
            num_input = layer.weight.size(-1)
            layer.weight.uniform_(-math.sqrt(6 / num_input) / 30, math.sqrt(6 / num_input) / 30)


def first_layer_sine_init(layer):
    with torch.no_grad():
        if hasattr(layer, 'weight'):
            num_input = layer.weight.size(-1)
            layer.weight.uniform_(-1 / num_input, 1 / num_input)


class SubsetSolver(nn.Module):
    def __init__(self,
                 in_features,
                 subset_size,
                 features,
                 sine_factor):
        super().__init__()
        self.subset_size = subset_size
        in_f = in_features * subset_size

        layers = []
        for out_f in features:
            layers += [nn.Linear(in_f, out_f),
                       SineActivation(sine_factor)]
            in_f = out_f

        self.model = nn.Sequential(*layers)

    def forward(self, data):

        num_entries = data.size(1)

        subset = []
        for _ in range(self.subset_size):
            indices = torch.randperm(num_entries)
            subset.append(data[:, indices])
        subset = torch.cat(subset, dim=-1)

        return self.model(subset)


class MLP(nn.Module):
    """
    MLP-based network with sine activate to predict misalignment from readout
    """
    def __init__(self,
                 in_features,
                 out_features,
                 embedding_features,
                 subset_config,
                 sine_factor = 30,
                 sine_init   = True):

        super().__init__()

        in_f = in_features

        # pointwise embedding
        layers = []
        for out_f in embedding_features:
            layers += [nn.Linear(in_f, out_f),
                       SineActivation(sine_factor)]
            in_f = out_f
        self.embed = nn.Sequential(*layers)

        # subset solvers
        layers = []
        for config in subset_config:
            layers += [SubsetSolver(in_features = in_f,
                                    subset_size = config[0],
                                    features    = config[1:],
                                    sine_factor = sine_factor)]
            in_f = config[-1]
        self.solver = nn.Sequential(*layers)

        # output layer
        self.output = nn.Linear(in_f, out_features)

        if sine_init:
            self.initialize()

    def forward(self, data):
        """
        data of shape (batch_size, num_entries, in_features)
        """
        data = self.embed(data)
        data = self.solver(data).mean(dim=1)
        return self.output(data)

    def initialize(self):
        self.apply(sine_init)
        self.embed[0].apply(first_layer_sine_init)
