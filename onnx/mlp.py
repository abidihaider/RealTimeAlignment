"""
An MLP-based network to predict misalignment from readout
Rewritten for ONNX export.
"""
import math
import torch
from torch import nn

class SineActivation(nn.Module):
    """
    Sine activate with a frequency factor
    """
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, data):
        return torch.sin(self.factor * data)


def sine_init(layer):
    """
    SIREN's way of initializing a linear layer
    """
    with torch.no_grad():
        if hasattr(layer, 'weight'):
            num_input = layer.weight.size(-1)
            vrange = math.sqrt(6 / num_input) / 30
            layer.weight.uniform_(-vrange, vrange)


def first_layer_sine_init(layer):
    """
    SIREN's way of initializing the first linear layer
    """
    with torch.no_grad():
        if hasattr(layer, 'weight'):
            num_input = layer.weight.size(-1)
            layer.weight.uniform_(-1 / num_input, 1 / num_input)


class ResLinear(nn.Module):
    """
    Linear layer with a layer normalization, a sine activation,
    and a residual connection.
    """
    def __init__(self,
                 in_features,
                 out_features,
                 rezero,
                 sine_factor):

        super().__init__()

        self.norm = nn.LayerNorm(in_features)
        self.linear = nn.Linear(in_features, out_features)
        self.activ = SineActivation(sine_factor)

        if rezero:
            self.re_alpha = nn.Parameter(torch.zeros((1, )))
        else:
            self.re_alpha = 1

    def forward(self, data):
        return data + self.re_alpha * self.activ(self.linear(self.norm(data)))


class SubsetSolver(nn.Module):
    def __init__(self,
                 in_features,
                 subset_size,
                 features,
                 rezero,
                 sine_factor):

        super().__init__()
        self.subset_size = subset_size
        in_f = in_features * subset_size

        layers = []
        for out_f in features:
            if in_f == out_f:
                layers += [ResLinear(in_features  = in_f,
                                     out_features = out_f,
                                     rezero       = rezero,
                                     sine_factor  = sine_factor)]
            else:
                layers += [nn.LayerNorm(in_f),
                           nn.Linear(in_f, out_f),
                           SineActivation(sine_factor)]
            in_f = out_f

        self.model = nn.Sequential(*layers)

    def __assemble(self, data):
        """
        assemble the subset by shifting the input
        """
        return torch.cat([torch.roll(data, i, dims=1)
                          for i in range(self.subset_size)],
                         dim=-1)

    def forward(self, data):
        subset = self.__assemble(data)
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
                 sine_factor,
                 use_sine_init,
                 rezero):

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
                                    sine_factor = sine_factor,
                                    rezero      = rezero)]
            in_f = config[-1]
        self.solvers = nn.ModuleList(layers)

        # output layer
        self.output = nn.Linear(in_f, out_features)

        if use_sine_init:
            self.initialize()

    def forward(self, data):
        """
        data of shape (batch_size, num_entries, in_features)
        """
        data = self.embed(data)
        for _, solver in enumerate(self.solvers):
            data = solver(data)
        return self.output(data.mean(dim=1))

    def initialize(self):
        self.apply(sine_init)
        self.embed[0].apply(first_layer_sine_init)
