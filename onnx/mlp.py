"""
An MLP-based network to predict misalignment from readout
"""
import torch
from torch import nn

from rtal.utils.utils import get_activ_layer


class CloudNorm(nn.Module):
    """
    Apply batch or instance normalization to data in 
    the shape (batch_size, num_points, num_features).
    In the forward, we switch the last two dimension,
    apply the norm, and switch back.
    """
    def __init__(self, norm, num_features):
        super().__init__()
        if norm == 'batch':
            self.norm = nn.BatchNorm1d(num_features)
        elif norm == 'instance':
            self.norm = nn.InstanceNorm1d(num_features)
        else:
            raise KeyError(f'Unknown norm type {norm}!')

    def forward(self, tensor):
        """
        tensor: (Batch_size, num_points, num_features)
        """
        tensor = tensor.permute(0, 2, 1)
        tensor = self.norm(tensor)
        tensor = tensor.permute(0, 2, 1)
        return tensor


class ResLinear(nn.Module):
    """
    Linear layer with a layer normalization, a sine activation,
    and a residual connection.
    """
    def __init__(self,
                 in_features,
                 out_features,
                 rezero,
                 activ,
                 norm):

        super().__init__()

        if norm is None:
            self.norm_layer = nn.Identity()
        else:
            self.norm_layer = CloudNorm(norm, in_features)

        self.linear = nn.Linear(in_features, out_features)
        self.activ = get_activ_layer(activ)

        if rezero:
            self.re_alpha = nn.Parameter(torch.zeros((1, )))
        else:
            self.re_alpha = 1

    def forward(self, data):
        residual = data
        data = self.activ(self.linear(self.norm_layer(data)))
        return residual + self.re_alpha * data


class SubsetSolver(nn.Module):
    def __init__(self,
                 in_features,
                 subset_size,
                 features,
                 rezero,
                 activ,
                 norm):

        super().__init__()
        self.subset_size = subset_size

        in_f = in_features * subset_size

        layers = []
        for out_f in features:
            if in_f == out_f:
                layers += [ResLinear(in_features  = in_f,
                                     out_features = out_f,
                                     rezero       = rezero,
                                     activ        = activ,
                                     norm         = norm)]
            else:
                if norm is None:
                    norm_layer = nn.Identity()
                else:
                    norm_layer = CloudNorm(norm, in_f)
                layers += [norm_layer,
                           nn.Linear(in_f, out_f),
                           get_activ_layer(activ)]
            in_f = out_f

        self.model = nn.Sequential(*layers)

        self.assemble = self._assemble_shift

    def _assemble_shift(self, data):
        """
        assemble the subset by shifting the input
        """
        return torch.cat([torch.roll(data, i, dims=1)
                          for i in range(self.subset_size)],
                         dim=-1)

    def forward(self, data):
        subset = self.assemble(data)
        return self.model(subset)

    def inference(self, data):
        """
        Have the option to turn off random permutation
        for inference.
        """
        subset = self._assemble_shift(data)
        return self.model(subset)


class MLP(nn.Module):
    """
    MLP-based network with subset solvers
    """
    def __init__(self,
                 in_features,
                 out_features,
                 embedding_features,
                 subset_config,
                 rezero,
                 activ,
                 norm):

        super().__init__()
        in_f = in_features

        # pointwise embedding        
        if norm is None:
            norm_layer = nn.Identity()
        else:
            norm_layer = CloudNorm(norm, in_f)
    
        layers = []
        for out_f in embedding_features:
            layers += [norm_layer,
                       nn.Linear(in_f, out_f),
                       get_activ_layer(activ)]
            in_f = out_f

        self.embed = nn.Sequential(*layers)

        # subset solvers
        layers = []
        for config in subset_config:
            layers += [SubsetSolver(in_features = in_f,
                                    subset_size = config[0],
                                    features    = config[1:],
                                    rezero      = rezero,
                                    activ       = activ,
                                    norm        = norm)]
            in_f = config[-1]
        self.solvers = nn.ModuleList(layers)

        # output layer
        self.output = nn.Linear(in_f, out_features)

    def forward(self, data):
        """
        data of shape (batch_size, num_entries, in_features)
        """
        data = self.embed(data)
        for _, solver in enumerate(self.solvers):
            data = solver(data)
        return self.output(data.mean(dim=1))

    def inference(self, data):
        """
        Provide the option to turn of random permutation for inference.
        Input:
            data of shape (batch_size, num_entries, in_features)
        """
        data = self.embed(data)
        for _, solver in enumerate(self.solvers):
            data = solver.inference(data)
        return self.output(data.mean(dim=1))