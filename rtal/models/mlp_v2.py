"""
An MLP-based network to predict misalignment from readout
"""
import torch
from torch import nn

from rtal.utils.utils import get_activ_layer


class CloudBatchNorm(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        # self.bn = nn.BatchNorm1d(num_features)
        self.norm = nn.InstanceNorm1d(num_features)

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
                 batchnorm):

        super().__init__()

        if batchnorm:
            self.norm = CloudBatchNorm(in_features)
        else:
            self.norm = nn.Identity()

        self.linear = nn.Linear(in_features, out_features)
        self.activ = get_activ_layer(activ)

        if rezero:
            self.re_alpha = nn.Parameter(torch.zeros((1, )))
        else:
            self.re_alpha = 1

    def forward(self, data):
        residual = data
        data = self.activ(self.linear(self.norm(data)))
        return residual + self.re_alpha * data


class SubsetSolver(nn.Module):
    def __init__(self,
                 in_features,
                 subset_size,
                 features,
                 rezero,
                 activ,
                 batchnorm,
                 randperm):

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
                                     batchnorm    = batchnorm)]
            else:
                if batchnorm:
                    norm = CloudBatchNorm(in_f)
                else:
                    norm = nn.Identity()
                layers += [norm,
                           nn.Linear(in_f, out_f),
                           get_activ_layer(activ)]
            in_f = out_f

        self.model = nn.Sequential(*layers)

        # specify how to assemble the equation subsets
        if randperm:
            self.assemble = self._assemble_rand
        else:
            self.assemble = self._assemble_shift

    def _assemble_rand(self, data):
        """
        assemble the subset with randomly permutate the input
        """
        num_entries = data.size(1)

        subset = []
        for _ in range(self.subset_size):
            indices = torch.randperm(num_entries)
            subset.append(data[:, indices])

        return torch.cat(subset, dim=-1)

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

    def inference(self, data, randperm):
        """
        Have the option to turn off random permutation
        for inference.
        """
        if randperm:
            subset = self._assemble_rand(data)
        else:
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
                 batchnorm,
                 randperm):

        super().__init__()

        in_f = in_features

        # pointwise embedding
        layers = []
        for out_f in embedding_features:
            if batchnorm:
                norm = CloudBatchNorm(in_f)
            else:
                norm = nn.Identity()
            layers += [norm,
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
                                    batchnorm   = batchnorm,
                                    randperm    = randperm)]
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

    def inference(self, data, randperm):
        """
        Provide the option to turn of random permutation for inference.
        Input:
            data of shape (batch_size, num_entries, in_features)
        """
        data = self.embed(data)
        for _, solver in enumerate(self.solvers):
            data = solver.inference(data, randperm)
        return self.output(data.mean(dim=1))
