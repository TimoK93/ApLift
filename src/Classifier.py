"""
Here the simple multi layer perceptron based classifier is implemented
"""

from collections import defaultdict
import torch
import torch.nn as nn

EPSILLON = 1e-5


class FeatureTransform(torch.nn.Module):
    def __init__(self, dim):
        """
        For input x, each feature x_i is normalized: x = (a_i*x_i+b_i)_i with trainable parameters a_i and b_i
        """
        super(FeatureTransform, self).__init__()
        self.feature_transformer = torch.nn.ModuleList()
        self.feature_dim = dim
        for feature_pos in range(dim):
            self.feature_transformer.append(nn.Linear(1, 1))

    def forward(self, x):

        updated_features = list()
        for feature in range(self.feature_dim):
            updated_features.append(self.feature_transformer[feature](x[:, feature].unsqueeze(1)))
        x = torch.cat(updated_features, dim=1)
        return x


class MLP(torch.nn.Module):
    """ The lightweight multi layer perceptron """
    def __init__(
            self, in_features, features):
        super().__init__()
        self.min_time = [i / 10 for i in range(0, 21)] + [1000]
        self.layer_blocks = torch.nn.ModuleList()
        self.layer_blocks2 = torch.nn.ModuleList()
        for i in range(21):
            self.layer_blocks.append(nn.Linear(in_features=in_features - 1, out_features=in_features - 1))
            self.layer_blocks2.append(nn.Linear(in_features=in_features - 1, out_features=1))
        self.feature_transform = FeatureTransform(len(features) - 1)
        self.leakyReLU = nn.LeakyReLU(0.1)
        self.features = features

    def forward(self, x):
        x_in = x
        dt_i = self.features.index("delta_t")
        inds = [_ for _ in range(len(self.features)) if _ != dt_i]
        x[:, inds] = self.feature_transform(x[:, inds])
        inds = [i for i in range(0, x.size()[1]) if i != dt_i]
        x_without_t = x_in[:, inds]
        t = x_in[:, dt_i:dt_i + 1]
        x_stack = list()
        for start, end, layer in zip(self.min_time[:-1], self.min_time[1:], range(0, len(self.layer_blocks))):
            x_out = self.layer_blocks[layer](x_without_t.clone())  # Timo_time_simple
            x_out = self.leakyReLU(x_out)
            x_out = self.layer_blocks2[layer](x_out)
            valid = (end >= t) * (t > start)
            x_out = x_out * valid
            x_stack.append(x_out)
        x = torch.cat(x_stack, dim=1)
        x = x.sum(dim=1, keepdim=True)
        return x


class Classifier(nn.Module):
    """
    The classifier that transforms features to a dictionary with in-, out-, node- and edge-costs.
    """

    def __init__(self, features):
        super().__init__()
        self.edge_features_keys = features
        self.edge_feature_dimensions = defaultdict(lambda: 1)
        in_features_mlp = sum([self.edge_feature_dimensions[f] for f in features])
        self.edge_mlp = MLP(in_features=in_features_mlp, features=features)

    def forward(self, nodes, edges):
        edge_features = [edges[key][0].float() for key in self.edge_features_keys]
        edge_features = [f[:, None] if len(f.shape) == 1 else f for f in edge_features]
        for key in self.edge_features_keys:
            value = edges[key][0][:, None] if len(edges[key][0].shape) == 1 else edges[key][0]
            assert self.edge_feature_dimensions[key] == value.shape[-1], (
            key, self.edge_feature_dimensions[key], value.shape)

        edge_cost_input = torch.cat(edge_features, dim=1).float()
        edge_costs = self.edge_mlp(edge_cost_input.clone())
        edge_costs = edge_costs[:, 0]
        edge_costs = edge_costs[None, :]
        node_costs = torch.ones_like(nodes["conf"])
        io_costs = 0
        det_costs = 0
        inout_var = io_costs * torch.ones(1)
        det_costs = det_costs * torch.ones(1)
        det_costs = -det_costs * node_costs
        inout_costs = inout_var * node_costs
        costs = {
            "in_costs": inout_costs.float(), "out_costs": inout_costs.float(),
            "node_costs": det_costs.float(), "edge_sources": edges["source"],
            "edge_sinks": edges["sink"], "edge_costs": edge_costs, "node_frames": nodes["frame"]
        }

        return costs

