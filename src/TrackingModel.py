"""
Implements the tracking model that combines the Classifier and the solver as a torch module.

"""

import torch
import torch.nn as nn

from src.LiftedSolver import LiftedSolver
from src.Classifier import Classifier



class TrackingModel(nn.Module):
    """
    A pytorch wrapper for the full Tracking method.
    During training the solver just forwards the input costs, and applies sigmoid on them.

    """
    def __init__(self, solver_config, features):
        """
        Create a tracking model that consists of the following parts:
            1. Edge Classifier
            2. Solver

        :param solver_config: Configuration for the solver
        :param features: A list of feature names used in the classifier
        """
        super(TrackingModel, self).__init__()
        self.cost_generator = Classifier(features)
        self.solver = LiftedSolver(**solver_config)

    def training_step(self, costs):
        result = dict()
        result["activated_nodes"] = torch.ones_like(costs["in_costs"])
        result["activated_edges"] = torch.nn.Sigmoid()(-costs["edge_costs"])
        result["in_nodes"] = torch.zeros_like(costs["in_costs"])
        result["out_nodes"] = torch.zeros_like(costs["in_costs"])
        return result

    def calculate_costs(self, nodes, edges):
        costs = self.cost_generator(nodes, edges)
        if 'gt_edge_costs' in edges.keys():
            costs['gt_edges'] = edges['gt_edge_costs']
        return costs

    def run_solver(self, costs):
        result = self.solver(costs)
        return result


def load_model(checkpoint_path=None, **model_config):
    """
    Loads a model

    :param checkpoint_path: If this is not None, the given model is loaded and the weights are used
    :param model_config: The config for the model

    :return A TrackingModel instance
    """
    model = TrackingModel(**model_config)
    model.eval()
    if checkpoint_path:
        state_dict = torch.load(checkpoint_path)
        model.load_state_dict(state_dict["model_state_dict"], strict=False)

    return model


def store_model(model_path: str, model: TrackingModel):
    """ Stores the model to the given path """
    is_train = True if model.training is True else False
    model.eval()
    torch.save({'model_state_dict': model.state_dict()}, model_path)
    if is_train:
        model.train()









