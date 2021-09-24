"""
Focal loss used for our framework
"""

import torch

EPSILLON = 0.001


class FocalLoss(object):
    def __init__(self, focal_lambda=1, positive_weight=1, positive_weight_max=10):
        """ Implementation of our extended focal loss """
        self.focal_lambda = focal_lambda
        self.positive_weight = positive_weight
        self.positive_weight_max = positive_weight_max

    def __call__(self, data: dict):

        gt = data["edges"]["gt_edge_costs"]
        pred = data["prediction"]["activated_edges"]

        time_frames = torch.unique(data["edges"]["delta_t"])
        truth = data["edges"]["delta_t"] * gt
        false = data["edges"]["delta_t"] * (1 - gt)

        ''' Calculate unweighted focal loss '''
        p = pred * gt + (1 - pred).abs() * (1 - gt)
        p = p.clamp(0.001, 0.999)
        focal_loss = (1 - p).abs().pow(self.focal_lambda)

        ''' Count positives and negative edges '''
        num_positives, num_negatives = torch.zeros_like(time_frames), torch.zeros_like(time_frames)
        for i in range(time_frames.numel()):
            num_positives[i] = (truth == time_frames[i]).sum()
            num_negatives[i] = (false == time_frames[i]).sum()
        ratio = num_positives / num_negatives.clamp(1)
        ''' Calculate weights '''
        time_weights_positive = ratio.clamp(self.positive_weight, self.positive_weight_max) / num_positives.clamp(1)
        time_weights_negative = 1 / num_negatives.clamp(1)
        weight = torch.zeros_like(gt).float()
        for i in range(time_frames.numel()):
            weight[truth == time_frames[i]] = time_weights_positive[i]
            weight[false == time_frames[i]] = time_weights_negative[i]
        weight = weight / time_frames.numel()
        ''' Apply weight '''
        focal_loss = weight * focal_loss
        factor = -p.log()
        loss = factor * focal_loss
        loss = loss.sum()

        return loss

