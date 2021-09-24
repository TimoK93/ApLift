"""
Defines geometric functions
"""
import numpy as np


def iou(box1, box2):
    """
    Intersection over union
    :param box1: np.ndarray [[x1, y1, x2, y2]]
    :param box2: np.ndarray [[x1, y1, x2, y2]]
    :return: np.ndarray [iou_value]
    """
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = np.maximum(box1[:, 0], box2[:, 0])
    yA = np.maximum(box1[:, 1], box2[:, 1])
    xB = np.minimum(box1[:, 2], box2[:, 2])
    yB = np.minimum(box1[:, 3], box2[:, 3])
    # Compute the area of intersection rectangle
    interArea = np.abs((np.maximum((xB - xA), 0)) * np.maximum((yB - yA), 0))
    # Compute the area of both the prediction and ground-truth ectangles
    boxAArea = np.abs((box1[:, 0] - box1[:, 2]) * (box1[:, 1] - box1[:, 3]))
    boxBArea = np.abs((box2[:, 0] - box2[:, 2]) * (box2[:, 1] - box2[:, 3]))
    # Compute the intersection over union by taking the intersection area and dividing it by the sum of
    #   prediction + ground-truth areas - the interesection area
    iou = interArea / (boxAArea + boxBArea - interArea)
    return iou
