"""
Methods to calculate evaluations
"""

import os
import motmetrics as mm
import numpy as np

from src.TrackingModel import TrackingModel
from src.datasets import Data
from src.utilities.geometrics import iou
from src.Tracker import Tracker


def evaluate_sequence(model: TrackingModel, dataset: Data, result_dir):
    """
    Evaluates a sequence and stores the results
    :param model: The tracking model to use
    :param dataset: A dataset with exactly one sequence for inference
    :param result_dir: A directory to store the result file and metrics
    :return: A dictionary with metrics
    """
    ''' Prepare the data'''
    assert len(dataset.sequences_for_inference) == 1, "Only one sequence is allowed!"
    name = dataset.sequences_for_inference[0]["name"]
    set_environment(name)
    ''' Track the sequence'''
    tracking_data = Tracker.track(model, dataset)
    ''' Create a result file in MOT format '''
    result_text = to_result_text(tracking_data)
    with open(os.path.join(result_dir, name + ".txt"), "w+") as file:
        file.write(result_text)
    ''' Check against groud truth if ground truth is existing '''
    has_ground_truth = bool(tracking_data["ground_truth"])
    if has_ground_truth:  # Replace this with condition for test sequences
        remove_overlap_threshold = 0.75 if "MOT20" in dataset.root else 0.5  # As defined by MotChallenge.org
        metrics = compute_metrics(tracking_data, remove_overlap_threshold=remove_overlap_threshold)
        with open(os.path.join(result_dir, "val_metrics.txt"), "w+") as file:
            for key, metric in metrics["metrics"].items():
                file.write("%s:%s, " % (key, metric))
    else:
        metrics = dict()
    return metrics["metrics"]


def compute_metrics(data: dict, remove_overlap_threshold=0.5):
    """
    Evaluates the tracking result against ground truth. Works for MotChallenge classes.
    For Evaluation script details look at https://github.com/cheind/py-motmetrics

    :param remove_overlap_threshold:
        A IoU threshold to match Detections to Ground truth
    :param data:
        A dictionary with the data prediction and ground truth data.
        Must at least contain the keys:
            - prediction
                - node_ids
            - ground_truth
                - id
                - frame
                - x1, x2, y1, y2
                - class
            - nodes
                - frame
                - x1, x2, y1, y2

    :return Dictionary with all metrics computed by the mot_metrics package
    """
    # Load detection and ground_truth geometry, id and time data
    ids_det, frames_det, x1_det, x2_det, y1_det, y2_det = \
        data["prediction"]["node_ids"], \
        data["nodes"]["frame"], data["nodes"]["x1"], data["nodes"]["x2"], data["nodes"]["y1"], data["nodes"]["y2"]

    ids_gt, frames_gt, x1_gt, x2_gt, y1_gt, y2_gt, is_pedestrian = \
        data["ground_truth"]["id"], data["ground_truth"]["frame"],  data["ground_truth"]["x1"], \
        data["ground_truth"]["x2"], data["ground_truth"]["y1"], data["ground_truth"]["y2"], \
        data["ground_truth"]["class"] == 1,

    remove_gt = \
        (data["ground_truth"]["class"] == 2) + (data["ground_truth"]["class"] == 7) + \
        (data["ground_truth"]["class"] == 8) + (data["ground_truth"]["class"] == 12)

    ''' Fill accumulator with detection/gt data '''
    acc = mm.MOTAccumulator(auto_id=True)
    frames = np.sort(np.unique(np.concatenate([np.unique(frames_det), np.unique(frames_gt)])))
    for frame in frames:
        # Get indices of detections and ground_truth that are in the current frame
        det_inds, gt_inds = np.where(frames_det == frame)[0], np.where(frames_gt == frame)[0]
        det_inds = det_inds[ids_det[det_inds] != -1]
        if det_inds.size == 0 and gt_inds.size == 0:
            continue
        # Remove all unimportant detections
        box1 = np.stack([x1_gt[gt_inds], y1_gt[gt_inds], x2_gt[gt_inds], y2_gt[gt_inds]], axis=1)
        box2 = np.stack([x1_det[det_inds], y1_det[det_inds], x2_det[det_inds], y2_det[det_inds]], axis=1)
        ious = np.zeros(shape=(gt_inds.size, det_inds.size))
        for box in range(box1.shape[0]):
            _box1 = np.repeat(box1[box:box + 1], axis=0, repeats=box2.shape[0])
            ious[box] = iou(_box1, box2)
        ious[ious < remove_overlap_threshold] = 0
        ious = ious * np.expand_dims(remove_gt[gt_inds], axis=1)
        valid_detections = np.sum(ious, axis=0) == 0
        # Remove all invalid gt that are not pedestrians
        det_inds = det_inds[valid_detections]
        gt_inds = gt_inds[is_pedestrian[gt_inds]]
        gt_objects = ids_gt[gt_inds]
        det_objects = ids_det[det_inds]
        # Use (1 - intersection_over_union) as cost value
        box1 = np.stack([x1_gt[gt_inds], y1_gt[gt_inds], x2_gt[gt_inds], y2_gt[gt_inds]], axis=1)
        box2 = np.stack([x1_det[det_inds], y1_det[det_inds], x2_det[det_inds], y2_det[det_inds]], axis=1)
        ious = np.zeros(shape=(gt_objects.size, det_objects.size))
        for box in range(box1.shape[0]):
            _box1 = np.repeat(box1[box:box + 1], axis=0, repeats=box2.shape[0])
            ious[box] = iou(_box1, box2)
        # Remove all invalid IoUs by setting them to NaN
        ious = 1 - ious  # <- to costs
        ious[ious > 0.5] = np.nan  # <- NaN is like infinite costs, so they cannot be matched
        if gt_objects.size == 0 and det_objects.size == 0:
            continue
        acc.update(gt_objects, det_objects, ious)
    ''' Compute the metrics'''
    mh = mm.metrics.create()
    metrics = [
        'mota', 'idf1', 'idp', 'idr', 'idtp', 'idfn', 'idfp', 'num_switches', 'num_false_positives', 'num_misses',
        'mostly_tracked', 'partially_tracked', 'mostly_lost', 'num_fragmentations', 'precision', 'recall', 'num_objects'
    ]
    summary = mh.compute(acc, metrics=metrics, name='acc')
    _ = dict()
    for key, item in summary.items():
        _[key] = item[0]
    summary = _
    return {"metrics": summary}


def to_result_text(data: dict):
    """
    Creates a result file text

    :param data:
        A dictionary with the data prediction.  Must at least contain the keys:
            - prediction
                - node_ids
            - nodes
                - frame
                - x1, x2, y1, y2

    :return
    """
    det, pred = data["nodes"], data["prediction"]
    ids_det = pred["node_ids"]
    frames_det = det["frame"]
    x1_det, y1_det, w_det, h_det  = det["x1"],  det["y1"], det["w"], det["h"]

    result_text = list()
    for i in range(ids_det.size):
        if int(ids_det[i]) == -1:
            continue
        line = [
            str(int(frames_det[i])), str(int(ids_det[i])), str(int(x1_det[i])), str(int(y1_det[i])),
            str(int(w_det[i])), str(int(h_det[i])), str(1.0), str(-1), str(-1), str(-1)
        ]
        result_text.append(", ".join(line))
    result_text = "\n".join(result_text)
    return result_text


def set_environment(name):
    """
    Decrease or increase threshold or sequences with bird eye view or moving cameras (as explained in the appendix).
    The environment variables will be read out in the graph pruning step.
    """
    if "MOT17-03" in name or "MOT17-04" in name:
        os.environ["DECREASE_GEOMETRIC_THRESHOLD"] = "1"
    else:
        os.environ["DECREASE_GEOMETRIC_THRESHOLD"] = "0"
    if "MOT17-10" in name or "MOT17-14" in name or "MOT17-13" in name:
        os.environ["INCREASE_GEOMETRIC_THRESHOLD"] = "1"
    else:
        os.environ["INCREASE_GEOMETRIC_THRESHOLD"] = "0"
    os.environ["TRAINING"] = "0"

