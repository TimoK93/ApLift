"""
Data wrapper for MOT17 dataset
"""

import os
import pandas
import configparser
import numpy as np
from src.utilities.geometrics import iou
from scipy.optimize import linear_sum_assignment


class MOT17:
    """
    Loads MOT17 with all test and train sequences with additional data like visual embedding, ground truth mapping etc.
    """
    def __init__(
            self, detector,
            root_path,
            load_data=True,
            load_visual_embedding=False,
            preprocessed=False,
            calc_gt_mapping=False,
            **kwargs
    ):

        self.calc_gt_mapping = calc_gt_mapping
        self.detector = detector
        os.environ["DATASET"] = "MOT17"

        assert self.detector in ["FRCNN", "DPM", "SDP", "ALL"], "Detector not existing!"
        if preprocessed:
            self.root = os.path.join(root_path, "MOT17-Preprocessed")
        else:
            self.root = os.path.join(root_path, "MOT17")
        self.load_visual_embedding = load_visual_embedding
        self.test_sequences = list()
        self.train_sequences = list()
        if load_data:
            self.load_sequences()

    def load_sequences(self):
        # Load train and test sequences
        test_sequences = os.listdir(os.path.join(self.root, "test"))
        test_sequences.sort()
        self.test_sequences = [
            self.load_sequence(os.path.join(self.root, "test", _)) for _ in test_sequences
            if _.endswith(self.detector) or self.detector == "ALL"
        ]
        train_sequences = os.listdir(os.path.join(self.root, "train"))
        train_sequences.sort()
        self.train_sequences = [
            self.load_sequence(os.path.join(self.root, "train", _)) for _ in train_sequences
            if _.endswith(self.detector) or self.detector == "ALL"
        ]

    def load_sequence(self, directory):
        """ Loads the data of the sequence and returns a sequence dict """
        assert os.path.isdir(directory)

        # Read ini file
        ini_file = configparser.ConfigParser()
        ini_dir = directory
        ini_dir = ini_dir.replace("-Preprocessed", "")
        ini_file.read(os.path.join(ini_dir, "seqinfo.ini"))

        def image_path_function(frame: int) -> str:
            """ Generate the name of a image in frame "frame" """
            return os.path.join(ini_dir, "img1", str(frame).zfill(6) + ".jpg")

        optical_flow = np.load(os.path.join(ini_dir, "OPTICAL_FLOW.npy"))
        max_optical_flow = np.load(os.path.join(ini_dir, "MAX_OPTICAL_FLOW.npy"))

        # Create basic information
        seq = dict()
        seq["name"] = os.path.basename(directory)
        seq["image_path"] = image_path_function
        seq["image_width"] = int(ini_file["Sequence"]["imWidth"])
        seq["image_height"] = int(ini_file["Sequence"]["imHeight"])
        seq["frames"] = int(ini_file["Sequence"]["seqLength"])
        seq["fps"] = float(ini_file["Sequence"]["frameRate"])

        # Read the gt and detection file
        det_file = os.path.join(directory,  "det", "det.txt")
        if os.path.exists(os.path.join(directory, "gt", "gt.txt")):
            gt_file = os.path.join(directory, "gt", "gt.txt")
        else:
            gt_file = os.path.join(ini_dir, "gt", "gt.txt")

        if os.path.exists(det_file):
            det = pandas.read_csv(
                det_file, sep=",", index_col=False, header=None,
                names=["frame", "id", "x1", "y1", "w", "h", "conf", "3D_x", "3D_y", "3D_z"]
            )
            _det = dict()
            _det["w"], _det["h"], _det["x1"], _det["x2"], _det["y1"], _det["y2"], _det["conf"], _det["frame"], _det["id"] =\
                det["w"].to_numpy(), det["h"].to_numpy(), det["x1"].to_numpy(), det["x1"].to_numpy() + det["w"].to_numpy(),\
                det["y1"].to_numpy(), det["y1"].to_numpy() + det["h"].to_numpy(), det["conf"].to_numpy(), \
                det["frame"].to_numpy().astype(int), det["id"].to_numpy().astype(int),
            _det["row"] = np.arange(_det["w"].size).astype(int)
            seq["det"] = _det
            if self.load_visual_embedding:
                embedding_file = os.path.join(directory, "det", "DG-Net-features.npy")
                embedding_file_untuned = os.path.join(directory, "det", "DG-Net-features_untuned.npy")
                if os.path.exists(embedding_file_untuned):
                    embedding_file = embedding_file_untuned
                seq["det"]["visual_embedding"] = np.load(embedding_file, mmap_mode="r")
                seq["det"]["visual_embedding"] = np.load(embedding_file)
            seq["det"]["optical_flow"] = optical_flow[seq["det"]["frame"] - 1]
            seq["det"]["max_optical_flow"] = max_optical_flow[seq["det"]["frame"] - 1]
            seq["det"]["time"] = np.copy(seq["det"]["frame"]) / seq["fps"]
        else:
            seq["det"] = None

        if os.path.exists(gt_file):
            gt = pandas.read_csv(
                gt_file, sep=",", index_col=False, header=None,
                names=["frame", "id", "x1", "y1", "w", "h", "conf", "class", "visibility"]
            )
            _gt = dict()
            _gt["w"], _gt["h"], _gt["x1"], _gt["x2"], _gt["y1"], _gt["y2"], _gt["conf"], _gt["frame"], _gt["id"], \
            _gt["class"], _gt["visibility"] = \
                gt["w"].to_numpy(), gt["h"].to_numpy(), gt["x1"].to_numpy(), gt["x1"].to_numpy() + gt["w"].to_numpy(), \
                gt["y1"].to_numpy(), gt["y1"].to_numpy() + gt["h"].to_numpy(), \
                gt["conf"].to_numpy(), gt["frame"].to_numpy(), gt["id"].to_numpy(), gt["class"].to_numpy(), \
                gt["visibility"].to_numpy()
            _gt["row"] = np.arange(_gt["w"].size).astype(int)
            _gt["visibility"] = np.abs(_gt["visibility"])
            _gt["class"][_gt["class"] == -1] = 1

            seq["gt"] = _gt
            if self.load_visual_embedding:
                embedding_file = os.path.join(ini_dir, "gt", "DG-Net-features.npy")
                embedding_file_untuned = os.path.join(directory, "gt", "DG-Net-features_untuned.npy")
                if os.path.exists(embedding_file_untuned):
                    embedding_file = embedding_file_untuned
                seq["gt"]["visual_embedding"] = np.load(embedding_file, mmap_mode="r")
            seq["gt"]["optical_flow"] = optical_flow[seq["gt"]["frame"] - 1]
            seq["gt"]["max_optical_flow"] = max_optical_flow[seq["gt"]["frame"] - 1]
            seq["gt"]["time"] = np.copy(seq["gt"]["frame"]) / seq["fps"]
        else:
            seq["gt"] = None

        if self.calc_gt_mapping and seq["gt"]:
            seq["det"]["gt_id"] = create_gt_id_mapping(seq["det"], seq["gt"])

        clip_at_timeframe = os.getenv("CLIP_SEQUENCES", None)
        if clip_at_timeframe:
            clip_at_timeframe = int(clip_at_timeframe)
            inds = seq["det"]["frame"] <= clip_at_timeframe
            for key, item in seq["det"].items():
                seq["det"][key] = seq["det"][key][inds]
            if seq["gt"]:
                inds = seq["gt"]["frame"] <= clip_at_timeframe
                for key, item in seq["gt"].items():
                    seq["gt"][key] = seq["gt"][key][inds]

        return seq


def create_gt_id_mapping(det, gt):
    """ Creates a mappting from bounding boxes to ground truth """
    gt_id = -np.ones_like(det["frame"])
    if os.getenv("DATASET") == "MOT20":
        iou_thresh = 0.75
    else:
        iou_thresh = 0.5

    bb_det = np.stack([det["x1"], det["y1"], det["x2"], det["y2"]], axis=1)
    bb_gt = np.stack([gt["x1"], gt["y1"], gt["x2"], gt["y2"]], axis=1)

    for frame in np.unique(det["frame"]):
        det_inds = np.where(det["frame"] == frame)[0]
        gt_inds = np.where(gt["frame"] == frame)[0]
        if gt_inds.size == 0 or det_inds.size == 0:
            continue
        box_det, box_gt = bb_det[det_inds], bb_gt[gt_inds]
        box_det = np.repeat(box_det, gt_inds.size, axis=0)
        box_gt = np.tile(box_gt, (det_inds.size, 1))
        _ious = iou(box_det, box_gt)
        ious = np.zeros((det_inds.size, gt_inds.size))
        for i in range(det_inds.size):
            ious[i] = _ious[i*gt_inds.size:(i+1)*gt_inds.size]
        costs = 1 - ious + (ious < iou_thresh) * 10000
        row_ind, col_ind = linear_sum_assignment(costs)
        is_valid = ious[row_ind, col_ind] >= iou_thresh
        if is_valid.size == 0:
            continue
        row_ind, col_ind = row_ind[is_valid], col_ind[is_valid]
        gt_id[det_inds[row_ind]] = gt["id"][gt_inds[col_ind]]

    return gt_id


def calculate_max_velocity_per_frame(detections):
    """ Returns the maximal velocity per frame based on optical flow of bounding boxes"""
    max_velocities = list()
    for f in sorted(np.unique(detections["frame"])):
        dets = np.where(detections["frame"] == f)[0]
        velocities = detections["mean_and_max_velocity"][dets, 1, :]
        velocities = np.linalg.norm(velocities, 2, axis=1)
        max_velocity = np.max(velocities)
        max_velocities.append(max_velocity)
    return np.asarray(max_velocities)

