"""
A script to run the main script with all sequences of a dataset.
To use the script a config.yaml needs to be specified.

Example usage:

    python3 main.py config/example_config.yaml

if "pretrained_model_path" is passed as an argument in the config, training is skipped and pretrained models are used
for the inference.

"""

import os
import shutil
from copy import copy

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # GPUs are not necessary!

from main import run_pipeline
from src.utilities.config_reader import main_function


def copyanything(src, dst):
    for root, dirs, files in os.walk(src):
        for name in files:
            dir = root.replace(src, dst)
            dst_file = os.path.join(dir, name)
            if os.path.exists(dst_file):
                print("Model", dst_file, "is already existing!")
            os.makedirs(dir, exist_ok=True)
            shutil.copy(os.path.join(root, name), os.path.join(dir, name))


@main_function
def main(working_dir, dataset: str, pretrained_models_path=None, **kwargs):
    """ Runs the main pipeline on all sequences of a dataset """
    ''' Create directory an copy pretrained models '''
    os.makedirs(working_dir, exist_ok=True)
    if pretrained_models_path is not None:
        copyanything(os.path.join(pretrained_models_path), working_dir)

    ''' Creates a list of jobs to be executed'''
    jobs = list()
    if dataset == "MOT17":
        detectors = ["FRCNN", "DPM", "SDP"]
        train_sequences = [2, 4, 5, 9, 10, 11, 13]
        test_sequences = [1, 3, 6, 7, 8, 12, 14]

        for d in detectors:
            for t in train_sequences:
                train = ["MOT17-%s-%s" % (str(_).rjust(2, "0"), d) for _ in train_sequences if _ != t]
                val = ["MOT17-%s-%s" % (str(t).rjust(2, "0"), d)]
                jobs.append(dict(train=train, val=val, detector=d))
            for t in test_sequences:
                train = ["MOT17-%s-%s" % (str(_).rjust(2, "0"), d) for _ in train_sequences]
                val = ["MOT17-%s-%s" % (str(t).rjust(2, "0"), d)]
                jobs.append(dict(train=train, val=val, detector=d))
    elif dataset == "MOT20":
        test_sequences = [4, 6, 7, 8]
        train_sequences = [1, 2, 3, 5]

        for t in train_sequences:
            train = ["MOT20-%s" % str(_).rjust(2, "0") for _ in train_sequences if _ != t]
            val = ["MOT20-%s" % str(t).rjust(2, "0")]
            jobs.append(dict(train=train, val=val, detector="FRCNN"))
        for t in test_sequences:
            train = ["MOT20-%s" % str(_).rjust(2, "0") for _ in train_sequences]
            val = ["MOT20-%s" % str(t).rjust(2, "0")]
            jobs.append(dict(train=train, val=val, detector="FRCNN"))
    elif dataset == "MOT15":
        test_sequences = [
            'Venice-1', 'TUD-Crossing', 'PETS09-S2L2', 'KITTI-19', 'KITTI-16', 'ETH-Jelmoli', 'ETH-Linthescher',
            'ETH-Crossing', 'AVG-TownCentre', 'ADL-Rundle-3', 'ADL-Rundle-1'
        ]
        train_sequences = [
            'Venice-2', 'KITTI-17', 'KITTI-13', 'ETH-Sunnyday', 'ETH-Pedcross2', 'ETH-Bahnhof', 'ADL-Rundle-8',
            'TUD-Stadtmitte', 'TUD-Campus', 'ADL-Rundle-6', 'PETS09-S2L1'
        ]
        for t in train_sequences:
            train = [_ for _ in train_sequences if _ != t]
            val = [t]
            jobs.append(dict(train=train, val=val, detector="FRCNN"))
        for t in test_sequences:
            train = [_ for _ in train_sequences if _ != t]
            val = [t]
            jobs.append(dict(train=train, val=val, detector="FRCNN"))

    ''' Runs the jobs sequentially '''
    features = copy(kwargs["data_config"]["edge_features"])
    for job in jobs:
        print("Run Job:", job)
        if os.path.exists(os.path.join(working_dir, job["val"][0], job["val"][0] + ".txt")):
            print("... Result file already existing!")
            continue
        kwargs["data_config"]["edge_features"] = copy(features)
        kwargs["data_config"]["dataset"]["detector"] = job["detector"]
        kwargs["training_config"]["sequences_for_training"] = job["train"]
        kwargs["training_config"]["sequences_for_inference"] = job["val"]
        run_pipeline(working_dir=working_dir, **kwargs)


if __name__ == "__main__":
    main()
