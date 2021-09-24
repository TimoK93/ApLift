"""
Evaluation script for MOT15, MOT17 and MOT20.
Run this script on working directories containing tracked sequences.

Example usage for a tracking result on MOT17 sequences stored in ./results/ExampleRun:

  python3 evaluate.py --working_directory results/ExampleRun --challenge MOT17

"""


import argparse
import os
import shutil


def parse_arguments():
    """ Parses arguments and checks if they are valid """
    parser = argparse.ArgumentParser(description='Evaluates tracking results')
    parser.add_argument('--challenge', type=str, help='The Challenge like MOT15, MOT17 or MOT20')
    parser.add_argument('--working_directory', type=str, help='The tracking working directory like ./results/TestRun')
    args = parser.parse_args()
    assert args.challenge in ["MOT15", "MOT17", "MOT20"], "Dataset is not existing!"
    assert os.path.exists(args.working_directory), "Working directory is not existing!"
    working_directory = os.path.abspath(args.working_directory)
    return args.challenge, working_directory


def collect_data(working_directory, dst_directory):
    """ Searches for result.txt files and copies them to a directory """
    os.makedirs(dst_directory, exist_ok=True)
    for dir in os.listdir(working_directory):
        result_file = os.path.join(working_directory, dir, dir + ".txt")
        dst_result_file = os.path.join(dst_directory, dir + ".txt")
        if os.path.exists(result_file):
            print("Found result for sequence", dir)
            if os.path.exists(dst_result_file):
                os.remove(dst_result_file)
            shutil.copy(result_file, dst_result_file)


def link_ground_truth_sequences(result_directory, gt_directory, new_gt_directory):
    """ Creates a pseudo ground-truth folder and adds symlinks to original ground-truth directories into it """
    os.makedirs(new_gt_directory, exist_ok=True)
    for file in os.listdir(result_directory):
        if "txt" not in file:
            continue
        file = file.replace(".txt", "")
        src_file = os.path.abspath(os.path.join(gt_directory, file))
        dst_file = os.path.join(new_gt_directory, file)
        if not os.path.exists(dst_file) and not os.path.islink(dst_file):
            os.symlink(src_file, dst_file)


def run_official_evaluation_package(tracker_dir, result_dir, working_dir):
    """
    Evaluates sequences with official evaluation package provided by motchallenge.net
    """
    os.makedirs(result_dir, exist_ok=True)
    trackers_folder, tracker_to_eval = os.path.split(os.path.normpath(tracker_dir))
    output_folder = os.path.abspath(result_dir)
    gt_to_use = "gt"

    eval_string = f"export PYTHONPATH={os.path.join(os.getcwd(), 'third_party_code', 'HOTA-metrics')}; " \
                  f"cd {os.path.join(os.getcwd(), 'third_party_code', 'HOTA-metrics')}; " \
                  f"python3 eval_code/Scripts/run_MOTChallenge.py " \
                  f"--TRACKERS_FOLDER {trackers_folder} " \
                  f"--GT_FOLDER {working_dir} " \
                  f"--OUTPUT_FOLDER {output_folder} " \
                  f"--TRACKERS_TO_EVAL {tracker_to_eval} " \
                  f"--GT_TO_USE {gt_to_use} "
    cwd = os.getcwd()
    os.chdir(os.path.join(cwd, "third_party_code", "HOTA-metrics"))
    os.system(eval_string)
    os.chdir(cwd)


if __name__ == "__main__":
    ''' Read arguments and prepare parameter '''
    challenge, working_directory = parse_arguments()
    tracking_directory = os.path.join(working_directory, "results")
    evaluation_directory = os.path.join(working_directory, "evaluation")
    gt_data_root = os.path.join("data", "tmp")
    gt_directory = os.path.join(gt_data_root, challenge, "train")
    new_gt_directory = os.path.join(working_directory, "gt")
    ''' Prepare the directory '''
    collect_data(working_directory, tracking_directory)
    link_ground_truth_sequences(tracking_directory, gt_directory, new_gt_directory)
    ''' Evaluate '''
    print(">>> Evaluation without postprocessing:")
    run_official_evaluation_package(tracking_directory, evaluation_directory, working_directory)
    ''' Evaluate post processed directory '''
    postprocessed_tracking_directory = os.path.join(working_directory, "results_postprocessed")
    postprocessed_evaluation_directory = os.path.join(working_directory, "evaluation_postprocessed")
    if os.path.exists(postprocessed_tracking_directory):
        print(">>> Evaluation with postprocessing:")
        run_official_evaluation_package(
            postprocessed_tracking_directory, postprocessed_evaluation_directory, working_directory
        )
