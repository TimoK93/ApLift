"""
Post processing script for MOT15, MOT17 and MOT20.
Run this script on a working directory containing result files of tracked sequences.

Example usage for a tracking result on MOT17 sequences stored in ./results/ExampleRun:

  python3 run_postprocessing.py --working_directory results/ExampleRun/ --challenge MOT17

Parameter "challenge":
    Runs post processing depending if sequences are crowded and static (MOT20) or not (MOT15/17).

    MOT17:
       - Interpolation
    MOT20:
       - Remove Short ends from trajectories
       - Remove wrong assignments at borders
       - Remove irregular velocity trajectories determined by inter trajectory velocities
       - Remove irregular velocity trajectories determined by intra trajectory velocity
       - Interpolation

"""


import argparse
import numpy as np
import pandas as pd
import os
import shutil


BORDERS = {  # Image sizes needed for "split at border" post processing
    "MOT20-01.txt": dict(w=1920, h=1080),
    "MOT20-02.txt": dict(w=1920, h=1080),
    "MOT20-03.txt": dict(w=1173, h=880),
    "MOT20-04.txt": dict(w=1545, h=1080),
    "MOT20-05.txt": dict(w=1654, h=1080),
    "MOT20-06.txt": dict(w=1920, h=734),
    "MOT20-07.txt": dict(w=1920, h=1080),
    "MOT20-08.txt": dict(w=1920, h=734),
}


def parse_arguments():
    """ Parses arguments and checks if they are valid """
    parser = argparse.ArgumentParser(description='Postprocesses tracking results')
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
    for dir in sorted(os.listdir(working_directory)):
        result_file = os.path.join(working_directory, dir, dir + ".txt")
        dst_result_file = os.path.join(dst_directory, dir + ".txt")
        if os.path.exists(result_file):
            print("Found result for sequence", dir)
            if not os.path.exists(dst_result_file):
                shutil.copy(result_file, dst_result_file)


def write_to_file(data, file):
    """ Writes a result dataframe to a file """
    with open(file, "w+") as file:
        lines = list()
        for i in range(data["id"].size):
            line = ", ".join([
                str(int(data["frame"][i])), str(int(data["id"][i])), str(round(data["x1"][i])), str(round(data["y1"][i])),
                str(round(data["w"][i])), str(round(data["h"][i])), str(int(data["conf"][i])), str(int(data["x"][i])),
                str(int(data["y"][i])), str(int(data["z"][i]))
            ])
            lines.append(line)
        lines = "\n".join(lines)
        file.write(lines)


def short_ends(src_name, res_name, min_time_gap=10, min_ends=2):
    """
    Checks if a trajecotry has a "short end" which is a skip connection to a single deteftion in the beginning or
    end of a trajectory and removes the short end if existing
    """
    data = pd.read_csv(
        src_name, sep=",", index_col=False, header=None,
        names=["frame", "id", "x1", "y1", "w", "h", "conf", "x", "y", "z"]
    )
    totally_splitted = 0
    ids = np.unique(data["id"])
    next_id = np.max(data["id"]) + 1
    for _id in list(ids):
        inds = np.where(data["id"] == _id)[0]
        frames = np.sort(np.unique(data["frame"][inds]))
        for i, (frame1, frame2) in enumerate(zip(frames[0:-1], frames[1:])):
            if frame2 - frame1 < min_time_gap:
                continue
            # Check if at end
            if (min_ends - 1) <= i < (frames.size - min_ends):
                continue
            # Split trajectory
            #print("    Split with too short end")
            new_inds = (data["id"] == _id) & (data["frame"] > frame2)
            data.loc[new_inds, "id"] = next_id
            _id = next_id
            next_id += 1
            totally_splitted += 1
    write_to_file(data, res_name)


def split_at_border(src_name, res_name, borders, pixel_to_border=5, min_time_gap=10):
    """
    Checks if a trajectory has potentially wrong assignment at the borders of the scene and removes these.
    A potentially wrong assignment is a long skip connection if both detetctions are at th image border.
    """
    data = pd.read_csv(
        src_name, sep=",", index_col=False, header=None,
        names=["frame", "id", "x1", "y1", "w", "h", "conf", "x", "y", "z"]
    )
    left, top = pixel_to_border, pixel_to_border
    right, bottom = borders["w"] - pixel_to_border, borders["h"] - pixel_to_border
    is_left = data["x1"] <= left
    is_top = data["y1"] <= top
    is_right = (data["x1"] + data["w"]) >= right
    is_bottom = (data["y1"] + data["h"]) >= bottom
    is_at_border = is_bottom | is_top | is_right | is_left
    totally_splitted = 0
    ids = np.unique(data["id"])
    next_id = np.max(data["id"]) + 1
    for _id in list(ids):
        inds = np.where(data["id"] == _id)[0]
        frames = np.sort(np.unique(data["frame"][inds]))
        for frame1, frame2 in zip(frames[0:-1], frames[1:]):
            if frame2 - frame1 < min_time_gap:
                continue
            # Check if at border
            ind1, ind2 = \
                np.where((data["id"] == _id) & (data["frame"] == frame1))[0], \
                np.where((data["id"] == _id) & (data["frame"] == frame2))[0]
            if not is_at_border[ind1].any() and not is_at_border[ind2].any():
                continue
            # Split trajectory
            data.loc[(data["id"] == _id) & (data["frame"] > frame1), "id"] = next_id
            _id = next_id
            next_id += 1
            totally_splitted += 1
    write_to_file(data, res_name)


def extrinsic_trajectory_motion(src_name, res_name, min_time_gap=10):
    """ Compares the velocity caused by skip edges with the maximal existing motion in trajectories without longer
     skip edges. If the velocity is too high, the skip connections are splitted."""
    data = pd.read_csv(
        src_name, sep=",", index_col=False, header=None,
        names=["frame", "id", "x1", "y1", "w", "h", "conf", "x", "y", "z"]
    )
    motions = list()
    totally_splitted = 0
    ids = np.unique(data["id"])
    for _id in list(ids):
        inds = np.where(data["id"] == _id)[0]
        frames = np.sort(np.unique(data["frame"][inds]))
        start_frame, end_frame = frames[0], frames[0]
        for frame1, frame2 in zip(frames[0:-1], frames[1:]):
            end_frame = frame2
            if frame2 - frame1 <= 2 and frame2 != frames[-1]:
                continue
            if end_frame - start_frame > 10:
                ind1, ind2 = \
                    np.where((data["id"] == _id) & (data["frame"] == start_frame))[0][0], \
                    np.where((data["id"] == _id) & (data["frame"] == end_frame))[0][0]
                offset_x = \
                    (np.abs(data["x1"][ind1] - data["x1"][ind2]) +
                    np.abs(data["x1"][ind1] + data["w"][ind1] - data["x1"][ind2] - data["w"][ind2])) / 2
                offset_y = \
                    (np.abs(data["y1"][ind1] - data["y1"][ind2]) +
                     np.abs(data["y1"][ind1] + data["h"][ind1] - data["y1"][ind2] - data["h"][ind2])) / 2
                offset = np.sqrt(offset_x * offset_x + offset_y * offset_y)
                velocity = offset / (end_frame - start_frame)
                motions.append(velocity)
            start_frame = frame2

    next_id = np.max(data["id"]) + 1
    # Detect trajectory gaps and split if they are to fast
    for _id in list(ids):
        inds = np.where(data["id"] == _id)[0]
        frames = np.sort(np.unique(data["frame"][inds]))

        for frame1, frame2 in zip(frames[0:-1], frames[1:]):
            if frame2 - frame1 < min_time_gap:
                continue
            # Compare velocity
            ind1, ind2 = \
                np.where((data["id"] == _id) & (data["frame"] == frame1))[0][0], \
                np.where((data["id"] == _id) & (data["frame"] == frame2))[0][0]
            offset_x = \
                (np.abs(data["x1"][ind1] - data["x1"][ind2]) +
                 np.abs(data["x1"][ind1] + data["w"][ind1] - data["x1"][ind2] - data["w"][ind2])) / 2
            offset_y = \
                (np.abs(data["y1"][ind1] - data["y1"][ind2]) +
                 np.abs(data["y1"][ind1] + data["h"][ind1] - data["y1"][ind2] - data["h"][ind2])) / 2
            offset = np.sqrt(offset_x * offset_x + offset_y * offset_y)
            velocity = offset / (frame2 - frame1)
            is_to_fast = velocity > np.max(motions)
            if not is_to_fast:
                continue
            # Split trajectory
            new_inds = (data["id"] == _id) & (data["frame"] > frame1)
            data.loc[new_inds, "id"] = next_id
            _id = next_id
            totally_splitted += 1
            next_id += 1
    write_to_file(data, res_name)


def intrinsic_trajectory_motion(src_name, res_name, min_time_gap=10, motion_similarity=5, max_angle_difference=np.pi/2):
    """  Compares the velocity caused by skip edges with the motion in the rest of the trajectory. If the velocity is
    too high, the skip connections are splitted."""
    data = pd.read_csv(
        src_name, sep=",", index_col=False, header=None,
        names=["frame", "id", "x1", "y1", "w", "h", "conf", "x", "y", "z"]
    )
    totally_splitted = 0
    ids = np.unique(data["id"])
    next_id = np.max(data["id"]) + 1
    # Detect trajectory gaps and split if they motions do not match are to fast
    for _id in list(ids):
        inds = np.where(data["id"] == _id)[0]
        frames = np.sort(np.unique(data["frame"][inds]))
        # Detect gaps
        begins, ends = [frames[0]], list()
        for frame1, frame2 in zip(frames[0:-1], frames[1:]):
            if frame2 - frame1 < min_time_gap:
                continue
            ends.append(frame1)
            begins.append(frame2)
        ends.append(frames[-1])
        for i in range(1, len(begins)):
            def calc_motion(ind1, ind2):
                frame1, frame2 = data["frame"][ind1], data["frame"][ind2]
                offset_x = \
                    (np.abs(data["x1"][ind1] - data["x1"][ind2]) +
                     np.abs(data["x1"][ind1] + data["w"][ind1] - data["x1"][ind2] - data["w"][ind2])) / 2
                offset_y = \
                    (np.abs(data["y1"][ind1] - data["y1"][ind2]) +
                     np.abs(data["y1"][ind1] + data["h"][ind1] - data["y1"][ind2] - data["h"][ind2])) / 2
                offset = np.sqrt(offset_x * offset_x + offset_y * offset_y)
                velocity = offset / np.maximum(0.001, frame2 - frame1)
                angle = np.arctan2(offset_x, offset_y)
                return velocity, angle, frame2 -frame1

            def compare_motions(vel1, angle1, vel2, angle2):
                difference_vel = np.maximum(vel1, vel2) / np.maximum(0.001, np.minimum(vel1, vel2))
                difference_angle = angle1 - angle2
                difference_angle = ((difference_angle + np.pi) % (2 * np.pi)) - np.pi
                if vel1 == 0 or vel2 == 0:
                    return True
                if difference_vel > motion_similarity:
                    return False
                if np.abs(difference_angle) > max_angle_difference:
                    return False
                return True
            indA1, indA2 = \
                np.where((data["id"] == _id) & (data["frame"] == begins[i - 1]))[0][0], \
                np.where((data["id"] == _id) & (data["frame"] == ends[i - 1]))[0][0]
            indB1, indB2 = \
                np.where((data["id"] == _id) & (data["frame"] == begins[i]))[0][0], \
                np.where((data["id"] == _id) & (data["frame"] == ends[i]))[0][0]
            # Calculate first motion
            velocityA, angleA, delta_tA = calc_motion(indA1, indA2)
            velocityB, angleB, delta_tB = calc_motion(indB1, indB2)
            velocitySKIP, angleSKIP, delta_tSKIP = calc_motion(indA2, indB1)
            # Compare motions
            compA = compare_motions(velocityA, angleA, velocitySKIP, angleSKIP)
            compB = compare_motions(velocityB, angleB, velocitySKIP, angleSKIP)
            if (compA or delta_tA < 2) and (compB or delta_tB < 2):
                continue
            # Split trajectory
            new_inds = (data["id"] == _id) & (data["frame"] > ends[i - 1])
            data.loc[new_inds, "id"] = next_id
            _id = next_id
            next_id += 1
            totally_splitted += 1
    write_to_file(data, res_name)


def interpolate(src_name, res_name, max_interpolation_length=60):
    """ Interpolates gaps in trajectories linear """
    data = pd.read_csv(
        src_name, sep=",", index_col=False, header=None,
        names=["frame", "id", "x1", "y1", "w", "h", "conf", "x", "y", "z"]
    )
    new_dets = dict()
    new_dets["id"] = list()
    new_dets["frame"] = list()
    new_dets["x1"] = list()
    new_dets["y1"] = list()
    new_dets["w"] = list()
    new_dets["h"] = list()
    ids, frames, x1, y1, w, h = data["id"], data["frame"], data["x1"], data["y1"], data["w"], data["h"]
    x2, y2 = x1 + w, y1 + h
    id_name, counts = np.unique(ids, return_counts=True)
    for n, c in zip(id_name, counts):
        if n == -1 or c < 2:
            continue
        inds = np.where(ids == n)[0]
        _frames = frames[inds]
        if np.sum(np.abs(_frames - np.sort(_frames))) != 0:
            continue
        for i, fi_pair in enumerate(zip(_frames[:-1], _frames[1:], inds[:-1], inds[1:])):
            f1, f2, i1, i2 = fi_pair
            if f1 == f2 - 1 or np.abs(f1 - f2) > max_interpolation_length:
                continue
            for dt in range(1, f2 - f1):
                l1, l2 = (f2 - f1 - dt) / (f2 - f1), dt / (f2 - f1)
                new_x1, new_x2, new_y1, new_y2 = \
                    x1[i1] * l1 + x1[i2] * l2, x2[i1] * l1 + x2[i2] * l2, \
                    y1[i1] * l1 + y1[i2] * l2, y2[i1] * l1 + y2[i2] * l2
                new_dets["id"].append(n)
                new_dets["frame"].append(f1 + dt)
                new_dets["x1"].append(new_x1)
                new_dets["y1"].append(new_y1)
                new_dets["w"].append(new_x2 - new_x1)
                new_dets["h"].append(new_y2 - new_y1)
    if len(new_dets["id"]) > 0:
        new_dets["id"] = np.asarray(new_dets["id"])
        new_dets["frame"] = np.asarray(new_dets["frame"])
        new_dets["x1"] = np.asarray(new_dets["x1"])
        new_dets["y1"] = np.asarray(new_dets["y1"])
        new_dets["w"] = np.asarray(new_dets["w"])
        new_dets["h"] = np.asarray(new_dets["h"])
        new_dets["conf"] = np.ones_like(new_dets["h"])
        new_dets["x"] = -np.ones_like(new_dets["h"])
        new_dets["y"] = -np.ones_like(new_dets["h"])
        new_dets["z"] = -np.ones_like(new_dets["h"])
        new_dets = pd.DataFrame.from_dict(new_dets)
        data = pd.concat([data, new_dets], ignore_index=True)
    write_to_file(data, res_name)


if __name__ == "__main__":
    challenge, working_directory = parse_arguments()
    interpolation_directory = os.path.join(working_directory, "results_postprocessed")
    if os.path.exists(interpolation_directory):
        shutil.rmtree(interpolation_directory)
    os.makedirs(interpolation_directory, exist_ok=True)
    collect_data(working_directory, interpolation_directory)
    for name in sorted(os.listdir(interpolation_directory)):
        print(">>> Postprocess", name)
        if challenge == "MOT17" or challenge == "MOT15":
            src_file = os.path.join(interpolation_directory, name)
            result_file = os.path.join(interpolation_directory, name)
            interpolate(src_file, result_file)
        elif challenge == "MOT20":
            src_file = os.path.join(interpolation_directory, name)
            result_file = os.path.join(interpolation_directory, name)
            short_ends(src_file, result_file)
            split_at_border(result_file, result_file, borders=BORDERS[name])
            extrinsic_trajectory_motion(result_file, result_file)
            intrinsic_trajectory_motion(result_file, result_file)
            interpolate(src_file, result_file)
        else:
            raise NotImplementedError()