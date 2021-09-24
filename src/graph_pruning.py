"""

This fileimplements the graph pruning described by the paper. The graph is pruned by removing "obvious cuts" and
determine "obvious joins" which gets very lof costs later to induce soft constrains.

obvious cuts are removed from the edge list
obvious joins are labeled as "high confident"
"""

import numpy as np
import math
import os


""" The API for pruning methods """


def prune(pre_pruning, nodes, max_edge_len):
    """ Selects one of pruning methods and creates the nodes """
    if pre_pruning == "none":
        return none(nodes, max_edge_len)
    elif pre_pruning == "geometric":
        return geometric(nodes, max_edge_len)
    elif pre_pruning == "geometric_with_heuristic":
        return geometric_with_heuristic(nodes, max_edge_len)
    elif pre_pruning == "geometric_with_dg_heuristic":
        return geometric_with_dg_heuristic(nodes, max_edge_len)
    elif pre_pruning == "optical_flow_with_dg_heuristic":
        return optical_flow_with_dg_heuristic(nodes, max_edge_len)
    else:
        raise NotImplementedError


""" The pruning methods """


def none(nodes, max_edge_len):
    """
    Creates a full graph for given nodes
    """
    num_nodes = nodes["frame"].size
    sources, sinks = list(), list()
    for i in range(int(np.ceil(num_nodes / 1000))):
        for j in range(int(np.ceil(num_nodes / 1000))):
            # Create all possible edges between nodes in a given time gap, even recurrent edges
            num_nodes1, num_nodes2 = nodes["frame"][i*1000:(i+1)*1000].size, nodes["frame"][j*1000:(j+1)*1000].size
            node1_ids = np.arange(num_nodes1, dtype=np.int32) + i * 1000
            node2_ids = np.arange(num_nodes2, dtype=np.int32) + j * 1000
            if np.max(nodes["frame"][node2_ids]) - np.min(nodes["frame"][node1_ids]) < 0 or \
                    np.min(nodes["frame"][node2_ids]) - np.max(nodes["frame"][node1_ids]) > max_edge_len:
                continue
            _sources = np.tile(np.expand_dims(np.copy(node1_ids), 1), (1, node2_ids.size)).astype(np.int64)
            _sinks = np.tile(np.expand_dims(np.copy(node2_ids), 0), (node1_ids.size, 1)).astype(np.int64)
            # Remove all edges that are not directed to the future
            try:
                edges_directed_to_future = (nodes["frame"][_sinks] - nodes["frame"][_sources]) > 0
                _sources, _sinks = _sources[edges_directed_to_future].astype(np.int64), _sinks[edges_directed_to_future].astype(np.int64)
                valid_edges = (nodes["frame"][_sinks] - nodes["frame"][_sources]) <= max_edge_len
                _sources, _sinks = _sources[valid_edges], _sinks[valid_edges]
            except Exception as e:
                print(np.min(_sources), np.max(_sources), np.min(_sinks), np.max(_sinks))
                raise e
            sources.append(_sources)
            sinks.append(_sinks)
            del num_nodes2, num_nodes1, _sources, _sinks, edges_directed_to_future, valid_edges
    if len(sources) > 0:
        sources, sinks = np.concatenate(sources), np.concatenate(sinks)
    else:
        sources, sinks = np.zeros(0), np.zeros(0)

    return {"source": sources.astype(int), "sink": sinks.astype(int)}


def geometric(nodes, max_edge_len):
    """
    Creates a full graph from    none()   and removes all edges which would cause too much velocity. This step is used
    to pre-filter the size of the batch.

    The velocity is determined by the displacement devided by the time distance.
    The maximal velocity is experimental determined and is different for the datasets:
            MOT20 is generally 0.12
            MOT17 is generally 0.30
                - static sequence with bird view can also get 0.12 because its the same setup than MOT20
                - moving sequences get 0.80 to compensate motion caused displacements

    The threshold is induced by setting an environment variable
    """
    # Choose threshold
    dataset = os.getenv("DATASET", "MOT17")
    SAFETY_FACTOR = 0.12 if dataset == "MOT20" else 0.30
    decrease = os.getenv("DECREASE_GEOMETRIC_THRESHOLD", "0")  # You can control the factor for static scenes
    training = os.getenv("TRAINING", "0")  # To get more variance, you can increase the threshold during training
    if decrease == "1" and dataset == "MOT17":
        SAFETY_FACTOR = 0.12
    increase = os.getenv("INCREASE_GEOMETRIC_THRESHOLD", "0")
    if (increase == "1" or training == "1") and dataset == "MOT17":
        SAFETY_FACTOR = 0.80
    # Start pruning
    edges = none(nodes, max_edge_len)
    sinks, sources = edges["sink"].astype(np.int64), edges["source"].astype(np.int64)
    inds = list()
    bz = 1000000000  # Batch size so the memory does not collapse
    for i in range(math.ceil(sinks.size / bz)):
        try:
            # Calculate distance between boxes
            delta_frame = nodes["frame"][sinks[i * bz: (i + 1) * bz]] - nodes["frame"][sources[i * bz: (i + 1) * bz]]
            delta_t = nodes["time"][sinks[i * bz: (i+1) * bz]] - nodes["time"][sources[i * bz: (i+1) * bz]]
            mean_w = (nodes["w"][sinks[i * bz: (i+1) * bz]] + nodes["w"][sources[i * bz: (i+1) * bz]]) / 2
            mean_h = (nodes["h"][sinks[i * bz: (i+1) * bz]] + nodes["h"][sources[i * bz: (i+1) * bz]]) / 2
            x1, x2, y1, y2 = \
                (nodes["x1"][sinks[i * bz: (i+1) * bz]] + nodes["x2"][sinks[i * bz: (i+1) * bz]]) / 2, \
                (nodes["x1"][sources[i * bz: (i+1) * bz]] + nodes["x2"][sources[i * bz: (i+1) * bz]]) / 2, \
                (nodes["y1"][sinks[i * bz: (i+1) * bz]] + nodes["y2"][sinks[i * bz: (i+1) * bz]]) / 2, \
                (nodes["y1"][sources[i * bz: (i+1) * bz]] + nodes["y2"][sources[i * bz: (i+1) * bz]]) / 2,
            distance_x, distance_y = np.abs(x1 - x2), np.abs(y1 - y2)
            x_per_frame, y_per_frame = distance_x / delta_t, distance_y / delta_t
            # Increase safety factor for boxes with very small time gap to avoid wrong matches caused by detection noise
            one = 25 * (delta_frame == 1) * delta_t
            two = 25 * (delta_frame == 2) * delta_t / 2
            three = 25 * (delta_frame == 3) * delta_t / 3
            four = 25 * (delta_frame == 4) * delta_t / 4
            five = 25 * (delta_frame == 5) * delta_t / 5
            if dataset == "MOT17":
                safety_factor = 25 * (
                        one * 0.5 + two * 0.15 + three * 0.05 + four * 0.0 * five * 0.0 +
                        SAFETY_FACTOR
                )
            else:
                safety_factor = 25 * (
                        one * 0.1 + two * 0.05 + three * 0.03 + four * 0.025 * five * 0.01 +
                        SAFETY_FACTOR
                )
            # Remove all that have bigger distance per step than their own size
            safety_factor_x, safety_factor_y = safety_factor, safety_factor / 2  # y-direction is unlikely
            _inds = np.where((x_per_frame <= mean_w * safety_factor_x) * (y_per_frame <= mean_h * safety_factor_y))[0]
            if _inds.size > 0:
                inds.append(_inds + bz * i)
        except Exception as e:
            raise e
    if len(inds) > 0:
        inds = np.concatenate(inds)
    else:
        inds = np.zeros(0, dtype=int)

    return {"source": sources[inds], "sink": sinks[inds]}


def geometric_with_heuristic(nodes, max_edge_len):
    """
    Creates a geometrical pruned graph from    geometric()   and selects "high confident" edges with an IoU of
    0.5 and higher after motion compensation which are in consecutive frames
    """
    output = geometric(nodes, max_edge_len)

    # Geometric heuristic
    iou_threshold = 0.5
    sources, sinks = output["source"], output["sink"]

    x1_a, x1_b, x2_a, x2_b = nodes["x1"][sources], nodes["x1"][sinks], nodes["x2"][sources], nodes["x2"][sinks]
    y1_a, y1_b, y2_a, y2_b = nodes["y1"][sources], nodes["y1"][sinks], nodes["y2"][sources], nodes["y2"][sinks]
    delta_frame = nodes["frame"][sinks] - nodes["frame"][sources]
    w_a, h_a, w_b, h_b = nodes["w"][sources], nodes["h"][sources], nodes["w"][sinks], nodes["h"][sinks]

    xA = np.maximum(x1_a, x1_b)
    yA = np.maximum(y1_a, y1_b)
    xB = np.minimum(x2_a, x2_b)
    yB = np.minimum(y2_a, y2_b)

    # compute the area of intersection rectangle
    interArea = np.maximum(xB - xA, 0) * np.maximum(yB - yA, 0)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = np.abs(w_a * h_b)
    boxBArea = np.abs(w_b * h_b)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    ious = interArea / (boxAArea + boxBArea - interArea)

    high_confident = np.logical_and(delta_frame == 1, ious >= iou_threshold) * 1

    if np.sum(high_confident) > 0 and os.getenv("DATASET", "") == "MOT17":
        # Calculate if is high confident after shift
        inds = high_confident == 1
        optical_flow = nodes["optical_flow"][sinks][inds]
        optical_flow = np.round(optical_flow)
        x1_a, x1_b, x2_a, x2_b = x1_a[inds], x1_b[inds], x2_a[inds], x2_b[inds]
        y1_a, y1_b, y2_a, y2_b = y1_a[inds], y1_b[inds], y2_a[inds], y2_b[inds]
        w_a, h_a, w_b, h_b = w_a[inds], h_a[inds], w_b[inds], h_b[inds]

        a_is_right = x1_a >= x1_b
        x1_a, x2_a = x1_a + a_is_right * optical_flow, x2_a + a_is_right * optical_flow
        x1_b, x2_b = x1_b + (1 - a_is_right) * optical_flow, x2_b + (1 - a_is_right) * optical_flow

        xA = np.maximum(x1_a, x1_b)
        yA = np.maximum(y1_a, y1_b)
        xB = np.minimum(x2_a, x2_b)
        yB = np.minimum(y2_a, y2_b)
        # compute the area of intersection rectangle
        interArea = np.maximum(xB - xA, 0) * np.maximum(yB - yA, 0)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = np.abs(w_a * h_b)
        boxBArea = np.abs(w_b * h_b)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        ious = interArea / (boxAArea + boxBArea - interArea)

        high_confident[inds] = high_confident[inds] * (ious >= iou_threshold)

    return {"source": sources, "sink": sinks, "high_confident": high_confident}


def geometric_with_dg_heuristic(nodes, max_edge_len):
    """
    Creates a geometrical pruned graph from    geometric_with_heuristic()   and adds "high confident" edges which have
    a very similar appearance vector with cosine similarity of 1.95 and higher which are in consecutive frames
    """
    output = geometric_with_heuristic(nodes, max_edge_len)

    dg_threshold = 1.95
    sources, sinks, high_confident = output["source"], output["sink"], output["high_confident"]
    delta_frame = nodes["frame"][sinks] - nodes["frame"][sources]
    # Use DG net as constraint

    distance = list()
    bz = 10000  # Batch size so the memory does not collapse
    for i in range(int(np.ceil(sources.size / bz))):
        embedding_src, embedding_snk = \
            nodes["visual_embedding"][sources[i*bz:(i+1)*bz]], nodes["visual_embedding"][sinks[i*bz:(i+1)*bz]]
        _distance = embedding_src * embedding_snk
        _distance = np.sum(_distance, 1)
        distance.append(_distance)
    if len(distance) > 0:
        distance = np.concatenate(distance)
        high_confident = np.maximum(high_confident, np.logical_and(delta_frame == 1, distance >= dg_threshold) * 1)

    return {"source": sources, "sink": sinks, "high_confident": high_confident}


def optical_flow_with_dg_heuristic(nodes, max_edge_len):
    """
    Creates a pruned graph from    geometric_with_dg_heuristic()   and filters out edges which should not be possible
    by given displacement determined by the optical flow.
    """
    output = geometric_with_dg_heuristic(nodes, max_edge_len)
    sources, sinks, high_confident = output["source"], output["sink"], output["high_confident"]

    distance_x, distance_y = \
        ((nodes["x1"][sinks] + nodes["x2"][sinks]) - (nodes["x1"][sources] + nodes["x2"][sources])) / 2, \
        ((nodes["y1"][sinks] + nodes["y2"][sinks]) - (nodes["y1"][sources] + nodes["y2"][sources])) / 2,
    distance_squared = distance_x * distance_x + distance_y * distance_y

    k_d = 175  # Additional displacement to avoid wrong decision caused by noise
    max_distance = (nodes["max_optical_flow"][sinks] - nodes["max_optical_flow"][sources] + k_d)
    max_distance_squared = max_distance * max_distance

    inds = distance_squared <= max_distance_squared

    return {"source": sources[inds], "sink": sinks[inds], "high_confident": high_confident[inds]}


if __name__ == "__main__":
    x = np.asarray([1, 2, 3])
    x = np.tile(x, 3)
    print(x)
