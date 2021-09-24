"""
This file gives methods to create features for the tracker
"""

import numpy as np
import torch

EPS = 0.00001


"""
API functions to access all existing features
"""


def calculate_edge_features(nodes, edges, feature_list, sequence_info):
    """ Selects one of feature calculation methods and calculates the feature """
    for feature in feature_list:
        if feature not in edges and not "NORM_" in feature:
            edges[feature] = globals()[feature](nodes, edges, sequence_info)
    # Check if the feature should be normalized
    create_forward_backward_norm = False
    create_forward_backward_batch_norm = False
    create_forward_backward_mean = False
    iterator = 0
    while (not create_forward_backward_norm) and iterator < len(feature_list):
        entry = feature_list[iterator]
        if "_NORM_FORW" in entry or "_NORM_BACKW" in entry:
            create_forward_backward_norm = True
        iterator += 1
    iterator = 0
    while (not create_forward_backward_batch_norm) and iterator < len(feature_list):
        entry = feature_list[iterator]
        if "_BNORM_FORW" in entry or "_BNORM_BACKW" in entry:
            create_forward_backward_batch_norm = True
        iterator += 1
    iterator = 0
    while (not create_forward_backward_mean) and iterator < len(feature_list):
        entry = feature_list[iterator]
        if "_MNORM_" in entry:
            create_forward_backward_mean = True
        iterator += 1
    # Create normalizations if necessary
    if create_forward_backward_norm:
        edges = forward_backward_normalization(nodes, edges, feature_list, sequence_info)
    if create_forward_backward_batch_norm:
        edges = forward_backward_batch_normalization(nodes, edges, feature_list, sequence_info)
    if create_forward_backward_mean:
        edges = normalize_by_mean(nodes, edges, feature_list, sequence_info)
    return nodes, edges


"""
Normalization functions for global context normalization
"""


def augment_features(edge_features, valid_normalization):
    """ Create a list with new feature names of the augmented features """
    edge_features_augmented = edge_features
    for type in ["FRAME", "BATCH", "MEAN"]:
        if type == "FRAME":
            directions = ["_NORM_FORW","_NORM_BACKW", "_NORM_RATIO_FORW", "_NORM_RATIO_BACKW"]
        elif type == "BATCH":
            directions = ["_BNORM_FORW", "_BNORM_BACKW"]
        elif type == "MEAN":
            directions = ["_MNORM_"]
        else:
            raise NotImplementedError()
        for norm_direction in directions:
            edge_features_augmented += \
                [feature + norm_direction for feature in edge_features if feature in valid_normalization]
    return edge_features_augmented


def forward_backward_normalization(nodes, edges, feature_list, sequence_info):
    """ Normalization over all edges of a node directed to the future (forward) or the past (backward)"""
    normalizing_feature_names = [feature for feature in feature_list if feature+"_NORM_FORW" in feature_list]

    for feature in normalizing_feature_names:
            edges[feature + "_NORM_FORW"] = edges[feature].clone()
            edges[feature + "_NORM_BACKW"] = edges[feature].clone()
            edges[feature + "_NORM_RATIO_FORW"] = edges[feature].clone()
            edges[feature + "_NORM_RATIO_BACKW"] = edges[feature].clone()

    edge_features = [edges[key].float() for key in normalizing_feature_names]
    edge_features = [f[:, None] if len(f.shape) == 1 else f for f in edge_features]
    edge_features = torch.cat(edge_features, dim=1).float()
    edge_features_to_normalize = edge_features
    edges["frame_source"] = frame_source(nodes, edges, sequence_info)
    edges["frame_sink"] = frame_sink(nodes, edges, sequence_info)

    min_frame = nodes["frame"].min()
    max_frame = nodes["frame"].max()
    unique_frames = nodes["frame"].unique()
    unique_frames = np.sort(unique_frames)

    t_mapping = np.zeros(max_frame + 1, dtype=np.int)
    for i, fr in enumerate(unique_frames):
        t_mapping[fr] = i

    # Norm over all forward directed (outgoing) edges
    x, y, t, value = \
        edges["source"], edges["sink"], nodes["frame"][edges["sink"]] - min_frame, edge_features_to_normalize
    num_nodes = nodes["frame"].numel()
    num_features = len(normalizing_feature_names)
    map = torch.zeros(torch.Size([num_nodes, num_nodes, unique_frames.size, num_features]))
    map[x, y, t_mapping[t]] = value
    max_per_frame = map.max(1)[0]
    ratio = value.clone() / (max_per_frame[x, t_mapping[t]] + EPS)
    normed = value.clone() * ratio
    for feature_pos, feature in enumerate(normalizing_feature_names):
        edges[feature + "_NORM_FORW"] = normed[:, feature_pos].clone()
        edges[feature + "_NORM_RATIO_FORW"] = ratio[:, feature_pos].clone()
    # Norm over all backward directed (incoming) edges
    x, y, t, value = \
        edges["source"], edges["sink"], nodes["frame"][edges["source"]] - min_frame, edge_features_to_normalize
    num_nodes = nodes["frame"].numel()
    num_features = len(normalizing_feature_names)
    map = torch.zeros(torch.Size([num_nodes, num_nodes, unique_frames.size, num_features]))
    map[x, y, t_mapping[t]] = value
    max_per_frame = map.max(0)[0]
    ratio = value.clone() / (max_per_frame[y, t_mapping[t]] + EPS)
    normed = value.clone() * ratio
    for feature_pos, feature in enumerate(normalizing_feature_names):
        edges[feature + "_NORM_BACKW"] = normed[:, feature_pos].clone()
        edges[feature + "_NORM_RATIO_BACKW"] = ratio[:, feature_pos].clone()
    return edges


def forward_backward_batch_normalization(nodes, edges, feature_list, sequence_info):
    """ Normalization over all edges in the current frame directed to the future (forward) or the past (backward) """

    normalizing_feature_names = [feature for feature in feature_list if feature+"_BNORM_FORW" in feature_list]
    for feature in normalizing_feature_names:
        edges[feature + "_BNORM_FORW"] = edges[feature].clone()
        edges[feature + "_BNORM_BACKW"] = edges[feature].clone()

    num_nodes = nodes["frame"].shape[0]

    if edges["source"].shape[0] > 0:
        for feature in normalizing_feature_names:
            distances = torch.zeros((num_nodes, num_nodes)).double()
            distances[edges["source"], edges["sink"]] = edges[feature]
            max_forward = distances.max(dim=1, keepdim=True)[0]
            max_backward = distances.max(dim=0, keepdim=True)[0]
            norm_forward = distances / (max_forward + EPS)
            norm_backward = distances / (max_backward + EPS)
            norm_forward = norm_forward[edges["source"], edges["sink"]]
            norm_backward = norm_backward[edges["source"], edges["sink"]]
            edges[feature + "_BNORM_FORW"] = norm_forward
            edges[feature + "_BNORM_BACKW"] = norm_backward
    return edges


def normalize_by_mean(nodes, edges, feature_list, sequence_info):
    """ Normalize over all existing edges in the batch """
    normalizing_feature_names = [feature for feature in feature_list if feature+"_MNORM_" in feature_list]
    for feature in normalizing_feature_names:
        data = edges[feature].clone()
        maximum = data.max()
        edges[feature + "_MNORM_"] = data / (maximum + 0.0001)
    return edges


''' Edge features '''


def embedding_distance(nodes, edges, sequence_info):
    if "embedding_distance" in edges.keys():
        return edges["embedding_distance"]
    sink_emb = nodes["visual_embedding"][edges["sink"]]
    source_emb = nodes["visual_embedding"][edges["source"]]
    similarity = (sink_emb * source_emb).sum(axis=1).clamp(0, 2)
    return similarity


def frame_source(nodes, edges, sequence_info):
    edges["frame_source"] = nodes["frame"][edges["source"]]
    return edges["frame_source"]


def frame_sink(nodes, edges, sequence_info):
    edges["frame_sink"] = nodes["frame"][edges["sink"]]
    return edges["frame_sink"]


def center_iou(nodes, edges, sequence_info):
    """ Position invariant Intersection over union """
    # determine the (x, y)-coordinates of the intersection rectangle
    x1_1, x2_1, y1_1, y2_1, w_1, h_1 = \
        torch.zeros_like(nodes["x1"][edges["source"]]), nodes["x2"][edges["source"]] - nodes["x1"][edges["source"]], \
        torch.zeros_like(nodes["x1"][edges["source"]]), nodes["y2"][edges["source"]] - nodes["y1"][edges["source"]], \
        nodes["w"][edges["source"]], nodes["h"][edges["source"]]
    x1_2, x2_2, y1_2, y2_2, w_2, h_2 = \
        torch.zeros_like(nodes["x1"][edges["sink"]]), nodes["x2"][edges["sink"]] - nodes["x1"][edges["sink"]], \
        torch.zeros_like(nodes["x1"][edges["sink"]]), nodes["y2"][edges["sink"]] - nodes["y1"][edges["sink"]], \
        nodes["w"][edges["sink"]], nodes["h"][edges["sink"]]

    # Replace boxes to upper left corner

    xA = torch.max(x1_1, x1_2)
    yA = torch.max(y1_1, y1_2)
    xB = torch.min(x2_1, x2_2)
    yB = torch.min(y2_1, y2_2)

    # compute the area of intersection rectangle
    interArea = torch.relu(xB - xA) * torch.relu(yB - yA)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = torch.abs(w_1 * h_1)
    boxBArea = torch.abs(w_2 * h_2)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea.float() / (boxAArea + boxBArea - interArea)
    iou = iou.pow(3)
    # return the intersection over union value
    return iou


def delta_t(nodes, edges, sequence_info):
    """ Distance of time """
    delta = nodes["frame"][edges["sink"]] - nodes["frame"][edges["source"]]
    return delta.float() / sequence_info["sequence"]["fps"]
