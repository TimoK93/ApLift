import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import linear_sum_assignment
import random
import re
from torch._six import string_classes
import collections.abc as container_abcs
import math

from src.datasets.MOT17 import MOT17
from src.datasets.MOT17Preprocessed import MOT17Preprocessed
from src.datasets.MOT15 import MOT15
from src.datasets.MOT15Preprocessed import MOT15Preprocessed
from src.datasets.MOT20 import MOT20
from src.datasets.MOT20Preprocessed import MOT20Preprocessed
from src.graph_pruning import prune
from src.calculate_features import calculate_edge_features
from src.utilities.geometrics import iou
from src.utilities.colors import PrintColors
from src.utilities.conversions import dict_to_torch

np_str_obj_array_pattern = re.compile(r'[SaUO]')


def create_dataset(name, **kwargs):
    """ Loads a dataset determined by its name"""
    if name == "MOT17":
        return MOT17(**kwargs)
    elif name == "MOT17Preprocessed":
        return MOT17Preprocessed(**kwargs)
    elif name == "MOT15":
        return MOT15(**kwargs)
    elif name == "MOT15Preprocessed":
        return MOT15Preprocessed(**kwargs)
    elif name == "MOT20":
        return MOT20(**kwargs)
    elif name == "MOT20Preprocessed":
        return MOT20Preprocessed(**kwargs)
    else:
        raise NotImplementedError


def get_sequences(sequences, selection, strict=True):
    name_to_seq = {s['name']: s for s in sequences}
    result_strings = \
        [name_to_seq[name] for name in selection if type(name) == str and (name in name_to_seq.keys() or strict)]
    result_numbers = [sequences[i] for i in selection if type(i) == int]
    return result_strings + result_numbers


class Data(Dataset):
    """ A wrapper for all datasets """

    def __init__(self, dataset, graph_config, features, features_to_normalize):
        """
        Loads a dataset from the dataset directory
        """
        print("%sLoad dataset: %s" % (PrintColors.OKBLUE, PrintColors.ENDC))
        ''' Load data'''
        data = create_dataset(**dataset)
        self.root = data.root
        self.train_sequences = data.train_sequences
        self.test_sequences = data.test_sequences
        print("Train sequences")
        for seq in self.train_sequences:
            print("    %s" % seq["name"])
        print("Test sequences")
        for seq in self.test_sequences:
            print("    %s" % seq["name"])
        ''' Container to define sequences for training loop or inference'''
        self.sequences_for_training_loop = None
        self.sequences_for_inference = None
        ''' Define variables '''
        self.batches = list()
        self.is_inference = True
        self.shuffle_seed = random.randint(0, 100)
        self.features = features
        self.graph_config = graph_config
        self.finisher_function = lambda x: x

    def get_sequence_by_name(self, name):
        """ Returns the sequence data if existing """
        for _ in self.train_sequences + self.test_sequences:
            if _["name"] == name:
                return _
        raise Exception("Sequence %s not existing!" % name)

    def create_batches(
            self, sequences: list, is_inference: bool, part=1, total_parts=1, shuffle=True, finisher_function=None
    ):
        """
        Create batches for training of inference

        :param sequences: A list of sequences to be used
        :param is_inference: if True, create batches for inference. False for training
        :param finisher_function: A function that is processed on the batch in __get_item__() call
        :param shuffle: Shuffle during training
        :param part: The current part
        :param total_parts: Divide the dataset to n parts

        """
        ''' Check if inputs are valid '''
        for seq in sequences:
            assert seq in self.train_sequences or seq in self.test_sequences, "Not a sequence!"
        self.is_inference = is_inference
        self.finisher_function = finisher_function if finisher_function is not None else lambda x: x
        config = self.graph_config
        ''' Create all valid batches for the sequences'''
        self.batches = list()
        for seq in sequences:
            name = seq["name"]
            last_frame = seq["frames"]
            end_gap = config["sparse_frame_indices"][-1] - config["sparse_frame_indices"][-2]
            for i in range(-config["max_edge_len"] + end_gap + 2, last_frame, 1):
                inds = np.asarray(config["sparse_frame_indices"], dtype=int)
                inds = inds + (i - max(i, 1))  # shift if start of index set < 1
                batch = {
                    "sequence": name, "start_frame": max(i, 1),
                    "end_frame": min(last_frame, i + config["max_edge_len"]),
                    "intended_indices": inds
                }
                batch["frames_in_batch"] = 1 + batch["end_frame"] - batch["start_frame"]
                is_node_in_batch = 0
                for i in batch["intended_indices"]:
                    if is_node_in_batch >= 2:
                        break
                    inds = np.where(seq["det"]["frame"] == batch["start_frame"] + i)[0]
                    if inds.size > 0:
                        is_node_in_batch += 1
                if is_node_in_batch >= 2:
                    if batch["end_frame"] > batch["start_frame"]:
                        self.batches.append(batch)
        ''' Shuffle batches for training and split it into parts if necessary'''
        if shuffle:
            random.Random(self.shuffle_seed).shuffle(self.batches)
        if total_parts != 1:
            start = math.floor((part - 1) * (len(self.batches) / total_parts))
            end = math.floor(part * (len(self.batches) / total_parts))
            self.batches = self.batches[start:end]

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        assert idx < len(self.batches)
        batch = self.batches[idx]
        batch["sequence"] = [_ for _ in self.train_sequences + self.test_sequences if _["name"] == batch["sequence"]][0]
        nodes, edges, ground_truth = self._get_nodes_and_edges(batch)
        batch.update(dict(nodes=nodes, edges=edges, ground_truth=ground_truth))
        ''' If gt is existing: Create assignment between gt and det '''
        if len(ground_truth.keys()) != 0:
            ious = self._create_det_gt_iou_values(nodes, ground_truth)
            ground_truth["detection_gt_ious"] = ious
            ground_truth["gt_edge_activation"] = self._create_gt_edge_activation_matrix(ground_truth, True)
            ground_truth["gt_edge_activation_sparse"] = self._create_gt_edge_activation_matrix(ground_truth, False)
            ground_truth["gt_ids"] = self._create_gt_id_mapping(nodes, ground_truth, ious)
            gt_edge_cost, sparse_gt_edge_cost = self._create_gt_edge_costs(ground_truth["gt_ids"], edges, nodes)
            edges["gt_edge_costs"] = gt_edge_cost
            edges["sparse_gt_edge_costs"] = sparse_gt_edge_cost
        # Calculate edge and node features
        if nodes["time"].shape[0] == 0:
            batch.pop("sequence")
        nodes, edges = calculate_edge_features(nodes, edges, self.features, batch)
        batch = {
            "nodes": dict_to_torch(nodes), "edges": dict_to_torch(edges), "ground_truth": dict_to_torch(ground_truth)
        }
        batch = self.finisher_function(batch)
        return batch

    ''' Internal functions for batch creation '''

    def _get_nodes_and_edges(self, batch, is_global_graph=False, return_edges=True):
        """ Creates nodes and edges for a batch """
        config = self.graph_config
        ''' Check which frames should be extracted '''
        frames_to_extract = np.arange(batch["start_frame"], batch["end_frame"] + 1)
        if not is_global_graph:
            inds = batch["intended_indices"]
            inds = inds[inds < len(frames_to_extract)]
            inds = inds[0 <= inds]
            frames_to_extract = frames_to_extract[inds]
            number_of_valid_detections = batch["sequence"]["det"]["frame"].size
            frames_to_extract = frames_to_extract[frames_to_extract < number_of_valid_detections]
            frames_to_extract = frames_to_extract[batch["sequence"]["det"]["frame"][frames_to_extract] > 0]
        ''' Load detections and ground truth'''
        # Ground truth
        nodes, ground_truth = dict(), dict()
        if batch["sequence"]["gt"] is not None:
            gt = batch["sequence"]["gt"]
            inds = np.where((gt["frame"] >= batch["start_frame"]) * (gt["frame"] <= batch["end_frame"]))[0]
            if not self.is_inference:
                # Discard all ground truth detections that have a too low visibility during training
                inds = inds[gt["visibility"][inds] >= config["gt_min_visibility"]]
            all_inds = inds[np.isin(gt["frame"][inds], frames_to_extract)]
            class_inds = [np.where(gt["class"][all_inds] == c)[0] for c in self.graph_config["gt_classes_to_train_on"]]
            valid_classes = all_inds[np.concatenate(class_inds)]
            inds = valid_classes
            inds = sorted(inds, key=lambda ind: gt["frame"][ind])
            for key, item in gt.items():
                if key == "visual_embedding" or key == "l2_embedding":
                    continue
                ground_truth[key] = np.copy(item[inds])
        else:
            ground_truth = {}
        # Detections
        det = batch["sequence"]["det"]
        inds = np.where((det["frame"] >= batch["start_frame"]) * (det["frame"] <= batch["end_frame"]))[0]
        inds = inds[np.isin(det["frame"][inds], frames_to_extract)]
        inds = sorted(inds, key=lambda ind: np.copy(det["frame"][ind]))
        # Sample random detections during training
        if not self.is_inference:
            if config["sample_detections"]:
                num_samples = math.ceil(len(inds) * config["detection_sample_rate"])
                inds = random.sample(inds, num_samples)
                inds = sorted(inds)
        for key, item in det.items():
            nodes[key] = np.copy(item[inds])
        nodes["id"] = -np.ones(shape=nodes["id"].shape)
        # Use extended sampling which just keeps nodes that are in the same region
        if not self.is_inference:
            if config["sample_top_k_around_random_point"] > -1:
                x, y = 0.5 * (nodes["x1"] + nodes["x2"]), 0.5 * (nodes["y1"] + nodes["y2"])
                anchor_x = int(random.random() * batch["sequence"]["image_width"])
                anchor_y = int(random.random() * batch["sequence"]["image_height"])
                distances = np.abs(x-anchor_x) + np.abs(y - anchor_y)
                indices = np.argsort(distances)[:math.ceil(config["sample_top_k_around_random_point"])]
                indices = np.unique(indices)
                indices.sort()
                for key in nodes.keys():
                    nodes[key] = nodes[key][indices]
        ''' Create edges for the graph with a specified pruning method '''
        max_len = min(int(self.graph_config["max_time_distance"] * batch["sequence"]["fps"]), config["max_edge_len"])
        if return_edges:
            pruning = self.graph_config["pre_pruning"]
            edges = prune(pruning, nodes, max_len)
        else:
            edges = dict()

        return dict_to_torch(nodes), dict_to_torch(edges), dict_to_torch(ground_truth)

    def _create_det_gt_iou_values(self, nodes, ground_truth):
        """
        Calculates the intersection over union for all detection/ground_truth pairs. The output is a 2d Matrix with
        size (num_dets x num_gt). If detections and frames are not in the same frame, the iou value will be set to 0.

        """
        num_det, num_gt = nodes["id"].shape[0], ground_truth["id"].shape[0]
        t_det, t_gt = nodes["frame"], ground_truth["frame"]
        bb_det = np.stack([
            np.copy(nodes["x1"]), np.copy(nodes["y1"]), np.copy(nodes["x2"]), np.copy(nodes["y2"])
        ], axis=1)
        bb_gt = np.stack([
            np.copy(ground_truth["x1"]), np.copy(ground_truth["y1"]),
            np.copy(ground_truth["x2"]), np.copy(ground_truth["y2"])
        ], axis=1)
        t_det, t_gt = np.tile(np.expand_dims(t_det, 1), (1, num_gt)), np.tile(np.expand_dims(t_gt, 0), (num_det, 1))
        delta_t = t_det - t_gt
        in_same_frame = delta_t == 0
        ious = np.zeros(shape=(bb_det.shape[0], bb_gt.shape[0]))
        for i in range(num_det):
            ious[i] = iou(np.tile(np.copy(bb_det[i:i+1]), (num_gt, 1)), np.copy(bb_gt))
        ious = ious * in_same_frame
        return ious

    def _create_gt_edge_activation_matrix(self, ground_truth, activate_higher_order_edges=False):
        num_gt = ground_truth["id"].shape[0]
        ids = ground_truth["id"]
        ids1, ids2 = \
            np.repeat(np.expand_dims(np.copy(ids), 1), num_gt, axis=1), \
            np.repeat(np.expand_dims(np.copy(ids), 0), num_gt, axis=0)
        delta_id = ids1 - ids2
        is_same_id = ((delta_id == 0) * 1).astype(int)
        if not activate_higher_order_edges:
            time_ranks = ranks(ground_truth["frame"].cpu().detach().numpy())
            time_diff_table = time_ranks[None, :] - time_ranks[:, None]
            result = is_same_id * (time_diff_table == 1)
            return result
        return is_same_id

    def _create_gt_id_mapping(self, nodes, ground_truth, ious):
        costs = 1 - np.copy(ious)
        costs[costs >  0.5] = 100000
        rows, cols = linear_sum_assignment(costs)
        _valid = costs[rows, cols] < 0.5
        rows, cols = rows[_valid], cols[_valid]
        real_ids = np.copy(nodes["id"])
        real_ids[rows] = ground_truth["id"][cols]
        return real_ids

    def _create_gt_edge_costs(self, real_ids, edges, nodes):
        num_nodes = real_ids.size
        ids_1, ids_2 = np.expand_dims(np.copy(real_ids), axis=0), np.expand_dims(np.copy(real_ids), axis=1)
        ids_1, ids_2 = np.repeat(ids_1, num_nodes, axis=0), np.repeat(ids_2, num_nodes, axis=1)
        id_diff = ids_1 - ids_2
        id_diff[id_diff != 0] = -1
        id_diff[ids_1 == -1] = -1
        id_diff += 1
        time_ranks = ranks(nodes["frame"].cpu().detach().numpy())
        time_diff_table = time_ranks[None, :] - time_ranks[:, None]
        sparse_id_diff = id_diff * (time_diff_table == 1)
        gt_edge_costs = id_diff[edges["source"], edges["sink"]]
        sparse_gt_edge_costs = sparse_id_diff[edges["source"], edges["sink"]]
        return gt_edge_costs, sparse_gt_edge_costs

    def return_batch_with_full_graph(self, sequence, return_edges=True):
        old_is_inference_state = self.is_inference
        self.is_inference = True
        last_frame = sequence["frames"]
        batch = {"sequence": sequence, "start_frame": 1,"end_frame": last_frame}
        batch["frames_in_batch"] = 1 + batch["end_frame"] - batch["start_frame"]
        nodes, edges, ground_truth = self._get_nodes_and_edges(batch, is_global_graph=True, return_edges=return_edges)
        # Remove all higher order edges with morge than desired edge length
        if return_edges:
            max_time_distance = self.graph_config["max_edge_len"]
            delta_t = nodes["frame"][edges["sink"]] - nodes["frame"][edges["source"]]
            inds = delta_t <= max_time_distance
            edges["source"] = edges["source"][inds]
            edges["sink"] = edges["sink"][inds]
        batch.update(dict(nodes=nodes, edges=edges, ground_truth=dict_to_torch(ground_truth)))
        self.is_inference = old_is_inference_state
        return dict_to_torch(batch)


def ranks(sequence):
    """ Ranks an numpy.ndarray """
    as_list = list(sequence.cpu().detach().numpy()) if type(sequence) != np.ndarray else list(sequence)
    all_values = sorted(list(set(as_list)))
    mapping = {value: index for index, value in enumerate(all_values)}
    result = np.array([mapping[value] for value in as_list])
    if type(sequence) != np.ndarray:
        result = torch.from_numpy(result)
        result = result.to(sequence.device).to(sequence.dtype)
    return result


''' The following code implements a dataloader that allows to load the dataset partially to save memory space '''


class SplittedDataloader:
    def __init__(self, dataset: Data, dataset_config: dict, dataloader_config, total_parts):
        """
        The splitted dataloader is able to load a dataset without loading it into ram completely. It divides the
        dataset into N subparts and loads only one subpart into memory simultaneously
        """
        self.shuffle = dataloader_config.get('shuffle', True)
        self.dataset = dataset
        self.dataset.create_batches(**dataset_config, part=1, total_parts=1)
        self.num_batches = len(dataset)
        self.dataloader_config = dataloader_config
        self.dataloader_config["collate_fn"] = self.do_nothing
        self.dataset_config = dataset_config
        self.dataset.create_batches(
            **dataset_config, part=1, total_parts=total_parts, shuffle=self.shuffle,
            finisher_function=self.finisher_function
        )
        self.dataloader = DataLoader(self.dataset, **dataloader_config)
        self.dataloader_iterator = iter(self.dataloader)
        self.current_part = 1
        self.total_parts = total_parts

    def __iter__(self):
        return self

    def __next__(self):
        """ Returns the next data item or loads the next dataset part. """
        try:
            element = next(self.dataloader_iterator)
        except StopIteration:
            self.current_part += 1
            if self.current_part > self.total_parts:
                raise StopIteration
            self.dataset.create_batches(
                **self.dataset_config, part=self.current_part, total_parts=self.total_parts, shuffle=self.shuffle,
                finisher_function=self.finisher_function
            )
            self.dataloader = DataLoader(self.dataset, **self.dataloader_config)
            self.dataloader_iterator = iter(self.dataloader)
            element = next(self.dataloader_iterator)
        return element

    def __len__(self):
        return self.num_batches

    @staticmethod
    def do_nothing(batch):
        if len(batch) == 1:
            return batch[0]
        return batch

    @staticmethod
    def default_collate(batch):
        r"""Puts each data field into a tensor with outer dimension batch size"""
        elem = batch[0]
        elem_type = type(elem)
        if isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                numel = sum([x.numel() for x in batch])
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            return torch.stack(batch, 0, out=out)
        elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
            elem = batch[0]
            if elem_type.__name__ == 'ndarray':
                # array of string classes and object
                return SplittedDataloader.default_collate([torch.as_tensor(b) for b in batch])
            elif elem.shape == ():  # scalars
                return torch.as_tensor(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float64)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, string_classes):
            return batch
        elif isinstance(elem, container_abcs.Mapping):
            return {key: SplittedDataloader.default_collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
            return elem_type(*(SplittedDataloader.default_collate(samples) for samples in zip(*batch)))
        elif isinstance(elem, container_abcs.Sequence):
            # check to make sure that the elements in batch have consistent size
            it = iter(batch)
            elem_size = len(next(it))
            if not all(len(elem) == elem_size for elem in it):
                raise RuntimeError('each element in list of batch should be of equal size')
            transposed = zip(*batch)
            return [SplittedDataloader.default_collate(samples) for samples in transposed]

    @staticmethod
    def finisher_function(batch):
        batch = SplittedDataloader.default_collate([batch])
        return batch

    @staticmethod
    def collate_to_lists(batch):
        batch = [SplittedDataloader.default_collate([_]) for _ in batch] if len(batch) > 1 \
            else SplittedDataloader.default_collate(batch)
        return batch