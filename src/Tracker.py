"""
Implements pipelines to track a sequence: Obtain costs, solve the instance (global or instance wise)
"""

import torch
import numpy as np
from scipy.sparse import csc_matrix
from tqdm import tqdm
import math
import os

from src.TrackingModel import TrackingModel
from src.datasets import Data, SplittedDataloader
from src.utilities.conversions import to_numpy


''' Cost update functions '''


def temporal_decay(delta_time):
    """ creates  a temporal decay factor based on the temporal distance """
    return 1 / (10 * delta_time.clamp(0, 2) + 0.1)


def induce_soft_constraints(data, result):
    """
    Induces Soft-constraints by adding High cost value to hingh confident edges.
    """
    if "high_confident" in data["edges"].keys():
        high_confidention_cost = -1000
        result['edge_costs'] = result['edge_costs'] + data["edges"]["high_confident"] * high_confidention_cost
    return result


''' The tracker class to solve instances '''


class Tracker:
    node_cost_keys = ['out_costs', 'in_costs', 'node_costs']
    dataloader_cfg = dict(shuffle=False, num_workers=0, pin_memory=False, batch_size=1)

    @staticmethod
    def track(model: TrackingModel, dataset: Data):
        if not model.solver.solve_instance_wise:
            return Tracker.track_global(model, dataset)
        else:
            return Tracker.track_instance_wise(model, dataset)

    @staticmethod
    def track_global(model: TrackingModel, dataset: Data):
        """
        This function infers and associates the data set with a given model

        :param model: The model to evaluate
        :param dataset: The dataset of class data. BE SURE THAT ONLY ONE SEQUENCE IS LOADED!

        :return Dictionariers with numpy arrays
        """
        model.eval()
        seq = dataset.sequences_for_inference[0]
        ''' Create global graph for the sequence'''
        full_graph_data = dataset.return_batch_with_full_graph(seq)
        number_of_nodes = full_graph_data["nodes"]["frame"].shape[0]
        node_row = full_graph_data["nodes"]["row"].cpu().numpy().astype(int)
        node_row_to_index_mapping = np.ones(np.max(node_row) + 1)
        node_row_to_index_mapping[node_row] = np.arange(node_row.shape[0])
        node_row_to_index_mapping = node_row_to_index_mapping.astype(int)
        ''' Create edge cost and node cost container '''
        node_costs = dict()
        for key in Tracker.node_cost_keys:
            node_costs[key] = torch.zeros_like(full_graph_data["nodes"]["id"][None, :])
        edge_cost_matrix = csc_matrix((number_of_nodes, number_of_nodes), dtype=np.float32)
        edge_calculations = csc_matrix((number_of_nodes, number_of_nodes), dtype=np.int16)
        node_calculations = np.zeros(number_of_nodes, dtype=np.int16)
        ''' Iterate over dataset and fill cost container with cost values '''
        dataset_cfg = dict(sequences=[seq], is_inference=True)
        dataloader = SplittedDataloader(dataset, dataset_cfg, Tracker.dataloader_cfg, total_parts=25)
        with torch.no_grad():
            progress_bar = tqdm(iter(dataloader), desc="Track sequence with global graph")
            for data in progress_bar:
                if data["edges"]["sink"].numel() == 0:
                    continue
                result = model.calculate_costs(data['nodes'], data['edges'])
                result = induce_soft_constraints(data, result)
                _rows, _sources, _sinks = \
                    data["nodes"]["row"].cpu().numpy().astype(int)[0], \
                    data["edges"]["source"].cpu().numpy().astype(int)[0], \
                    data["edges"]["sink"].cpu().numpy().astype(int)[0]
                # Map detection_rows, sinks and sources to indices of the global graph
                detection_indices = node_row_to_index_mapping[_rows]
                _sources, _sinks = detection_indices[_sources], detection_indices[_sinks]
                node_calculations[detection_indices] += 1
                for key in Tracker.node_cost_keys:
                    node_costs[key][0, detection_indices] += result[key][0]
                edge_calculations[_sources, _sinks] += 1
                edge_cost_matrix[_sources, _sinks] += result["edge_costs"][0].numpy().astype(np.float32)
        ''' Convert aggregated edge costs to solver format '''
        edge_counter = edge_calculations[
             full_graph_data["edges"]["source"].numpy().astype(int),
             full_graph_data["edges"]["sink"].numpy().astype(int)]
        global_edge_costs = edge_cost_matrix[
            full_graph_data["edges"]["source"].numpy().astype(int),
            full_graph_data["edges"]["sink"].numpy().astype(int)]
        global_edge_costs = global_edge_costs / np.maximum(1, edge_counter)
        node_calculations = torch.from_numpy(node_calculations[None, :])
        for key in Tracker.node_cost_keys:
            node_costs[key] /= node_calculations.clamp(1, 10000)
        costs = dict(
            node_frames=full_graph_data["nodes"]["frame"][None, :], node_costs=node_costs['node_costs'],
            edge_sources=full_graph_data["edges"]["source"][None, :], out_costs=node_costs['out_costs'],
            edge_sinks=full_graph_data["edges"]["sink"][None, :], edge_costs=torch.from_numpy(global_edge_costs),
            in_costs=node_costs['in_costs'],
        )
        ''' Weight costs with the time '''
        delta_time = \
            (costs['node_frames'][0][costs['edge_sinks']] - costs['node_frames'][0][costs['edge_sources']]).float() / \
            seq["fps"]
        weight = temporal_decay(delta_time)
        costs['edge_costs'] = costs['edge_costs'][0] * weight
        ''' Solve global instance and return full graph data '''
        with torch.no_grad():
            result = model.run_solver(costs=costs)
        full_graph_data["prediction"] = result
        full_graph_data["edges"]["costs"] = costs['edge_costs']
        full_graph_data = to_numpy(full_graph_data)
        return full_graph_data

    @staticmethod
    def track_instance_wise(model: TrackingModel, dataset: Data):
        """ Tracks a sequence splitted into instances """

        solver = model.solver.instance_solver
        ''' Create dataset specific values '''
        seq = dataset.sequences_for_inference[0]
        dataset_cfg = dict(sequences=[seq], is_inference=True)
        full_graph_data = dataset.return_batch_with_full_graph(seq, return_edges=False)
        number_of_nodes = full_graph_data["nodes"]["frame"].shape[0]
        fps = seq["fps"]
        datase_name = os.getenv("DATASET", "MOT17")
        batchsize = 3 * 50 if datase_name == "MOT20" else 3 * 60
        node_row = full_graph_data["nodes"]["row"].cpu().numpy().astype(int)
        node_row_to_index_mapping = np.ones(np.max(node_row) + 1)
        node_row_to_index_mapping[node_row] = np.arange(node_row.shape[0])
        node_row_to_index_mapping = node_row_to_index_mapping.astype(int)
        ''' Update solver parameter for "irregular" videos with different framerate than 30 '''
        if datase_name == "MOT17" and fps != 30:
            new_len = str(int(math.floor(2 * fps)))
            params = {"MAX_TIMEGAP_BASE": new_len, "MAX_TIMEGAP_LIFTED": new_len, "MAX_TIMEGAP_COMPLETE": new_len}
            model.solver.batched_solver.update_params_map(params)

        def init_tracker_container():
            """ Create data containers required for a tracking run """
            node_costs = dict()
            for key in Tracker.node_cost_keys:
                node_costs[key] = torch.zeros_like(full_graph_data["nodes"]["id"][None, :])
            dataloader = SplittedDataloader(dataset, dataset_cfg, Tracker.dataloader_cfg, total_parts=50)
            return dataloader, node_costs

        def prepare_local_instance(
                edge_calculations, edge_cost_matrix, node_calculations, node_costs,
                first_frame, last_frame
        ):
            """ Converts the sparse global graph to a local instance """
            source, sink = edge_calculations.nonzero()
            frames = full_graph_data["nodes"]["frame"].numpy()
            if last_frame is not None:
                valid = (frames[source] <= last_frame) * (frames[sink] <= last_frame)
                source, sink = source[valid], sink[valid]
            if first_frame is not None:
                valid = (frames[source] >= first_frame) * (frames[sink] >= first_frame)
                source, sink = source[valid], sink[valid]
            edge_counter = edge_calculations[source, sink]
            global_edge_costs = edge_cost_matrix[source, sink]
            global_edge_costs = global_edge_costs / edge_counter
            node_calculations = torch.from_numpy(node_calculations[None, :])
            for key in node_costs:
                node_costs[key] = node_costs[key] / node_calculations.float().clamp(1, 10000)
            # Convert to cost tensor
            costs = dict(
                node_frames=full_graph_data["nodes"]["frame"][None, :], edge_sources=torch.from_numpy(source)[None, :],
                edge_sinks=torch.from_numpy(sink)[None, :], edge_costs=torch.from_numpy(global_edge_costs),
                in_costs=node_costs['in_costs'], out_costs=node_costs['out_costs'], node_costs=node_costs['node_costs']
            )
            return costs

        def delete_old_nodes_and_edges(
                edge_calculations, edge_cost_matrix, node_calculations, node_costs, min_frame
        ):
            """ Removes entries from the sparse matrix for frames smaller than the current minimal frame"""
            frames_to_be_removed = np.where(full_graph_data["nodes"]["frame"] < min_frame)[0]
            edge_calculations[edge_calculations[frames_to_be_removed, :].nonzero()] = 0
            edge_calculations[edge_calculations[:, frames_to_be_removed].nonzero()] = 0
            edge_cost_matrix[edge_cost_matrix[frames_to_be_removed, :].nonzero()] = 0
            edge_cost_matrix[edge_cost_matrix[:, frames_to_be_removed].nonzero()] = 0
            edge_cost_matrix.eliminate_zeros()
            edge_calculations.eliminate_zeros()
            node_calculations[frames_to_be_removed] = 0
            for key in node_costs.keys():
                node_costs[key][0, frames_to_be_removed] = 0
            return edge_calculations, edge_cost_matrix, node_calculations, node_costs

        def iterate_through_dataset(node_costs):
            """ Iterates over the sequence and solves batches"""
            ''' Create empty data container to accumulate costs '''
            edge_cost_matrix, edge_calculations, node_calculations = \
                csc_matrix((number_of_nodes, number_of_nodes), dtype=np.float32), \
                csc_matrix((number_of_nodes, number_of_nodes), dtype=np.int16), \
                np.zeros(number_of_nodes, dtype=np.int16)
            data_stack = list()
            ''' Iterate over sequence and calculate all edges '''
            progress_bar = tqdm(iter(dataloader), desc="Track sequence batchwise graph")
            with torch.no_grad():
                for datas in progress_bar:
                    datas = [datas] if type(datas) != list else datas
                    for data in datas:
                        if data["edges"]["sink"].numel() == 0:
                            continue
                        l_bound, u_bound = solver.time_bounds[0], solver.time_bounds[1]
                        ''' Do inference for current batch'''
                        result = model.calculate_costs(data['nodes'], data['edges'])
                        result = induce_soft_constraints(data, result)
                        min_frame, max_frame = data["nodes"]["frame"].min().item(), data["nodes"]["frame"].max().item()
                        if max_frame < l_bound:
                            continue
                        ''' Add calculated node and edge costs to accumulator '''
                        _rows, _sources, _sinks = \
                            data["nodes"]["row"].cpu().numpy().astype(int)[0], \
                            data["edges"]["source"].cpu().numpy().astype(int)[0], \
                            data["edges"]["sink"].cpu().numpy().astype(int)[0]
                        # Map detection_rows, sinks and sources to indices of the global graph
                        detection_indices = node_row_to_index_mapping[_rows]
                        _sources, _sinks = detection_indices[_sources], detection_indices[_sinks]
                        node_calculations[detection_indices] += 1
                        # Weight costs with time
                        delta_time = data["edges"]["delta_t"]
                        delta_time = delta_time.float()
                        weight = temporal_decay(delta_time)
                        result['edge_costs'][0] = result['edge_costs'][0] * weight
                        for key in Tracker.node_cost_keys:
                            node_costs[key][0, detection_indices] += result[key][0]
                        # Aggregate some data, cause updateing the sparse matrix ist slow
                        _ = result["edge_costs"][0].numpy().astype(np.float32)
                        data_stack.append([_sources, _sinks, _])
                        ''' If all frames for the current batch are processed: Merge data and solve graph '''
                        solve = min_frame >= solver.time_bounds[1]
                        if solve:
                            ''' Update sparse matrix with collected data '''
                            _sources = np.concatenate([_[0] for _ in data_stack])
                            _sinks = np.concatenate([_[1] for _ in data_stack])
                            _data = np.concatenate([_[2] for _ in data_stack])
                            edge_cost_matrix[_sources, _sinks] += _data
                            edge_calculations[_sources, _sinks] += 1
                            data_stack = list()
                            ''' Solve graph '''
                            costs = prepare_local_instance(
                                edge_calculations, edge_cost_matrix, node_calculations, node_costs, l_bound, u_bound)
                            solver.process_next_batch(costs)
                            updated_sparse = delete_old_nodes_and_edges(
                                edge_calculations, edge_cost_matrix, node_calculations, node_costs, min_frame=l_bound)
                            edge_calculations, edge_cost_matrix, node_calculations, node_costs = updated_sparse
            ''' Solve the last batch if ot already done '''
            if len(data_stack) > 0:
                _sources, _sinks, _data = \
                    np.concatenate([_[0] for _ in data_stack]), np.concatenate([_[1] for _ in data_stack]), \
                    np.concatenate([_[2] for _ in data_stack])
                edge_cost_matrix[_sources, _sinks] += _data
                edge_calculations[_sources, _sinks] += 1
                costs = prepare_local_instance(
                    edge_calculations, edge_cost_matrix, node_calculations, node_costs, l_bound, u_bound)
                solver.process_next_batch(costs)

        ''' First stage: Solve sequence instance wise'''
        dataloader, node_costs = init_tracker_container()
        solver.init_new_global_instance(batch_size=batchsize, node_frames=full_graph_data["nodes"]["frame"])
        iterate_through_dataset(node_costs)
        ''' Second stage: Connect instances '''
        solver.init_connection_stage()
        dataloader, node_costs = init_tracker_container()
        iterate_through_dataset(node_costs)
        ''' Merge results and return the data '''
        solver.solve_global()
        result = {"node_ids": solver.final_solution}
        full_graph_data["prediction"] = result
        full_graph_data = to_numpy(full_graph_data)
        return full_graph_data

