"""
API to the the solver API

Set
    VERBOSE = "1"
to generate tracker outputs.

Set
    SAVE_INTERMEDIATE = "1"
to store intermediate tracking results to the working directory

"""

import time
import multiprocessing as mp
import math
import copy
import numpy as np
import torch.nn as nn

from src.utilities.decorators import simple_multiple_to_numpy_and_back, input_to_numpy
import lpmp_py


VERBOSE = "1"
SAVE_INTERMEDIATE = "0"


class LiftedSolver(nn.Module):
    """
    A wrapper for the solver API
    """
    def __init__(self, lpmp_config, tighten_config, solve_instance_wise=False):
        """

        """
        super(LiftedSolver, self).__init__()
        self.solve_instance_wise = solve_instance_wise
        self.tighten_config = tighten_config
        lpmp_config.update(dict(save_intermediate=0))
        self.paramsMap_eval = {key.upper(): str(value) for key, value in lpmp_config.items()}

        self.instance_solver = InstanceSolver(self.paramsMap_eval, tighten_config)

    def forward(self, costs: dict):
        """
        Calculates a tracking result

        costs = {
            "in_costs": (node_len), "out_costs": (node_len), "node_costs": (node_len), "edge_costs": (edge_len),
        }

        result = {"node_ids": (node_len)}
        """

        in_nodes, out_nodes, activated_nodes, activated_edges, node_ids = self.solve(
            costs['node_frames'], costs["edge_costs"], costs['edge_sources'], costs["edge_sinks"], self.paramsMap_eval,
            self.tighten_config
        )

        result = {
            "in_nodes": in_nodes, "out_nodes": out_nodes, "activated_nodes": activated_nodes, "node_ids": node_ids,
            "activated_edges": activated_edges,
        }

        return result

    @staticmethod
    @simple_multiple_to_numpy_and_back
    def solve(node_frames, edge_costs, edge_in_nodes, edge_out_nodes, paramsMap, tighten_config):
        """ Solves an problem instance """
        # Build solver instance
        solver, instance_params = LiftedSolver.get_solver(paramsMap, tighten_config)
        number_of_nodes = node_frames.size
        # Count frequencies of frames
        _, counts = np.unique(node_frames[0], return_counts=True)
        assert list(node_frames[0]) == sorted(list(node_frames[0]))
        # Define how many nodes are in a frame
        time_frames = lpmp_py.lpd.TimeFramesToVertices()
        time_frames.init_from_vector(counts)
        graph_struct = lpmp_py.lpd.GraphStructure(time_frames)
        # Edge vector. Each edge is represented by a pair of vertices. Vertices ar numbered from zero
        edge_vector = np.stack([edge_in_nodes[0], edge_out_nodes[0]], axis=1)
        graph_struct.add_edges_from_vectors(
            edge_vector.astype(np.int32), edge_costs[0].astype(np.float32), instance_params, 0
        )
        instance = lpmp_py.lpd.LdpInstance(instance_params, graph_struct)
        lpmp_py.lpd.construct(solver, instance)
        solver.solve()
        paths = solver.get_best_primal()
        indicators = LiftedSolver.recover_indicators(paths, number_of_nodes, edge_in_nodes, edge_out_nodes)
        return indicators

    @staticmethod
    def get_solver(paramsMap, tighten_config=None):
        """ Creates a new solver instances for the specified parameters """
        iterations = paramsMap['ITERATIONS']
        reduced_params = {key: value for key, value in paramsMap.items() if key != 'ITERATIONS'}
        instance_params = lpmp_py.lpd.LdpParams(reduced_params)
        solver_params = ["solveFromFiles", "--maxIter", str(iterations), "-v", VERBOSE]
        if tighten_config:
            arg_list = [['--'+name, str(value)] for name, value in tighten_config.items()]
            solver_params.append('--tighten')
            for arg in arg_list:
                solver_params.extend(arg)
        solver = lpmp_py.lpd.Solver(solver_params)
        return solver, instance_params


    @staticmethod
    def recover_indicators(paths, num_nodes, edge_in_nodes, edge_out_nodes):
        """ Recovers the original ndoe indices from the tracking result"""
        in_nodes, out_nodes, activated_nodes, node_ids = \
            np.zeros(num_nodes), np.zeros(num_nodes), np.zeros(num_nodes), -np.ones(num_nodes)
        edge_activation_map = np.zeros((num_nodes, num_nodes))
        for i, t in enumerate(paths):
            in_nodes[t[0]] = 1
            out_nodes[t[-1]] = 1
            activated_nodes[t] = 1
            node_ids[t] = i
            sources, sinks = np.repeat(t, len(t)), np.tile(t, len(t))  # <-- creates rich matrix with higher order edges
            edge_activation_map[sources, sinks] = 1.0
        node_ids = np.expand_dims(node_ids, axis=0)
        in_nodes = np.expand_dims(in_nodes, axis=0)
        out_nodes = np.expand_dims(out_nodes, axis=0)
        activated_nodes = np.expand_dims(activated_nodes, axis=0)
        activated_edges = np.expand_dims(
            edge_activation_map[edge_in_nodes[0].astype(int), edge_out_nodes[0].astype(int)], axis=0)
        return in_nodes, out_nodes, activated_nodes, activated_edges, node_ids


class InstanceSolver:
    def __init__(self, params_map, tighten_config):
        """ Creates an uninitiated instance solver """
        ''' Store solver configs '''
        self.tighten_config = tighten_config
        self.params_map = params_map
        ''' Store solver outputs/inputs for parallel runs '''
        self.parallel_processor = Processor(4)
        self.stored_batches = list()  # list of tuples with (path_number, stored_paths)  # First stage
        self.paths = list()  # list of tuples with (path_number, stored_paths)  # Final merging stage
        self.connections = list()  # list of tuples with (start_path_number, end_path_number, connection)
        ''' Define empty variables necessary for instance tracking '''
        self.time_frames, self.total_nodes, self.total_frames, self.frame_indices, self.counts, self.overlap, \
                self.batch_size, self.time_bounds, self.current_stage, self.current_batch, self.max_frame, \
                self.total_batches, self.node_frames = \
            None, None, None, None, None, None, None, None, None, None, None, None, None
        self.stage1_tasks, self.stage2_tasks = None, None
        self.global_labels = []

    def solve_global(self):
        """
        Waits for all instances to be solved and merges the ids to a global tracking result
        :return:
        """
        ''' Wait for all tasks to be done and append it to the paths/connection list'''
        batches = 0
        while batches < len(self.stage2_tasks):
            print("Wait for stage two task %s of %s to be solved," % (batches + 1, len(self.stage2_tasks)))
            _ = self.stage2_tasks[batches].get()
            batches += 1
            if _[0] == 0:
                self.paths.append((_[0] * 2 + 0, _[1]))
                self.paths.append((_[0] * 2 + 1, _[2]))
                self.paths.append((_[0] * 2 + 2, _[3]))
            else:
                self.paths.append((1 + _[0] * 2, _[2]))
                self.paths.append((1 + _[0] * 2 + 1, _[3]))
            self.connections.append((_[0] * 2, _[4]))
            self.connections.append((_[0] * 2 + 1, _[5]))
        ''' Sort paths and connections '''
        _final_paths, _final_connections = list(), list()
        for p in self.paths:
            merged = False
            for i, ref_p in enumerate(_final_paths):
                if ref_p[0] > p[0]:
                    _final_paths.insert(i, p)
                    merged = True
                    break
            if not merged:
                _final_paths.append(p)
        for c in self.connections:
            merged = False
            for i, ref_c in enumerate(_final_connections):
                if ref_c[0] > c[0]:
                    _final_connections.insert(i, c)
                    merged = True
                    break
            if not merged:
                _final_connections.append(c)
        ''' Merge labels of all paths '''
        label_assignment = lpmp_py.lpd.LabelAssignment()
        label_assignment.init(_final_paths[0][1])  # Init is called with the paths extracted from the first interval
        for p, c in zip(_final_paths[1:], _final_connections):
            label_assignment.update(p[1], c[1])
        self.global_labels = label_assignment.get_labels()

    @property
    def final_solution(self):
        """ Returns the solution as numpy array """
        ids = -np.ones_like(self.node_frames, dtype=int)
        for _ in self.global_labels:
            ids[int(_[0])] = int(_[1])
        return ids

    def init_new_global_instance(self, batch_size: int, node_frames: np.ndarray):
        """
        Inititates solver parameter needed to solve the sequence instance-wise.
        :param batch_size: The size of an batch of frames that is solved as one instance (in frames).
        :param node_frames: ndarray with the number of the frame for each node.
        :return:
        """
        self.node_frames, self.batch_size = np.copy(node_frames), batch_size
        self.time_frames, self.total_nodes, self.total_frames, self.frame_indices, self.counts, self.max_frame = \
            self.init_time_frames(node_frames)
        self.global_labels = []
        self.time_bounds = (1, batch_size)
        self.total_batches = math.ceil(self.time_frames.get_max_time() / batch_size)
        self.overlap = int(self.params_map["MAX_TIMEGAP_LIFTED"])
        self.current_stage, self.current_batch = "batches", 0
        self.stored_batches, self.paths, self.connections = list(), list(), list()
        self.stage1_tasks, self.stage2_tasks = list(), list()

    @staticmethod
    def init_time_frames(node_frames):
        """
        Creates a lpmp_py.lpd.TimeFramesToVertices() object
        :param node_frames: A np.ndarray with the frame number for all nodes.
        :return:
        """
        # Count frequencies of frames
        _, counts = np.unique(node_frames, return_counts=True)
        assert list(node_frames) == sorted(list(node_frames))
        # Define how many nodes are in a frame
        total_nodes = len(node_frames)
        time_frames = lpmp_py.lpd.TimeFramesToVertices()
        time_frames.init_from_vector(counts)
        return time_frames, total_nodes, len(counts), _, counts, int(np.max(_))

    def init_connection_stage(self):
        """
        If all instances are processed, this initiates the connection state to connect the instances.
        """
        self.current_batch, self.current_stage = 0, "connections"
        self.time_bounds = \
            max(0, self.batch_size - (2 * self.overlap)),\
            min(self.batch_size + self.overlap * 2, self.time_frames.get_max_time())

    def update_params_map(self, params: dict):
        """ Updates the parameter map with new values """
        for param, value in params.items():
            assert param in self.params_map.keys(), "Cant update parameter %s because it is not existing!" % param
            self.params_map[param] = value

    def get_solver_params_and_instance_params(self):
        """ Creates solver and instace parameter """
        instance_params = lpmp_py.lpd.LdpParams(self.params_map)
        solver_parameters = ["solveFromFiles", "--maxIter", str(self.params_map["ITERATIONS"]), "-v", VERBOSE]
        if self.tighten_config:
            arg_list = [['--' + name, str(value)] for name, value in self.tighten_config.items()]
            solver_parameters.append('--tighten')
            for arg in arg_list:
                solver_parameters.extend(arg)
        return solver_parameters, instance_params

    def prepare_time_frames_set(self, start_frame, end_frame):
        """
        Returns a set of time frames.
        :param start_frame: The start frame
        :param end_frame: The end frame
        :return:
            - A lpmp_py.lpd.TimeFramesToVertices() object
            - A list with number of nodes per frame
            - The ids of the nodes in the frame (global instance IDs)
        """
        node_ids = 0
        frames, counts = list(), list()
        for frame, count in zip(copy.copy(self.frame_indices), copy.copy(self.counts)):
            if start_frame <= frame <= end_frame:
                frames.append(frame)
                counts.append(count)
            elif start_frame > frame:
                node_ids += count
        time_frames = lpmp_py.lpd.TimeFramesToVertices()
        time_frames.init_from_vector(counts)
        return time_frames, counts, node_ids

    @staticmethod
    @input_to_numpy
    def extract_costs(edge_sources, edge_sinks, edge_costs, **kwargs):
        """ Creates a edge cost vector in the correct format """
        edge_vector = np.stack([edge_sources[0], edge_sinks[0]], axis=1)
        return np.copy(edge_vector), np.copy(edge_costs[0])

    def process_next_batch(self, costs):
        """ Solves the next batch of the sequence """
        print("Process batch in stage", self.current_stage, "and call with time bounds",  self.time_bounds)
        edges, edge_costs = self.extract_costs(**costs)
        start, end = self.time_bounds
        solver_params, instance_params = self.get_solver_params_and_instance_params()

        if self.current_stage == "batches":
            ''' Process the next instance if the solver is in the first stage '''
            if self.current_batch >= self.total_batches:
                return
            time_frames, time_frames_counts, node_index = self.prepare_time_frames_set(start, end)
            args = [
                solver_params, self.params_map, edges, edge_costs, time_frames_counts, self.current_batch, node_index,
            ]
            self.stage1_tasks.append(self.parallel_processor.run_batch(*args))
            new_start = end + 1
            new_end = min(self.time_frames.get_max_time(), new_start + self.batch_size - 1)
            self.time_bounds = new_start, new_end

        elif self.current_stage == "connections":
            ''' Process the next connection if the solver is in the second stage '''
            if self.current_batch + 1 >= self.total_batches:
                return
            batch1, batch2 = None, None
            it = 0
            while batch1 is None or batch2 is None:
                for b in self.stored_batches:
                    if type(b) != tuple:
                        continue
                    if batch1 is None and b[0] == self.current_batch:
                        batch1 = b
                    if batch2 is None and b[0] == self.current_batch + 1:
                        batch2 = b
                if batch1 is not None and batch2 is not None:
                    break
                if len(self.stage1_tasks) > it:
                    if type(self.stage1_tasks[it]) != tuple:
                        batch = self.stage1_tasks[it].get()
                        self.stage1_tasks[it] = batch
                    batch = self.stage1_tasks[it]
                    it += 1
                    self.stored_batches.append(batch)
                else:
                    time.sleep(5)
            time_frames1, time_frames2 = copy.copy(batch1[3]), copy.copy(batch2[3])
            min_vertex_id1, min_vertex_id2 = batch1[2], batch2[2]
            paths1, paths2 = copy.copy(batch1[1]), copy.copy(batch2[1])
            batch_id1, batch_id2 = batch1[0], batch2[0]
            is_first = True if batch_id1 == 0 else False
            is_last = True if (batch_id2 + 1) * self.batch_size >= self.time_frames.get_max_time() else False
            args = [
                solver_params, self.params_map, edges, edge_costs, time_frames1, time_frames2, self.current_batch,
                is_first, is_last, self.overlap, min_vertex_id1, min_vertex_id2, paths1, paths2
            ]
            self.stage2_tasks.append(self.parallel_processor.run_connection(*args))
            new_start = start + self.batch_size
            new_end = min(self.time_frames.get_max_time(), end + self.batch_size)
            self.time_bounds = new_start, new_end
        self.current_batch += 1


"""
Parallel Implementation

Please integrate your own parallel processor into "Processor" class to use real parallelization with your cluster
"""

class Processor:
    """
    A parallel processor for solving multiple instances in parallel. If you want to implement your own Processor
    make sure that a queue is returned by run_batch() and run_connection()
    """

    """ MODIFY FROM HERE BELOW TO INTEGRATE YOUR CLUSTER """
    def __init__(self, num_processes):
        self.num_processes = num_processes
        self.processes = list()
        self.manager = mp.Manager()

    def create_process(self, f, *args):
        while len(self.processes) >= self.num_processes:
            time.sleep(1)
            for p in self.processes:
                if p.is_alive() is False:
                    self.processes.remove(p)
                    break
        q = self.manager.Queue()
        p = mp.Process(target=self.put_to_queue, args=tuple((q, f) + args), daemon=True)
        p.start()
        self.processes.append(p)
        return p, q

    @staticmethod
    def put_to_queue(q, f, *args):
        q.put(f(*args))

    def run_batch(self, *args):
        p, q = self.create_process(self._run_batch, *args)
        return q

    def run_connection(self, *args):
        p, q = self.create_process(self._run_connection, *args)
        return q

    """ MODIFY ABOVE TO INTEGRATE YOUR CLUSTER """

    @staticmethod
    def _run_batch(solver_params, instance_params, edges, edge_costs, time_frames, batch_id, node_index):
        """
        Solves an instance

        DO NOT MODIFY THIS!
        """
        instance_params = lpmp_py.lpd.LdpParams(instance_params)
        _time_frames = lpmp_py.lpd.TimeFramesToVertices()
        _time_frames.init_from_vector(time_frames)
        solver_params[4] = VERBOSE  # Verbosity
        solver = lpmp_py.lpd.Solver(solver_params)
        complete_graph_structure = lpmp_py.lpd.GraphStructure(_time_frames)
        minimal_vertex_id = node_index
        complete_graph_structure.add_edges_from_vectors(edges, edge_costs, instance_params, minimal_vertex_id)
        instance = lpmp_py.lpd.LdpInstance(instance_params, complete_graph_structure)
        lpmp_py.lpd.construct(solver, instance)
        solver.solve()
        paths = solver.get_best_primal()
        return batch_id, paths, minimal_vertex_id, time_frames

    @staticmethod
    def _run_connection(
            solver_params, instance_params, edges, edge_costs, time_frames1, time_frames2, batch_id, is_first, is_last,
            max_lifted, min_vertex_id1, min_vertex_id2, paths1, paths2
    ):
        """
        Solves a connection between two instances

        DO NOT MODIFY THIS!
        """
        _time_frames1 = lpmp_py.lpd.TimeFramesToVertices()
        _time_frames1.init_from_vector(time_frames1)
        _time_frames2 = lpmp_py.lpd.TimeFramesToVertices()
        _time_frames2.init_from_vector(time_frames2)
        cut_off1, cut_off2 =\
            min(max_lifted, max(2, len(time_frames1) - 20)), max(2, min(max_lifted, len(time_frames2) - 20))
        instance_params = lpmp_py.lpd.LdpParams(instance_params)
        solver_params[4] = VERBOSE # Verbosity
        solver = lpmp_py.lpd.Solver(solver_params)
        paths_extractor1 = lpmp_py.lpd.PathsExtractor(_time_frames1, paths1, cut_off1, is_first, False, min_vertex_id1)
        paths_extractor2 = lpmp_py.lpd.PathsExtractor(_time_frames2, paths2, cut_off2, False, is_last, min_vertex_id2)
        interval_connection = lpmp_py.lpd.IntervalConnection(paths_extractor1, paths_extractor2)
        interval_connection.init_edges_from_vectors(edges, edge_costs)
        instance = lpmp_py.lpd.LdpInstance(instance_params, interval_connection)
        lpmp_py.lpd.construct(solver, instance)
        solver.solve()
        connected_paths = solver.get_best_primal()
        interval_connection.create_result_structures(connected_paths)
        middle_paths = interval_connection.get_middle_paths()
        first_to_middle = interval_connection.get_first_to_middle()
        middle_to_second = interval_connection.get_middle_to_second()
        extracted_paths1 = paths_extractor1.get_extracted_paths()
        extracted_paths2 = paths_extractor2.get_extracted_paths()
        return batch_id, extracted_paths1, middle_paths, extracted_paths2, first_to_middle, middle_to_second







