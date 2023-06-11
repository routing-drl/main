import torch.nn.functional as F
from environment.util import  get_flows, get_new_route, compute_reward, get_max_neighbors, adjust_lat_band
from helper.graph import get_neighbors
from gym.spaces import MultiDiscrete, Discrete
from networkx import barabasi_albert_graph
import networkx as nx
import gym
import pandas as pd
from typing import Tuple
import torch
import random
from config import device
from torch_geometric.data import Data
import numpy as np

class Env(gym.Env):
    # Constructor, create graphs, set some variables for gym, house keeping stuff
    def __init__(self, save_file: str, num_nodes_in_graph: int = 5, max_neighbors=5, graph=None) -> None:

        self.graph = graph if graph is not None else self.generate_graph()
        self.num_nodes_in_graph = num_nodes_in_graph
        self.max_neighbors = get_max_neighbors(self.graph)

        # self.neighbor_dict = self.get_neighbor_list()
        self.removed_nodes = []
        self.edge_list, self.edge_weights = self.get_edge_list()

        # Declare the spaces for the Env
        # self.max_neighbors = 5

        self.observation_space: MultiDiscrete = MultiDiscrete(
            [self.num_nodes(), self.num_nodes()])
        self.action_space: Discrete = Discrete(self.num_nodes_in_graph)

        # Counters
        self.steps: int = 0
        self.hops: int = 0

        # Log dir
        self.save_file: str = save_file

        # Path information
        self.source: int = -1
        self.target: int = -1
        self.current_node: int = -1
        self.path: list = []
        self.neighbors = []
        self.eps = 0

        self.episode_reward = 0
        self.latency_reward = 0
        self.bandwidth_reward = 0

        f = open(self.save_file, "w+")
        f.write("steps,reward,latency,bandwidth\n")
        f.close()

#         f = open("training_data/step_data.csv", "w+")
#         f.write("steps,reward\n")
#         f.close()

        self.df = pd.DataFrame(
            columns=['steps', 'reward'])
        self.dict_list = []
        # self.writer = SummaryWriter()
        self.latency_record = []

    # Perform the action and compute reward

    def step(self, action: int):
        rewards = []
        try:
            next_node = action
        except:
            self.neighbors = list(self.graph.neighbors(self.current_node))
            for i in range(len(self.valid_actions)):
                if i < len(self.neighbors):
                    self.valid_actions[i] = 1
                else:
                    self.valid_actions[i] = 0
            return [self.current_node, self.target, self.get_node_data(), self.get_edge_list()], -12345, False, {'valid_actions': self.valid_actions}
        self.path.append(next_node)
        self.current_node = next_node
        self.steps += 1
        rewards, done = self._get_reward()
        self.episode_reward += round(rewards[0], ndigits=3)
        self.record_data(rewards, done)

        self.neighbors = list(self.graph.neighbors(self.current_node))
        valid_actions = self.get_valid_actions()
        node_data = self.format_src_tgt(self.current_node, self.target)
        return Data(x=node_data, edge_index=self.edge_list, edge_attr=self.edge_weights), rewards[0], done, {'valid_actions': valid_actions}

    def get_valid_actions(self):
        valid_actions = torch.zeros(
            (1, self.num_nodes_in_graph), dtype=torch.bool, device=device)
        neighbors = list(nx.neighbors(self.graph, self.current_node))
        valid_actions[0][neighbors] = True
        return valid_actions

    def format_src_tgt(self, src, tgt):
        src_tgt_mat = torch.zeros(
            size=(self.num_nodes_in_graph, 3), device=device)
        src_tgt_mat[src][0] = 1  # src
        src_tgt_mat[tgt][1] = 1  # tgt
        for neighbor in self.neighbors:
            src_tgt_mat[neighbor][2] = 1
        return src_tgt_mat

    def record_data(self, rewards, done):
        if done:
            self.eps += 1
            self.latency_record.append(rewards[1])

            with open(self.save_file, 'a') as fd:
                fd.write(str(self.steps))
                fd.write(',' + str(round(self.episode_reward, ndigits=3)))
                fd.write(',' + str(round(rewards[1], ndigits=3)))
                fd.write(',' + str(round(rewards[2], ndigits=8)))
                fd.write('\n')
                fd.close()

    # Called when an environment is finished, creates a new "environment"
    def reset(self) -> Tuple[int, int]:
        return self._reset()

    # Reset counters to zero, get an observation store it and return it
    def _reset(self) -> Tuple[int, int]:
        # self.graph = self.generate_graph()
        # if self.eps % 50 == 0:
        #     self.update_graph_weights()
        self.source, self.target = get_new_route(self.graph)
        self.current_node = self.source
        self.path = []
        self.neighbors = get_neighbors(self.graph, self.current_node)
        self.path.append(self.source)
        self.hops = 0
        self.episode_reward = 0
        self.neighbors = list(self.graph.neighbors(self.current_node))

        valid_actions = self.get_valid_actions()
        # edge_list, edge_weights = self.get_edge_list()

        if self.eps % 50 == 0:
            latency_record = np.asarray(self.latency_record)
            print("latency_data", latency_record.mean(),
                  np.median(latency_record))
            self.latency_record = []
        node_data = self.format_src_tgt(self.current_node, self.target)

        return Data(x=node_data, edge_index=self.edge_list, edge_attr=self.edge_weights), {'valid_actions': valid_actions}

    # not used / doesn't make sense to use given the problem
    def _render(self, mode: str = 'human', close: bool = False) -> None:
        pass

    # compute the reward
    def _get_reward(self) -> Tuple[list, bool]:
        return compute_reward(self.graph, self.target, tuple(self.path))

    def num_nodes(self) -> int:
        return len(self.graph.nodes)

    def generate_graph(self):
        self.graph = barabasi_albert_graph(self.num_nodes_in_graph, 2)
        while get_max_neighbors(self.graph) > self.max_neighbors:
            self.graph = barabasi_albert_graph(self.num_nodes_in_graph, 2)

        self.update_graph_weights()
        return self.graph

    def update_graph_weights(self):
        for e in self.graph.edges():
            self.graph[e[0]][e[1]]["weight"] = random.uniform(0, 1)
            self.graph[e[0]][e[1]]["capacity"] = random.uniform(0, 1)
        adjust_lat_band(self.graph, get_flows(self.graph, 20))

    def get_edge_list(self):
        edge_list = torch.zeros(size=(self.graph.number_of_edges(), 2),
                                dtype=torch.int64, device=device)
        edge_list = edge_list.transpose(0, 1)
        raw_edge_list = list(self.graph.edges(data=True))
        edge_weights = []
        for i in range(self.graph.number_of_edges()):
            edge_list[0][i] = raw_edge_list[i][0]
            edge_list[1][i] = raw_edge_list[i][1]
            edge_weights.append(raw_edge_list[i][2]['weight'])

        return edge_list, torch.as_tensor(edge_weights, device=device).reshape(-1, 1)

    def update_edges_weights(self):
        self.edge_list, self.edge_weights = self.get_edge_list()

    def remove_graph_nodes(self, nodes: list):
        self.graph.remove_nodes_from(nodes)
        self.edge_list, self.edge_weights = self.get_edge_list()

    def get_neighbor_list(self):
        neighbor_dict = {}
        for node in self.graph.nodes():
            neighbor_dict[node] = nx.neighbors(self.graph, node)
        return neighbor_dict


