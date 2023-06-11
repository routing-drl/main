from environment.util import create_graph, get_new_route, compute_reward, get_max_neighbors
from helper.graph import get_neighbors
from gym.spaces import MultiDiscrete, Discrete
from networkx import Graph
import gym
import pandas as pd
from typing import Tuple
from copy import deepcopy


class Env(gym.Env):
    # Constructor, create graphs, set some variables for gym, house keeping stuff
    def __init__(self, save_file: str, graph: Graph = None) -> None:

        # Create our graphs, each with a unique set of edge weights
        if graph is None:
            self.graph: Graph = create_graph()
        else:
            self.graph = deepcopy(graph)

        # Declare the spaces for the Env
        self.max_neighbors = get_max_neighbors(self.graph)

        self.observation_space: MultiDiscrete = MultiDiscrete(
            [self.num_nodes(), self.num_nodes()])
        self.action_space: Discrete = Discrete(self.max_neighbors)
        self.valid_actions = [1]*self.max_neighbors

        # Counters
        self.steps: int = 0
        self.hops: int = 0

        # Log dir
        self.save_file: str = save_file
        # self.test_file: str = save_file[:save_file.index(".csv")]+"_test.csv"
        # print(self.test_file)

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

        f = open("training_data/step_data.csv", "w+")
        f.write("steps,reward\n")
        f.close()

        # f = open(self.test_file, 'w+')
        # f.write("steps,reward,latency,bandwidth\n")
        # f.close()

        self.df = pd.DataFrame(
            columns=['steps', 'reward'])
        self.dict_list = []
        # self.writer = SummaryWriter()

    # Preform the action and compute reward

    def step(self, action: int, train_mode=True):
        rewards = []
        try:
            next_node = self.neighbors[action]
        except:
            # print("error")
            self.neighbors = list(self.graph.neighbors(self.current_node))
            for i in range(len(self.valid_actions)):
                if i < len(self.neighbors):
                    self.valid_actions[i] = 1
                else:
                    self.valid_actions[i] = 0
            return [self.current_node, self.target], -1, False, {'valid_actions': self.valid_actions}
        self.path.append(next_node)
        self.current_node = next_node
        self.steps += 1
        rewards, done = self._get_reward()
        self.episode_reward += round(rewards[0], ndigits=3)
        self.record_data(rewards, done, train_mode)

        self.neighbors = list(self.graph.neighbors(self.current_node))
        self.update_valid_actions()

        return [self.current_node, self.target], rewards[0], done, {'valid_actions': self.valid_actions}

    def update_valid_actions(self):
        for i in range(len(self.valid_actions)):
            if i < len(self.neighbors):
                self.valid_actions[i] = 1
            else:
                self.valid_actions[i] = 0

    def record_data(self, rewards, done, train_mode):
        if done:
            self.eps += 1
            f_name = self.save_file if train_mode else self.test_file

            with open(f_name, 'a') as fd:
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
        self.source, self.target = get_new_route(self.graph)
        self.current_node = self.source
        self.path = []
        self.neighbors = get_neighbors(self.graph, self.current_node)
        self.path.append(self.source)
        self.hops = 0
        self.episode_reward = 0
        self.neighbors = list(self.graph.neighbors(self.current_node))
        for i in range(len(self.valid_actions)):
            if i < len(self.neighbors):
                self.valid_actions[i] = 1
            else:
                self.valid_actions[i] = 0

        return self.source, self.target, {'valid_actions': self.valid_actions}

    # not used / doesn't make sense to use given the problem
    def _render(self, mode: str = 'human', close: bool = False) -> None:
        pass

    # compute the reward
    def _get_reward(self) -> Tuple[list, bool]:
        return compute_reward(self.graph, self.target, tuple(self.path))

    def num_nodes(self) -> int:
        return len(self.graph.nodes)
