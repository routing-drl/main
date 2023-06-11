from tqdm import  tqdm
from itertools import count
import random
from torch_geometric.nn import  GATConv
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch import nn
import torch
import math
from models.DQN import GAMMA, Agent, Transition, ReplayMemory
from torch.optim.lr_scheduler import StepLR
from config import device
import pandas as pd
TARGET_UPDATE = 14_000
EPS_DECAY = 17_000  # larger means slower decay
EPS_START = .95
BATCH_SIZE = 128
EPS_END = 0.001
GAMMA = 0.99


class GCN(nn.Module):
    def __init__(self, num_nodes, out):
        super().__init__()
        # graph
        conv1_out = num_nodes//3
        conv2_out = num_nodes//2

        # num_nodes = 5
        self.conv1 = GATConv(3, conv1_out, edge_dim=1)
        self.conv2 = GATConv(conv1_out, conv2_out, edge_dim=1)

        self.x1 = nn.Linear(num_nodes*conv2_out, num_nodes)
        self.x2 = nn.Linear(num_nodes, out)

        self.num_nodes = num_nodes

    def my_sigmoid(self, tensor):
        return torch.sigmoid(tensor)

    def forward(self, state):
        """
        data: N x 2 array. Each row contains (latency,bancdwidth)
        """
        loader: DataLoader = state[0]
        mask = state[1]

        for batch in loader:
            out = self.conv1(batch.x, batch.edge_index, batch.edge_attr)
            out = torch.selu(out)
            out = self.conv2(out, batch.edge_index, batch.edge_attr)
        out = torch.stack(out.split(self.num_nodes))
        out = out.flatten(start_dim=1)
        conv_out = torch.selu(out)
        x = self.x1(conv_out)
        x = torch.selu(x)
        x = self.x2(x)

        x[~mask] = float('-inf')

        return x

recorder = [0, 0]

class GCN_Agent(Agent):
    def __init__(self, outputs, policy_net, target_net, num_nodes, env):
        super().__init__(outputs, policy_net, target_net)
        self.num_nodes = num_nodes
        self.env = env
        self.memory = ReplayMemory(3_000)
        self.steps_done = 0
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=0.001)

        self.scheduler = StepLR(
            self.optimizer,
            step_size=80,
            gamma=.9)
        self.metrics = {
            'loss': [],
            'reward': [],
            'path_length': [],
            'eps_reward': []
        }

    def select_action(self, state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)

        if sample > eps_threshold:
            state_as_loader = DataLoader(
                [state[0]], batch_size=1, shuffle=False)
            recorder[0] += 1
            with torch.no_grad():
                self.policy_net.eval()
                action = self.policy_net((state_as_loader, state[1])).max(1)[
                    1].view(1, 1)
                self.policy_net.train()
                return action

        else:
            recorder[1] += 1

            mask = state[1][0]
            indices = (mask == True).nonzero().flatten()
            random_action = indices[torch.randint(
                0, indices.shape[0], size=(1, 1))]

            return random_action

    def _format_input(self, state):
        """
        state here is (src,tgt,mask) where src, tgt are ints and mask is an array
        """
        src_tgt_mat = torch.zeros(size=(self.num_nodes, 2), device=device)
        src_tgt_mat[state[0]][0] = 1  # src
        src_tgt_mat[state[1]][1] = 1  # tgt
        return torch.unsqueeze(src_tgt_mat, 0)

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            self.steps_done = 0
            return
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.as_tensor(tuple(map(lambda s: s is not None,
                                                   batch.next_state)), device=device, dtype=torch.bool)

        non_final_next_graph_list = [s[0]
                                     for s in batch.next_state if s is not None]
        non_final_next_states = (DataLoader(non_final_next_graph_list, batch_size=len(
            non_final_next_graph_list), shuffle=False),
            torch.cat([s[1] for s in batch.next_state if s is not None]))
        state_batch_graph_list = [s[0] for s in batch.state]
        state_batch = (DataLoader(state_batch_graph_list, batch_size=len(
            state_batch_graph_list), shuffle=False),
            torch.cat([s[1] for s in batch.state]))

        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        state_action_values = self.policy_net(
            state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_net(
            non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (
            next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values,
                                expected_state_action_values.unsqueeze(1))
        self.metrics['loss'].append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()

        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update_lr(self):
        for g in self.optimizer.param_groups:
            g['lr'] = 0.001

    def run(self, num_episodes=1000):
        step_count = 0
        global recorder
        for i_episode in tqdm(range(num_episodes), position=0, leave=True):
            # Initialize the environment and state
            _state, data = self.env.reset()
            mask = data['valid_actions']
            state = _state, mask
            eps_rew = 0

            for t in count():
                step_count += 1
                self.steps_done += 1

                action = self.select_action(state)
                _next_state, reward, done, data = self.env.step(action.item())
                eps_rew += reward
                self.metrics['reward'].append(reward)

                next_state_mask = data['valid_actions']

                next_state = _next_state, next_state_mask

                reward = torch.tensor([reward], device=device)

                if done:
                    next_state = None

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                self.optimize_model()


                if done:
                    self.episode_durations.append(t + 1)
                    self.metrics['path_length'].append(t+1)
                    self.metrics['eps_reward'].append(eps_rew)

                    break
                # Update the target network, copying all weights and biases in DQN
                if step_count % TARGET_UPDATE == 0:
                    self.target_net.load_state_dict(
                        self.policy_net.state_dict())
                    # print("here")
            self.scheduler.step()

            if i_episode % 50 == 0:
                print(i_episode, self.steps_done, reward)
                metrics = pd.DataFrame(
                    dict([(k, pd.Series(v)) for k, v in self.metrics.items()]))

                print(metrics.describe().loc[[
                      'count', 'mean', '50%', '25%', '75%']])
                self.metrics = {
                    'loss': [],
                    'reward': [],
                    'path_length': [],
                    'eps_reward': []
                }
                print("decay:", recorder[0]/sum(recorder))
                print("lr:", self.scheduler.get_last_lr())
                recorder = [0, 0]
                print("-"*50)
